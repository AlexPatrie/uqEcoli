# RFC006 Compliance Audit — Gaps Found and Resolved

**Audit date:** 2026-03-17
**Scope:** Full review of RFC006 requirements against the `uq/` implementation, with focus on whether Mode B (local parallel) and Mode C (HPC batch) satisfy the RFC.

---

## Gap 1: Gene Knockouts Not Applied to Simulations (FIXED)

**RFC006 Activity 1** requires three input variable types:
> "Inputs: vio pathway presence, mecillinam condition, **gene knockouts**"

**What existed:**
- `GeneKnockoutParams` dataclass defined (`uq/pipeline/models.py:274-287`) with `gene_deletions` and `translation_knockouts` fields
- `UQInputParametersVecoli.knockouts` field existed (`models.py:542`)
- `XSpaceVecoli.sample_to_params()` accepted `knockouts` parameter (`xspace.py:371-372`)

**What was missing:**
- `UQInputParametersVecoli.to_simulation_config()` (`models.py:547-574`) **never emitted knockout variants**. The knockouts field was silently ignored — simulations ran without any gene knockouts regardless of what was set.

**Fix applied:**
- `to_simulation_config()` now handles both knockout types:
  - **`translation_knockouts`**: Merged into the `mecillinam_timeline` variant's `knockouts` field. The `mecillinam_timeline.apply_variant()` function in vEcoli already handles this by setting `translation_efficiencies_by_monomer[ko_idx] = 0.0`. If no mecillinam concentrations are active, a zero-concentration timeline is used as the vehicle.
  - **`gene_deletions`**: Emitted under `parca_options.gene_deletions` in the config dict. These require ParCa-level changes (before sim_data generation) and are consumed by vEcoli's `knowledge_base_raw.py` when rebuilding sim_data from scratch.

**Files changed:** `uq/pipeline/models.py`

**Verification:** 4 unit tests covering: standalone translation KO, merged translation KO + mecillinam, gene deletions to parca_options, no-knockout baseline (no spurious variants).

---

## Remaining Limitations (Not Gaps — Documented Design Decisions)

### Gene Deletions Require ParCa Rebuild

`gene_deletions` work differently from `translation_knockouts`:
- Translation knockouts are applied at simulation time (variant function sets translation efficiency to 0)
- Gene deletions are applied at ParCa time (before sim_data is built), removing genes from the knowledge base entirely

**Impact on batch workflows:**
- **`export-configs` (Mode C)**: The exported sim_data pickles are built from an existing baseline sim_data. Gene deletions cannot be applied post-hoc to an already-built sim_data — they require re-running ParCa. The `parca_options` field in the config supports this for Nextflow workflows that include a ParCa step.
- **`generate-samples` (Modes A/B)**: Same limitation — the `VecoliSimulationFunc` deep-copies baseline sim_data and applies variant functions, but variant functions cannot retroactively delete genes from already-built sim_data.

**Workaround:** For gene deletion studies, run ParCa separately with the desired deletions to produce a new sim_data pickle, then use that as the baseline for UQ.

### Knockout Parameters Not in Continuous Parameter Space

`XSpaceVecoli` currently treats knockouts as a discrete list (passed via kwargs), not as continuous parameters with bounds. This means:
- Knockouts are not varied by LHS sampling — they're fixed per pipeline run
- Morris/PCE/Sobol analysis operates on the continuous parameters (vio expression, translation efficiency, mecillinam concentration)
- To study knockout effects, you'd run separate pipeline instances with different knockout sets and compare results

This is a design decision, not a bug: gene knockouts are binary (on/off), not continuous, so they don't fit naturally into the LHS → PCE → Sobol workflow which assumes continuous parameter spaces.

---

## Full RFC006 Compliance Status (Post-Fix)

| # | Activity | Status | Implementation |
|---|----------|--------|----------------|
| **Phase 1** | | | |
| 1 | Input/output variables | **COMPLETE** | `XSpaceVecoli` (vio, mecillinam, knockouts), `OutputExtractor` (transcriptome, proteome, fluxes, properties) |
| 2 | Output via emitter | **COMPLETE** | ParquetEmitter + hive partitioning (deviation from XarrayEmitter per RFC note) |
| 3 | Wrapper functions | **COMPLETE** | `VecoliSimulationFunc`, `SimulationWrapper`, `DataDrivenWrapper`, UQPy/PyTUQ factories |
| 4 | PCE-based GSA (strategies 1-3) | **COMPLETE** | `SensitivityAnalyzer.analyze_with_pce()`, Morris prescreening, Sobol indices |
| 5 | Apply to representative sims | **PARTIAL** | See Gap 2 below |
| **Phase 2** | | | |
| 6 | Cell cycle stratification | **COMPLETE** | 6 variable types including GSA-informed, Koopman spectral (PyDMD backend) |
| 7 | Per-stage GSA | **COMPLETE** | `run_phase2()`, `Strategy4Wrapper`, per-stage Sobol |
| **Infrastructure** | | | |
| - | UQPy integration | **COMPLETE** | `create_uqpy_model()`, Uniform distributions, PceSensitivity |
| - | PyTUQ integration | **COMPLETE** | `create_pytuq_model()`, PCSobol, lreg fitting |
| - | PyDMD integration | **COMPLETE** | `DynamicModeDecomposition` and `ExtendedDMD` delegate to `pydmd.DMD`; domain API (`KoopmanSpectrum`, `KoopmanMode`) unchanged |
| - | Batch execution (Mode B) | **COMPLETE** | `ProcessPoolExecutor` in `evaluate_batch()` |
| - | HPC batch (Mode C) | **COMPLETE** | `export-configs`, `collect-results` CLI commands |

---

## Gap 2: Activity 5 — Not Yet Applied to Real Multi-Parameter Simulations (OPEN)

**RFC006 Activity 5** requires:
> "Apply [the UQ framework] to representative simulations"

**What exists and passes (19/19 `@pytest.mark.real_data` tests):**
- Data loading from real `api_simulation_default` Parquet (hive-partitioned, DuckDB)
- All 4 aggregation strategies on real data
- Variance decomposition on real data
- Cell cycle variable computation (mass-based, DNA, Koopman/DMD) on real trajectories
- Cell cycle stratification (`calculate_cell_cycle()`) on real data

**What is tested only with synthetic model functions:**
- Morris screening (Step 4): runs against `synthetic_model(x) = Σ(i+1)*x_i + 0.1*x_i²`, not a real simulation wrapper or precomputed (X, Y) pairs
- PCE surrogate fitting (Step 5): fits to synthetic model evaluations, not real simulation responses
- Sobol indices (Step 6): `test_rfc006_full_workflow_with_real_data` constructs **mock Sobol indices** via `np.random.rand()` rather than computing from PCE coefficients

**Root cause:** The pipeline's sensitivity analysis path (Morris → PCE → Sobol) requires outputs at **multiple parameter configurations** (LHS samples across the vio/mecillinam input space). The existing real data (`api_simulation_default`) is a single experiment at one parameter configuration. Running LHS samples requires either:
1. **`generate-samples` (Mode B)**: Each sample triggers `VecoliSimulationFunc` → `EcoliSim` for a full whole-cell simulation (~minutes per sample, ~50-200 samples needed)
2. **`export-configs` + Nextflow (Mode C)**: Exports per-sample configs, runs on HPC, collects results

**What's needed to close:**
1. Run `uv run uq generate-samples` against real vEcoli sims (requires `ecoli` importable — confirmed available) to produce a `PrecomputedCache` with multi-parameter (X, Y) pairs
2. Run `uv run uq quantify --precomputed-path` against that cache to produce real Sobol indices
3. Update `test_rfc006_full_workflow_with_real_data` to use real (X, Y) pairs instead of `synthetic_model` and mock Sobol indices

**Estimated effort:** The infrastructure is complete; the bottleneck is wall-clock time for simulation evaluation (~50 samples × ~3-10 min/sample = 2.5-8 hours serial, or proportionally less with `--max-workers`).
