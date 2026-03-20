# UQ Pipeline Output Reference

> Traceability map for `uq_results.json` — each field traced from RFC006 requirement to CLI verification.

---

## 1. `parameter_names` / `n_parameters`

**What it captures:** The input parameters varied during sensitivity analysis (here: `vio_expression`, `vio_trl_eff`, `mecillinam_concentration`).

**RFC006 basis (Activity 1):**
> "Identify the scientifically most relevant input and output variables. Expected inputs: vio pathway presence, mecillinam condition, gene knockouts..."

Each parameter is a `SimDataParameter` specifying a dot-path into `SimulationDataEcoli` (e.g., `process.metabolism.kinetic_objective_weight`) with sampling bounds. The parameter space is defined via `XSpaceVecoli` and can be customized with `--params-file`.

**CLI verification:**
```bash
# Custom parameters via JSON spec:
uv run uq sample <exp_ids> --params-file params.json --cache-dir ./cache --live

# Default parameters (see DEFAULT_SIM_DATA_PARAMETERS):
uv run uq quantify <exp_ids> <outdir>
```

**Code path:** `SimDataParameter` (`pipeline/models.py`) -> `ParameterDataset.to_parameter_space()` (`pipeline/param_loader.py`) -> `XSpaceVecoli` (`inputs.py`)

---

## 2. `n_cell_cycle_stages`

**What it captures:** Number of cell-cycle bins (here: 10) used to stratify Phase 2 sensitivity analysis.

**RFC006 basis (Section 3, Strategy 4):**
> "A low-dimensional (possibly scalar) 'cell cycle variable' computed from omics variables...will be used for deterministically binning simulation data into cell stages."

Timepoints are assigned a cell-cycle coordinate theta in [0, 1] (via Koopman DMD or mass progression), then binned into `n_bins` equal-width stages. Each stage gets its own Sobol decomposition.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --n-bins 10
```

**Code path:** `Pipeline(n_bins=10)` (`pipe.py`) -> `run_phase2(n_bins=...)` (`pipeline/workflow.py`) -> `Strategy4Wrapper` bins by theta

---

## 3. `variance_decomposition`

**What it captures:** ANOVA-style decomposition of total output variance into generation effects (convergence), seed effects (exogenous stochasticity), and residual (cell cycle + parameter sensitivity).

**RFC006 basis (Section 3, Strategies 1-3):**
> "Variance Decomposition -- compute_variance_decomposition(agg_uniform, agg_by_gen, agg_by_seed) -> per observable: generation_fraction, seed_fraction, residual_fraction (cell-cycle-related, key output)"

| Field | Meaning |
|-------|---------|
| `total_variance` | Var(Y) across all cells/times |
| `generation_fraction` | Fraction attributable to between-generation differences |
| `seed_fraction` | Fraction attributable to between-seed (exogenous) differences |
| `residual_fraction` | Remainder -- primarily cell cycle and parameter effects |
| `between_generation_variance` | Absolute between-generation variance |
| `between_seed_variance` | Absolute between-seed variance |
| `within_group_variance` | Absolute within-group (residual) variance |

In this example, residual dominates (99.9%), meaning almost all variance is driven by parameter sensitivity and cell-cycle dynamics rather than generation convergence or seed stochasticity.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --export-path ./results
# -> results/variance_decomposition.json
```

**Code path:** `aggregate_timeseries()` -> `get_variance_decomposition()` (`pipeline/workflow.py`) -> stored in `PipelineResult.variance_decomposition`

---

## 4. `phase1_population_sobol`

**What it captures:** Population-level (bulk) Sobol sensitivity indices -- first-order (S_i, main effects) and total-order (S_Ti, including interactions).

**RFC006 basis (Activity 4, Phase 1):**
> "Implement well established global sensitivity analysis methods based on the aggregation strategies (1-3)...PCE Surrogate (Strategies 1-3) -> PCESurrogate...Sobol Indices (from PCE) -- S_i, S_Ti from PCE coefficients -> SobolIndices"

Phase 1 asks: *"Across all cells, all times, all generations -- which parameters drive the most variance in bulk output?"* In this example, `vio_expression` (S_Ti=0.55) and `mecillinam_concentration` (S_Ti=0.50) dominate, while `vio_trl_eff` is minor (S_Ti=0.08). The gap between S_i and S_Ti indicates parameter interactions.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --pce-polynomial-order 3 --n-samples 200
# Terminal report shows "GSA PHASE 1 // POPULATION (bulk)" table
```

**Code path:** `run_phase1()` (`pipeline/workflow.py`) -> `SensitivityAnalyzer.analyze_with_pce()` (`sensitivity.py`) -> `SobolIndices`

---

## 5. `phase2_cell_cycle_sobol_per_stage`

**What it captures:** Per-stage Sobol indices conditioned on cell-cycle position theta. Each of the 10 stages has its own first-order and total-order decomposition.

**RFC006 basis (Section 4, Phase 2):**
> "Phase 2 (Strategy 4 -- Cell Cycle): PCE + Sobol on Strategy 4 -> per-stage PCESurrogate, list[SobolIndices]...answering: 'During DNA replication, mecillinam_conc drives 80% of variance'"

Phase 2 asks: *"Within a single cell cycle stage, which parameters drive variance?"* This is orthogonal to Phase 1 -- not a temporal refinement of it. Key insight from this example:

- **Early stages (theta 0.0-0.3):** `vio_expression` dominates (S_Ti ~ 0.48-0.68)
- **Late stages (theta 0.7-1.0):** `mecillinam_concentration` takes over (S_Ti ~ 0.41-0.49)
- **`vio_trl_eff`** remains minor throughout (S_Ti ~ 0.08-0.12)

This crossover pattern -- violacein expression matters early, mecillinam matters late -- is the kind of stage-specific insight that Phase 1 alone cannot provide.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --n-bins 10
# Terminal report shows "GSA PHASE 2 // CELL CYCLE (10 stages)" with per-stage tables
# Export: results/cell_cycle_sobol_stage_0/ through .../cell_cycle_sobol_stage_9/
```

**Code path:** `run_phase2()` -> `Strategy4Wrapper` -> `compute_strategy4_sobol()` -> `_split_multi_output_sobol()` (`pipeline/workflow.py`)

---

## 6. `cell_cycle_profile`

**What it captures:** Per-stage mean values for key observables (dry mass, growth rate) across the 10 cell-cycle stages.

**RFC006 basis (Phase 2, per-stage statistics):**
> "Per-stage observable means/stds" -- required for the dashboard's observable waveform panel and for interpreting per-stage Sobol indices in physical units.

The profile here shows the expected pattern: dry mass increases monotonically (1.027 -> 1.908) while growth rate peaks early (0.01165 at stage 2) then drops through mid-cycle before recovering -- consistent with known *E. coli* cell cycle physiology.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --export-path ./results
# -> results/cell_cycle_profile.json
# Also consumed by: uv run uq dashboard --run-mode tk
```

**Code path:** `Pipeline._compute_cell_cycle_profile()` (`pipe.py`) -> `PipelineResult.cell_cycle_profile`

---

## 7. `morris_screening`

**What it captures:** Morris elementary effects prescreening -- identifies which parameters are influential before the more expensive PCE analysis.

**RFC006 basis (Activity 4, Phase 1, Step 5a):**
> "Morris Prescreening -- EE_i = [f(x+De_i) - f(x)] / D, mu* = mean(|EE_i|), sigma = std(EE_i), n params -> K params (K << n) -> MorrisIndices"

| Metric | Meaning |
|--------|---------|
| `mu` | Mean elementary effect (direction-sensitive) |
| `mu_star` | Mean absolute elementary effect (importance, preferred ranking metric) |
| `sigma` | Std dev of elementary effects (high = nonlinear or interactive) |

Classification rule: if sigma < 0.5 * mu_star, the parameter effect is approximately linear; otherwise it is nonlinear or interactive. Here, `mecillinam_concentration` (sigma/mu_star = 0.55) shows nonlinear/interactive behavior, while `vio_expression` (0.33) is more linear.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> \
    --pce-n-trajectories 10 --pce-n-selected-params 5
# Terminal report shows "MORRIS PRESCREENING" table with mu*, sigma, and classification
```

**Code path:** `SensitivityAnalyzer.analyze_with_morris()` (`sensitivity.py`) -> `MorrisIndices`, called from `run_phase1()` when `prescreen_config` is provided

---

## 8. `cell_cycle_relevance`

**What it captures:** GSA-informed selection of which observables are most relevant for cell-cycle-conditioned analysis (Phase 2).

**RFC006 basis (Section 3 + Phase 2, Step 5b):**
> "The choice of the 'cell cycle variable' will be informed by the sensitivity analyses (1-3)...GSA-Informed Observable Selection -- identify_cell_cycle_relevant_observables(decomp)"

The algorithm ranks observables by `residual_fraction` from the variance decomposition (Step 4). Observables with high residual variance are candidates for Phase 2 -- their variance is not explained by generation or seed effects, so it likely comes from cell-cycle dynamics and parameter sensitivity.

Here, `dry_mass` (relevance 0.92) and `growth` (relevance 0.78) are selected. This is the feedback loop from Phase 1 to Phase 2: variance decomposition informs which observables merit cell-cycle-resolved analysis.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir>
# Terminal report shows "CELL CYCLE RELEVANT OBSERVABLES" table
# Automatic -- no separate flag needed
```

**Code path:** `identify_cell_cycle_relevant_observables()` (`sensitivity.py`) -> `CellCycleRelevanceResult`, called from `run_phase2()` (`pipeline/workflow.py`)

---

## 9. `surrogates`

**What it captures:** Metadata for the two PCE (Polynomial Chaos Expansion) surrogate models -- one for population-level (Phase 1) and one for cell-cycle-conditioned (Phase 2) analysis.

**RFC006 basis (Activity 4):**
> "Expected to use PCE surrogate method for the stochastic function (sim_data -> SIM output)...f(x) ~ sum c_a Psi_a(x), Legendre basis, least-squares on LHS samples -> PCESurrogate"

| Field | Population Surrogate | Cell Cycle Surrogate |
|-------|---------------------|---------------------|
| `basis_type` | Legendre | Legendre |
| `polynomial_order` | 3 | 2 |
| `input_dim` | 3 (parameters) | 3 (parameters) |
| `output_dim` | 2 (observables) | 10 (stages) |
| `r_squared` | 0.92 | 0.87 |
| `n_terms` | 20 | 10 |

The surrogates enable: (a) Sobol index computation from PCE coefficients without additional simulation runs, (b) fast response-surface evaluation in the dashboard (draggable parameter sliders), and (c) prediction uncertainty via R-squared quality metric.

**CLI verification:**
```bash
uv run uq quantify <exp_ids> <outdir> --export-path ./results
# -> results/population_surrogate/   (coefficients.npy, multi_indices.npy)
# -> results/cell_cycle_surrogate/   (coefficients.npy, multi_indices.npy)
# Dashboard loads these for interactive exploration:
uv run uq dashboard
```

**Code path:** `SensitivityAnalyzer.analyze_with_pce()` -> `PCESurrogate` (`sensitivity.py`), stored in `PipelineResult.population.surrogate` and `PipelineResult.cell_cycle.surrogate`

---

## Quick Reference: Full Pipeline Command

```bash
# Stage 1: Generate samples (compute-intensive)
uv run uq sample <exp_ids> \
    --sim-base-path /path/to/sims \
    --cache-dir ./cache --n-samples 200 --live \
    --params-file params.json

# Stage 2: Analyze (fast, produces all 9 output sections)
uv run uq quantify <exp_ids> /path/to/sims \
    --precomputed-path ./cache \
    --export-path ./results \
    --n-bins 10 \
    --pce-polynomial-order 3 \
    --pce-n-trajectories 10

# Stage 3: Explore interactively
uv run uq dashboard
```
