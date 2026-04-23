# UQ Framework for vEcoli

## Dev Environment
- **Always use `uv run`** instead of `python` or `python3`
- Tests: `uv run pytest tests/`
- Marimo notebooks: `uv run marimo run tutorials/<notebook>.py`
- Marimo reactivity rules: no duplicate variable names across cells, use `_` prefix for temporaries, use `mo.ui.array()` for reactive slider collections
- Live sample generation: `make test-sampling` (runs 20 LHS samples with `--live --max-workers 4`)

## Architecture (RFC006 — MS-08.4.2)
UQ framework for tracking prediction confidence in vEcoli whole-cell simulations.

### Core Pipeline (7 steps)
1. **Parameter space** (`inputs.XSpaceVecoli`) — generic `SimDataParameter` specs (any scalar `simData` attribute)
2. **Load data** — `pipe.initialize_data()` loads Parquet via `TimeseriesLoaderParquet` + `ParameterDataset` from `simData.cPickle`
3. **Aggregate** via 4 strategies (`pipeline/workflow.aggregate_timeseries()`):
   - Uniform (baseline), by generation, by lineage seed, by cell cycle stage
4. **Variance decomposition** (`pipeline/workflow.get_variance_decomposition()`)
5. **Morris prescreening** (`sensitivity.SensitivityAnalyzer.analyze_with_morris()`) — identify top-K important params
6. **PCE surrogate** (`pce.generate_surrogate()`) — Legendre/Hermite basis, LHS sampling, least squares/LASSO/OMP fitting
7. **Sobol indices** from PCE coefficients via `SensitivityAnalyzer.analyze_with_pce()` → PyTUQ `PCSobol.compute()` → `SobolIndices`

### Module Map (`uq/`)
| Module | Role |
|--------|------|
| `models.py` | Pydantic config models: `PipelineConfig`, `SamplingConfig`, `XPrescreenConfig` |
| `inputs.py` | `XSpace` (ABC) + `XSpaceVecoli` (working implementation), `load_dataset()`, `get_available_columns()` |
| `outputs.py` | OutputExtractor — DuckDB extraction of transcriptome, proteome, fluxes, properties from Parquet |
| `aggregation.py` | 4 aggregation strategies, Aggregator class, variance decomposition |
| `wrappers.py` | SimulationWrapper, PrecomputedWrapper, DataDrivenWrapper (response surface from data stats) |
| `sampling.py` | Two-stage workflow: `PrecomputedCache` (on-disk X/Y/timeseries cache), `generate_lhs_samples()`, `run_and_cache()` |
| `sensitivity.py` | SensitivityAnalyzer, SobolIndices, MorrisIndices, PCESurrogate, GSA-informed cell cycle analysis |
| `synthetic.py` | `generate_synthetic_simulation_data()`, `generate_signal()` — for test/demo only |
| `pce/surrogate.py` | PCE math: fit coefficients, multi-indices, basis matrices, LHS sampling, generate_surrogate() |
| `pce/models.py` | PCE-specific models (PCESolverConfig, PCESurrogateConfig, PCEParameterSelectionConfig, etc.) |
| ~~`cell_cycle.py`~~ | *Removed from public — see `private/spectral-music-apollo` branch* |
| ~~`koopman.py`~~ | *Removed from public — see `private/spectral-music-apollo` branch* |
| ~~`viz.py`~~ | *Removed from public — see `private/spectral-music-apollo` branch* |
| `pipe.py` | `Pipeline` dataclass (full orchestration), `execute_pipeline()`, `initialize_data()` → `DatasetMultiExperiment` |
| `handlers.py` | CLI handler functions: `generate_samples()`, `export_configs()`, `collect_results()`, `pipeline()` |
| `cli.py` | Typer CLI app: `quantify`, `generate-samples`, `export-configs`, `collect-results`, `configure-pipeline`, `demo`, `readme` |
| `pipeline/workflow.py` | `aggregate_timeseries()`, `get_variance_decomposition()`, `run_phase1()`, `run_phase2()`, `AggregationResult`, `Strategy4Wrapper` |
| `pipeline/models.py` | Domain models: `PipelineResult`, `UqProfile`, `SimDataParameter`, `GenericSimDataParams`, `CellCycleVariable` |
| `pipeline/output_loader.py` | `TimeseriesLoaderParquet`, `load_timeseries()` — loads Parquet with hive partitioning |
| `pipeline/param_loader.py` | `ParameterDataset` — loads `simData.cPickle`, `DEFAULT_SIM_DATA_PARAMETERS`, `to_parameter_space(parameters=...)` |
| `generators/vecoli.py` | `TimeseriesGeneratorVecoli` — subprocess-based vEcoli simulation wrapper, `_apply_sim_data_mutations()`, `export_batch_configs()`, `collect_batch_results()` |
| `remote.py` | `SmsApiClient` — httpx wrapper for SMS-API REST endpoints, cd1 TSV parsing (`parse_cd1_tsvs`, `find_cd1_tsvs`), ALB 502/504 retry logic, `CD1_MODULE_MAP` (preset→module mapping) |
| `multi_condition.py` | `MultiConditionResult`, `compute_rank_stability()`, `find_universal_drivers()`, `find_condition_specific()`, `compute_differential_sobol()`, `quantify_multi_condition()` |
| `common/` | Shared utilities (`BaseClass`, `get_repo_root()`) |
| `io.py` | DataclassIO — serialize dataclasses + numpy arrays |

### CLI Commands
| Command | Handler | Description |
|---------|---------|-------------|
| `uq quantify` | `handlers.pipeline()` | Full RFC006 pipeline (Phase 1 + Phase 2) |
| `uq generate-samples` | `handlers.generate_samples()` | LHS sampling + evaluation + cache to disk |
| `uq export-configs` | `handlers.export_configs()` | Write per-sample vEcoli configs for HPC batch |
| `uq collect-results` | `handlers.collect_results()` | Assemble HPC batch outputs into PrecomputedCache |
| `uq configure-pipeline` | writes JSON | Default PipelineConfig to JSON |
| `uq demo` | `handlers.demo()` | Demo with default experiments |
| `uq readme` | prints diagram | RFC006 workflow ASCII art |

### `uq` CLI Commands (simplified pipeline — `uq/cli.py`)
| Command | Description |
|---------|-------------|
| `uq sample` | UQPC Steps 1-3: PCRV sampling + vEcoli execution (local subprocess or SMS-API remote) |
| `uq quantify` | UQPC Steps 4-5: fit PCE surrogates, compute Sobol indices (all 4 strategies) |
| `uq fetch` | Download + parse cd1 analysis TSVs from a completed SMS-API simulation |
| `uq dashboard` | Interactive visualization (tk DAW, marimo, or web) |
| `uq tui` | Textual terminal UI |
| `uq gui` | Marimo browser GUI |
| `uq show-config` | Preview vEcoli workflow config JSON without running |
| `uq compare` | Side-by-side Sobol comparison across multiple UQ experiments |
| `uq export-figures` | Publication-ready PDFs and LaTeX from results |
| `uq suggest-experiment` | Identify max-uncertainty parameter region |
| `uq init` | Guided project setup wizard |

### Key Input Variables (per RFC006)
Any scalar `SimulationDataEcoli` attribute by dot-path, specified as `SimDataParameter` objects. Default 5 parameters (`DEFAULT_SIM_DATA_PARAMETERS`):
- `process.metabolism.kinetic_objective_weight` — FBA kinetic vs homeostatic objective
- `process.metabolism.secretion_penalty_coeff` — penalty on secretion fluxes
- `process.transcription.fraction_active_rnap_free` — RNAP active fraction (ppGpp-free)
- `process.transcription.fraction_active_rnap_bound` — RNAP active fraction (ppGpp-bound)
- `mass.cell_dry_mass_fraction` — dry mass fraction

Custom parameters via `--params-file params.json` (see `examples/uq_artifacts/params/params_demo.json`).

### Key Output Variables (observable presets — `uq/observables.py`)
Configurable via `--observables` flag on `uq sample`. Each preset mirrors a cd1 analysis module:
- **mass** — raw mass/growth scalars (5 features, default)
- **higher_order** — derived cd1 metrics: doubling time, growth rate, DNA/RNA/dry mass fractions, volume (6 features)
- **exchange_fluxes** — external metabolite fluxes (~87 features)
- **transcriptome** — mRNA cistron counts (~4,300 genes)
- **proteome** — protein monomer counts (~4,300 monomers)
- **fluxome** — base reaction fluxes, dry-mass normalized (~2,800 reactions)

Presets compose: `--observables higher_order --observables transcriptome` concatenates both.
`--generation-lower-bound N` filters early transient generations (cd1 pattern).

### Pipeline Entry Points

**`Pipeline` class** (`uq/pipe.py`) — preferred entry point:

```python
from libuq.pipe import Pipeline

pipe = Pipeline(
    experiment_ids=["my_experiment"],
    sim_base_path="/path/to/sims",
    n_samples=50,
    polynomial_order=2,
)
pipe.run()
result = pipe.result  # PipelineResult
```

**`execute_pipeline()`** — functional entry point:

```python
from libuq.pipe import execute_pipeline

result = execute_pipeline(
    experiment_ids=["my_experiment"],
    sim_base_path="/path/to/sims",
    precomputed_path="./cache",
    export_path="./uq_results",
)
```

**`initialize_data()`** — data loading only:

```python
from libuq.pipe import initialize_datasets

ds = initialize_datasets(
    experiment_ids=["my_experiment"],
    sim_base_path="/path/to/sims",
)
# ds.parameter_space → XSpaceVecoli
# ds.y → polars.DataFrame (timeseries)
# ds.observables → list[str]
# ds.x → list[ParameterDataset]
```

### Two-Stage Workflow (CLI)
```bash
# Stage 1: generate and cache with LIVE vEcoli simulations (subprocess, no in-process EcoliSim)
# Default: 5 physiologically relevant sim_data parameters
uv run uq sample \
    api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 20 --live --max-workers 4

# With custom parameters from JSON:
uv run uq sample \
    api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 20 --live --params-file examples/uq_artifacts/params/params_demo.json

# Stage 2: analyze from cache (fast, repeatable)
uv run uq quantify \
    api_simulation_default \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results
```

### HPC Batch Workflow (CLI)
```bash
# 1. Export per-sample configs
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200

# 2. Run on cluster (Nextflow, Slurm, etc.)

# 3. Collect results into cache
uv run uq collect-results ./batch ./batch_outputs --cache-dir ./uq_cache

# 4. Analyze
uv run uq quantify exp1 --sim-base-path /path/to/sims --precomputed-path ./uq_cache
```

### Pipeline Output Structure
The full pipeline produces a `PipelineResult` with two `UqProfile` instances plus intermediate outputs:
- **Population** (Phase 1, bulk): 1 `SobolIndices` + 1 `PCESurrogate` — static variance decomposition across all cells
- **Cell Cycle** (Phase 2, per-θ-bin): `n_bins` `SobolIndices` + 1 `PCESurrogate` — variance decomposition conditioned on cell cycle stage
- **`variance_decomposition`**: Dict with `generation_fraction`, `seed_fraction`, `between_generation_variance`, `between_seed_variance`, `within_group_variance`
- **`aggregation`**: `AggregationResult` (uniform, generation, seed `AggregatedOutput` instances)
- **`morris_indices`**: `MorrisIndices` from prescreening (if `prescreen_config` provided)
- **`cell_cycle_relevance`**: `CellCycleRelevanceResult` from GSA-informed observable selection

### Function Return Types
- `run_phase1(...)` → `tuple[SobolIndices, PCESurrogate, MorrisIndices | None]`
- `run_phase2(...)` → `tuple[list[SobolIndices], PCESurrogate, CellCycleRelevanceResult]`

`SobolIndices.first_order` and `.total_order` are `np.ndarray` of floats (fractions of variance explained), not integer indices. Use `.select(n=K)` to get top-K `(param_name, float)` tuples.

### Sobol Index Computation Path
`SensitivityAnalyzer.analyze_with_pce()` supports two modes:

**With wrapper (live evaluation):**
1. `PCSobol(dom=bounds, pctype="LU", order=p)` — creates PyTUQ PCE+Sobol object
2. `pc_sobol.sample(N)` → LHS samples in germ space
3. `wrapper.evaluate_batch(xsam)` → model evaluations
4. `pc_sobol.compute(ysam)` → fits PCE coefficients, decomposes variance → `sens["main"]` (S_i), `sens["total"]` (S_Ti)

**With precomputed samples (cached mode):**
1. Scale X to germ space [-1, 1], build Legendre basis via `pc.pcrv.evalBases(X_germ, 0)`
2. Fit coefficients via `lsq().fita(Amat, y_col)` per output column
3. Compute Sobol from `pcrv.computeSens()`, variance-weight across outputs

### Tutorials
| Tutorial | Description |
|----------|-------------|
| `01_introduction.py` | UQ framework basics |
| `02_aggregation_strategies.py` | Aggregation strategies |
| `03_sensitivity_analysis.py` | PCE and Sobol indices |
| `03b_reactive_sensitivity.py` | Reactive parameter exploration (vEcoli-specific) |
| `03c_reactive_sensitivity_generalized.py` | Generalized reactive exploration |
| ~~`04_cell_cycle_and_koopman.py`~~ | *Removed from public* |
| `07_full_workflow.py` | Complete Morris → PCE → reactive exploration |
| `08_nextflow_execution.py` | Nextflow HPC execution |
| `09_batch_pipeline.py` | Batch pipeline patterns |
| `10_dashboard.py` | **Full RFC006 interactive dashboard** — GUI alternative to CLI, uses real data |

### Dashboard (`app/`)
| File | Description |
|------|-------------|
| `app/dashboard.py` | Marimo notebook dashboard (`uq dashboard --run-mode mo`) |
| `app/uq_daw.py` | Tkinter DAW dashboard (`uq dashboard` or `--run-mode tk`) |
| `app/README.md` | Dashboard documentation — panels, math, PCE evaluation |

**Launch:** `uv run uq dashboard` (tk, default) or `uv run uq dashboard --run-mode mo` (marimo)

The dashboard loads `uq_results.json` + surrogate binaries from `PipelineResult.export()` and provides:
- **PCE Response Curves** — draggable parameter markers (tk) / slider-reactive (marimo)
- **Observable Waveform** — per-stage Y(θ) in physical units, baseline vs modulated
- **Sensitivity Spectrogram** — S_Ti heatmap (params × stages) with per-stage prediction curve
- **Heatmap Minimap** — toggleable observable-domain heatmap in physical units

All panels use pipeline outputs only (PCE coefficients, per-stage Sobol indices, cell cycle profile). The observable waveform modulation formula:
```
Y_obs_k(x) = baseline_obs_k * (1 + sum_i [|dY/dx_i| * (x_i - mid) * S_Ti^(k)] / |baseline|)
```

### Sampling (`uq/sampling.py`)
`run_batch_and_cache()` (live mode) generates LHS samples and delegates to `TimeseriesGeneratorVecoli._run_batch()`, which runs all simulations as **subprocesses** via `ecoli_master_sim.py` with the Parquet emitter. No `EcoliSim` is held in the UQ process memory. Y (aggregated) is derived by `ts.mean(axis=0)`. `run_and_cache()` remains for synthetic mode (DataDrivenWrapper).

### Export (`PipelineResult.export()`)
Writes a complete artifact directory:
```
export_dir/
├── uq_results.json              # Comprehensive JSON (Phase 1 + Phase 2 Sobol, profile, Morris, relevance)
├── cell_cycle_profile.json       # Per-stage observable means
├── population_surrogate/         # PCE coefficients + multi-indices (.npy)
├── cell_cycle_surrogate/         # PCE coefficients + multi-indices (.npy)
├── cell_cycle_sobol_stage_N/     # Per-stage Sobol indices (.npy)
├── variance_decomposition.json
├── morris_indices/
└── metadata.json
```

## Detailed Docs
- Full RFC: `readmes/RFC006.md`
- Extended context: `readmes/CONTEXT.md`
- Dashboard guide: `app/README.md`
- About vEcoli: `readmes/VECOLI.md`

## Most Recent Work

### SMS-API remote execution (2026-04-23)
`uq/remote.py` — SMS-API client for remote vEcoli execution via AWS Batch:
- `SmsApiClient` wraps httpx for the REST API at `http://localhost:8080` (stanford-test)
- cd1 TSV parsing: `parse_cd1_tsvs()`, `find_cd1_tsvs()`, `parse_cd1_tsv()`
- `CD1_MODULE_MAP` maps UQ presets → cd1 modules (transcriptome→cd1_transcriptomics, etc.)
- ALB 502/504 auto-retry via `_request_with_retry()`
- `uq sample --api-url http://localhost:8080 --simulator-id 11` for remote mode
- `uq fetch <sim_id>` downloads + parses cd1 outputs from completed simulations
- Verified: sim 48 (10 gens, 1000 seeds) → 11,644 observables across 5 cd1 modules

### Cross-condition GSA (2026-04-16)
`uq/multi_condition.py` — cross-condition sensitivity analysis via multi-parca:
- `uq sample --conditions glucose_minus_aas --conditions glucose_plus_aas`
- `uq quantify` auto-detects multi-condition cache, runs per-condition + cross-condition
- `MultiConditionResult`: rank stability, universal drivers, condition-specific params
- Dashboard: condition selector combobox (Tk), grouped bar chart (Marimo)
- CLI: cross-condition comparison table with rank stability indicators (●●●○○)

### PyTUQ PCE refactor (2026-03-27)
`uq/workflow.py` uses `pytuq.surrogates.pce.PCE` directly (UQPC workflow):
- `_fit_pce_and_sobol()` — core function: scales to germ space, fits PCE per output, syncs coefficients via `setCfs`, computes Sobol from PCRV
- `regression` parameter ('lsq', 'bcs', 'anl') threaded through all functions and CLI (`--regression`)
- Critical: after `PCE.build(regression='lsq')`, must call `pce.pcrv.setCfs([pce.lreg.cf])` before Sobol computation

### Default parameters expanded to 6 (2026-03-27)
`DEFAULT_SIM_DATA_PARAMETERS` in `uq/pipeline/param_loader.py` now includes:
- `process.transcription.fraction_active_rnap_free` — ppGpp-free RNAP
- `process.transcription.fraction_active_rnap_bound` — ppGpp-bound RNAP
- `process.translation.basal_elongation_rate` — ribosome speed (aa/s)
- `process.metabolism.kinetic_objective_weight` — FBA kinetic objective
- `process.metabolism.secretion_penalty_coeff` — overflow metabolism
- `mass.cell_dry_mass_fraction` — dry mass fraction

## Public Fork: Atlantis

This repo (`uqEcoli`, private) is the development source. A **public subset** is
cherry-picked into:

- **Repo:** https://github.com/vivarium-collective/atlantis
- **Local path:** `../atlantis`

### What goes into Atlantis
Atlantis is the public-facing, open-source release of the UQ framework. It should
contain the **generalizable** parts of this repo — the parts that are not specific
to private vEcoli internals, proprietary data, or unpublished results. Specifically:

**Include:**
- `uq/workflow.py` — the PyTUQ UQPC pipeline (generic, no private deps)
- `uq/remote.py` — SMS-API client + cd1 TSV parsing
- `uq/multi_condition.py` — cross-condition GSA
- `uq/cli.py` — the Typer CLI (all commands)
- `uq/tui.py` — the Textual TUI
- `uq/models.py` — Pydantic config models
- `uq/growth.py` — growth-stage binning
- `uq/observables.py` — observable presets + collection
- `app/uq_daw_simple.py` — tkinter DAW dashboard
- `app/dashboard_simple.py` — marimo dashboard
- `app/gui.py` — marimo GUI
- `tests/` — all tests that don't require private data
- `docs/` — Sphinx documentation
- `README.md`, `SAMPLING.md` — public documentation
- `tutorials/` — marimo tutorial notebooks

**Exclude (keep private in uqEcoli):**
- `libuq/` — the older full pipeline (pre-PyTUQ refactor)
- `uq/pipeline/` — older pipeline models/loaders tightly coupled to libuq
- `uq_cache_*/`, `uq_results_*/` — cached simulation data and results
- `sim_data/` or any `simData.cPickle` references that embed proprietary data paths
- `uq/bigraph/` — process-bigraph integration (experimental)
- `patches/`, `assets/` — internal development artifacts
- Any hardcoded paths to `../vEcoli` or private repos

### Relationship
- **uqEcoli** = private development repo (this one). All work happens here first.
- **Atlantis** = public release repo. Cherry-picked content, cleaned of private paths.
- The Atlantis agent should understand: it is downstream of uqEcoli. If something
  looks incomplete or references missing modules, check uqEcoli for the source.
- The SMS-API at https://github.com/vivarium-collective/sms-api is the REST API
  that Atlantis's `uq/remote.py` talks to. It is a separate public repo.