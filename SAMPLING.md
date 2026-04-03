# Sampling in `uq sample`

How the CLI generates, evaluates, and caches parameter samples for downstream UQ analysis.

---

## Quick Reference

```bash
# Default: 5 physiological sim_data parameters, live vEcoli subprocess
uv run uq sample api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 200 --live --max-workers 4

# Custom parameters from JSON
uv run uq sample api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 50 --live \
    --params-file examples/uq_artifacts/params/params_demo.json

# Downstream: analyze cached samples (no simulation needed)
uv run uq quantify api_simulation_default \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results
```

---

## End-to-End Data Flow

```
uv run uq sample ...
        │
        ▼
┌─ cli.sample() ──────────────────────────────────────────────────────────┐
│  Parse CLI args, validate experiment dirs exist                         │
│  Delegate to handlers.generate_samples()                                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─ handlers.generate_samples() ───────────────────────────────────────────┐
│                                                                         │
│  1. Load baseline data                                                  │
│     initialize_data(experiment_ids, sim_base_path)                      │
│     → DatasetMultiExperiment with simData.cPickle + Parquet timeseries  │
│                                                                         │
│  2. Resolve parameter specifications                                    │
│     --params-file → JSON list of SimDataParameter specs                 │
│     (or DEFAULT_SIM_DATA_PARAMETERS if omitted)                         │
│                                                                         │
│  3. Build validated parameter space                                     │
│     ParameterDataset.to_parameter_space(parameters=specs)               │
│     → XSpaceVecoli (names, bounds, types)                               │
│                                                                         │
│  4. Branch on --live flag                                               │
│     ├─ live=True  → TimeseriesGeneratorVecoli + run_batch_and_cache()   │
│     └─ live=False → DataDrivenWrapper + run_and_cache()                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ (live mode shown)
                                     ▼
┌─ sampling.run_batch_and_cache() ────────────────────────────────────────┐
│                                                                         │
│  5. Generate Latin Hypercube Samples                                    │
│     generate_lhs_samples(param_space, n_samples, seed)                  │
│     → X: numpy array (n_samples × n_params), values in physical bounds  │
│                                                                         │
│  6. Run vEcoli simulations                                              │
│     TimeseriesGeneratorVecoli._run_batch(X)                             │
│     → Y: (n_samples × n_obs), Y_timeseries: list of per-sample arrays  │
│                                                                         │
│  7. Persist to disk                                                     │
│     PrecomputedCache(X, Y, Y_timeseries, ...).save()                    │
│     → cache_dir/{X.npy, Y.npy, metadata.json, timeseries/*.npy}        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Walkthrough

### 1. CLI Entry — `uq/cli.py:sample()`

The Typer command accepts:

| Argument | Default | Purpose |
|----------|---------|---------|
| `experiment_ids` | (required) | vEcoli experiment directory names |
| `--sim-base-path` | None | Root of vEcoli simulation outputs |
| `--cache-dir` | None | Where to write cached (X, Y) |
| `--n-samples` | 200 | Number of LHS parameter samples |
| `--seed` | 42 | RNG seed for reproducibility |
| `--live` | True | Run real vEcoli simulations (subprocess) |
| `--max-workers` | None | Parallelism (currently reserved for Nextflow) |
| `--max-duration` | 10800.0 | Max simulation wall-clock time (seconds) |
| `--generations` | 1 | Number of cell generations per sample |
| `--params-file` | None | JSON file with custom `SimDataParameter` specs |
| `--batch-dir` | None | Explicit working directory (temp if omitted) |

Validates that each `experiment_id` directory exists under `sim_base_path`, then delegates to `handlers.generate_samples()`.

### 2. Load Baseline Data — `handlers.generate_samples()`

```python
ds = initialize_data(experiment_ids, sim_base_path, observable_columns)
```

`initialize_data()` (in `uq/pipe.py`) loads:
- **`simData.cPickle`** from each experiment directory via `ParameterDataset` — the ParCa-generated parameter object that defines every tunable attribute of the vEcoli model.
- **Hive-partitioned Parquet** timeseries via `TimeseriesLoaderParquet` — baseline simulation outputs with metadata columns (`variant`, `lineage_seed`, `generation`, `agent_id`) extracted from directory structure.

Returns a `DatasetMultiExperiment` containing the parameter datasets, timeseries DataFrame, resolved observable column names, and a constructed `XSpaceVecoli`.

If `observable_columns` is not provided, defaults to:
```
listeners__mass__dry_mass, listeners__mass__cell_mass,
listeners__mass__volume, listeners__mass__growth
```

### 3. Resolve Parameters

**Generic mode (default):** Parameters are `SimDataParameter` objects, each defining:
- `name` — human-readable label
- `attr_path` — dot-separated path into `SimulationDataEcoli` (e.g. `"process.transcription.fraction_active_rnap_free"`)
- `bounds` — (min, max) tuple for sampling range
- `index` — (optional) for array-valued attributes

When `--params-file` is provided, specs are loaded from JSON. Otherwise, `DEFAULT_SIM_DATA_PARAMETERS` from `uq/pipeline/param_loader.py` supplies 5 defaults:

| Parameter | Attribute Path | Bounds |
|-----------|---------------|--------|
| `kinetic_objective_weight` | `process.metabolism.kinetic_objective_weight` | (0.0, 1.0) |
| `secretion_penalty_coeff` | `process.metabolism.secretion_penalty_coeff` | (0.0, 1.0) |
| `fraction_active_rnap_free` | `process.transcription.fraction_active_rnap_free` | (0.25, 0.47) |
| `fraction_active_rnap_bound` | `process.transcription.fraction_active_rnap_bound` | (0.12, 0.22) |
| `cell_dry_mass_fraction` | `mass.cell_dry_mass_fraction` | (0.25, 0.35) |

### 4. Build & Validate Parameter Space

```python
param_space = ds.x[0].to_parameter_space(parameters=sim_data_parameters)
```

`ParameterDataset.to_parameter_space()` in `uq/pipeline/param_loader.py`:
1. **Validates** every `attr_path` by traversing the actual `sim_data` object — raises `ValueError` if any path segment is missing.
2. Returns `XSpaceVecoli` (from `uq/inputs.py`) with `parameter_names`, `parameter_bounds`, `parameter_types`, and `_sim_data_parameters` stored as plain list attributes.

### 5. Generate Latin Hypercube Samples — `uq/sampling.py`

```python
X = generate_lhs_samples(parameter_space, n_samples, seed)
```

Uses `scipy.stats.qmc.LatinHypercube` to produce space-filling samples:
1. Draw `n_samples` points in the unit hypercube [0,1]^n_params
2. Scale each dimension to its physical bounds

Result: `X` with shape `(n_samples, n_params)` — each row is a parameter vector.

### 6. Run vEcoli Simulations — `uq/generators/vecoli.py`

This is the core of live mode. `TimeseriesGeneratorVecoli._run_batch(X)` executes all samples through vEcoli as a **single subprocess call**:

#### 6a. Write baseline `simData.cPickle`

The loaded `SimulationDataEcoli` object is pickled to `batch_dir/kb/simData.cPickle` — this serves as the unmodified baseline that vEcoli's workflow will load and mutate per-variant.

#### 6b. Encode samples as vEcoli variants

`_build_variants_section_generic(X, param_specs)` converts the LHS matrix into a vEcoli-native variant specification:

```json
{
  "sim_data_setattr": {
    "mutations": {
      "value": [
        {"process.transcription.fraction_active_rnap_free": 0.36, "mass.cell_dry_mass_fraction": 0.30},
        {"process.transcription.fraction_active_rnap_free": 0.40, "mass.cell_dry_mass_fraction": 0.28},
        ...
      ]
    }
  }
}
```

Each entry in `value` is one sample's mutations dict. vEcoli's `create_variants` with `op: "zip"` interprets each dict as a set of `setattr` calls on the baseline `sim_data` — producing N independent simulation variants.

#### 6c. Build workflow config

A complete vEcoli workflow config JSON is assembled:

```json
{
  "sim_data_path": "/path/to/batch_dir/kb/simData.cPickle",
  "experiment_id": "uq_batch",
  "emitter": "parquet",
  "emitter_arg": {"out_dir": "/path/to/batch_dir/output", "batch_size": 10800},
  "max_duration": 10800.0,
  "n_init_sims": 1,
  "generations": 1,
  "single_daughters": true,
  "variants": { ... }
}
```

#### 6d. Launch subprocess

```python
subprocess.run([sys.executable, "<vecoli_root>/runscripts/workflow.py", "--config", config_path])
```

vEcoli's `workflow.py` reads the config, loads the baseline `simData.cPickle`, applies each variant's mutations, and runs a full whole-cell simulation per variant. Outputs are emitted as **hive-partitioned Parquet** files:

```
output/uq_batch/history/
├── experiment_id=uq_batch/
│   ├── variant=0/lineage_seed=0/...    ← baseline (unmodified)
│   ├── variant=1/lineage_seed=0/...    ← sample 0
│   ├── variant=2/lineage_seed=0/...    ← sample 1
│   ...
```

**No `EcoliSim` object is ever held in UQ process memory.** The entire simulation lifecycle is isolated in the subprocess.

#### 6e. Collect and aggregate Parquet outputs

After the subprocess completes:

1. **Read** all Parquet files with `polars.read_parquet(hive_partitioning=True)` — automatically extracts `variant`, `lineage_seed`, `generation` columns from directory names.
2. **Filter** by variant index to isolate each sample's timeseries.
3. **Aggregate** each sample by time-mean:

```python
for i in range(n_samples):
    sample_df = df.filter(pl.col("variant") == i + 1)  # variant 0 = baseline
    ts = sample_df.select(observable_cols).to_numpy()   # shape (n_timesteps, n_obs)
    Y_timeseries.append(ts)
    Y_list.append(ts.mean(axis=0))                      # collapse time dimension

Y = np.vstack(Y_list)  # shape (n_samples, n_obs)
```

Returns `(Y, Y_timeseries)` — aggregated means for Phase 1, raw timeseries for Phase 2.

### 7. Persist to Disk — `PrecomputedCache`

`PrecomputedCache.save()` writes:

```
cache_dir/
├── X.npy               (n_samples × n_params)  — LHS sample points
├── Y.npy               (n_samples × n_obs)     — time-aggregated outputs
├── metadata.json        — parameter names, bounds, seed, counts
└── timeseries/
    ├── sample_0000.npy  (n_timesteps × n_obs)  — full timeseries, sample 0
    ├── sample_0001.npy  (n_timesteps × n_obs)  — full timeseries, sample 1
    ...
```

The cache is self-contained and portable — everything needed for downstream PCE fitting is on disk.

---

## How This Satisfies RFC006

RFC006 (Section 4) requires a framework that parametrizes three computational steps:
1. **Selection/extraction** of subsampled time points and variables
2. **Temporal aggregation** into output variables Y
3. **Sensitivity analysis** applied to the aggregated outputs

The two-stage sampling workflow maps directly to this:

### Stage 1: `uq sample` (this document)

| RFC006 Requirement | Implementation |
|---|---|
| **Input variables** from `SimulationDataEcoli` | `SimDataParameter` specs validated against actual sim_data via dot-path traversal |
| **Output variables** (transcriptome, proteome, fluxes, properties) | Observable columns from hive-partitioned Parquet (configurable) |
| **Wrapper functions callable from numerical libraries** | `TimeseriesGeneratorVecoli` wraps vEcoli as subprocess; `PrecomputedWrapper` loads cache for PCE |
| **Latin Hypercube Sampling** for PCE training data | `scipy.stats.qmc.LatinHypercube` with reproducible seed |
| **Aggregation strategy 1** (uniform across cells/times) | `Y[i] = timeseries[i].mean(axis=0)` — time-averaged per-sample |

### Stage 2: `uq quantify --precomputed-path ./uq_cache`

The cached `(X, Y)` feeds directly into the RFC006 pipeline (`uq/pipe.py`):

```python
cache = PrecomputedCache.load(precomputed_path)
```

| RFC006 Phase | What Happens | Data Source |
|---|---|---|
| **Phase 1 Steps 5a-7a** | Morris prescreening → PCE surrogate → Sobol indices (bulk) | `cache.X`, `cache.Y` |
| **Phase 1 Steps 3-4** | Aggregation strategies 1-3 + variance decomposition | Baseline Parquet timeseries |
| **Phase 2 Steps 5b-7b** | GSA-informed observable selection → Koopman DMD → per-stage PCE + Sobol | `cache.Y_timeseries` + variance decomposition residuals |

The separation is deliberate:
- **Sampling is expensive** (hours for live vEcoli, minutes to days for HPC batches) and should only run once.
- **Analysis is cheap** (seconds for PCE fitting) and can be iterated with different polynomial orders, bin counts, or observable selections without re-running simulations.

---

## Synthetic Mode (No Live Simulation)

When `--live` is omitted or set to False, `DataDrivenWrapper` builds a synthetic linear response surface from the baseline data's mean and standard deviation:

```
generate_samples() → aggregate_timeseries(baseline_data) → DataDrivenWrapper(mean, std)
                   → run_and_cache() → evaluate wrapper at each LHS point → cache
```

This is useful for quick demos and pipeline testing without requiring a vEcoli installation.

---

## HPC Batch Workflow (Alternative Path)

For large-scale runs on clusters, `uq sample` is decomposed into three CLI commands:

```bash
# 1. Export per-sample vEcoli configs (no simulation)
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200

# 2. Run on cluster via Nextflow, Slurm, etc.
#    Each config is an independent vEcoli run

# 3. Collect outputs into PrecomputedCache
uv run uq collect-results ./batch ./batch_outputs --cache-dir ./uq_cache

# 4. Analyze (same as above)
uv run uq quantify ... --precomputed-path ./uq_cache
```

`export_batch_configs()` and `collect_batch_results()` in `uq/generators/vecoli.py` handle the config serialization and Parquet collection respectively.

---

## Code Map

| Step | File | Function/Class |
|------|------|----------------|
| CLI entry | `uq/cli.py` | `sample()` |
| Handler | `uq/handlers.py` | `generate_samples()` |
| Data loading | `uq/pipe.py` | `initialize_data()` |
| Parameter specs | `uq/pipeline/param_loader.py` | `ParameterDataset.to_parameter_space()`, `DEFAULT_SIM_DATA_PARAMETERS` |
| Parameter validation | `uq/pipeline/param_loader.py` | `_validate_sim_data_parameters()` |
| Parameter space | `uq/inputs.py` | `XSpaceVecoli` |
| LHS generation | `uq/sampling.py` | `generate_lhs_samples()` |
| Live batch orchestration | `uq/sampling.py` | `run_batch_and_cache()` |
| vEcoli subprocess runner | `uq/generators/vecoli.py` | `TimeseriesGeneratorVecoli._run_batch()` |
| Variant encoding | `uq/generators/vecoli.py` | `_build_variants_section_generic()` |
| Workflow config builder | `uq/generators/vecoli.py` | `_build_workflow_config()` |
| Subprocess launch | `uq/generators/vecoli.py` | `_run_workflow()` |
| Synthetic mode | `uq/sampling.py` | `run_and_cache()` |
| Synthetic wrapper | `uq/wrappers.py` | `DataDrivenWrapper` |
| Cache persistence | `uq/sampling.py` | `PrecomputedCache` |
| Downstream pipeline | `uq/pipe.py` | `Pipeline.run()`, `execute_pipeline()` |
| HPC export | `uq/generators/vecoli.py` | `export_batch_configs()` |
| HPC collection | `uq/generators/vecoli.py` | `collect_batch_results()` |
