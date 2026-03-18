# vEcoli Uncertainty Quantification (UQ) Framework

Sensitivity analysis and uncertainty quantification for [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell simulations, built on Polynomial Chaos Expansion (PCE) surrogates and Sobol variance decomposition.

**RFC006 — Milestone 08.4.2**

## What It Does

The framework answers two questions about your simulation outputs:

1. **Population level** — Which input parameters drive the most variance in bulk outputs (e.g., dry mass, growth rate)?
2. **Cell cycle level** — How does parameter importance shift across cell cycle stages?

It does this through a 7-step pipeline:

```
Parameter Space → Load Data → Aggregate (4 strategies) → Variance Decomposition
                                                                │
Sobol Indices  ←  PCE Surrogate  ←  Morris Screening  ←────────┘
```

## Installation

```bash
uv sync
```

---

## CLI Reference

The `uq` CLI exposes the following commands:

| Command | Purpose |
|---------|---------|
| `uq generate-samples` | Generate LHS samples and cache (X, Y) pairs to disk |
| `uq quantify` | Run the full RFC006 UQ pipeline (from data or cache) |
| `uq demo` | Run a demo pipeline with default experiments |
| `uq export-configs` | Export per-sample vEcoli configs for HPC/Nextflow batch execution |
| `uq collect-results` | Assemble completed HPC batch outputs into a precomputed cache |
| `uq configure-pipeline` | Write a default pipeline config JSON |
| `uq readme` | Print the RFC006 workflow diagram |

---

## Getting Started

### 1. Generate Samples

Generate Latin Hypercube samples and evaluate them against a simulation function, caching the results to disk:

```bash
uv run uq generate-samples \
    api_simulation_default mecillinam test_violacein_with_metabolism \
    --sim-base-path /path/to/vEcoli/api_integration/sims \
    --cache-dir ./uq_cache \
    --n-samples 200
```

**Important:** By default this uses a **synthetic response surface** (`DataDrivenWrapper`) — it builds a linear model from the statistics of your existing Parquet data and evaluates it instantly. This is useful for testing the pipeline but does not produce biologically meaningful sensitivity indices.

To run **real vEcoli simulations** at each sample point, add the `--live` flag:

```bash
uv run uq generate-samples \
    api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 50 \
    --live \
    --max-duration 300 \
    --max-workers 4
```

**`generate-samples` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sim-base-path` | (required) | Root directory containing simulation Parquet outputs |
| `--cache-dir` | None | Where to write cached X.npy, Y.npy, metadata.json |
| `--n-samples` | 200 | Number of LHS sample points |
| `--seed` | 42 | Random seed for reproducible sampling |
| `--live` | off | Run real `EcoliSim` per sample (requires `ecoli` package) |
| `--max-duration` | 10800 | Simulation wall-clock limit in seconds (live mode) |
| `--generations` | 1 | Generations per simulation (live mode) |
| `--max-workers` | None | Parallel workers for evaluation (None = sequential) |
| `--include-vio` | auto | Include violacein pathway parameters (auto-detected from sim_data) |
| `--include-mecillinam` | True | Include mecillinam concentration parameter |
| `--observable-columns` | mass cols | Which output columns to extract |

### 2. Run the Pipeline

Once you have a cache, run the full UQ pipeline:

```bash
uv run uq quantify \
    api_simulation_default mecillinam test_violacein_with_metabolism \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results
```

This loads the cached (X, Y) data, fits PCE surrogates, computes Sobol indices, and prints a rich terminal report. No simulation calls are made.

**`quantify` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sim-base-path` | (required) | Root simulation output directory |
| `--precomputed-path` | None | Path to cache from `generate-samples` |
| `--export-path` | None | Directory to write artifacts (Sobol, surrogates, Koopman PDF) |
| `--lb-generation` | 2 | Skip initial transient generations |
| `--lb-time` | 100.0 | Skip early transient timesteps |
| `--n-bins` | 10 | Number of cell cycle bins (Phase 2) |
| `--pce-polynomial-order` | 3 | PCE polynomial order |
| `--n-samples` | 20 | Samples for PCE fitting (ignored if --precomputed-path used) |
| `--expected-cycle-time` | 3600.0 | Expected cell cycle duration in seconds |
| `--pce-n-trajectories` | 10 | Morris screening trajectories |
| `--pce-n-selected-params` | 5 | Top-K parameters from Morris screening |

### 3. Demo Mode

Run the pipeline with default experiment IDs and built-in paths:

```bash
# Synthetic sample generation demo (fast)
uv run uq demo

# Full pipeline demo
uv run uq demo --demo-type full
```

---

## HPC Batch Workflow

For production-scale analysis where each simulation takes minutes to hours, use the three-step HPC batch workflow. This avoids running simulations locally by exporting per-sample configs that a job scheduler (Nextflow, Slurm, etc.) can execute in parallel.

### Step 1: Export Configs

Generate LHS samples, apply parameter variants to `sim_data`, and write per-sample configs:

```bash
uv run uq export-configs \
    /path/to/simData.cPickle \
    ./batch \
    --n-samples 200 \
    --include-vio \
    --include-mecillinam \
    --generations 1
```

This produces:

```
batch/
├── configs/          # Per-sample JSON configs (sample_0000.json, ...)
├── sim_data/         # Per-sample pickled sim_data with variants applied
└── metadata.json     # Maps sample index → parameter values
```

**`export-configs` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `sim_data_path` | (positional) | Path to baseline `simData.cPickle` |
| `batch_dir` | (positional) | Output directory for batch configs |
| `--n-samples` | 200 | Number of LHS sample points |
| `--seed` | 42 | Random seed |
| `--include-vio` | True | Include violacein pathway parameters |
| `--include-mecillinam` | True | Include mecillinam concentration parameter |
| `--base-config-path` | None | Optional base JSON config to merge into |
| `--generations` | 1 | Generations per simulation |
| `--emitter` | parquet | Output emitter type |

### Step 2: Run on Cluster

Submit the configs to your job scheduler. Each sample is an independent simulation:

```bash
# Example with Nextflow (adapt to your scheduler)
nextflow run vEcoli_batch.nf \
    --configs ./batch/configs \
    --sim_data ./batch/sim_data \
    --outdir ./batch_outputs

# Example with a simple parallel loop
for cfg in ./batch/configs/sample_*.json; do
    ecoli_master_sim --config "$cfg" --outdir ./batch_outputs/$(basename "$cfg" .json) &
done
wait
```

Each job reads its `sim_data` pickle + JSON config, runs `EcoliSim`, and writes Parquet output to the output directory.

### Step 3: Collect Results

Assemble the per-sample Parquet outputs into a `PrecomputedCache`:

```bash
uv run uq collect-results \
    ./batch \
    ./batch_outputs \
    --cache-dir ./uq_cache
```

**`collect-results` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `batch_dir` | (positional) | Directory from `export-configs` (contains metadata.json) |
| `output_dir` | (positional) | Root dir with per-sample Parquet outputs |
| `--observable-columns` | None | Which columns to extract (default: mass columns) |
| `--cache-dir` | `{batch_dir}/cache` | Where to save the assembled cache |

### Step 4: Analyze

Run the pipeline against the collected cache — this is the same `quantify` command:

```bash
uv run uq quantify \
    api_simulation_default \
    --sim-base-path /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results
```

### Full HPC workflow at a glance

```
export-configs              →  run on cluster  →  collect-results  →  quantify
(write per-sample configs)     (parallel sims)    (assemble cache)    (fit PCE + Sobol)
         │                          │                    │                  │
    batch/configs/             Parquet output        uq_cache/          uq_results/
    batch/sim_data/            per sample            X.npy, Y.npy       sobol indices
    batch/metadata.json                              metadata.json      koopman PDF
```

---

## Python API

```python
from uq import XSpaceVecoli
from uq.pipeline.workflow import execute_pipeline

result = execute_pipeline(
    experiment_ids=["mecillinam"],
    sim_base_path="/path/to/vEcoli/api_integration/sims",
    simulation_func=your_simulation_wrapper,
    polynomial_order=3,
    n_samples=200,
    export_path="./uq_results",
)

# Population-level Sobol indices (Phase 1)
for name, value in result.population.sobol_indices[0].select(n=5):
    print(f"{name}: {value:.4f}")

# Variance decomposition
print(result.variance_decomposition)

# Per-cell-cycle-stage Sobol indices (Phase 2)
for i, stage_sobol in enumerate(result.cell_cycle.sobol_indices):
    print(f"Stage {i}: {stage_sobol.select(n=3)}")

# Reload results later
from uq.pipeline.models import PipelineResult
loaded = PipelineResult.from_export("./uq_results")
```

## Pipeline Output

The pipeline produces a `PipelineResult` with two profiles:

| Profile | What it tells you | Key outputs |
|---------|-------------------|-------------|
| **Population** (Phase 1) | Which parameters drive bulk variance | 1 set of Sobol indices + PCE surrogate |
| **Cell Cycle** (Phase 2) | How sensitivity varies across the cell cycle | *n* sets of Sobol indices (one per stage) + PCE surrogate |

When `--export-path` is provided, the following artifacts are written:

| Artifact | Description |
|----------|-------------|
| `population_sobol/` | Serialized Phase 1 Sobol indices |
| `cell_cycle_surrogate/` | Serialized Phase 2 PCE surrogate |
| `koopman_spectrum.pdf` | 4-panel Koopman spectral decomposition |
| `variance_decomposition.json` | Variance decomposition results |
| `morris_indices/` | Morris screening results (if applicable) |

## Input Parameters

| Parameter | Description |
|-----------|-------------|
| Vio expression factor | Violacein pathway gene expression level |
| Vio translation efficiency | Translation efficiency for vio pathway |
| Mecillinam concentration | Antibiotic concentration timeline |
| Gene knockouts | Genes deleted or silenced at translation level |

## Key Outputs Analyzed

- **Transcriptome** — mRNA cistron counts
- **Proteome** — monomer counts
- **Metabolic fluxes** — FBA exchange/internal fluxes
- **Cell properties** — mass, volume, growth rate

## Documentation

| Resource | Location |
|----------|----------|
| Full RFC specification | [`readmes/RFC006.md`](readmes/RFC006.md) |
| Extended technical context | [`readmes/CONTEXT.md`](readmes/CONTEXT.md) |
| Sphinx docs | [`docs/`](docs/) |
| End-to-end example | [`examples/uq_pipeline.py`](examples/uq_pipeline.py) |
| Tutorials (Marimo) | [`tutorials/`](tutorials/) |

## Running Tests

```bash
uv run pytest tests/
```

## Dependencies

Built on [UQPy](https://uqpyproject.readthedocs.io/) and [PyTUQ](https://sandialabs.github.io/pytuq/) for PCE construction and Sobol analysis, with [Polars](https://pola.rs/) and [DuckDB](https://duckdb.org/) for data handling.
