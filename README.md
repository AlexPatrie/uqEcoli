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

## How it works

### A.) The full workflow provided by `uq quantify` and `uq.pipe.pipeline()` allows for sensitivity analysis on a non-linear, stochastic system.

- The pipeline performs global sensitivity analysis (Morris screening + PCE + Sobol indices) on vEcoli, which is indeed a nonlinear, stochastic whole-cell model. The stochasticity comes from multiple lineage seeds and cell-to-cell variation; the nonlinearity from the biochemical kinetics.

### B.) Sensitivity analysis on this nonlinear stochastic system is enabled by PCE(Polynomial Chaos Expansion), which builds polynomial surrogates that capture nonlinear input-output relationships. Koopman/DMD is used specifically in Phase 2 to extract a data-driven cell cycle coordinate for stratification, enabling stage-resolved sensitivity analysis.

- Phase 1 (population-level GSA) performs sensitivity analysis using PCE surrogates and Sobol indices
   without any Koopman analysis at all. It works on aggregated statistics (uniform mean/std across
  cells). PCE itself handles nonlinearity just fine — it's a polynomial approximation of the
  input→output map, and Sobol indices decompose variance regardless of whether the underlying system is
   linear or nonlinear. Phase 1 is the primary sensitivity analysis and does not use Koopman.

- Phase 2 (cell-cycle-stratified GSA) is the only phase that uses Koopman/DMD. Its role is specifically to provide a cell cycle coordinate θ ∈ [0,1] so that data can be binned by cell cycle stage. Then PCE+Sobol are run per stage. Koopman here is a binning/stratification tool, not the mechanism that enables sensitivity analysis on a nonlinear system.

### C.) The Koopman operator is an infinite-dimensional linear operator that acts on the space of observable functions. For a nonlinear dynamical system x_{t+1} = F(x_t), the Koopman operator K propagates observables: (Kg)(x) = g(F(x)). Linearity in this infinite-dimensional function space is exact, regardless of the nonlinearity of F.

- It is not parameterized by a single timepoint as you stated. The Koopman operator K acts on
  observable functions: `(Kg)(x) = g(F(x))` where F is the one-step dynamics map. It describes how
  functions of the state evolve forward by one timestep. For continuous-time systems, there's a Koopman
   semigroup K(t), but it's a family parameterized by duration, not "a given timepoint."

- The operator acts on the space of observables (functions of state), not on the state space itself.
It doesn't map `Y[t_i] → Y[t_i+1]` directly; it maps functions of state forward.

### D.) Since K is infinite-dimensional, any computational method must approximate it with a finite-rank matrix. The resulting eigendecomposition naturally yields complex eigenvalues (encoding oscillation frequency and growth/decay rate) and complex eigenvectors (Koopman modes). Taking real parts for reconstruction is standard linear algebra, not a special 'decryption' — it's the same operation as reconstructing a real signal from its complex Fourier coefficients

- Finite-rank approximation is needed because we can't work with infinite-dimensional operators
  computationally — this part is correct.

- Complex eigenvalues arise because the Koopman operator (like any linear operator) can have complex eigenvalues, which encode oscillatory dynamics. The imaginary part gives frequency, the real part gives growth/decay. This is standard spectral theory, not something unique to "needing a decryption key." It's the same reason a rotation matrix has complex eigenvalues.

### E.) DMD finds the best-fit linear operator `A` such that `x_{t+1} ≈ Ax_t`, then eigendecomposes `A`. The eigenvalues encode frequencies and growth rates; the eigenvectors are spatial modes. This is a finite-rank approximation to the Koopman operator, not a reversible transform

What DMD actually does (as implemented in `koopman.py:239-258`):

1. Takes snapshot matrices `X = [x_1, ..., x_{N-1}]` and `X' = [x_2, ..., x_N]`
2. Finds the best-fit linear operator A such that X' ≈ AX (via SVD-based pseudoinverse)
3. Computes the eigendecomposition of A → eigenvalues λ_i and eigenvectors (modes) φ_i

- This is a data-driven eigendecomposition of an approximate linear dynamics operator, not a  time-to-frequency transform. It's lossy (rank-truncated SVD), so it is not reversible — the  reconstruction error in KoopmanSpectrum.reconstruction_error quantifies this loss.

### F.) DMD eigenvalues `λ_i` do encode frequencies: `ω_i = Im(log(λ_i))/(2π)`, as implemented at `koopman.py:86`. And the DMD modes are indeed finite-rank approximations of Koopman eigenfunctions projected onto the observable space.

- Together both eigenvalues and eigenvectors define a modal decomposition, NOT a spectrum in the Fourier sense.

### G.) DMD's final/second step recomposes/sums all harmonics from approximated frequency domain frequencies (`ω_i = Im(log(λ_i))/(2π)`) back to the original system's signal domain.

`KoopmanSpectrum.reconstruct()` at line (154-172) is exactly this:

```
x(t) = Σ_j amplitude_j × mode_j × λ_j^(t/dt)
```

- This sums contributions from each mode (weighted by amplitude, with time evolution governed by the
eigenvalue).

- This is analogous to inverse Fourier synthesis but using DMD modes/eigenvalues instead of sinusoids. However, it's important to note this reconstruction is approximate — the number of modes is finite and rank-truncated.


## Installation

```bash
uv sync
```

---

## `uq_simple` — Simplified Pipeline (RFC006 Compliant)

The `uq_simple` package provides a scientifically transparent implementation of all four RFC006 aggregation strategies, built on [PyTUQ's UQPC workflow](https://sandialabs.github.io/pytuq/apps/uqpc.html) (Sandia National Labs). No Koopman spectral analysis.

| Strategy | Aggregation | Purpose |
|----------|-------------|---------|
| 1 (Uniform) | Time-mean across all cells/times | Baseline bulk sensitivity |
| 2 (By Generation) | Mean per generation | Controls for convergence toward steady-state growth |
| 3 (By Lineage Seed) | Mean per lineage seed | Controls for exogenous stochastic variance |
| 4 (Growth-Stratified) | Binned by θ = normalized log(dry_mass) | How parameter importance changes as cells grow |

PCE surrogates are fitted via `pytuq.surrogates.pce.PCE` with configurable regression: `lsq` (least squares, default), `bcs` (Bayesian Compressed Sensing — sparse), or `anl` (analytical with uncertainty).

### Usage

```bash
# Stage 1: generate + cache LHS samples (same as uq sample)
uv run uq-simple sample api_simulation_default \
    --sim-base-path /path/to/sims --cache-dir ./uq_cache \
    --n-samples 50 --live --generations 3

# Stage 2: PCE / Sobol across all 4 strategies
uv run uq-simple quantify api_simulation_default \
    /path/to/sims ./uq_cache ./uq_results \
    --polynomial-order 2 --n-bins 10 --regression lsq

# Stage 3: interactive dashboard
uv run uq-simple dashboard ./uq_results/uq_results.json
```

### Python API

```python
from uq.sampling import PrecomputedCache
from uq.pipe import initialize_datasets
from uq_simple.pipeline import run_pipeline

cache = PrecomputedCache.load("./uq_cache")
ds = initialize_datasets(experiment_ids=["exp1"], sim_base_path="/path/to/sims")
result = run_pipeline(
    cache=cache,
    param_space=ds.parameter_space,
    export_path="./uq_results",
    regression="lsq",  # or "bcs", "anl"
)

# Strategy 1: bulk Sobol
result.population_sobol.total_order

# Strategy 2: per-generation Sobol
result.per_generation_sobol  # dict[int, SobolIndices]

# Strategy 3: per-seed Sobol
result.per_seed_sobol  # dict[int, SobolIndices]

# Strategy 4: growth-stratified Sobol
result.per_stage_sobol  # list[SobolIndices]
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

Generate Latin Hypercube samples and evaluate them against vEcoli, caching the results to disk:

```bash
# Default: 5 physiologically relevant sim_data parameters
uv run uq sample \
    api_simulation_default \
    --sim-base-path /path/to/vEcoli/api_integration/sims \
    --cache-dir ./uq_cache \
    --n-samples 50 --live --max-workers 4
```

**Custom parameters** — create a JSON file specifying any `SimulationDataEcoli` attributes:

```json
[
  {"name": "kinetic_obj_wt", "attr_path": "process.metabolism.kinetic_objective_weight", "bounds": [0.0, 1.0]},
  {"name": "rnap_active_free", "attr_path": "process.transcription.fraction_active_rnap_free", "bounds": [0.1, 1.0]}
]
```

```bash
uv run uq sample api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 50 --live --params-file my_params.json
```

**Legacy vio/mecillinam mode** (for violacein/antibiotic-enabled sim_data):

```bash
uv run uq sample api_simulation_default \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 50 --live --include-vio --include-mecillinam
```

Live mode runs vEcoli simulations as **subprocesses** — no `EcoliSim` is held in the UQ process memory. Without `--live`, uses a synthetic response surface for fast testing.

**`sample` options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sim-base-path` | (required) | Root directory containing simulation Parquet outputs |
| `--cache-dir` | None | Where to write cached X.npy, Y.npy, metadata.json |
| `--n-samples` | 200 | Number of LHS sample points |
| `--seed` | 42 | Random seed for reproducible sampling |
| `--live` | on | Run real vEcoli simulations as subprocesses |
| `--params-file` | None | JSON file with custom `SimDataParameter` specs |
| `--max-duration` | 10800 | Simulation wall-clock limit in seconds (live mode) |
| `--generations` | 1 | Generations per simulation (live mode) |
| `--max-workers` | None | Parallel subprocesses (None = sequential) |
| `--include-vio` | None | Include violacein parameters (legacy mode) |
| `--include-mecillinam` | None | Include mecillinam parameter (legacy mode) |
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
| `uq_results.json` | Comprehensive summary (Phase 1 + Phase 2 Sobol, cell cycle profile, Morris, surrogate metadata) |
| `population_surrogate/` | PCE coefficients + multi-indices (.npy) — used by dashboard for interactive prediction |
| `cell_cycle_surrogate/` | Phase 2 PCE surrogate |
| `population_sobol/` | Serialized Phase 1 Sobol indices |
| `cell_cycle_sobol_stage_N/` | Per-stage Sobol indices (one directory per θ-bin) |
| `cell_cycle_profile.json` | Per-stage observable means in physical units |
| `koopman_spectrum.pdf` | 4-panel Koopman spectral decomposition |
| `koopman_spectrum.html` | Interactive Plotly version |
| `variance_decomposition.json` | Variance decomposition results |
| `morris_indices/` | Morris screening results (if applicable) |
| `metadata.json` | Pipeline metadata (n_stages, param names, stratification) |

## Interactive Dashboard

```bash
# Tkinter DAW (default) — draggable parameter markers on response curves
uv run uq dashboard
uv run uq dashboard --results-path ./uq_results/uq_results.json

# Marimo notebook — slider-reactive, with collapsible info panels
uv run uq dashboard --run-mode mo
```

The dashboard loads export artifacts and provides interactive PCE surrogate exploration: drag parameter values on response curves (tk) or use sliders (marimo) to see predicted outputs, per-stage waveforms, and sensitivity spectrograms update in real time. All computations use pipeline outputs only (PCE coefficients + Sobol indices). See [`app/README.md`](app/README.md) for details.

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
| Full RFC specification | [`readmes/RFC006.md`](readmes/start/tools/RFC006.md) |
| Extended technical context | [`readmes/CONTEXT.md`](readmes/start/tools/CONTEXT.md) |
| Dashboard guide | [`app/README.md`](app/README.md) |
| Sphinx docs | [`docs/`](docs/) |
| End-to-end example | [`examples/uq_pipeline.py`](examples/uq_pipeline.py) |
| Tutorials (Marimo) | [`tutorials/`](tutorials/) |

## Running Tests

```bash
uv run pytest tests/
```

## Dependencies

Built on [UQPy](https://uqpyproject.readthedocs.io/) and [PyTUQ](https://sandialabs.github.io/pytuq/) for PCE construction and Sobol analysis, with [Polars](https://pola.rs/) and [DuckDB](https://duckdb.org/) for data handling.
