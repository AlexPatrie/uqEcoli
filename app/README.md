# UQ Results Dashboard

Interactive visualization of RFC006 UQ pipeline outputs. Built with [Marimo](https://marimo.io) and Plotly.

```bash
uv run marimo run app/dashboard.py
```

## Input

The dashboard loads a `uq_results.json` file produced by `PipelineResult.export()` (via `uv run uq quantify --export-path ./output`). It also reads the sibling binary files in the same export directory:

```
export_dir/
├── uq_results.json                 # Comprehensive summary (loaded by default)
├── population_surrogate/           # PCE coefficients + multi-indices (Phase 1)
│   ├── coefficients.npy
│   ├── multi_indices.npy
│   └── input_bounds.npy
├── cell_cycle_surrogate/           # PCE coefficients + multi-indices (Phase 2)
├── cell_cycle_sobol_stage_N/       # Per-stage Sobol indices (binary)
├── variance_decomposition.json
├── cell_cycle_profile.json
├── morris_indices/
├── metadata.json
├── koopman_spectrum.pdf
└── koopman_spectrum.html
```

The JSON file contains Sobol indices, variance decomposition, Morris screening, cell cycle profile, and surrogate metadata. The `.npy` files contain the PCE surrogate coefficients used by the interactive prediction panel.

## Panels

### 1. Solo Parameter Selector
Dropdown to isolate ("solo") one parameter. All downstream panels highlight the selected parameter and dim the others.

### 2. Header Strip
At-a-glance trust indicators: parameter count, cell cycle stages, surrogate R^2 values, Morris trajectory count.

### 3. Parametric EQ
Per-stage total-order Sobol indices (S_Ti) plotted as spline curves across the cell cycle variable theta. The soloed parameter is thick with a fill; others are ghosted. A dashed overlay shows the first-order index (S_i), and the gap between them reveals interaction effects.

### 4. Channel Strip
Three sub-panels: population-level Sobol faders, Morris mu\*/sigma scatter (with linear/nonlinear classification), and variance decomposition donut chart.

### 5. Sensitivity Spectrogram + Cell Cycle Profile
Top: heatmap of S_Ti (parameters x stages). Bottom: per-stage mass and growth rate means with B/C/D period annotations. Shared x-axis links sensitivity patterns to biological observables.

### 6. Observable Relevance + Surrogate Specs
Which observables were selected as cell-cycle-relevant (Step 5b), and a comparison table of the two PCE surrogates.

### 7. Mixer
Grouped bar chart of S_Ti per stage for all parameters. Complementary to the EQ — the EQ shows trends across stages for one parameter; the mixer compares parameters at each stage.

### 8. Sidechain / Interaction Analyzer
Plots (S_Ti - S_i) across stages — the fraction of variance from parameter interactions. Reveals where parameters amplify or cancel each other's effects.

### 9. PCE Prediction EQ (Surrogate Knobs)
**Interactive surrogate evaluation.** This is the panel where parameter values directly control predicted outputs.

## How the PCE Prediction EQ Works

### What it does

One slider per input parameter, bounded by the parameter space limits from the pipeline. Dragging any slider instantly re-evaluates the PCE surrogate and updates the response curves for **all** parameters simultaneously.

### How it connects to the pipeline

The PCE surrogate is the polynomial fitted during Phase 1 of `uq quantify`:

1. **During the pipeline** (`run_phase1`): N Latin Hypercube samples are drawn from the parameter space. Each sample is evaluated by the simulation function (either live EcoliSim or precomputed cache). The resulting (X, Y) pairs are used to fit Legendre polynomial coefficients via least-squares regression.

2. **After the pipeline** (`PipelineResult.export`): The fitted coefficients, multi-index matrix, and input bounds are serialized to `population_surrogate/coefficients.npy`, `multi_indices.npy`, and `input_bounds.npy`.

3. **In the dashboard**: These files are loaded as numpy arrays. When a slider changes, the dashboard evaluates:

```
Y_hat(x) = sum_alpha c_alpha * prod_i P_{alpha_i}(x_i)
```

where `c_alpha` are the stored coefficients, `alpha` are the multi-indices, and `P_n` are Legendre polynomials evaluated via the standard recurrence relation. The parameter vector `x` is normalized from physical bounds to [-1, 1] before evaluation.

### What the response curves show

For each parameter `p_i`, the dashboard sweeps `p_i` across its full range while holding all other parameters at their current slider values, producing a response curve. All curves share a normalized x-axis (0 = parameter minimum, 1 = parameter maximum) so they can be overlaid.

- **Curve shape** = how the predicted output changes as that parameter varies
- **Curve steepness** = local sensitivity of the output to that parameter
- **Cross-parameter effects** = when you move one slider and another parameter's curve changes shape, that's the interaction terms in the PCE (the same interactions quantified by S_Ti - S_i in the Sidechain panel)

### Cost

The polynomial evaluation is pure arithmetic — no simulation, no I/O. Each slider change triggers ~180 multiply-add operations (60 sweep points x 3 parameters). This is the design intent of the PCE surrogate approach: pay the simulation cost once during `uq generate-samples`, then explore the fitted response surface interactively.

### Relationship to other panels

| Panel | Shows | Derived from |
|-------|-------|-------------|
| Parametric EQ | *Which* parameters matter at each stage | Sobol indices (from PCE coefficients) |
| Sidechain | *How much* parameters interact | S_Ti - S_i (from PCE coefficients) |
| **PCE Prediction EQ** | *What happens* when you set specific values | Direct PCE evaluation (using the same coefficients) |

The Sobol panels answer "what fraction of variance does this parameter explain?" The PCE Prediction EQ answers "if I set this parameter to 3.5, what output do I get?" They use the same underlying polynomial — one analytically decomposes its variance, the other evaluates it at a point.

## Generating Pipeline Artifacts

### Quick start (synthetic data for demo)

```bash
uv run pytest tests/test_export_artifacts.py -s -v
```

Writes artifacts to `examples/uq_artifacts/test_export_output/`. The dashboard defaults to this path.

### From real simulations

```bash
# Stage 1: generate and cache LHS samples (compute-intensive)
uv run uq generate-samples \
    api_simulation_default mecillinam test_violacein_with_metabolism \
    --sim-base-path /path/to/sims \
    --cache-dir ./uq_cache \
    --n-samples 200 --live --max-workers 4

# Stage 2: fit PCE and export (fast)
uv run uq quantify \
    api_simulation_default mecillinam test_violacein_with_metabolism \
    --outdir-root /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_output

# View results
uv run marimo run app/dashboard.py
# Then paste the path to ./uq_output/uq_results.json in the file loader
```

### HPC batch workflow

```bash
# 1. Export per-sample configs for cluster
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200

# 2. Run on cluster (Nextflow, Slurm, etc.)

# 3. Collect results into cache
uv run uq collect-results ./batch ./batch_outputs --cache-dir ./uq_cache

# 4. Analyze and export
uv run uq quantify exp1 \
    --outdir-root /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_output
```
