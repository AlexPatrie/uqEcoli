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

The polynomial evaluation is pure arithmetic — no simulation, no I/O. Each slider change triggers ~180 multiply-add operations (60 sweep points x 3 parameters). This is the design intent of the PCE surrogate approach: pay the simulation cost once during `uq sample`, then explore the fitted response surface interactively.

### Per-stage prediction curve

Below the sensitivity spectrogram (heatmap), a green line chart shows the predicted output at each cell cycle stage, recomputed live as you drag sliders or dots. This is the bridge between the population-level prediction (one number) and the per-stage Sobol indices (static variance fractions).

The per-stage prediction combines three pipeline outputs — no synthetic data:

```
Y_hat_k = Y_hat_pop + sum_i [ |dY/dx_i| * (x_i - midpoint) * S_Ti^(k) ]
```

| Term | Source | Meaning |
|------|--------|---------|
| `Y_hat_pop` | PCE surrogate `predict(x)` | Bulk prediction at current slider values |
| `\|dY/dx_i\|` | Finite difference on PCE | Local sensitivity at current operating point |
| `(x_i - midpoint)` | Slider position in germ space | How far the parameter deviates from center |
| `S_Ti^(k)` | Pipeline Sobol indices | How much parameter i matters at stage k |

This distributes the population prediction across stages, weighted by each parameter's per-stage importance. When you increase mecillinam (high S_Ti at stages 7-9), the late-cycle stages respond more than the early ones. The peak stage is highlighted with a labeled dot.

### Relationship to other panels

| Panel | Shows | Derived from |
|-------|-------|-------------|
| Parametric EQ | *Which* parameters matter at each stage | Sobol indices (from PCE coefficients) |
| Sidechain | *How much* parameters interact | S_Ti - S_i (from PCE coefficients) |
| **PCE Prediction EQ** | *What happens* when you set specific values | Direct PCE evaluation (using the same coefficients) |
| **Stage Prediction** | *Where in the cell cycle* the effect concentrates | PCE + Sobol (population prediction weighted by per-stage S_Ti) |

The Sobol panels answer "what fraction of variance does this parameter explain?" The PCE Prediction EQ answers "if I set this parameter to 3.5, what output do I get?" The stage prediction answers "at this configuration, which cell cycle stage is most affected?" All three use the same pipeline outputs — they decompose the same information from different angles.

### What the heatmap x-axis is

The cell cycle stages (θ-bins) were computed during the pipeline by:

1. Running simulations at LHS sample points → getting timeseries Y(t)
2. Running Koopman DMD on those timeseries → extracting `θ(t) ∈ [0,1]` per timepoint
3. Binning timepoints by `θ` into 10 stages (`n_bins`)
4. Fitting PCE and computing Sobol indices within each bin separately

#### So `S_Ti^(k)` for parameter `X_i` at stage `k` answers:

>> "Of the variance in `Y` observed among timepoints that fall in `θ`-bin `k`, what fraction is attributable to `X_i?`"

### What the spectrogram shows when you change `X_i`

The heatmap colors (`S_Ti` values) don't change — they're fixed pipeline outputs. _**What changes is**_:

1. The white cursor dot on `X_i`'s row slides to reflect your new slider position
2. The green prediction curve below reshapes _**because the per-stage prediction formula weights the population prediction by `S_Ti^(k)`**_:

```
Y_hat_k = Y_hat_pop + sum_i [ |dY/dx_i| * (x_i - midpoint) * S_Ti^(k) ]
```

> So if mecillinam has `S_Ti = 0.05` at stage `0` but `S_Ti = 0.55` at stage `9`, and you crank mecillinam from `2` to `8`, the green curve shifts _much more_ at stage `9` than at stage `0`.
> The green curve is answering:
>> "given that I changed X_i, which stages feel it most?" — and the answer comes directly from the
per-stage Sobol indices that the pipeline already computed."

### To state it plainly

- The spectrogram tells you: "Parameter `X_i` explains `S_Ti^(k)` fraction of output variance at cell cycle stage `k`."

- The green prediction curve tells you: "If I set `X_i` to this specific value, stage `k`'s predicted output shifts by an amount proportional to `S_Ti^(k)`."

- The first is a global variance attribution (pipeline output). The second applies that attribution to  your specific parameter choice (dashboard computation using pipeline outputs only). The curve doesn't  recompute `θ` or re-run Koopman — it uses the already-computed per-stage Sobol weights to distribute the effect of your slider change across stages.

- Dragging `X_i` shows you which cell cycle stages are most affected by the value of `X_i`, exactly because the per-stage Sobol indices encode that stage-specific sensitivity.

### Observable-domain heatmap (Tkinter mode)

The Tkinter dashboard (`--run-mode tk`) adds a second heatmap that shows the same cell cycle stages on the x-axis, but with **output observables** on the y-axis and **physical-unit values** as cell colors — not variance fractions. This heatmap updates in lock-step with the parameter sliders.

The link between the two heatmaps is valid because they share the same decomposition chain:

1. The pipeline computed `cell_cycle_profile` by binning real simulation timeseries into θ-stages (Step 6c). This gives **baseline per-stage means** for each observable — e.g., `dry_mass_mean[7] = 1.653` at stage 7. These are measured in the original physical units of the simulation output.

2. The pipeline also computed `S_Ti^(k)` — the fraction of variance in those same θ-binned outputs attributable to each input parameter (Step 7b). The sensitivity spectrogram visualizes these fractions.

3. The observable heatmap modulates the baselines from (1) using the Sobol weights from (2):

```
Y_obs_k(x) = baseline_obs_k * (1 + sum_i [ |dY/dx_i| * (x_i - mid) * S_Ti^(k) ] / |baseline_obs_k|)
```

This is valid because `S_Ti^(k)` quantifies exactly how much parameter `X_i` influences the output at stage `k`. If `S_Ti^(7) = 0.55` for mecillinam at stage 7, then 55% of the variance in stage-7 output comes from mecillinam — so changing mecillinam's value should shift stage-7's predicted output proportionally to that 55% attribution. The Sobol index IS the weight.

The two heatmaps are therefore dual views of the same pipeline decomposition:

| Heatmap | Y-axis | Cell value | Units | Changes with sliders? |
|---------|--------|-----------|-------|----------------------|
| Sensitivity spectrogram | Parameters | `S_Ti^(k)` | Variance fraction [0,1] | No (pipeline constant) |
| Observable domain | Observables | `Y_obs_k(x)` | Physical (fg, 1/s, etc.) | Yes (modulated by slider position) |

The sensitivity spectrogram answers "which parameter drives variance at which stage." The observable heatmap answers "what does the output actually look like at each stage for my chosen parameter values." They share the same x-axis (θ-bins), the same underlying data (θ-binned simulation outputs), and the same attribution weights (per-stage Sobol indices). One is the explanation; the other is the consequence.

## Generating Pipeline Artifacts

### Quick start (synthetic data for demo)

```bash
uv run pytest tests/test_export_artifacts.py -s -v
```

Writes artifacts to `examples/uq_artifacts/test_export_output/`. The dashboard defaults to this path.

### From real simulations

```bash
# Stage 1: sample + run vEcoli (compute-intensive)
uv run uq sample /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --n-samples 50

# Stage 2: fit PCE + export (fast, auto-generates report.html)
uv run uq quantify /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --export-path ./uq_results

# View results — pick any surface
uv run uq dashboard --results-path ./uq_results   # tkinter DAW
uv run uq report --results-path ./uq_results       # standalone HTML report
open ./uq_results/report.html                       # or just open the auto-generated report
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
