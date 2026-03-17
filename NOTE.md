# Running the Full UQ Workflow

## Overview

The UQ pipeline answers two questions about your vEcoli simulations:

1. **Population (Phase 1):** "Which input parameters drive the most variance in bulk output?"
2. **Cell Cycle (Phase 2):** "Which parameters matter most *within each cell cycle stage*?"

The pipeline has two stages: **sample generation** (slow, run once) and **analysis** (fast, repeatable).

---

## Stage 1: Generate Samples

Generate parameter samples, evaluate the simulation function at each sample point, and cache the results. This is the expensive step.

```bash
uv run uq generate-samples \
    mecillinam api_simulation_default test_violacein_with_metabolism \
    /path/to/sims \
    ./uq_cache \
    --n-samples 200 \
    --seed 42
```

This produces a cache directory (`./uq_cache/`) containing:

| File | Contents |
|------|----------|
| `X.npy` | Input samples, shape `(n_samples, n_params)` |
| `Y.npy` | Aggregated outputs, shape `(n_samples, n_outputs)` |
| `metadata.json` | Parameter names, bounds, seed, dimensions |
| `timeseries/sample_NNNN.npy` | Per-sample raw timeseries for Phase 2 (optional) |

### What gets stored in metadata

The cache records exactly which parameters were sampled and their bounds:

```json
{
  "parameter_names": ["vio_expression", "vio_trl_eff", "mecillinam_concentration"],
  "bounds": [[0.0, 5.0], [0.0, 2.0], [0.0, 10.0]],
  "n_samples": 200,
  "n_params": 3,
  "n_outputs": 4,
  "seed": 42
}
```

This metadata is the contract between Stage 1 and Stage 2.

---

## Stage 2: Analyze (quantify)

Run the full GSA pipeline using the cached samples. No simulations are re-run.

```bash
uv run uq quantify \
    mecillinam api_simulation_default test_violacein_with_metabolism \
    /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results \
    --n-bins 10 \
    --pce-polynomial-order 3 \
    --n-samples 200
```

The `quantify` command:

1. Loads timeseries data from the experiment directories
2. Runs aggregation (uniform, by-generation, by-seed) and variance decomposition
3. Loads the precomputed `(X, Y)` from the cache
4. Fits PCE surrogates and computes Sobol indices (Phase 1: population, Phase 2: cell cycle)
5. Prints a rich terminal report and optionally exports artifacts

---

## How Samples Match the Pipeline

**The critical invariant:** the parameter names and bounds in the cache must match the parameter space the pipeline constructs from the same `experiment_ids` + `sim_base_path`.

The parameter space is derived deterministically from the `simData.cPickle` files. Given the same experiments and sim data path, both `generate-samples` and `quantify` will construct the same parameter space. This means:

- **Same experiments + same sim data = compatible cache.** You can re-run `quantify` with different `--n-bins`, `--pce-polynomial-order`, or `--export-path` without regenerating samples.
- **Different experiments or sim data = incompatible cache.** The parameter names/bounds will differ, and the pipeline will fail with a dimension mismatch rather than silently produce wrong results.
- **The seed is recorded** in metadata so results are reproducible.

### When you need to regenerate samples

| Changed | Regenerate? | Why |
|---------|-------------|-----|
| `--n-bins`, `--pce-polynomial-order` | No | Analysis-only settings |
| `--export-path` | No | Just changes where artifacts are written |
| `experiment_ids` | **Yes** | Different experiments = different parameter space |
| `sim_base_path` (pointing to new sim data) | **Yes** | Different sim data = different parameter bounds |
| `--n-samples` (want more) | **Yes** | Need more sample evaluations |
| `--observable-columns` | **Yes** | Changes Y (output) dimensionality |

---

## What the Report Shows

After `quantify` completes, a terminal report is printed with:

| Section | What it shows |
|---------|---------------|
| **Variance Decomposition** | How total variance splits across generation (convergence), seed (exogenous noise), and residual (cell cycle) |
| **Morris Prescreening** | Which parameters passed initial screening, with linearity classification |
| **GSA Phase 1 (Population)** | Sobol indices S_i and S_Ti ranking parameters by bulk variance contribution |
| **GSA Phase 2 (Cell Cycle)** | Per-stage Sobol indices showing how parameter importance shifts across the cell cycle |
| **Cell Cycle Relevant Observables** | Which observables are most informative for defining the cell cycle variable |

---

## Export Artifacts

When `--export-path` is provided, the pipeline writes:

```
uq_results/
  population_surrogate/     # Phase 1 PCE surrogate
  population_sobol/         # Phase 1 Sobol indices
  cell_cycle_surrogate/     # Phase 2 PCE surrogate
  cell_cycle_sobol_stage_N/ # Phase 2 per-stage Sobol indices
  variance_decomposition.json
  morris_indices/           # Morris screening results
  metadata.json             # Pipeline metadata
```

These can be reloaded with `PipelineResult.from_export("./uq_results")`.

---

## Quick Reference

```bash
# Full two-stage workflow:
uv run uq generate-samples exp1 exp2 /sims ./cache --n-samples 200
uv run uq quantify exp1 exp2 /sims --precomputed-path ./cache --export-path ./results

# Re-analyze with different PCE order (no regeneration needed):
uv run uq quantify exp1 exp2 /sims --precomputed-path ./cache --pce-polynomial-order 4

# Demo with default experiments:
uv run uq demo
```
