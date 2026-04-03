# UQ Framework Tutorial

Two-stage workflow for global sensitivity analysis of the vEcoli whole-cell
model, following [RFC006](readmes/start/tools/RFC006.md) and the
[PyTUQ UQPC workflow](https://sandialabs.github.io/pytuq/apps/uqpc.html).

```
  Stage 1: sample                      Stage 2: quantify
  ┌────────────────────────────────┐   ┌─────────────────────────────────┐
  │ UQPC Step 1: bounds → PCRV    │   │ UQPC Step 4: pc_fit (lsq/bcs/  │
  │ UQPC Step 2: PCRV.sampleGerm  │──▶│   anl) → PCRV + Sobol          │
  │ UQPC Step 3: vEcoli workflow.py│   │ UQPC Step 5: post-process      │
  └────────────────────────────────┘   │   (all 4 RFC006 strategies)    │
                                       └─────────────────────────────────┘
```

Three clients expose the same workflow in different formats:

| Client | Launch | Format |
|--------|--------|--------|
| **CLI** | `uv run uq sample / quantify` | Terminal with Rich progress bar |
| **TUI** | `uv run uq tui` | Textual interactive app |
| **Dashboard** | `uv run marimo run app/dashboard_simple.py` | Marimo notebook |

---

## Prerequisites

```bash
uv sync
```

You need:
- A `simData.cPickle` file (from vEcoli's parca step)
- vEcoli installed as an editable dependency (for `runscripts/workflow.py`)
- `ecoli/variants/sim_data_setattr.py` in the vEcoli repo

---

## Stage 1: Sampling (UQPC Steps 1-3)

Sampling follows the PyTUQ UQPC workflow:
- **Step 1:** Load `simData.cPickle`, build parameter space, construct
  input PCRV (Legendre basis, order 1) from bounds.
- **Step 2:** Draw samples via `PCRV.sampleGerm()` → `PCRV.evalPC()`
  (PyTUQ-native random sampling, not scipy LHS).
- **Step 3:** Build a single vEcoli config JSON with `sim_data_setattr`
  variants, launch `workflow.py` as a subprocess. Live progress shown
  via stdout streaming + variant polling.

### CLI

```bash
uv run uq sample /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --n-samples 20 \
    --generations 2 \
    --n-init-sims 1 \
    --max-duration 10800
```

The CLI shows:
- Live Nextflow stdout (ANSI-stripped, color-coded)
- Rich progress bar: `Simulating  5/21  [45s]  ━━━━━━━━━━━━━━── 23.8%`
- Final summary: `Cached 20 samples (6 params, 4 outputs)`

Options:
```
--n-samples      Number of PCRV samples (= number of variants)
--generations    Generations per sim (>= 2 enables Strategy 2)
--n-init-sims    Seeds per variant (> 1 enables Strategy 3)
--max-duration   Wall-clock limit per sim in seconds
--params-file    JSON file with custom SimDataParameter specs
--cache-dir      Where to write the cache (default: ./uq_cache)
```

### TUI

```bash
uv run uq tui
```

The sidebar has a **CONFIG** section with all parameters editable:
- `simData` path, cache dir, samples, generations, seeds
- **VARIANT PARAMETERS** — checkboxes to toggle each of the 6 default
  parameters on/off, with editable lo/hi bounds

Click **Run Sampling** — live Nextflow output streams in the log panel
with a progress bar showing `Launching → Variants → Simulating` phases.

### Python API

```python
from uq.workflow import sample

cache = sample(
    sim_data_path="/path/to/simData.cPickle",
    cache_dir="./uq_cache",
    n_samples=50,
    generations=2,
)
```

### What gets cached

```
uq_cache/
├── X.npy                        # (n_samples, n_params) — physical space
├── Y.npy                        # (n_samples, n_outputs) — time-averaged
├── germ_train.npy               # (n_samples, n_params) — germ space [-1,1]
├── metadata.json                # parameter names, bounds, observable columns
└── timeseries/
    ├── sample_0000.npy          # (n_timesteps, n_obs) — raw timeseries
    ├── sample_0000_meta.npz     # generation + lineage_seed labels per row
    └── ...
```

### Custom parameters

By default, 6 physiologically relevant parameters are varied
(`DEFAULT_SIM_DATA_PARAMETERS`). To use custom parameters:

```bash
# CLI: pass a JSON file
uv run uq sample /path/to/simData.cPickle --params-file params.json
```

```python
# Python API: pass SimDataParameter objects directly
from libuq.pipeline.models import SimDataParameter

cache = sample(
    sim_data_path="/path/to/simData.cPickle",
    cache_dir="./uq_cache",
    n_samples=100,
    parameters=[
        SimDataParameter(
            name="kinetic_objective_weight",
            attr_path="process.metabolism.kinetic_objective_weight",
            bounds=(5e-8, 5e-7),
        ),
    ],
)
```

In the TUI, uncheck parameters in the **VARIANT PARAMETERS** section
and edit bounds directly in the sidebar.

---

## Stage 2: Quantification (UQPC Steps 4-5)

Load the cache from Stage 1 and run all four RFC006 strategies.

### CLI

```bash
uv run uq quantify /path/to/simData.cPickle \
    --cache-dir ./uq_cache \
    --export-path ./uq_results \
    --polynomial-order 2 \
    --n-bins 10 \
    --regression lsq
```

Prints a Rich report with Sobol tables for all 4 strategies.

### TUI

Click **Run Quantify** in the sidebar. Results populate the log panel
inline, then click **S1 Population** / **S2 By Generation** / etc. to
view detailed DataTables.

### Python API

```python
from uq.workflow import quantify

result = quantify(
    cache_dir="./uq_cache",
    sim_data_path="/path/to/simData.cPickle",
    polynomial_order=2,
    n_bins=10,
    regression="lsq",
    export_path="./uq_results",
)
```

### Inspecting results

```python
# Strategy 1: bulk sensitivity
for name, st in zip(result.parameter_names, result.strategy1.sobol.total_order):
    print(f"  {name}: S_T = {st:.4f}")

# Strategy 2: per-generation
for gen, r in sorted(result.strategy2.items()):
    print(f"\n  Generation {gen}:")
    for name, st in zip(result.parameter_names, r.sobol.total_order):
        print(f"    {name}: S_T = {st:.4f}")

# Strategy 4: per-growth-stage
for i, r in enumerate(result.strategy4_per_stage):
    print(f"\n  Growth stage {i}:")
    for name, st in zip(result.parameter_names, r.sobol.total_order):
        print(f"    {name}: S_T = {st:.4f}")
```

### Surrogate quality

```python
s1 = result.strategy1
print(f"Training relative error: {s1.relerr_train}")
print(f"Mean prediction std: {s1.Y_train_pc_std.mean():.6f}")

# Raw PyTUQ objects
pcrv = s1.pcrv          # PCRV — evaluate, sample, compute moments
linregs = s1.linregs    # per-output regression objects
```

### Export structure

```
uq_results/
├── uq_results.json              # dashboard-compatible summary
├── population_surrogate/        # PCE coefficients + multi-indices
├── population_sobol/            # first_order.npy, total_order.npy
├── generation_0_sobol/
├── generation_1_sobol/
├── seed_0_sobol/
├── growth_stage_0_sobol/
├── growth_stage_1_sobol/
├── ...
└── growth_stratified_surrogate/
```

---

## Full end-to-end (CLI)

```bash
# Stage 1: sample
uv run uq sample ./sim_data/baseline/kb/simData.cPickle \
    --n-samples 20 --generations 1

# Stage 2: quantify
uv run uq quantify ./sim_data/baseline/kb/simData.cPickle \
    --polynomial-order 2 --regression lsq --export-path ./uq_results

# View dashboard
uv run marimo run app/dashboard_simple.py
```

---

## Regression methods

| Method | Flag | When to use |
|--------|------|-------------|
| **LSQ** | `lsq` | Default. Fast, exact on noiseless polynomial data. n_samples >> n_terms. |
| **BCS** | `bcs` | Sparse PCE. Retains only significant terms. n_samples ≈ n_terms. |
| **ANL** | `anl` | Analytical Bayesian. Calibrated uncertainty estimates. |

Rule of thumb: PCE terms = `C(n_params + order, order)`.
For 6 params, order 2 → 28 terms → need ≥ 56 samples for stable LSQ.

---

## Interpreting Sobol indices

- **S_i (first-order):** Fraction of output variance explained by parameter `i` alone.
- **S_Ti (total-order):** Fraction explained by `i` plus all its interactions.
- For additive models: `sum(S_i) ≈ 1` and `S_Ti ≈ S_i`.
- Large `S_Ti - S_i` indicates the parameter participates in interactions.

Example:
```
  fraction_active_rnap_free:    S_T = 45.2%   ← controls ~half of bulk variance
  basal_elongation_rate:        S_T = 30.1%   ← translation speed matters
  kinetic_objective_weight:     S_T = 12.3%   ← FBA tuning is secondary
  cell_dry_mass_fraction:       S_T =  8.7%   ← composition matters less
  secretion_penalty_coeff:      S_T =  3.1%   ← overflow metabolism is minor
  fraction_active_rnap_bound:   S_T =  0.6%   ← ppGpp-bound RNAP negligible
```

---

## Regarding Strategies 1-4

The PCE fitting and Sobol computation are identical across all four
strategies — same `_fit_surrogate` → `_compute_sobol` path, same PyTUQ
PCRV + lsq/bcs/anl machinery. The only thing that changes is how you
aggregate the raw timeseries into the Y matrix that gets fed to `run_uqpc`.

### Raw data shape from a batch run:

`(n_variants, n_lineage_seeds, n_generations, n_agents, n_timesteps, n_observables)`

where `n_agents` per generation = 1 (single daughters, `generation=3/agent_id=000`).

### Each strategy collapses different axes before PCE sees it:

```
┌──────────┬──────────────────────────────────────┬─────────────────────────────────────────┬──────────────────────────────────────┐
│ Strategy │ What gets collapsed                  │ Y shape into PCE                        │ Lens                                 │
├──────────┼──────────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
│ 1        │ seeds, gens, agents, timesteps        │ (n_variants, n_obs)                     │ Everything averaged                  │
│ 2        │ seeds, agents, timesteps (per gen)    │ (n_variants, n_obs) × per generation    │ Hold generation fixed                │
│ 3        │ gens, agents, timesteps (per seed)    │ (n_variants, n_obs) × per seed          │ Hold seed fixed                      │
│ 4        │ nothing temporally — binned by θ      │ (n_variants, n_bins×n_obs) × per stage  │ Hold cell-cycle stage fixed           │
└──────────┴──────────────────────────────────────┴─────────────────────────────────────────┴──────────────────────────────────────┘
```

The `n_variants` axis is always the row dimension — that's what PCE
regresses over. The strategies just change what's in the columns.
