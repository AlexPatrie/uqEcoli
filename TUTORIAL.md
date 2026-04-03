# UQ Framework Tutorial — `uq.common.handlers`

Two-stage workflow for global sensitivity analysis of the vEcoli whole-cell
model, following [RFC006](readmes/start/tools/RFC006.md) and the
[PyTUQ UQPC workflow](https://sandialabs.github.io/pytuq/).

```
  Stage 1: sample()                       Stage 2: quantify()
  ┌────────────────────────────────┐      ┌─────────────────────────────────┐
  │ UQPC Step 1: bounds → PCRV    │      │ UQPC Step 4: pc_fit (lsq/bcs/  │
  │ UQPC Step 2: PCRV.sampleGerm  │─────▶│   anl) → PCRV + Sobol          │
  │ UQPC Step 3: vEcoli workflow.py│cache │ UQPC Step 5: post-process      │
  └────────────────────────────────┘      │   (all 4 RFC006 strategies)    │
                                          └─────────────────────────────────┘
```

---

## Prerequisites

```bash
# From the uqEcoli repo root
uv sync
```

You need:
- A `simData.cPickle` file (from vEcoli's parca step)
- vEcoli installed as an editable dependency (for `runscripts/workflow.py`)

---

## Stage 1: Sampling (UQPC Steps 1-3)

Sampling follows the PyTUQ UQPC workflow directly:
- **Step 1:** Load `simData.cPickle`, build parameter space, construct
  input PCRV (Legendre basis, order 1) from bounds.
- **Step 2:** Draw samples via `PCRV.sampleGerm()` → `PCRV.evalPC()`
  (PyTUQ-native random sampling, not scipy LHS).
- **Step 3:** Evaluate vEcoli at each sample via `TimeseriesGeneratorVecoli`
  (subprocess call to `runscripts/workflow.py`).

### Python API

```python
from uq.common.workflow import sample

cache = sample(
    sim_data_path="/path/to/sims/my_experiment/parca/kb/simData.cPickle",
    cache_dir="./uq_cache",
    n_samples=50,
    seed=42,
    # generations >= 2 enables Strategy 2 (by-generation GSA)
    generations=2,
    # max_workers controls subprocess parallelism
    max_workers=4,
)

print(f"Cached: {cache.X.shape[0]} samples, {cache.X.shape[1]} params")
print(f"Outputs: {cache.Y.shape[1]} observables")
print(f"Timeseries: {len(cache.Y_timeseries)} samples")
```

### What gets cached

```
uq_cache/
├── X.npy                        # (n_samples, n_params) — physical space
├── Y.npy                        # (n_samples, n_outputs) — time-averaged
├── germ_train.npy               # (n_samples, n_params) — germ space [-1,1]
├── metadata.json                # parameter names, bounds, germ samples
└── timeseries/
    ├── sample_0000.npy          # (n_timesteps, n_obs) — raw timeseries
    ├── sample_0000_meta.npz     # generation + seed labels per row
    ├── sample_0001.npy
    ├── sample_0001_meta.npz
    └── ...
```

The `germ_train.npy` file stores the PCRV germ-space samples so that
`quantify()` can use them directly — no inverse transform needed.

### Custom parameters

By default, `sample()` uses 6 physiologically relevant parameters
(`DEFAULT_SIM_DATA_PARAMETERS`). To use custom parameters:

```python
# Option A: pass SimDataParameter objects directly
from uq.pipeline.models import SimDataParameter

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
        # ... more params
    ],
)

# Option B: pass a JSON file with SimDataParameter dicts
cache = sample(
    sim_data_path="/path/to/simData.cPickle",
    cache_dir="./uq_cache",
    n_samples=100,
    params_file="params.json",
)
```

---

## Stage 2: Quantification (UQPC Steps 4-5)

Load the cache from Stage 1 and run all four RFC006 strategies.
Germ-space samples are loaded directly from the cache, ensuring
exact consistency with the Legendre basis used in PCE fitting.

### Python API

```python
from uq.common.workflow import quantify

result = quantify(
    cache_dir="./uq_cache",
    sim_data_path="/path/to/simData.cPickle",
    polynomial_order=2,
    n_bins=10,
    regression="lsq",  # or "bcs" (sparse) or "anl" (Bayesian)
    export_path="./uq_results",
)
```

### Inspecting results

```python
# Strategy 1: Which parameters drive the most variance in bulk output?
s1 = result.strategy1
for name, st in zip(result.parameter_names, s1.sobol.total_order):
    print(f"  {name}: S_T = {st:.4f}")

# Strategy 2: Does parameter importance differ across generations?
for gen, r in sorted(result.strategy2.items()):
    print(f"\n  Generation {gen}:")
    for name, st in zip(result.parameter_names, r.sobol.total_order):
        print(f"    {name}: S_T = {st:.4f}")

# Strategy 3: Does it differ across lineage seeds?
for seed, r in sorted(result.strategy3.items()):
    print(f"\n  Seed {seed}:")
    for name, st in zip(result.parameter_names, r.sobol.total_order):
        print(f"    {name}: S_T = {st:.4f}")

# Strategy 4: How does parameter importance change across the cell cycle?
for i, r in enumerate(result.strategy4_per_stage):
    lo, hi = i / result.strategy4_combined.sobol.total_order.shape[0], 0
    print(f"\n  Growth stage {i}:")
    for name, st in zip(result.parameter_names, r.sobol.total_order):
        print(f"    {name}: S_T = {st:.4f}")
```

### Surrogate quality

Each `UQPCResult` includes diagnostics from the UQPC workflow:

```python
s1 = result.strategy1

# Relative error: ||Y - Y_pc|| / ||Y|| per output
print(f"Training relative error: {s1.relerr_train}")

# Prediction uncertainty at training points
print(f"Mean prediction std: {s1.Y_train_pc_std.mean():.6f}")

# Raw PyTUQ objects for advanced use
pcrv = s1.pcrv          # PCRV — evaluate, sample, compute moments
linregs = s1.linregs    # per-output regression objects
```

### Export structure

When `export_path` is provided, `quantify()` writes:

```
uq_results/
├── uq_results.json                     # summary with Sobol indices
├── strategy1_population/
│   ├── first_order.npy
│   ├── total_order.npy
│   ├── relerr_train.npy
│   └── surrogate/                      # PCE coefficients + multi-indices
├── strategy2_generation_0/
│   ├── first_order.npy
│   └── total_order.npy
├── strategy2_generation_1/
│   └── ...
├── strategy3_seed_0/
│   └── ...
├── strategy4_stage_0/
│   ├── first_order.npy
│   └── total_order.npy
├── strategy4_stage_1/
│   └── ...
└── strategy4_surrogate/
    └── ...
```

---

## Full end-to-end example

```python
from uq.common.workflow import sample, quantify

SIM_DATA = "/path/to/sims/api_simulation_default/parca/kb/simData.cPickle"

# ── Stage 1: Sample (UQPC Steps 1-3) ──
cache = sample(
    sim_data_path=SIM_DATA,
    cache_dir="./uq_cache",
    n_samples=50,
    generations=2,
    max_workers=4,
)
print(f"Cached {cache.X.shape[0]} samples")

# ── Stage 2: Quantify (UQPC Steps 4-5) ──
result = quantify(
    cache_dir="./uq_cache",
    sim_data_path=SIM_DATA,
    polynomial_order=2,
    n_bins=5,
    regression="lsq",
    export_path="./uq_results",
)

# ── Interpret ──
print("\nPopulation-level sensitivity (Strategy 1):")
for name, st in zip(result.parameter_names, result.strategy1.sobol.total_order):
    pct = st * 100
    bar = "█" * int(pct / 2)
    print(f"  {name:<35s} {pct:5.1f}% {bar}")

if result.strategy4_per_stage:
    print(f"\nGrowth-stratified sensitivity ({len(result.strategy4_per_stage)} stages):")
    for i, r in enumerate(result.strategy4_per_stage):
        dominant = result.parameter_names[r.sobol.total_order.argmax()]
        print(f"  Stage {i}: dominated by {dominant} "
              f"(S_T = {r.sobol.total_order.max():.3f})")
```

---

## Using individual workflow steps

For finer control, use the step-level functions directly:

```python
from uq.common.workflow import run_uqpc, run_strategy1_uniform
from uq.pipeline.param_loader import ParameterDataset
from uq.sampling import PrecomputedCache

# Load existing cache + rebuild parameter space
cache = PrecomputedCache.load("./uq_cache")
ds = ParameterDataset(sim_data_path="/path/to/simData.cPickle")
param_space = ds.to_parameter_space()

# Run just Strategy 1 with BCS (sparse) regression
result = run_strategy1_uniform(
    param_space, cache.X, cache.Y,
    polynomial_order=3,
    regression="bcs",
)

# Or run the raw UQPC workflow on any (X, Y) pair
result = run_uqpc(
    param_space,
    Y_train=cache.Y,
    X_train=cache.X,
    polynomial_order=2,
    regression="lsq",
)
```

---

## Regression methods

| Method | Flag | When to use |
|--------|------|-------------|
| **LSQ** | `"lsq"` | Default. Fast, exact on noiseless polynomial data. Good when n_samples >> n_terms. |
| **BCS** | `"bcs"` | Sparse PCE. Retains only statistically significant terms. Good when n_samples ≈ n_terms or data is noisy. |
| **ANL** | `"anl"` | Analytical Bayesian. Provides posterior predictive uncertainty. Good when you need calibrated error bars. |

Rule of thumb for PCE terms: `C(n_params + order, order)`.
For 6 params, order 2 → 28 terms → need ≥ 56 samples for stable LSQ.

---

## Interpreting Sobol indices

- **S_i (first-order):** Fraction of output variance explained by parameter `i` alone.
- **S_Ti (total-order):** Fraction explained by `i` plus all its interactions.
- For additive models: `sum(S_i) ≈ 1` and `S_Ti ≈ S_i`.
- Large `S_Ti - S_i` indicates the parameter participates in interactions.

Example interpretation:
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

The PCE fitting and Sobol computation are identical across all four strategies — same _fit_surrogate → _compute_sobol path, same PyTUQ PCRV + lsq/bcs/anl machinery. The only thing that changes is how you aggregate the raw timeseries into the Y matrix that gets fed to run_uqpc.                                               
                                                                                                                                                                                                                                                                                                                                                  
### The raw data shape from a batch run is conceptually: 

`(n_variants, n_lineage_seeds, n_generations, n_agents, n_timesteps, n_observables)`                                                                                                                                                                                                                                                          

#### where:

`n_agents per generation = 1 (single daughters), so generation and agent_id are 1:1 (i.e.; /generation=3/agent_id=000)`                                                                                                                                                                                                                                         


### Each strategy collapses different axes before PCE sees it:                                                                                                                                                                                                                                                                                        

```
┌──────────┬───────────────────────────────────────────────────┬──────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐                                                                                                                                                                 
│ Strategy │                What gets collapsed                │                 Y shape into PCE                 │                            Lens                             │                                                                                                                                                               
├──────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                                                                                                                                                                 
│ 1        │ seeds, generations, agents, timesteps             │ (n_variants, n_obs)                              │ Everything averaged — one scalar per variant per observable │                                                                                                                                                               
├──────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2        │ seeds, agents, timesteps (within each gen)        │ (n_variants, n_obs) × one PCE per generation     │ Hold generation fixed, average the rest                     │                                                                                                                                                                 
├──────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                                                                                                                                                                 
│ 3        │ generations, agents, timesteps (within each seed) │ (n_variants, n_obs) × one PCE per seed           │ Hold seed fixed, average the rest                           │                                                                                                                                                                 
├──────────┼───────────────────────────────────────────────────┼──────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                                                                                                                                                                 
│ 4        │ nothing collapsed temporally — binned by θ        │ (n_variants, n_bins × n_obs) × one PCE per stage │ Hold cell-cycle stage fixed                                 │                                                                                                                                                               
└──────────┴───────────────────────────────────────────────────┴──────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘                                                                                                                                                                 
```

The n_variants axis (`= n_samples = number of LHS points in parameter space`) is _always the row dimension_ — that's the axis PCE regresses over. _The strategies just change what's in the columns._                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                                  
So in the code, the following (for example) exists:

1. `run_strategy2_by_generation calls _aggregate_by_group(..., "generation")` to partition timeseries rows by their generation label, 
2. computes per-generation means, 
3. calls the exact same run_uqpc() that Strategy 1 uses. 

Strategy 3 does the same with `"lineage_seed"`. _The GSA algorithm is completely agnostic to which lens produced the Y matrix._

---

## Interactive TUI

A Textual-based terminal UI wraps both `sample()` and `quantify()`:

```bash
uv run python -m uq.common.tui
```

Four tabs: **Sample** (fill in simData path, click Run), **Quantify**
(paths auto-fill from sampling, pick regression method, click Run),
**Results** (Sobol tables for all 4 strategies), **Log** (timestamped output).

Keyboard shortcuts: `s` Sample, `u` Quantify, `r` Results, `l` Log, `q` Quit.

Uses ANSI-only colors (`ansi_color=True`) — no truecolor backgrounds, so it
adapts to any terminal theme automatically.