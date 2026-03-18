# UQ Artifact Analysis: Violacein–Mecillinam Cell Cycle Sensitivity

---

## 1. Top-Level Metadata

```json
"n_parameters": 3,
"n_cell_cycle_stages": 10
```

Three parameters are being varied: `vio_expression`, `vio_trl_eff`, and `mecillinam_concentration`. The cell cycle is discretized into 10 equal-width bins of the normalized cell cycle coordinate θ ∈ [0,1]. This θ is likely a phase variable — 0 is birth, 1 is division.

---

## 2. Variance Decomposition

This section answers: **where does the variability in your output come from structurally?** The two outputs tracked are likely `dry_mass` and `growth` (matching `cell_cycle_relevance`).

```json
"total_variance": [0.086, 2e-06]
```

The total variance in `dry_mass` across all cells/conditions is 0.086. Growth variance is essentially 0 (2e-6). This already tells you something important — **growth rate is nearly deterministic** given your parameter ranges, while dry_mass accumulates meaningful variance.

```json
"generation_fraction": [0.0001, 0.0],
"seed_fraction":       [5e-05,  0.0],
"residual_fraction":   [0.9999,  1.0]
```

This is a variance components decomposition — it partitions total variance into:

- **generation_fraction**: how much variance is explained by *which generation* a cell is in. Answer: essentially zero (0.01%). The generation number carries almost no explanatory power.
- **seed_fraction**: how much variance comes from stochastic seed differences (random number generator seeds used in simulation). Also essentially zero (0.005%). Your model is **not meaningfully stochastic** — different random seeds produce nearly identical outcomes.
- **residual_fraction**: everything not explained by generation or seed — which is 99.99% of the variance. This residual is driven by **parameter variation**, which is exactly what you want in a sensitivity analysis. It means the Sobol' analysis is operating on a clean signal.

```json
"between_generation_variance": [8.6e-06, 0.0],
"between_seed_variance":       [4.3e-06, 0.0],
"within_group_variance":       [0.0859,  2e-06]
```

These are the absolute variance quantities corresponding to those fractions. Within-group variance (0.0859) dominates completely. The system's output variance is almost entirely **within-condition variance driven by parameter inputs**, not by biological noise or generational effects.

> **Key takeaway**: your simulation is clean, near-deterministic, and parameter-driven. The UQ analysis is well-posed.

---

## 3. Phase 1 — Population Sobol Indices

This is a global sensitivity analysis on **population-level outputs** (collapsed across the cell cycle).

```json
"first_order":  {vio_expression: 0.45, vio_trl_eff: 0.05, mecillinam_concentration: 0.40}
"total_order":  {vio_expression: 0.55, vio_trl_eff: 0.08, mecillinam_concentration: 0.50}
```

**First order sum**: 0.45 + 0.05 + 0.40 = 0.90. The remaining ~10% of variance is due to interactions between parameters.

**vio_expression (S1=0.45, ST=0.55)**
Gap = 0.10. Strong independent effect, moderate interaction involvement. This parameter drives nearly half the output variance on its own, and participates in interactions contributing another 10%.

**vio_trl_eff (S1=0.05, ST=0.08)**
Gap = 0.03. Small but non-negligible. At population level this parameter looks weak — consistent with Morris (μ*=0.03). However the total order being notably larger than first order proportionally (60% larger) hints that what little influence it has is interaction-mediated. Flag this for the cell-cycle-resolved analysis.

**mecillinam_concentration (S1=0.40, ST=0.50)**
Gap = 0.10. Nearly as important as `vio_expression` independently. The gap structure is symmetric to `vio_expression`, suggesting they interact **with each other** primarily — which makes biological sense: antibiotic concentration and gene expression level likely have a coupled effect on growth.

**The 0.45/0.40 split between `vio_expression` and `mecillinam_concentration`** is striking — these two parameters together explain 90% of variance at population level, nearly equally. Your system has two co-dominant drivers.

---

## 4. Phase 2 — Cell Cycle Resolved Sobol Per Stage

This is where the artifact gets scientifically deep. Instead of collapsing across the cell cycle, you're asking: **does parameter importance shift across the cell cycle?** θ goes from 0 (birth) to 1 (division).

### vio_expression — First Order across stages:

```
θ=0.0: 0.564
θ=0.1: 0.526
θ=0.2: 0.435
θ=0.3: 0.365
θ=0.4: 0.310
θ=0.5: 0.290
θ=0.6: 0.196
θ=0.7: 0.100
θ=0.8: 0.080
θ=0.9: 0.002  ← nearly vanishes
```

This is a **dramatic monotonic decline**. `vio_expression` dominates early in the cell cycle and becomes essentially irrelevant by division. At birth (θ=0), 56% of output variance is driven by `vio_expression` alone. By late cycle (θ=0.9), it contributes virtually nothing (0.2%).

Biologically this makes sense: violacein expression level sets the initial state of the cell — how much of the relevant machinery is present at birth. But as the cell progresses through its cycle, that initial condition becomes less and less predictive of where it ends up. The cell cycle dynamics wash out the expression-level signal.

### mecillinam_concentration — First Order across stages:

```
θ=0.0: 0.062
θ=0.1: 0.087
θ=0.2: 0.143
θ=0.3: 0.197
θ=0.4: 0.229
θ=0.5: 0.284
θ=0.6: 0.317
θ=0.7: 0.380
θ=0.8: 0.451
θ=0.9: 0.479  ← peaks near division
```

This is the **mirror image** of `vio_expression` — a monotonic increase across the cell cycle. Mecillinam (a β-lactam antibiotic targeting PBP2, disrupting cell elongation) has negligible influence early in the cycle but becomes the dominant driver as the cell approaches division.

This is a profound biological signal: **mecillinam's effect accumulates over the cell cycle**. Its mechanism — disrupting the elongasome and cell wall synthesis — is a process that compounds over time within a single cycle. Early in the cycle there hasn't been enough time for the drug to differentially affect cells. By late cycle, cumulative drug exposure has become the primary determinant of cell state.

### vio_trl_eff — First Order across stages:

```
θ=0.0: 0.043
θ=0.1: 0.054
...
θ=0.9: 0.055
```

Essentially flat across the entire cell cycle, always low (0.04–0.06). This parameter is consistently unimportant at every stage. However — note the **total order**:

```
θ=0.9 total: 0.108
```

At late cycle, `vio_trl_eff`'s total order is double its first order. It's picking up interaction effects with `mecillinam_concentration` at division time. It never acts alone, but it modulates the antibiotic response slightly near division.

### The Crossover Point

`vio_expression` and `mecillinam_concentration` **cross over** somewhere around θ=0.5–0.6:

```
θ=0.5: vio_expression=0.290, mecillinam=0.284  ← nearly equal
θ=0.6: vio_expression=0.196, mecillinam=0.317  ← crossover complete
```

At the midpoint of the cell cycle, neither parameter dominates — the system is in a transition regime. This crossover is a quantitative fingerprint of when antibiotic effects begin to outweigh initial expression state as the determinant of cell behavior.

---

## 5. Cell Cycle Profile

```json
"dry_mass_mean": [1.027, 1.104, 1.179, ..., 1.908]
"growth_mean":   [0.01086, 0.01138, 0.01165, ..., 0.00894]
```

**Dry mass** increases monotonically from ~1.03 to ~1.91 across the cycle — roughly a doubling, as expected for a cell growing toward division. The growth is approximately exponential (consistent with the pipelining biology discussed earlier).

**Growth rate** (instantaneous) shows a non-monotonic profile:

- Rises from θ=0 to θ=0.2 (peak ~0.01165)
- Declines from θ=0.2 to θ=0.7 (trough ~0.00833)
- Slightly recovers at θ=0.8–0.9

This bell-shaped growth rate profile is biologically meaningful. Growth rate peaks early-to-mid cycle then slows — possibly due to mecillinam effects on cell wall synthesis accumulating and throttling elongation. The slight recovery near division may reflect cell cycle checkpoint release or pre-division metabolic upregulation. This non-monotonicity is exactly what the cell-cycle-resolved Sobol' analysis is capturing — the drug's increasing dominance corresponds precisely to the period when growth rate is declining.

---

## 6. Cell Cycle Relevance

```json
"relevance_scores": {
    "listeners__mass__dry_mass": 0.92,
    "listeners__mass__growth":   0.78
},
"residual_variance_fraction": [0.9999, 1.0]
```

These relevance scores (likely R² or a correlation-based metric) say how well the cell cycle stage alone predicts each observable. Dry mass is highly predictable from cell cycle position (0.92) — knowing where you are in the cycle tells you most of what you need to know about mass. Growth rate is less predictable (0.78) — it has more parameter-dependent variability.

The residual variance fraction of 0.9999 reiterates what the variance decomposition showed: essentially all variance is within-group, parameter-driven. The observables are **almost entirely determined by your three input parameters**, not by noise or generational effects.

---

## 7. Surrogates

```json
"population": {
    "basis_type": "legendre",
    "polynomial_order": 3,
    "input_dim": 3, "output_dim": 2,
    "r_squared": 0.92,
    "n_terms": 10
}
```

A **degree-3 Legendre PCE** was fit to map the 3 parameters → 2 population outputs. With 3 inputs and degree 3, the full basis has C(3+3,3)=20 terms, but only 10 were used — suggesting sparse PCE or truncation, keeping only the most important basis functions. R²=0.92 is good but not exceptional — the cubic terms are needed to capture the nonlinearity you saw in Morris (mecillinam's σ/μ*=0.55).

```json
"cell_cycle": {
    "basis_type": "legendre",
    "polynomial_order": 2,
    "input_dim": 3,
    "output_dim": 10,
    "r_squared": 0.87,
    "n_terms": 10
}
```

A **degree-2 Legendre PCE** maps 3 parameters → 10 cell cycle stage outputs simultaneously. Lower polynomial order than the population surrogate (degree 2 vs 3), same number of terms (10), but lower R²=0.87. The lower fit quality makes sense — you're now trying to capture stage-dependent behavior across 10 outputs simultaneously, which is a harder problem. The degree-2 limitation may be slightly underfitting the nonlinear crossover dynamics you see in the Sobol' stage profiles. A degree-3 cell cycle surrogate might recover some of that missing 13%.

---

## Overall Scientific Narrative

Reading all sections together, a coherent story emerges:

The system has **two co-dominant parameters that control different phases of the cell cycle**. `vio_expression` sets the initial conditions that govern early-cycle behavior, while `mecillinam_concentration` accumulates influence across the cycle and dominates near division. They cross over at mid-cycle (θ≈0.55), which coincides with the inflection point in the growth rate profile. `vio_trl_eff` is essentially a spectator at population level but has weak interaction effects near division. The system is clean, near-deterministic, and well-suited for PCE surrogate modeling, though the cell-cycle surrogate may benefit from a degree-3 expansion to better capture the crossover dynamics.
