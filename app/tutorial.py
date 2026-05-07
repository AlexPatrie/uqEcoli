# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23", "plotly", "numpy"]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import json
    from pathlib import Path
    import plotly.graph_objects as go

    return Path, go, json, mo, np


@app.cell
def __(mo):
    mo.md(r"""
# 🧬 UQ Framework — Interactive Tutorial

> **Uncertainty Quantification for the vEcoli Whole-Cell Model**
>
> This is a **living tutorial**. Every chart, table, and number below is
> computed in real time using the actual UQ pipeline. Change any slider,
> toggle any option, and the results update instantly.
>
> By the end, you will understand the entire `uq sample` → `uq quantify`
> workflow — the same commands used for published research.
""")
    return


@app.cell
def __(mo):
    level = mo.ui.slider(
        start=0,
        stop=7,
        step=1,
        value=0,
        label="Progress Level",
        show_value=True,
        full_width=True,
    )
    return (level,)


@app.cell
def __(level, mo):
    levels = [
        ("🎯", "The Problem", "Why UQ matters"),
        ("📐", "Parameter Space", "Defining what we perturb"),
        ("🎲", "Sampling", "PCRV & Latin Hypercube"),
        ("📦", "PCE Surrogate", "Fitting the math model"),
        ("📊", "Sobol Indices", "What drives variance"),
        ("🔬", "4 Strategies", "Aggregation deep-dive"),
        ("🚀", "Full Pipeline", "sample → quantify"),
        ("⚡", "Advanced", "PCA, baseline, remote"),
    ]
    lv = int(level.value)
    emoji, title, desc = levels[lv]
    mo.md(f"## {emoji} Level {lv + 1}/8: {title}\n\n*{desc}*")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 0 — THE PROBLEM
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(go, level, mo, np):
    mo.stop(int(level.value) < 0)

    rng = np.random.default_rng(42)
    _n = 200
    _t = np.linspace(0, 10, _n)
    _k_true = 0.5
    _k_range = (0.3, 0.7)

    _mass_curves = []
    for _k in np.linspace(*_k_range, 15):
        _mass_curves.append(np.exp(_k * _t) * (1 + 0.05 * np.sin(_t)))

    _fig = go.Figure()
    for _i, _c in enumerate(_mass_curves):
        _opacity = 0.15 if _i != 7 else 0.6
        _color = "#00d4aa" if _i == 7 else "#636efa"
        _fig.add_trace(
            go.Scatter(
                x=_t,
                y=_c,
                mode="lines",
                line=dict(color=_color, width=1.5 if _i == 7 else 1),
                opacity=_opacity,
                showlegend=False,
            )
        )
    _fig.add_trace(
        go.Scatter(
            x=_t,
            y=_mass_curves[-1],
            mode="lines",
            line=dict(color="#00d4aa", width=1.5, dash="dash"),
            name="parameter range",
        )
    )
    _fig.update_layout(
        title=dict(text="Output Variance from Parameter Uncertainty", x=0.5),
        xaxis_title="Time (hours)",
        yaxis_title="Dry Mass (pg)",
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return


@app.cell
def __(mo):
    mo.md(r"""
### The Core Problem

Every vEcoli simulation depends on **parameters** — kinetic constants,
mass fractions, transcription efficiencies. These values come from
experiments with measurement error. So which parameter uncertainty
actually matters?

The UQ framework answers:

| Question | Method |
|---|---|
| **Which parameters drive output variance?** | Sobol sensitivity indices |
| **How much variance is explained?** | Polynomial Chaos Expansion |
| **Does sensitivity change during the cell cycle?** | Growth-stratified analysis |
| **Is the result robust to stochastic noise?** | Multi-seed aggregation |

**The workflow has two stages:**

1. **`uq sample`** — perturb parameters, run vEcoli, cache results
2. **`uq quantify`** — fit PCE surrogate, compute Sobol indices, export report

> Increase the progress slider above to explore each level.
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 1 — PARAMETER SPACE
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(go, level, mo, np):
    mo.stop(int(level.value) < 1)

    default_params = [
        ("kinetic_obj_weight", 5e-8, 5e-7, "Metabolism FBA objective"),
        ("secretion_penalty", 1e-4, 1e-2, "Overflow metabolism penalty"),
        ("rnap_free_frac", 0.25, 0.47, "ppGpp-free RNAP active fraction"),
        ("rnap_bound_frac", 0.25, 0.47, "ppGpp-bound RNAP active fraction"),
        ("dry_mass_frac", 0.25, 0.35, "Cell dry mass fraction"),
        ("basal_elong_rate", 10.0, 22.0, "Ribosome elongation rate (aa/s)"),
    ]

    _names = [p[0] for p in default_params]
    _lows = [p[1] for p in default_params]
    _highs = [p[2] for p in default_params]

    _fig = go.Figure()
    _fig.add_trace(
        go.Bar(
            y=_names,
            x=[(h - l) / ((h + l) / 2) * 100 for l, h in zip(_lows, _highs)],
            orientation="h",
            marker=dict(
                color=np.linspace(0.4, 1.0, 6),
                colorscale="Tealgrn",
                showscale=False,
            ),
            text=[f"±{(h - l) / ((h + l) / 2) * 100:.0f}%" for l, h in zip(_lows, _highs)],
            textposition="outside",
            textfont=dict(color="#00d4aa", size=12),
        )
    )
    _fig.update_layout(
        title=dict(text="Parameter Uncertainty Ranges (default 6 parameters)", x=0.5),
        xaxis_title="Relative Range (%)",
        template="plotly_dark",
        height=260,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        margin=dict(l=160, r=80, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )

    _rows = []
    for name, lo, hi, desc in default_params:
        _mid = (lo + hi) / 2
        _range_pct = (hi - lo) / _mid * 100
        _rows.append(f"| `{name}` | {lo:.2g} | {hi:.2g} | {_mid:.2g} | ±{_range_pct:.0f}% | {desc} |")
    _table = "\n".join(_rows)

    return (default_params,)


@app.cell
def __(default_params, mo):
    _rows = []
    for name, lo, hi, desc in default_params:
        _mid = (lo + hi) / 2
        _range_pct = (hi - lo) / _mid * 100
        _rows.append(f"| `{name}` | {lo:.2g} | {hi:.2g} | {_mid:.2g} | ±{_range_pct:.0f}% | {desc} |")
    _table = "\n".join(_rows)

    mo.md(f"""
### Default Parameters

These 6 `SimDataParameter` specs are loaded from `simData.cPickle`:

| Parameter | Lower | Upper | Mid | Range | Description |
|---|---|---|---|---|---|
{_table}

**CLI:** `--params-file params.json` overrides these with custom bounds.
""")
    return


@app.cell
def __(mo):
    mo.md(r"""
### How Parameters Map to vEcoli

Each parameter is a **dot-path** into the `SimulationDataEcoli` object:

```
process.metabolism.kinetic_objective_weight  →  FBA objective blending
process.transcription.fraction_active_rnap_free  →  RNAP activation
mass.cell_dry_mass_fraction  →  mass conversion factor
```

During sampling, each LHS point becomes a **variant**:
`{"value": [{"process.metabolism.kinetic_objective_weight": 3.2e-7}]}`

vEcoli's `sim_data_setattr` mutates the simulation data before each run.
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 2 — SAMPLING
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(level, mo, np):
    mo.stop(int(level.value) < 2)

    n_samples_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Number of Samples (N)",
        show_value=True,
        full_width=True,
    )
    n_params_slider = mo.ui.slider(
        start=2,
        stop=6,
        step=1,
        value=3,
        label="Number of Parameters (d)",
        show_value=True,
        full_width=True,
    )
    return n_params_slider, n_samples_slider


@app.cell
def __(go, mo, n_params_slider, n_samples_slider, np):
    _N = int(n_samples_slider.value)
    _d = int(n_params_slider.value)

    rng = np.random.RandomState(42)
    _lhs = np.zeros((_N, _d))
    for _dim in range(_d):
        _perm = rng.permutation(_N)
        _lhs[:, _dim] = (_perm + rng.random(_N)) / _N

    from math import comb

    _p = 3
    _n_terms = comb(_d + _p, _p)

    _fig = go.Figure()
    if _d == 2:
        _fig.add_trace(
            go.Scatter(
                x=_lhs[:, 0],
                y=_lhs[:, 1],
                mode="markers",
                marker=dict(size=8, color="#00d4aa", opacity=0.7),
                name=f"LHS (N={_N})",
            )
        )
        _fig.update_layout(
            xaxis_title="Parameter 1 [0, 1]",
            yaxis_title="Parameter 2 [0, 1]",
            xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#2a2a3a"),
            yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#2a2a3a"),
        )
    else:
        _fig.add_trace(
            go.Scatter3d(
                x=_lhs[:, 0],
                y=_lhs[:, 1],
                z=_lhs[:, 2],
                mode="markers",
                marker=dict(size=5, color=_lhs[:, 0], colorscale="Tealgrn", opacity=0.8, showscale=False),
                name=f"LHS (N={_N})",
            )
        )
        _fig.update_layout(
            scene=dict(
                xaxis_title="Param 1",
                yaxis_title="Param 2",
                zaxis_title="Param 3",
                bgcolor="#12121a",
            ),
        )

    _fig.update_layout(
        title=dict(text=f"Latin Hypercube Samples: N={_N}, d={_d}", x=0.5),
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0a0a0f",
        margin=dict(l=30, r=30, t=50, b=40),
    )
    return


@app.cell
def __(mo, n_params_slider, n_samples_slider):
    _N = int(n_samples_slider.value)
    _d = int(n_params_slider.value)
    from math import comb

    _n_terms = comb(_d + 3, 3)

    mo.md(f"""
### Sampling Math

**PyTUQ PCRV** generates samples in *germ space* `ξ ∈ [-1, 1]ᵈ`:

```python
input_pc = PCRV(dom=bounds, pctype="LU", order=1)  # Legendre, uniform
germ = input_pc.sampleGerm(N)     # LHS in [-1, 1]
X = input_pc.evalPC(germ)          # map to physical space
```

**Physical space:** `x(ξ) = ½(lb + ub) + ½(ub - lb) · ξ`

**PCE terms:** `{_n_terms}` (for d={_d}, p=3)
**Rule of thumb:** N ≥ 2 × {_n_terms} = {2 * _n_terms} samples recommended

**`uq sample` does:**
1. Load `simData.cPickle` → build `XSpaceVecoli`
2. `PCRV.sampleGerm(N)` → LHS germ samples
3. Run vEcoli (subprocess or SMS-API) for each sample
4. Cache `(X, Y, timeseries)` → `./uq_cache/`
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 3 — PCE SURROGATE
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(go, level, mo, np):
    mo.stop(int(level.value) < 3)

    from libuq.inputs import XSpaceVecoli
    from libuq.pipeline.models import SimDataParameter
    from uq.workflow import run_uqpc

    rng = np.random.RandomState(42)
    _n_samp = 80
    _n_p = 3
    _X = rng.uniform(0, 1, (_n_samp, _n_p))
    _Y = (3 * _X[:, 0] + 1.5 * _X[:, 1] + 0.3 * _X[:, 2] + 0.5 * _X[:, 0] * _X[:, 1]).reshape(-1, 1)

    _params = [SimDataParameter(name=f"p{i}", attr_path=f"dummy.p{i}", bounds=(0.0, 1.0)) for i in range(_n_p)]
    _param_space = XSpaceVecoli(
        parameter_names=[],
        parameter_bounds=[],
        parameter_types=[],
        experiment_id="tutorial",
        parameters=_params,
    )

    _pce_result = run_uqpc(
        param_space=_param_space,
        Y_train=_Y,
        X_train=_X,
        polynomial_order=2,
        regression="lsq",
        seed=42,
    )

    _sweep = np.linspace(0, 1, 100)
    _X_sweep = np.zeros((100, _n_p))
    _X_sweep[:, 1] = 0.5
    _X_sweep[:, 2] = 0.5
    _X_sweep[:, 0] = _sweep
    _Y_pred = _pce_result.surrogate.predict(_X_sweep)

    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_X[:, 0],
            y=_Y.flatten(),
            mode="markers",
            marker=dict(size=6, color="#636efa", opacity=0.5),
            name="Training data",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_sweep,
            y=_Y_pred.flatten(),
            mode="lines",
            line=dict(color="#00d4aa", width=3),
            name="PCE prediction (p=2)",
        )
    )
    _fig.update_layout(
        title=dict(text="PCE Surrogate: Fitted Response Curve", x=0.5),
        xaxis_title="Parameter p₁ [0, 1]",
        yaxis_title="Output Y",
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        margin=dict(l=50, r=20, t=50, b=40),
    )

    _relerr = _pce_result.relerr_train
    _avg_relerr = float(np.mean(_relerr)) if _relerr is not None else 0.0
    _n_terms = len(_pce_result.surrogate.multi_indices)
    _p_order = _pce_result.surrogate.polynomial_order

    return XSpaceVecoli, SimDataParameter, _avg_relerr, _n_p, _n_samp, _n_terms, _p_order, _pce_result, go, np, run_uqpc


@app.cell
def __(mo):
    mo.md(r"""
### PCE Surrogate — Live Fit

Just now fitted a **Legendre PCE** (order 2, least squares) on synthetic data.

**The math:**

```
Ŷ(ξ) = Σ c_α · Φ_α(ξ)    where |α| ≤ p
```

- `Φ_α` = multivariate Legendre polynomial (orthogonal on [-1,1])
- `c_α` = coefficients fitted by least squares
- `α` = multi-index (e.g., [1,0,0] = linear in p₁)

**Why PCE?** 80 samples → analytical Sobol. Monte Carlo would need 10,000+.
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 4 — SOBOL INDICES
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(go, level, mo, np):
    mo.stop(int(level.value) < 4)

    from libuq.inputs import XSpaceVecoli
    from libuq.pipeline.models import SimDataParameter
    from uq.workflow import run_uqpc

    rng = np.random.RandomState(123)
    _n = 100
    _d = 3
    _X2 = rng.uniform(0, 1, (_n, _d))
    _Y2 = (5 * _X2[:, 0] + 2 * _X2[:, 1] + 0.1 * _X2[:, 2] + 1.5 * _X2[:, 0] * _X2[:, 1]).reshape(-1, 1)

    _params2 = [SimDataParameter(name=f"p{i}", attr_path=f"dummy.p{i}", bounds=(0.0, 1.0)) for i in range(_d)]
    _ps2 = XSpaceVecoli(
        parameter_names=[],
        parameter_bounds=[],
        parameter_types=[],
        experiment_id="tutorial2",
        parameters=_params2,
    )

    _sobol_result = run_uqpc(
        param_space=_ps2,
        Y_train=_Y2,
        X_train=_X2,
        polynomial_order=3,
        regression="lsq",
        seed=42,
    )

    _s1 = _sobol_result.sobol.first_order
    _st = _sobol_result.sobol.total_order
    _names2 = ["p₁ (dominant)", "p₂ (moderate)", "p₃ (negligible)"]

    _fig = go.Figure()
    _x = np.arange(3)
    _w = 0.35
    _fig.add_trace(
        go.Bar(
            x=_x - _w / 2,
            y=_s1,
            width=_w,
            name="Sᵢ (main effect)",
            marker_color="#636efa",
        )
    )
    _fig.add_trace(
        go.Bar(
            x=_x + _w / 2,
            y=_st,
            width=_w,
            name="S_Ti (total effect)",
            marker_color="#00d4aa",
        )
    )
    _interactions = _st - _s1

    _fig.update_layout(
        barmode="group",
        title=dict(text="Sobol Sensitivity Indices (analytical from PCE)", x=0.5),
        xaxis=dict(ticktext=_names2, tickvals=list(range(3)), title="Parameter"),
        yaxis_title="Fraction of Variance Explained",
        yaxis=dict(range=[0, max(_st.max(), _s1.max()) * 1.2]),
        template="plotly_dark",
        height=380,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        margin=dict(l=50, r=20, t=50, b=80),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    return XSpaceVecoli, SimDataParameter, _fig, _interactions, _s1, _st, go, np, run_uqpc


@app.cell
def __(mo):
    mo.md(r"""
### Sobol Indices — What Drives Variance?

**First-order Sᵢ** = main effect of parameter i alone.
**Total-order S_Ti** = main + all interaction effects involving i.
**S_Ti − Sᵢ** = higher-order interactions.

In real vEcoli analyses, this tells you:
- Which parameters to measure more precisely
- Which parameters can be fixed at nominal values
- Whether parameters interact synergistically
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 5 — 4 AGGREGATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(level, mo):
    mo.stop(int(level.value) < 5)

    strategy_selector = mo.ui.dropdown(
        options={
            "Strategy 1: Uniform (bulk)": 1,
            "Strategy 2: By Generation": 2,
            "Strategy 3: By Lineage Seed": 3,
            "Strategy 4: Growth-Stratified (θ)": 4,
        },
        value=1,
        label="Aggregation Strategy",
    )
    return (strategy_selector,)


@app.cell
def __(mo, strategy_selector):
    _strat_desc = {
        1: """
        **Uniform** — average over all cells, all times.

        This is the baseline "bulk" sensitivity. Answers:
        *Which parameters matter for the population as a whole?*

        **CLI:** Default. No special flag needed.
        """,
        2: """
        **By Generation** — compute Sobol indices per cell generation.

        Controls for convergence toward steady-state. Early generations
        may show different sensitivity patterns than later ones.

        **CLI:** Requires `--generations >= 2` in `uq sample`.
        """,
        3: """
        **By Lineage Seed** — compute Sobol per initial stochastic seed.

        Controls for gene expression noise and stochastic partitioning.
        If Sobol indices vary across seeds, stochastic effects dominate.

        **CLI:** Requires `--n-init-sims >= 2` in `uq sample`.
        """,
        4: """
        **Growth-Stratified (θ)** — bin cells by cell cycle position.

        Cell cycle coordinate:
        `θ = [log(m) − log(m_birth)] / [log(m_div) − log(m_birth)]`

        Each θ-bin gets its own PCE + Sobol. Answers:
        *When during the cell cycle is each parameter most influential?*

        **CLI:** Requires `--n-bins N` in `uq quantify`.
        """,
    }

    _s = int(strategy_selector.value)
    mo.md(f"""
### Strategy {_s}

{_strat_desc.get(_s, "")}

| Strategy | Groups By | Purpose | CLI Requirement |
|---|---|---|---|
| 1 | None | Population average | None |
| 2 | Generation | Convergence check | `--generations >= 2` |
| 3 | Lineage seed | Stochastic variance | `--n-init-sims >= 2` |
| 4 | θ-bin (cell cycle) | Phase-specific GSA | `--n-bins N` |

**All 4 strategies run automatically** in a single `uq quantify` call.
Each produces its own set of Sobol indices.
""")
    return


@app.cell
def __(mo):
    mo.md(r"""
### Why 4 Strategies?

vEcoli simulations produce **timeseries** — thousands of data points
per cell. We must reduce this to a single Y value per sample for
PCE fitting. *How* we aggregate determines *what* we learn.

```
Raw data:  ──────────────────────────────────► Y per sample
              Uniform    Generation    Seed    θ-bin
```

The **variance decomposition** step tells you how much total output
variance comes from generation drift vs stochastic seed noise vs
cell cycle dynamics — before any PCE is fit.

**In `uq quantify` output:**
- Strategy 1 → `population_surrogate/`, `population_sobol/`
- Strategy 2 → `generation_0_sobol/`, `generation_1_sobol/`, ...
- Strategy 3 → `seed_0_sobol/`, `seed_1000_sobol/`, ...
- Strategy 4 → `growth_stage_0_sobol/`, ..., `growth_stage_9_sobol/`
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 6 — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(level, mo):
    mo.stop(int(level.value) < 6)

    mo.md(r"""
### The Complete Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  uq sample  (compute-intensive, runs vEcoli)                │
│                                                              │
│  1. Load simData.cPickle → XSpaceVecoli                     │
│  2. PCRV.sampleGerm(N) → LHS samples in germ space          │
│  3. Build variants → vEcoli config JSON                     │
│  4. Run runscripts/workflow.py (subprocess or SMS-API)      │
│  5. Collect Parquet → extract observables                   │
│  6. Cache X, Y, timeseries → ./uq_cache/                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  uq quantify  (fast, repeatable, no vEcoli)                 │
│                                                              │
│  1. Load cache → X (n×d), Y (n×m)                          │
│  2. For each of 4 strategies:                                │
│     a. Aggregate Y by strategy                                │
│     b. Scale X to germ space [-1, 1]                         │
│     c. Build Legendre basis, fit PCE coefficients             │
│     d. Compute Sobol indices analytically                     │
│  3. Export: uq_results.json + surrogate binaries + HTML      │
└──────────────────────────────────────────────────────────────┘
```

**The key design principle:** `sample` is expensive (hours of vEcoli).
`quantify` is fast (seconds of math). This separation means you can
iterate on analysis without re-running simulations.
""")
    return


@app.cell
def __(mo):
    sample_n = mo.ui.slider(10, 200, step=10, value=50, label="--n-samples", show_value=True)
    sample_gens = mo.ui.slider(1, 10, step=1, value=2, label="--generations", show_value=True)
    sample_seeds = mo.ui.slider(1, 5, step=1, value=2, label="--n-init-sims", show_value=True)
    sample_obs = mo.ui.dropdown(
        options=["mass", "higher_order", "exchange_fluxes", "transcriptome", "proteome", "fluxome"],
        value="higher_order",
        label="--observables",
    )
    quantify_order = mo.ui.slider(1, 5, step=1, value=2, label="--polynomial-order", show_value=True)
    quantify_reg = mo.ui.dropdown(
        options={"lsq (least squares)": "lsq", "bcs (sparse)": "bcs", "anl (analytical)": "anl"},
        value="lsq",
        label="--regression",
    )
    return (
        quantify_order,
        quantify_reg,
        sample_gens,
        sample_n,
        sample_obs,
        sample_seeds,
    )


@app.cell
def __(
    mo,
    quantify_order,
    quantify_reg,
    sample_gens,
    sample_n,
    sample_obs,
    sample_seeds,
):
    _cmd_sample = f"""```bash
uv run uq sample /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --n-samples {int(sample_n.value)} \\
    --generations {int(sample_gens.value)} \\
    --n-init-sims {int(sample_seeds.value)} \\
    --observables {sample_obs.value}
```"""

    _cmd_quantify = f"""```bash
uv run uq quantify /path/to/simData.cPickle \\
    --cache-dir ./uq_cache \\
    --export-path ./uq_results \\
    --polynomial-order {int(quantify_order.value)} \\
    --regression {quantify_reg.value}
```"""

    mo.md(f"""
### Build Your Command

Adjust the sliders above to construct real CLI commands:

{_cmd_sample}

{_cmd_quantify}

**After quantify:** `report.html` opens in your browser with interactive
Sobol bar charts, PCE response curves, and a sensitivity spectrogram.
""")
    return


# ═══════════════════════════════════════════════════════════════════
#  LEVEL 7 — ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(level, mo):
    mo.stop(int(level.value) < 7)

    feature_selector = mo.ui.dropdown(
        options={
            "🔬 Output PCA": "pca",
            "📊 Baseline (Q1)": "baseline",
            "☁️  Remote (SMS-API)": "remote",
            "🔄 Multi-Condition": "multicond",
            "📐 Held-out Validation": "validation",
            "📦 HPC Batch": "hpc",
        },
        value="pca",
        label="Feature",
    )
    return (feature_selector,)


@app.cell
def __(feature_selector, mo):
    _features = {
        "pca": """
### Output-Side PCA for High-Dimensional Observables

When Y has thousands of columns (transcriptome, proteome), fitting
individual PCE surrogates is expensive and noisy.

```bash
uv run uq quantify ... --output-pca 5
```

**What happens:**
1. Center Y, compute SVD → top-K principal components
2. Fit K PCE surrogates (one per PC) instead of n_obs
3. Sobol indices are per-PC: "which parameter drives PC1?"
4. Top loadings reveal biology: "PC1 ≈ growth signature"

**Files exported:**
- `pca/pca_loadings.npy` — (K, n_obs) loading matrix
- `pca/pca_explained_variance.npy` — variance ratio per PC
- `pca/pca_top_loadings.json` — top-10 observables per PC

**Recommended:** `--observables transcriptome --output-pca 5`
""",
        "baseline": """
### Q1 Baseline Variance Accounting (--no-perturbation)

Run vEcoli at **fixed baseline** parameters (no LHS perturbation)
to decompose observable variance into natural sources:

```bash
uv run uq sample /path/to/simData.cPickle --no-perturbation \\
    --n-init-sims 5 --generations 3
uv run uq quantify /path/to/simData.cPickle --no-perturbation
```

**Decomposition:** `σ²_total = σ²_gen + σ²_seed + σ²_θ + σ²_residual`

| Component | Meaning |
|---|---|
| σ²_gen | Drift across generations (convergence) |
| σ²_seed | Stochastic seed-to-seed variation |
| σ²_θ | Cell cycle phase variation |
| σ²_residual | Unexplained (numerical noise) |

**This is the variance denominator** — the natural variability
that parameter perturbations must exceed to be detectable.
""",
        "remote": """
### Remote Execution via SMS-API

Run vEcoli on the cluster instead of locally:

```bash
uv run uq sample /path/to/simData.cPickle \\
    --api-url https://sms.cam.uchc.edu \\
    --simulator-id 11 \\
    --n-samples 50 \\
    --conditions glucose_minus_aas \\
    --conditions glucose_plus_aas
```

**What happens:**
1. Build N variants from LHS samples
2. Submit ONE simulation with N variants to AWS Batch
3. Poll until complete (with progress bar)
4. Download cd1 analysis TSVs
5. Parse → cache (same format as local)

**Key advantage:** No local vEcoli installation needed.
The `uq quantify` step works identically regardless of execution mode.

**Fetch outputs:** `uv run uq fetch <sim_id> --api-url URL`
""",
        "multicond": """
### Cross-Condition GSA

Run the same UQ analysis under multiple growth conditions:

```bash
uv run uq sample /path/to/simData.cPickle \\
    --conditions vecoli_m9_glucose_minus_aas \\
    --conditions vecoli_m9_glucose_plus_aas
uv run uq quantify /path/to/simData.cPickle  # auto-detects
```

**Auto-detected:** `quantify` sees the multi-condition cache and runs
per-condition + cross-condition analysis:

| Metric | Meaning |
|---|---|
| Rank stability | How consistent is parameter ranking across conditions? |
| Universal drivers | Parameters with S_Ti > 10% in ALL conditions |
| Condition-specific | Parameters that matter in only one condition |
| Differential Sobol | ΔS_Ti = |S_Ti(cond_A) − S_Ti(cond_B)| |

**Output:** Comparison table with rank stability indicators (●●●○○)
""",
        "validation": """
### Held-out Validation (--n-test)

Reserve samples for out-of-sample PCE quality assessment:

```bash
uv run uq sample ... --n-samples 50 --n-test 10
```

**What happens:**
- 50 training samples → fit PCE
- 10 test samples → compute relative error `||Y − Ŷ|| / ||Y||`

**In quantify output:**
```
SURROGATE QUALITY
┌──────────┬───────────┬──────────┐
│ OUTPUT   │   TRAIN   │   TEST   │
├──────────┼───────────┼──────────┤
│ output[0]│  1.2e-03  │  1.8e-03 │
│ output[1]│  2.1e-03  │  2.9e-03 │
└──────────┴───────────┴──────────┘
```

If test error >> train error → overfitting. Consider lower `--polynomial-order`.
""",
        "hpc": """
### HPC Batch Workflow

For large-scale runs on clusters:

```bash
# 1. Export per-sample configs (no vEcoli run)
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200

# 2. Run on cluster (Nextflow, Slurm, etc.)
# Each job gets its own simData/{i}.cPickle + config/{i}.json

# 3. Collect results into cache
uv run uq collect-results ./batch ./batch_outputs --cache-dir ./uq_cache

# 4. Analyze
uv run uq quantify exp1 --cache-dir ./uq_cache
```

**Key insight:** Variants are baked into pickled `simData` at export time.
HPC jobs don't need the UQ codebase — just vEcoli.

**Metadata:** `metadata.json` maps sample index → parameter values.
""",
    }
    mo.md(_features.get(feature_selector.value, ""))
    return


# ═══════════════════════════════════════════════════════════════════
#  INTERACTIVE PCE EXPLORER (always visible)
# ═══════════════════════════════════════════════════════════════════


@app.cell
def __(mo, np):
    from libuq.inputs import XSpaceVecoli
    from libuq.pipeline.models import SimDataParameter
    from uq.workflow import run_uqpc
    import plotly.graph_objects as go

    rng = np.random.RandomState(42)
    _N = 100
    _d = 3
    _X_live = rng.uniform(0, 1, (_N, _d))
    _Y_live = (
        3 * _X_live[:, 0]
        + 1.5 * _X_live[:, 1]
        + 0.3 * _X_live[:, 2]
        + 0.5 * _X_live[:, 0] * _X_live[:, 1]
        + 0.2 * _X_live[:, 0] ** 2
    ).reshape(-1, 1)

    _params_live = [SimDataParameter(name=f"p{i}", attr_path=f"dummy.p{i}", bounds=(0.0, 1.0)) for i in range(_d)]
    _ps_live = XSpaceVecoli(
        parameter_names=[],
        parameter_bounds=[],
        parameter_types=[],
        experiment_id="live",
        parameters=_params_live,
    )

    _live_result = run_uqpc(
        param_space=_ps_live,
        Y_train=_Y_live,
        X_train=_X_live,
        polynomial_order=3,
        regression="lsq",
        seed=42,
    )

    return XSpaceVecoli, SimDataParameter, _d, _live_result, go, np, run_uqpc


@app.cell
def __(mo):
    p1 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₁", show_value=True)
    p2 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₂", show_value=True)
    p3 = mo.ui.slider(0, 1, step=0.01, value=0.5, label="p₃", show_value=True)
    return p1, p2, p3


@app.cell
def __(_live_result, go, mo, np, p1, p2, p3):
    _x_curr = np.array([[p1.value, p2.value, p3.value]])
    _y_curr = float(_live_result.surrogate.predict(_x_curr)[0, 0])

    _s1 = _live_result.sobol.first_order
    _st = _live_result.sobol.total_order

    _n_sweep = 50
    _fig = go.Figure()
    _colors = ["#00d4aa", "#636efa", "#EF553B"]
    _param_names = ["p₁", "p₂", "p₃"]

    for _i in range(3):
        _X_sw = np.array([[p1.value, p2.value, p3.value]] * _n_sweep)
        _X_sw[:, _i] = np.linspace(0, 1, _n_sweep)
        _Y_sw = _live_result.surrogate.predict(_X_sw).flatten()

        _fig.add_trace(
            go.Scatter(
                x=np.linspace(0, 1, _n_sweep),
                y=_Y_sw,
                mode="lines",
                name=f"Sweep {_param_names[_i]}",
                line=dict(color=_colors[_i], width=2),
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=[_x_curr[0, _i]],
                y=[_y_curr],
                mode="markers",
                name=f"Current {_param_names[_i]}",
                marker=dict(symbol="diamond", size=12, color=_colors[_i]),
                showlegend=False,
            )
        )

    _fig.update_layout(
        title=dict(text="PCE Response Curves — Move Sliders to Explore", x=0.5),
        xaxis_title="Parameter Value",
        yaxis_title="Predicted Output Ŷ",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    _fig2 = go.Figure()
    _x_pos = np.arange(3)
    _w = 0.35
    _fig2.add_trace(
        go.Bar(
            x=_x_pos - _w / 2,
            y=_s1,
            width=_w,
            name="Sᵢ (main)",
            marker_color="#636efa",
        )
    )
    _fig2.add_trace(
        go.Bar(
            x=_x_pos + _w / 2,
            y=_st,
            width=_w,
            name="S_Ti (total)",
            marker_color="#00d4aa",
        )
    )
    _fig2.update_layout(
        barmode="group",
        title=dict(text="Sobol Indices", x=0.5),
        xaxis=dict(ticktext=["p₁", "p₂", "p₃"], tickvals=list(range(3))),
        yaxis_title="Variance Fraction",
        template="plotly_dark",
        height=280,
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    _relerr = _live_result.relerr_train
    _avg_err = float(np.mean(_relerr)) if _relerr is not None else 0.0

    _info = mo.md(
        f"""
        **Current point:** Ŷ({p1.value:.2f}, {p2.value:.2f}, {p3.value:.2f}) = **{_y_curr:.3f}**

        | Metric | Value |
        |---|---|
        | Avg train error | {_avg_err:.2e} |
        | PCE order | {_live_result.surrogate.polynomial_order} |
        | Basis terms | {len(_live_result.surrogate.multi_indices)} |
        """
    )
    return _fig, _fig2, _info


@app.cell
def __(mo):
    mo.md(r"""
---

### 🎮 Interactive PCE Explorer

Move the sliders to see how the PCE surrogate responds. The curves
show the predicted output as each parameter varies (holding others
constant). The Sobol indices quantify each parameter's contribution
to total output variance.

This is the same interactive experience available in `uq dashboard`
and `uq gui` after running the full pipeline on real vEcoli data.
""")
    return


@app.cell
def __(_fig, _fig2, _info, mo):
    mo.vstack([
        mo.hstack([
            mo.md("### 🔬 PCE Explorer"),
            _info,
        ]),
        _fig,
        _fig2,
    ])
    return


@app.cell
def __(mo):
    mo.md(r"""
---

## 🏁 Summary

You've explored the complete UQ workflow:

| Level | Concept | CLI Command |
|---|---|---|
| 1 | **Why UQ** | — |
| 2 | **Parameter Space** | `--params-file` |
| 3 | **Sampling** | `uq sample` (Steps 1-3) |
| 4 | **PCE Surrogate** | fitted in `uq quantify` |
| 5 | **Sobol Indices** | computed in `uq quantify` |
| 6 | **4 Strategies** | all 4 run automatically |
| 7 | **Full Pipeline** | `sample` → `quantify` |
| 8 | **Advanced** | PCA, baseline, remote |

**Next steps:**
- `uv run uq gui` — full browser GUI with live workflow
- `uv run uq tui` — terminal-based interactive interface
- `uv run uq dashboard` — explore existing results
- Read the full docs at ReadTheDocs
""")
    return


if __name__ == "__main__":
    app.run()
