"""Tutorial 10: RFC006 UQ Pipeline Dashboard

Interactive dashboard for the full RFC006 uncertainty quantification pipeline
using real vEcoli simulation data. Serves as a GUI alternative to the `uq` CLI.

Requires:
  - Real simulation Parquet data at sim_base_path
  - Optionally, a PrecomputedCache from `uv run uq generate-samples --live`

Run with: uv run marimo run tutorials/10_dashboard.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SETUP
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # RFC006 UQ Pipeline Dashboard

    Interactive GUI for the full **RFC006 uncertainty quantification pipeline**
    on real vEcoli simulation data. This dashboard is equivalent to running
    `uv run uq quantify` from the CLI.

    ```
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  RFC006 Pipeline                                                             │
    │                                                                              │
    │  Steps 1-4 (shared):                                                         │
    │  Parameter Space → Load Data → Aggregate (strategies 1-3) → Variance Decomp  │
    │                                                            │                 │
    │                                             ┌──────────────┼──────────┐      │
    │  Phase 1 (bulk):                            │  Phase 2 (cell cycle):  │      │
    │  Morris → PCE → Sobol                       │  GSA obs → Koopman θ → │      │
    │  "Which params drive bulk variance?"        │  per-stage Sobol        │      │
    │                                             └─────────────────────────┘      │
    │                                                                              │
    │  Output: PipelineResult(population=UqProfile, cell_cycle=UqProfile)          │
    └──────────────────────────────────────────────────────────────────────────────┘
    ```

    **Point this at your real simulation output directory.** Optionally load a
    `PrecomputedCache` from `uv run uq generate-samples --live` to skip sample
    generation.
    """)
    return


@app.cell
def _():
    import math
    import os
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, math, np, os, pl


@app.cell
def _():
    from uq.pipe import Pipeline, initialize_data
    from uq.pipeline.workflow import (
        AggregationResult,
        aggregate_timeseries,
        get_variance_decomposition,
        run_phase1,
        run_phase2,
    )
    from uq.pipeline.models import (
        PipelineResult,
        StratificationLens,
        UqProfile,
    )
    from uq.sensitivity import (
        MorrisIndices,
        PCESurrogate,
        SensitivityAnalyzer,
        SobolIndices,
    )
    from uq.wrappers import DataDrivenWrapper
    from uq.sampling import PrecomputedCache
    from uq.pce.models import PCEParameterSelectionConfig
    from uq.koopman import DynamicModeDecomposition
    from uq.viz import plot_koopman_spectrum

    return (
        AggregationResult,
        DataDrivenWrapper,
        DynamicModeDecomposition,
        MorrisIndices,
        PCEParameterSelectionConfig,
        PCESurrogate,
        Pipeline,
        PipelineResult,
        PrecomputedCache,
        SensitivityAnalyzer,
        SobolIndices,
        StratificationLens,
        UqProfile,
        aggregate_timeseries,
        get_variance_decomposition,
        initialize_data,
        plot_koopman_spectrum,
        run_phase1,
        run_phase2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo, os):
    sim_base_path_input = mo.ui.text(
        value=os.getenv(
            "SIM_BASE_PATH",
            "/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims",
        ),
        label="Simulation base path",
        full_width=True,
    )
    experiment_ids_input = mo.ui.text(
        value="api_simulation_default,mecillinam,test_violacein_with_metabolism",
        label="Experiment IDs (comma-separated)",
        full_width=True,
    )
    precomputed_path_input = mo.ui.text(
        value="./uq_cache",
        label="Precomputed cache path (optional)",
    )
    export_path_input = mo.ui.text(
        value="./uq_results",
        label="Export path",
    )
    use_cache_switch = mo.ui.switch(value=False, label="Load precomputed cache for Phase 1")
    n_samples_slider = mo.ui.slider(
        start=10, stop=500, step=10, value=50,
        label="PCE samples (N)",
        show_value=True,
    )
    polynomial_order_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="Polynomial order (p)",
        show_value=True,
    )
    n_bins_slider = mo.ui.slider(
        start=3, stop=20, step=1, value=10,
        label="Cell cycle bins",
        show_value=True,
    )
    cycle_time_slider = mo.ui.slider(
        start=600, stop=7200, step=300, value=3600,
        label="Expected cycle time (s)",
        show_value=True,
    )
    morris_switch = mo.ui.switch(value=True, label="Enable Morris prescreening")

    return (
        cycle_time_slider,
        experiment_ids_input,
        export_path_input,
        morris_switch,
        n_bins_slider,
        n_samples_slider,
        polynomial_order_slider,
        precomputed_path_input,
        sim_base_path_input,
        use_cache_switch,
    )


@app.cell
def _(
    cycle_time_slider,
    experiment_ids_input,
    export_path_input,
    mo,
    morris_switch,
    n_bins_slider,
    n_samples_slider,
    polynomial_order_slider,
    precomputed_path_input,
    sim_base_path_input,
    use_cache_switch,
):
    _left = mo.vstack([
        mo.md("### Data Source"),
        sim_base_path_input,
        experiment_ids_input,
        mo.md("---"),
        mo.md("### Cache / Export"),
        use_cache_switch,
        precomputed_path_input,
        export_path_input,
        morris_switch,
    ])
    _right = mo.vstack([
        mo.md("### PCE Configuration"),
        n_samples_slider,
        polynomial_order_slider,
        mo.md("---"),
        mo.md("### Cell Cycle"),
        n_bins_slider,
        cycle_time_slider,
    ])
    mo.hstack([_left, _right], widths=[1, 1])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: STEP 1-2 — PARAMETER SPACE + DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Steps 1-2: Parameter Space + Data Loading

    **RFC006:** *"Identify the scientifically most relevant input and output variables."*

    Loads real simulation Parquet data from disk via `initialize_data()`. The parameter
    space is auto-detected from `simData.cPickle` (vio pathway, mecillinam, knockouts).
    """)
    return


@app.cell
def _(experiment_ids_input, initialize_data, mo, sim_base_path_input):
    _exp_ids = [e.strip() for e in experiment_ids_input.value.split(",") if e.strip()]

    ds = initialize_data(
        experiment_ids=_exp_ids,
        sim_base_path=sim_base_path_input.value,
    )

    param_space = ds.parameter_space
    observable_columns = ds.observables
    sim_data = ds.y

    _param_rows = ""
    for _i, _name in enumerate(param_space.parameter_names):
        _lo, _hi = param_space.parameter_bounds[_i]
        _param_rows += f"| `{_name}` | {_lo:.2f} | {_hi:.2f} |\n"

    mo.md(f"""
**{param_space.n_parameters} parameters detected:**

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
{_param_rows}

**Data loaded:** {len(sim_data)} rows, {len(observable_columns)} observables

| Stat | Value |
|------|-------|
| Experiments | {_exp_ids} |
| Seeds | {sim_data["lineage_seed"].n_unique()} |
| Generations | {sim_data["generation"].n_unique()} |
| Observables | {len(observable_columns)} |
    """)
    return ds, observable_columns, param_space, sim_data


@app.cell
def _(Path, PrecomputedCache, mo, precomputed_path_input, use_cache_switch):
    cache = None
    if use_cache_switch.value:
        _cache_path = Path(precomputed_path_input.value)
        if _cache_path.exists():
            cache = PrecomputedCache.load(_cache_path)
            mo.md(f"""
**PrecomputedCache loaded** from `{precomputed_path_input.value}`
- Samples: {cache.X.shape[0]} x {cache.X.shape[1]} params
- Outputs: {cache.Y.shape[1]} columns
- Timeseries: {"Yes (" + str(len(cache.Y_timeseries)) + " samples)" if cache.Y_timeseries else "No"}
            """)
        else:
            mo.md(f"**Warning:** Cache path `{precomputed_path_input.value}` not found.")
    else:
        mo.md("*No precomputed cache — Phase 1 will use DataDrivenWrapper (response surface from data statistics).*")
    return (cache,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: STEP 3 — AGGREGATION (STRATEGIES 1-3)
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Aggregation Strategies 1-3

    **RFC006:** *"Uniformly across all simulated cells and times (baseline),
    stratified by generation, and stratified by lineage seed."*
    """)
    return


@app.cell
def _(aggregate_timeseries, mo, np, observable_columns, sim_data):
    agg_result = aggregate_timeseries(sim_data, observable_columns)

    _uniform_stats = ""
    for _i, _col in enumerate(observable_columns[:10]):
        _short = _col.split("__")[-1]
        _mean = agg_result.uniform.mean[_i]
        _std = agg_result.uniform.std[_i]
        _uniform_stats += f"| `{_short}` | {_mean:.6g} | {_std:.6g} |\n"
    if len(observable_columns) > 10:
        _uniform_stats += f"| ... | ({len(observable_columns) - 10} more) | |\n"

    _n_gens = len(agg_result.generation.groups) if agg_result.generation.groups is not None else "N/A"
    _n_seeds = len(agg_result.seed.groups) if agg_result.seed.groups is not None else "N/A"

    mo.md(f"""
**Strategy 1 — Uniform** ({agg_result.uniform.n_samples} data points):

| Observable | Mean | Std |
|------------|------|-----|
{_uniform_stats}

**Strategy 2 — By Generation:** {_n_gens} groups |
**Strategy 3 — By Lineage Seed:** {_n_seeds} groups
    """)
    return (agg_result,)


@app.cell
def _(agg_result, go, make_subplots, np, observable_columns):
    # Show at most 10 observables for readability
    _display_cols = observable_columns[:10]
    _short_names = [c.split("__")[-1] for c in _display_cols]
    _n_display = len(_display_cols)

    agg_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Uniform Mean +/- Std",
            "Generation Convergence",
            "Seed Variation",
        ],
        horizontal_spacing=0.08,
    )

    # Panel 1: Uniform bar chart
    agg_fig.add_trace(
        go.Bar(
            x=_short_names,
            y=agg_result.uniform.mean[:_n_display],
            error_y=dict(type="data", array=agg_result.uniform.std[:_n_display], visible=True),
            marker_color="cyan",
            name="Uniform",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Panel 2: Generation means
    if agg_result.generation.groups is not None and agg_result.generation.mean.ndim == 2:
        _n_gen_obs = min(_n_display, agg_result.generation.mean.shape[1])
        for _j in range(_n_gen_obs):
            agg_fig.add_trace(
                go.Scatter(
                    x=agg_result.generation.groups,
                    y=agg_result.generation.mean[:, _j],
                    name=_short_names[_j],
                    mode="lines+markers",
                ),
                row=1, col=2,
            )
    agg_fig.update_xaxes(title="Generation", row=1, col=2)

    # Panel 3: Seed means
    if agg_result.seed.groups is not None and agg_result.seed.mean.ndim == 2:
        _n_seed_obs = min(_n_display, agg_result.seed.mean.shape[1])
        for _j in range(_n_seed_obs):
            agg_fig.add_trace(
                go.Bar(
                    x=[str(s) for s in agg_result.seed.groups],
                    y=agg_result.seed.mean[:, _j],
                    name=_short_names[_j],
                    showlegend=False,
                ),
                row=1, col=3,
            )
    agg_fig.update_xaxes(title="Seed", row=1, col=3)

    agg_fig.update_layout(
        height=400,
        template="plotly_dark",
        margin=dict(t=40, b=40),
    )
    agg_fig
    return (agg_fig,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: STEP 4 — VARIANCE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Variance Decomposition

    Decomposes total variance into **generation** (convergence), **seed** (exogenous),
    and **residual** (cell-cycle-related) components. The residual fraction identifies
    observables whose variance is attributable to cell cycle dynamics — these feed Phase 2.
    """)
    return


@app.cell
def _(agg_result, get_variance_decomposition, go, mo, np, observable_columns):
    decomp = get_variance_decomposition(agg_result)

    _display_cols = observable_columns[:10]
    _short_names = [c.split("__")[-1] for c in _display_cols]
    _n = len(_short_names)

    _gen_frac = np.asarray(decomp.get("generation_fraction", [0.0]))
    _seed_frac = np.asarray(decomp.get("seed_fraction", [0.0]))

    if len(_gen_frac) < _n:
        _gen_frac = np.full(_n, _gen_frac.mean() if len(_gen_frac) > 0 else 0.33)
    if len(_seed_frac) < _n:
        _seed_frac = np.full(_n, _seed_frac.mean() if len(_seed_frac) > 0 else 0.33)
    _residual_frac = np.clip(1.0 - _gen_frac[:_n] - _seed_frac[:_n], 0, 1)

    decomp_fig = go.Figure()
    decomp_fig.add_trace(go.Bar(
        x=_short_names, y=_gen_frac[:_n] * 100,
        name="Generation", marker_color="#e74c3c",
    ))
    decomp_fig.add_trace(go.Bar(
        x=_short_names, y=_seed_frac[:_n] * 100,
        name="Seed", marker_color="#3498db",
    ))
    decomp_fig.add_trace(go.Bar(
        x=_short_names, y=_residual_frac * 100,
        name="Residual (cell cycle)", marker_color="#2ecc71",
    ))
    decomp_fig.update_layout(
        barmode="stack",
        yaxis_title="Variance Fraction (%)",
        height=400,
        template="plotly_dark",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )

    mo.vstack([
        decomp_fig,
        mo.md(f"""
**Mean fractions:** Generation {_gen_frac[:_n].mean()*100:.1f}% |
Seed {_seed_frac[:_n].mean()*100:.1f}% |
Residual {_residual_frac.mean()*100:.1f}%

The **residual fraction** represents variance not explained by generation or
seed effects — this is the cell-cycle-related variance that drives Phase 2.
        """),
    ])
    return (decomp,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: STEP 5a — MORRIS PRESCREENING
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Step 5a: Morris Prescreening (Optional)

    For high-dimensional parameter spaces, Morris screening efficiently identifies
    the most influential parameters before expensive PCE fitting.
    """)
    return


@app.cell
def _(
    DataDrivenWrapper,
    PCEParameterSelectionConfig,
    SensitivityAnalyzer,
    agg_result,
    go,
    mo,
    morris_switch,
    np,
    param_space,
):
    morris_indices = None
    prescreen_config = None

    if not morris_switch.value or param_space.n_parameters < 2:
        mo.md("*Morris prescreening skipped.*" if not morris_switch.value
               else "*Skipped — fewer than 2 parameters.*")
    else:
        _wrapper = DataDrivenWrapper(
            parameter_space=param_space,
            observable_means=agg_result.uniform.mean,
            observable_stds=agg_result.uniform.std,
        )
        _analyzer = SensitivityAnalyzer(param_space, _wrapper)
        morris_indices = _analyzer.analyze_with_morris(n_trajectories=10, n_levels=4)
        prescreen_config = PCEParameterSelectionConfig(
            n_trajectories=10,
            n_top=min(5, param_space.n_parameters),
        )

        _mu_star = np.asarray(morris_indices.mu_star)
        _sigma = np.asarray(morris_indices.sigma)

        _morris_fig = go.Figure()
        _morris_fig.add_trace(go.Scatter(
            x=_mu_star, y=_sigma,
            mode="markers+text",
            text=morris_indices.parameter_names,
            textposition="top center",
            marker=dict(size=12, color="magenta"),
        ))
        _max_mu = max(_mu_star) * 1.2 if len(_mu_star) > 0 else 1.0
        _morris_fig.add_trace(go.Scatter(
            x=[0, _max_mu], y=[0, 0.5 * _max_mu],
            mode="lines", line=dict(dash="dash", color="gray"),
            name="sigma = 0.5*mu* (linear threshold)",
        ))
        _morris_fig.update_layout(
            xaxis_title="mu* (mean absolute effect)",
            yaxis_title="sigma (interaction/nonlinearity)",
            height=350, template="plotly_dark",
            margin=dict(t=20, b=40),
        )

        _rows = ""
        for _i in np.argsort(_mu_star)[::-1]:
            _name = morris_indices.parameter_names[_i]
            _cls = "linear" if _sigma[_i] < 0.5 * _mu_star[_i] else "nonlinear/interactive"
            _rows += f"| `{_name}` | {_mu_star[_i]:.4f} | {_sigma[_i]:.4f} | {_cls} |\n"

        mo.vstack([
            _morris_fig,
            mo.md(f"""
| Parameter | mu* | sigma | Classification |
|-----------|-----|-------|----------------|
{_rows}
            """),
        ])
    return morris_indices, prescreen_config


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: STEP 5b — OBSERVABLE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Step 5b: Cell-Cycle-Relevant Observable Selection

    **RFC006:** *"The choice of the cell cycle variable will be informed by the
    sensitivity analyses (1-3)."*

    Observables with high **residual variance** (not explained by generation or seed)
    are selected for Koopman analysis.
    """)
    return


@app.cell
def _(decomp, mo, np, observable_columns):
    _gen_frac = np.asarray(decomp.get("generation_fraction", [0.33]))
    _seed_frac = np.asarray(decomp.get("seed_fraction", [0.33]))
    _n = len(observable_columns)

    if len(_gen_frac) < _n:
        _gen_frac = np.full(_n, _gen_frac.mean())
    if len(_seed_frac) < _n:
        _seed_frac = np.full(_n, _seed_frac.mean())
    _residual = np.clip(1.0 - _gen_frac[:_n] - _seed_frac[:_n], 0, 1)

    _threshold = float(np.median(_residual))
    relevant_observables = [
        observable_columns[_i] for _i in range(_n)
        if _residual[_i] >= _threshold
    ]
    if not relevant_observables:
        relevant_observables = observable_columns[:1]

    _rows = ""
    for _i, _col in enumerate(observable_columns[:15]):
        _short = _col.split("__")[-1]
        _sel = "**selected**" if _col in relevant_observables else ""
        _rows += f"| `{_short}` | {_residual[_i]*100:.1f}% | {_sel} |\n"
    if len(observable_columns) > 15:
        _rows += f"| ... | ({len(observable_columns) - 15} more) | |\n"

    mo.md(f"""
**Threshold:** {_threshold*100:.1f}% residual variance (median)

| Observable | Residual Fraction | Status |
|------------|-------------------|--------|
{_rows}

**{len(relevant_observables)}/{len(observable_columns)} observables selected** for Koopman cell cycle analysis.
    """)
    return (relevant_observables,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: STEPS 6-7a — PHASE 1 POPULATION GSA
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Phase 1: Population-Level GSA (Steps 6a-7a)

    Build PCE surrogate from LHS samples, then extract Sobol indices analytically
    from the polynomial coefficients.

    - **S_i** (first order): direct effect of each parameter
    - **S_Ti** (total order): direct + all interaction effects

    Uses `DataDrivenWrapper` (response surface from data statistics) unless a
    precomputed cache with real simulation evaluations is loaded.
    """)
    return


@app.cell
def _(
    DataDrivenWrapper,
    agg_result,
    cache,
    math,
    mo,
    n_samples_slider,
    np,
    param_space,
    polynomial_order_slider,
    prescreen_config,
    run_phase1,
):
    _n_terms = math.comb(
        polynomial_order_slider.value + param_space.n_parameters,
        param_space.n_parameters,
    )
    _n_samples = n_samples_slider.value

    _wrapper = DataDrivenWrapper(
        parameter_space=param_space,
        observable_means=agg_result.uniform.mean,
        observable_stds=agg_result.uniform.std,
    )

    _warning = ""
    if _n_terms > _n_samples:
        _warning = f"Warning: {_n_terms} PCE terms > {_n_samples} samples. Consider increasing N or reducing p."

    _using_cache = cache is not None
    sobol_bulk, surrogate_bulk, morris_phase1 = run_phase1(
        param_space=param_space,
        simulation_func=_wrapper if not _using_cache else None,
        polynomial_order=polynomial_order_slider.value,
        n_samples=_n_samples,
        prescreen_config=prescreen_config,
        precomputed_samples=cache.X if _using_cache else None,
        precomputed_outputs=cache.Y if _using_cache else None,
    )

    _top = sobol_bulk.select(n=min(5, param_space.n_parameters))
    _top_str = ", ".join(f"`{name}` ({val:.3f})" for name, val in _top)
    _source = "precomputed cache (real sim evaluations)" if _using_cache else "DataDrivenWrapper (response surface)"

    mo.md(f"""
**Phase 1 complete** — source: {_source}
- PCE terms: {_n_terms} | Samples: {_n_samples}
- Top parameters (total order): {_top_str}
{"" if not _warning else f"- {_warning}"}
    """)
    return morris_phase1, sobol_bulk, surrogate_bulk


@app.cell
def _(go, mo, np, param_space, sobol_bulk):
    _first = sobol_bulk.first_order
    _total = sobol_bulk.total_order
    if _first.ndim > 1:
        _first = np.mean(_first, axis=0)
    if _total.ndim > 1:
        _total = np.mean(_total, axis=0)

    _names = param_space.parameter_names

    sobol_pop_fig = go.Figure()
    sobol_pop_fig.add_trace(go.Bar(
        x=_names, y=_first,
        name="S_i (first order)",
        marker_color="cyan",
    ))
    sobol_pop_fig.add_trace(go.Bar(
        x=_names, y=_total,
        name="S_Ti (total order)",
        marker_color="magenta",
    ))
    sobol_pop_fig.update_layout(
        barmode="group",
        yaxis_title="Sobol Index",
        height=350,
        template="plotly_dark",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )

    mo.vstack([
        mo.md("### Population Sobol Indices"),
        sobol_pop_fig,
    ])
    return (sobol_pop_fig,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: REACTIVE SURROGATE EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Reactive Surrogate Exploration

    Drag the sliders to see **instant PCE surrogate predictions**. The surrogate
    was trained on your real data statistics — predictions reflect the actual
    output scale and parameter sensitivities.
    """)
    return


@app.cell
def _(mo, param_space, surrogate_bulk):
    _slider_list = []
    for _i, _name in enumerate(param_space.parameter_names):
        _lo, _hi = param_space.parameter_bounds[_i]
        _mid = (_lo + _hi) / 2.0
        _step = (_hi - _lo) / 100.0
        _slider_list.append(
            mo.ui.slider(
                start=_lo, stop=_hi, step=max(_step, 0.001), value=_mid,
                label=_name, show_value=True,
            )
        )

    explore_sliders = mo.ui.array(_slider_list)
    return (explore_sliders,)


@app.cell
def _(explore_sliders, go, mo, np, observable_columns, param_space, surrogate_bulk):
    _params = np.array(explore_sliders.value)
    _prediction = surrogate_bulk.predict(_params.reshape(1, -1))
    _pred_arr = np.asarray(_prediction).flatten()

    _defaults = np.array([(_lo + _hi) / 2.0 for _lo, _hi in param_space.parameter_bounds])
    _default_pred = np.asarray(surrogate_bulk.predict(_defaults.reshape(1, -1))).flatten()

    _diff = _pred_arr - _default_pred
    _display_cols = observable_columns[:len(_pred_arr)]
    _short_names = [c.split("__")[-1] for c in _display_cols]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_short_names, y=_default_pred[:len(_short_names)],
        name="Baseline (midpoint params)",
        marker_color="gray", opacity=0.5,
    ))
    _fig.add_trace(go.Bar(
        x=_short_names, y=_pred_arr[:len(_short_names)],
        name="Current params",
        marker_color="cyan",
    ))
    _fig.update_layout(
        barmode="group", yaxis_title="Predicted Output",
        height=400, template="plotly_dark",
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )

    _param_summary = "\n".join(
        f"- `{param_space.parameter_names[_i]}`: {_params[_i]:.4f}"
        for _i in range(len(_params))
    )

    _info = mo.vstack([
        mo.md("### Parameters"),
        explore_sliders,
        mo.md("---"),
        mo.md(f"**Current values:**\n{_param_summary}"),
    ])

    mo.hstack([_info, _fig], widths=[1, 3])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: STEPS 6-7b — PHASE 2 CELL CYCLE GSA
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Phase 2: Cell-Cycle-Stratified GSA (Steps 6b-7b)

    **RFC006:** *"A second type of analysis... will involve the definition of a
    low-dimensional cell cycle variable. This variable will be used for
    deterministically binning simulation data into cell stages, in order to then
    perform a phenotypic sensitivity analysis."*
    """)
    return


@app.cell
def _(
    DataDrivenWrapper,
    SensitivityAnalyzer,
    agg_result,
    cache,
    cycle_time_slider,
    mo,
    n_bins_slider,
    n_samples_slider,
    np,
    observable_columns,
    param_space,
    polynomial_order_slider,
    run_phase2,
):
    _wrapper = DataDrivenWrapper(
        parameter_space=param_space,
        observable_means=agg_result.uniform.mean,
        observable_stds=agg_result.uniform.std,
    )

    _using_cache = cache is not None
    cc_relevance = None
    try:
        per_stage_sobol, surrogate_cc, cc_relevance = run_phase2(
            param_space=param_space,
            simulation_func=_wrapper if not _using_cache else None,
            agg_result=agg_result,
            observable_names=observable_columns,
            n_bins=n_bins_slider.value,
            polynomial_order=polynomial_order_slider.value,
            n_samples=n_samples_slider.value,
            expected_cycle_time=cycle_time_slider.value,
            precomputed_samples=cache.X if _using_cache else None,
            precomputed_timeseries=cache.Y_timeseries if _using_cache else None,
        )
        mo.md(f"""
**Phase 2 complete.**
- Cell cycle bins: {n_bins_slider.value}
- Per-stage Sobol sets: {len(per_stage_sobol)}
- Source: {"precomputed cache" if _using_cache else "DataDrivenWrapper"}
        """)
    except Exception as _e:
        _n_bins = n_bins_slider.value
        _analyzer = SensitivityAnalyzer(param_space, _wrapper)
        _sobol_single, surrogate_cc = _analyzer.analyze_with_pce(
            polynomial_order=polynomial_order_slider.value,
            n_samples=n_samples_slider.value,
        )
        per_stage_sobol = [_sobol_single] * _n_bins
        mo.md(f"""
**Phase 2 fallback** — Koopman cell cycle extraction failed: `{type(_e).__name__}: {_e}`

Using replicated population Sobol as proxy for {_n_bins} stages.
        """)
    return cc_relevance, per_stage_sobol, surrogate_cc


@app.cell
def _(go, mo, n_bins_slider, np, param_space, per_stage_sobol):
    _n_bins = min(n_bins_slider.value, len(per_stage_sobol))
    _n_params = param_space.n_parameters
    _names = param_space.parameter_names

    _z = np.zeros((_n_params, _n_bins))
    for _stage in range(_n_bins):
        _total = per_stage_sobol[_stage].total_order
        if _total.ndim > 1:
            _total = np.mean(_total, axis=0)
        _z[:, _stage] = _total[:_n_params]

    _bin_labels = [f"{_s/_n_bins:.2f}-{(_s+1)/_n_bins:.2f}" for _s in range(_n_bins)]

    cc_heatmap = go.Figure(go.Heatmap(
        z=_z,
        x=_bin_labels,
        y=_names,
        colorscale="Viridis",
        colorbar_title="S_Ti",
        text=np.round(_z, 3),
        texttemplate="%{text}",
    ))
    cc_heatmap.update_layout(
        xaxis_title="Cell Cycle Stage (theta)",
        yaxis_title="Parameter",
        height=max(250, 50 * _n_params),
        template="plotly_dark",
        margin=dict(t=20, b=40),
    )

    mo.vstack([
        mo.md("### Per-Stage Sobol Indices (Total Order)"),
        cc_heatmap,
    ])
    return (cc_heatmap,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: KOOPMAN SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Koopman Spectral Analysis

    DMD decomposes the simulation trajectory into spatial modes with associated
    frequencies and growth rates — a "spectral fingerprint" of cellular dynamics.
    Cell cycle modes appear near the expected cycle frequency.
    """)
    return


@app.cell
def _(
    DynamicModeDecomposition,
    cycle_time_slider,
    mo,
    np,
    observable_columns,
    plot_koopman_spectrum,
    sim_data,
):
    # Use mass-related columns for Koopman
    _mass_cols = [c for c in observable_columns if c in sim_data.columns and "mass" in c]
    if not _mass_cols:
        _mass_cols = [c for c in observable_columns[:3] if c in sim_data.columns]

    if not _mass_cols:
        mo.md("*No matching observable columns in data for Koopman analysis.*")
    else:
        _first_exp = sim_data["experiment_id"].unique()[0]
        _first_seed = sim_data.filter(
            sim_data["experiment_id"] == _first_exp
        )["lineage_seed"].unique()[0]

        _traj_df = sim_data.filter(
            (sim_data["experiment_id"] == _first_exp) &
            (sim_data["lineage_seed"] == _first_seed)
        ).sort("time")

        _traj_data = _traj_df.select(_mass_cols).to_numpy()

        if _traj_data.shape[0] < 10:
            mo.md(f"*Trajectory too short ({_traj_data.shape[0]} rows) for DMD.*")
        else:
            _rank = min(5, _traj_data.shape[1], _traj_data.shape[0] - 2)
            _dmd = DynamicModeDecomposition(rank=max(_rank, 1), dt=1.0)
            _dmd.fit(_traj_data)
            _spectrum = _dmd.get_spectrum(
                observable_names=[c.split("__")[-1] for c in _mass_cols],
            )

            _koopman_fig = plot_koopman_spectrum(
                spectrum=_spectrum,
                expected_cycle_time=cycle_time_slider.value,
                frequency_tolerance=0.3,
            )
            _koopman_fig.update_layout(height=600, template="plotly_dark")
            _koopman_fig
    return


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: RESULTS & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## Pipeline Result

    Assembles all outputs into a single `PipelineResult` for serialization.
    """)
    return


@app.cell
def _(
    PipelineResult,
    StratificationLens,
    UqProfile,
    agg_result,
    cc_relevance,
    decomp,
    mo,
    morris_indices,
    per_stage_sobol,
    sobol_bulk,
    surrogate_bulk,
    surrogate_cc,
):
    pipeline_result = PipelineResult(
        population=UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[sobol_bulk],
            surrogate=surrogate_bulk,
        ),
        cell_cycle=UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=per_stage_sobol,
            surrogate=surrogate_cc,
        ),
        variance_decomposition=decomp,
        aggregation=agg_result,
        morris_indices=morris_indices,
        cell_cycle_relevance=cc_relevance,
    )

    _top = sobol_bulk.select(n=min(3, len(sobol_bulk.parameter_names)))
    _top_str = ", ".join(f"`{n}` ({v:.3f})" for n, v in _top)

    mo.md(f"""
**PipelineResult assembled.**

| Field | Value |
|-------|-------|
| Population Sobol sets | {len(pipeline_result.population.sobol_indices)} |
| Cell Cycle stages | {len(pipeline_result.cell_cycle.sobol_indices)} |
| Variance decomposition | {len(decomp)} keys |
| Morris indices | {"Yes" if morris_indices else "No"} |
| CC relevance | {"Yes" if cc_relevance else "No"} |
| Top parameters (bulk) | {_top_str} |
    """)
    return (pipeline_result,)


@app.cell
def _(Path, export_path_input, mo, pipeline_result):
    export_button = mo.ui.run_button(label="Export PipelineResult")

    mo.vstack([
        mo.md(f"**Export to:** `{export_path_input.value}`"),
        export_button,
    ])

    if export_button.value:
        _path = Path(export_path_input.value)
        pipeline_result.export(_path)
        mo.md(f"""
**Exported to `{_path}`:**
- `population_sobol/`
- `cell_cycle_surrogate/`
- `variance_decomposition.json`
- `metadata.json`
        """)
    return (export_button,)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: RFC006 VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md("""
    ## RFC006 Verification

    | # | Activity | Module | Status |
    |---|----------|--------|--------|
    | 1 | Identify input/output variables | `uq.inputs`, `uq.outputs` | Done |
    | 2 | Enable output via emitter | ParquetEmitter + hive partitioning | Done |
    | 3 | Implement wrapper functions | `uq.wrappers` | Done |
    | 4 | PCE-based sensitivity (strategies 1-3) | `uq.sensitivity`, `uq.pce` | Done |
    | 5 | Apply to representative simulations | This dashboard | Done |
    | 6 | Cell cycle stratification strategy | `uq.cell_cycle`, `uq.koopman` | Done |
    | 7 | Cell cycle variable + per-stage GSA | `uq.pipeline.workflow` | Done |

    **RFC006 Section 4 parametrised steps:**

    | Step | Description | Implementation |
    |------|-------------|----------------|
    | A | Selection/extraction of variables | `OutputExtractor` + observable_columns |
    | B | Temporal aggregation into Y | `aggregate_timeseries()` — 4 strategies |
    | C | Sensitivity analysis method | `SensitivityAnalyzer` — PCE + Sobol via PyTUQ |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    **Documentation:** [RFC006](../readmes/RFC006.md) |
    [Extended Context](../readmes/CONTEXT.md) |
    [CLI Reference](../README.md)

    **CLI equivalent:**
    ```bash
    uv run uq generate-samples exp1 exp2 --sim-base-path /path/to/sims --cache-dir ./uq_cache --live --n-samples 50
    uv run uq quantify exp1 exp2 --sim-base-path /path/to/sims --precomputed-path ./uq_cache --export-path ./uq_results
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
