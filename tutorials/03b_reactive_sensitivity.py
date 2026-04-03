"""Tutorial 3b: Reactive Parameter Exploration with PCE Surrogates

This tutorial demonstrates real-time parameter exploration using trained
PCE surrogate models with Marimo's reactivity.

Run with: marimo run 03b_reactive_sensitivity.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(
    width="full",
    layout_file="layouts/03b_reactive_sensitivity.grid.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Reactive Parameter Exploration with PCE Surrogates

    This notebook demonstrates **real-time parameter → output visualization**.

    ## How It Works

    1. A **PCE surrogate** predicts output values from input parameters (instantly!)
    2. We generate a **synthetic timeseries** whose dynamics depend on the parameters
    3. **Drag any slider** → the output timeseries updates immediately

    ## Parameter Effects

    | Parameter | Effect on Timeseries |
    |-----------|---------------------|
    | `vio_expression` | Increases baseline level and growth rate |
    | `vio_trl_eff` | Changes oscillation frequency (metabolic cycles) |
    | `mecillinam_conc` | Dampens growth (antibiotic effect) |

    **Try it:** Drag the sliders and watch the output timeseries change!
    """)
    return


@app.cell
def _():
    import numpy as np
    from dataclasses import dataclass

    return (np,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Building a PCE Surrogate

    First, we'll create a PCE surrogate from synthetic data. In practice,
    you would train this on actual simulation outputs.
    """)
    return


@app.cell
def _(np):
    # Import the PCE surrogate (with our new predict method!)
    from libuq.sensitivity import PCESurrogate

    # Define parameter space bounds
    PARAM_NAMES = ["vio_expression", "vio_trl_eff", "mecillinam_conc"]
    PARAM_BOUNDS = np.array([
        [0.5, 5.0],  # vio_expression
        [0.5, 2.0],  # vio_trl_eff
        [0.0, 10.0],  # mecillinam_concentration
    ])

    # Create a PCE surrogate with known coefficients
    # This represents: Y = 1.5 + 0.8*P1(x1) + 0.3*P1(x2) + 0.1*P1(x3) + 0.2*P2(x1) + ...
    # Where P_n are Legendre polynomials and x_i are normalized inputs

    # Multi-indices: each row is [order_x1, order_x2, order_x3]
    multi_indices = np.array([
        [0, 0, 0],  # Constant term
        [1, 0, 0],  # Linear in x1 (vio_expression)
        [0, 1, 0],  # Linear in x2 (vio_trl_eff)
        [0, 0, 1],  # Linear in x3 (mecillinam_conc)
        [2, 0, 0],  # Quadratic in x1
        [0, 2, 0],  # Quadratic in x2
        [1, 1, 0],  # Interaction x1*x2
        [1, 0, 1],  # Interaction x1*x3
    ])

    # Coefficients chosen to show clear parameter effects
    # Higher coefficient = more influence on output
    coefficients = np.array([
        1.5,  # Constant (baseline output)
        0.8,  # x1 linear (strong effect)
        0.3,  # x2 linear (medium effect)
        -0.15,  # x3 linear (negative effect - mecillinam reduces output)
        0.2,  # x1 quadratic (nonlinear)
        0.1,  # x2 quadratic
        0.15,  # x1*x2 interaction
        -0.1,  # x1*x3 interaction (mecillinam dampens vio effect)
    ])

    # Create surrogate
    pce_surrogate = PCESurrogate(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type="legendre",
        polynomial_order=2,
        input_dim=3,
        output_dim=1,
        r_squared=0.95,
        input_bounds=PARAM_BOUNDS,
    )

    print(f"PCE Surrogate created:")
    print(f"  - Input parameters: {PARAM_NAMES}")
    print(f"  - Polynomial order: {pce_surrogate.polynomial_order}")
    print(f"  - Number of terms: {len(coefficients)}")
    print(f"  - R-squared: {pce_surrogate.r_squared}")
    return PARAM_BOUNDS, PARAM_NAMES, pce_surrogate


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(PARAM_BOUNDS, PARAM_NAMES, mo):
    # Create sliders for each parameter
    vio_expr_slider = mo.ui.slider(
        start=PARAM_BOUNDS[0, 0],
        stop=PARAM_BOUNDS[0, 1],
        step=0.1,
        value=2.75,  # Start at midpoint
        label=f"{PARAM_NAMES[0]}",
        show_value=True,
    )

    vio_trl_slider = mo.ui.slider(
        start=PARAM_BOUNDS[1, 0],
        stop=PARAM_BOUNDS[1, 1],
        step=0.05,
        value=1.25,
        label=f"{PARAM_NAMES[1]}",
        show_value=True,
    )

    mecillinam_slider = mo.ui.slider(
        start=PARAM_BOUNDS[2, 0],
        stop=PARAM_BOUNDS[2, 1],
        step=0.5,
        value=5.0,
        label=f"{PARAM_NAMES[2]}",
        show_value=True,
    )
    return mecillinam_slider, vio_expr_slider, vio_trl_slider


@app.cell
def _(
    go,
    make_subplots,
    mecillinam_slider,
    mo,
    np,
    pce_surrogate,
    vio_expr_slider,
    vio_trl_slider,
):
    # =====================================================================
    # REACTIVE OUTPUT TIMESERIES
    # The timeseries updates instantly when you adjust parameter sliders
    # =====================================================================

    # Get current parameter values from sliders
    current_params = np.array([
        vio_expr_slider.value,
        vio_trl_slider.value,
        mecillinam_slider.value,
    ])

    # Predict using the PCE surrogate
    _prediction = pce_surrogate.predict(current_params)
    _pred_mean, _pred_std = pce_surrogate.predict_with_uncertainty(current_params.reshape(1, -1))
    _pred_val = float(_prediction.flat[0])
    _std_val = float(_pred_std.flat[0])

    # =====================================================================
    # GENERATE OUTPUT TIMESERIES based on current parameters
    # This simulates what a vEcoli simulation might produce
    # =====================================================================
    _T = 1000  # timesteps (e.g., seconds of simulation)
    _t = np.arange(_T)

    # Parameter effects on timeseries dynamics:
    # - vio_expression: affects baseline level and growth rate
    # - vio_trl_eff: affects oscillation frequency (metabolic cycles)
    # - mecillinam_conc: dampens growth (antibiotic effect)

    _baseline = _pred_val  # PCE prediction sets baseline
    _growth_rate = 0.002 * current_params[0]  # vio_expression drives growth
    _osc_freq = 0.005 + 0.01 * current_params[1]  # vio_trl_eff affects oscillations
    _damping = 0.001 * current_params[2]  # mecillinam dampens growth

    # Build the timeseries
    _trend = _baseline * np.exp((_growth_rate - _damping) * _t)
    _oscillation = 0.15 * _baseline * np.sin(2 * np.pi * _osc_freq * _t)
    np.random.seed(42)  # Deterministic for smooth updates
    _noise = 0.03 * _baseline * np.random.randn(_T)

    _timeseries = _trend + _oscillation * np.exp(-0.001 * _t) + _noise

    # Also generate a "baseline" timeseries (all params at default)
    _default_params = np.array([2.75, 1.25, 5.0])
    _default_pred = float(pce_surrogate.predict(_default_params).flat[0])
    _default_growth = 0.002 * _default_params[0]
    _default_osc = 0.005 + 0.01 * _default_params[1]
    _default_damp = 0.001 * _default_params[2]
    _default_trend = _default_pred * np.exp((_default_growth - _default_damp) * _t)
    _default_osc_signal = 0.15 * _default_pred * np.sin(2 * np.pi * _default_osc * _t)
    _default_ts = _default_trend + _default_osc_signal * np.exp(-0.001 * _t) + _noise

    # =====================================================================
    # BUILD FIGURE
    # =====================================================================
    ts_fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Output Timeseries (drag sliders to see changes)", "Difference from Baseline"],
        vertical_spacing=0.12,
    )

    # Main timeseries plot
    ts_fig.add_trace(
        go.Scatter(
            x=_t,
            y=_default_ts,
            name="Baseline (default params)",
            line=dict(color="rgba(100, 100, 100, 0.5)", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    ts_fig.add_trace(
        go.Scatter(
            x=_t,
            y=_timeseries,
            name="Current (your params)",
            line=dict(color="cyan", width=2),
            fill="tonexty",
            fillcolor="rgba(0, 255, 255, 0.1)",
        ),
        row=1,
        col=1,
    )

    # Difference plot
    _diff = _timeseries - _default_ts
    ts_fig.add_trace(
        go.Scatter(
            x=_t,
            y=_diff,
            name="Difference",
            line=dict(color="magenta", width=1),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 255, 0.2)",
        ),
        row=2,
        col=1,
    )

    # Add zero line to difference plot
    ts_fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)

    ts_fig.update_xaxes(title="Time (s)", row=2, col=1)
    ts_fig.update_yaxes(title="Observable Value", row=1, col=1)
    ts_fig.update_yaxes(title="Diff from baseline", row=2, col=1)

    ts_fig.update_layout(
        height=550,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(t=40, b=40),
    )

    # =====================================================================
    # LAYOUT: Sliders on left, timeseries on right
    # =====================================================================
    _slider_panel = mo.vstack([
        mo.md("### Parameters"),
        mo.md("*Drag sliders to see effect on output*"),
        vio_expr_slider,
        vio_trl_slider,
        mecillinam_slider,
        mo.md("---"),
        mo.md(f"""
**Current values:**
- vio_expression: `{current_params[0]:.2f}`
- vio_trl_eff: `{current_params[1]:.2f}`
- mecillinam_conc: `{current_params[2]:.2f}`

**PCE Prediction:** `{_pred_val:.4f}` ± {_std_val:.4f}

**Timeseries stats:**
- Final value: `{_timeseries[-1]:.4f}`
- Net growth: `{(_growth_rate - _damping) * 1000:.2f}‰/step`
        """),
    ])

    mo.hstack(
        [
            _slider_panel,
            ts_fig,
        ],
        widths=[1, 3],
        gap=2,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    This tutorial demonstrated:

    1. **PCE Surrogates** can be evaluated instantly (polynomial math)
    2. **Marimo's reactivity** triggers updates when sliders change
    3. **Real-time exploration** of parameter space without running simulations
    4. **Visualization** of individual parameter effects and interactions
    5. **Timeseries generation** that responds to parameter changes

    ## Next Steps

    In practice, you would:
    1. Train a PCE surrogate on actual vEcoli simulation outputs
    2. Use this reactive interface to explore "what if" scenarios
    3. Identify optimal parameter combinations for desired outputs
    4. Use Sobol indices to rank parameter importance

    Continue to **Tutorial 4** for cell cycle analysis and Koopman spectral methods.
    """)
    return


if __name__ == "__main__":
    app.run()
