"""Tutorial 3b: Reactive Parameter Exploration with PCE Surrogates

This tutorial demonstrates real-time parameter exploration using trained
PCE surrogate models with Marimo's reactivity.

Run with: marimo run 03b_reactive_sensitivity.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Reactive Parameter Exploration with PCE Surrogates

    This notebook demonstrates **real-time parameter sensitivity exploration**.

    ## The Key Insight

    Once we have a trained **PCE (Polynomial Chaos Expansion) surrogate**, we can:
    1. Evaluate it **instantly** (it's just polynomial math)
    2. Use **Marimo's reactivity** to update outputs as sliders change
    3. **Explore parameter space interactively** without running simulations

    ## What You'll See

    - Sliders for each input parameter
    - Real-time output prediction as you adjust parameters
    - Visualization of parameter effects on model outputs
    """)
    return


@app.cell
def _():
    import numpy as np
    from dataclasses import dataclass

    return dataclass, np


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
    from uq.sensitivity import PCESurrogate

    # Define parameter space bounds
    PARAM_NAMES = ["vio_expression", "vio_trl_eff", "mecillinam_conc"]
    PARAM_BOUNDS = np.array([
        [0.5, 5.0],   # vio_expression
        [0.5, 2.0],   # vio_trl_eff
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
        1.5,   # Constant (baseline output)
        0.8,   # x1 linear (strong effect)
        0.3,   # x2 linear (medium effect)
        -0.15, # x3 linear (negative effect - mecillinam reduces output)
        0.2,   # x1 quadratic (nonlinear)
        0.1,   # x2 quadratic
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

    return PARAM_BOUNDS, PARAM_NAMES, PCESurrogate, coefficients, multi_indices, pce_surrogate


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
def _(PARAM_BOUNDS, PARAM_NAMES, go, make_subplots, mecillinam_slider, mo, np, pce_surrogate, vio_expr_slider, vio_trl_slider):
    # =====================================================================
    # UNIFIED REACTIVE VISUALIZATION
    # All plots update instantly when sliders change
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
    # 1. SENSITIVITY CURVES (1D parameter sweeps)
    # =====================================================================
    _n_points = 50
    sensitivity_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Effect of {n}" for n in PARAM_NAMES],
        horizontal_spacing=0.08,
    )

    for _idx, (_param_name, _bounds) in enumerate(zip(PARAM_NAMES, PARAM_BOUNDS)):
        _sweep_values = np.linspace(_bounds[0], _bounds[1], _n_points)
        _predictions = [pce_surrogate.predict(
            np.array([_sweep_values[_k] if _idx == 0 else current_params[0],
                      _sweep_values[_k] if _idx == 1 else current_params[1],
                      _sweep_values[_k] if _idx == 2 else current_params[2]])
        ).flat[0] for _k in range(_n_points)]

        sensitivity_fig.add_trace(
            go.Scatter(x=_sweep_values, y=_predictions, mode='lines',
                      name=_param_name, line=dict(width=2)),
            row=1, col=_idx+1
        )
        sensitivity_fig.add_trace(
            go.Scatter(x=[current_params[_idx]], y=[_pred_val],
                      mode='markers', marker=dict(size=12, color='red', symbol='diamond'),
                      showlegend=False),
            row=1, col=_idx+1
        )
        sensitivity_fig.update_xaxes(title_text=_param_name, row=1, col=_idx+1)

    sensitivity_fig.update_layout(height=280, template='plotly_dark', showlegend=False,
                                   margin=dict(t=40, b=40))

    # =====================================================================
    # 2. RESPONSE SURFACE (2D interaction)
    # =====================================================================
    _n_grid = 25
    _x1_range = np.linspace(PARAM_BOUNDS[0, 0], PARAM_BOUNDS[0, 1], _n_grid)
    _x2_range = np.linspace(PARAM_BOUNDS[1, 0], PARAM_BOUNDS[1, 1], _n_grid)
    _X1, _X2 = np.meshgrid(_x1_range, _x2_range)
    _Z = np.array([[pce_surrogate.predict(np.array([_X1[_i, _j], _X2[_i, _j], current_params[2]])).flat[0]
                   for _j in range(_n_grid)] for _i in range(_n_grid)])

    surface_fig = go.Figure()
    surface_fig.add_trace(go.Surface(x=_X1, y=_X2, z=_Z, colorscale='Viridis', showscale=False))
    surface_fig.add_trace(go.Scatter3d(
        x=[current_params[0]], y=[current_params[1]], z=[_pred_val],
        mode='markers', marker=dict(size=6, color='red', symbol='diamond'),
        name='Current'
    ))
    surface_fig.update_layout(
        height=350, template='plotly_dark', margin=dict(t=30, b=10, l=10, r=10),
        scene=dict(xaxis_title='vio_expr', yaxis_title='vio_trl', zaxis_title='Output'),
    )

    # =====================================================================
    # 3. SYNTHETIC TIMESERIES
    # =====================================================================
    _T = 500
    _t = np.arange(_T)
    _base_output = _pred_val
    _growth_rate = 0.001 * current_params[0]
    _oscillation_freq = 0.01 * current_params[1]
    _damping = 0.0005 * current_params[2]
    _growth = np.exp((_growth_rate - _damping) * _t)
    _oscillation = 0.1 * np.sin(2 * np.pi * _oscillation_freq * _t) * np.exp(-0.001 * _t)
    np.random.seed(42)  # Deterministic noise for smoother updates
    _noise = 0.02 * np.random.randn(_T) * _growth
    _timeseries = _base_output * _growth + _oscillation + _noise

    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=_t, y=_timeseries, mode='lines',
                                line=dict(color='cyan', width=1)))
    ts_fig.update_layout(height=200, template='plotly_dark', margin=dict(t=30, b=40),
                         xaxis_title='Time', yaxis_title='Value',
                         title='Synthetic Timeseries')

    # =====================================================================
    # UNIFIED LAYOUT
    # =====================================================================
    _slider_panel = mo.vstack([
        mo.md("### Parameters"),
        vio_expr_slider,
        vio_trl_slider,
        mecillinam_slider,
        mo.md(f"""
**Prediction:** `{_pred_val:.4f}` ± {_std_val:.4f}
        """),
    ])

    _plots_panel = mo.vstack([
        sensitivity_fig,
        mo.hstack([surface_fig, ts_fig], justify="space-around"),
    ])

    mo.hstack([
        _slider_panel,
        _plots_panel,
    ], widths=[1, 4], gap=2)

    return current_params, sensitivity_fig, surface_fig, ts_fig


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
