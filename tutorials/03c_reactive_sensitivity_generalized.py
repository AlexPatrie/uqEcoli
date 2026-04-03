"""Tutorial 3c: Generalized Reactive Parameter Exploration

This tutorial provides a GENERALIZED version of the reactive sensitivity demo.
Users can define their own parameters, bounds, and timeseries generation logic.

Run with: uv run marimo run tutorials/03c_reactive_sensitivity_generalized.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Generalized Reactive Parameter Exploration

    This notebook allows you to explore **any parameter space** with reactive visualization.

    ## How to Customize

    1. **Edit the `PARAMETER_CONFIG`** cell below to define your own parameters
    2. **Edit the `timeseries_generator`** function to define how parameters affect output
    3. **Run the notebook** and drag sliders to explore!

    The PCE surrogate and sliders are automatically generated from your configuration.
    """)
    return


@app.cell
def _():
    import numpy as np
    from dataclasses import dataclass, asdict
    from typing import Callable, Optional
    from itertools import combinations_with_replacement

    return Callable, combinations_with_replacement, np


@app.cell
def _():
    """
    =========================================================================
    PARAMETER CONFIGURATION - CUSTOMIZE THIS FOR YOUR USE CASE
    =========================================================================

    Define your parameters as a list of dictionaries with:
        - name: str           - Parameter name (used in labels)
        - bounds: [min, max]  - Parameter range
        - default: float      - Default/baseline value (optional, defaults to midpoint)
        - step: float         - Slider step size (optional, auto-calculated if omitted)
        - description: str    - What this parameter does (optional, for documentation)

    Example configurations are provided below. Uncomment/modify as needed.
    """

    # -------------------------------------------------------------------------
    # EXAMPLE 1: Default vEcoli-like parameters (3 params)
    # -------------------------------------------------------------------------
    PARAMETER_CONFIG = [
        {
            "name": "expression_factor",
            "bounds": [0.5, 5.0],
            "default": 2.75,
            "step": 0.1,
            "description": "Gene expression multiplier (1.0 = baseline)",
        },
        {
            "name": "translation_efficiency",
            "bounds": [0.5, 2.0],
            "default": 1.25,
            "step": 0.05,
            "description": "Translation efficiency factor",
        },
        {
            "name": "inhibitor_conc",
            "bounds": [0.0, 10.0],
            "default": 5.0,
            "step": 0.5,
            "description": "Inhibitor concentration (reduces output)",
        },
    ]

    # -------------------------------------------------------------------------
    # EXAMPLE 2: Simple 2-parameter system (uncomment to use)
    # -------------------------------------------------------------------------
    # PARAMETER_CONFIG = [
    #     {
    #         "name": "amplitude",
    #         "bounds": [0.1, 10.0],
    #         "default": 5.0,
    #         "description": "Signal amplitude",
    #     },
    #     {
    #         "name": "frequency",
    #         "bounds": [0.001, 0.1],
    #         "default": 0.01,
    #         "description": "Oscillation frequency (Hz)",
    #     },
    # ]

    # -------------------------------------------------------------------------
    # EXAMPLE 3: 5-parameter complex system (uncomment to use)
    # -------------------------------------------------------------------------
    # PARAMETER_CONFIG = [
    #     {"name": "param_a", "bounds": [0, 1], "description": "First parameter"},
    #     {"name": "param_b", "bounds": [0, 1], "description": "Second parameter"},
    #     {"name": "param_c", "bounds": [-1, 1], "description": "Third parameter"},
    #     {"name": "param_d", "bounds": [0, 10], "description": "Fourth parameter"},
    #     {"name": "param_e", "bounds": [0.1, 2], "description": "Fifth parameter"},
    # ]

    # -------------------------------------------------------------------------
    # PCE Configuration
    # -------------------------------------------------------------------------
    PCE_ORDER = 2  # Polynomial order for the surrogate (1, 2, or 3 recommended)

    # -------------------------------------------------------------------------
    # Timeseries Configuration
    # -------------------------------------------------------------------------
    TIMESERIES_LENGTH = 1000  # Number of timesteps
    RANDOM_SEED = 42  # For reproducible noise
    return PARAMETER_CONFIG, PCE_ORDER, RANDOM_SEED, TIMESERIES_LENGTH


@app.cell
def _(Callable, np):
    """
    =========================================================================
    TIMESERIES GENERATOR - CUSTOMIZE THIS FOR YOUR USE CASE
    =========================================================================

    Define a function that takes:
        - params: np.ndarray of shape (n_params,) with current parameter values
        - param_names: list of parameter names
        - baseline_value: float, the PCE-predicted baseline
        - n_timesteps: int, number of timesteps to generate
        - random_seed: int, for reproducible noise

    Returns:
        - np.ndarray of shape (n_timesteps,) with the timeseries

    The default implementation below provides a generic growth + oscillation model.
    Replace it with your own physics/biology if needed.
    """

    def default_timeseries_generator(
        params: np.ndarray,
        param_names: list[str],
        baseline_value: float,
        n_timesteps: int,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Default timeseries generator: exponential growth + damped oscillation + noise.

        Parameter effects (generic interpretation):
        - First parameter: affects growth rate (higher = faster growth)
        - Second parameter: affects oscillation frequency (if present)
        - Third+ parameters: damping effects (reduce growth)

        Customize this function for your specific use case!
        """
        t = np.arange(n_timesteps)
        n_params = len(params)

        # Normalize parameters to [0, 1] for generic effects
        # (In practice, you'd use domain-specific logic)

        # Growth rate: first param drives growth
        growth_rate = 0.002 * params[0] if n_params > 0 else 0.002

        # Oscillation frequency: second param affects frequency
        osc_freq = 0.005 + 0.01 * params[1] if n_params > 1 else 0.01

        # Damping: remaining params contribute to damping
        damping = 0.0
        if n_params > 2:
            for i in range(2, n_params):
                damping += 0.0005 * params[i]

        # Build timeseries components
        trend = baseline_value * np.exp((growth_rate - damping) * t)
        oscillation = 0.15 * baseline_value * np.sin(2 * np.pi * osc_freq * t)
        oscillation *= np.exp(-0.001 * t)  # Damped oscillation

        # Add reproducible noise
        np.random.seed(random_seed)
        noise = 0.03 * baseline_value * np.random.randn(n_timesteps)

        return trend + oscillation + noise

    # You can define alternative generators and assign them here
    # Example: a pure sinusoidal generator
    def sinusoidal_generator(
        params: np.ndarray,
        param_names: list[str],
        baseline_value: float,
        n_timesteps: int,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Simple sinusoidal timeseries: amplitude * sin(2*pi*freq*t)"""
        t = np.arange(n_timesteps)
        amplitude = params[0] if len(params) > 0 else 1.0
        frequency = params[1] if len(params) > 1 else 0.01
        np.random.seed(random_seed)
        noise = 0.05 * amplitude * np.random.randn(n_timesteps)
        return baseline_value + amplitude * np.sin(2 * np.pi * frequency * t) + noise

    # =========================================================================
    # SELECT WHICH GENERATOR TO USE (change this to switch generators)
    # =========================================================================
    TIMESERIES_GENERATOR: Callable = default_timeseries_generator
    # TIMESERIES_GENERATOR: Callable = sinusoidal_generator
    return (TIMESERIES_GENERATOR,)


@app.cell
def _(combinations_with_replacement, np):
    def generate_multi_indices(n_params: int, max_order: int) -> np.ndarray:
        """
        Generate multi-indices for PCE basis up to given order.

        Returns array where each row is [order_x1, order_x2, ...] and
        sum of each row <= max_order.
        """
        indices = []
        for total_order in range(max_order + 1):
            for combo in combinations_with_replacement(range(n_params), total_order):
                idx = [0] * n_params
                for i in combo:
                    idx[i] += 1
                if idx not in indices:
                    indices.append(idx)
        return np.array(indices)

    def generate_synthetic_coefficients(
        multi_indices: np.ndarray,
        seed: int = 123,
    ) -> np.ndarray:
        """
        Generate synthetic PCE coefficients with realistic structure.

        - Constant term: ~1.5 (baseline)
        - Linear terms: larger coefficients (main effects)
        - Higher-order terms: smaller coefficients
        - Some negative coefficients for variety
        """
        np.random.seed(seed)
        n_terms = len(multi_indices)
        coefficients = np.zeros(n_terms)

        for i, idx in enumerate(multi_indices):
            total_order = sum(idx)
            if total_order == 0:
                # Constant term
                coefficients[i] = 1.5
            elif total_order == 1:
                # Linear terms: larger, some positive, some negative
                coefficients[i] = np.random.uniform(0.2, 0.8) * np.random.choice([1, -1], p=[0.7, 0.3])
            else:
                # Higher-order terms: smaller
                coefficients[i] = np.random.uniform(0.05, 0.2) / total_order * np.random.choice([1, -1])

        return coefficients

    def compute_slider_step(bounds: list[float], n_steps: int = 50) -> float:
        """Compute a reasonable step size for a slider given bounds."""
        range_size = bounds[1] - bounds[0]
        step = range_size / n_steps
        # Round to nice values
        magnitude = 10 ** np.floor(np.log10(step))
        normalized = step / magnitude
        if normalized < 1.5:
            nice_step = 1
        elif normalized < 3.5:
            nice_step = 2
        elif normalized < 7.5:
            nice_step = 5
        else:
            nice_step = 10
        return nice_step * magnitude

    return (
        compute_slider_step,
        generate_multi_indices,
        generate_synthetic_coefficients,
    )


@app.cell
def _(
    PARAMETER_CONFIG,
    PCE_ORDER,
    generate_multi_indices,
    generate_synthetic_coefficients,
    np,
):
    from libuq.sensitivity import PCESurrogate

    # Extract parameter info from config
    PARAM_NAMES = [p["name"] for p in PARAMETER_CONFIG]
    PARAM_BOUNDS = np.array([p["bounds"] for p in PARAMETER_CONFIG])
    PARAM_DEFAULTS = np.array([p.get("default", (p["bounds"][0] + p["bounds"][1]) / 2) for p in PARAMETER_CONFIG])
    N_PARAMS = len(PARAMETER_CONFIG)

    # Generate PCE basis and coefficients
    multi_indices = generate_multi_indices(N_PARAMS, PCE_ORDER)
    coefficients = generate_synthetic_coefficients(multi_indices)

    # Create surrogate
    pce_surrogate = PCESurrogate(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type="legendre",
        polynomial_order=PCE_ORDER,
        input_dim=N_PARAMS,
        output_dim=1,
        r_squared=0.95,
        input_bounds=PARAM_BOUNDS,
    )

    print(f"PCE Surrogate created from config:")
    print(f"  - Parameters: {PARAM_NAMES}")
    print(f"  - Bounds: {PARAM_BOUNDS.tolist()}")
    print(f"  - Defaults: {PARAM_DEFAULTS.tolist()}")
    print(f"  - Polynomial order: {PCE_ORDER}")
    print(f"  - Number of PCE terms: {len(coefficients)}")
    return PARAM_BOUNDS, PARAM_DEFAULTS, PARAM_NAMES, pce_surrogate


@app.cell
def _(
    PARAMETER_CONFIG,
    PARAM_BOUNDS,
    PARAM_DEFAULTS,
    PARAM_NAMES,
    compute_slider_step,
    mo,
):
    # Create sliders dynamically using mo.ui.array for reactivity
    # (per CONTEXT.md best practices)

    _slider_list = []
    for _i, _cfg in enumerate(PARAMETER_CONFIG):
        _step = _cfg.get("step", compute_slider_step(_cfg["bounds"]))
        _slider = mo.ui.slider(
            start=PARAM_BOUNDS[_i, 0],
            stop=PARAM_BOUNDS[_i, 1],
            step=_step,
            value=PARAM_DEFAULTS[_i],
            label=PARAM_NAMES[_i],
            show_value=True,
        )
        _slider_list.append(_slider)

    # Wrap in mo.ui.array for proper reactivity
    param_sliders = mo.ui.array(_slider_list)
    return (param_sliders,)


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(
    PARAMETER_CONFIG,
    PARAM_DEFAULTS,
    PARAM_NAMES,
    RANDOM_SEED,
    TIMESERIES_GENERATOR: "Callable",
    TIMESERIES_LENGTH,
    go,
    make_subplots,
    mo,
    np,
    param_sliders,
    pce_surrogate,
):
    # =========================================================================
    # REACTIVE OUTPUT TIMESERIES
    # Updates instantly when any slider changes
    # =========================================================================

    # Get current parameter values from sliders
    current_params = np.array(param_sliders.value)

    # Predict using the PCE surrogate
    _prediction = pce_surrogate.predict(current_params)
    _pred_mean, _pred_std = pce_surrogate.predict_with_uncertainty(current_params.reshape(1, -1))
    _pred_val = float(_prediction.flat[0])
    _std_val = float(_pred_std.flat[0])

    # Generate timeseries using the configured generator
    _t = np.arange(TIMESERIES_LENGTH)
    _timeseries = TIMESERIES_GENERATOR(
        params=current_params,
        param_names=PARAM_NAMES,
        baseline_value=_pred_val,
        n_timesteps=TIMESERIES_LENGTH,
        random_seed=RANDOM_SEED,
    )

    # Generate baseline timeseries (at default params)
    _default_pred = float(pce_surrogate.predict(PARAM_DEFAULTS).flat[0])
    _default_ts = TIMESERIES_GENERATOR(
        params=PARAM_DEFAULTS,
        param_names=PARAM_NAMES,
        baseline_value=_default_pred,
        n_timesteps=TIMESERIES_LENGTH,
        random_seed=RANDOM_SEED,
    )

    # =========================================================================
    # BUILD FIGURE
    # =========================================================================
    ts_fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            "Output Timeseries (drag sliders to see changes)",
            "Difference from Baseline",
        ],
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

    ts_fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)

    ts_fig.update_xaxes(title="Time (steps)", row=2, col=1)
    ts_fig.update_yaxes(title="Observable Value", row=1, col=1)
    ts_fig.update_yaxes(title="Diff from baseline", row=2, col=1)

    ts_fig.update_layout(
        height=550,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(t=40, b=40),
    )

    # =========================================================================
    # BUILD PARAMETER INFO PANEL
    # =========================================================================
    _param_lines = "\n".join([f"- **{PARAM_NAMES[_i]}**: `{current_params[_i]:.4f}`" for _i in range(len(PARAM_NAMES))])

    _desc_lines = ""
    for _cfg in PARAMETER_CONFIG:
        if "description" in _cfg:
            _desc_lines += f"- *{_cfg['name']}*: {_cfg['description']}\n"

    _slider_panel = mo.vstack([
        mo.md("### Parameters"),
        mo.md("*Drag sliders to see effect on output*"),
        param_sliders,
        mo.md("---"),
        mo.md(
            f"""
    **Current values:**
    {_param_lines}

    **PCE Prediction:** `{_pred_val:.4f}` +/- {_std_val:.4f}

    **Timeseries stats:**
    - Final value: `{_timeseries[-1]:.4f}`
    - Mean: `{np.mean(_timeseries):.4f}`
    - Std: `{np.std(_timeseries):.4f}`
    """
        ),
    ])

    mo.hstack([_slider_panel, ts_fig], widths=[1, 3], gap=2)
    return


@app.cell
def _(PARAMETER_CONFIG, mo):
    _desc_rows = []
    for _cfg in PARAMETER_CONFIG:
        _name = _cfg["name"]
        _bounds = _cfg["bounds"]
        _default = _cfg.get("default", (_bounds[0] + _bounds[1]) / 2)
        _desc = _cfg.get("description", "No description provided")
        _desc_rows.append(f"| `{_name}` | [{_bounds[0]}, {_bounds[1]}] | {_default} | {_desc} |")

    _table = """
    | Parameter | Bounds | Default | Description |
    |-----------|--------|---------|-------------|
    """ + "\n".join(_desc_rows)

    mo.md(f"""
    ## Parameter Reference

    {_table}

    *Edit the `PARAMETER_CONFIG` cell above to customize these parameters.*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## How to Customize This Notebook

    ### 1. Define Your Parameters

    Edit the `PARAMETER_CONFIG` list to define your own parameters:

    ```python
    PARAMETER_CONFIG = [
        {
            "name": "my_param",
            "bounds": [0.0, 100.0],
            "default": 50.0,  # optional
            "step": 1.0,      # optional
            "description": "What this parameter controls",
        },
        # Add more parameters...
    ]
    ```

    ### 2. Define Timeseries Generation

    Edit the `TIMESERIES_GENERATOR` function to define how parameters affect the output:

    ```python
    def my_custom_generator(params, param_names, baseline_value, n_timesteps, random_seed):
        t = np.arange(n_timesteps)
        # Your custom timeseries logic here
        # params[0], params[1], etc. are the current slider values
        return my_timeseries
    ```

    ### 3. Advanced: Load Real Data

    Instead of synthetic generation, you could:
    - Load a pre-trained PCE surrogate from file
    - Load actual simulation timeseries and interpolate
    - Connect to a live simulation API

    ## Use Cases

    - **Parameter sensitivity exploration** for any dynamical system
    - **"What if" scenario analysis** without running full simulations
    - **Teaching tool** for demonstrating parameter effects
    - **Rapid prototyping** of control strategies
    """)
    return


if __name__ == "__main__":
    app.run()
