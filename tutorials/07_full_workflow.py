"""Tutorial 07: Full Sensitivity Analysis Workflow

This tutorial demonstrates the complete workflow for sensitivity analysis:
1. Morris screening (cheap) to identify important parameters
2. Detailed PCE analysis on the screened subset
3. Reactive parameter exploration with sliders

This is the recommended approach for high-dimensional parameter spaces.

Run with: uv run marimo run tutorials/07_full_workflow.py
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
    # Full Sensitivity Analysis Workflow

    This tutorial demonstrates the **complete workflow** for sensitivity analysis
    on high-dimensional parameter spaces:

    ```
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Many Params    │ ──► │ Morris Screening│ ──► │  Top K Params   │
    │  (10-100+)      │     │  (O(n) cheap)   │     │  (3-10)         │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ Reactive Explore│ ◄── │  PCE Surrogate  │ ◄── │ Detailed PCE    │
    │ (sliders + viz) │     │  (instant pred) │     │ (Sobol indices) │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
    ```

    **Why this workflow?**
    - Morris screening is O(n) - feasible for 50+ parameters
    - PCE/Sobol is O(n²) - only practical for <20 parameters
    - Screening identifies which parameters to focus on
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


# =============================================================================
# STEP 1: DEFINE A HIGH-DIMENSIONAL PARAMETER SPACE
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Define High-Dimensional Parameter Space

    We'll simulate a system with **15 parameters** - too many for direct PCE
    analysis, but perfect for Morris screening.
    """)
    return


@app.cell
def _(np):
    # Simulate a high-dimensional parameter space
    # In practice, this would come from InputParameterSpace

    FULL_PARAM_NAMES = [
        "gene_expression_A",
        "gene_expression_B",
        "gene_expression_C",
        "translation_eff_A",
        "translation_eff_B",
        "translation_eff_C",
        "degradation_rate",
        "diffusion_coeff",
        "binding_affinity",
        "inhibitor_conc",
        "activator_conc",
        "temperature_factor",
        "pH_factor",
        "nutrient_level",
        "stress_response",
    ]

    FULL_PARAM_BOUNDS = [
        (0.1, 5.0),  # gene_expression_A (strong effect)
        (0.1, 5.0),  # gene_expression_B (medium effect)
        (0.1, 5.0),  # gene_expression_C (weak effect)
        (0.5, 2.0),  # translation_eff_A (strong effect)
        (0.5, 2.0),  # translation_eff_B (weak effect)
        (0.5, 2.0),  # translation_eff_C (negligible)
        (0.01, 0.5),  # degradation_rate (medium effect)
        (0.001, 0.1),  # diffusion_coeff (negligible)
        (0.1, 10.0),  # binding_affinity (medium effect)
        (0.0, 10.0),  # inhibitor_conc (strong negative effect)
        (0.0, 5.0),  # activator_conc (medium effect)
        (0.8, 1.2),  # temperature_factor (negligible)
        (0.9, 1.1),  # pH_factor (negligible)
        (0.1, 2.0),  # nutrient_level (weak effect)
        (0.0, 1.0),  # stress_response (weak effect)
    ]

    N_PARAMS = len(FULL_PARAM_NAMES)
    print(f"Total parameters: {N_PARAMS}")
    print(f"Direct PCE would need ~{(N_PARAMS + 1) * (N_PARAMS + 2) // 2} terms (order 2)")
    return FULL_PARAM_BOUNDS, FULL_PARAM_NAMES, N_PARAMS


# =============================================================================
# STEP 2: DEFINE A SYNTHETIC MODEL
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Define Model Function

    We define a synthetic model where we **know** which parameters are important.
    This lets us verify that Morris screening correctly identifies them.

    **Ground truth** (built into our model):
    - Strong effects: `gene_expression_A`, `translation_eff_A`, `inhibitor_conc`
    - Medium effects: `gene_expression_B`, `degradation_rate`, `binding_affinity`, `activator_conc`
    - Weak effects: `gene_expression_C`, `translation_eff_B`, `nutrient_level`, `stress_response`
    - Negligible: `translation_eff_C`, `diffusion_coeff`, `temperature_factor`, `pH_factor`
    """)
    return


@app.cell
def _(np):
    def synthetic_model(params: np.ndarray) -> np.ndarray:
        """
        Synthetic model with known parameter sensitivities.

        Returns a scalar output representing some biological observable.
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        outputs = []
        for p in params:
            # Unpack parameters
            expr_A, expr_B, expr_C = p[0], p[1], p[2]
            trl_A, trl_B, trl_C = p[3], p[4], p[5]
            deg_rate, diff_coeff, bind_aff = p[6], p[7], p[8]
            inhib, activ = p[9], p[10]
            temp, ph, nutrient, stress = p[11], p[12], p[13], p[14]

            # Model with known sensitivities
            # Strong effects
            y = 2.0 * expr_A * trl_A  # Strong positive
            y -= 0.3 * inhib  # Strong negative

            # Medium effects
            y += 0.5 * expr_B
            y -= 0.2 * deg_rate * 10  # Scale up
            y += 0.3 * np.log1p(bind_aff)
            y += 0.4 * activ

            # Weak effects
            y += 0.1 * expr_C
            y += 0.05 * trl_B
            y += 0.08 * nutrient
            y -= 0.05 * stress

            # Negligible effects (< 1% of output variance)
            y += 0.01 * trl_C
            y += 0.005 * diff_coeff * 100
            y += 0.01 * (temp - 1.0) * 10
            y += 0.01 * (ph - 1.0) * 10

            # Add small interaction term
            y += 0.1 * expr_A * inhib  # Nonlinear interaction

            outputs.append(y)

        return np.array(outputs)

    # Test it
    test_params = np.array([1.0] * 15)
    test_output = synthetic_model(test_params)
    print(f"Test output at all-ones params: {test_output[0]:.4f}")
    return (synthetic_model,)


# =============================================================================
# STEP 3: MORRIS SCREENING
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Morris Screening

    Now we run Morris screening to identify which parameters are most influential.
    This requires only **O(n_trajectories × (n_params + 1))** model evaluations.

    For 15 parameters with 20 trajectories: **320 evaluations**
    (vs ~2000+ for full PCE analysis)
    """)
    return


@app.cell
def _(FULL_PARAM_BOUNDS, FULL_PARAM_NAMES, np, synthetic_model):
    from uq.sensitivity import MorrisIndices

    def run_morris_screening(
        model_func,
        param_names: list[str],
        param_bounds: list[tuple[float, float]],
        n_trajectories: int = 20,
        n_levels: int = 4,
        seed: int = 42,
    ) -> MorrisIndices:
        """
        Run Morris screening on a model function.

        This is a simplified implementation for demonstration.
        In practice, use SensitivityAnalyzer.analyze_with_morris().
        """
        np.random.seed(seed)
        n_params = len(param_names)
        bounds = np.array(param_bounds)
        lb, ub = bounds[:, 0], bounds[:, 1]

        # Grid step size
        delta = n_levels / (2 * (n_levels - 1))

        elementary_effects = np.zeros((n_trajectories, n_params))

        for traj in range(n_trajectories):
            # Random starting point on grid
            x_base = np.random.randint(0, n_levels - 1, n_params) / (n_levels - 1)

            # Random permutation of parameters
            perm = np.random.permutation(n_params)

            # Build trajectory
            trajectory = np.zeros((n_params + 1, n_params))
            trajectory[0] = x_base.copy()

            for i, param_idx in enumerate(perm):
                trajectory[i + 1] = trajectory[i].copy()
                if trajectory[i, param_idx] + delta <= 1.0:
                    trajectory[i + 1, param_idx] += delta
                else:
                    trajectory[i + 1, param_idx] -= delta

            # Scale to actual bounds
            trajectory_scaled = lb + trajectory * (ub - lb)

            # Evaluate model
            outputs = model_func(trajectory_scaled).flatten()

            # Compute elementary effects
            for i, param_idx in enumerate(perm):
                dy = outputs[i + 1] - outputs[i]
                dx = trajectory[i + 1, param_idx] - trajectory[i, param_idx]
                param_range = ub[param_idx] - lb[param_idx]
                elementary_effects[traj, param_idx] = dy / (dx * param_range) if dx != 0 else 0

        # Compute statistics
        mu = np.mean(elementary_effects, axis=0)
        mu_star = np.mean(np.abs(elementary_effects), axis=0)
        sigma = np.std(elementary_effects, axis=0)

        return MorrisIndices(
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            parameter_names=param_names,
            elementary_effects=elementary_effects,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
        )

    # Run Morris screening
    morris_results = run_morris_screening(
        model_func=synthetic_model,
        param_names=FULL_PARAM_NAMES,
        param_bounds=FULL_PARAM_BOUNDS,
        n_trajectories=30,
        n_levels=6,
    )

    print(morris_results.summary())
    return MorrisIndices, morris_results, run_morris_screening


# =============================================================================
# STEP 4: IDENTIFY IMPORTANT PARAMETERS
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Identify Important Parameters

    Based on Morris screening, we select the **top parameters** for detailed analysis.
    """)
    return


@app.cell
def _(FULL_PARAM_BOUNDS, morris_results):
    # Get top 5 most influential parameters
    TOP_N = 5
    important_params = morris_results.get_screening_candidates(top_n=TOP_N)
    print(f"Top {TOP_N} parameters for detailed analysis:")
    for name, mu_star in morris_results.select(n=TOP_N):
        print(f"  {name}: μ*={mu_star:.4f}")

    # Classify all parameters
    classification = morris_results.classify_parameters()
    print(f"\nClassification:")
    print(f"  Negligible (can fix): {classification['negligible']}")
    print(f"  Linear effects: {classification['linear']}")
    print(f"  Nonlinear/interactions: {classification['nonlinear']}")

    # Convert to PARAMETER_CONFIG format
    PARAMETER_CONFIG = morris_results.to_parameter_config(
        parameter_bounds=FULL_PARAM_BOUNDS,
        top_n=TOP_N,
    )

    print(f"\nGenerated PARAMETER_CONFIG for reactive tutorial:")
    for cfg in PARAMETER_CONFIG:
        print(f"  {cfg['name']}: bounds={cfg['bounds']}")
    return PARAMETER_CONFIG, TOP_N, classification, important_params


# =============================================================================
# STEP 4b: VARIANCE DECOMPOSITION
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 4b: Variance Decomposition

    In the full RFC006 pipeline, after aggregation with strategies 1-3,
    variance decomposition reveals *where* uncertainty comes from:
    generation effects, stochastic seeding, or intrinsic cell-cycle dynamics.
    """)
    return


@app.cell
def _(np):
    # Synthetic variance decomposition (in practice, from Aggregator)
    variance_decomposition = {
        "total_variance": np.array([1.0, 0.8, 0.5]),
        "between_generation_variance": np.array([0.5, 0.3, 0.2]),
        "between_seed_variance": np.array([0.3, 0.3, 0.1]),
        "generation_fraction": np.array([0.5, 0.375, 0.4]),
        "seed_fraction": np.array([0.3, 0.375, 0.2]),
    }

    for i, name in enumerate(["transcriptome", "proteome", "fluxes"]):
        gen = variance_decomposition["generation_fraction"][i]
        seed = variance_decomposition["seed_fraction"][i]
        residual = 1 - gen - seed
        print(f"{name:15s}  generation={gen:.0%}  seed={seed:.0%}  residual(CC)={residual:.0%}")
    return (variance_decomposition,)


# =============================================================================
# STEP 5: REACTIVE EXPLORATION WITH SLIDERS
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Reactive Parameter Exploration

    Now we create sliders for only the **important parameters** identified by
    Morris screening. The other parameters are fixed at their default values.

    **Drag the sliders** to see how the output changes!
    """)
    return


@app.cell
def _(PARAMETER_CONFIG, mo, np):
    # Create sliders dynamically from PARAMETER_CONFIG
    _slider_list = []
    for _cfg in PARAMETER_CONFIG:
        _slider = mo.ui.slider(
            start=_cfg["bounds"][0],
            stop=_cfg["bounds"][1],
            step=_cfg["step"],
            value=_cfg["default"],
            label=_cfg["name"],
            show_value=True,
        )
        _slider_list.append(_slider)

    param_sliders = mo.ui.array(_slider_list)

    # Fixed values for non-important parameters (at midpoint)
    FIXED_DEFAULTS = {
        "gene_expression_C": 2.55,
        "translation_eff_B": 1.25,
        "translation_eff_C": 1.25,
        "diffusion_coeff": 0.05,
        "temperature_factor": 1.0,
        "pH_factor": 1.0,
        "nutrient_level": 1.05,
        "stress_response": 0.5,
    }
    return FIXED_DEFAULTS, param_sliders


@app.cell
def _(
    FIXED_DEFAULTS,
    FULL_PARAM_BOUNDS,
    FULL_PARAM_NAMES,
    PARAMETER_CONFIG,
    mo,
    np,
    param_sliders,
    synthetic_model,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Build full parameter vector from sliders + fixed values
    _full_params = []
    _slider_idx = 0
    _slider_names = [cfg["name"] for cfg in PARAMETER_CONFIG]

    for _i, _name in enumerate(FULL_PARAM_NAMES):
        if _name in _slider_names:
            _full_params.append(param_sliders.value[_slider_names.index(_name)])
        elif _name in FIXED_DEFAULTS:
            _full_params.append(FIXED_DEFAULTS[_name])
        else:
            # Use midpoint of bounds
            _lb, _ub = FULL_PARAM_BOUNDS[_i]
            _full_params.append((_lb + _ub) / 2)

    current_params = np.array(_full_params)
    current_output = synthetic_model(current_params)[0]

    # Also compute baseline (all at defaults)
    _baseline_params = []
    for _i, _name in enumerate(FULL_PARAM_NAMES):
        _lb, _ub = FULL_PARAM_BOUNDS[_i]
        _baseline_params.append((_lb + _ub) / 2)
    baseline_output = synthetic_model(np.array(_baseline_params))[0]

    # Generate a timeseries (synthetic dynamics)
    _t = np.arange(100)
    _baseline_ts = baseline_output * np.exp(0.01 * _t) + 0.5 * np.sin(0.1 * _t)
    _current_ts = current_output * np.exp(0.01 * _t) + 0.5 * np.sin(0.1 * _t)

    # Create figure
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Output Trajectory", "Difference from Baseline"],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=_t,
            y=_baseline_ts,
            name="Baseline",
            line=dict(color="gray", dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=_t,
            y=_current_ts,
            name="Current",
            line=dict(color="cyan", width=2),
            fill="tonexty",
            fillcolor="rgba(0, 255, 255, 0.1)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=_t,
            y=_current_ts - _baseline_ts,
            name="Difference",
            line=dict(color="magenta"),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 255, 0.2)",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)

    fig.update_layout(
        height=500,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    # Parameter info panel
    _param_info = "\n".join([
        f"- **{cfg['name']}**: `{param_sliders.value[i]:.3f}`" for i, cfg in enumerate(PARAMETER_CONFIG)
    ])

    _info_panel = mo.vstack([
        mo.md("### Important Parameters"),
        mo.md("*Identified by Morris screening*"),
        param_sliders,
        mo.md("---"),
        mo.md(f"""
**Current output:** `{current_output:.4f}`

**Baseline output:** `{baseline_output:.4f}`

**Change:** `{current_output - baseline_output:+.4f}` ({100 * (current_output - baseline_output) / baseline_output:+.1f}%)

---

**Fixed parameters** (negligible influence):
{", ".join(FIXED_DEFAULTS.keys())}
        """),
    ])

    mo.hstack([_info_panel, fig], widths=[1, 3], gap=2)
    return (
        baseline_output,
        current_output,
        current_params,
        fig,
        go,
        make_subplots,
    )


# =============================================================================
# SUMMARY
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Summary: The Complete Workflow

    ### What We Did

    1. **Started with 15 parameters** - too many for direct PCE/Sobol
    2. **Ran Morris screening** - 320 evaluations (vs ~2000+ for full PCE)
    3. **Identified top 5 parameters** - reduced problem dimensionality by 3x
    4. **Generated PARAMETER_CONFIG** - ready for reactive exploration
    5. **Created reactive sliders** - instant exploration of important parameters

    ### Key Takeaways

    | Step | Method | Cost | Output |
    |------|--------|------|--------|
    | Screening | Morris | O(n) | Important param names |
    | Detailed | PCE/Sobol | O(n²) | Sensitivity indices |
    | Exploration | PCE predict | O(1) | Instant predictions |

    ### In Practice

    ```python
    from uq import SensitivityAnalyzer, InputParameterSpace

    # 1. Full parameter space
    full_space = InputParameterSpace(include_vio=True, include_mecillinam=True, ...)

    # 2. Morris screening
    analyzer = SensitivityAnalyzer(full_space, wrapper)
    morris = analyzer.analyze_with_morris(n_trajectories=20)

    # 3. Get PARAMETER_CONFIG for tutorial
    PARAMETER_CONFIG = morris.to_parameter_config(
        parameter_bounds=full_space.parameter_bounds,
        top_n=5,
    )

    # 4. Use in 03c_reactive_sensitivity_generalized.py
    ```

    ### Next Steps

    - Use `morris.classify_parameters()` to understand effect types
    - Run detailed PCE on the reduced parameter space
    - Analyze interactions between important parameters
    """)
    return


if __name__ == "__main__":
    app.run()
