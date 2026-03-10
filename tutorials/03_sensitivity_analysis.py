"""Tutorial 3: Sensitivity Analysis with PCE and Sobol Indices

This tutorial demonstrates global sensitivity analysis using Polynomial
Chaos Expansion (PCE) surrogate models and Sobol sensitivity indices.

Run with: marimo run 03_sensitivity_analysis.py
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
    # Tutorial 3: Sensitivity Analysis with PCE and Sobol Indices

    Global Sensitivity Analysis (GSA) helps answer a crucial question:
    **Which input parameters have the biggest impact on model outputs?**

    ## What You'll Learn

    1. **Polynomial Chaos Expansion (PCE)** - Building surrogate models
    2. **Sobol Indices** - Quantifying parameter importance
    3. **First-Order vs Total-Order** - Main effects vs interactions
    4. **Practical Workflow** - From samples to insights

    ## Why PCE?

    PCE is recommended over Monte Carlo Sobol analysis because:
    - **Fewer samples needed** - PCE builds an analytical surrogate
    - **Analytical Sobol indices** - Computed directly from coefficients
    - **Reusable surrogate** - Can make predictions without new simulations
    """)
    return


@app.cell
def _():
    import numpy as np
    from uq import (
        InputParameterSpace,
        SensitivityAnalyzer,
        SobolIndices,
        PCESurrogate,
        SensitivityMethod,
    )

    return (
        InputParameterSpace,
        PCESurrogate,
        SensitivityAnalyzer,
        SobolIndices,
        np,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Setting Up the Parameter Space

    First, we define the input parameters and their ranges. This tells the
    sensitivity analysis what parameters to vary and within what bounds.
    """)
    return


@app.cell
def _(InputParameterSpace, mo):
    # Define the parameter space
    param_space = InputParameterSpace(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.5, 5.0),  # Expression factor
        vio_trl_eff_bounds=(0.5, 2.0),  # Translation efficiency
        mecillinam_conc_bounds=(0.0, 10.0),  # Antibiotic concentration
    )

    mo.md(f"""
    ### Parameter Space Configuration

    | Parameter | Lower Bound | Upper Bound |
    |-----------|-------------|-------------|
    | `vio_expression` | 0.5 | 5.0 |
    | `vio_trl_eff` | 0.5 | 2.0 |
    | `mecillinam_concentration` | 0.0 | 10.0 |

    **Total Parameters:** {param_space.n_parameters}
    """)
    return (param_space,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Creating a Synthetic Model Function

    For this tutorial, we'll use a synthetic function with **known sensitivities**.
    This lets us verify that our analysis recovers the correct importance ranking.

    Our function: `Y = 3*x₁ + x₂ + 0.5*x₃ + ε`

    Where:
    - x₁ = vio_expression (should be most important)
    - x₂ = vio_trl_eff (medium importance)
    - x₃ = mecillinam_concentration (least important)
    - ε = small noise
    """)
    return


@app.cell
def _(np, param_space):
    def synthetic_model(X: np.ndarray) -> np.ndarray:
        """
        Synthetic model with known sensitivities.

        Y = 3*x1 + 1*x2 + 0.5*x3 + noise

        For uniform [0,1] inputs:
        - Var(Y) ≈ 9*var(x1) + 1*var(x2) + 0.25*var(x3)
        - S1 ≈ 9/10.25 ≈ 0.878 (most important)
        - S2 ≈ 1/10.25 ≈ 0.098
        - S3 ≈ 0.25/10.25 ≈ 0.024 (least important)
        """
        # Normalize inputs to [0, 1]
        lb, ub = param_space.get_pytuq_bounds()
        X_norm = (X - lb) / (ub - lb)

        # Compute output
        Y = 3.0 * X_norm[:, 0] + 1.0 * X_norm[:, 1] + 0.5 * X_norm[:, 2]

        # Add small noise
        Y += 0.01 * np.random.randn(len(Y))

        return Y.reshape(-1, 1)

    print("Synthetic model defined with known sensitivities:")
    print("  x1 (vio_expression): ~87.8% of variance")
    print("  x2 (vio_trl_eff): ~9.8% of variance")
    print("  x3 (mecillinam_conc): ~2.4% of variance")
    return (synthetic_model,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Generating Samples

    PCE requires samples from the input space. Latin Hypercube Sampling (LHS)
    provides good coverage with fewer samples than pure random sampling.
    """)
    return


@app.cell
def _(mo, np, param_space, synthetic_model):
    # Generate samples using Latin Hypercube Sampling (simplified version)
    rng = np.random.default_rng(42)
    n_samples = 100

    lb, ub = param_space.get_pytuq_bounds()
    X = rng.uniform(lb, ub, size=(n_samples, param_space.n_parameters))

    # Evaluate the model
    Y = synthetic_model(X)

    mo.md(f"""
    ### Generated Samples

    - **Number of samples:** {n_samples}
    - **Input shape:** {X.shape}
    - **Output shape:** {Y.shape}

    **Sample Statistics:**
    | Parameter | Mean | Std | Min | Max |
    |-----------|------|-----|-----|-----|
    | vio_expression | {X[:, 0].mean():.2f} | {X[:, 0].std():.2f} | {X[:, 0].min():.2f} | {X[:, 0].max():.2f} |
    | vio_trl_eff | {X[:, 1].mean():.2f} | {X[:, 1].std():.2f} | {X[:, 1].min():.2f} | {X[:, 1].max():.2f} |
    | mecillinam_conc | {X[:, 2].mean():.2f} | {X[:, 2].std():.2f} | {X[:, 2].min():.2f} | {X[:, 2].max():.2f} |

    **Output:** Mean = {Y.mean():.3f}, Std = {Y.std():.3f}
    """)
    return X, Y


@app.cell
def _(mo):
    mo.md("""
    ## 4. Creating the Sensitivity Analyzer

    The `SensitivityAnalyzer` class handles PCE construction and Sobol
    index computation. We can provide precomputed samples and outputs.
    """)
    return


@app.cell
def _(SensitivityAnalyzer, X, Y, mo, param_space):
    # Create analyzer with precomputed data
    analyzer = SensitivityAnalyzer(
        parameter_space=param_space,
        samples=X,
        outputs=Y,
    )

    mo.md(f"""
    ### Sensitivity Analyzer Created

    - **Parameter space:** {param_space.n_parameters} parameters
    - **Samples loaded:** {analyzer.samples.shape[0]}
    - **Outputs loaded:** {analyzer.outputs.shape}

    The analyzer is ready to perform PCE-based sensitivity analysis.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Understanding Sobol Indices

    Before running the analysis, let's understand what Sobol indices tell us:

    ### First-Order Index (S₁)
    - Measures the **main effect** of a parameter
    - "How much variance is explained by this parameter alone?"
    - Sum of all S₁ ≤ 1 (equals 1 if no interactions)

    ### Total-Order Index (Sᴛ)
    - Measures **total effect** including all interactions
    - "How much variance involves this parameter?"
    - Sᴛ ≥ S₁ always

    ### Interpreting the Difference
    - **Sᴛ - S₁** = interaction effects
    - If Sᴛ ≈ S₁: Parameter acts independently
    - If Sᴛ >> S₁: Parameter has strong interactions
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Computing Sobol Indices

    Since we don't have UQPy installed, we'll demonstrate with manually
    computed indices. In practice, you would use:

    ```python
    sobol, pce = analyzer.analyze_with_pce(
        polynomial_order=3,
        n_samples=100,
        use_uqpy=True,  # or False for PyTUQ
    )
    ```
    """)
    return


@app.cell
def _(SobolIndices, mo, np, param_space):
    # Create Sobol indices with known analytical values
    # For Y = 3*x1 + x2 + 0.5*x3 with uniform [0,1] inputs:
    # Total variance ∝ 9 + 1 + 0.25 = 10.25

    total_var = 10.25
    sobol_indices = SobolIndices(
        first_order=np.array([9 / total_var, 1 / total_var, 0.25 / total_var]),
        total_order=np.array([9 / total_var, 1 / total_var, 0.25 / total_var]),  # No interactions
        parameter_names=param_space.parameter_names,
        output_names=["model_output"],
    )

    mo.md(f"""
    ### Computed Sobol Indices

    | Parameter | First-Order (S₁) | Total-Order (Sᴛ) | Interaction (Sᴛ-S₁) |
    |-----------|------------------|------------------|---------------------|
    | vio_expression | {sobol_indices.first_order[0]:.3f} | {sobol_indices.total_order[0]:.3f} | {sobol_indices.total_order[0] - sobol_indices.first_order[0]:.3f} |
    | vio_trl_eff | {sobol_indices.first_order[1]:.3f} | {sobol_indices.total_order[1]:.3f} | {sobol_indices.total_order[1] - sobol_indices.first_order[1]:.3f} |
    | mecillinam_conc | {sobol_indices.first_order[2]:.3f} | {sobol_indices.total_order[2]:.3f} | {sobol_indices.total_order[2] - sobol_indices.first_order[2]:.3f} |

    **Sum of First-Order:** {sobol_indices.first_order.sum():.3f} (should be ~1.0 for additive model)
    """)
    return (sobol_indices,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Ranking Parameters by Importance

    The `get_most_influential()` method returns parameters sorted by importance.
    """)
    return


@app.cell
def _(mo, sobol_indices):
    # Get most influential parameters
    top_params = sobol_indices.get_most_influential(n=3, index_type="total")

    mo.md(
        """
    ### Parameter Ranking (by Total-Order Index)

    """
        + "\n".join([
            f"**{i + 1}. {name}**: Sᴛ = {value:.3f} ({100 * value:.1f}% of variance)"
            for i, (name, value) in enumerate(top_params)
        ])
        + """

    **Interpretation:**
    - `vio_expression` dominates - it explains ~88% of output variance
    - `vio_trl_eff` has moderate influence at ~10%
    - `mecillinam_concentration` has minimal impact at ~2%

    This matches our synthetic model where the coefficient for x₁ is 3x larger
    than x₂ and 6x larger than x₃.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Visualizing Sensitivity

    Let's create a simple bar chart visualization of the indices.
    """)
    return


@app.cell
def _(mo, param_space, sobol_indices):
    def bar_chart(values, labels, max_width=40):
        """Create a text-based bar chart."""
        max_val = max(values)
        lines = []
        for label, val in zip(labels, values):
            bar_len = int(max_width * val / max_val)
            bar = "█" * bar_len + "░" * (max_width - bar_len)
            lines.append(f"{label:25s} {bar} {val:.3f}")
        return "\n".join(lines)

    chart = f"""
    ### First-Order Indices (Main Effects)
    ```
    {bar_chart(sobol_indices.first_order, param_space.parameter_names)}
    ```

    ### Total-Order Indices (Including Interactions)
    ```
    {bar_chart(sobol_indices.total_order, param_space.parameter_names)}
    ```
    """

    mo.md(chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Working with Multiple Outputs

    Real simulations have multiple outputs (mass, growth rate, fluxes, etc.).
    Sobol indices can be computed for each output separately.
    """)
    return


@app.cell
def _(SobolIndices, mo, np, param_space):
    # Multi-output example: mass and growth_rate respond differently
    multi_output_sobol = SobolIndices(
        first_order=np.array([
            [0.70, 0.20, 0.08],  # Mass: dominated by vio_expression
            [0.30, 0.50, 0.15],  # Growth: more influenced by vio_trl_eff
        ]),
        total_order=np.array([
            [0.75, 0.22, 0.10],
            [0.35, 0.55, 0.18],
        ]),
        parameter_names=param_space.parameter_names,
        output_names=["dry_mass", "growth_rate"],
    )

    mo.md(f"""
    ### Multi-Output Sobol Indices

    **For Dry Mass:**
    | Parameter | S₁ | Sᴛ |
    |-----------|----|----|
    | vio_expression | {multi_output_sobol.first_order[0, 0]:.2f} | {multi_output_sobol.total_order[0, 0]:.2f} |
    | vio_trl_eff | {multi_output_sobol.first_order[0, 1]:.2f} | {multi_output_sobol.total_order[0, 1]:.2f} |
    | mecillinam_conc | {multi_output_sobol.first_order[0, 2]:.2f} | {multi_output_sobol.total_order[0, 2]:.2f} |

    **For Growth Rate:**
    | Parameter | S₁ | Sᴛ |
    |-----------|----|----|
    | vio_expression | {multi_output_sobol.first_order[1, 0]:.2f} | {multi_output_sobol.total_order[1, 0]:.2f} |
    | vio_trl_eff | {multi_output_sobol.first_order[1, 1]:.2f} | {multi_output_sobol.total_order[1, 1]:.2f} |
    | mecillinam_conc | {multi_output_sobol.first_order[1, 2]:.2f} | {multi_output_sobol.total_order[1, 2]:.2f} |

    **Key Insight:** Different outputs may be sensitive to different parameters!
    - Mass is most sensitive to vio_expression
    - Growth rate is most sensitive to vio_trl_eff
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. PCE Surrogate Model

    The PCE surrogate is a polynomial approximation of the simulation model.
    It can be used for:
    - **Fast predictions** without running new simulations
    - **Optimization** (finding optimal parameters)
    - **Uncertainty propagation** (predicting output distributions)
    """)
    return


@app.cell
def _(PCESurrogate, mo, np):
    # Example PCE surrogate (simplified)
    pce_surrogate = PCESurrogate(
        coefficients=np.array([1.5, 0.8, 0.3, 0.1]),  # Polynomial coefficients
        multi_indices=np.array([
            [0, 0, 0],  # Constant term
            [1, 0, 0],  # Linear in x1
            [0, 1, 0],  # Linear in x2
            [0, 0, 1],  # Linear in x3
        ]),
        basis_type="legendre",
        polynomial_order=3,
        input_dim=3,
        output_dim=1,
        r_squared=0.98,
    )

    mo.md(f"""
    ### PCE Surrogate Properties

    | Property | Value |
    |----------|-------|
    | Polynomial Order | {pce_surrogate.polynomial_order} |
    | Input Dimension | {pce_surrogate.input_dim} |
    | Output Dimension | {pce_surrogate.output_dim} |
    | Basis Type | {pce_surrogate.basis_type} |
    | R² (fit quality) | {pce_surrogate.r_squared:.2f} |
    | Number of Terms | {len(pce_surrogate.coefficients)} |

    The surrogate captures {100 * pce_surrogate.r_squared:.0f}% of the variance in the
    original model - a good fit!
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Complete Workflow with Real Data

    Here's how you would run a complete sensitivity analysis with real
    vEcoli simulation data:

    ```python
    from uq import run_sensitivity_analysis, AggregationStrategy

    # Run complete analysis
    sobol_indices, pce_surrogate = run_sensitivity_analysis(
        sim_data_path="./sim_data.cPickle",
        output_dir="./uq_outputs",
        aggregation_strategy=AggregationStrategy.BY_GENERATION,
        polynomial_order=3,
        n_samples=100,
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.5, 5.0),
        mecillinam_conc_bounds=(0.0, 10.0),
        use_uqpy=True,  # Use UQPy library
    )

    # Get top parameters
    print("Most influential parameters:")
    for name, value in sobol_indices.get_most_influential(n=5):
        print(f"  {name}: {value:.4f}")
    ```

    Or with precomputed simulation results:

    ```python
    from uq import analyze_precomputed_results

    sobol, pce = analyze_precomputed_results(
        data_dir="./simulation_outputs",
        aggregation_strategy=AggregationStrategy.UNIFORM,
        polynomial_order=3,
    )
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    In this tutorial, you learned:

    1. **PCE** builds a polynomial surrogate from simulation samples
    2. **First-Order Indices (S₁)** measure main effects
    3. **Total-Order Indices (Sᴛ)** include interactions (Sᴛ ≥ S₁)
    4. **Parameter Ranking** identifies which inputs matter most
    5. **Multi-Output Analysis** reveals output-specific sensitivities

    ## Key Takeaways

    - PCE is efficient: ~100 samples can build a good surrogate
    - Sobol indices are normalized: sum of S₁ ≤ 1
    - Different outputs may have different important parameters
    - The surrogate enables fast predictions and optimization

    ## Next Steps

    Continue to **Tutorial 4** to learn about cell cycle stratification
    and Koopman spectral analysis for understanding system dynamics.
    """)
    return


if __name__ == "__main__":
    app.run()
