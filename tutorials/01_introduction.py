"""Tutorial 1: Introduction to UQ Framework

This tutorial introduces the core concepts of the vEcoli Uncertainty Quantification
framework, covering input parameters and basic data structures.

Run with: marimo run 01_introduction.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Tutorial 1: Introduction to the UQ Framework

        Welcome to the vEcoli Uncertainty Quantification (UQ) framework! This tutorial
        will introduce you to the core concepts and data structures used throughout
        the package.

        ## What You'll Learn

        1. **Input Parameters** - How to define experimental conditions
        2. **Parameter Spaces** - Setting up ranges for sensitivity analysis
        3. **Basic Data Structures** - Understanding the core types

        ## Prerequisites

        - Basic Python knowledge
        - Familiarity with numpy arrays
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 1. Input Parameters

        The UQ framework supports three categories of input parameters that represent
        scientifically relevant experimental conditions in vEcoli simulations.
        """
    )
    return


@app.cell
def __():
    # Import the input parameter classes
    from uq import VioPathwayParams, MecillinamParams, GeneKnockoutParams
    return GeneKnockoutParams, MecillinamParams, VioPathwayParams


@app.cell
def __(mo):
    mo.md(
        """
        ### 1.1 Violacein (vio) Pathway Parameters

        The vio pathway represents a heterologous gene expression system.
        You can control when it's induced and how strongly it's expressed.
        """
    )
    return


@app.cell
def __(VioPathwayParams, mo):
    # Create vio pathway parameters
    vio_params = VioPathwayParams(
        enabled=True,                    # Enable the pathway
        induction_gen=1,                 # Induce at generation 1
        expression=2.5,                  # 2.5x expression factor
        translation_efficiency=1.2,      # 1.2x translation efficiency
        condition="basal",               # Base media condition
    )

    mo.md(f"""
    **Created VioPathwayParams:**
    - Enabled: `{vio_params.enabled}`
    - Induction Generation: `{vio_params.induction_gen}`
    - Expression Factor: `{vio_params.expression}`
    - Translation Efficiency: `{vio_params.translation_efficiency}`
    """)
    return (vio_params,)


@app.cell
def __(mo):
    mo.md(
        """
        ### 1.2 Mecillinam Antibiotic Parameters

        Mecillinam is a beta-lactam antibiotic that affects cell wall synthesis.
        You can define a timeline of concentrations to simulate antibiotic exposure.
        """
    )
    return


@app.cell
def __(MecillinamParams, mo):
    # Create mecillinam parameters with a step increase
    mec_params = MecillinamParams(
        times=[0.0, 1800.0, 3600.0],           # Time points in seconds
        concentrations=[0.0, 2.5, 5.0],        # Concentration at each time
        knockouts=["murG"],                     # Genes to knock out
    )

    mo.md(f"""
    **Created MecillinamParams:**
    - Times (s): `{mec_params.times}`
    - Concentrations (mM): `{mec_params.concentrations}`
    - Knockouts: `{mec_params.knockouts}`

    This defines a step-wise increase in antibiotic concentration over the simulation.
    """)
    return (mec_params,)


@app.cell
def __(mo):
    mo.md(
        """
        ### 1.3 Gene Knockout Parameters

        Gene knockouts can be applied at different levels:
        - **Gene deletions**: Complete removal at the ParCa level
        - **Translation knockouts**: Set translation efficiency to zero
        """
    )
    return


@app.cell
def __(GeneKnockoutParams, mo):
    # Create knockout parameters
    ko_params = GeneKnockoutParams(
        gene_deletions=["lacZ", "galK"],        # Delete these genes
        translation_knockouts=["murG"],          # Block translation of these
    )

    mo.md(f"""
    **Created GeneKnockoutParams:**
    - Gene Deletions: `{ko_params.gene_deletions}`
    - Translation Knockouts: `{ko_params.translation_knockouts}`
    """)
    return (ko_params,)


@app.cell
def __(mo):
    mo.md(
        """
        ## 2. Combined Input Parameters

        The `UQInputParameters` class combines all parameter types into a single
        container that fully specifies a simulation configuration.
        """
    )
    return


@app.cell
def __(ko_params, mec_params, mo, vio_params):
    from uq import UQInputParameters

    # Combine all parameters
    full_params = UQInputParameters(
        vio=vio_params,
        mecillinam=mec_params,
        knockouts=ko_params,
        seed=42,               # Random seed for reproducibility
        generations=8,         # Number of generations to simulate
    )

    mo.md(f"""
    **Combined UQInputParameters:**
    - Seed: `{full_params.seed}`
    - Generations: `{full_params.generations}`
    - Vio Enabled: `{full_params.vio.enabled}`
    - Mecillinam Times: `{full_params.mecillinam.times}`

    This object can be converted to a simulation configuration dictionary.
    """)
    return UQInputParameters, full_params


@app.cell
def __(full_params, mo):
    # Convert to configuration dictionary
    config_dict = full_params.to_config_dict()

    mo.md(f"""
    **Configuration Dictionary:**
    ```python
    {config_dict}
    ```

    This dictionary format is compatible with vEcoli's simulation API.
    """)
    return (config_dict,)


@app.cell
def __(mo):
    mo.md(
        """
        ## 3. Parameter Spaces for Sensitivity Analysis

        For sensitivity analysis, we need to define **ranges** for parameters,
        not just single values. The `InputParameterSpace` class handles this.
        """
    )
    return


@app.cell
def __(mo):
    from uq import InputParameterSpace
    import numpy as np

    # Define a parameter space
    param_space = InputParameterSpace(
        include_vio=True,                          # Include vio parameters
        include_mecillinam=True,                   # Include mecillinam
        vio_expression_bounds=(0.5, 5.0),          # Expression range
        vio_trl_eff_bounds=(0.5, 2.0),             # Translation efficiency range
        mecillinam_conc_bounds=(0.0, 10.0),        # Concentration range
    )

    mo.md(f"""
    **Created InputParameterSpace:**
    - Number of Parameters: `{param_space.n_parameters}`
    - Parameter Names: `{param_space.parameter_names}`
    - Parameter Bounds: `{param_space.parameter_bounds}`
    """)
    return InputParameterSpace, np, param_space


@app.cell
def __(mo, np, param_space):
    # Get bounds in different formats
    bounds_array = param_space.bounds_array
    lb, ub = param_space.get_pytuq_bounds()

    mo.md(f"""
    ### Bounds Formats

    **As NumPy Array (for UQPy):**
    ```
    {bounds_array}
    ```

    **As Lower/Upper Bounds (for PyTUQ):**
    - Lower: `{lb}`
    - Upper: `{ub}`
    """)
    return bounds_array, lb, ub


@app.cell
def __(mo):
    mo.md(
        """
        ## 4. Sampling and Converting Parameters

        The parameter space can generate samples and convert them back to
        `UQInputParameters` objects for simulation.
        """
    )
    return


@app.cell
def __(lb, mo, np, param_space, ub):
    # Generate random samples within bounds
    rng = np.random.default_rng(42)
    n_samples = 5
    samples = rng.uniform(lb, ub, size=(n_samples, param_space.n_parameters))

    mo.md(f"""
    **Generated {n_samples} Random Samples:**
    ```
    {samples}
    ```

    Each row is a different parameter combination within the specified bounds.
    """)
    return n_samples, rng, samples


@app.cell
def __(mo, param_space, samples):
    # Convert a sample to UQInputParameters
    sample = samples[0]  # Take first sample
    params_from_sample = param_space.sample_to_params(
        sample,
        seed=100,
        generations=8,
    )

    mo.md(f"""
    **Sample Converted to Parameters:**

    Sample values: `{sample}`

    Resulting parameters:
    - Vio Expression: `{params_from_sample.vio.expression:.3f}`
    - Vio Translation Eff: `{params_from_sample.vio.translation_efficiency:.3f}`
    - Mecillinam Conc: `{params_from_sample.mecillinam.concentrations}`
    """)
    return params_from_sample, sample


@app.cell
def __(mo):
    mo.md(
        """
        ## 5. Aggregation Strategies Overview

        The UQ framework defines four aggregation strategies for characterizing
        different types of uncertainty:
        """
    )
    return


@app.cell
def __(mo):
    from uq import AggregationStrategy

    strategies = list(AggregationStrategy)

    mo.md(f"""
    **Available Aggregation Strategies:**

    | Strategy | Value | Purpose |
    |----------|-------|---------|
    | `UNIFORM` | `{AggregationStrategy.UNIFORM.value}` | Baseline "bulk" population average |
    | `BY_GENERATION` | `{AggregationStrategy.BY_GENERATION.value}` | Track convergence to steady state |
    | `BY_LINEAGE_SEED` | `{AggregationStrategy.BY_LINEAGE_SEED.value}` | Quantify exogenous (stochastic) variance |
    | `BY_CELL_CYCLE` | `{AggregationStrategy.BY_CELL_CYCLE.value}` | Phenotypic analysis across cell cycle |

    These strategies are explored in detail in Tutorial 2.
    """)
    return AggregationStrategy, strategies


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        In this tutorial, you learned:

        1. **VioPathwayParams** - Control heterologous gene expression
        2. **MecillinamParams** - Define antibiotic exposure timelines
        3. **GeneKnockoutParams** - Specify gene deletions and knockouts
        4. **UQInputParameters** - Combine all parameters for simulation
        5. **InputParameterSpace** - Define ranges for sensitivity analysis
        6. **AggregationStrategy** - Four strategies for characterizing uncertainty

        ## Next Steps

        Continue to **Tutorial 2** to learn about aggregation strategies and
        variance decomposition in detail.
        """
    )
    return


if __name__ == "__main__":
    app.run()
