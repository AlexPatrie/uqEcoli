"""Tutorial 2: Aggregation Strategies and Variance Decomposition

This tutorial explores the four aggregation strategies and demonstrates
how to decompose variance to understand different sources of uncertainty.

Run with: marimo run 02_aggregation_strategies.py
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
        # Tutorial 2: Aggregation Strategies and Variance Decomposition

        This tutorial explores how to aggregate simulation data using different
        strategies to characterize various types of uncertainty.

        ## What You'll Learn

        1. **Uniform Aggregation** - The baseline "bulk" average
        2. **Generation Stratification** - Tracking convergence
        3. **Lineage Seed Stratification** - Quantifying stochastic variance
        4. **Variance Decomposition** - Attributing variance to sources

        ## Key Insight

        Different aggregation strategies reveal different aspects of model behavior:
        - **Uniform**: What does the average cell look like?
        - **By Generation**: Has the model converged to steady state?
        - **By Seed**: How much variance comes from stochastic effects?
        """
    )
    return


@app.cell
def __():
    # Import required modules
    import numpy as np
    import polars as pl
    from uq import (
        AggregationStrategy,
        AggregatedOutput,
        compute_variance_decomposition,
    )
    return (
        AggregatedOutput,
        AggregationStrategy,
        compute_variance_decomposition,
        np,
        pl,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## 1. Creating Synthetic Simulation Data

        For this tutorial, we'll create synthetic data that mimics the structure
        of vEcoli simulation output. This lets us demonstrate concepts without
        requiring actual simulation data.
        """
    )
    return


@app.cell
def __(np, pl):
    def generate_synthetic_data(
        n_seeds: int = 4,
        n_generations: int = 8,
        n_timepoints: int = 20,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Generate synthetic simulation data."""
        rng = np.random.default_rng(seed)
        rows = []

        for seed_val in range(n_seeds):
            seed_effect = rng.normal(0, 0.3)  # Seed-specific offset

            for gen in range(n_generations):
                gen_effect = 0.1 * gen  # Generation trend

                for t in range(n_timepoints):
                    time_in_gen = t / n_timepoints

                    # Mass grows exponentially within generation
                    base_mass = 1.0 * (2.0 ** time_in_gen)
                    mass = base_mass * (1 + seed_effect + gen_effect + rng.normal(0, 0.05))

                    # Growth rate with cell cycle variation
                    growth = 0.01 * (1 + 0.2 * np.sin(2 * np.pi * time_in_gen))
                    growth += rng.normal(0, 0.001)

                    rows.append({
                        "lineage_seed": seed_val * 1000,
                        "generation": gen,
                        "time": gen * 100 + t,
                        "agent_id": f"agent_{seed_val}_{gen}_{t}",
                        "dry_mass": mass,
                        "growth_rate": growth,
                    })

        return pl.DataFrame(rows)

    # Generate the data
    sim_data = generate_synthetic_data()
    print(f"Generated {len(sim_data)} data points")
    print(f"Columns: {sim_data.columns}")
    return generate_synthetic_data, sim_data


@app.cell
def __(mo, sim_data):
    mo.md(f"""
    **Synthetic Data Summary:**
    - Total rows: `{len(sim_data)}`
    - Lineage seeds: `{sim_data['lineage_seed'].unique().to_list()}`
    - Generations: `{sorted(sim_data['generation'].unique().to_list())}`
    - Features: `dry_mass`, `growth_rate`
    """)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 2. Strategy 1: Uniform Aggregation

        Uniform aggregation computes statistics across **all samples** regardless
        of generation, seed, or time. This gives us the "bulk" population average.
        """
    )
    return


@app.cell
def __(AggregatedOutput, mo, np, sim_data):
    # Extract data as numpy array
    data = sim_data.select(["dry_mass", "growth_rate"]).to_numpy()

    # Compute uniform aggregation
    uniform_result = AggregatedOutput(
        mean=np.mean(data, axis=0),
        std=np.std(data, axis=0),
        n_samples=len(data),
        groups=None,  # No groups for uniform
    )

    mo.md(f"""
    ### Uniform Aggregation Results

    | Metric | Dry Mass | Growth Rate |
    |--------|----------|-------------|
    | **Mean** | {uniform_result.mean[0]:.4f} | {uniform_result.mean[1]:.6f} |
    | **Std** | {uniform_result.std[0]:.4f} | {uniform_result.std[1]:.6f} |
    | **N Samples** | {uniform_result.n_samples} | {uniform_result.n_samples} |

    This represents what a "bulk" measurement would observe - the average
    across all cells at all times.
    """)
    return data, uniform_result


@app.cell
def __(mo):
    mo.md(
        """
        ## 3. Strategy 2: Stratified by Generation

        Generation stratification computes separate statistics for each generation.
        This helps us understand:
        - Has the model **converged** to steady-state growth?
        - How do outputs **evolve** across cell divisions?
        """
    )
    return


@app.cell
def __(AggregatedOutput, data, mo, np, sim_data):
    # Get generation labels
    generations = sim_data["generation"].to_numpy()
    unique_gens = np.unique(generations)

    # Compute per-generation statistics
    gen_means = np.array([np.mean(data[generations == g], axis=0) for g in unique_gens])
    gen_stds = np.array([np.std(data[generations == g], axis=0) for g in unique_gens])
    gen_counts = np.array([np.sum(generations == g) for g in unique_gens])

    by_generation = AggregatedOutput(
        mean=gen_means,
        std=gen_stds,
        n_samples=gen_counts,
        groups=unique_gens,
    )

    mo.md(f"""
    ### By-Generation Results

    **Shape:** {by_generation.mean.shape} (generations x features)

    | Generation | Mean Mass | Std Mass | Mean Growth | Std Growth | N |
    |------------|-----------|----------|-------------|------------|---|
    """ + "\n".join([
        f"| {g} | {by_generation.mean[i, 0]:.3f} | {by_generation.std[i, 0]:.3f} | "
        f"{by_generation.mean[i, 1]:.5f} | {by_generation.std[i, 1]:.5f} | {by_generation.n_samples[i]} |"
        for i, g in enumerate(unique_gens)
    ]))
    return (
        by_generation,
        gen_counts,
        gen_means,
        gen_stds,
        generations,
        unique_gens,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ### Interpreting Generation Trends

        Notice how the mean mass increases with generation number. This is expected
        because:
        1. Cells grow and divide, passing biomass to daughter cells
        2. Early generations may not have reached "physiological steady state"

        **Key Question**: At what generation does the model converge?

        Look for when the statistics stabilize (stop trending).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 4. Strategy 3: Stratified by Lineage Seed

        Lineage seed stratification computes statistics for each stochastic
        initialization. This quantifies **exogenous variance** - the variance
        that comes from random initial conditions.
        """
    )
    return


@app.cell
def __(AggregatedOutput, data, mo, np, sim_data):
    # Get seed labels
    seeds = sim_data["lineage_seed"].to_numpy()
    unique_seeds = np.unique(seeds)

    # Compute per-seed statistics
    seed_means = np.array([np.mean(data[seeds == s], axis=0) for s in unique_seeds])
    seed_stds = np.array([np.std(data[seeds == s], axis=0) for s in unique_seeds])
    seed_counts = np.array([np.sum(seeds == s) for s in unique_seeds])

    by_seed = AggregatedOutput(
        mean=seed_means,
        std=seed_stds,
        n_samples=seed_counts,
        groups=unique_seeds,
    )

    mo.md(f"""
    ### By-Seed Results

    **Shape:** {by_seed.mean.shape} (seeds x features)

    | Seed | Mean Mass | Std Mass | Mean Growth | Std Growth | N |
    |------|-----------|----------|-------------|------------|---|
    """ + "\n".join([
        f"| {s} | {by_seed.mean[i, 0]:.3f} | {by_seed.std[i, 0]:.3f} | "
        f"{by_seed.mean[i, 1]:.5f} | {by_seed.std[i, 1]:.5f} | {by_seed.n_samples[i]} |"
        for i, s in enumerate(unique_seeds)
    ]))
    return by_seed, seed_counts, seed_means, seed_stds, seeds, unique_seeds


@app.cell
def __(by_seed, mo, np):
    # Compute between-seed variance
    between_seed_var = np.var(by_seed.mean, axis=0)

    mo.md(f"""
    ### Between-Seed Variance

    The variance of the seed means represents **exogenous stochasticity**:

    - Dry Mass: `{between_seed_var[0]:.6f}`
    - Growth Rate: `{between_seed_var[1]:.10f}`

    **Interpretation**: If simulations with different seeds give very different
    results, there's high exogenous variance. If they're similar, the model
    is more deterministic.
    """)
    return (between_seed_var,)


@app.cell
def __(mo):
    mo.md(
        """
        ## 5. Variance Decomposition

        Now we can decompose the total variance into components:
        - **Between-generation variance**: Due to convergence effects
        - **Between-seed variance**: Due to stochastic initialization
        - **Within-group variance**: Residual (intrinsic) variance
        """
    )
    return


@app.cell
def __(by_generation, by_seed, compute_variance_decomposition, mo, uniform_result):
    # Compute variance decomposition
    decomp = compute_variance_decomposition(
        by_generation,
        by_seed,
        uniform_result,
    )

    mo.md(f"""
    ### Variance Decomposition Results

    | Component | Dry Mass | Growth Rate |
    |-----------|----------|-------------|
    | **Total Variance** | {decomp['total_variance'][0]:.6f} | {decomp['total_variance'][1]:.10f} |
    | **Between-Generation** | {decomp['between_generation_variance'][0]:.6f} | {decomp['between_generation_variance'][1]:.10f} |
    | **Between-Seed** | {decomp['between_seed_variance'][0]:.6f} | {decomp['between_seed_variance'][1]:.10f} |

    ### Variance Fractions

    | Source | Dry Mass | Growth Rate |
    |--------|----------|-------------|
    | **Generation** | {100*decomp['generation_fraction'][0]:.1f}% | {100*decomp['generation_fraction'][1]:.1f}% |
    | **Seed** | {100*decomp['seed_fraction'][0]:.1f}% | {100*decomp['seed_fraction'][1]:.1f}% |
    """)
    return (decomp,)


@app.cell
def __(decomp, mo):
    mo.md(f"""
    ### Interpreting the Decomposition

    **For Dry Mass:**
    - {100*decomp['generation_fraction'][0]:.1f}% of variance is due to generation effects
    - {100*decomp['seed_fraction'][0]:.1f}% is due to stochastic seeding

    **For Growth Rate:**
    - {100*decomp['generation_fraction'][1]:.1f}% of variance is due to generation effects
    - {100*decomp['seed_fraction'][1]:.1f}% is due to stochastic seeding

    **Takeaway**: Mass shows strong generation effects (cells grow larger over
    generations in our synthetic data), while growth rate is more stable.
    """)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## 6. Visualizing the Strategies

        Let's create a simple visualization to compare the strategies.
        """
    )
    return


@app.cell
def __(by_generation, by_seed, mo, uniform_result):
    # Create a text-based visualization
    def bar(value, max_val, width=30):
        filled = int(width * value / max_val)
        return "█" * filled + "░" * (width - filled)

    max_mass = max(
        uniform_result.mean[0],
        by_generation.mean[:, 0].max(),
        by_seed.mean[:, 0].max(),
    )

    viz_text = """
    ### Mass Distribution by Strategy

    **Uniform (Global):**
    """
    viz_text += f"  Mean: {bar(uniform_result.mean[0], max_mass)} {uniform_result.mean[0]:.3f}\n\n"

    viz_text += "**By Generation:**\n"
    for i, g in enumerate(by_generation.groups):
        viz_text += f"  Gen {g}: {bar(by_generation.mean[i, 0], max_mass)} {by_generation.mean[i, 0]:.3f}\n"

    viz_text += "\n**By Seed:**\n"
    for i, s in enumerate(by_seed.groups):
        viz_text += f"  Seed {s}: {bar(by_seed.mean[i, 0], max_mass)} {by_seed.mean[i, 0]:.3f}\n"

    mo.md(viz_text)
    return bar, max_mass, viz_text


@app.cell
def __(mo):
    mo.md(
        """
        ## 7. Working with Real Data

        When working with actual vEcoli simulations, you'll use the `Aggregator`
        class which handles SQL queries and data extraction:

        ```python
        from uq import Aggregator, AggregationStrategy
        from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

        # Connect to simulation data
        conn = create_duckdb_conn()
        history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

        # Create aggregator
        aggregator = Aggregator(conn, history_sql, config_sql)

        # Aggregate transcriptome data
        result, cistron_ids = aggregator.aggregate_transcriptome(
            AggregationStrategy.BY_GENERATION,
            generation_lower_bound=2,  # Skip initial transients
        )
        ```

        The `Aggregator` supports:
        - `aggregate_transcriptome()` - mRNA counts
        - `aggregate_proteome()` - Protein counts
        - `aggregate_fluxes()` - Metabolic fluxes
        - `aggregate_higher_order_properties()` - Mass, volume, growth rate
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Summary

        In this tutorial, you learned:

        1. **Uniform Aggregation** - Computes global statistics (the "bulk" view)
        2. **Generation Stratification** - Reveals convergence behavior
        3. **Seed Stratification** - Quantifies exogenous stochastic variance
        4. **Variance Decomposition** - Attributes variance to different sources

        ## Key Formulas

        - **Total Variance** = Uniform std²
        - **Between-Group Variance** = Variance of group means
        - **Fraction** = Between-group variance / Total variance

        ## Next Steps

        Continue to **Tutorial 3** to learn about sensitivity analysis using
        Polynomial Chaos Expansion (PCE) and Sobol indices.
        """
    )
    return


if __name__ == "__main__":
    app.run()
