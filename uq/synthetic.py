import json

import numpy as np
import polars as pl

from uq import AggregatedOutput


def generate_synthetic_simulation_data(
    n_experiments: int = 3,
    n_seeds: int = 4,
    n_generations: int = 8,
    n_timepoints_per_gen: int = 100,
    n_cistrons: int = 50,
    n_proteins: int = 50,
    n_fluxes: int = 20,
) -> pl.DataFrame:
    """
    Generate synthetic simulation data mimicking vEcoli output structure.

    This creates realistic-looking data for demonstration purposes when
    real simulation outputs are not available.
    """
    np.random.seed(42)

    rows = []
    time = 0.0

    for exp_id in range(n_experiments):
        # Each experiment has different parameter values
        vio_expression = 0.5 + exp_id * 2.0  # 0.5, 2.5, 4.5
        mec_concentration = exp_id * 3.0  # 0, 3, 6

        for seed in range(n_seeds):
            seed_value = seed * 1000

            for gen in range(n_generations):
                # Simulate cell growth within generation
                for t_idx in range(n_timepoints_per_gen):
                    time_in_gen = t_idx / n_timepoints_per_gen

                    # Mass grows exponentially within generation
                    base_mass = 1.0 * (2.0**time_in_gen)

                    # Add stochastic variation
                    mass_noise = np.random.normal(0, 0.05)
                    dry_mass = base_mass * (1 + mass_noise)

                    # DNA mass increases during replication (C period)
                    if 0.3 < time_in_gen < 0.7:
                        dna_mass = 0.03 * (1 + (time_in_gen - 0.3) / 0.4)
                    else:
                        dna_mass = 0.03 if time_in_gen < 0.3 else 0.06

                    # Growth rate varies with cell cycle
                    growth_rate = 0.01 * (1 + 0.2 * np.sin(2 * np.pi * time_in_gen))

                    # Transcriptome: influenced by vio_expression
                    transcriptome = np.random.poisson(100 * (1 + 0.1 * vio_expression), size=n_cistrons).astype(float)

                    # Add cell cycle variation to some genes
                    cell_cycle_genes = np.sin(2 * np.pi * time_in_gen + np.random.rand(10) * np.pi)
                    transcriptome[:10] *= 1 + 0.3 * cell_cycle_genes

                    # Proteome: correlated with transcriptome but delayed
                    proteome = np.random.poisson(500 * (1 + 0.05 * vio_expression), size=n_proteins).astype(float)

                    # Fluxes: influenced by mecillinam
                    base_flux = max(1.0, 10.0 * (1 - 0.05 * mec_concentration))
                    fluxes = np.random.exponential(base_flux, size=n_fluxes)

                    # Exchange fluxes (subset with EX_ prefix behavior)
                    exchange_fluxes = fluxes[:5] * np.random.uniform(0.8, 1.2, 5)

                    row = {
                        "experiment_id": exp_id,
                        "variant": f"vio_{vio_expression:.1f}_mec_{mec_concentration:.1f}",
                        "lineage_seed": seed_value,
                        "generation": gen,
                        "agent_id": f"agent_{seed}_{gen}",
                        "time": time,
                        "listeners__mass__dry_mass": dry_mass,
                        "listeners__mass__cell_mass": dry_mass * 1.3,
                        "listeners__mass__dna_mass": dna_mass,
                        "listeners__mass__protein_mass": dry_mass * 0.5,
                        "listeners__fba_results__growth": growth_rate,
                        # Store arrays as JSON strings for demonstration
                        "listeners__rna_counts__mRNA_cistron_counts": json.dumps(transcriptome.tolist()),
                        "listeners__monomer_counts": json.dumps(proteome.tolist()),
                        "listeners__fba_results__base_reaction_fluxes": json.dumps(fluxes.tolist()),
                        # Parameter values for this experiment
                        "_param_vio_expression": vio_expression,
                        "_param_mec_concentration": mec_concentration,
                    }
                    rows.append(row)
                    time += 1.0

    return pl.DataFrame(rows)


def create_synthetic_aggregated_outputs(
    data: pl.DataFrame,
    n_features: int = 10,
) -> dict[str, AggregatedOutput]:
    """Create aggregated outputs from synthetic data."""

    # Strategy 1: Uniform aggregation
    uniform_mean = (
        data.select([
            "listeners__mass__dry_mass",
            "listeners__fba_results__growth",
        ])
        .mean()
        .to_numpy()
        .flatten()
    )

    uniform_std = (
        data.select([
            "listeners__mass__dry_mass",
            "listeners__fba_results__growth",
        ])
        .std()
        .to_numpy()
        .flatten()
    )

    agg_uniform = AggregatedOutput(
        mean=uniform_mean,
        std=uniform_std,
        n_samples=len(data),
        groups=None,
    )

    # Strategy 2: By generation
    by_gen = (
        data.group_by("generation")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__fba_results__growth").mean().alias("growth_mean"),
            pl.col("listeners__fba_results__growth").std().alias("growth_std"),
            pl.count().alias("n"),
        ])
        .sort("generation")
    )

    agg_by_gen = AggregatedOutput(
        mean=np.column_stack([
            by_gen["mass_mean"].to_numpy(),
            by_gen["growth_mean"].to_numpy(),
        ]),
        std=np.column_stack([
            by_gen["mass_std"].to_numpy(),
            by_gen["growth_std"].to_numpy(),
        ]),
        n_samples=by_gen["n"].to_numpy(),
        groups=by_gen["generation"].to_numpy(),
    )

    # Strategy 3: By lineage seed
    by_seed = (
        data.group_by("lineage_seed")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__fba_results__growth").mean().alias("growth_mean"),
            pl.col("listeners__fba_results__growth").std().alias("growth_std"),
            pl.count().alias("n"),
        ])
        .sort("lineage_seed")
    )

    agg_by_seed = AggregatedOutput(
        mean=np.column_stack([
            by_seed["mass_mean"].to_numpy(),
            by_seed["growth_mean"].to_numpy(),
        ]),
        std=np.column_stack([
            by_seed["mass_std"].to_numpy(),
            by_seed["growth_std"].to_numpy(),
        ]),
        n_samples=by_seed["n"].to_numpy(),
        groups=by_seed["lineage_seed"].to_numpy(),
    )

    return {
        "uniform": agg_uniform,
        "by_generation": agg_by_gen,
        "by_lineage_seed": agg_by_seed,
    }


def generate_signal(
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
