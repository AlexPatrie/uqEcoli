"""
Pytest fixtures for UQ package tests.

These fixtures provide both synthetic data and real simulation data for testing.
Real data is loaded from api_simulation_default experiment using uq.inputs.load_dataset().
"""

import glob
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# =============================================================================
# Real Data Loading
# =============================================================================


# Standard columns used in UQ analysis (column name mapping)
COLUMN_MAPPING = {
    "dry_mass": "listeners__mass__dry_mass",
    "cell_mass": "listeners__mass__cell_mass",
    "dna_mass": "listeners__mass__dna_mass",
    "protein_mass": "listeners__mass__protein_mass",
    "growth": "listeners__mass__growth",
    "instantaneous_growth_rate": "listeners__mass__instantaneous_growth_rate",
    "time": "time",
}

# Columns to load from real data
REAL_DATA_COLUMNS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__dna_mass",
    "listeners__mass__protein_mass",
    "listeners__mass__growth",
    "listeners__mass__instantaneous_growth_rate",
    "time",
]


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    # Try to find repo root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


def _load_real_dataset_safe(
    experiment_id: str = "api_simulation_default",
    max_files: int | None = None,
) -> pl.DataFrame | None:
    """
    Load real simulation data safely, handling schema mismatches.

    Args:
        experiment_id: Experiment identifier
        max_files: Maximum number of parquet files to load (for faster testing)

    Returns:
        DataFrame or None if data not available
    """
    repo_root = _get_repo_root()
    base_path = repo_root / "api_integration/sims" / experiment_id / "history" / f"experiment_id={experiment_id}"

    if not base_path.exists():
        return None

    # Find parquet files
    files = sorted(glob.glob(str(base_path / "**/*.pq"), recursive=True))
    if not files:
        return None

    if max_files:
        files = files[:max_files]

    # Load files individually and concatenate
    dfs = []
    for f in files:
        try:
            # Extract metadata from path
            parts = Path(f).parts
            variant = None
            lineage_seed = None
            generation = None
            agent_id = None

            for part in parts:
                if part.startswith("variant="):
                    variant = int(part.split("=")[1])
                elif part.startswith("lineage_seed="):
                    lineage_seed = int(part.split("=")[1])
                elif part.startswith("generation="):
                    generation = int(part.split("=")[1])
                elif part.startswith("agent_id="):
                    agent_id = part.split("=")[1]

            # Read only needed columns
            df = pl.read_parquet(f, columns=REAL_DATA_COLUMNS)

            # Add metadata columns
            df = df.with_columns([
                pl.lit(variant).alias("variant"),
                pl.lit(lineage_seed).alias("lineage_seed"),
                pl.lit(generation).alias("generation"),
                pl.lit(agent_id).alias("agent_id"),
                pl.lit(experiment_id).alias("experiment_id"),
            ])

            dfs.append(df)
        except Exception:
            # Skip files with schema issues
            continue

    if not dfs:
        return None

    return pl.concat(dfs)


@pytest.fixture(scope="session")
def real_simulation_dataframe() -> pl.DataFrame | None:
    """
    Load real simulation data from api_simulation_default experiment.

    This fixture loads actual vEcoli simulation output for integration testing.
    Returns None if data is not available.
    """
    return _load_real_dataset_safe("api_simulation_default", max_files=50)


@pytest.fixture
def real_data_available(real_simulation_dataframe) -> bool:
    """Check if real simulation data is available."""
    return real_simulation_dataframe is not None


@pytest.fixture
def simulation_dataframe(real_simulation_dataframe, synthetic_simulation_dataframe):
    """
    Get simulation data, preferring real data if available.

    This fixture provides a unified interface that uses real data when available
    and falls back to synthetic data otherwise.
    """
    if real_simulation_dataframe is not None:
        # Rename columns to match expected schema
        df = real_simulation_dataframe.rename({
            "listeners__mass__growth": "listeners__fba_results__growth",
        })
        return df
    return synthetic_simulation_dataframe


# =============================================================================
# Synthetic Data Generation
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_trajectory(rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic trajectory data for Koopman analysis.

    Returns:
        Array of shape (n_timesteps, n_observables)
    """
    n_timesteps = 200
    n_observables = 3

    # Create trajectory with growth and oscillation
    t = np.arange(n_timesteps)

    # Observable 1: Exponential growth with oscillation (mass-like)
    obs1 = np.exp(0.01 * t) * (1 + 0.1 * np.sin(2 * np.pi * t / 50))

    # Observable 2: Damped oscillation (growth rate-like)
    obs2 = 0.01 * (1 + 0.3 * np.sin(2 * np.pi * t / 50) * np.exp(-0.005 * t))

    # Observable 3: Noisy linear trend
    obs3 = 0.5 + 0.001 * t + 0.02 * rng.normal(size=n_timesteps)

    trajectory = np.column_stack([obs1, obs2, obs3])
    return trajectory


@pytest.fixture
def synthetic_simulation_dataframe(rng: np.random.Generator) -> pl.DataFrame:
    """
    Generate synthetic simulation data mimicking vEcoli output structure.

    This fixture provides realistic data for testing aggregation, variance
    decomposition, and sensitivity analysis without requiring actual simulations.
    """
    n_experiments = 5
    n_seeds = 4
    n_generations = 8
    n_timepoints_per_gen = 20

    rows = []
    time = 0.0

    for exp_id in range(n_experiments):
        # Each experiment has different parameter values
        vio_expression = 0.5 + exp_id * 1.0
        mec_concentration = exp_id * 2.0

        for seed in range(n_seeds):
            seed_value = seed * 1000

            for gen in range(n_generations):
                for t_idx in range(n_timepoints_per_gen):
                    time_in_gen = t_idx / n_timepoints_per_gen

                    # Mass grows exponentially within generation
                    base_mass = 1.0 * (2.0**time_in_gen)
                    mass_noise = rng.normal(0, 0.05)
                    dry_mass = base_mass * (1 + mass_noise)

                    # DNA mass
                    if 0.3 < time_in_gen < 0.7:
                        dna_mass = 0.03 * (1 + (time_in_gen - 0.3) / 0.4)
                    else:
                        dna_mass = 0.03 if time_in_gen < 0.3 else 0.06

                    # Growth rate with cell cycle variation
                    growth_rate = 0.01 * (1 + 0.2 * np.sin(2 * np.pi * time_in_gen))

                    # Transcriptome influenced by vio_expression
                    transcriptome = rng.poisson(100 * (1 + 0.1 * vio_expression), size=10).astype(float).tolist()

                    # Fluxes influenced by mecillinam
                    base_flux = max(1.0, 10.0 * (1 - 0.03 * mec_concentration))
                    fluxes = rng.exponential(base_flux, size=5).tolist()

                    row = {
                        "experiment_id": exp_id,
                        "variant": f"vio_{vio_expression:.1f}_mec_{mec_concentration:.1f}",
                        "lineage_seed": seed_value,
                        "generation": gen,
                        "agent_id": f"agent_{seed}_{gen}_{t_idx}",
                        "time": time,
                        "listeners__mass__dry_mass": dry_mass,
                        "listeners__mass__cell_mass": dry_mass * 1.3,
                        "listeners__mass__dna_mass": dna_mass,
                        "listeners__mass__protein_mass": dry_mass * 0.5,
                        "listeners__fba_results__growth": growth_rate,
                        "listeners__rna_counts__mRNA_cistron_counts": json.dumps(transcriptome),
                        "listeners__fba_results__base_reaction_fluxes": json.dumps(fluxes),
                        "_param_vio_expression": vio_expression,
                        "_param_mec_concentration": mec_concentration,
                    }
                    rows.append(row)
                    time += 1.0

    return pl.DataFrame(rows)


@pytest.fixture
def parameter_output_samples(
    synthetic_simulation_dataframe: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract parameter-output sample pairs for sensitivity analysis.

    Returns:
        Tuple of (X, Y) where X is inputs and Y is outputs
    """
    df = synthetic_simulation_dataframe

    # Group by experiment to get parameter -> output mapping
    grouped = df.group_by("experiment_id").agg([
        pl.col("_param_vio_expression").first(),
        pl.col("_param_mec_concentration").first(),
        pl.col("listeners__mass__dry_mass").mean().alias("mean_mass"),
        pl.col("listeners__fba_results__growth").mean().alias("mean_growth"),
    ])

    X = np.column_stack([
        grouped["_param_vio_expression"].to_numpy(),
        np.zeros(len(grouped)),  # vio_trl_eff placeholder
        grouped["_param_mec_concentration"].to_numpy(),
    ])

    Y = np.column_stack([
        grouped["mean_mass"].to_numpy(),
        grouped["mean_growth"].to_numpy(),
    ])

    return X, Y


# =============================================================================
# Input Parameter Fixtures
# =============================================================================


@pytest.fixture
def vio_params():
    """Sample VioPathwayParams for testing."""
    from uq import VioPathwayParams

    return VioPathwayParams(
        enabled=True,
        induction_gen=1,
        expression=2.5,
        translation_efficiency=1.2,
        condition="basal",
    )


@pytest.fixture
def mecillinam_params():
    """Sample MecillinamParams for testing."""
    from uq import MecillinamParams

    return MecillinamParams(
        times=[0.0, 3600.0],
        concentrations=[0.0, 5.0],
        knockouts=["murG"],
    )


@pytest.fixture
def knockout_params():
    """Sample GeneKnockoutParams for testing."""
    from uq import GeneKnockoutParams

    return GeneKnockoutParams(
        gene_deletions=["lacZ", "galK"],
        translation_knockouts=["murG"],
    )


@pytest.fixture
def input_parameter_space():
    """InputParameterSpace configured for testing."""
    from uq import InputParameterSpaceVecoli

    return InputParameterSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.0, 5.0),
        vio_trl_eff_bounds=(0.0, 2.0),
        mecillinam_conc_bounds=(0.0, 10.0),
    )


# =============================================================================
# Aggregation Fixtures
# =============================================================================


@pytest.fixture
def aggregated_uniform(synthetic_simulation_dataframe: pl.DataFrame):
    """AggregatedOutput for uniform strategy."""
    from uq import AggregatedOutput

    df = synthetic_simulation_dataframe
    mean = (
        df.select([
            "listeners__mass__dry_mass",
            "listeners__fba_results__growth",
        ])
        .mean()
        .to_numpy()
        .flatten()
    )

    std = (
        df.select([
            "listeners__mass__dry_mass",
            "listeners__fba_results__growth",
        ])
        .std()
        .to_numpy()
        .flatten()
    )

    return AggregatedOutput(
        mean=mean,
        std=std,
        n_samples=len(df),
        groups=None,
    )


@pytest.fixture
def aggregated_by_generation(synthetic_simulation_dataframe: pl.DataFrame):
    """AggregatedOutput stratified by generation."""
    from uq import AggregatedOutput

    df = synthetic_simulation_dataframe
    by_gen = (
        df.group_by("generation")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__fba_results__growth").mean().alias("growth_mean"),
            pl.col("listeners__fba_results__growth").std().alias("growth_std"),
            pl.len().alias("n"),
        ])
        .sort("generation")
    )

    return AggregatedOutput(
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


@pytest.fixture
def aggregated_by_seed(synthetic_simulation_dataframe: pl.DataFrame):
    """AggregatedOutput stratified by lineage seed."""
    from uq import AggregatedOutput

    df = synthetic_simulation_dataframe
    by_seed = (
        df.group_by("lineage_seed")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__fba_results__growth").mean().alias("growth_mean"),
            pl.col("listeners__fba_results__growth").std().alias("growth_std"),
            pl.len().alias("n"),
        ])
        .sort("lineage_seed")
    )

    return AggregatedOutput(
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


# =============================================================================
# Sensitivity Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_sobol_indices():
    """Sample SobolIndices for testing."""
    from uq import SobolIndices

    return SobolIndices(
        first_order=np.array([0.45, 0.25, 0.20]),
        total_order=np.array([0.55, 0.30, 0.25]),
        second_order=None,
        parameter_names=["vio_expression", "vio_trl_eff", "mecillinam_concentration"],
    )


# =============================================================================
# Real Data Aggregation Fixtures
# =============================================================================


@pytest.fixture
def real_aggregated_uniform(real_simulation_dataframe):
    """AggregatedOutput for uniform strategy using real data."""
    from uq import AggregatedOutput

    if real_simulation_dataframe is None:
        pytest.skip("Real simulation data not available")

    df = real_simulation_dataframe
    cols = ["listeners__mass__dry_mass", "listeners__mass__growth"]

    mean = df.select(cols).mean().to_numpy().flatten()
    std = df.select(cols).std().to_numpy().flatten()

    return AggregatedOutput(
        mean=mean,
        std=std,
        n_samples=len(df),
        groups=None,
    )


@pytest.fixture
def real_aggregated_by_generation(real_simulation_dataframe):
    """AggregatedOutput stratified by generation using real data."""
    from uq import AggregatedOutput

    if real_simulation_dataframe is None:
        pytest.skip("Real simulation data not available")

    df = real_simulation_dataframe
    by_gen = (
        df.group_by("generation")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__mass__growth").mean().alias("growth_mean"),
            pl.col("listeners__mass__growth").std().alias("growth_std"),
            pl.len().alias("n"),
        ])
        .sort("generation")
    )

    return AggregatedOutput(
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


@pytest.fixture
def real_aggregated_by_seed(real_simulation_dataframe):
    """AggregatedOutput stratified by lineage seed using real data."""
    from uq import AggregatedOutput

    if real_simulation_dataframe is None:
        pytest.skip("Real simulation data not available")

    df = real_simulation_dataframe
    by_seed = (
        df.group_by("lineage_seed")
        .agg([
            pl.col("listeners__mass__dry_mass").mean().alias("mass_mean"),
            pl.col("listeners__mass__dry_mass").std().alias("mass_std"),
            pl.col("listeners__mass__growth").mean().alias("growth_mean"),
            pl.col("listeners__mass__growth").std().alias("growth_std"),
            pl.len().alias("n"),
        ])
        .sort("lineage_seed")
    )

    return AggregatedOutput(
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


@pytest.fixture
def real_trajectory(real_simulation_dataframe) -> np.ndarray:
    """
    Extract a trajectory from real data for Koopman analysis.

    Returns:
        Array of shape (n_timesteps, n_observables)
    """
    if real_simulation_dataframe is None:
        pytest.skip("Real simulation data not available")

    df = real_simulation_dataframe

    # Get a single cell's trajectory (first lineage_seed, generation 1, first agent)
    cell = df.filter((pl.col("lineage_seed") == df["lineage_seed"].min()) & (pl.col("generation") == 1)).sort("time")

    if len(cell) < 10:
        pytest.skip("Not enough data points for trajectory")

    # Extract observables
    trajectory = cell.select([
        "listeners__mass__dry_mass",
        "listeners__mass__growth",
        "listeners__mass__dna_mass",
    ]).to_numpy()

    return trajectory


# =============================================================================
# Markers for Test Categories
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "milestone: marks tests that verify specific milestone requirements")
    config.addinivalue_line("markers", "e2e: marks end-to-end integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
    config.addinivalue_line("markers", "slow: marks tests that take a long time to run")
    config.addinivalue_line("markers", "real_data: marks tests that use real simulation data")
