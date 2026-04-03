"""
Pytest fixtures for UQ package tests.

These fixtures provide both synthetic data and real simulation data for testing.
Real data is loaded from api_simulation_default experiment using uq.inputs.load_dataset().
"""

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


# Default path for real simulation data
REAL_DATA_OUTDIR = Path("/Users/alexanderpatrie/sms/sms-api/artifacts/sims")


def _load_real_dataset_safe(
    experiment_id: str = "api_simulation_default",
    max_files: int | None = None,
) -> pl.DataFrame | None:
    """
    Load real simulation data using uq.inputs.load_dataset.

    Args:
        experiment_id: Experiment identifier
        max_files: Maximum number of parquet files to load (for faster testing)

    Returns:
        DataFrame or None if data not available
    """
    from uq.inputs import load_dataset

    repo_root = _get_repo_root()
    observables_file = repo_root / "baseline_observables.json"

    # Check if data directory exists
    if not REAL_DATA_OUTDIR.exists():
        return None

    try:
        # Load observables from JSON
        if observables_file.exists():
            with open(observables_file) as f:
                obs = [col for col in json.load(f) if col.startswith("listener")]
            obs.append("time")
        else:
            # Fallback to standard columns
            obs = REAL_DATA_COLUMNS

        # Load dataset using uq.inputs.load_dataset
        df = load_dataset(
            experiment_id=experiment_id,
            outdir_root=REAL_DATA_OUTDIR,
            observables=obs,
        )

        # Handle both DataFrame and LazyFrame
        if hasattr(df, "collect"):
            df = df.collect()

        # Filter to Float64 columns only (numeric observables)
        schema = df.schema
        float_observables = []
        for obs_i in obs:
            if obs_i in schema:
                coltype = schema[obs_i]
                if coltype == pl.Float64:
                    float_observables.append(obs_i)

        # Ensure we have required columns (including all metadata from hive partitioning)
        required = ["time", "lineage_seed", "generation", "variant", "agent_id"]
        for col in required:
            if col in schema and col not in float_observables:
                float_observables.append(col)

        df = df.select([c for c in float_observables if c in schema])

        if len(df) == 0:
            return None

        return df

    except Exception as e:
        # Log the error for debugging but don't fail
        print(f"Warning: Could not load real data: {e}")
        return None


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
def sim_data_params():
    """Sample SimDataParameter specs for testing (equivalent to former vio/mecillinam params)."""
    from uq.pipeline.models import SimDataParameter

    return [
        SimDataParameter(
            name="vio_expression", attr_path="process.transcription.new_gene_expression_baselines", bounds=(0.0, 5.0)
        ),
        SimDataParameter(
            name="vio_trl_eff", attr_path="process.transcription.translation_efficiencies_by_gene", bounds=(0.0, 2.0)
        ),
        SimDataParameter(
            name="mecillinam_concentration", attr_path="process.metabolism.secretion_penalty_coeff", bounds=(0.0, 10.0)
        ),
    ]


@pytest.fixture
def input_parameter_space():
    """InputParameterSpace configured for testing."""
    from uq.inputs import XSpaceVecoli
    from uq.pipeline.models import SimDataParameter

    return XSpaceVecoli(
        parameters=[
            SimDataParameter(
                name="vio_expression",
                attr_path="process.transcription.new_gene_expression_baselines",
                bounds=(0.0, 5.0),
            ),
            SimDataParameter(
                name="vio_trl_eff",
                attr_path="process.transcription.translation_efficiencies_by_gene",
                bounds=(0.0, 2.0),
            ),
            SimDataParameter(
                name="mecillinam_concentration",
                attr_path="process.metabolism.secretion_penalty_coeff",
                bounds=(0.0, 10.0),
            ),
        ]
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
# PCE Fixtures
# =============================================================================


@pytest.fixture
def pce_parameters():
    """Sample Parameter list for PCE testing."""
    from uq.models import Parameter

    return [
        Parameter(name="p1", bounds=(0.0, 1.0), default=0.5, step=0.1, description="Parameter 1"),
        Parameter(name="p2", bounds=(0.0, 2.0), default=1.0, step=0.2, description="Parameter 2"),
        Parameter(name="p3", bounds=(-1.0, 1.0), default=0.0, step=0.1, description="Parameter 3"),
    ]


@pytest.fixture
def pce_sample_data(rng):
    """Generate sample X, Y data for PCE fitting."""
    n_samples = 100
    n_params = 3

    X = rng.uniform(-1, 1, (n_samples, n_params))
    # Known function: y = 1 + 2*x1 + 0.5*x2^2 + 0.1*x1*x3
    Y = 1 + 2 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * X[:, 0] * X[:, 2]

    return X, Y


@pytest.fixture
def pce_surrogate_config(pce_parameters):
    """Sample PCESurrogateConfig for testing."""
    from uq.models import PCESurrogateConfig

    return PCESurrogateConfig(
        parameters=pce_parameters,
        n_samples=100,
        polynomial_order=2,
    )


@pytest.fixture
def pce_fit_result(pce_sample_data):
    """Pre-fitted PCE result for testing."""
    from uq.pce.surrogate import fit_pce_coefficients

    X, Y = pce_sample_data
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])

    return fit_pce_coefficients(X, Y, polynomial_order=2, bounds=bounds)


@pytest.fixture
def simple_stochastic_function(rng):
    """Simple stochastic function for testing."""

    def f(x):
        # Mean = sum of inputs, with small noise
        return np.sum(x) + rng.normal(0, 0.01)

    return f


@pytest.fixture
def deterministic_function():
    """Simple deterministic function for testing."""

    def f(x):
        return x[0] + 0.5 * x[1] + 0.25 * x[2]

    return f


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
    config.addinivalue_line("markers", "pce: marks tests related to PCE functionality")
    config.addinivalue_line("markers", "pipeline: marks tests for the RFC006 pipeline module")
