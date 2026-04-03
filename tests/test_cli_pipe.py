"""
Integration test for uq.cli.pipe() end-to-end — REAL DATA, no mocking.

Uses real simulation data from vEcoli/api_integration/sims for
the full RFC006 pipeline: data loading → aggregation → variance decomposition
→ Phase 1 (Morris + PCE + Sobol) → Phase 2 (Koopman + per-stage Sobol)
→ PipelineResult.

Low n_samples/n_trajectories to keep runtime and memory manageable.
"""

from pathlib import Path

import numpy as np
import pytest

from libuq import SobolIndices
from libuq.pipeline.models import PipelineResult, StratificationLens, UqProfile
from libuq.sensitivity import CellCycleRelevanceResult, MorrisIndices, PCESurrogate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIM_BASE_PATH = Path("/Users/alexanderpatrie/sms/vEcoli/api_integration/sims")
EXPERIMENT_IDS = ["mecillinam", "api_simulation_default", "test_violacein_with_metabolism"]
N_BINS = 5

# Explicit observable columns — avoids DuckDB parse errors from columns
# with brackets (e.g. "TRANS-CPLX-201[m]") when querying all columns.
OBSERVABLE_COLUMNS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__volume",
    "listeners__mass__growth",
]

# Keep these small so Phase 1/2 run fast without OOM
N_SAMPLES = 20
POLYNOMIAL_ORDER = 2
N_TRAJECTORIES = 10
N_TOP = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _skip_if_no_data():
    """Skip if the real sim data directory doesn't exist."""
    for exp_id in EXPERIMENT_IDS:
        pickle_path = SIM_BASE_PATH / exp_id / "parca" / "kb" / "simData.cPickle"
        if not pickle_path.exists():
            pytest.skip(f"simData.cPickle not found for {exp_id}: {pickle_path}")
        history_path = SIM_BASE_PATH / exp_id / "history"
        if not history_path.exists():
            pytest.skip(f"history dir not found for {exp_id}: {history_path}")


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline once and share across tests in this module.

    This avoids re-running the expensive Phase 1/2 for every test method.
    If data is missing, all tests using this fixture will be skipped.
    """
    for exp_id in EXPERIMENT_IDS:
        pickle_path = SIM_BASE_PATH / exp_id / "parca" / "kb" / "simData.cPickle"
        if not pickle_path.exists():
            pytest.skip(f"simData.cPickle not found for {exp_id}: {pickle_path}")
        history_path = SIM_BASE_PATH / exp_id / "history"
        if not history_path.exists():
            pytest.skip(f"history dir not found for {exp_id}: {history_path}")

    from libuq.pce.models import PCEParameterSelectionConfig
    from libuq.pipe import execute_pipeline

    prescreen_config = PCEParameterSelectionConfig(n_trajectories=N_TRAJECTORIES, n_top=N_TOP)
    return execute_pipeline(
        experiment_ids=EXPERIMENT_IDS,
        sim_base_path=str(SIM_BASE_PATH),
        observable_columns=OBSERVABLE_COLUMNS,
        prescreen_config=prescreen_config,
        n_bins=N_BINS,
        n_samples=N_SAMPLES,
        polynomial_order=POLYNOMIAL_ORDER,
    )


# ---------------------------------------------------------------------------
# Tests: Data Loading (Steps 1-2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.execute_pipeline
class TestInitializeData:
    """Tests for initialize_data — real sim data loading."""

    def test_loads_multi_experiment(self, _skip_if_no_data):
        from libuq.pipe import initialize_datasets

        ds = initialize_datasets(
            experiment_ids=EXPERIMENT_IDS,
            sim_base_path=str(SIM_BASE_PATH),
            observable_columns=OBSERVABLE_COLUMNS,
        )

        assert ds.parameter_space is not None
        assert ds.parameter_space.n_parameters > 0
        assert len(ds.y) > 0
        assert len(ds.observables) > 0
        assert len(ds.x) == len(EXPERIMENT_IDS)

    def test_loads_single_experiment(self, _skip_if_no_data):
        from libuq.pipe import initialize_datasets

        ds = initialize_datasets(
            experiment_ids="mecillinam",
            sim_base_path=str(SIM_BASE_PATH),
            observable_columns=OBSERVABLE_COLUMNS,
        )
        assert len(ds.x) == 1
        assert ds.experiment_ids == ["mecillinam"]
        assert len(ds.y) > 0


# ---------------------------------------------------------------------------
# Tests: Aggregation + Variance Decomposition (Steps 3-4)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.execute_pipeline
class TestAggregation:
    def test_aggregation_strategies(self, _skip_if_no_data):
        from libuq.pipe import initialize_datasets
        from libuq.pipeline.workflow import aggregate_timeseries, get_variance_decomposition

        ds = initialize_datasets(
            experiment_ids=EXPERIMENT_IDS,
            sim_base_path=str(SIM_BASE_PATH),
            observable_columns=OBSERVABLE_COLUMNS,
        )

        agg = aggregate_timeseries(ds.y, ds.observables)

        # Uniform strategy
        assert agg.uniform.mean.shape == (len(ds.observables),)
        assert np.all(np.isfinite(agg.uniform.mean))

        # Generation strategy
        assert agg.generation.groups is not None
        assert agg.generation.mean.ndim == 2

        # Seed strategy
        assert agg.seed.groups is not None

        # Variance decomposition
        decomp = get_variance_decomposition(agg)
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp


# ---------------------------------------------------------------------------
# Tests: Full Pipeline (Steps 1-7, real Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.execute_pipeline
class TestFullPipeline:
    """End-to-end tests against the real pipeline result."""

    def test_returns_pipeline_result(self, pipeline_result):
        assert isinstance(pipeline_result, PipelineResult)

    def test_population_profile(self, pipeline_result):
        pop = pipeline_result.population
        assert pop.stratification == StratificationLens.POPULATION
        assert len(pop.sobol_indices) == 1
        assert isinstance(pop.surrogate, PCESurrogate)

        sobol = pop.sobol_indices[0]
        assert len(sobol.parameter_names) > 0
        assert sobol.first_order.shape[0] > 0
        assert np.all(np.isfinite(sobol.first_order))
        assert np.all(np.isfinite(sobol.total_order))

    def test_cell_cycle_profile(self, pipeline_result):
        cc = pipeline_result.cell_cycle
        assert cc.stratification == StratificationLens.CELL_CYCLE
        assert len(cc.sobol_indices) >= 1  # at least 1 stage
        assert isinstance(cc.surrogate, PCESurrogate)

        for stage_sobol in cc.sobol_indices:
            assert isinstance(stage_sobol, SobolIndices)
            assert len(stage_sobol.parameter_names) > 0

    def test_variance_decomposition(self, pipeline_result):
        decomp = pipeline_result.variance_decomposition
        assert isinstance(decomp, dict)
        assert len(decomp) > 0
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp

    def test_aggregation_result(self, pipeline_result):
        from libuq.pipeline.workflow import AggregationResult

        assert isinstance(pipeline_result.aggregation, AggregationResult)

    def test_morris_indices(self, pipeline_result):
        morris = pipeline_result.morris_indices
        assert isinstance(morris, MorrisIndices)
        assert len(morris.parameter_names) > 0
        assert morris.mu_star.shape[0] > 0

    def test_cell_cycle_relevance(self, pipeline_result):
        relevance = pipeline_result.cell_cycle_relevance
        assert isinstance(relevance, CellCycleRelevanceResult)
        assert isinstance(relevance.residual_fractions, dict)

    def test_export_round_trip(self, pipeline_result, tmp_path):
        export_dir = tmp_path / "uq_export"
        pipeline_result.export(export_dir)

        # Verify export artifacts exist
        assert (export_dir / "metadata.json").exists()
        assert (export_dir / "population_surrogate").exists()
        assert (export_dir / "population_sobol").exists()
        assert (export_dir / "variance_decomposition.json").exists()
        assert (export_dir / "morris_indices").exists()

        # Round-trip
        loaded = PipelineResult.from_export(export_dir)
        assert loaded.population.stratification == StratificationLens.POPULATION
        assert loaded.cell_cycle.stratification == StratificationLens.CELL_CYCLE
        assert len(loaded.population.sobol_indices) == 1

        np.testing.assert_array_almost_equal(
            loaded.population.sobol_indices[0].first_order,
            pipeline_result.population.sobol_indices[0].first_order,
        )

        # Morris round-trip
        assert loaded.morris_indices is not None
        np.testing.assert_array_almost_equal(
            loaded.morris_indices.mu_star,
            pipeline_result.morris_indices.mu_star,
        )


# ---------------------------------------------------------------------------
# Tests: CLI Demo Command
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.execute_pipeline
class TestCliDemo:
    """Test the demo CLI command end-to-end with real data."""

    def test_cli_demo_invokes_pipeline(self, _skip_if_no_data):
        """Verify the demo CLI command runs the full pipeline."""
        from typer.testing import CliRunner

        from libuq.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["demo"])

        if result.exit_code != 0:
            print(result.output)
            if result.exception:
                import traceback

                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

        assert result.exit_code == 0
        assert "Pipeline complete" in result.output
