"""
Tests for uq.pipeline — the full RFC006 UQ pipeline.

Covers:
- aggregate_timeseries() (strategies 1-3)
- get_variance_decomposition()
- _split_multi_output_sobol()
- Strategy4Wrapper
- run_phase1()
- compute_strategy4_sobol()
- execute_pipeline()
- PipelineResult.export() / .from_export()
- UqProfile validation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from libuq import AggregatedOutput, SobolIndices, XSpaceVecoli
from libuq.pipeline.models import PipelineResult, StratificationLens, UqProfile
from libuq.pipeline.workflow import (
    AggregationResult,
    Strategy4Wrapper,
    _split_multi_output_sobol,
    aggregate_timeseries,
    get_variance_decomposition,
)
from libuq.sensitivity import PCESurrogate

# =============================================================================
# Helpers
# =============================================================================

OBSERVABLE_COLUMNS = ["listeners__mass__dry_mass", "listeners__fba_results__growth"]


def _make_mock_sobol(n_params=3, param_names=None):
    """Create a mock SobolIndices."""
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    return SobolIndices(
        first_order=np.random.rand(n_params),
        total_order=np.random.rand(n_params),
        parameter_names=param_names,
    )


def _make_mock_surrogate(n_params=3):
    """Create a mock PCESurrogate."""
    return PCESurrogate(
        coefficients=np.random.rand(6),
        multi_indices=np.zeros((6, n_params), dtype=int),
        basis_type="legendre",
        polynomial_order=2,
        input_dim=n_params,
        output_dim=1,
        r_squared=0.95,
        input_bounds=np.array([[0.0, 5.0], [0.0, 2.0], [0.0, 10.0]])[:n_params],
    )


def _make_mock_multi_sobol(n_bins=5, n_params=3, param_names=None):
    """Create a mock multi-output SobolIndices (2D arrays)."""
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    return SobolIndices(
        first_order=np.random.rand(n_bins, n_params),
        total_order=np.random.rand(n_bins, n_params),
        parameter_names=param_names,
    )


class SyntheticBulkWrapper:
    """Deterministic wrapper for Phase 1 testing: params → scalar output."""

    def __init__(self, param_names: list[str]):
        self.param_names = param_names

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[0] * 2.0 + x[1] * 0.5 + x[2] * 0.1])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self(x) for x in X])


# =============================================================================
# Tests: aggregate_timeseries (Step 3)
# =============================================================================


class TestAggregateTimeseries:
    def test_returns_aggregation_result(self, synthetic_simulation_dataframe):
        result = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        assert isinstance(result, AggregationResult)

    def test_uniform_strategy(self, synthetic_simulation_dataframe):
        result = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        uniform = result.uniform

        assert isinstance(uniform, AggregatedOutput)
        assert uniform.mean.shape == (len(OBSERVABLE_COLUMNS),)
        assert uniform.std.shape == (len(OBSERVABLE_COLUMNS),)
        assert uniform.n_samples == len(synthetic_simulation_dataframe)
        assert uniform.groups is None

    def test_generation_strategy(self, synthetic_simulation_dataframe):
        result = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        gen = result.generation

        assert isinstance(gen, AggregatedOutput)
        n_gens = synthetic_simulation_dataframe["generation"].n_unique()
        assert gen.mean.shape == (n_gens, len(OBSERVABLE_COLUMNS))
        assert gen.std.shape == (n_gens, len(OBSERVABLE_COLUMNS))
        assert gen.groups is not None
        assert len(gen.groups) == n_gens

    def test_seed_strategy(self, synthetic_simulation_dataframe):
        result = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        seed = result.seed

        assert isinstance(seed, AggregatedOutput)
        n_seeds = synthetic_simulation_dataframe["lineage_seed"].n_unique()
        assert seed.mean.shape == (n_seeds, len(OBSERVABLE_COLUMNS))
        assert seed.groups is not None
        assert len(seed.groups) == n_seeds

    def test_means_are_finite(self, synthetic_simulation_dataframe):
        result = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        assert np.all(np.isfinite(result.uniform.mean))
        assert np.all(np.isfinite(result.generation.mean))
        assert np.all(np.isfinite(result.seed.mean))


# =============================================================================
# Tests: get_variance_decomposition (Step 4)
# =============================================================================


class TestVarianceDecomposition:
    def test_returns_dict_with_expected_keys(self, synthetic_simulation_dataframe):
        agg = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        decomp = get_variance_decomposition(agg)

        assert isinstance(decomp, dict)
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp

    def test_fractions_are_bounded(self, synthetic_simulation_dataframe):
        agg = aggregate_timeseries(synthetic_simulation_dataframe, OBSERVABLE_COLUMNS)
        decomp = get_variance_decomposition(agg)

        gen_frac = decomp["generation_fraction"]
        seed_frac = decomp["seed_fraction"]

        assert np.all(gen_frac >= 0)
        assert np.all(gen_frac <= 1)
        assert np.all(seed_frac >= 0)
        assert np.all(seed_frac <= 1)


# =============================================================================
# Tests: _split_multi_output_sobol
# =============================================================================


class TestSplitMultiOutputSobol:
    def test_single_output_returns_list_of_one(self):
        sobol = SobolIndices(
            first_order=np.array([0.5, 0.3, 0.2]),
            total_order=np.array([0.6, 0.35, 0.25]),
            parameter_names=["a", "b", "c"],
        )
        result = _split_multi_output_sobol(sobol)
        assert len(result) == 1
        assert result[0] is sobol

    def test_multi_output_splits_correctly(self):
        n_outputs = 4
        n_params = 3
        rng = np.random.default_rng(42)
        fo = rng.random((n_outputs, n_params))
        to = rng.random((n_outputs, n_params))
        sobol = SobolIndices(
            first_order=fo,
            total_order=to,
            parameter_names=["a", "b", "c"],
        )
        result = _split_multi_output_sobol(sobol)

        assert len(result) == n_outputs
        for i, s in enumerate(result):
            np.testing.assert_array_equal(s.first_order, fo[i])
            np.testing.assert_array_equal(s.total_order, to[i])
            assert s.parameter_names == ["a", "b", "c"]
            assert s.output_names == [f"stage_{i}"]


# =============================================================================
# Tests: Strategy4Wrapper
# =============================================================================


class TestStrategy4Wrapper:
    @pytest.fixture
    def mock_koopman(self):
        """A mock KoopmanCellCycleVariable that returns linearly spaced θ."""
        from libuq.pipeline.models import CellCycleVariable

        class MockKoopman:
            def __init__(self):
                self.observable_columns = ["obs_0"]

            def compute(self, df: pl.DataFrame) -> CellCycleVariable:
                n = len(df)
                return CellCycleVariable(
                    values=np.linspace(0, 1, n),
                    normalized=True,
                    variable_name="mock_theta",
                )

        return MockKoopman()

    def test_call_returns_per_stage_means(self, mock_koopman):
        n_bins = 5
        rng = np.random.default_rng(0)

        def base_wrapper(params):
            return rng.random(100)

        wrapper = Strategy4Wrapper(
            base_wrapper=base_wrapper,
            koopman_cc=mock_koopman,
            n_bins=n_bins,
        )
        result = wrapper(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (n_bins,)
        assert np.all(np.isfinite(result))

    def test_evaluate_batch(self, mock_koopman):
        n_bins = 5
        rng = np.random.default_rng(0)

        def base_wrapper(params):
            return rng.random(100)

        wrapper = Strategy4Wrapper(
            base_wrapper=base_wrapper,
            koopman_cc=mock_koopman,
            n_bins=n_bins,
        )
        X = np.random.rand(10, 3)
        result = wrapper.evaluate_batch(X)
        assert result.shape == (10, n_bins)

    def test_multi_observable_output(self, mock_koopman):
        n_bins = 5
        n_obs = 3
        rng = np.random.default_rng(0)

        def base_wrapper(params):
            return rng.random((100, n_obs))

        wrapper = Strategy4Wrapper(
            base_wrapper=base_wrapper,
            koopman_cc=mock_koopman,
            n_bins=n_bins,
            observable_columns=[f"obs_{i}" for i in range(n_obs)],
        )
        result = wrapper(np.array([1.0, 2.0, 3.0]))
        assert result.shape == (n_bins * n_obs,)

    def test_custom_timeseries_to_dataframe(self, mock_koopman):
        n_bins = 5
        rng = np.random.default_rng(0)

        def base_wrapper(params):
            return rng.random(50)

        custom_called = [False]

        def custom_to_df(raw):
            custom_called[0] = True
            return pl.DataFrame({"obs_0": raw})

        wrapper = Strategy4Wrapper(
            base_wrapper=base_wrapper,
            koopman_cc=mock_koopman,
            n_bins=n_bins,
            timeseries_to_dataframe=custom_to_df,
        )
        wrapper(np.array([1.0, 2.0, 3.0]))
        assert custom_called[0]


# =============================================================================
# Tests: run_phase1 (Steps 5a-7a) — mocked PyTUQ
# =============================================================================


class TestRunPhase1:
    """Test run_phase1 orchestration with mocked SensitivityAnalyzer."""

    def test_returns_sobol_and_surrogate(self, input_parameter_space):
        from libuq.pipeline.workflow import run_phase1

        mock_sobol = _make_mock_sobol(n_params=3, param_names=input_parameter_space.parameter_names)
        mock_surrogate = _make_mock_surrogate(n_params=3)

        with patch("uq.pipeline.workflow.SensitivityAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_with_pce.return_value = (mock_sobol, mock_surrogate)

            wrapper = SyntheticBulkWrapper(input_parameter_space.parameter_names)
            sobol, surrogate, morris = run_phase1(
                param_space=input_parameter_space,
                simulation_func=wrapper,
                polynomial_order=2,
                n_samples=30,
            )

        assert isinstance(sobol, SobolIndices)
        assert isinstance(surrogate, PCESurrogate)
        assert len(sobol.parameter_names) == input_parameter_space.n_parameters
        assert morris is None  # no prescreen_config provided

    def test_export_path(self, input_parameter_space, tmp_path):
        from libuq.pipeline.workflow import run_phase1

        mock_sobol = _make_mock_sobol(n_params=3)
        mock_surrogate = MagicMock(spec=PCESurrogate)

        with patch("uq.pipeline.workflow.SensitivityAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_with_pce.return_value = (mock_sobol, mock_surrogate)

            wrapper = SyntheticBulkWrapper(input_parameter_space.parameter_names)
            run_phase1(
                param_space=input_parameter_space,
                simulation_func=wrapper,
                polynomial_order=2,
                n_samples=30,
                export_path=tmp_path,
            )

        mock_surrogate.export.assert_called_once_with(tmp_path / "population_surrogate")


# =============================================================================
# Tests: compute_strategy4_sobol (Step 7b) — mocked PyTUQ
# =============================================================================


class TestComputeStrategy4Sobol:
    def test_returns_per_stage_sobol(self, input_parameter_space):
        from libuq.pipeline.workflow import compute_strategy4_sobol

        n_bins = 5
        mock_multi_sobol = _make_mock_multi_sobol(
            n_bins=n_bins, n_params=3, param_names=input_parameter_space.parameter_names
        )
        mock_surrogate = _make_mock_surrogate(n_params=3)

        # Mock the SensitivityAnalyzer that compute_strategy4_sobol creates
        with patch("uq.pipeline.workflow.SensitivityAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_with_pce.return_value = (mock_multi_sobol, mock_surrogate)

            # A wrapper-like object (doesn't matter — mocked)
            f_stage4 = MagicMock()

            per_stage, surrogate = compute_strategy4_sobol(
                param_space=input_parameter_space,
                f_stage4=f_stage4,
                polynomial_order=2,
                n_samples=30,
            )

        assert isinstance(per_stage, list)
        assert len(per_stage) == n_bins
        for s in per_stage:
            assert isinstance(s, SobolIndices)
            assert len(s.parameter_names) == 3

        assert isinstance(surrogate, PCESurrogate)


# =============================================================================
# Tests: execute_pipeline (full orchestration) — mocked PyTUQ
# =============================================================================


class TestExecutePipeline:
    @pytest.fixture
    def _mock_phases(self, input_parameter_space, synthetic_simulation_dataframe):
        """Patch run_phase1, run_phase2, ParameterDataset, and data loading to avoid PyTUQ/DuckDB calls."""
        n_params = input_parameter_space.n_parameters
        param_names = input_parameter_space.parameter_names

        mock_sobol_bulk = _make_mock_sobol(n_params, param_names)
        mock_surrogate_bulk = _make_mock_surrogate(n_params)

        n_bins = 5
        mock_cc_sobols = [_make_mock_sobol(n_params, param_names) for _ in range(n_bins)]
        mock_surrogate_cc = _make_mock_surrogate(n_params)

        # Mock the DuckDB data loading path (Steps 1-2)
        mock_outputs = MagicMock()
        mock_outputs.higher_order_properties = {col: np.random.rand(100) for col in OBSERVABLE_COLUMNS}
        mock_extractor = MagicMock()
        mock_extractor.extract_all.return_value = mock_outputs
        mock_extractor.load_timeseries.return_value = synthetic_simulation_dataframe

        # Mock ParameterDataset so execute_pipeline can build param_space from sim_data_path
        mock_param_dataset = MagicMock()
        mock_param_dataset.to_parameter_space.return_value = input_parameter_space

        with (
            patch("uq.pipeline.workflow.run_phase1", return_value=(mock_sobol_bulk, mock_surrogate_bulk, None)),
            patch("uq.pipeline.workflow.run_phase2", return_value=(mock_cc_sobols, mock_surrogate_cc, None)),
            patch("ecoli.library.parquet_emitter.create_duckdb_conn", return_value=MagicMock()),
            patch("ecoli.library.parquet_emitter.dataset_sql", return_value=("hist", "conf", "succ")),
            patch("uq.outputs.OutputExtractor", return_value=mock_extractor),
            patch("uq.pipeline.param_loader.ParameterDataset", return_value=mock_param_dataset),
        ):
            yield {
                "n_bins": n_bins,
                "sobol_bulk": mock_sobol_bulk,
                "surrogate_bulk": mock_surrogate_bulk,
                "cc_sobols": mock_cc_sobols,
                "surrogate_cc": mock_surrogate_cc,
            }

    def test_returns_pipeline_result(
        self,
        _mock_phases,
    ):
        from libuq.pipeline.workflow import execute_pipeline

        result = execute_pipeline(
            sim_data_path="/tmp/sim_data/simData.cPickle",
            simulation_func=MagicMock(),
            experiment_ids="test_experiment",
            sim_base_path="/tmp/sims",
            observable_columns=OBSERVABLE_COLUMNS,
            n_bins=5,
            polynomial_order=2,
            n_samples=30,
        )

        assert isinstance(result, PipelineResult)
        assert isinstance(result.population, UqProfile)
        assert isinstance(result.cell_cycle, UqProfile)

    def test_population_profile(
        self,
        _mock_phases,
    ):
        from libuq.pipeline.workflow import execute_pipeline

        result = execute_pipeline(
            sim_data_path="/tmp/sim_data/simData.cPickle",
            simulation_func=MagicMock(),
            experiment_ids="test_experiment",
            sim_base_path="/tmp/sims",
            observable_columns=OBSERVABLE_COLUMNS,
        )

        assert result.population.stratification == StratificationLens.POPULATION
        assert len(result.population.sobol_indices) == 1

    def test_cell_cycle_profile(
        self,
        _mock_phases,
    ):
        from libuq.pipeline.workflow import execute_pipeline

        result = execute_pipeline(
            sim_data_path="/tmp/sim_data/simData.cPickle",
            simulation_func=MagicMock(),
            experiment_ids="test_experiment",
            sim_base_path="/tmp/sims",
            observable_columns=OBSERVABLE_COLUMNS,
        )

        assert result.cell_cycle.stratification == StratificationLens.CELL_CYCLE
        assert len(result.cell_cycle.sobol_indices) == _mock_phases["n_bins"]


# =============================================================================
# Tests: PipelineResult serialization
# =============================================================================


class TestPipelineResultSerialization:
    @pytest.fixture
    def sample_pipeline_result(self, sample_sobol_indices):
        """Create a minimal PipelineResult for serialization tests."""
        surrogate = PCESurrogate(
            coefficients=np.array([1.0, 0.5, 0.3]),
            multi_indices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            basis_type="legendre",
            polynomial_order=2,
            input_dim=3,
            output_dim=1,
            r_squared=0.95,
            input_bounds=np.array([[0.0, 5.0], [0.0, 2.0], [0.0, 10.0]]),
        )

        cc_sobols = [
            SobolIndices(
                first_order=np.array([0.4, 0.3, 0.2]),
                total_order=np.array([0.5, 0.35, 0.25]),
                parameter_names=["vio_expression", "vio_trl_eff", "mecillinam_concentration"],
            )
            for _ in range(5)
        ]

        return PipelineResult(
            population=UqProfile(
                stratification=StratificationLens.POPULATION,
                sobol_indices=[sample_sobol_indices],
                surrogate=surrogate,
            ),
            cell_cycle=UqProfile(
                stratification=StratificationLens.CELL_CYCLE,
                sobol_indices=cc_sobols,
                surrogate=surrogate,
            ),
        )

    def test_export_creates_files(self, sample_pipeline_result, tmp_path):
        sample_pipeline_result.export(tmp_path / "output")
        out = tmp_path / "output"
        assert (out / "metadata.json").exists()
        assert (out / "population_surrogate").exists()
        assert (out / "population_sobol").exists()
        for i in range(5):
            assert (out / f"cell_cycle_sobol_stage_{i}").exists()

    def test_round_trip(self, sample_pipeline_result, tmp_path):
        out = tmp_path / "output"
        sample_pipeline_result.export(out)
        loaded = PipelineResult.from_export(out)

        assert loaded.population.stratification == StratificationLens.POPULATION
        assert loaded.cell_cycle.stratification == StratificationLens.CELL_CYCLE
        assert len(loaded.population.sobol_indices) == 1
        assert len(loaded.cell_cycle.sobol_indices) == 5

        # Verify Sobol values survived round-trip
        np.testing.assert_array_almost_equal(
            loaded.population.sobol_indices[0].first_order,
            sample_pipeline_result.population.sobol_indices[0].first_order,
        )

    def test_metadata_content(self, sample_pipeline_result, tmp_path):
        import json

        out = tmp_path / "output"
        sample_pipeline_result.export(out)
        meta = json.loads((out / "metadata.json").read_text())

        assert meta["n_cell_cycle_stages"] == 5
        assert meta["population_stratification"] == "population"
        assert meta["cell_cycle_stratification"] == "cell_cycle"


# =============================================================================
# Tests: UqProfile validation
# =============================================================================


class TestUqProfile:
    def test_population_rejects_multiple_sobol(self, sample_sobol_indices):
        surrogate = PCESurrogate(
            coefficients=np.array([1.0]),
            multi_indices=np.array([[0, 0, 0]]),
            input_dim=3,
            output_dim=1,
        )
        with pytest.raises(ValueError, match="population level"):
            UqProfile(
                stratification=StratificationLens.POPULATION,
                sobol_indices=[sample_sobol_indices, sample_sobol_indices],
                surrogate=surrogate,
            )

    def test_cell_cycle_accepts_multiple_sobol(self, sample_sobol_indices):
        surrogate = PCESurrogate(
            coefficients=np.array([1.0]),
            multi_indices=np.array([[0, 0, 0]]),
            input_dim=3,
            output_dim=1,
        )
        profile = UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=[sample_sobol_indices] * 5,
            surrogate=surrogate,
        )
        assert len(profile.sobol_indices) == 5

    def test_population_accepts_single_sobol(self, sample_sobol_indices):
        surrogate = _make_mock_surrogate()
        profile = UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[sample_sobol_indices],
            surrogate=surrogate,
        )
        assert len(profile.sobol_indices) == 1
