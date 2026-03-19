"""
End-to-end test for the `uq quantify` CLI command.

Exercises the full RFC006 pipeline via the CLI entrypoint:
  1. Generate precomputed samples (Stage 1)
  2. Run `quantify` with --precomputed-path (Stage 2)
  3. Verify PipelineResult structure, report output, and export artifacts

Uses real simulation data from vEcoli-private/api_integration/sims.
Tests are skipped when data is not available.
"""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from uq import SobolIndices
from uq.pipe import Pipeline
from uq.pipeline.models import PipelineResult, StratificationLens
from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices, PCESurrogate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIM_BASE_PATH = Path("/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims")
EXPERIMENT_IDS = ["mecillinam", "api_simulation_default", "test_violacein_with_metabolism"]

OBSERVABLE_COLUMNS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__volume",
    "listeners__mass__growth",
]

N_BINS = 5
N_SAMPLES = 20
POLYNOMIAL_ORDER = 2
N_TRAJECTORIES = 10
N_TOP = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_available() -> bool:
    for exp_id in EXPERIMENT_IDS:
        pickle_path = SIM_BASE_PATH / exp_id / "parca" / "kb" / "simData.cPickle"
        history_path = SIM_BASE_PATH / exp_id / "history"
        if not pickle_path.exists() or not history_path.exists():
            return False
    return True


skip_if_no_data = pytest.mark.skipif(
    not _data_available(),
    reason="Real simulation data not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def precomputed_cache_dir(tmp_path_factory) -> Path:
    """Generate precomputed samples (Stage 1) once per module."""
    if not _data_available():
        pytest.skip("Real simulation data not available")

    from uq.handlers import generate_samples

    cache_dir = tmp_path_factory.mktemp("uq_cache")
    generate_samples(
        experiment_ids=EXPERIMENT_IDS,
        sim_base_path=str(SIM_BASE_PATH),
        cache_dir=str(cache_dir),
        n_samples=N_SAMPLES,
        seed=42,
        observable_columns=OBSERVABLE_COLUMNS,
    )
    return cache_dir


@pytest.fixture(scope="module")
def quantify_pipeline(precomputed_cache_dir) -> Pipeline:
    """Run the full pipeline via handlers.pipeline() once per module."""
    from uq import handlers
    from uq.pce.models import PCEParameterSelectionConfig

    return handlers.pipeline(
        experiment_ids=EXPERIMENT_IDS,
        sim_base_path=str(SIM_BASE_PATH),
        observable_columns=OBSERVABLE_COLUMNS,
        n_bins=N_BINS,
        n_samples=N_SAMPLES,
        polynomial_order=POLYNOMIAL_ORDER,
        prescreen_config=PCEParameterSelectionConfig(
            n_trajectories=N_TRAJECTORIES,
            n_top=N_TOP,
        ),
        precomputed_path=str(precomputed_cache_dir),
        execute=True,
    )


# ---------------------------------------------------------------------------
# Tests: Precomputed Samples (Stage 1)
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestPrecomputedSamples:
    """Verify that generate-samples produces a valid cache."""

    def test_cache_files_exist(self, precomputed_cache_dir):
        assert (precomputed_cache_dir / "X.npy").exists()
        assert (precomputed_cache_dir / "Y.npy").exists()
        assert (precomputed_cache_dir / "metadata.json").exists()

    def test_cache_shapes(self, precomputed_cache_dir):
        X = np.load(precomputed_cache_dir / "X.npy")
        Y = np.load(precomputed_cache_dir / "Y.npy")
        assert X.shape[0] == N_SAMPLES
        assert Y.shape[0] == N_SAMPLES
        assert X.shape[1] > 0  # at least 1 parameter
        assert Y.shape[1] > 0  # at least 1 output

    def test_cache_metadata_has_parameter_names(self, precomputed_cache_dir):
        import json

        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        assert "parameter_names" in meta
        assert isinstance(meta["parameter_names"], list)
        assert len(meta["parameter_names"]) > 0

    def test_cache_metadata_has_bounds(self, precomputed_cache_dir):
        import json

        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        assert "bounds" in meta
        bounds = np.array(meta["bounds"])
        assert bounds.ndim == 2
        assert bounds.shape[1] == 2
        assert np.all(bounds[:, 0] < bounds[:, 1])

    def test_samples_within_bounds(self, precomputed_cache_dir):
        import json

        X = np.load(precomputed_cache_dir / "X.npy")
        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        bounds = np.array(meta["bounds"])
        # LHS samples should be within parameter bounds (with small tolerance)
        assert np.all(bounds[:, 0] - 1e-10 <= X)
        assert np.all(bounds[:, 1] + 1e-10 >= X)


# ---------------------------------------------------------------------------
# Tests: Full Pipeline Result (Stage 2)
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestQuantifyPipelineResult:
    """Verify PipelineResult from the quantify workflow."""

    def test_returns_pipeline_with_result(self, quantify_pipeline):
        assert quantify_pipeline.result is not None
        assert isinstance(quantify_pipeline.result, PipelineResult)

    def test_population_profile(self, quantify_pipeline):
        pop = quantify_pipeline.result.population
        assert pop.stratification == StratificationLens.POPULATION
        assert len(pop.sobol_indices) == 1
        assert isinstance(pop.surrogate, PCESurrogate)

        sobol = pop.sobol_indices[0]
        assert len(sobol.parameter_names) > 0
        assert sobol.first_order.shape[0] > 0
        assert np.all(np.isfinite(sobol.first_order))
        assert np.all(np.isfinite(sobol.total_order))

    def test_cell_cycle_profile(self, quantify_pipeline):
        cc = quantify_pipeline.result.cell_cycle
        assert cc.stratification == StratificationLens.CELL_CYCLE
        assert len(cc.sobol_indices) >= 1
        assert isinstance(cc.surrogate, PCESurrogate)

        for stage_sobol in cc.sobol_indices:
            assert isinstance(stage_sobol, SobolIndices)
            assert len(stage_sobol.parameter_names) > 0

    def test_variance_decomposition(self, quantify_pipeline):
        decomp = quantify_pipeline.result.variance_decomposition
        assert isinstance(decomp, dict)
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp
        gen_frac = np.asarray(decomp["generation_fraction"])
        seed_frac = np.asarray(decomp["seed_fraction"])
        assert np.all(gen_frac >= 0)
        assert np.all(seed_frac >= 0)

    def test_cell_cycle_relevance(self, quantify_pipeline):
        ccr = quantify_pipeline.result.cell_cycle_relevance
        assert isinstance(ccr, CellCycleRelevanceResult)
        assert len(ccr.relevant_observables) > 0

    def test_sobol_parameter_names_match_cache(self, quantify_pipeline, precomputed_cache_dir):
        """Sobol parameter names should match the cached parameter names."""
        import json

        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        cached_names = meta["parameter_names"]
        sobol_names = quantify_pipeline.result.population.sobol_indices[0].parameter_names
        assert set(sobol_names).issubset(set(cached_names))


# ---------------------------------------------------------------------------
# Tests: Export / Round-Trip
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestQuantifyExport:
    def test_export_and_reload(self, quantify_pipeline, tmp_path):
        export_dir = tmp_path / "quantify_export"
        quantify_pipeline.result.export(export_dir)

        assert (export_dir / "metadata.json").exists()
        assert (export_dir / "population_surrogate").exists()
        assert (export_dir / "population_sobol").exists()
        assert (export_dir / "variance_decomposition.json").exists()

        loaded = PipelineResult.from_export(export_dir)
        np.testing.assert_array_almost_equal(
            loaded.population.sobol_indices[0].first_order,
            quantify_pipeline.result.population.sobol_indices[0].first_order,
        )


# ---------------------------------------------------------------------------
# Tests: CLI Entrypoint
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestQuantifyCli:
    """Test the quantify CLI command end-to-end."""

    def test_quantify_with_precomputed(self, precomputed_cache_dir, tmp_path):
        """Run `uq quantify` via CliRunner with precomputed samples."""
        from uq.cli import app

        export_dir = tmp_path / "cli_export"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "quantify",
                *EXPERIMENT_IDS,
                str(SIM_BASE_PATH),
                "--n-bins",
                str(N_BINS),
                "--n-samples",
                str(N_SAMPLES),
                "--pce-polynomial-order",
                str(POLYNOMIAL_ORDER),
                "--pce-n-trajectories",
                str(N_TRAJECTORIES),
                "--pce-n-selected-params",
                str(N_TOP),
                "--precomputed-path",
                str(precomputed_cache_dir),
                "--export-path",
                str(export_dir),
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )

        assert result.exit_code == 0

        # Report should contain key sections
        assert "UQ REPORT" in result.output
        assert "POPULATION" in result.output or "Phase 1" in result.output or "PHASE 1" in result.output

        # Export artifacts should exist
        assert (export_dir / "metadata.json").exists()

    def test_quantify_report_contains_sobol(self, precomputed_cache_dir):
        """Report output should contain Sobol index information."""
        from uq.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "quantify",
                *EXPERIMENT_IDS,
                str(SIM_BASE_PATH),
                "--n-samples",
                str(N_SAMPLES),
                "--pce-polynomial-order",
                str(POLYNOMIAL_ORDER),
                "--precomputed-path",
                str(precomputed_cache_dir),
            ],
        )

        assert result.exit_code == 0
        # Should show percentage values from Sobol
        assert "%" in result.output


# ---------------------------------------------------------------------------
# Tests: Sample/Pipeline Specification Compatibility
# ---------------------------------------------------------------------------


@skip_if_no_data
class TestSamplePipelineCompatibility:
    """Verify that precomputed samples are correctly matched to pipeline specs.

    The key invariant: the parameter_names and bounds stored in the cache
    must match the parameter space that the pipeline constructs from the
    same experiment_ids + sim_base_path. If they diverge, the Sobol indices
    would be attributed to wrong parameters.
    """

    def test_cache_params_match_pipeline_params(self, precomputed_cache_dir):
        """Cache parameter_names must match what initialize_data produces."""
        import json

        from uq.pipe import initialize_data

        ds = initialize_data(
            experiment_ids=EXPERIMENT_IDS,
            sim_base_path=str(SIM_BASE_PATH),
            observable_columns=OBSERVABLE_COLUMNS,
        )
        pipeline_names = ds.parameter_space.parameter_names
        pipeline_bounds = ds.parameter_space.parameter_bounds

        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        cached_names = meta["parameter_names"]
        cached_bounds = np.array(meta["bounds"])

        assert cached_names == pipeline_names, (
            f"Parameter name mismatch: cache has {cached_names}, pipeline expects {pipeline_names}"
        )
        np.testing.assert_array_almost_equal(
            cached_bounds,
            np.array(pipeline_bounds),
            err_msg="Parameter bounds mismatch between cache and pipeline",
        )

    def test_cache_n_params_matches_X_columns(self, precomputed_cache_dir):
        """X.shape[1] must equal len(parameter_names) in metadata."""
        import json

        X = np.load(precomputed_cache_dir / "X.npy")
        meta = json.loads((precomputed_cache_dir / "metadata.json").read_text())
        assert X.shape[1] == len(meta["parameter_names"])

    def test_mismatched_cache_raises_or_warns(self, tmp_path):
        """Using a cache generated for different params should not silently succeed.

        This test creates a deliberately mismatched cache (wrong n_params)
        and verifies the pipeline catches the incompatibility.
        """
        from uq.sampling import PrecomputedCache

        # Create a fake cache with 1 parameter (pipeline expects more)
        fake_cache = PrecomputedCache(
            cache_dir=tmp_path / "bad_cache",
            X=np.random.rand(10, 1),
            Y=np.random.rand(10, 2),
            parameter_names=["fake_param"],
            metadata={"bounds": [[0.0, 1.0]], "seed": 0},
        )
        fake_cache.save()

        # The pipeline should fail or raise when dimensions don't match
        from uq import handlers
        from uq.pce.models import PCEParameterSelectionConfig

        with pytest.raises(Exception):
            handlers.pipeline(
                experiment_ids=EXPERIMENT_IDS,
                sim_base_path=str(SIM_BASE_PATH),
                observable_columns=OBSERVABLE_COLUMNS,
                n_samples=10,
                precomputed_path=str(tmp_path / "bad_cache"),
                prescreen_config=PCEParameterSelectionConfig(n_trajectories=5, n_top=3),
                execute=True,
            )
