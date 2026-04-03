"""
Tests for uq_simple pipeline: aggregation strategies, PyTUQ PCE integration, regression modes.

Verifies:
- PrecomputedCache round-trips timeseries metadata (generation/seed labels)
- _aggregate_by_group correctly groups and averages timeseries
- run_by_generation produces per-generation Sobol indices
- run_by_seed produces per-seed Sobol indices
- Graceful degradation when metadata is absent (old caches)
- End-to-end SimplePipelineResult with all 4 strategies
- Export produces correct artifact directories
- _fit_pce_and_sobol uses PyTUQ PCE directly with correct Sobol computation
- Regression modes (lsq, bcs) produce valid results
- Surrogate coefficients are non-zero (not the previous broken extraction)
"""

import json

import numpy as np
import pytest

from libuq.inputs import XSpaceVecoli
from libuq.pipeline.models import SimDataParameter
from libuq.sampling import PrecomputedCache
from uq.pipeline import (
    SimplePipelineResult,
    _aggregate_by_group,
    _fit_pce_and_sobol,
    run_by_generation,
    run_by_seed,
    run_pipeline,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def param_space():
    return XSpaceVecoli(
        parameters=[
            SimDataParameter(
                name="rnap_free",
                attr_path="process.transcription.fraction_active_rnap_free",
                bounds=(0.25, 0.47),
            ),
            SimDataParameter(
                name="dry_mass_frac",
                attr_path="mass.cell_dry_mass_fraction",
                bounds=(0.25, 0.35),
            ),
        ]
    )


def _make_synthetic_cache(
    tmp_path,
    n_samples: int = 20,
    n_params: int = 2,
    n_obs: int = 4,
    n_timesteps: int = 100,
    n_generations: int = 3,
    n_seeds: int = 2,
    include_meta: bool = True,
) -> PrecomputedCache:
    """Build a synthetic PrecomputedCache with generation/seed metadata.

    Each sample's timeseries has rows tagged with generation and seed IDs.
    Observable 0 is a monotonically increasing "mass" column (for Phase 2).
    The outputs are deterministic functions of X so PCE can fit them.
    """
    rng = np.random.default_rng(42)
    bounds = np.array([[0.25, 0.47], [0.25, 0.35]])[:n_params]
    X = bounds[:, 0] + rng.random((n_samples, n_params)) * (bounds[:, 1] - bounds[:, 0])

    Y_timeseries = []
    Y_timeseries_meta = [] if include_meta else None
    Y_list = []

    rows_per_gen = n_timesteps // n_generations

    for i in range(n_samples):
        # Build timeseries: obs 0 = monotonic mass, others = f(x) + noise
        ts = np.zeros((n_timesteps, n_obs))
        ts[:, 0] = np.linspace(1.0, 2.0, n_timesteps)  # dry mass (monotonic)
        for j in range(1, n_obs):
            ts[:, j] = X[i, 0] * (j + 1) + X[i, 1 % n_params] * 0.5 + rng.normal(0, 0.01, n_timesteps)
        Y_timeseries.append(ts)
        Y_list.append(ts.mean(axis=0))

        if include_meta:
            gen_labels = np.zeros(n_timesteps, dtype=np.int64)
            seed_labels = np.zeros(n_timesteps, dtype=np.int64)
            for g in range(n_generations):
                start = g * rows_per_gen
                end = start + rows_per_gen if g < n_generations - 1 else n_timesteps
                gen_labels[start:end] = g
                seed_labels[start:end] = g % n_seeds
            Y_timeseries_meta.append({
                "generation": gen_labels,
                "lineage_seed": seed_labels,
            })

    Y = np.vstack(Y_list)

    cache = PrecomputedCache(
        cache_dir=tmp_path / "cache",
        X=X,
        Y=Y,
        parameter_names=[f"p{i}" for i in range(n_params)],
        metadata={"bounds": bounds.tolist(), "seed": 42},
        Y_timeseries=Y_timeseries,
        Y_timeseries_meta=Y_timeseries_meta,
    )
    return cache


# ── PrecomputedCache metadata round-trip ────────────────────────────


class TestCacheMetadataRoundTrip:
    def test_save_load_with_metadata(self, tmp_path):
        """Metadata files are written and loaded correctly."""
        cache = _make_synthetic_cache(tmp_path, n_samples=5)
        cache.save()

        loaded = PrecomputedCache.load(tmp_path / "cache")

        assert loaded.Y_timeseries_meta is not None
        assert len(loaded.Y_timeseries_meta) == 5
        for i in range(5):
            assert "generation" in loaded.Y_timeseries_meta[i]
            assert "lineage_seed" in loaded.Y_timeseries_meta[i]
            np.testing.assert_array_equal(
                loaded.Y_timeseries_meta[i]["generation"],
                cache.Y_timeseries_meta[i]["generation"],
            )
            np.testing.assert_array_equal(
                loaded.Y_timeseries_meta[i]["lineage_seed"],
                cache.Y_timeseries_meta[i]["lineage_seed"],
            )

    def test_save_load_without_metadata(self, tmp_path):
        """Old-style caches (no metadata) load with Y_timeseries_meta=None."""
        cache = _make_synthetic_cache(tmp_path, n_samples=5, include_meta=False)
        cache.save()

        loaded = PrecomputedCache.load(tmp_path / "cache")

        assert loaded.Y_timeseries is not None
        assert loaded.Y_timeseries_meta is None

    def test_meta_files_exist_on_disk(self, tmp_path):
        """sample_NNNN_meta.npz files are written alongside timeseries."""
        cache = _make_synthetic_cache(tmp_path, n_samples=3)
        cache.save()

        ts_dir = tmp_path / "cache" / "timeseries"
        for i in range(3):
            assert (ts_dir / f"sample_{i:04d}.npy").exists()
            assert (ts_dir / f"sample_{i:04d}_meta.npz").exists()


# ── _aggregate_by_group ─────────────────────────────────────────────


class TestAggregateByGroup:
    def test_group_by_generation(self, tmp_path):
        """Grouping by generation returns correct number of groups."""
        cache = _make_synthetic_cache(tmp_path, n_samples=10, n_generations=3)
        grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "generation")
        assert len(grouped) == 3
        for gen, Y_g in grouped.items():
            assert Y_g.shape == (10, 4)  # n_samples, n_obs

    def test_group_by_seed(self, tmp_path):
        """Grouping by lineage_seed returns correct number of groups."""
        cache = _make_synthetic_cache(tmp_path, n_samples=10, n_seeds=2)
        grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "lineage_seed")
        assert len(grouped) == 2
        for seed, Y_s in grouped.items():
            assert Y_s.shape == (10, 4)

    def test_empty_when_key_missing(self, tmp_path):
        """Returns empty dict when the group key is not in metadata."""
        cache = _make_synthetic_cache(tmp_path, n_samples=5)
        # Remove lineage_seed from metadata
        for m in cache.Y_timeseries_meta:
            del m["lineage_seed"]
        grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "lineage_seed")
        assert grouped == {}

    def test_per_group_means_are_correct(self, tmp_path):
        """Verify that group means match manual computation."""
        cache = _make_synthetic_cache(tmp_path, n_samples=2, n_timesteps=6, n_generations=2, n_obs=2)
        grouped = _aggregate_by_group(cache.Y_timeseries, cache.Y_timeseries_meta, "generation")
        # Check first sample, generation 0
        ts = cache.Y_timeseries[0]
        gen_labels = cache.Y_timeseries_meta[0]["generation"]
        mask_gen0 = gen_labels == 0
        expected = ts[mask_gen0].mean(axis=0)
        np.testing.assert_array_almost_equal(grouped[0][0], expected)


# ── run_by_generation ───────────────────────────────────────────────


class TestRunByGeneration:
    def test_returns_sobol_per_generation(self, param_space, tmp_path):
        """Each generation should have its own SobolIndices."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, n_generations=3)
        result = run_by_generation(
            param_space,
            cache.X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order=1,
        )
        assert len(result) == 3
        for gen, sobol in result.items():
            assert sobol.first_order.shape == (2,)  # 2 params
            assert sobol.total_order.shape == (2,)
            # Sobol indices should be non-negative
            assert np.all(sobol.total_order >= -0.1)  # small numerical tolerance

    def test_empty_without_metadata(self, param_space, tmp_path):
        """Returns empty dict when no generation labels exist."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, include_meta=False)
        # Can't call run_by_generation without meta — pipeline handles this
        # Test _aggregate_by_group directly
        result = _aggregate_by_group(cache.Y_timeseries, [], "generation")
        assert result == {}


# ── run_by_seed ─────────────────────────────────────────────────────


class TestRunBySeed:
    def test_returns_sobol_per_seed(self, param_space, tmp_path):
        """Each seed should have its own SobolIndices."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, n_seeds=2)
        result = run_by_seed(
            param_space,
            cache.X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order=1,
        )
        assert len(result) == 2
        for seed, sobol in result.items():
            assert sobol.first_order.shape == (2,)
            assert sobol.total_order.shape == (2,)
            assert np.all(sobol.total_order >= -0.1)


# ── End-to-end pipeline ────────────────────────────────────────────


class TestPipelineWithAllStrategies:
    def test_full_pipeline_with_metadata(self, param_space, tmp_path):
        """Pipeline produces results for all 4 strategies."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, n_generations=3, n_seeds=2)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
        )

        # Strategy 1: population
        assert result.population_sobol is not None
        assert result.population_surrogate is not None

        # Strategy 2: by generation
        assert result.per_generation_sobol is not None
        assert len(result.per_generation_sobol) == 3

        # Strategy 3: by seed
        assert result.per_seed_sobol is not None
        assert len(result.per_seed_sobol) == 2

        # Strategy 4: growth-stratified
        assert len(result.per_stage_sobol) == 5

    def test_pipeline_without_metadata_degrades_gracefully(self, param_space, tmp_path):
        """Pipeline works without metadata — strategies 2-3 are None."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, include_meta=False)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
        )

        assert result.population_sobol is not None
        assert result.per_generation_sobol is None
        assert result.per_seed_sobol is None
        assert len(result.per_stage_sobol) == 5


# ── Export artifacts ────────────────────────────────────────────────


class TestExportWithStrategies:
    def test_export_creates_generation_directories(self, param_space, tmp_path):
        """Export writes generation_N_sobol/ directories."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, n_generations=3, n_seeds=2)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
        )

        export_dir = tmp_path / "export"
        result.export(export_dir)

        # Strategy 2 artifacts
        for gen in range(3):
            gen_dir = export_dir / f"generation_{gen}_sobol"
            assert gen_dir.exists(), f"Missing generation_{gen}_sobol/"
            assert (gen_dir / "first_order.npy").exists()
            assert (gen_dir / "total_order.npy").exists()
            meta = json.loads((gen_dir / "metadata.json").read_text())
            assert meta["generation"] == gen

        # Strategy 3 artifacts
        for seed in range(2):
            seed_dir = export_dir / f"seed_{seed}_sobol"
            assert seed_dir.exists(), f"Missing seed_{seed}_sobol/"
            assert (seed_dir / "first_order.npy").exists()
            assert (seed_dir / "total_order.npy").exists()
            meta = json.loads((seed_dir / "metadata.json").read_text())
            assert meta["lineage_seed"] == seed

    def test_export_uq_results_json_has_strategies(self, param_space, tmp_path):
        """uq_results.json includes strategy2 and strategy3 sections."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, n_generations=2, n_seeds=2)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
        )

        export_dir = tmp_path / "export"
        result.export(export_dir)

        summary = json.loads((export_dir / "uq_results.json").read_text())

        # Strategy 2
        s2 = summary["strategy2_by_generation"]
        assert "generations" in s2
        assert s2["n_generations"] == 2
        for gen_entry in s2["generations"]:
            assert "sobol_total_order" in gen_entry

        # Strategy 3
        s3 = summary["strategy3_by_seed"]
        assert "seeds" in s3
        assert s3["n_seeds"] == 2
        for seed_entry in s3["seeds"]:
            assert "sobol_total_order" in seed_entry

    def test_export_without_metadata_marks_unavailable(self, param_space, tmp_path):
        """uq_results.json strategy sections show not_available when no metadata."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20, include_meta=False)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
        )

        export_dir = tmp_path / "export"
        result.export(export_dir)

        summary = json.loads((export_dir / "uq_results.json").read_text())
        assert summary["strategy2_by_generation"]["status"] == "not_available"
        assert summary["strategy3_by_seed"]["status"] == "not_available"


# ── PyTUQ PCE direct integration ────────────────────────────────────


class TestFitPCEAndSobol:
    """Tests for _fit_pce_and_sobol — the core PyTUQ PCE wrapper."""

    def test_known_linear_relationship(self):
        """For Y = 2*X0 + 0.5*X1, X0 should dominate Sobol indices."""
        rng = np.random.default_rng(42)
        n = 50
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = rng.random((n, 2)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        Y = 2.0 * X[:, 0] + 0.5 * X[:, 1]

        sobol, surr = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=1,
        )

        # X0 has 4x the coefficient → ~16x the variance → S_Ti(x0) >> S_Ti(x1)
        assert sobol.total_order[0] > sobol.total_order[1]
        assert sobol.total_order[0] > 0.5  # x0 should dominate

    def test_surrogate_coefficients_nonzero(self):
        """Surrogate coefficients should be populated (not zeros)."""
        rng = np.random.default_rng(42)
        n = 30
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = rng.random((n, 2))
        Y = X[:, 0] ** 2 + 0.3 * X[:, 1]

        _, surr = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=2,
        )

        assert surr.coefficients is not None
        assert np.any(surr.coefficients != 0), "Surrogate coefficients should be non-zero"

    def test_multi_output_variance_weighting(self):
        """Multi-output should produce variance-weighted Sobol."""
        rng = np.random.default_rng(42)
        n = 50
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = rng.random((n, 2))
        # Y0 has high variance driven by x0, Y1 has low variance
        Y = np.column_stack([
            3.0 * X[:, 0] + 0.1 * X[:, 1],
            0.01 * X[:, 0] + 0.01 * X[:, 1],
        ])

        sobol, _ = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=1,
        )

        # x0 drives the high-variance output → should dominate
        assert sobol.total_order[0] > sobol.total_order[1]

    def test_per_output_mode(self):
        """per_output=True returns (n_outputs, n_params) arrays."""
        rng = np.random.default_rng(42)
        n = 40
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = rng.random((n, 2))
        Y = np.column_stack([X[:, 0], X[:, 1]])

        sobol, _ = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=1,
            per_output=True,
        )

        assert sobol.first_order.shape == (2, 2)  # (n_outputs, n_params)
        assert sobol.total_order.shape == (2, 2)


class TestRegressionModes:
    """Test that different PyTUQ regression backends work."""

    def _make_data(self):
        rng = np.random.default_rng(42)
        n = 60
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = rng.random((n, 2))
        Y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.01, n)
        return X, Y, bounds

    def test_lsq_regression(self):
        """Default LSQ regression produces valid Sobol."""
        X, Y, bounds = self._make_data()
        sobol, surr = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=2,
            regression="lsq",
        )
        assert sobol.total_order[0] > 0.3
        assert np.any(surr.coefficients != 0)

    def test_bcs_regression(self):
        """BCS regression produces valid (possibly sparse) Sobol."""
        X, Y, bounds = self._make_data()
        sobol, surr = _fit_pce_and_sobol(
            X,
            Y,
            bounds,
            parameter_names=["x0", "x1"],
            polynomial_order=2,
            regression="bcs",
        )
        assert sobol.total_order[0] > 0.3
        assert np.any(surr.coefficients != 0)

    def test_pipeline_with_regression_param(self, param_space, tmp_path):
        """run_pipeline accepts regression parameter."""
        cache = _make_synthetic_cache(tmp_path, n_samples=20)
        result = run_pipeline(
            cache=cache,
            param_space=param_space,
            polynomial_order=1,
            n_bins=5,
            regression="lsq",
        )
        assert result.population_sobol is not None
        assert np.any(result.population_surrogate.coefficients != 0)
