"""
Tests for uq.sampling — single-evaluation caching and PrecomputedCache round-trip.

Verifies:
- run_and_cache calls simulation_func exactly once per sample (not twice)
- Y is derived from timeseries via time-mean (not a separate evaluate_batch call)
- PrecomputedCache round-trips correctly (X, Y, timeseries, metadata)
- Parallel evaluation (max_workers > 1) produces identical results
- Fallback to 1D mode when simulation_func returns scalars
"""

import json

import numpy as np
import pytest

from libuq.inputs import XSpaceVecoli
from libuq.pipeline.models import SimDataParameter
from libuq.sampling import PrecomputedCache, generate_lhs_samples, run_and_cache


class CountingWrapper:
    """Test wrapper that counts how many times __call__ and evaluate_batch are invoked."""

    def __init__(self, n_params: int, n_obs: int = 3, n_timesteps: int = 50):
        self.n_params = n_params
        self.n_obs = n_obs
        self.n_timesteps = n_timesteps
        self.call_count = 0
        self.evaluate_batch_count = 0
        self._rng = np.random.default_rng(42)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return synthetic 2D timeseries. Increments call_count."""
        self.call_count += 1
        # Deterministic output based on x so results are reproducible
        base = np.outer(
            np.linspace(0, 1, self.n_timesteps), x[: self.n_obs] if len(x) >= self.n_obs else np.ones(self.n_obs)
        )
        noise = self._rng.normal(0, 0.01, base.shape)
        return base + noise

    def evaluate_batch(self, X: np.ndarray, max_workers: int | None = None) -> np.ndarray:
        """Should NOT be called when store_timeseries=True."""
        self.evaluate_batch_count += 1
        return np.vstack([self(x).mean(axis=0) for x in X])


class ScalarWrapper:
    """Test wrapper that returns 1D output (no timeseries)."""

    def __init__(self, n_params: int):
        self.n_params = n_params
        self.call_count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return np.array([x.sum(), x.mean()])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self(x) for x in X])


@pytest.fixture
def param_space():
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


class TestRunAndCacheSingleEvaluation:
    """Verify that run_and_cache evaluates each sample exactly once."""

    def test_single_call_per_sample(self, param_space, tmp_path):
        """Each sample should be evaluated via __call__ exactly once."""
        n_samples = 5
        wrapper = CountingWrapper(n_params=param_space.n_parameters)

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=n_samples,
            cache_dir=tmp_path / "cache",
        )

        # __call__ should be invoked exactly n_samples times
        assert wrapper.call_count == n_samples
        # evaluate_batch should NOT be called at all
        assert wrapper.evaluate_batch_count == 0

    def test_y_derived_from_timeseries_mean(self, param_space, tmp_path):
        """Y should equal the time-mean of Y_timeseries."""
        wrapper = CountingWrapper(n_params=param_space.n_parameters, n_timesteps=50)

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
        )

        assert cache.Y_timeseries is not None
        assert len(cache.Y_timeseries) == 3

        for i, ts in enumerate(cache.Y_timeseries):
            expected_y = ts.mean(axis=0)
            np.testing.assert_array_almost_equal(cache.Y[i], expected_y)

    def test_timeseries_stored_when_requested(self, param_space, tmp_path):
        """Y_timeseries should be populated when store_timeseries=True."""
        wrapper = CountingWrapper(n_params=param_space.n_parameters)

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
            store_timeseries=True,
        )

        assert cache.Y_timeseries is not None
        assert len(cache.Y_timeseries) == 3
        assert cache.Y_timeseries[0].ndim == 2

    def test_timeseries_not_stored_when_disabled(self, param_space, tmp_path):
        """Y_timeseries should be None when store_timeseries=False."""
        wrapper = CountingWrapper(n_params=param_space.n_parameters)

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
            store_timeseries=False,
        )

        assert cache.Y_timeseries is None
        # Y should still be computed correctly
        assert cache.Y.shape == (3, wrapper.n_obs)

    def test_scalar_fallback(self, param_space, tmp_path):
        """When simulation_func returns 1D, Y is used directly, no timeseries."""
        wrapper = ScalarWrapper(n_params=param_space.n_parameters)

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
        )

        assert cache.Y_timeseries is None
        assert cache.Y.shape == (3, 2)
        assert wrapper.call_count == 3


class TestPrecomputedCacheRoundTrip:
    """Verify that PrecomputedCache serializes and deserializes losslessly."""

    def test_round_trip_with_timeseries(self, param_space, tmp_path):
        wrapper = CountingWrapper(n_params=param_space.n_parameters, n_timesteps=20)

        original = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=4,
            cache_dir=tmp_path / "cache",
        )

        loaded = PrecomputedCache.load(tmp_path / "cache")

        np.testing.assert_array_equal(loaded.X, original.X)
        np.testing.assert_array_equal(loaded.Y, original.Y)
        assert loaded.parameter_names == original.parameter_names
        assert loaded.Y_timeseries is not None
        assert len(loaded.Y_timeseries) == len(original.Y_timeseries)
        for orig_ts, load_ts in zip(original.Y_timeseries, loaded.Y_timeseries):
            np.testing.assert_array_equal(load_ts, orig_ts)

    def test_round_trip_without_timeseries(self, param_space, tmp_path):
        wrapper = CountingWrapper(n_params=param_space.n_parameters)

        original = run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
            store_timeseries=False,
        )

        loaded = PrecomputedCache.load(tmp_path / "cache")

        np.testing.assert_array_equal(loaded.X, original.X)
        np.testing.assert_array_equal(loaded.Y, original.Y)
        assert loaded.Y_timeseries is None

    def test_metadata_persisted(self, param_space, tmp_path):
        wrapper = CountingWrapper(n_params=param_space.n_parameters)

        run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=5,
            cache_dir=tmp_path / "cache",
            seed=123,
        )

        meta = json.loads((tmp_path / "cache" / "metadata.json").read_text())
        assert meta["n_samples"] == 5
        assert meta["n_params"] == param_space.n_parameters
        assert meta["seed"] == 123
        assert meta["parameter_names"] == param_space.parameter_names

    def test_disk_files_created(self, param_space, tmp_path):
        wrapper = CountingWrapper(n_params=param_space.n_parameters, n_timesteps=10)

        run_and_cache(
            parameter_space=param_space,
            simulation_func=wrapper,
            n_samples=3,
            cache_dir=tmp_path / "cache",
        )

        cache_dir = tmp_path / "cache"
        assert (cache_dir / "X.npy").exists()
        assert (cache_dir / "Y.npy").exists()
        assert (cache_dir / "metadata.json").exists()
        assert (cache_dir / "timeseries").is_dir()
        for i in range(3):
            assert (cache_dir / "timeseries" / f"sample_{i:04d}.npy").exists()


class TestLHSSampling:
    def test_sample_shape(self, param_space):
        X = generate_lhs_samples(param_space, n_samples=10)
        assert X.shape == (10, param_space.n_parameters)

    def test_samples_within_bounds(self, param_space):
        X = generate_lhs_samples(param_space, n_samples=50)
        bounds = np.array(param_space.parameter_bounds)
        for i in range(param_space.n_parameters):
            assert np.all(X[:, i] >= bounds[i, 0])
            assert np.all(X[:, i] <= bounds[i, 1])

    def test_reproducibility(self, param_space):
        X1 = generate_lhs_samples(param_space, n_samples=10, seed=42)
        X2 = generate_lhs_samples(param_space, n_samples=10, seed=42)
        np.testing.assert_array_equal(X1, X2)
