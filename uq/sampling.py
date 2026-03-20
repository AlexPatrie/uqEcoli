"""
Pre-computation and caching of (X, Y) sample pairs for UQ analysis.

Stage 1 of the two-stage UQ workflow:
  1. Generate LHS sample points from the parameter space
  2. Evaluate the simulation function at each sample
  3. Cache (X, Y) to disk for later PCE fitting

Stage 2 (pipeline analysis) loads the cached data and fits PCE
directly without re-running simulations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

from uq.inputs import XSpace


@dataclass
class PrecomputedCache:
    """On-disk cache of (X, Y) sample pairs.

    Attributes:
        cache_dir: Directory containing the cached files.
        X: Input samples, shape (n_samples, n_params).
        Y: Aggregated outputs, shape (n_samples, n_outputs).
        parameter_names: Names of the input parameters.
        metadata: Additional metadata (bounds, polynomial_order, etc.).
        Y_timeseries: Per-sample raw timeseries for Phase 2.
            List of arrays, each (n_timesteps, n_obs). None if not cached.
    """

    cache_dir: Path
    X: np.ndarray
    Y: np.ndarray
    parameter_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    Y_timeseries: list[np.ndarray] | None = None

    def save(self) -> None:
        """Save X.npy, Y.npy, metadata.json (and optional timeseries) to cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.cache_dir / "X.npy", self.X)
        np.save(self.cache_dir / "Y.npy", self.Y)

        meta = {
            "parameter_names": self.parameter_names,
            "n_samples": int(self.X.shape[0]),
            "n_params": int(self.X.shape[1]),
            "n_outputs": int(self.Y.shape[1]),
            **self.metadata,
        }
        (self.cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Save per-sample timeseries if available (for Phase 2)
        if self.Y_timeseries is not None:
            ts_dir = self.cache_dir / "timeseries"
            ts_dir.mkdir(exist_ok=True)
            for i, ts in enumerate(self.Y_timeseries):
                np.save(ts_dir / f"sample_{i:04d}.npy", ts)

    @classmethod
    def load(cls, cache_dir: str | Path) -> PrecomputedCache:
        """Load from cache_dir."""
        cache_dir = Path(cache_dir)
        X = np.load(cache_dir / "X.npy")
        Y = np.load(cache_dir / "Y.npy")
        meta = json.loads((cache_dir / "metadata.json").read_text())
        parameter_names = meta.pop("parameter_names")

        # Load timeseries if available
        ts_dir = cache_dir / "timeseries"
        Y_timeseries = None
        if ts_dir.exists():
            n_samples = X.shape[0]
            Y_timeseries = []
            for i in range(n_samples):
                ts_path = ts_dir / f"sample_{i:04d}.npy"
                if ts_path.exists():
                    Y_timeseries.append(np.load(ts_path))

        return cls(
            cache_dir=cache_dir,
            X=X,
            Y=Y,
            parameter_names=parameter_names,
            metadata=meta,
            Y_timeseries=Y_timeseries,
        )


def generate_lhs_samples(
    parameter_space: XSpace,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate LHS samples scaled to parameter bounds.

    Latin Hypercube Samples — a space-filling sampling strategy for exploring
    a multi-dimensional parameter space.

    Instead of pure random sampling (which can leave gaps and clusters), LHS
    divides each parameter's range into n equal strata and ensures exactly one
     sample falls in each stratum per dimension. This guarantees better
    coverage of the parameter space with fewer samples than Monte Carlo.

    In this codebase, generate_lhs_samples() in uq/sampling.py:100 uses
    scipy.stats.qmc.LatinHypercube to generate unit-cube samples, then scales
    them to the physical parameter bounds (e.g., vio_expression in [0, 5],
    mecillinam_concentration in [0, 10]).

    The result is an (n_samples, n_params) array where each row is a parameter
     vector x that gets fed to the simulation function f(x) -> y for PCE
    surrogate fitting.

    Args:
        parameter_space: Input parameter space with bounds.
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, n_params) in the physical parameter domain.
    """
    bounds = np.array(parameter_space.parameter_bounds)
    sampler = qmc.LatinHypercube(d=parameter_space.n_parameters, seed=seed)
    X_unit = sampler.random(n=n_samples)
    return qmc.scale(X_unit, bounds[:, 0], bounds[:, 1])


def _evaluate_sample(simulation_func: Callable, x: np.ndarray) -> np.ndarray:
    """Evaluate one sample — top-level function for ProcessPoolExecutor."""
    return simulation_func(x)


def run_and_cache(
    parameter_space: XSpace,
    simulation_func: Callable,
    n_samples: int,
    cache_dir: Path,
    seed: int = 42,
    store_timeseries: bool = True,
    max_workers: int | None = None,
) -> PrecomputedCache:
    """Generate LHS samples, evaluate simulation_func, save to disk.

    Each sample is evaluated **once** via ``simulation_func(x)``, which
    returns raw timeseries of shape ``(n_timesteps, n_obs)``.  The
    aggregated output Y (for Phase 1) is derived by taking the time-mean
    of each timeseries.  This avoids the previous double-evaluation where
    ``evaluate_batch`` and ``__call__`` each ran a full simulation.

    If ``simulation_func(x)`` returns a 1D array (no timeseries), the
    function falls back to ``evaluate_batch`` for Y and skips timeseries
    storage.

    Args:
        parameter_space: Input parameter space with bounds.
        simulation_func: Callable with ``__call__(x)`` returning
            ``(n_timesteps, n_obs)`` timeseries.  Also used as fallback
            via ``evaluate_batch(X)`` if ``__call__`` returns 1D.
        n_samples: Number of LHS samples to generate.
        cache_dir: Directory to write cached data.
        seed: Random seed for LHS generation.
        store_timeseries: Whether to cache per-sample raw timeseries
            for Phase 2.
        max_workers: If > 1, use parallel evaluation via
            ProcessPoolExecutor.  None = sequential.

    Returns:
        PrecomputedCache with X, Y, and optionally Y_timeseries.
    """
    X = generate_lhs_samples(parameter_space, n_samples, seed=seed)

    # Single-pass evaluation: call simulation_func(x) once per sample,
    # collect raw timeseries, derive aggregated Y by time-mean.
    if max_workers and max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        timeseries_list: list[np.ndarray | None] = [None] * len(X)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(_evaluate_sample, simulation_func, X[i]): i for i in range(len(X))}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                timeseries_list[idx] = future.result()
    else:
        timeseries_list = [simulation_func(x) for x in X]

    # Check if we got 2D timeseries or 1D scalars
    has_timeseries = timeseries_list[0] is not None and timeseries_list[0].ndim == 2

    if has_timeseries:
        # Derive Y from timeseries (time-mean) — no second evaluation needed
        Y = np.vstack([ts.mean(axis=0) for ts in timeseries_list])
        Y_timeseries = timeseries_list if store_timeseries else None
    else:
        # Fallback: simulation_func returns 1D, use evaluate_batch for Y
        Y = np.vstack([ts if ts.ndim == 1 else ts.ravel() for ts in timeseries_list])
        Y_timeseries = None

    bounds = np.array(parameter_space.parameter_bounds)
    cache = PrecomputedCache(
        cache_dir=Path(cache_dir),
        X=X,
        Y=Y,
        parameter_names=parameter_space.parameter_names,
        metadata={
            "bounds": bounds.tolist(),
            "seed": seed,
        },
        Y_timeseries=Y_timeseries,
    )
    cache.save()
    return cache


def run_batch_and_cache(
    parameter_space: XSpace,
    simulation_func: Any,
    n_samples: int,
    cache_dir: Path,
    seed: int = 42,
    store_timeseries: bool = True,
    max_workers: int | None = None,
    batch_dir: Path | None = None
) -> PrecomputedCache:
    """Generate LHS samples, run batch simulation via subprocesses, cache.

    Unlike ``run_and_cache()`` which calls ``simulation_func(x)`` per
    sample, this function delegates to ``simulation_func._run_batch(X)``
    which runs all simulations as subprocesses with Parquet output and
    collects results in a single pass.

    This is the preferred path for vEcoli live simulations: no EcoliSim
    is held in the UQ process memory.

    Args:
        parameter_space: Input parameter space with bounds.
        simulation_func: Object with ``_run_batch(X, max_workers)``
            method (e.g., ``TimeseriesGeneratorVecoli``).
        n_samples: Number of LHS samples to generate.
        cache_dir: Directory to write cached data.
        seed: Random seed for LHS generation.
        store_timeseries: Whether to cache per-sample raw timeseries.
        max_workers: Max parallel subprocesses. None = sequential.
        batch_dir: Destination in which batch artifacts will be saved.

    Returns:
        PrecomputedCache with X, Y, and optionally Y_timeseries.
    """
    X = generate_lhs_samples(parameter_space, n_samples, seed=seed)

    Y, Y_timeseries = simulation_func._run_batch(
        X, max_workers=max_workers, batch_dir=batch_dir
    )

    if not store_timeseries:
        Y_timeseries = None

    bounds = np.array(parameter_space.parameter_bounds)
    cache = PrecomputedCache(
        cache_dir=Path(cache_dir),
        X=X,
        Y=Y,
        parameter_names=parameter_space.parameter_names,
        metadata={
            "bounds": bounds.tolist(),
            "seed": seed,
        },
        Y_timeseries=Y_timeseries,
    )
    cache.save()
    return cache
