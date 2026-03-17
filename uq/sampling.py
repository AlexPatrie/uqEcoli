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

    The simulation_func is called twice per sample when ``store_timeseries``
    is True:
      - ``simulation_func(x)`` returns raw timeseries (n_timesteps, n_obs)
      - ``simulation_func.evaluate_batch(X)`` returns aggregated (n_samples, n_outputs)

    If simulation_func does not return 2D timeseries from ``__call__``,
    timeseries storage is skipped.

    Args:
        parameter_space: Input parameter space with bounds.
        simulation_func: Callable with ``__call__(x)`` and ``evaluate_batch(X)``.
        n_samples: Number of LHS samples to generate.
        cache_dir: Directory to write cached data.
        seed: Random seed for LHS generation.
        store_timeseries: Whether to cache per-sample raw timeseries for Phase 2.
        max_workers: If > 1, use parallel evaluation (passed to
            ``evaluate_batch``).  None = sequential.

    Returns:
        PrecomputedCache with X, Y, and optionally timeseries.
    """
    X = generate_lhs_samples(parameter_space, n_samples, seed=seed)

    # Evaluate aggregated outputs (Phase 1)
    # Pass max_workers if the simulation_func supports it
    if max_workers and hasattr(simulation_func, "evaluate_batch"):
        import inspect

        sig = inspect.signature(simulation_func.evaluate_batch)
        if "max_workers" in sig.parameters:
            Y = simulation_func.evaluate_batch(X, max_workers=max_workers)
        else:
            Y = simulation_func.evaluate_batch(X)
    else:
        Y = simulation_func.evaluate_batch(X)

    # Optionally collect per-sample timeseries (Phase 2)
    Y_timeseries = None
    if store_timeseries:
        Y_timeseries = []
        for x in X:
            ts = simulation_func(x)
            if ts.ndim == 2:
                Y_timeseries.append(ts)
            else:
                # simulation_func returns 1D — no timeseries to store
                Y_timeseries = None
                break

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
