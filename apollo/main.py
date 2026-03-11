import numpy as np

from apollo import LosslessScore, encode_spectrum, encode_spectrum_lossless
from apollo.encoding import encode_trajectory_lossless
from uq.koopman import ExtendedDMD, KoopmanSpectrum

# Run simulation: y = f(x, theta, t)
# y = run_simulation(params)  # shape: (n_timesteps, n_observables)


def generate_spectrum(y: np.ndarray) -> KoopmanSpectrum:
    # Extract Koopman spectral representation
    dmd = ExtendedDMD(rank=10, dt=1.0)
    dmd.fit(y)
    return dmd.get_spectrum(observable_names=["dry_mass", "growth_rate", ...])


def generate_score(spectrum, y) -> LosslessScore:
    return encode_spectrum_lossless(
        spectrum,
        duration=len(y),
        cell_cycle_time=3600.0,
    )


def reconstruct_timeseries(score: LosslessScore) -> np.ndarray:
    return score.reconstruct()


def _encode_trajectory(y: np.ndarray, cell_cycle_time: float = 3600.0, rank: int = 10):
    return encode_trajectory_lossless(
        X=y, observable_names=["dry_mass", "volume", "growth_rate"], cell_cycle_time=cell_cycle_time, rank=rank
    )


def encode_trajectory(
    y: np.ndarray,
    observable_names: list[str],
    dt: float = 1.0,
    cell_cycle_time: float | None = None,
    rank: int | None = None,
    energy_threshold: float = 0.99,
) -> LosslessScore:
    """Encode with automatic parameter selection."""

    # Auto-detect cell cycle time from mass if not provided
    if cell_cycle_time is None:
        # Assume exponential growth, estimate period
        if "mass" in observable_names[0].lower():
            log_y = np.log(np.maximum(y[:, 0], 1e-10))
            # Fit exponential to get growth rate
            growth_rate = np.polyfit(np.arange(len(log_y)) * dt, log_y, 1)[0]
            # Doubling time = ln(2) / growth_rate
            cell_cycle_time = np.log(2) / growth_rate if growth_rate > 0 else 3600.0
        else:
            cell_cycle_time = 3600.0  # Default 1 hour

    # Auto-select rank if not provided
    if rank is None:
        from uq.koopman import DynamicModeDecomposition

        dmd = DynamicModeDecomposition(rank=None, dt=dt)
        dmd.fit(y)

        # Find rank for desired energy retention
        sv = dmd._S
        cumulative = np.cumsum(sv**2) / np.sum(sv**2)
        rank = int(np.searchsorted(cumulative, energy_threshold) + 1)
        rank = max(3, min(rank, len(y) // 4))  # Bounds

    return encode_trajectory_lossless(
        X=y,
        observable_names=observable_names,
        cell_cycle_time=cell_cycle_time,
        rank=rank,
    )


def decode_trajectory(score: LosslessScore):
    return score.reconstruct()


def test_encode_trajectory():
    obs = ["x", "y", "z"]
    n_obs = len(obs)
    total_duration = 1111
    y = np.random.random((total_duration, n_obs))
    score = encode_trajectory(y=y, observable_names=obs)
    print()
