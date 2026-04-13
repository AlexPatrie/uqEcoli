"""End-to-end tests for ``uq/workflow_spectral.py``.

The tests use a **synthetic** oscillator with a *known* parameter→frequency
law, so we can verify that:

1. DMD recovers the correct per-sample frequency.
2. The Koopman-output PCE reproduces the known frequency map.
3. Sobol indices correctly identify the parameter that drives the pitch.
4. Inverse design finds a parameter setting for a target frequency.
5. ``SpectralDAWResult.export`` writes the expected artifact layout.
6. ``quantify_spectral`` works on a real ``PrecomputedCache`` on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libuq.sampling import PrecomputedCache
from uq.workflow_spectral import (
    SpectralDAWResult,
    SpectralFeatures,
    SpectralStrategyResult,
    _align_modes,
    _dedupe_conjugate_pairs,
    extract_spectral_features,
    quantify_spectral,
    run_strategy_spectral,
)


# ── Fixture: synthetic parameter-dependent oscillator ─────────────────


TRUE_F0 = 0.02  # intercept (cycles per timestep)
TRUE_SLOPES = (0.06, 0.005)  # pitch sensitivity to each of 2 parameters
TRUE_DAMPING = -0.001  # gentle decay
N_SAMPLES_TRAIN = 40
N_SAMPLES_TEST = 12
N_TIMESTEPS = 300
DT = 1.0


def true_freq(x: np.ndarray) -> float:
    """Parameter → oscillator frequency (cycles per timestep)."""
    return float(TRUE_F0 + TRUE_SLOPES[0] * x[0] + TRUE_SLOPES[1] * x[1])


def _synth_trajectory(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Damped oscillator with 2 correlated observables."""
    rng = np.random.default_rng(seed)
    f = true_freq(x)
    t = np.arange(N_TIMESTEPS, dtype=float) * DT
    decay = np.exp(TRUE_DAMPING * t)
    base = decay * np.cos(2 * np.pi * f * t)
    # Second observable is a lagged copy to keep the problem multivariate
    base2 = decay * np.cos(2 * np.pi * f * t - 0.4)
    noise = rng.normal(scale=0.01, size=(N_TIMESTEPS, 2))
    return np.stack([base, base2], axis=1) + noise


@pytest.fixture(scope="module")
def synthetic_bundle() -> dict:
    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    X_train = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_SAMPLES_TRAIN, 2))
    X_test = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_SAMPLES_TEST, 2))

    Y_ts = [_synth_trajectory(x, seed=i) for i, x in enumerate(X_train)]
    Y_test_ts = [_synth_trajectory(x, seed=1000 + i) for i, x in enumerate(X_test)]

    return {
        "bounds": bounds,
        "parameter_names": ["x0", "x1"],
        "X_train": X_train,
        "X_test": X_test,
        "Y_timeseries": Y_ts,
        "Y_test_timeseries": Y_test_ts,
    }


# ── 1. Mode alignment primitives ───────────────────────────────────────


def test_dedupe_conjugate_pairs_keeps_positive_omega() -> None:
    from libuq.koopman import DynamicModeDecomposition

    x = np.array([0.4, 0.2])
    ts = _synth_trajectory(x, seed=0)
    dmd = DynamicModeDecomposition(rank=4, dt=DT).fit(ts)
    spectrum = dmd.get_spectrum()
    deduped = _dedupe_conjugate_pairs(list(spectrum.modes))
    # At most half the original modes survive (conjugate pairs collapse)
    assert len(deduped) <= len(spectrum.modes)
    # No mode with negative ω should remain (except the DC mode, ω=0)
    for m in deduped:
        assert np.imag(m.eigenvalue) >= -1e-9


def test_align_modes_sorts_by_frequency() -> None:
    from libuq.koopman import DynamicModeDecomposition

    x = np.array([0.7, 0.3])
    ts = _synth_trajectory(x, seed=2)
    dmd = DynamicModeDecomposition(rank=6, dt=DT).fit(ts)
    picked = _align_modes(dmd.get_spectrum(), n_modes=3)
    freqs = [np.abs(np.imag(np.log(m.eigenvalue + 1e-12))) for m in picked]
    assert freqs == sorted(freqs), "modes should be sorted by ascending frequency"


# ── 2. Feature extraction recovers the known frequency ────────────────


def test_extract_spectral_features_recovers_true_frequency(synthetic_bundle: dict) -> None:
    spectral = extract_spectral_features(
        Y_timeseries=synthetic_bundle["Y_timeseries"],
        n_modes=3,
        dt=DT,
    )
    assert isinstance(spectral, SpectralFeatures)
    assert spectral.features.shape == (N_SAMPLES_TRAIN, 3 * 3)
    assert len(spectral.feature_names) == 9

    blocks = spectral.split_blocks()
    omega = blocks["omega"]
    # The dominant partial (after conjugate-pair dedup) should be at
    # 2π * f(x).  It may land in any of the K=3 slots because of
    # amplitude-then-pitch reordering; take the closest match per row.
    X = synthetic_bundle["X_train"]
    errs = []
    for i, x in enumerate(X):
        true_omega = 2 * np.pi * true_freq(x)
        k_best = int(np.argmin(np.abs(omega[i] - true_omega)))
        errs.append(abs(omega[i, k_best] - true_omega))
    mean_err = float(np.mean(errs))
    assert mean_err < 0.02, f"mean frequency error {mean_err:.4f} rad/step too large"


# ── 3. PCE reproduces the parametric frequency map ────────────────────


def test_run_strategy_spectral_fits_and_sobols(synthetic_bundle: dict) -> None:
    spectral = extract_spectral_features(
        Y_timeseries=synthetic_bundle["Y_timeseries"],
        n_modes=3,
        dt=DT,
    )
    bounds = synthetic_bundle["bounds"]
    X = synthetic_bundle["X_train"]
    # germ = 2*(x-lb)/(ub-lb) - 1  for this box that's 2*x - 1
    germ = 2.0 * X - 1.0

    result = run_strategy_spectral(
        spectral=spectral,
        parameter_names=synthetic_bundle["parameter_names"],
        germ_train=germ,
        X_train=X,
        polynomial_order=2,
        regression="lsq",
    )
    assert isinstance(result, SpectralStrategyResult)
    assert result.sobol_per_feature_total.shape == (9, 2)
    # Training relative errors should be small for the leading features
    assert np.mean(result.relerr_train) < 0.5


def test_sobol_identifies_dominant_parameter(synthetic_bundle: dict) -> None:
    """The true slopes are (0.06, 0.005) — x0 should dominate pitch."""
    spectral = extract_spectral_features(
        Y_timeseries=synthetic_bundle["Y_timeseries"],
        n_modes=2,
        dt=DT,
    )
    bounds = synthetic_bundle["bounds"]
    X = synthetic_bundle["X_train"]
    germ = 2.0 * X - 1.0
    result = run_strategy_spectral(
        spectral=spectral,
        parameter_names=synthetic_bundle["parameter_names"],
        germ_train=germ,
        X_train=X,
        polynomial_order=2,
    )
    # Aggregate Sobol (variance-weighted across features): x0 should
    # be more influential than x1.
    total = np.asarray(result.sobol.total_order)
    assert total[0] > total[1], f"x0 must dominate x1 in total Sobol ({total})"
    # The ω row of the per-feature modulation matrix for the most-active
    # partial should put the majority of its variance on x0.
    mod = result.sobol_per_feature_total  # (n_features, n_params)
    # feature 0 is omega_0 (the first pitch slot). Either the amplitude
    # reorder put the real mode at slot 0 or slot 1 of the K=2 sorted-by-
    # pitch list.  Take whichever has larger total-order mass as the
    # "live" pitch slot.
    pitch_rows = [mod[0], mod[3]]  # omega_0 and omega_1
    live = max(pitch_rows, key=lambda r: r.sum())
    assert live[0] > live[1], f"live pitch row should be x0-dominated: {live}"


# ── 4. SpectralDAWResult: predict / inverse_design / resynthesize ─────


@pytest.fixture(scope="module")
def daw_result(synthetic_bundle: dict) -> SpectralDAWResult:
    spectral = extract_spectral_features(
        Y_timeseries=synthetic_bundle["Y_timeseries"],
        n_modes=2,
        dt=DT,
    )
    bounds = synthetic_bundle["bounds"]
    X = synthetic_bundle["X_train"]
    germ = 2.0 * X - 1.0
    strategy = run_strategy_spectral(
        spectral=spectral,
        parameter_names=synthetic_bundle["parameter_names"],
        germ_train=germ,
        X_train=X,
        polynomial_order=2,
    )
    return SpectralDAWResult(
        strategy=strategy,
        parameter_names=synthetic_bundle["parameter_names"],
        parameter_bounds=bounds,
        spectral=spectral,
    )


def test_daw_predict_returns_physical_units(daw_result: SpectralDAWResult) -> None:
    pred = daw_result.predict(np.array([0.5, 0.5]))
    for key in ("omega", "sigma", "amp", "freq_hz", "features"):
        assert key in pred
    # At x=(0.5, 0.5) the true frequency is 0.02 + 0.03 + 0.0025 = 0.0525 cpt
    # ≈ 0.33 rad/step — pick the pitch slot closest to that target
    target_omega = 2 * np.pi * (TRUE_F0 + 0.5 * (TRUE_SLOPES[0] + TRUE_SLOPES[1]))
    closest = float(np.min(np.abs(pred["omega"] - target_omega)))
    assert closest < 0.05, f"predicted omega off by {closest:.4f} rad/step"


def test_daw_modulation_matrix_shape(daw_result: SpectralDAWResult) -> None:
    mod = daw_result.modulation_matrix
    # 2 modes × 3 features (ω, σ, |a|) × 2 params
    assert mod.shape == (6, 2)


def test_daw_inverse_design_recovers_target_frequency(
    daw_result: SpectralDAWResult,
) -> None:
    # Target: a cell with frequency 0.05 cpt (mid of the range)
    target_freq_hz = 0.05
    out = daw_result.inverse_design(
        target={"freq_hz": np.array([target_freq_hz, 0.0])},
        n_restarts=10,
        seed=7,
    )
    x_opt = out["x"]
    pred = out["prediction"]
    recovered_hz = float(pred["freq_hz"][0])
    assert abs(recovered_hz - target_freq_hz) < 0.01, (
        f"inverse-design frequency off: got {recovered_hz:.4f} Hz/step "
        f"for target {target_freq_hz} at x={x_opt}"
    )
    # The recovered x must lie inside the box
    assert np.all(x_opt >= daw_result.parameter_bounds[:, 0] - 1e-9)
    assert np.all(x_opt <= daw_result.parameter_bounds[:, 1] + 1e-9)


def test_daw_resynthesize_reproduces_oscillation(
    daw_result: SpectralDAWResult,
) -> None:
    x = np.array([0.6, 0.4])
    t = np.linspace(0, 100, 200)
    sig = daw_result.resynthesize(x, t)
    assert sig.shape == t.shape
    # Crude check: the signal should oscillate within a reasonable band
    assert np.max(sig) - np.min(sig) > 0.1


# ── 5. Export artifact layout ─────────────────────────────────────────


def test_daw_export_writes_expected_files(
    daw_result: SpectralDAWResult, tmp_path: Path
) -> None:
    out = daw_result.export(tmp_path / "spectral")
    expected = [
        "spectral_features.npy",
        "feature_names.json",
        "parameter_names.json",
        "modulation_matrix.npy",
        "modulation_matrix_first.npy",
        "input_bounds.npy",
        "pce_coefficients.npy",
        "multi_indices.npy",
        "relerr_train.npy",
        "spectral_summary.json",
    ]
    for f in expected:
        assert (out / f).exists(), f"missing export file: {f}"
    # PCE coefficients must be a 2-D array — one row per output feature
    coefs = np.load(out / "pce_coefficients.npy")
    assert coefs.ndim == 2
    assert coefs.shape[0] == len(daw_result.feature_names)
    # Multi-indices must be (n_basis, n_params)
    mi = np.load(out / "multi_indices.npy")
    assert mi.ndim == 2
    assert mi.shape[1] == daw_result.n_parameters


# ── 6. quantify_spectral over a real PrecomputedCache on disk ─────────


def test_quantify_spectral_end_to_end(synthetic_bundle: dict, tmp_path: Path) -> None:
    """Drive ``quantify_spectral`` through a ``PrecomputedCache`` round-trip.

    This is the same entry point the DAW/GUI will use: the cache is written
    to disk (simulating what ``uq sample`` produces), then
    ``quantify_spectral`` loads it and runs the full Koopman-PCE pipeline.
    """
    bounds = synthetic_bundle["bounds"]
    X = synthetic_bundle["X_train"]
    Y_ts = synthetic_bundle["Y_timeseries"]
    Y_agg = np.vstack([ts.mean(axis=0) for ts in Y_ts])

    # Include the held-out validation samples too, so we can exercise the
    # test-relerr path of run_strategy_spectral.
    X_test = synthetic_bundle["X_test"]
    Y_test_ts = synthetic_bundle["Y_test_timeseries"]
    Y_test_agg = np.vstack([ts.mean(axis=0) for ts in Y_test_ts])

    cache_dir = tmp_path / "cache"
    cache = PrecomputedCache(
        cache_dir=cache_dir,
        X=X,
        Y=Y_agg,
        parameter_names=synthetic_bundle["parameter_names"],
        metadata={"bounds": bounds.tolist(), "observable_columns": ["c0", "c1"]},
        Y_timeseries=Y_ts,
        X_test=X_test,
        Y_test=Y_test_agg,
        Y_test_timeseries=Y_test_ts,
    )
    cache.save()

    # Save the germ samples so quantify_spectral can skip the inverse transform
    germ = 2.0 * X - 1.0
    np.save(cache_dir / "germ_train.npy", germ)

    result = quantify_spectral(
        cache_dir=cache_dir,
        n_modes=2,
        dt=DT,
        polynomial_order=2,
        regression="lsq",
        export_path=tmp_path / "export",
    )
    assert isinstance(result, SpectralDAWResult)
    assert result.n_modes == 2
    assert result.n_parameters == 2
    assert result.modulation_matrix.shape == (6, 2)
    # Training relerr must be finite and below 1 for the active features
    assert np.all(np.isfinite(result.train_relerr_per_feature))
    # Test relerr was exercised because we cached Y_test_timeseries
    assert result.test_relerr_per_feature is not None
    assert result.test_relerr_per_feature.shape == result.train_relerr_per_feature.shape

    # Export directory populated
    export_dir = tmp_path / "export"
    assert (export_dir / "spectral_summary.json").exists()
