"""Tests for the Koopman-PCE spectral layer in ``app/uq_daw_simple.py``.

These tests verify the data layer only — everything that can be exercised
without instantiating a Tk widget:

- the module imports cleanly (so the DAW can be launched),
- :func:`app.uq_daw_simple.load_spectral_bundle` round-trips a
  :class:`uq.workflow_spectral.SpectralDAWResult` export,
- :func:`app.uq_daw_simple.predict_spectral_features` matches the live
  :meth:`SpectralDAWResult.predict` output exactly,
- :func:`app.uq_daw_simple.resynthesize_waveform` matches
  :meth:`SpectralDAWResult.resynthesize`,
- :func:`app.uq_daw_simple.inverse_design_bundle` recovers a target
  frequency on the synthetic oscillator benchmark.

Rendering of the new ``SpectralSynthCanvas`` / ``ModulationMatrixCanvas``
/ ``ResynthOscilloscope`` Tk widgets is NOT exercised here — no display is
available in CI.  The important invariant is that the helpers the canvases
consume agree with the ground-truth values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.uq_daw_simple import (
    ModulationMatrixCanvas,  # noqa: F401  — smoke-test import
    ResynthOscilloscope,  # noqa: F401
    SpectralBundle,
    SpectralSynthCanvas,  # noqa: F401
    evaluate_multi_output_pce,
    inverse_design_bundle,
    load_spectral_bundle,
    predict_spectral_features,
    resynthesize_waveform,
)
from uq.workflow_spectral import (
    SpectralDAWResult,
    extract_spectral_features,
    run_strategy_spectral,
)


# Reuse the same synthetic oscillator as test_workflow_spectral.py so the
# two test suites stay in sync.

TRUE_F0 = 0.02
TRUE_SLOPES = (0.06, 0.005)
TRUE_DAMPING = -0.001
N_SAMPLES = 40
N_TIMESTEPS = 300
DT = 1.0


def _mktraj(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = TRUE_F0 + TRUE_SLOPES[0] * x[0] + TRUE_SLOPES[1] * x[1]
    t = np.arange(N_TIMESTEPS, dtype=float)
    decay = np.exp(TRUE_DAMPING * t)
    noise = rng.normal(scale=0.01, size=(N_TIMESTEPS, 2))
    return np.stack(
        [decay * np.cos(2 * np.pi * f * t), decay * np.cos(2 * np.pi * f * t - 0.4)],
        axis=1,
    ) + noise


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[SpectralDAWResult, Path]:
    """Fit a spectral DAW result and export it to disk.

    Returns the live ``SpectralDAWResult`` AND the export directory so the
    tests can compare live predictions to the ones rebuilt from the bundle.
    """
    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_SAMPLES, 2))
    Y_ts = [_mktraj(x, seed=i) for i, x in enumerate(X)]

    spectral = extract_spectral_features(Y_timeseries=Y_ts, n_modes=2, dt=DT)
    germ = 2.0 * X - 1.0
    strategy = run_strategy_spectral(
        spectral=spectral,
        parameter_names=["x0", "x1"],
        germ_train=germ,
        X_train=X,
        polynomial_order=2,
        regression="lsq",
    )
    live = SpectralDAWResult(
        strategy=strategy,
        parameter_names=["x0", "x1"],
        parameter_bounds=bounds,
        spectral=spectral,
    )

    export_dir = tmp_path_factory.mktemp("spectral_export")
    live.export(export_dir)
    return live, export_dir


def test_module_imports_without_tk() -> None:
    """Sanity check: the DAW module and all its canvases import without a
    display.  Tk widget *instantiation* is skipped — only the class
    references are touched here.
    """
    from app import uq_daw_simple

    assert hasattr(uq_daw_simple, "UQDawSimpleApp")
    assert hasattr(uq_daw_simple, "SpectralSynthCanvas")
    assert hasattr(uq_daw_simple, "ModulationMatrixCanvas")
    assert hasattr(uq_daw_simple, "ResynthOscilloscope")
    assert hasattr(uq_daw_simple, "load_spectral_bundle")
    assert hasattr(uq_daw_simple, "predict_spectral_features")
    assert hasattr(uq_daw_simple, "resynthesize_waveform")
    assert hasattr(uq_daw_simple, "inverse_design_bundle")


def test_load_spectral_bundle_roundtrip(exported_bundle) -> None:
    live, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None, "bundle failed to load"
    assert isinstance(bundle, SpectralBundle)
    assert bundle.parameter_names == live.parameter_names
    assert bundle.feature_names == live.feature_names
    assert bundle.n_modes == live.n_modes
    assert bundle.n_parameters == live.n_parameters
    assert bundle.input_bounds.shape == live.parameter_bounds.shape
    np.testing.assert_allclose(bundle.input_bounds, live.parameter_bounds)
    assert bundle.modulation_matrix.shape == live.modulation_matrix.shape
    np.testing.assert_allclose(bundle.modulation_matrix, live.modulation_matrix)


def test_load_spectral_bundle_missing_files_returns_none(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_spectral_bundle(empty) is None


def test_predict_spectral_features_matches_live(exported_bundle) -> None:
    """``app.uq_daw_simple.predict_spectral_features`` must produce the
    same numbers as ``SpectralDAWResult.predict`` for the same x.
    """
    live, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None

    rng = np.random.default_rng(123)
    for _ in range(8):
        x = rng.uniform(0.0, 1.0, size=2)
        live_pred = live.predict(x)
        bundle_pred = predict_spectral_features(bundle, x)
        for key in ("omega", "sigma", "amp", "freq_hz"):
            np.testing.assert_allclose(
                live_pred[key],
                bundle_pred[key],
                atol=1e-8,
                err_msg=f"bundle disagrees with live on {key} at x={x}",
            )


def test_evaluate_multi_output_pce_matches_per_feature(exported_bundle) -> None:
    """Multi-output evaluation must match per-feature evaluation row-wise."""
    live, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None

    x_norm = np.array([0.2, -0.3])
    y_multi = evaluate_multi_output_pce(
        x_norm, bundle.pce_coefficients, bundle.multi_indices
    )
    # Manually evaluate each feature using the same shared-basis formula
    # as the fallback path in app/uq_daw_simple.py
    from app.uq_daw_simple import legendre_eval

    for i in range(bundle.n_features):
        expected = legendre_eval(
            x_norm, bundle.pce_coefficients[i], bundle.multi_indices
        )
        np.testing.assert_allclose(y_multi[i], expected, atol=1e-10)


def test_resynthesize_waveform_matches_live(exported_bundle) -> None:
    live, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None

    x = np.array([0.6, 0.3])
    t = np.linspace(0, 80, 160)
    live_sig = live.resynthesize(x, t)
    bundle_sig = resynthesize_waveform(bundle, x, t)
    np.testing.assert_allclose(live_sig, bundle_sig, atol=1e-8)


def test_inverse_design_bundle_recovers_target(exported_bundle) -> None:
    """Inverse design on the bundle should hit a given target frequency."""
    _, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None

    # Mid-of-range target
    target_hz = 0.05
    out = inverse_design_bundle(
        bundle,
        target={"freq_hz": np.array([target_hz, 0.0])},
        n_restarts=12,
        seed=3,
    )
    pred = out["prediction"]
    recovered = float(pred["freq_hz"][0])
    assert abs(recovered - target_hz) < 0.01, (
        f"bundle inverse design got freq_hz={recovered:.4f} for target {target_hz}"
    )
    x = out["x"]
    assert np.all(x >= bundle.input_bounds[:, 0] - 1e-9)
    assert np.all(x <= bundle.input_bounds[:, 1] + 1e-9)


def test_spectral_bundle_optional_fields_loaded(exported_bundle) -> None:
    _, export_dir = exported_bundle
    bundle = load_spectral_bundle(export_dir)
    assert bundle is not None
    # These were written by the export — should be populated
    assert bundle.relerr_train is not None
    assert bundle.training_features is not None
    # relerr_test was NOT written (no test samples in fixture)
    assert bundle.relerr_test is None
    # Summary JSON was written
    assert bundle.summary
    assert bundle.summary.get("n_modes") == bundle.n_modes
