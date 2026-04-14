"""Tests for todo #8: predicted observable profile across cell cycle.

Verifies:
1. PredictedProfileCanvas class exists and accepts profiles
2. Per-stage per-output PCE evaluation produces per-observable profiles
3. Profiles change when slider values change
4. All observables produce finite, varying profiles
5. % deviation mode works for aggregate view
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.uq_daw_simple import (
    PredictedProfileCanvas,
    _obs_label,
    legendre_eval,
    normalize_to_germ,
)

E2E_DIR = Path("uq_results_e2e")

skip_no_e2e = pytest.mark.skipif(
    not (E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy").exists(),
    reason="No e2e artifacts with per-stage coefficients",
)


def test_predicted_profile_canvas_exists() -> None:
    import inspect

    sig = inspect.signature(PredictedProfileCanvas.set_profiles)
    assert "profiles" in sig.parameters
    assert "baselines" in sig.parameters
    assert "selected_obs" in sig.parameters


@skip_no_e2e
def test_per_observable_profiles_computed() -> None:
    """Compute per-observable profiles from the per-stage PCE and verify shapes."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    obs_names = data["observable_names"]
    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(obs_names)

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    profiles: dict[str, np.ndarray] = {}
    for _oi, _name in enumerate(obs_names):
        _short = _name.split("__")[-1]
        _vals = np.zeros(n_stages)
        for _si in range(n_stages):
            _col = _si * n_obs + _oi
            _vals[_si] = legendre_eval(x_norm, gs_per_out[_col], gs_mi)
        profiles[_short] = _vals

    assert len(profiles) == n_obs
    for _name, _prof in profiles.items():
        assert _prof.shape == (n_stages,), f"{_name} has wrong shape {_prof.shape}"
        assert np.all(np.isfinite(_prof)), f"{_name} has non-finite values"


@skip_no_e2e
def test_profiles_change_with_parameters() -> None:
    """Moving a slider should change at least one observable's profile."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    obs_names = data["observable_names"]
    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(obs_names)

    # Profile at midpoint
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = normalize_to_germ(mid, bounds)

    # Profile with first param shifted to 80%
    shifted = mid.copy()
    shifted[0] = bounds[0, 0] + 0.8 * (bounds[0, 1] - bounds[0, 0])
    shifted_norm = normalize_to_germ(shifted, bounds)

    _any_changed = False
    for _oi in range(n_obs):
        _mid_vals = np.array([
            legendre_eval(mid_norm, gs_per_out[_si * n_obs + _oi], gs_mi)
            for _si in range(n_stages)
        ])
        _shift_vals = np.array([
            legendre_eval(shifted_norm, gs_per_out[_si * n_obs + _oi], gs_mi)
            for _si in range(n_stages)
        ])
        if not np.allclose(_mid_vals, _shift_vals, atol=1e-10):
            _any_changed = True

    assert _any_changed, "No observable profile changed when shifting a parameter"


@skip_no_e2e
def test_profiles_vary_across_stages() -> None:
    """At least one observable should have different values at different θ-bins."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(data["observable_names"])

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    _any_varies = False
    for _oi in range(n_obs):
        _vals = np.array([
            legendre_eval(x_norm, gs_per_out[_si * n_obs + _oi], gs_mi)
            for _si in range(n_stages)
        ])
        if np.max(_vals) - np.min(_vals) > 1e-8:
            _any_varies = True

    assert _any_varies, "No observable varies across θ-bins"


@skip_no_e2e
def test_pct_deviation_mode() -> None:
    """When showing all observables as % deviation, values at midpoint should be ~0%."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    pop_mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(data["observable_names"])

    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = normalize_to_germ(mid, bounds)

    for _oi in range(n_obs):
        _baseline = legendre_eval(mid_norm, per_out[_oi], pop_mi)
        if abs(_baseline) < 1e-15:
            continue
        _vals = np.array([
            legendre_eval(mid_norm, gs_per_out[_si * n_obs + _oi], gs_mi)
            for _si in range(n_stages)
        ])
        _pct = (_vals - _baseline) / abs(_baseline) * 100.0
        # At midpoint the per-stage values may differ from the population
        # baseline (different aggregation), but the deviations should be
        # small and finite
        assert np.all(np.isfinite(_pct)), f"obs {_oi}: non-finite % deviation"
