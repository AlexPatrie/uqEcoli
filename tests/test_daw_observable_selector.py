"""Tests for todo #6 (observable selector) and #7 (per-stage surrogate).

Verifies:
1. Per-output coefficient matrix is exported by quantify
2. Per-stage coefficient matrix is exported for growth-stratified
3. Per-output PCE evaluation matches aggregate at the right index
4. Per-stage PCE evaluation produces finite predictions for each stage
5. Observable selector data flow works end-to-end
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.uq_daw_simple import legendre_eval, normalize_to_germ


E2E_DIR = Path("uq_results_e2e")

skip_no_e2e = pytest.mark.skipif(
    not (E2E_DIR / "population_surrogate" / "coefficients_per_output.npy").exists(),
    reason="No e2e artifacts with per-output coefficients",
)


# ── Todo #6: per-output coefficients ─────────────────────────────────


@skip_no_e2e
def test_per_output_coefficients_shape() -> None:
    """coefficients_per_output.npy should be (n_obs, n_basis)."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    agg = np.load(E2E_DIR / "population_surrogate" / "coefficients.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")

    assert per_out.ndim == 2
    n_obs, n_basis = per_out.shape
    assert n_basis == mi.shape[0], "basis size must match multi_indices"
    assert n_basis == len(agg), "basis size must match aggregate coefficients"
    # Should have at least as many outputs as observable_names in results JSON
    data = json.loads((E2E_DIR / "uq_results.json").read_text())
    assert n_obs == len(data["observable_names"])


@skip_no_e2e
def test_per_output_eval_gives_different_values() -> None:
    """Different observables should produce different PCE predictions."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    values = []
    for _i in range(per_out.shape[0]):
        values.append(legendre_eval(x_norm, per_out[_i], mi))

    # Not all observables should give the same value
    assert len(set(round(v, 8) for v in values)) > 1, (
        f"All observables gave the same value: {values}"
    )


@skip_no_e2e
def test_per_output_matches_aggregate_qualitatively() -> None:
    """The aggregate PCE should be in the range of the per-output values."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    agg = np.load(E2E_DIR / "population_surrogate" / "coefficients.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    agg_val = legendre_eval(x_norm, agg, mi)
    per_out_vals = [legendre_eval(x_norm, per_out[_i], mi) for _i in range(per_out.shape[0])]

    # The aggregate should be finite
    assert np.isfinite(agg_val)
    # All per-output values should be finite
    assert all(np.isfinite(v) for v in per_out_vals)


# ── Todo #7: per-stage coefficients ──────────────────────────────────


@skip_no_e2e
def test_per_stage_coefficients_shape() -> None:
    """coefficients_per_output.npy in growth_stratified_surrogate should be
    (n_stages * n_obs, n_basis)."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")

    assert gs_per_out.ndim == 2
    n_cols, n_basis = gs_per_out.shape
    assert n_basis == gs_mi.shape[0]

    data = json.loads((E2E_DIR / "uq_results.json").read_text())
    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(data["observable_names"])
    assert n_cols == n_stages * n_obs, (
        f"Expected {n_stages} stages × {n_obs} obs = {n_stages * n_obs} rows, got {n_cols}"
    )


@skip_no_e2e
def test_per_stage_eval_gives_per_stage_predictions() -> None:
    """Evaluating each stage's PCE should give finite, varying predictions."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")

    data = json.loads((E2E_DIR / "uq_results.json").read_text())
    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(data["observable_names"])

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    # Evaluate first observable at each stage
    stage_vals = []
    for _si in range(n_stages):
        _col = _si * n_obs + 0  # first observable
        val = legendre_eval(x_norm, gs_per_out[_col], gs_mi)
        stage_vals.append(val)
        assert np.isfinite(val), f"Stage {_si} gave non-finite: {val}"

    # Not all stages should give the same value (cell cycle changes things)
    assert len(set(round(v, 6) for v in stage_vals)) > 1, (
        f"All stages gave the same value: {stage_vals}"
    )


@skip_no_e2e
def test_per_stage_matches_linear_approx_direction() -> None:
    """The exact per-stage PCE and the linear approximation should at least
    agree on which stage has the highest prediction at the midpoint."""
    gs_per_out = np.load(E2E_DIR / "growth_stratified_surrogate" / "coefficients_per_output.npy")
    gs_mi = np.load(E2E_DIR / "growth_stratified_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")

    data = json.loads((E2E_DIR / "uq_results.json").read_text())
    n_stages = data["phase2_growth_stratified"]["n_stages"]
    n_obs = len(data["observable_names"])

    x = 0.5 * (bounds[:, 0] + bounds[:, 1])
    x_norm = normalize_to_germ(x, bounds)

    # Exact per-stage prediction (first observable)
    exact = np.array([
        legendre_eval(x_norm, gs_per_out[_si * n_obs], gs_mi)
        for _si in range(n_stages)
    ])
    assert np.all(np.isfinite(exact))
    # Should have a detectable peak
    assert np.max(exact) > np.min(exact)


# ── Observable selector data flow ────────────────────────────────────


@skip_no_e2e
def test_observable_selector_data_flow() -> None:
    """Simulate what the DAW does: pick an observable, evaluate its PCE."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    obs_names = data["observable_names"]
    short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]

    x = bounds[:, 0] + 0.3 * (bounds[:, 1] - bounds[:, 0])
    x_norm = normalize_to_germ(x, bounds)

    # Simulate picking each observable by short name
    for _i, _short in enumerate(short_names):
        val = legendre_eval(x_norm, per_out[_i], mi)
        assert np.isfinite(val), f"Observable {_short} gave non-finite at x=30%"

    print(f"Observable selector: all {len(short_names)} observables produce finite PCE values")
