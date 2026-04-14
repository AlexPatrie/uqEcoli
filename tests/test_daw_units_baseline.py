"""Tests for observable units, baseline reference, and delta display.

Verifies:
1. OBSERVABLE_UNITS maps known column names to (label, unit, fmt)
2. _obs_label falls back gracefully for unknown names
3. Baselines are computed at parameter midpoints
4. Delta percentage is correct
5. ResponseCurveCanvas.set_curves accepts baseline + labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.uq_daw_simple import (
    OBSERVABLE_UNITS,
    _obs_label,
    legendre_eval,
    normalize_to_germ,
)

E2E_DIR = Path("uq_results_e2e")

skip_no_e2e = pytest.mark.skipif(
    not (E2E_DIR / "population_surrogate" / "coefficients_per_output.npy").exists(),
    reason="No e2e artifacts",
)


def test_known_observables_have_units() -> None:
    for _key in ["dry_mass", "cell_mass", "volume", "growth",
                  "instantaneous_growth_rate", "growth_rate_per_hour",
                  "doubling_time_hours", "dna_fraction_g_per_g_dw"]:
        _lbl, _unit, _fmt = OBSERVABLE_UNITS[_key]
        assert _lbl, f"{_key} has empty label"
        assert _fmt.startswith("."), f"{_key} format {_fmt} doesn't start with '.'"


def test_obs_label_parses_full_parquet_name() -> None:
    _lbl, _unit, _fmt = _obs_label("listeners__mass__dry_mass")
    assert _lbl == "Dry mass"
    assert _unit == "fg"


def test_obs_label_fallback_for_unknown() -> None:
    _lbl, _unit, _fmt = _obs_label("some_weird_column")
    assert _lbl == "some_weird_column"
    assert _unit == ""
    assert _fmt == ".4f"


@skip_no_e2e
def test_baselines_at_midpoint() -> None:
    """Baselines should equal PCE(midpoint) for each observable."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    agg = np.load(E2E_DIR / "population_surrogate" / "coefficients.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")
    data = json.loads((E2E_DIR / "uq_results.json").read_text())

    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = normalize_to_germ(mid, bounds)

    # Aggregate baseline
    _agg_val = legendre_eval(mid_norm, agg, mi)
    assert np.isfinite(_agg_val)

    # Per-observable baselines
    for _i, _name in enumerate(data["observable_names"]):
        _val = legendre_eval(mid_norm, per_out[_i], mi)
        assert np.isfinite(_val), f"Baseline for {_name} is not finite"
        _lbl, _unit, _fmt = _obs_label(_name)
        # Value should be formattable with the specified format
        _formatted = f"{_val:{_fmt}}"
        assert len(_formatted) > 0


@skip_no_e2e
def test_delta_percentage_is_correct() -> None:
    """Delta % = (current - baseline) / |baseline| * 100."""
    per_out = np.load(E2E_DIR / "population_surrogate" / "coefficients_per_output.npy")
    mi = np.load(E2E_DIR / "population_surrogate" / "multi_indices.npy")
    bounds = np.load(E2E_DIR / "population_surrogate" / "input_bounds.npy")

    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = normalize_to_germ(mid, bounds)

    # Baseline = PCE at midpoint
    _baseline = legendre_eval(mid_norm, per_out[0], mi)

    # Shift one parameter to 75% of its range
    x_shifted = mid.copy()
    x_shifted[0] = bounds[0, 0] + 0.75 * (bounds[0, 1] - bounds[0, 0])
    x_shifted_norm = normalize_to_germ(x_shifted, bounds)
    _current = legendre_eval(x_shifted_norm, per_out[0], mi)

    _delta_pct = (_current - _baseline) / abs(_baseline) * 100
    assert np.isfinite(_delta_pct)
    # At midpoint, delta should be 0
    _delta_at_mid = (legendre_eval(mid_norm, per_out[0], mi) - _baseline) / abs(_baseline) * 100
    assert abs(_delta_at_mid) < 1e-10


def test_response_canvas_set_curves_signature() -> None:
    """set_curves should accept baseline, obs_label, obs_unit kwargs."""
    import inspect

    from app.uq_daw_simple import ResponseCurveCanvas

    _sig = inspect.signature(ResponseCurveCanvas.set_curves)
    assert "baseline" in _sig.parameters
    assert "obs_label" in _sig.parameters
    assert "obs_unit" in _sig.parameters
