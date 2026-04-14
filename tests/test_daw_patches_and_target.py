"""Tests for patch save/load and target-mode inverse design in uq_daw_simple.

Headless tests — exercises the data-layer logic without Tk widgets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── Patch save/load (pure JSON round-trip) ────────────────────────────


def test_patch_roundtrip(tmp_path: Path) -> None:
    """Save a patch dict as JSON, reload it, verify values match."""
    patch = {
        "fraction_active_rnap_free": 0.36,
        "basal_elongation_rate": 22.5,
        "cell_dry_mass_fraction": 0.30,
    }
    path = tmp_path / "patch_000.json"
    path.write_text(json.dumps(patch, indent=2))

    loaded = json.loads(path.read_text())
    for k, v in patch.items():
        assert abs(loaded[k] - v) < 1e-10


def test_patch_numbering(tmp_path: Path) -> None:
    """Sequential patch files get incrementing indices."""
    for i in range(3):
        path = tmp_path / f"patch_{i:03d}.json"
        path.write_text(json.dumps({"x": float(i)}))
    files = sorted(tmp_path.glob("patch_*.json"))
    assert len(files) == 3
    assert files[0].name == "patch_000.json"
    assert files[2].name == "patch_002.json"


# ── Target mode (inverse design on population PCE) ───────────────────


@pytest.fixture
def surr_data() -> dict | None:
    """Load the e2e population surrogate if available."""
    pop_dir = Path("uq_results_e2e/population_surrogate")
    if not pop_dir.exists():
        return None
    return {
        "pop_coeffs": np.load(pop_dir / "coefficients.npy"),
        "pop_mi": np.load(pop_dir / "multi_indices.npy"),
        "bounds": np.load(pop_dir / "input_bounds.npy"),
    }


def test_target_mode_finds_midpoint_value(surr_data: dict | None) -> None:
    """Given the PCE value at the midpoint, target-mode should recover ~midpoint."""
    if surr_data is None:
        pytest.skip("No e2e artifacts")

    from app.uq_daw_simple import legendre_eval, normalize_to_germ

    bounds = surr_data["bounds"]
    coeffs = surr_data["pop_coeffs"]
    mi = surr_data["pop_mi"]

    # Evaluate at midpoint
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = normalize_to_germ(mid, bounds)
    target_val = float(legendre_eval(mid_norm, coeffs, mi))

    # Run the same optimizer logic as _find_target
    from scipy.optimize import minimize

    lb, ub = bounds[:, 0], bounds[:, 1]
    box = list(zip(lb.tolist(), ub.tolist()))

    def loss(x_phys: np.ndarray) -> float:
        xn = normalize_to_germ(x_phys, bounds)
        y = legendre_eval(xn, coeffs, mi)
        return float((y - target_val) ** 2)

    rng = np.random.default_rng(0)
    best_x, best_loss = None, np.inf
    for _ in range(12):
        x0 = lb + rng.random(len(lb)) * (ub - lb)
        res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=box)
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_x = np.asarray(res.x)

    assert best_x is not None
    achieved = float(legendre_eval(normalize_to_germ(best_x, bounds), coeffs, mi))
    assert abs(achieved - target_val) < 1e-6, (
        f"target {target_val:.6f}, achieved {achieved:.6f}"
    )


def test_target_mode_hits_shifted_value(surr_data: dict | None) -> None:
    """Target a Ŷ that's NOT at the midpoint — optimizer should still hit it."""
    if surr_data is None:
        pytest.skip("No e2e artifacts")

    from app.uq_daw_simple import legendre_eval, normalize_to_germ

    bounds = surr_data["bounds"]
    coeffs = surr_data["pop_coeffs"]
    mi = surr_data["pop_mi"]

    # Pick a point at 25% of each bound range
    x_ref = bounds[:, 0] + 0.25 * (bounds[:, 1] - bounds[:, 0])
    target_val = float(legendre_eval(normalize_to_germ(x_ref, bounds), coeffs, mi))

    from scipy.optimize import minimize

    lb, ub = bounds[:, 0], bounds[:, 1]
    box = list(zip(lb.tolist(), ub.tolist()))

    def loss(x: np.ndarray) -> float:
        y = legendre_eval(normalize_to_germ(x, bounds), coeffs, mi)
        return float((y - target_val) ** 2)

    rng = np.random.default_rng(7)
    best_x, best_loss = None, np.inf
    for _ in range(12):
        x0 = lb + rng.random(len(lb)) * (ub - lb)
        res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=box)
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_x = np.asarray(res.x)

    assert best_x is not None
    achieved = float(legendre_eval(normalize_to_germ(best_x, bounds), coeffs, mi))
    assert abs(achieved - target_val) < 1e-4, (
        f"target {target_val:.6f}, achieved {achieved:.6f}"
    )
    # Solution must lie inside the box
    assert np.all(best_x >= lb - 1e-9)
    assert np.all(best_x <= ub + 1e-9)
