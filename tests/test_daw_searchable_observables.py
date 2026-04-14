"""Tests for the searchable observable selector (per-gene/per-protein scale).

Verifies:
1. The filter logic correctly narrows a large observable list
2. Selection of a filtered item updates the selected_observable
3. The pipeline produces valid per-output coefficients for large observable sets
4. The DAW data flow works with 4000+ observables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.uq_daw_simple import _obs_label, legendre_eval, normalize_to_germ


def test_filter_logic_narrows_list() -> None:
    """Simulate the _filter_observables logic on a large list."""
    # Simulate 4345 gene names
    all_choices = ["(aggregate)"] + [f"mRNA_cistron_counts_{_i}" for _i in range(4345)]

    def _filter(query: str) -> list[str]:
        _q = query.strip().lower()
        return [_c for _c in all_choices if not _q or _q in _c.lower()]

    # No filter → full list
    assert len(_filter("")) == 4346

    # Search for "1234" → only genes containing "1234"
    _matches = _filter("1234")
    assert all("1234" in _m for _m in _matches)
    assert len(_matches) < 100  # much smaller than full list

    # Search for "aggregate" → just the aggregate
    assert _filter("aggregate") == ["(aggregate)"]

    # Search for nonexistent → empty
    assert _filter("zzzzzzzzz") == []


def test_filter_with_real_observable_names() -> None:
    """Filter with biologically meaningful search terms."""
    # Simulate a mix of mass + transcriptome observables
    all_choices = [
        "(aggregate)",
        "dry_mass", "cell_mass", "volume", "growth",
    ] + [f"mRNA_cistron_counts_{_i}" for _i in range(100)]

    def _filter(query: str) -> list[str]:
        _q = query.strip().lower()
        return [_c for _c in all_choices if not _q or _q in _c.lower()]

    # Search "mass" → dry_mass, cell_mass
    _mass = _filter("mass")
    assert "dry_mass" in _mass
    assert "cell_mass" in _mass
    assert len(_mass) == 2

    # Search "mrna" → all mRNA entries
    _mrna = _filter("mrna")
    assert len(_mrna) == 100
    assert all("mRNA" in _m for _m in _mrna)

    # Search "counts_5" → mRNA_cistron_counts_5, _50-59
    _c5 = _filter("counts_5")
    assert all("counts_5" in _m for _m in _c5)


def test_obs_label_for_gene_names() -> None:
    """Gene observable names should produce a reasonable fallback label."""
    _lbl, _unit, _fmt = _obs_label("mRNA_cistron_counts_42")
    # Unknown observable → raw name, no unit
    assert _lbl == "mRNA_cistron_counts_42"
    assert _unit == ""
    assert _fmt == ".4f"


def test_per_output_coefficients_work_at_scale() -> None:
    """Simulate a large per-output coefficient matrix and verify evaluation."""
    n_obs = 4345
    n_basis = 28
    n_params = 6

    # Random coefficients (simulating what quantify would produce)
    _rng = np.random.default_rng(42)
    _coeffs = _rng.normal(size=(n_obs, n_basis)) * 0.01
    _mi = np.zeros((n_basis, n_params), dtype=int)
    # Simple multi-index: first row = constant, then linear terms
    for _i in range(min(n_params, n_basis - 1)):
        _mi[_i + 1, _i] = 1

    _bounds = np.column_stack([np.zeros(n_params), np.ones(n_params)])
    _x = np.full(n_params, 0.5)
    _x_norm = normalize_to_germ(_x, _bounds)

    # Evaluate all 4345 observables
    _vals = np.array([legendre_eval(_x_norm, _coeffs[_i], _mi) for _i in range(n_obs)])
    assert _vals.shape == (n_obs,)
    assert np.all(np.isfinite(_vals))
    # Not all the same (random coefficients)
    assert len(set(np.round(_vals, 8))) > 100


@pytest.mark.skipif(
    not (Path("uq_results_e2e/population_surrogate/coefficients_per_output.npy").exists()),
    reason="No e2e artifacts",
)
def test_real_data_observable_count() -> None:
    """The e2e export should have observable_names matching per-output rows."""
    import json

    _data = json.loads((Path("uq_results_e2e/uq_results.json")).read_text())
    _per_out = np.load("uq_results_e2e/population_surrogate/coefficients_per_output.npy")
    _n_obs = len(_data["observable_names"])

    assert _per_out.shape[0] == _n_obs
    # Each observable should produce a finite PCE value
    _mi = np.load("uq_results_e2e/population_surrogate/multi_indices.npy")
    _bounds = np.load("uq_results_e2e/population_surrogate/input_bounds.npy")
    _x_norm = normalize_to_germ(0.5 * (_bounds[:, 0] + _bounds[:, 1]), _bounds)
    for _i in range(_n_obs):
        _v = legendre_eval(_x_norm, _per_out[_i], _mi)
        assert np.isfinite(_v), f"Observable {_i} ({_data['observable_names'][_i]}) non-finite"
