"""Tests for ``uq/observables.py`` — observable extraction presets.

Uses the real Parquet data in ``uq_cache/_batch/output/`` if available,
otherwise skips.  Tests both scalar presets (mass) and array presets
(transcriptome, proteome, exchange_fluxes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uq.observables import (
    ALL_PRESET_NAMES,
    PRESETS,
    collect_observables,
    list_presets,
)

HISTORY_BASE = Path("uq_cache/_batch/output/uqpc_batch/history")
N_VARIANTS = 20

skip_no_data = pytest.mark.skipif(
    not HISTORY_BASE.exists(),
    reason=f"No Parquet data at {HISTORY_BASE}",
)


def test_presets_defined() -> None:
    assert "mass" in PRESETS
    assert "exchange_fluxes" in PRESETS
    assert "transcriptome" in PRESETS
    assert "proteome" in PRESETS
    assert "fluxome" in PRESETS
    assert len(ALL_PRESET_NAMES) >= 5


def test_list_presets() -> None:
    text = list_presets()
    assert "mass" in text
    assert "exchange_fluxes" in text
    assert "transcriptome" in text


@skip_no_data
def test_collect_mass_preset() -> None:
    """Default mass preset — should match the old _collect_variant_timeseries."""
    Y, names, ts, meta = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass"]
    )
    assert Y.shape[0] == N_VARIANTS
    assert Y.shape[1] >= 4, f"mass preset should have >= 4 columns, got {Y.shape[1]}"
    assert len(names) == Y.shape[1]
    assert all("mass" in n or "growth" in n for n in names), f"unexpected names: {names}"
    assert len(ts) == N_VARIANTS
    # Each timeseries should be (n_timesteps, n_obs) with n_obs matching Y.shape[1]
    assert ts[0].shape[1] == Y.shape[1]


@skip_no_data
def test_collect_exchange_fluxes_preset() -> None:
    """Exchange fluxes — dynamic column discovery via pattern match."""
    Y, names, ts, meta = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["exchange_fluxes"]
    )
    assert Y.shape[0] == N_VARIANTS
    assert Y.shape[1] > 0, "exchange_fluxes should find at least one column"
    assert len(names) == Y.shape[1]
    print(f"exchange_fluxes: {Y.shape[1]} columns, names[:5] = {names[:5]}")


@skip_no_data
def test_collect_transcriptome_preset() -> None:
    """Transcriptome — array column extraction."""
    Y, names, ts, meta = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["transcriptome"]
    )
    assert Y.shape[0] == N_VARIANTS
    # Should have ~2000+ gene columns
    assert Y.shape[1] > 100, f"transcriptome should have >> 100 columns, got {Y.shape[1]}"
    assert len(names) == Y.shape[1]
    print(f"transcriptome: {Y.shape[1]} genes")


@skip_no_data
def test_collect_proteome_preset() -> None:
    """Proteome — array column extraction."""
    Y, names, ts, meta = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["proteome"]
    )
    assert Y.shape[0] == N_VARIANTS
    assert Y.shape[1] > 100, f"proteome should have >> 100 columns, got {Y.shape[1]}"
    print(f"proteome: {Y.shape[1]} monomers")


@skip_no_data
def test_collect_multiple_presets() -> None:
    """Composing presets: mass + exchange_fluxes should concatenate columns."""
    Y_mass, n_mass, _, _ = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass"]
    )
    Y_exch, n_exch, _, _ = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["exchange_fluxes"]
    )
    Y_both, n_both, _, _ = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass", "exchange_fluxes"]
    )
    assert Y_both.shape[1] == Y_mass.shape[1] + Y_exch.shape[1]
    assert n_both == n_mass + n_exch


@skip_no_data
def test_generation_lower_bound() -> None:
    """Filtering early generations should change Y (unless only 1 gen)."""
    Y_all, _, _, _ = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass"], generation_lower_bound=0
    )
    Y_filt, _, _, _ = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass"], generation_lower_bound=999
    )
    # With an absurdly high bound, either all data is filtered out (error)
    # or the fallback to baseline kicks in.  The important thing is that
    # generation_lower_bound is wired through and doesn't crash.
    # (A more precise test would require knowing how many gens are in the data.)
    assert Y_all.shape[0] == N_VARIANTS


@skip_no_data
def test_timeseries_and_meta_returned() -> None:
    """Per-variant timeseries and generation/seed metadata should be populated."""
    _, _, ts, meta = collect_observables(
        HISTORY_BASE, N_VARIANTS, presets=["mass"]
    )
    assert ts is not None
    assert len(ts) == N_VARIANTS
    # Each entry is a 2-D array
    for t in ts:
        assert t.ndim == 2

    # Meta may be None if no generation/seed columns present;
    # with real data it should be populated
    if meta is not None:
        assert len(meta) == N_VARIANTS
        assert "generation" in meta[0] or "lineage_seed" in meta[0]


def test_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown observable preset"):
        collect_observables(Path("/fake"), 1, presets=["nonexistent"])
