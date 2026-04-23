"""Tests for cross-condition GSA (PR2 §1.1-1.2).

Tests the metric functions and MultiConditionResult using synthetic
QuantifyResult objects — no vEcoli needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from libuq.sensitivity import SobolIndices
from uq.multi_condition import (
    MultiConditionResult,
    compute_differential_sobol,
    compute_rank_stability,
    find_condition_specific,
    find_universal_drivers,
    is_multi_condition_cache,
)


def _make_quantify_result(
    total_order: np.ndarray,
    parameter_names: list[str],
    observable_names: list[str] | None = None,
) -> MagicMock:
    """Create a mock QuantifyResult with given Sobol total-order indices."""
    sobol = SobolIndices(
        first_order=total_order * 0.8,
        total_order=total_order,
        second_order=np.zeros((len(parameter_names), len(parameter_names))),
        parameter_names=parameter_names,
    )
    strategy1 = MagicMock()
    strategy1.sobol = sobol

    result = MagicMock()
    result.strategy1 = strategy1
    result.parameter_names = parameter_names
    result.observable_names = observable_names or ["obs_0"]
    result.strategy2 = {}
    result.strategy3 = {}
    result.strategy4_per_stage = []
    return result


PARAMS = ["kinetic_obj_weight", "secretion_penalty", "rnap_free", "elongation_rate"]


class TestComputeRankStability:
    """Test rank stability computation."""

    def test_identical_rankings(self):
        """Same ranking in all conditions → stability = 1.0."""
        per_cond = {
            "cond_a": _make_quantify_result(np.array([0.5, 0.3, 0.15, 0.05]), PARAMS),
            "cond_b": _make_quantify_result(np.array([0.6, 0.25, 0.1, 0.05]), PARAMS),
        }
        # Both rank: kinetic > secretion > rnap > elongation
        stability = compute_rank_stability(per_cond)
        assert stability.shape == (4,)
        np.testing.assert_array_equal(stability, np.ones(4))

    def test_swapped_rankings(self):
        """Top two params swap between conditions → reduced stability for both."""
        per_cond = {
            "cond_a": _make_quantify_result(np.array([0.5, 0.3, 0.1, 0.05]), PARAMS),
            "cond_b": _make_quantify_result(np.array([0.3, 0.5, 0.1, 0.05]), PARAMS),
        }
        stability = compute_rank_stability(per_cond)
        # Params 0 and 1 have rank [1,2] and [2,1] → std=0.5, stability < 1
        assert stability[0] < 1.0
        assert stability[1] < 1.0
        # Params 2 and 3 are stable
        assert stability[2] == 1.0
        assert stability[3] == 1.0

    def test_single_condition(self):
        """Single condition → all stabilities = 1.0 (no variation)."""
        per_cond = {
            "cond_a": _make_quantify_result(np.array([0.5, 0.3, 0.1, 0.05]), PARAMS),
        }
        stability = compute_rank_stability(per_cond)
        np.testing.assert_array_equal(stability, np.ones(4))

    def test_empty(self):
        """Empty dict → empty array."""
        stability = compute_rank_stability({})
        assert len(stability) == 0

    def test_single_parameter(self):
        """Single param → stability = 1.0."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.8]), ["param"]),
            "b": _make_quantify_result(np.array([0.6]), ["param"]),
        }
        stability = compute_rank_stability(per_cond)
        assert stability.shape == (1,)
        assert stability[0] == 1.0


class TestFindUniversalDrivers:
    """Test universal driver identification."""

    def test_all_above_threshold(self):
        """Parameter above threshold in all conditions is universal."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.5, 0.3, 0.02]), PARAMS[:3]),
            "b": _make_quantify_result(np.array([0.4, 0.2, 0.05]), PARAMS[:3]),
        }
        universal = find_universal_drivers(per_cond, threshold=0.1)
        assert "kinetic_obj_weight" in universal
        assert "secretion_penalty" in universal
        assert "rnap_free" not in universal  # below 0.1 in both

    def test_none_universal(self):
        """No param above threshold in all conditions."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.5, 0.01]), ["p1", "p2"]),
            "b": _make_quantify_result(np.array([0.01, 0.5]), ["p1", "p2"]),
        }
        universal = find_universal_drivers(per_cond, threshold=0.1)
        assert universal == []

    def test_empty(self):
        assert find_universal_drivers({}) == []


class TestFindConditionSpecific:
    """Test condition-specific driver identification."""

    def test_one_condition_specific(self):
        """Parameter high in A but low in B → condition-specific for A."""
        per_cond = {
            "glucose_min": _make_quantify_result(np.array([0.5, 0.3, 0.05]), PARAMS[:3]),
            "glucose_rich": _make_quantify_result(np.array([0.4, 0.02, 0.05]), PARAMS[:3]),
        }
        specific = find_condition_specific(per_cond, threshold=0.1)
        # secretion_penalty: 0.3 in min, 0.02 in rich → specific to min
        assert "glucose_min" in specific
        assert "secretion_penalty" in specific["glucose_min"]
        # kinetic_obj_weight: 0.5 and 0.4 → both above threshold, NOT specific

    def test_no_specific(self):
        """All params similar across conditions → no condition-specific."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.5, 0.3]), ["p1", "p2"]),
            "b": _make_quantify_result(np.array([0.45, 0.28]), ["p1", "p2"]),
        }
        specific = find_condition_specific(per_cond, threshold=0.1)
        assert specific == {}

    def test_single_condition(self):
        """Single condition → no comparison possible."""
        per_cond = {"a": _make_quantify_result(np.array([0.5]), ["p"])}
        assert find_condition_specific(per_cond) == {}


class TestComputeDifferentialSobol:
    """Test pairwise differential Sobol computation."""

    def test_two_conditions(self):
        """Two conditions → one pair."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.5, 0.3]), ["p1", "p2"]),
            "b": _make_quantify_result(np.array([0.3, 0.5]), ["p1", "p2"]),
        }
        diff = compute_differential_sobol(per_cond)
        assert "a_vs_b" in diff
        np.testing.assert_allclose(diff["a_vs_b"], [0.2, -0.2])

    def test_three_conditions(self):
        """Three conditions → three pairs."""
        per_cond = {
            "a": _make_quantify_result(np.array([0.5, 0.3]), ["p1", "p2"]),
            "b": _make_quantify_result(np.array([0.4, 0.4]), ["p1", "p2"]),
            "c": _make_quantify_result(np.array([0.1, 0.7]), ["p1", "p2"]),
        }
        diff = compute_differential_sobol(per_cond)
        assert len(diff) == 3  # a_vs_b, a_vs_c, b_vs_c


class TestMultiConditionResult:
    """Test the result container and export."""

    def _make_result(self) -> MultiConditionResult:
        per_cond = {
            "glucose_min": _make_quantify_result(np.array([0.5, 0.3, 0.1, 0.05]), PARAMS),
            "glucose_rich": _make_quantify_result(np.array([0.3, 0.02, 0.4, 0.15]), PARAMS),
        }
        return MultiConditionResult(
            conditions=["glucose_min", "glucose_rich"],
            per_condition=per_cond,
            rank_stability=compute_rank_stability(per_cond),
            universal_drivers=find_universal_drivers(per_cond),
            condition_specific=find_condition_specific(per_cond),
            differential_sobol=compute_differential_sobol(per_cond),
        )

    def test_parameter_names(self):
        result = self._make_result()
        assert result.parameter_names == PARAMS

    def test_export_creates_files(self, tmp_path):
        result = self._make_result()
        out = result.export(tmp_path / "export")

        assert (out / "multi_condition_results.json").exists()
        # Per-condition subdirs are created by QuantifyResult.export(),
        # which is mocked here — just verify the JSON was written
        summary = json.loads((out / "multi_condition_results.json").read_text())
        assert "glucose_min" in summary["per_condition"]
        assert "glucose_rich" in summary["per_condition"]

    def test_export_json_schema(self, tmp_path):
        result = self._make_result()
        out = result.export(tmp_path / "export")

        summary = json.loads((out / "multi_condition_results.json").read_text())
        assert summary["conditions"] == ["glucose_min", "glucose_rich"]
        assert summary["n_conditions"] == 2
        assert "cross_condition" in summary
        assert "rank_stability" in summary["cross_condition"]
        assert "universal_drivers" in summary["cross_condition"]
        assert "condition_specific" in summary["cross_condition"]
        assert "differential_sobol" in summary["cross_condition"]
        assert "per_condition" in summary
        assert "glucose_min" in summary["per_condition"]
        assert "sobol_total_order" in summary["per_condition"]["glucose_min"]

    def test_universal_and_specific_correct(self):
        result = self._make_result()
        # kinetic_obj_weight: 0.5 and 0.3 → both > 0.1 → universal
        assert "kinetic_obj_weight" in result.universal_drivers
        # secretion_penalty: 0.3 in min but 0.05 in rich → specific to min
        assert "glucose_min" in result.condition_specific
        assert "secretion_penalty" in result.condition_specific["glucose_min"]


class TestIsMultiConditionCache:
    """Test cache detection."""

    def test_with_conditions_json(self, tmp_path):
        (tmp_path / "conditions.json").write_text('{"conditions": ["a", "b"]}')
        assert is_multi_condition_cache(tmp_path) is True

    def test_without_conditions_json(self, tmp_path):
        assert is_multi_condition_cache(tmp_path) is False
