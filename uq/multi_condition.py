"""Cross-condition global sensitivity analysis.

**Extension beyond RFC006** — RFC006 specifies 4 aggregation strategies
within a single growth condition.  This module extends that framework
to *multiple* conditions by leveraging vEcoli's multi-parca infrastructure.
Each condition runs the full RFC006-compliant pipeline independently;
cross-condition comparison is a derived analysis layer on top.

The same parameter samples (X) are evaluated under each condition;
Sobol indices are computed per condition and then compared to identify
universal vs. condition-specific drivers.

Two-stage API (mirrors ``uq.workflow``):

    # Stage 1: sample under multiple conditions
    uq sample ... --conditions glucose_minimal --conditions glucose_rich

    # Stage 2: quantify + cross-condition analysis
    result = quantify_multi_condition(cache_dir, sim_data_path)
    result.export(export_path)

The ``MultiConditionResult`` contains per-condition ``QuantifyResult``
objects plus cross-condition derived quantities: rank stability,
universal drivers, and condition-specific drivers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from uq.workflow import QuantifyResult, quantify

logger = logging.getLogger(__name__)


# ── Cross-condition metrics ──────────────────────────────────────────


def compute_rank_stability(
    per_condition: dict[str, QuantifyResult],
) -> np.ndarray:
    """Compute rank stability for each parameter across conditions.

    Stability = 1 - (std of ranks / max_possible_std).
    A parameter that ranks #1 in every condition has stability 1.0.
    A parameter whose rank varies maximally has stability near 0.0.

    Args:
        per_condition: Mapping of condition_id → QuantifyResult.

    Returns:
        Array of shape ``(n_params,)`` with stability scores in [0, 1].
    """
    if not per_condition:
        return np.array([])

    conditions = list(per_condition.keys())
    first = per_condition[conditions[0]]
    n_params = len(first.parameter_names)

    if n_params <= 1 or len(conditions) <= 1:
        return np.ones(n_params)

    # Collect S_Ti ranks per condition
    ranks = np.zeros((len(conditions), n_params))
    for ci, cond_id in enumerate(conditions):
        s_ti = per_condition[cond_id].strategy1.sobol.total_order
        if s_ti.ndim > 1:
            s_ti = np.mean(s_ti, axis=0)
        # Rank: highest S_Ti → rank 1
        order = np.argsort(-s_ti)
        for rank_pos, param_idx in enumerate(order):
            ranks[ci, param_idx] = rank_pos + 1

    # Stability: 1 - normalized std
    rank_std = np.std(ranks, axis=0)
    max_possible_std = (n_params - 1) / 2  # uniform over all ranks
    if max_possible_std > 0:
        stability = 1.0 - rank_std / max_possible_std
    else:
        stability = np.ones(n_params)

    return np.clip(stability, 0.0, 1.0)


def find_universal_drivers(
    per_condition: dict[str, QuantifyResult],
    threshold: float = 0.1,
) -> list[str]:
    """Parameters where S_Ti > threshold in ALL conditions."""
    if not per_condition:
        return []

    conditions = list(per_condition.keys())
    first = per_condition[conditions[0]]
    names = first.parameter_names

    universal = []
    for i, name in enumerate(names):
        all_above = True
        for cond_id in conditions:
            s_ti = per_condition[cond_id].strategy1.sobol.total_order
            if s_ti.ndim > 1:
                s_ti = np.mean(s_ti, axis=0)
            if s_ti[i] < threshold:
                all_above = False
                break
        if all_above:
            universal.append(name)

    return universal


def find_condition_specific(
    per_condition: dict[str, QuantifyResult],
    threshold: float = 0.1,
    low_threshold_factor: float = 0.5,
) -> dict[str, list[str]]:
    """Parameters important in one condition but not others.

    A parameter is condition-specific for condition C if:
    - S_Ti > threshold in C
    - S_Ti < threshold * low_threshold_factor in at least one other condition

    Returns:
        Dict mapping condition_id → list of condition-specific parameter names.
    """
    if not per_condition or len(per_condition) < 2:
        return {}

    conditions = list(per_condition.keys())
    first = per_condition[conditions[0]]
    names = first.parameter_names
    low_thresh = threshold * low_threshold_factor

    result: dict[str, list[str]] = {}
    for cond_id in conditions:
        specific = []
        s_ti_c = per_condition[cond_id].strategy1.sobol.total_order
        if s_ti_c.ndim > 1:
            s_ti_c = np.mean(s_ti_c, axis=0)

        for i, name in enumerate(names):
            if s_ti_c[i] < threshold:
                continue
            # Check if it's low in at least one other condition
            for other_id in conditions:
                if other_id == cond_id:
                    continue
                s_ti_o = per_condition[other_id].strategy1.sobol.total_order
                if s_ti_o.ndim > 1:
                    s_ti_o = np.mean(s_ti_o, axis=0)
                if s_ti_o[i] < low_thresh:
                    specific.append(name)
                    break

        if specific:
            result[cond_id] = specific

    return result


def compute_differential_sobol(
    per_condition: dict[str, QuantifyResult],
) -> dict[str, np.ndarray]:
    """Pairwise ΔS_Ti between all condition pairs.

    Returns:
        Dict mapping ``"condA_vs_condB"`` → array of shape ``(n_params,)``.
    """
    conditions = sorted(per_condition.keys())
    result = {}
    for i, a in enumerate(conditions):
        for b in conditions[i + 1 :]:
            s_a = per_condition[a].strategy1.sobol.total_order
            s_b = per_condition[b].strategy1.sobol.total_order
            if s_a.ndim > 1:
                s_a = np.mean(s_a, axis=0)
            if s_b.ndim > 1:
                s_b = np.mean(s_b, axis=0)
            result[f"{a}_vs_{b}"] = s_a - s_b
    return result


# ── Result container ──────────────────────────────────────────────────


@dataclass
class MultiConditionResult:
    """Cross-condition UQ results.

    Contains per-condition ``QuantifyResult`` objects plus derived
    cross-condition metrics.
    """

    conditions: list[str]
    per_condition: dict[str, QuantifyResult]
    rank_stability: np.ndarray
    universal_drivers: list[str]
    condition_specific: dict[str, list[str]]
    differential_sobol: dict[str, np.ndarray]

    @property
    def parameter_names(self) -> list[str]:
        first_key = self.conditions[0]
        return self.per_condition[first_key].parameter_names

    def export(self, export_dir: str | Path) -> Path:
        """Export per-condition results + cross-condition summary."""
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = self.parameter_names

        # Per-condition exports
        for cond_id, result in self.per_condition.items():
            cond_dir = out / f"condition_{cond_id}"
            result.export(cond_dir)

        # Cross-condition summary JSON
        summary: dict[str, Any] = {
            "conditions": self.conditions,
            "n_conditions": len(self.conditions),
            "parameter_names": names,
            "cross_condition": {
                "rank_stability": {
                    n: round(float(self.rank_stability[i]), 4)
                    for i, n in enumerate(names)
                },
                "universal_drivers": self.universal_drivers,
                "condition_specific": self.condition_specific,
                "differential_sobol": {
                    key: {n: round(float(v[i]), 6) for i, n in enumerate(names)}
                    for key, v in self.differential_sobol.items()
                },
            },
            "per_condition": {},
        }

        # Include per-condition S_Ti in the summary
        for cond_id, result in self.per_condition.items():
            s_ti = result.strategy1.sobol.total_order
            if s_ti.ndim > 1:
                s_ti = np.mean(s_ti, axis=0)
            summary["per_condition"][cond_id] = {
                "sobol_total_order": {
                    n: round(float(s_ti[i]), 6) for i, n in enumerate(names)
                },
            }

        (out / "multi_condition_results.json").write_text(
            json.dumps(summary, indent=2)
        )

        return out


# ── Public API ────────────────────────────────────────────────────────


def is_multi_condition_cache(cache_dir: str | Path) -> bool:
    """Check if a cache directory is a multi-condition cache."""
    return (Path(cache_dir) / "conditions.json").exists()


def quantify_multi_condition(
    cache_dir: str | Path,
    sim_data_path: str | Path,
    polynomial_order: int = 3,
    n_bins: int = 10,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    export_path: str | Path | None = None,
    seed: int | None = 42,
) -> MultiConditionResult:
    """Run per-condition quantification + cross-condition analysis.

    Auto-detects multi-condition cache by presence of ``conditions.json``.
    Runs the full UQPC workflow (all 4 strategies) for each condition,
    then computes cross-condition metrics.

    Args:
        cache_dir: Path to the multi-condition cache from ``uq sample --conditions``.
        sim_data_path: Path to ``simData.cPickle``.
        polynomial_order: PCE order.
        n_bins: Growth-stratified bins.
        regression: Fitting method ('lsq', 'bcs', 'anl').
        tolerance: BCS tolerance.
        export_path: If given, write all artifacts here.
        seed: Random seed.

    Returns:
        MultiConditionResult with per-condition and cross-condition analysis.
    """
    cache_dir = Path(cache_dir)
    cond_meta = json.loads((cache_dir / "conditions.json").read_text())
    conditions = cond_meta["conditions"]

    logger.info("Multi-condition quantify: %d conditions: %s", len(conditions), conditions)

    per_condition: dict[str, QuantifyResult] = {}
    for cond_id in conditions:
        cond_cache = cache_dir / f"condition_{cond_id}"
        if not cond_cache.exists():
            logger.warning("Condition cache not found: %s — skipping", cond_cache)
            continue

        logger.info("Quantifying condition: %s", cond_id)
        per_condition[cond_id] = quantify(
            cache_dir=str(cond_cache),
            sim_data_path=sim_data_path,
            polynomial_order=polynomial_order,
            n_bins=n_bins,
            regression=regression,
            tolerance=tolerance,
            seed=seed,
        )

    if not per_condition:
        raise RuntimeError(
            f"No conditions quantified from {cache_dir}. "
            f"Expected subdirectories: {['condition_' + c for c in conditions]}"
        )

    result = MultiConditionResult(
        conditions=list(per_condition.keys()),
        per_condition=per_condition,
        rank_stability=compute_rank_stability(per_condition),
        universal_drivers=find_universal_drivers(per_condition),
        condition_specific=find_condition_specific(per_condition),
        differential_sobol=compute_differential_sobol(per_condition),
    )

    if export_path is not None:
        result.export(export_path)
        logger.info("Multi-condition artifacts exported to %s", export_path)

    return result
