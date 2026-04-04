"""
Simplified RFC006 pipeline — biologically transparent implementation.

Four-strategy global sensitivity analysis for the vEcoli whole-cell model
(Macklin et al., Science 2020; Ahn-Horst et al., npj Syst Biol Appl 2022).

Strategy 1 (Population / Bulk)
    PCE surrogate fitted to time-averaged simulation outputs
    (dry mass, cell mass, volume, growth rate).  Sobol indices quantify
    which model parameters drive the most variance in bulk cellular
    phenotype across all cells and all times.

Strategy 2 (By Generation)
    PCE + Sobol per generation.  Controls for convergence toward
    steady-state growth — early generations may show transient dynamics.

Strategy 3 (By Lineage Seed)
    PCE + Sobol per lineage seed.  Controls for exogenous stochastic
    variance (gene expression noise, stochastic partitioning at division).

Strategy 4 (Growth-Stratified)
    The same PCE / Sobol machinery, but applied to outputs binned
    by growth progress (θ = normalized log dry mass).  This reveals
    how parameter importance *changes* as a cell grows from birth
    toward division.

Follows the UQPC workflow (https://sandialabs.github.io/pytuq/apps/uqpc.html):
    1. Input setup — parameter bounds from SimDataParameter specs
    2. Training data — (X, Y) from PrecomputedCache
    3. Surrogate construction — pytuq.surrogates.pce.PCE (lsq/bcs/anl)
    4. Sensitivity analysis — Sobol indices from PCE coefficients
    5. Post-processing — export to JSON + .npy artifacts

Methods:
    - PCRV.sampleGerm() — PyTUQ-native random sampling from germ measure
    - PCE surrogates via pytuq.surrogates.pce.PCE (Sandia National Labs)
    - Sobol indices from PCRV.computeSens() / computeTotSens() (Sudret, 2008)
    - Four aggregation strategies per RFC006 (no spectral decomposition)
    - Regression: lsq (default), bcs (Bayesian Compressed Sensing), anl (analytical)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from libuq.inputs import XSpace
from libuq.sampling import PrecomputedCache
from libuq.sensitivity import PCESurrogate, SobolIndices
from uq.growth import bin_by_growth_stage, compute_growth_fraction

logger = logging.getLogger(__name__)


# ── Biological parameter descriptions ──────────────────────────────
# Maps parameter attr_paths to descriptions grounded in the publications.

_PARAM_BIO_DESCRIPTIONS: dict[str, str] = {
    "fraction_active_rnap_free": (
        "Fraction of RNA polymerases actively transcribing when ppGpp-free. "
        "Controls rRNA/mRNA/tRNA synthesis rates and growth rate via the "
        "ppGpp regulatory circuit (Ahn-Horst et al. 2022, Fig. 1b). "
        "Tunable via relA/spoT mutations."
    ),
    "fraction_active_rnap_bound": (
        "Fraction of RNA polymerases actively transcribing when ppGpp-bound. "
        "ppGpp destabilizes RNAP open complex formation, reducing this "
        "fraction and downregulating stable RNA synthesis (Ahn-Horst et al. "
        "2022). Tunable via relA/spoT knockouts."
    ),
    "basal_elongation_rate": (
        "Ribosome elongation rate for non-ribosomal proteins (~22 aa/s). "
        "Controls translation capacity and protein synthesis rate. "
        "Experimentally tunable via sub-inhibitory chloramphenicol, fusidic "
        "acid, or growth temperature."
    ),
    "kinetic_objective_weight": (
        "Weight on kinetic vs homeostatic objective in FBA (~1e-7). Controls "
        "the balance between matching enzyme kinetics and maintaining "
        "metabolite homeostasis (Macklin et al. 2020)."
    ),
    "secretion_penalty_coeff": (
        "Penalty on metabolite secretion fluxes in FBA (~0.001). Higher "
        "values force the cell to retain metabolites. Controls acetate "
        "overflow metabolism, tunable via media composition."
    ),
    "cell_dry_mass_fraction": (
        "Fraction of total cell mass that is dry mass (~0.30). Determines "
        "the relationship between cell volume and biosynthetic capacity. "
        "Constrained by buoyant density measurements."
    ),
}


# ── Result container ────────────────────────────────────────────────


@dataclass
class SimplePipelineResult:
    """Complete output of the simplified UQ pipeline.

    Attributes:
        parameter_names: Input parameter names.
        population_sobol: Phase 1 / Strategy 1 Sobol indices (bulk).
        population_surrogate: Phase 1 PCE surrogate.
        per_stage_sobol: Phase 2 Sobol indices (one per growth bin).
        growth_surrogate: Phase 2 PCE surrogate.
        n_bins: Number of growth-progress bins.
        observable_names: Output observable names.
        per_generation_sobol: Strategy 2 — per-generation Sobol indices.
        per_seed_sobol: Strategy 3 — per-seed Sobol indices.
    """

    parameter_names: list[str]
    population_sobol: SobolIndices
    population_surrogate: PCESurrogate
    per_stage_sobol: list[SobolIndices]
    growth_surrogate: PCESurrogate
    n_bins: int
    observable_names: list[str]
    per_generation_sobol: dict[int, SobolIndices] | None = None
    per_seed_sobol: dict[int, SobolIndices] | None = None

    def export(self, export_dir: str | Path) -> Path:
        """Write all artifacts to *export_dir*."""
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Population Sobol
        pop_dir = out / "population_sobol"
        pop_dir.mkdir(exist_ok=True)
        np.save(pop_dir / "first_order.npy", self.population_sobol.first_order)
        np.save(pop_dir / "total_order.npy", self.population_sobol.total_order)
        (pop_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "parameter_names": self.parameter_names,
                    "description": "Phase 1: Sobol indices for population-averaged (bulk) outputs.",
                },
                indent=2,
            )
        )

        # Population surrogate
        self.population_surrogate.export(out / "population_surrogate")

        # Per-stage Sobol
        for i, sobol in enumerate(self.per_stage_sobol):
            stage_dir = out / f"growth_stage_{i}_sobol"
            stage_dir.mkdir(exist_ok=True)
            lo, hi = i / self.n_bins, (i + 1) / self.n_bins
            np.save(stage_dir / "first_order.npy", sobol.first_order)
            np.save(stage_dir / "total_order.npy", sobol.total_order)
            (stage_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameter_names": self.parameter_names,
                        "stage": i,
                        "theta_range": [lo, hi],
                        "description": (
                            f"Phase 2: Sobol indices for growth stage {i} "
                            f"(θ = {lo:.0%}–{hi:.0%} of mass doubling). "
                            f"θ is normalized log(dry_mass) — a monotonic "
                            f"proxy for growth progress."
                        ),
                    },
                    indent=2,
                )
            )

        # Growth-stratified surrogate
        self.growth_surrogate.export(out / "growth_stratified_surrogate")

        # Strategy 2: per-generation Sobol
        if self.per_generation_sobol is not None:
            for gen, sobol in self.per_generation_sobol.items():
                gen_dir = out / f"generation_{gen}_sobol"
                gen_dir.mkdir(exist_ok=True)
                np.save(gen_dir / "first_order.npy", sobol.first_order)
                np.save(gen_dir / "total_order.npy", sobol.total_order)
                (gen_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "parameter_names": self.parameter_names,
                            "generation": gen,
                            "description": (
                                f"Strategy 2: Sobol indices for generation {gen}. "
                                f"Controls for convergence toward steady-state growth."
                            ),
                        },
                        indent=2,
                    )
                )

        # Strategy 3: per-seed Sobol
        if self.per_seed_sobol is not None:
            for seed, sobol in self.per_seed_sobol.items():
                seed_dir = out / f"seed_{seed}_sobol"
                seed_dir.mkdir(exist_ok=True)
                np.save(seed_dir / "first_order.npy", sobol.first_order)
                np.save(seed_dir / "total_order.npy", sobol.total_order)
                (seed_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "parameter_names": self.parameter_names,
                            "lineage_seed": seed,
                            "description": (
                                f"Strategy 3: Sobol indices for lineage seed {seed}. "
                                f"Controls for exogenous stochastic variance."
                            ),
                        },
                        indent=2,
                    )
                )

        # Summary JSON — the primary artifact for downstream consumers
        summary = self._build_summary()
        (out / "uq_results.json").write_text(json.dumps(summary, indent=2))

        return out

    def _build_summary(self) -> dict[str, Any]:
        """Build the comprehensive summary JSON."""
        return {
            # ── Metadata ──
            "framework": "uq_simple (mass-based growth stratification)",
            "references": [
                "Macklin et al., Science 369:eaav3751, 2020",
                "Ahn-Horst et al., npj Syst Biol Appl 8:30, 2022",
            ],
            "methods": {
                "sampling": "PCRV.sampleGerm() (PyTUQ-native random germ sampling)",
                "surrogate": "Polynomial Chaos Expansion (PyTUQ, Legendre basis)",
                "sensitivity": "Variance-based Sobol indices (analytical from PCE coefficients)",
                "stratification": (
                    "Growth progress θ = normalized log(dry_mass). "
                    "θ = 0 at birth, θ = 1 at division. "
                    "Monotonic, no spectral decomposition."
                ),
            },
            # ── Parameters ──
            "parameters": {
                name: {
                    "index": i,
                    "biological_role": _PARAM_BIO_DESCRIPTIONS.get(name, ""),
                }
                for i, name in enumerate(self.parameter_names)
            },
            "n_parameters": len(self.parameter_names),
            # ── Observables ──
            "observable_names": self.observable_names,
            # ── Phase 1: Population (Bulk) ──
            "phase1_population": {
                "description": (
                    "Global sensitivity of time-averaged outputs to model "
                    "parameters. Answers: 'Across all cells and times, which "
                    "parameters drive the most variance in bulk phenotype?'"
                ),
                "sobol_total_order": {
                    n: round(float(v), 6) for n, v in zip(self.parameter_names, self.population_sobol.total_order)
                },
                "sobol_first_order": {
                    n: round(float(v), 6) for n, v in zip(self.parameter_names, self.population_sobol.first_order)
                },
            },
            # ── Strategy 2: By Generation ──
            "strategy2_by_generation": self._build_generation_summary(),
            # ── Strategy 3: By Lineage Seed ──
            "strategy3_by_seed": self._build_seed_summary(),
            # ── Phase 2: Growth-Stratified ──
            "phase2_growth_stratified": {
                "description": (
                    "Sensitivity analysis conditioned on growth progress. "
                    "Answers: 'Does parameter importance change as the cell "
                    "grows from birth toward division?' θ bins are equal "
                    "intervals of normalized log(dry_mass)."
                ),
                "n_stages": self.n_bins,
                "stages": [
                    {
                        "stage": i,
                        "theta_range": [
                            round(i / self.n_bins, 3),
                            round((i + 1) / self.n_bins, 3),
                        ],
                        "growth_description": _stage_description(i, self.n_bins),
                        "sobol_total_order": {
                            n: round(float(v), 6) for n, v in zip(self.parameter_names, s.total_order)
                        },
                    }
                    for i, s in enumerate(self.per_stage_sobol)
                ],
            },
        }

    def _build_generation_summary(self) -> dict[str, Any]:
        if self.per_generation_sobol is None:
            return {"status": "not_available", "reason": "No generation metadata in cache."}
        return {
            "description": (
                "Sensitivity analysis stratified by generation (RFC006 Strategy 2). "
                "Controls for convergence toward steady-state growth. "
                "Early generations may show transient dynamics."
            ),
            "n_generations": len(self.per_generation_sobol),
            "generations": [
                {
                    "generation": gen,
                    "sobol_total_order": {
                        n: round(float(v), 6) for n, v in zip(self.parameter_names, sobol.total_order)
                    },
                }
                for gen, sobol in sorted(self.per_generation_sobol.items())
            ],
        }

    def _build_seed_summary(self) -> dict[str, Any]:
        if self.per_seed_sobol is None:
            return {"status": "not_available", "reason": "No lineage seed metadata in cache."}
        return {
            "description": (
                "Sensitivity analysis stratified by lineage seed (RFC006 Strategy 3). "
                "Controls for exogenous stochastic variance (gene expression noise, "
                "stochastic partitioning at division)."
            ),
            "n_seeds": len(self.per_seed_sobol),
            "seeds": [
                {
                    "lineage_seed": seed,
                    "sobol_total_order": {
                        n: round(float(v), 6) for n, v in zip(self.parameter_names, sobol.total_order)
                    },
                }
                for seed, sobol in sorted(self.per_seed_sobol.items())
            ],
        }


def _stage_description(stage: int, n_bins: int) -> str:
    """Human-readable description of a growth stage."""
    frac = stage / n_bins
    if frac < 0.2:
        return "Early growth (recently divided, small cell)"
    elif frac < 0.4:
        return "Early-mid growth (active biosynthesis ramp-up)"
    elif frac < 0.6:
        return "Mid growth (steady-state biosynthesis)"
    elif frac < 0.8:
        return "Late-mid growth (approaching critical mass for division)"
    else:
        return "Late growth (pre-division, maximal cell size)"


# ── Core: PyTUQ PCE fitting + Sobol (UQPC workflow steps 3-4) ──────


def _fit_pce_and_sobol(
    X: np.ndarray,
    Y: np.ndarray,
    bounds: np.ndarray,
    parameter_names: list[str],
    polynomial_order: int,
    regression: str = "lsq",
    per_output: bool = False,
) -> tuple[SobolIndices, PCESurrogate]:
    """Fit PCE surrogate and compute Sobol indices via PyTUQ.

    Follows the UQPC workflow (https://sandialabs.github.io/pytuq/apps/uqpc.html):
      1. Scale X to germ space [-1, 1]
      2. Fit PCE per output column via ``pytuq.surrogates.pce.PCE``
      3. Extract Sobol main/total indices from PCRV coefficients
      4. Build exportable ``PCESurrogate``

    Args:
        X: Input samples in physical space, shape (n_samples, n_params).
        Y: Output values, shape (n_samples,) or (n_samples, n_outputs).
        bounds: Parameter bounds, shape (n_params, 2).
        parameter_names: Parameter names for SobolIndices.
        polynomial_order: PCE polynomial order.
        regression: PyTUQ regression method — 'lsq', 'bcs', or 'anl'.
        per_output: If True, return (n_outputs, n_params) Sobol arrays
            instead of variance-weighted scalars.
    """
    from pytuq.surrogates.pce import PCE as PyTUQ_PCE  # type: ignore[import-untyped]

    n_params = X.shape[1]

    # Step 1: Scale to germ space [-1, 1]
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb
    span[span == 0] = 1.0
    X_germ = 2.0 * (X - lb) / span - 1.0

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_outputs = Y.shape[1]

    # Step 2-3: Per-output PCE fit + Sobol extraction
    all_first, all_total = [], []
    last_pce = None

    for j in range(n_outputs):
        pce = PyTUQ_PCE(pce_dim=n_params, pce_order=polynomial_order, pce_type="LU")
        pce.set_training_data(X_germ, Y[:, j])
        pce.build(regression=regression)

        # For lsq/anl, build() does not sync coefficients to PCRV.
        # Explicit setCfs is required before computing Sobol indices.
        if regression != "bcs":
            pce.pcrv.setCfs([pce.lreg.cf])

        main = pce.pcrv.computeSens()[0]
        total = pce.pcrv.computeTotSens()[0]
        all_first.append(main)
        all_total.append(total)
        last_pce = pce

    # Step 4: Aggregate Sobol indices
    if n_outputs == 1:
        first_order = all_first[0]
        total_order = all_total[0]
    elif per_output:
        first_order = np.vstack(all_first)
        total_order = np.vstack(all_total)
    else:
        output_vars = np.var(Y, axis=0)
        total_var = output_vars.sum()
        weights = output_vars / total_var if total_var > 0 else np.ones(n_outputs) / n_outputs
        first_order = sum(w * fo for w, fo in zip(weights, all_first))
        total_order = sum(w * to for w, to in zip(weights, all_total))

    sobol = SobolIndices(
        first_order=first_order,
        total_order=total_order,
        parameter_names=parameter_names,
    )

    # Build exportable surrogate from the last fitted PCE
    coefficients = last_pce.pcrv.coefs[0] if last_pce else np.zeros(1)
    multi_indices = last_pce.pcrv.mindices[0] if last_pce else np.zeros((1, n_params), dtype=int)

    surrogate = PCESurrogate(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type="legendre",
        polynomial_order=polynomial_order,
        input_dim=n_params,
        output_dim=n_outputs,
        input_bounds=bounds,
    )

    return sobol, surrogate


# ── Phase 1: Bulk sensitivity ───────────────────────────────────────


def run_phase1(
    param_space: XSpace,
    X: np.ndarray,
    Y: np.ndarray,
    polynomial_order: int,
    regression: str = "lsq",
) -> tuple[SobolIndices, PCESurrogate]:
    """Strategy 1: PCE + Sobol on time-averaged (bulk) outputs."""
    bounds = np.array(param_space.parameter_bounds)
    return _fit_pce_and_sobol(
        X,
        Y,
        bounds,
        parameter_names=param_space.parameter_names,
        polynomial_order=polynomial_order,
        regression=regression,
    )


# ── Strategy 2: By-generation sensitivity ────────────────────────────


def _aggregate_by_group(
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    group_key: str,
) -> dict[int, np.ndarray]:
    """Aggregate timeseries by a metadata group key (generation or lineage_seed).

    For each unique group value, computes per-sample mean observables
    and stacks them into (n_samples, n_obs).

    Returns:
        Dict mapping group value -> Y array of shape (n_samples, n_obs).
    """
    n_samples = len(Y_timeseries)

    # Find the set of groups present across ALL samples
    group_sets = []
    for meta in Y_timeseries_meta:
        if group_key not in meta:
            return {}
        group_sets.append(set(meta[group_key].tolist()))

    common_groups = sorted(set.intersection(*group_sets)) if group_sets else []
    if not common_groups:
        return {}

    result: dict[int, np.ndarray] = {}
    n_obs = Y_timeseries[0].shape[1]
    for g in common_groups:
        Y_g = np.zeros((n_samples, n_obs))
        for i in range(n_samples):
            mask = Y_timeseries_meta[i][group_key] == g
            if np.any(mask):
                Y_g[i] = Y_timeseries[i][mask].mean(axis=0)
        result[int(g)] = Y_g

    return result


def run_by_generation(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    polynomial_order: int,
    regression: str = "lsq",
) -> dict[int, SobolIndices]:
    """Strategy 2: PCE + Sobol per generation.

    Aggregates each sample's timeseries by generation, then runs
    Phase-1-style bulk GSA independently per generation.  This controls
    for convergence toward steady-state growth — early generations may
    show transient dynamics that inflate or mask parameter effects.
    """
    grouped = _aggregate_by_group(Y_timeseries, Y_timeseries_meta, "generation")
    result: dict[int, SobolIndices] = {}
    for gen, Y_g in grouped.items():
        sobol, _ = run_phase1(param_space, X, Y_g, polynomial_order, regression=regression)
        result[gen] = sobol
    return result


def run_by_seed(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    Y_timeseries_meta: list[dict[str, np.ndarray]],
    polynomial_order: int,
    regression: str = "lsq",
) -> dict[int, SobolIndices]:
    """Strategy 3: PCE + Sobol per lineage seed.

    Aggregates each sample's timeseries by lineage seed, then runs
    Phase-1-style bulk GSA independently per seed.  This controls for
    exogenous stochastic variance (gene expression noise, stochastic
    partitioning at division, probabilistic initiation events).
    """
    grouped = _aggregate_by_group(Y_timeseries, Y_timeseries_meta, "lineage_seed")
    result: dict[int, SobolIndices] = {}
    for seed, Y_s in grouped.items():
        sobol, _ = run_phase1(param_space, X, Y_s, polynomial_order, regression=regression)
        result[seed] = sobol
    return result


# ── Phase 2: Growth-stratified sensitivity ──────────────────────────


def run_phase2(
    param_space: XSpace,
    X: np.ndarray,
    Y_timeseries: list[np.ndarray],
    n_bins: int,
    polynomial_order: int,
    mass_col_index: int = 0,
    regression: str = "lsq",
) -> tuple[list[SobolIndices], PCESurrogate]:
    """Strategy 4: per-stage PCE + Sobol using mass-based θ.

    For each cached timeseries:
      1. Compute θ = normalized log(dry_mass)
      2. Bin timesteps into *n_bins* growth stages
      3. Compute per-stage mean observables

    Then fit PCE to the stacked (n_samples, n_bins * n_obs) output
    and extract per-stage Sobol indices.
    """
    n_obs = Y_timeseries[0].shape[1]

    Y_stage_list: list[np.ndarray] = []
    for ts in Y_timeseries:
        theta = compute_growth_fraction(ts, mass_col_index=mass_col_index)
        bins = bin_by_growth_stage(theta, n_bins)

        stage_means = np.zeros(n_bins * n_obs)
        for s in range(n_bins):
            mask = bins == s
            if np.any(mask):
                stage_means[s * n_obs : (s + 1) * n_obs] = ts[mask].mean(axis=0)
        Y_stage_list.append(stage_means)

    Y_stage = np.vstack(Y_stage_list)
    bounds = np.array(param_space.parameter_bounds)

    sobol_multi, surrogate = _fit_pce_and_sobol(
        X,
        Y_stage,
        bounds,
        parameter_names=param_space.parameter_names,
        polynomial_order=polynomial_order,
        regression=regression,
        per_output=True,
    )

    # Split (n_bins*n_obs, n_params) → (n_bins, n_obs, n_params) → mean over obs
    fo = sobol_multi.first_order
    to = sobol_multi.total_order
    if fo.ndim > 1 and fo.shape[0] == n_bins * n_obs:
        fo_stages = fo.reshape(n_bins, n_obs, -1).mean(axis=1)
        to_stages = to.reshape(n_bins, n_obs, -1).mean(axis=1)
    else:
        fo_stages = fo.reshape(1, -1)
        to_stages = to.reshape(1, -1)

    per_stage = [
        SobolIndices(
            first_order=fo_stages[s],
            total_order=to_stages[s],
            parameter_names=param_space.parameter_names,
        )
        for s in range(fo_stages.shape[0])
    ]
    return per_stage, surrogate


# ── Full pipeline ───────────────────────────────────────────────────


def run_pipeline(
    cache: PrecomputedCache,
    param_space: XSpace,
    observable_names: list[str] | None = None,
    polynomial_order: int = 1,
    n_bins: int = 10,
    mass_col_index: int = 0,
    export_path: str | Path | None = None,
    regression: str = "lsq",
) -> SimplePipelineResult:
    """Run the full simplified UQ pipeline from a PrecomputedCache.

    Follows the UQPC workflow (PyTUQ, Sandia National Labs):
      1. Input setup — parameter bounds from ``param_space``
      2. Training data — (X, Y) from ``cache``
      3. Surrogate construction — PCE via ``pytuq.surrogates.pce.PCE``
      4. Sensitivity analysis — Sobol indices from PCE coefficients
      5. Post-processing — export to ``export_path``

    Args:
        cache: Cached (X, Y, Y_timeseries) from ``uq sample``.
        param_space: Parameter space (from ParameterDataset).
        observable_names: Observable column names.
        polynomial_order: PCE polynomial order.
        n_bins: Number of growth-progress bins.
        mass_col_index: Column index of dry mass in the timeseries
            arrays (default 0 = ``listeners__mass__dry_mass``).
        export_path: If given, write artifacts here.
        regression: PyTUQ regression method for PCE fitting.
            'lsq' (least squares, default), 'bcs' (Bayesian Compressed
            Sensing — sparse), or 'anl' (analytical).

    Returns:
        SimplePipelineResult with all 4 RFC006 strategy outputs.
    """
    X = cache.X
    Y = cache.Y
    obs_names = observable_names or cache.metadata.get(
        "observable_names",
        [f"obs_{i}" for i in range(Y.shape[1])],
    )

    # Phase 1 / Strategy 1: bulk Sobol (uniform aggregation)
    sobol_bulk, surrogate_bulk = run_phase1(
        param_space,
        X,
        Y,
        polynomial_order,
        regression=regression,
    )

    # Strategy 2: by-generation Sobol
    per_generation_sobol: dict[int, SobolIndices] | None = None
    if cache.Y_timeseries_meta is not None and cache.Y_timeseries is not None:
        per_generation_sobol = run_by_generation(
            param_space,
            X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order,
            regression=regression,
        )
        if not per_generation_sobol:
            logger.warning(
                "Strategy 2 (by generation): no generation metadata found. "
                "Skipping. Re-run sampling with generations >= 2 to enable."
            )
            per_generation_sobol = None
        elif len(per_generation_sobol) < 2:
            logger.warning(
                "Strategy 2 (by generation): only 1 generation found. "
                "Results are identical to Phase 1. Use generations >= 2 "
                "in sampling to get meaningful generation-stratified results."
            )
    else:
        logger.info(
            "Strategy 2 (by generation): skipped — no timeseries metadata "
            "in cache. Re-run sampling to generate metadata."
        )

    # Strategy 3: by-seed Sobol
    per_seed_sobol: dict[int, SobolIndices] | None = None
    if cache.Y_timeseries_meta is not None and cache.Y_timeseries is not None:
        per_seed_sobol = run_by_seed(
            param_space,
            X,
            cache.Y_timeseries,
            cache.Y_timeseries_meta,
            polynomial_order,
            regression=regression,
        )
        if not per_seed_sobol:
            logger.warning("Strategy 3 (by lineage seed): no seed metadata found. Skipping.")
            per_seed_sobol = None
    else:
        logger.info(
            "Strategy 3 (by lineage seed): skipped — no timeseries metadata "
            "in cache. Re-run sampling to generate metadata."
        )

    # Phase 2 / Strategy 4: growth-stratified Sobol
    per_stage_sobol, surrogate_cc = run_phase2(
        param_space,
        X,
        cache.Y_timeseries,  # type: ignore[arg-type]
        n_bins=n_bins,
        polynomial_order=polynomial_order,
        mass_col_index=mass_col_index,
        regression=regression,
    )

    result = SimplePipelineResult(
        parameter_names=param_space.parameter_names,
        population_sobol=sobol_bulk,
        population_surrogate=surrogate_bulk,
        per_stage_sobol=per_stage_sobol,
        growth_surrogate=surrogate_cc,
        n_bins=n_bins,
        observable_names=obs_names,
        per_generation_sobol=per_generation_sobol,
        per_seed_sobol=per_seed_sobol,
    )

    if export_path is not None:
        result.export(export_path)

    return result
