"""
workflow_spectral.py — Koopman-output PCE (``strategy_spectral``).

Additive, self-contained companion to :pymod:`uq.workflow`.  Implements the
lossless UQ ↔ DAW isomorphism described in the design discussion:

    Koopman diagonalizes the cell-level dynamics into complex eigenmodes,
    PCE parametrizes each Koopman feature as a function of the physical
    parameters, and Sobol indices on those PCE outputs become the
    modulation matrix of a synthesizer.

This file deliberately does **not** edit any existing module.  It imports
:func:`uq.workflow._setup_input_pc`, :func:`uq.workflow._fit_surrogate`,
:func:`uq.workflow._predict_and_variance`,
:func:`uq.workflow._compute_relative_errors`,
:func:`uq.workflow._compute_sobol`, and :func:`uq.workflow._physical_to_germ`
so the spectral strategy inherits every improvement to the main workflow
for free.

Pipeline
--------

.. code-block:: text

    Cached Y_timeseries (from `uq sample`)
             │
             ▼
    extract_spectral_features()           ← libuq.koopman.DynamicModeDecomposition
             │     (one DMD per sample, returns per-sample aligned Koopman features)
             ▼
    run_strategy_spectral()               ← uq.workflow._fit_surrogate / _compute_sobol
             │     (fits a single PCE where each output is a Koopman feature)
             ▼
    SpectralDAWResult
             │
             ├── synth.predict(x)                    — live PCE evaluation
             ├── synth.modulation_matrix             — Sobol S_Ti per feature
             ├── synth.inverse_design(target)        — cheap patch reverse-engineering
             ├── synth.resynthesize(x, t)            — rebuild a timeseries from PCE-ed modes
             └── synth.export(dir)                   — DAW-ready JSON + npy bundle

Everything downstream of ``_run_batch`` (i.e. all vEcoli work) is unchanged:
``quantify_spectral`` consumes the *same* ``PrecomputedCache`` that ``uq sample``
writes.  No new simulations required.

Mode alignment
--------------

Koopman modes do not come with a canonical ordering across samples, and real
signals produce complex-conjugate eigenvalue pairs.  The alignment rule used
here is the simplest one that is unit-stable and robust:

1. Drop the negative-frequency half of each conjugate pair (keep one of
   each pair; zero-frequency "DC" modes are kept separately).
2. Take the top-``n_modes`` remaining modes by amplitude magnitude
   ``|a_k|``.
3. Sort those by ascending frequency — this guarantees that feature index
   ``k`` refers to "the k-th loudest partial, re-sorted by pitch" for
   every sample, which is what the DAW lens expects.
4. Pad with zeros if a sample has fewer valid modes.

The resulting feature vector per sample is

    [ω_1, σ_1, |a_1|, ω_2, σ_2, |a_2|, …, ω_K, σ_K, |a_K|]

of length ``3 * n_modes``.  Mode shapes ``φ_k`` are optional (see
``include_mode_shapes``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# PyTUQ — re-used through uq.workflow helpers, not directly
from pytuq.rv.pcrv import PCRV  # type: ignore[import-untyped]

# libuq Koopman — read-only usage; we never mutate it
from libuq.koopman import DynamicModeDecomposition, KoopmanMode, KoopmanSpectrum
from libuq.sampling import PrecomputedCache

# Reuse workflow helpers without editing uq.workflow
from uq.workflow import (
    _compute_relative_errors,
    _compute_sobol,
    _fit_surrogate,
    _physical_to_germ,
    _predict_and_variance,
    _setup_input_pc,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Data containers
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SpectralFeatures:
    """Per-sample aligned Koopman features + metadata.

    Attributes:
        features: (n_samples, 3*n_modes) array of aligned Koopman features.
            Layout per row: [ω_1, σ_1, |a_1|, ω_2, σ_2, |a_2|, …, ω_K, σ_K, |a_K|].
        feature_names: Length 3*n_modes list of human labels, e.g.
            ``["omega_0", "sigma_0", "amp_0", "omega_1", …]``.
        n_modes: Number of retained modes (K).
        dt: Sample-to-sample timestep used for frequency conversion.
        observable_names: Names of the underlying observables (length
            equal to the number of channels per cell).
        mode_shapes: Optional (n_samples, n_modes, n_obs) array with
            real part of the Koopman mode vectors, if ``include_mode_shapes``
            was requested at extraction time.  None otherwise.
    """

    features: np.ndarray
    feature_names: list[str]
    n_modes: int
    dt: float
    observable_names: list[str]
    mode_shapes: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def split_blocks(self) -> dict[str, np.ndarray]:
        """Return {"omega": (N, K), "sigma": (N, K), "amp": (N, K)}."""
        K = self.n_modes
        block = self.features.reshape(self.n_samples, K, 3)
        return {
            "omega": block[:, :, 0],
            "sigma": block[:, :, 1],
            "amp": block[:, :, 2],
        }


# ═══════════════════════════════════════════════════════════════════
# 2. Koopman feature extraction (per cached sample)
# ═══════════════════════════════════════════════════════════════════


def _mode_is_dc(mode: KoopmanMode, tol: float = 1e-10) -> bool:
    """True if the mode is the real / DC (zero-frequency) mode."""
    return abs(np.imag(mode.eigenvalue)) < tol


def _dedupe_conjugate_pairs(modes: list[KoopmanMode]) -> list[KoopmanMode]:
    """Keep one mode from each complex-conjugate pair (the one with positive ω).

    Real signals produce eigenvalues in conjugate pairs
    ``λ`` and ``λ̄``; they carry the same physical information.  We keep the
    representative with positive imaginary part (positive ω).  DC modes are
    always kept.
    """
    kept: list[KoopmanMode] = []
    used: set[int] = set()
    for i, m in enumerate(modes):
        if i in used:
            continue
        if _mode_is_dc(m):
            kept.append(m)
            used.add(i)
            continue
        # Prefer the mode with positive imaginary part
        best = m if np.imag(m.eigenvalue) >= 0 else None
        for j in range(i + 1, len(modes)):
            if j in used:
                continue
            mj = modes[j]
            if _mode_is_dc(mj):
                continue
            if abs(m.eigenvalue - np.conj(mj.eigenvalue)) < 1e-9:
                used.add(j)
                if best is None or np.imag(mj.eigenvalue) > np.imag(best.eigenvalue):
                    best = mj
                break
        if best is not None:
            kept.append(best)
            used.add(i)
    return kept


def _align_modes(spectrum: KoopmanSpectrum, n_modes: int) -> list[KoopmanMode]:
    """Pick the top ``n_modes`` (by amplitude), then sort by ascending ω.

    This is the canonical per-sample ordering used across the entire file:
    feature index ``k`` refers to "the k-th loudest partial, re-sorted by
    pitch".
    """
    deduped = _dedupe_conjugate_pairs(list(spectrum.modes))
    by_amp = sorted(deduped, key=lambda m: float(np.abs(m.amplitude)), reverse=True)[:n_modes]
    return sorted(by_amp, key=lambda m: float(np.abs(np.imag(np.log(m.eigenvalue + 1e-12)) / (2 * np.pi))))


def _mode_to_triple(mode: KoopmanMode | None, dt: float) -> tuple[float, float, float]:
    """Convert one Koopman mode to a ``(ω, σ, |a|)`` triple in physical units.

    ``ω`` is in rad / (same time unit as ``dt``).  To get Hz, divide by
    ``2π``.  ``σ`` is in 1 / (same time unit as ``dt``).
    """
    if mode is None:
        return 0.0, 0.0, 0.0
    log_eig = np.log(mode.eigenvalue + 1e-12)
    omega = float(np.abs(np.imag(log_eig)) / dt)  # rad / time
    sigma = float(np.real(log_eig) / dt)  # 1 / time
    amp = float(np.abs(mode.amplitude))
    return omega, sigma, amp


def extract_spectral_features(
    Y_timeseries: list[np.ndarray],
    n_modes: int = 6,
    dt: float = 1.0,
    observable_names: list[str] | None = None,
    rank: int | None = None,
    include_mode_shapes: bool = False,
) -> SpectralFeatures:
    """Run DMD on every cached sample and stack the aligned features.

    Args:
        Y_timeseries: Per-sample raw timeseries, each ``(n_timesteps, n_obs)``.
        n_modes: Number of partials to retain per sample (K).
        dt: Sample-to-sample timestep (seconds) used by DMD for frequency
            conversion.  Divide ``ω`` by ``2π`` to get Hz.
        observable_names: Optional labels for the observable axis.
        rank: SVD truncation rank for DMD.  If ``None`` a sensible default
            ``2 * n_modes + 2`` is used (each oscillatory partial occupies a
            conjugate pair, so we need at least ``2 * n_modes`` latent
            dimensions; the extra +2 provides headroom for DC / decay modes).
        include_mode_shapes: If True, also return per-sample mode vectors
            (the "timbre" of each partial).  Stored in
            ``SpectralFeatures.mode_shapes``.

    Returns:
        :class:`SpectralFeatures` — one row per sample, ``3 * n_modes``
        columns per row (+ optional mode shapes).
    """
    if len(Y_timeseries) == 0:
        raise ValueError("extract_spectral_features: Y_timeseries is empty")

    n_obs = Y_timeseries[0].shape[1] if Y_timeseries[0].ndim == 2 else 1
    observable_names = observable_names or [f"obs_{i}" for i in range(n_obs)]

    # Default rank: 2*n_modes headroom for conjugate pairs + DC/decay slack.
    effective_rank = rank if rank is not None else max(2 * n_modes + 2, 4)

    N = len(Y_timeseries)
    features = np.zeros((N, 3 * n_modes), dtype=float)
    mode_shapes_arr: np.ndarray | None = (
        np.zeros((N, n_modes, n_obs), dtype=float) if include_mode_shapes else None
    )

    for i, ts in enumerate(Y_timeseries):
        ts = np.asarray(ts, dtype=float)
        if ts.ndim == 1:
            ts = ts.reshape(-1, 1)
        if ts.shape[0] < 4:
            logger.warning("Sample %d: %d timesteps too short for DMD, padding zeros", i, ts.shape[0])
            continue

        try:
            dmd = DynamicModeDecomposition(rank=effective_rank, dt=dt).fit(ts)
            spectrum = dmd.get_spectrum(observable_names)
        except Exception as e:
            logger.warning("Sample %d: DMD fit failed (%s), padding zeros", i, e)
            continue

        picked = _align_modes(spectrum, n_modes)
        for k in range(n_modes):
            m = picked[k] if k < len(picked) else None
            omega, sigma, amp = _mode_to_triple(m, dt)
            features[i, 3 * k] = omega
            features[i, 3 * k + 1] = sigma
            features[i, 3 * k + 2] = amp
            if include_mode_shapes and mode_shapes_arr is not None and m is not None:
                vec = np.real(m.mode)
                if vec.shape[0] == n_obs:
                    mode_shapes_arr[i, k] = vec

    feature_names = [
        label for k in range(n_modes) for label in (f"omega_{k}", f"sigma_{k}", f"amp_{k}")
    ]

    return SpectralFeatures(
        features=features,
        feature_names=feature_names,
        n_modes=n_modes,
        dt=dt,
        observable_names=observable_names,
        mode_shapes=mode_shapes_arr,
    )


# ═══════════════════════════════════════════════════════════════════
# 3. Strategy 5 — PCE over spectral features
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SpectralStrategyResult:
    """The UQPC output of the spectral strategy (mirrors ``UQPCResult``).

    Attributes:
        sobol: Variance-weighted Sobol indices across all spectral features.
        sobol_per_feature: Per-feature (first_order, total_order) pair —
            shape ``(n_features, n_params)``.  This is what becomes the
            modulation matrix of the DAW.
        pcrv: Fitted output PCRV (from uq.workflow._fit_surrogate).
        linregs: Per-output linear regression objects.
        germ_train: Germ-space samples used for fitting.
        X_train: Physical-space samples.
        Y_train: Spectral features at training samples.
        Y_train_pc: PCE predictions at training samples.
        relerr_train: Per-feature training relative errors.
        relerr_test: Per-feature test relative errors, if available.
        feature_names: Names of the spectral features.
    """

    sobol: Any
    sobol_per_feature_first: np.ndarray
    sobol_per_feature_total: np.ndarray
    pcrv: PCRV
    linregs: list[Any]
    germ_train: np.ndarray
    X_train: np.ndarray
    Y_train: np.ndarray
    Y_train_pc: np.ndarray
    relerr_train: np.ndarray
    relerr_test: np.ndarray | None
    feature_names: list[str]


def _per_output_sobol(
    pcrv: PCRV,
    parameter_names: list[str],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-output (first_order, total_order) Sobol matrices.

    Unlike :func:`uq.workflow._compute_sobol`, which variance-weights
    across outputs to return a single parameter ranking, the DAW modulation
    matrix needs the **full (n_features, n_params) tensor** — one row per
    Koopman feature, one column per parameter.
    """
    allsens_main = pcrv.computeSens()  # (n_out, n_params)
    allsens_total = pcrv.computeTotSens()  # (n_out, n_params)

    if allsens_main.ndim == 1:
        allsens_main = allsens_main.reshape(1, -1)
        allsens_total = allsens_total.reshape(1, -1)

    n_features = len(feature_names)
    n_params = len(parameter_names)
    if allsens_main.shape != (n_features, n_params):
        logger.warning(
            "PCRV sens shape %s != (n_features=%d, n_params=%d); truncating",
            allsens_main.shape,
            n_features,
            n_params,
        )
    return np.asarray(allsens_main), np.asarray(allsens_total)


def run_strategy_spectral(
    spectral: SpectralFeatures,
    parameter_names: list[str],
    germ_train: np.ndarray,
    X_train: np.ndarray,
    polynomial_order: int = 2,
    regression: Literal["lsq", "bcs", "anl"] = "lsq",
    tolerance: float = 1e-3,
    Y_test_features: np.ndarray | None = None,
    germ_test: np.ndarray | None = None,
) -> SpectralStrategyResult:
    """Fit a PCE where each output is a Koopman feature.

    This reuses the exact same :func:`uq.workflow._fit_surrogate` path that
    powers strategies 1-4, so everything (BCS, ANL, held-out validation,
    prediction variance) carries over without modification.
    """
    Y = spectral.features
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    pcrv, linregs = _fit_surrogate(
        germ_train,
        Y,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
    )

    n_features = Y.shape[1]
    Y_train_pc, _ = _predict_and_variance(pcrv, linregs, germ_train, n_features)
    relerr_train = _compute_relative_errors(Y, Y_train_pc)

    relerr_test: np.ndarray | None = None
    if Y_test_features is not None and germ_test is not None:
        Y_test_pc, _ = _predict_and_variance(pcrv, linregs, germ_test, n_features)
        relerr_test = _compute_relative_errors(Y_test_features, Y_test_pc)

    sobol_main, sobol_total = _per_output_sobol(pcrv, parameter_names, spectral.feature_names)
    # Variance-weighted aggregation (same convention as _compute_sobol)
    aggregate = _compute_sobol(pcrv, parameter_names, Y)

    return SpectralStrategyResult(
        sobol=aggregate,
        sobol_per_feature_first=sobol_main,
        sobol_per_feature_total=sobol_total,
        pcrv=pcrv,
        linregs=linregs,
        germ_train=germ_train,
        X_train=X_train,
        Y_train=Y,
        Y_train_pc=Y_train_pc,
        relerr_train=relerr_train,
        relerr_test=relerr_test,
        feature_names=spectral.feature_names,
    )


# ═══════════════════════════════════════════════════════════════════
# 4. SpectralDAWResult — the DAW-ready object
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SpectralDAWResult:
    """DAW-ready wrapper around :class:`SpectralStrategyResult`.

    Implements the isomorphism table from the design discussion:

    - :meth:`predict` — "play" the synth at an arbitrary knob setting.
    - :attr:`modulation_matrix` — which knob modulates which partial.
    - :meth:`inverse_design` — reverse-engineer a patch for a target spectrum.
    - :meth:`resynthesize` — reconstruct a timeseries from the PCE-predicted
      Koopman decomposition at a given parameter vector.
    - :meth:`export` — dump everything the GUI needs as JSON + .npy.
    """

    strategy: SpectralStrategyResult
    parameter_names: list[str]
    parameter_bounds: np.ndarray  # (n_params, 2)
    spectral: SpectralFeatures
    cache_dir: Path | None = None

    # ── core DAW accessors ────────────────────────────────────────

    @property
    def n_modes(self) -> int:
        return self.spectral.n_modes

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)

    @property
    def feature_names(self) -> list[str]:
        return list(self.strategy.feature_names)

    @property
    def modulation_matrix(self) -> np.ndarray:
        """Sobol total-order indices, shape ``(n_features, n_params)``.

        Row *f*, column *i* = "fraction of the variance of feature f
        explained by parameter i (including interactions)".  This is the
        synth's modulation routing — e.g. row 0 (``omega_0``) tells you
        which knobs move the pitch of partial 0.
        """
        return np.asarray(self.strategy.sobol_per_feature_total)

    @property
    def train_relerr_per_feature(self) -> np.ndarray:
        return np.asarray(self.strategy.relerr_train)

    @property
    def test_relerr_per_feature(self) -> np.ndarray | None:
        if self.strategy.relerr_test is None:
            return None
        return np.asarray(self.strategy.relerr_test)

    # ── forward evaluation (knob → spectrum) ──────────────────────

    def _x_to_germ(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(float)
        lb, ub = self.parameter_bounds[:, 0], self.parameter_bounds[:, 1]
        span = np.where(ub - lb > 0, ub - lb, 1.0)
        return 2.0 * (x - lb) / span - 1.0

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Predict the Koopman feature vector at a parameter setting.

        Args:
            x: Physical-space parameter vector, shape ``(n_params,)``.

        Returns:
            Dict with keys ``{"omega", "sigma", "amp", "features",
            "freq_hz"}`` — ``omega`` in rad/time, ``sigma`` in 1/time,
            ``amp`` in observable units, ``freq_hz`` = ``omega / (2π)``.
        """
        germ = self._x_to_germ(np.asarray(x))
        y = self.strategy.pcrv.function(germ).ravel()
        K = self.n_modes
        omega = y[0::3]
        sigma = y[1::3]
        amp = y[2::3]
        return {
            "features": y,
            "omega": omega[:K],
            "sigma": sigma[:K],
            "amp": amp[:K],
            "freq_hz": omega[:K] / (2 * np.pi),
        }

    # ── inverse design ────────────────────────────────────────────

    def inverse_design(
        self,
        target: np.ndarray | dict[str, np.ndarray],
        weights: np.ndarray | None = None,
        n_restarts: int = 8,
        seed: int = 0,
        bounds_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Reverse-engineer a parameter vector for a target spectrum.

        Solves ``min_x ||w ⊙ (Ŷ_features(x) − target)||²`` over the
        physical parameter box using L-BFGS-B with a random multi-start.

        Args:
            target: Target feature vector (length ``3*n_modes``) OR a dict
                with any subset of keys ``{"omega", "sigma", "amp",
                "freq_hz"}``.  Missing entries are ignored via zero
                weights.
            weights: Optional weighting per feature, same length as target
                vector.  Defaults to 1 for every specified target.
            n_restarts: Number of random L-BFGS-B restarts.
            seed: RNG seed for the restart points.
            bounds_scale: Shrink the physical box by this factor for the
                optimizer (``1.0`` uses the full box).

        Returns:
            Dict with ``{"x": best x, "loss": final loss, "prediction":
            prediction(best_x), "target": the resolved target vector,
            "mask": which indices were active}``.
        """
        from scipy.optimize import minimize  # lazy import

        K = self.n_modes
        target_vec = np.zeros(3 * K)
        mask = np.zeros(3 * K, dtype=bool)

        if isinstance(target, dict):
            if "freq_hz" in target:
                omega = np.asarray(target["freq_hz"], dtype=float) * (2 * np.pi)
                for k, v in enumerate(omega[:K]):
                    target_vec[3 * k] = v
                    mask[3 * k] = True
            if "omega" in target:
                for k, v in enumerate(np.asarray(target["omega"], dtype=float)[:K]):
                    target_vec[3 * k] = v
                    mask[3 * k] = True
            if "sigma" in target:
                for k, v in enumerate(np.asarray(target["sigma"], dtype=float)[:K]):
                    target_vec[3 * k + 1] = v
                    mask[3 * k + 1] = True
            if "amp" in target:
                for k, v in enumerate(np.asarray(target["amp"], dtype=float)[:K]):
                    target_vec[3 * k + 2] = v
                    mask[3 * k + 2] = True
        else:
            target_vec = np.asarray(target, dtype=float)
            mask = np.ones_like(target_vec, dtype=bool)

        w = np.ones_like(target_vec) if weights is None else np.asarray(weights, dtype=float)
        w = w * mask.astype(float)

        def loss(x_phys: np.ndarray) -> float:
            pred = self.predict(x_phys)["features"]
            diff = w * (pred - target_vec)
            return float(np.dot(diff, diff))

        lb = self.parameter_bounds[:, 0]
        ub = self.parameter_bounds[:, 1]
        span = ub - lb
        inner_lb = lb + 0.5 * (1 - bounds_scale) * span
        inner_ub = ub - 0.5 * (1 - bounds_scale) * span
        box = list(zip(inner_lb.tolist(), inner_ub.tolist()))

        rng = np.random.default_rng(seed)
        best_x: np.ndarray | None = None
        best_loss = np.inf
        for _ in range(max(1, n_restarts)):
            x0 = inner_lb + rng.random(len(lb)) * (inner_ub - inner_lb)
            try:
                res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=box)
            except Exception as e:
                logger.warning("inverse_design restart failed: %s", e)
                continue
            if res.fun < best_loss:
                best_loss = float(res.fun)
                best_x = np.asarray(res.x)

        if best_x is None:
            raise RuntimeError("inverse_design: all optimizer restarts failed")

        return {
            "x": best_x,
            "loss": best_loss,
            "prediction": self.predict(best_x),
            "target": target_vec,
            "mask": mask,
        }

    # ── resynthesis (knob → timeseries) ──────────────────────────

    def resynthesize(
        self,
        x: np.ndarray,
        t: np.ndarray,
        dc: float = 0.0,
    ) -> np.ndarray:
        """Reconstruct a 1-D observable from PCE-predicted Koopman modes.

        .. math::

            \\hat Y(t) \\approx \\mathrm{dc} + \\sum_{k=1}^{K}
                \\lvert a_k(x)\\rvert \\,
                \\exp\\bigl(\\sigma_k(x)\\,t\\bigr)\\,
                \\cos\\bigl(\\omega_k(x)\\,t\\bigr)

        This is an approximate reconstruction — it assumes real-valued
        observables and drops mode phases (the PCE outputs only carry
        ``|a_k|``, not the complex amplitude).  It is still faithful for
        the DAW use case: turning a knob audibly bends pitches and
        reshapes the envelope in real time.

        Args:
            x: Parameter vector.
            t: Time stamps.
            dc: Optional constant offset.

        Returns:
            1-D reconstruction, shape ``(len(t),)``.
        """
        pred = self.predict(x)
        omega, sigma, amp = pred["omega"], pred["sigma"], pred["amp"]
        t = np.asarray(t, dtype=float)
        y = np.full_like(t, fill_value=float(dc))
        for k in range(len(omega)):
            y += amp[k] * np.exp(sigma[k] * t) * np.cos(omega[k] * t)
        return y

    # ── export ─────────────────────────────────────────────────────

    def export(self, export_dir: str | Path) -> Path:
        """Write a DAW-ready artifact directory.

        Layout:

        .. code-block:: text

            export_dir/
            ├── spectral_features.npy          # (N, 3K) training features
            ├── feature_names.json
            ├── modulation_matrix.npy          # (3K, n_params) Sobol S_Ti
            ├── modulation_matrix_first.npy    # (3K, n_params) Sobol S_i
            ├── pce_coefficients.npy           # (3K, n_basis) — one row per feature
            ├── multi_indices.npy              # (n_basis, n_params) — shared basis
            ├── input_bounds.npy               # (n_params, 2) physical box
            ├── parameter_names.json
            ├── relerr_train.npy
            ├── relerr_test.npy                # optional
            └── spectral_summary.json          # DAW manifest

        .. note::

            The ``(pce_coefficients, multi_indices)`` pair assumes the PCE
            outputs share a single basis (the ``lsq`` and ``anl`` regression
            backends in ``uq.workflow._fit_surrogate`` satisfy this).  Sparse
            backends like ``bcs`` may use per-output sub-bases; in that case
            we still export the *union* multi-index and pad each row of
            ``pce_coefficients`` with zeros for the inactive terms.
        """
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "spectral_features.npy", self.strategy.Y_train)
        (out / "feature_names.json").write_text(json.dumps(self.feature_names, indent=2))
        (out / "parameter_names.json").write_text(json.dumps(self.parameter_names, indent=2))

        np.save(out / "modulation_matrix.npy", self.modulation_matrix)
        np.save(out / "modulation_matrix_first.npy", self.strategy.sobol_per_feature_first)
        np.save(out / "input_bounds.npy", self.parameter_bounds)
        np.save(out / "relerr_train.npy", self.train_relerr_per_feature)
        if self.test_relerr_per_feature is not None:
            np.save(out / "relerr_test.npy", self.test_relerr_per_feature)

        # PCE coefficients — (n_features, n_basis) against a shared multi-index.
        try:
            coefs = self.strategy.pcrv.coefs
            mindices = self.strategy.pcrv.mindices
            if coefs and mindices:
                # Build the union multi-index across all outputs (robust to sparse
                # bases).  Each output's coefficients are re-expanded onto the
                # union basis by matching multi-index rows.
                all_mi = [np.asarray(mi) for mi in mindices]
                union_rows: list[tuple[int, ...]] = []
                for mi in all_mi:
                    for row in mi:
                        union_rows.append(tuple(int(v) for v in row))
                # Preserve first-seen order for deterministic output
                seen: dict[tuple[int, ...], int] = {}
                for row in union_rows:
                    if row not in seen:
                        seen[row] = len(seen)
                n_basis = len(seen)
                n_dim = all_mi[0].shape[1] if all_mi else 0
                union_mi = np.zeros((n_basis, n_dim), dtype=int)
                for row, idx in seen.items():
                    union_mi[idx] = row

                n_features = len(coefs)
                stacked = np.zeros((n_features, n_basis), dtype=float)
                for j, (c, mi) in enumerate(zip(coefs, all_mi)):
                    c_arr = np.asarray(c, dtype=float)
                    for local_i, row in enumerate(mi):
                        key = tuple(int(v) for v in row)
                        stacked[j, seen[key]] = c_arr[local_i]

                np.save(out / "pce_coefficients.npy", stacked)
                np.save(out / "multi_indices.npy", union_mi)
        except Exception as e:
            logger.warning("Failed to export PCE coefficients: %s", e)

        summary = {
            "framework": "uq.workflow_spectral (Koopman-output PCE / strategy_spectral)",
            "parameter_names": self.parameter_names,
            "n_parameters": self.n_parameters,
            "n_modes": self.n_modes,
            "feature_names": self.feature_names,
            "observable_names": self.spectral.observable_names,
            "dt": self.spectral.dt,
            "units": {
                "omega": "rad / time",
                "freq_hz": "omega / (2*pi)",
                "sigma": "1 / time",
                "amp": "observable units",
            },
            "sobol_total_by_feature": {
                f: {p: round(float(self.modulation_matrix[i, j]), 6) for j, p in enumerate(self.parameter_names)}
                for i, f in enumerate(self.feature_names)
            },
            "relerr_train": [round(float(v), 6) for v in self.train_relerr_per_feature],
            "relerr_test": (
                None
                if self.test_relerr_per_feature is None
                else [round(float(v), 6) for v in self.test_relerr_per_feature]
            ),
        }
        (out / "spectral_summary.json").write_text(json.dumps(summary, indent=2))
        return out


# ═══════════════════════════════════════════════════════════════════
# 5. Top-level entry point
# ═══════════════════════════════════════════════════════════════════


def quantify_spectral(
    cache_dir: str | Path,
    parameter_bounds: np.ndarray | None = None,
    parameter_names: list[str] | None = None,
    n_modes: int = 6,
    dt: float = 1.0,
    polynomial_order: int = 2,
    regression: Literal["lsq", "bcs", "anl"] = "lsq",
    tolerance: float = 1e-3,
    include_mode_shapes: bool = False,
    export_path: str | Path | None = None,
) -> SpectralDAWResult:
    """Koopman-output PCE over an existing ``PrecomputedCache``.

    This runs *after* ``uq sample`` — no new vEcoli calls.

    Args:
        cache_dir: Cache produced by :func:`uq.workflow.sample`.
        parameter_bounds: ``(n_params, 2)`` bounds.  If ``None``, pulled
            from ``cache.metadata['bounds']``.
        parameter_names: If ``None``, pulled from
            ``cache.parameter_names``.
        n_modes: Number of Koopman partials to retain per sample.
        dt: Sample-to-sample timestep (used by DMD for frequency units).
        polynomial_order: Output PCE order (UQPC ``--outord``).
        regression: PyTUQ regression backend.
        tolerance: BCS sparsity tolerance.
        include_mode_shapes: If True, also fit PCE over mode vectors
            (stored in ``SpectralFeatures.mode_shapes``, not yet part of
            the PCE output).
        export_path: If given, call ``result.export(export_path)``.

    Returns:
        :class:`SpectralDAWResult` — the DAW synth handle.
    """
    cache_dir = Path(cache_dir)
    cache = PrecomputedCache.load(cache_dir)
    if cache.Y_timeseries is None:
        raise RuntimeError(
            f"Cache at {cache_dir} has no Y_timeseries; "
            "re-run `uq sample` so the raw cell-level timeseries are cached."
        )

    names = parameter_names or list(cache.parameter_names)
    bounds = parameter_bounds
    if bounds is None:
        raw_bounds = cache.metadata.get("bounds")
        if raw_bounds is None:
            raise RuntimeError(
                f"Cache at {cache_dir} has no parameter bounds; "
                "pass `parameter_bounds` explicitly."
            )
        bounds = np.asarray(raw_bounds, dtype=float)
    else:
        bounds = np.asarray(bounds, dtype=float)

    # Step 1: DMD per sample → aligned spectral features
    spectral = extract_spectral_features(
        Y_timeseries=cache.Y_timeseries,
        n_modes=n_modes,
        dt=dt,
        observable_names=cache.metadata.get("observable_columns"),
        include_mode_shapes=include_mode_shapes,
    )

    # Step 2: build input PC and load germ samples (reusing uq.workflow helpers)
    pc, _, _ = _setup_input_pc(bounds)
    germ_path = cache_dir / "germ_train.npy"
    if germ_path.exists():
        germ_train = np.load(germ_path)
    else:
        germ_train = _physical_to_germ(cache.X, bounds)
    del pc  # only needed for consistency checks; PCRV reinitialized in _fit_surrogate

    # Optional held-out test spectral features (UQPC --ntst)
    Y_test_features: np.ndarray | None = None
    germ_test: np.ndarray | None = None
    if cache.X_test is not None and cache.Y_test_timeseries is not None:
        test_spectral = extract_spectral_features(
            Y_timeseries=cache.Y_test_timeseries,
            n_modes=n_modes,
            dt=dt,
            observable_names=cache.metadata.get("observable_columns"),
            include_mode_shapes=False,
        )
        Y_test_features = test_spectral.features
        germ_test = _physical_to_germ(cache.X_test, bounds)

    # Step 3: fit PCE over spectral features
    strategy = run_strategy_spectral(
        spectral=spectral,
        parameter_names=names,
        germ_train=germ_train,
        X_train=cache.X,
        polynomial_order=polynomial_order,
        regression=regression,
        tolerance=tolerance,
        Y_test_features=Y_test_features,
        germ_test=germ_test,
    )

    result = SpectralDAWResult(
        strategy=strategy,
        parameter_names=names,
        parameter_bounds=bounds,
        spectral=spectral,
        cache_dir=cache_dir,
    )

    if export_path is not None:
        result.export(export_path)
        logger.info("Spectral DAW artifacts exported to %s", export_path)

    return result
