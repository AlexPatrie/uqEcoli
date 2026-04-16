"""
UQ Simple DAW — Tkinter dashboard for ``uq.workflow`` + ``uq.workflow_spectral``.

Provides DAW-style interactive visualization of:

  1. Uniform (bulk)          — PCE response curves with draggable markers
  2. By generation            — per-generation Sobol bar comparison
  3. By lineage seed          — per-seed Sobol bar comparison
  4. Growth-stratified        — sensitivity spectrogram + per-stage prediction
  5. **Koopman synth**       — pitches/dampings/amplitudes of Koopman partials,
     live-modulated by the PCE knobs, plus a Sobol modulation-matrix heatmap,
     a resynthesized-waveform oscilloscope, and an inverse-design button.

The spectral section is the genuine audio-DAW lens (Koopman diagonalizes the
cell-level dynamics, PCE parametrizes the diagonalization, Sobol becomes the
modulation matrix — every symbol has matching physical units on both sides).
It is enabled whenever the loaded results directory contains a ``spectral/``
sub-directory produced by :meth:`uq.workflow_spectral.SpectralDAWResult.export`.
If the sub-directory is missing the DAW falls back to strategies 1-4 only.

Launch:
    uv run python app/uq_daw_simple.py [path/to/uq_results.json]
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import numpy as np

# -- Colors (DAW dark theme) --------------------------------------------------

C = {
    "bg": "#0d0d0d",
    "panel": "#1a1a2e",
    "panel_light": "#252545",
    "border": "#2a2a4a",
    "text": "#e0e0e0",
    "text_dim": "#888888",
    "accent1": "#00f0ff",
    "accent2": "#ff3366",
    "accent3": "#33ff99",
    "accent4": "#ffaa00",
    "accent5": "#aa66ff",
    "accent_blue": "#2E86C1",
    "accent_gold": "#D4AC0D",
    "grid": "#222244",
}

DEFAULT_PARAM_COLORS = [C["accent1"], C["accent4"], C["accent2"], C["accent3"], C["accent5"]]


# -- Observable labels + units (biologist-facing) ------------------------------
#
# Maps Parquet column tails to (short label, unit string, python format spec).
# Falls back to the raw column name if not found.

OBSERVABLE_UNITS: dict[str, tuple[str, str, str]] = {
    "dry_mass": ("Dry mass", "fg", ".1f"),
    "cell_mass": ("Cell mass", "fg", ".1f"),
    "volume": ("Volume", "\u00b5m\u00b3", ".3f"),
    "growth": ("Mass growth", "fg/s", ".4f"),
    "instantaneous_growth_rate": ("Growth rate", "s\u207b\u00b9", ".2e"),
    "growth_rate_per_hour": ("Growth rate", "hr\u207b\u00b9", ".3f"),
    "doubling_time_hours": ("Doubling time", "hr", ".2f"),
    "dna_fraction_g_per_g_dw": ("DNA fraction", "g/g DW", ".4f"),
    "rna_fraction_g_per_g_dw": ("RNA fraction", "g/g DW", ".4f"),
    "dry_mass_fraction": ("Dry mass frac", "", ".3f"),
    "cell_volume_um3": ("Volume", "\u00b5m\u00b3", ".3f"),
}


def _obs_label(name: str) -> tuple[str, str, str]:
    """Return (short_label, unit, fmt) for an observable name."""
    tail = name.split("__")[-1] if "__" in name else name
    if tail in OBSERVABLE_UNITS:
        return OBSERVABLE_UNITS[tail]
    return (tail, "", ".4f")


# -- Legendre PCE evaluation --------------------------------------------------


def legendre_eval(x_norm, coeffs, multi_indices):
    """Evaluate PCE: sum c_alpha * prod P_{alpha_i}(x_i)."""
    max_ord = int(multi_indices.max())
    n_p = multi_indices.shape[1]
    P = np.zeros((max_ord + 1, n_p))
    P[0, :] = 1.0
    if max_ord >= 1:
        P[1, :] = x_norm
    for n in range(2, max_ord + 1):
        P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
    result = 0.0
    for t in range(len(coeffs)):
        term = coeffs[t]
        for p in range(n_p):
            term *= P[multi_indices[t, p], p]
        result += term
    return result


def normalize_to_germ(x, bounds):
    """Scale physical parameters to [-1, 1]."""
    return 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0


# -- Spectral (Koopman-PCE) data layer ----------------------------------------
#
# Everything below is UI-free: it can be imported, loaded, and exercised
# without instantiating a single Tk widget.  The canvases further down wrap
# these helpers but do not duplicate any math.


@dataclass
class SpectralBundle:
    """Loaded contents of a ``SpectralDAWResult.export(...)`` directory.

    This is the DAW-side mirror of :class:`uq.workflow_spectral.SpectralDAWResult`:
    it holds just enough to evaluate the PCE live and render the panels.
    """

    export_dir: Path
    parameter_names: list[str]
    feature_names: list[str]
    input_bounds: np.ndarray  # (n_params, 2)
    pce_coefficients: np.ndarray  # (n_features, n_basis)
    multi_indices: np.ndarray  # (n_basis, n_params)
    modulation_matrix: np.ndarray  # (n_features, n_params) Sobol S_Ti
    modulation_matrix_first: np.ndarray | None = None
    relerr_train: np.ndarray | None = None
    relerr_test: np.ndarray | None = None
    training_features: np.ndarray | None = None  # (N_samples, n_features)
    summary: dict[str, Any] = field(default_factory=dict)

    # --- derived shape helpers ---

    @property
    def n_parameters(self) -> int:
        return int(self.input_bounds.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.pce_coefficients.shape[0])

    @property
    def n_modes(self) -> int:
        # Features are laid out as (ω_k, σ_k, |a_k|) triples
        return self.n_features // 3

    @property
    def dt(self) -> float:
        try:
            return float(self.summary.get("dt", 1.0))
        except Exception:
            return 1.0


def load_spectral_bundle(export_dir: str | Path) -> SpectralBundle | None:
    """Load a spectral artifact directory.

    Returns ``None`` if any required file is missing — the DAW will then
    silently fall back to strategies 1-4 only.

    Args:
        export_dir: Path to the directory written by
            :meth:`uq.workflow_spectral.SpectralDAWResult.export`.
    """
    export_dir = Path(export_dir)
    required = [
        "pce_coefficients.npy",
        "multi_indices.npy",
        "input_bounds.npy",
        "modulation_matrix.npy",
        "feature_names.json",
        "parameter_names.json",
    ]
    if not all((export_dir / f).exists() for f in required):
        return None

    feature_names = json.loads((export_dir / "feature_names.json").read_text())
    parameter_names = json.loads((export_dir / "parameter_names.json").read_text())

    bundle = SpectralBundle(
        export_dir=export_dir,
        parameter_names=list(parameter_names),
        feature_names=list(feature_names),
        input_bounds=np.load(export_dir / "input_bounds.npy"),
        pce_coefficients=np.load(export_dir / "pce_coefficients.npy"),
        multi_indices=np.load(export_dir / "multi_indices.npy"),
        modulation_matrix=np.load(export_dir / "modulation_matrix.npy"),
    )

    opt = {
        "modulation_matrix_first": "modulation_matrix_first.npy",
        "relerr_train": "relerr_train.npy",
        "relerr_test": "relerr_test.npy",
        "training_features": "spectral_features.npy",
    }
    for attr, fname in opt.items():
        p = export_dir / fname
        if p.exists():
            setattr(bundle, attr, np.load(p))

    summary_path = export_dir / "spectral_summary.json"
    if summary_path.exists():
        bundle.summary = json.loads(summary_path.read_text())

    return bundle


def evaluate_multi_output_pce(
    x_norm: np.ndarray,
    coeffs_matrix: np.ndarray,
    multi_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate a multi-output Legendre PCE at a single germ-space point.

    Args:
        x_norm: Germ-space parameter vector, shape ``(n_params,)``.
        coeffs_matrix: ``(n_features, n_basis)`` coefficient matrix (shared basis).
        multi_indices: ``(n_basis, n_params)`` multi-index table.

    Returns:
        Length-``n_features`` prediction vector (one value per PCE output).
    """
    max_ord = int(multi_indices.max()) if multi_indices.size else 0
    n_params = multi_indices.shape[1]

    # Precompute Legendre polynomials at x_norm up to max_ord for each dim
    P = np.zeros((max_ord + 1, n_params))
    P[0, :] = 1.0
    if max_ord >= 1:
        P[1, :] = x_norm
    for n in range(2, max_ord + 1):
        P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n

    # term_vals[t] = product over params of P[alpha_t, p]
    n_basis = multi_indices.shape[0]
    term_vals = np.ones(n_basis)
    for t in range(n_basis):
        for p in range(n_params):
            term_vals[t] *= P[multi_indices[t, p], p]

    # One dot product per output feature — much faster than a Python inner loop
    return coeffs_matrix @ term_vals


def predict_spectral_features(
    bundle: SpectralBundle,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the Koopman-output PCE at a physical parameter setting.

    Returns the same dict shape as
    :meth:`uq.workflow_spectral.SpectralDAWResult.predict`:
    ``{"omega", "sigma", "amp", "freq_hz", "features"}``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != bundle.n_parameters:
        raise ValueError(
            f"predict_spectral_features: x has {x.shape[0]} params, "
            f"bundle has {bundle.n_parameters}"
        )
    x_norm = normalize_to_germ(x, bundle.input_bounds)
    y = evaluate_multi_output_pce(x_norm, bundle.pce_coefficients, bundle.multi_indices)
    K = bundle.n_modes
    omega = y[0::3][:K]
    sigma = y[1::3][:K]
    amp = y[2::3][:K]
    return {
        "features": y,
        "omega": omega,
        "sigma": sigma,
        "amp": amp,
        "freq_hz": omega / (2 * np.pi),
    }


def resynthesize_waveform(
    bundle: SpectralBundle,
    x: np.ndarray,
    t: np.ndarray,
    dc: float = 0.0,
) -> np.ndarray:
    """Reconstruct a 1-D observable from PCE-predicted Koopman modes.

    Mirrors :meth:`uq.workflow_spectral.SpectralDAWResult.resynthesize` but
    operates on a saved :class:`SpectralBundle` so the DAW does not require
    a live Python object.
    """
    pred = predict_spectral_features(bundle, x)
    omega, sigma, amp = pred["omega"], pred["sigma"], pred["amp"]
    t = np.asarray(t, dtype=float)
    y = np.full_like(t, fill_value=float(dc))
    for k in range(len(omega)):
        y += amp[k] * np.exp(sigma[k] * t) * np.cos(omega[k] * t)
    return y


def inverse_design_bundle(
    bundle: SpectralBundle,
    target: dict[str, np.ndarray],
    n_restarts: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """L-BFGS-B multi-start inverse design on top of a :class:`SpectralBundle`.

    The DAW uses this to implement the "inverse design" toolbar button.  It
    is identical in spirit to
    :meth:`uq.workflow_spectral.SpectralDAWResult.inverse_design` but takes
    a bundle instead of the live result — so it works from a saved export.
    """
    from scipy.optimize import minimize

    K = bundle.n_modes
    target_vec = np.zeros(3 * K)
    mask = np.zeros(3 * K, dtype=bool)

    if "freq_hz" in target:
        for k, v in enumerate(np.asarray(target["freq_hz"], dtype=float)[:K]):
            target_vec[3 * k] = float(v) * (2 * np.pi)
            mask[3 * k] = True
    if "omega" in target:
        for k, v in enumerate(np.asarray(target["omega"], dtype=float)[:K]):
            target_vec[3 * k] = float(v)
            mask[3 * k] = True
    if "sigma" in target:
        for k, v in enumerate(np.asarray(target["sigma"], dtype=float)[:K]):
            target_vec[3 * k + 1] = float(v)
            mask[3 * k + 1] = True
    if "amp" in target:
        for k, v in enumerate(np.asarray(target["amp"], dtype=float)[:K]):
            target_vec[3 * k + 2] = float(v)
            mask[3 * k + 2] = True

    w = mask.astype(float)

    def loss(x_phys: np.ndarray) -> float:
        pred = predict_spectral_features(bundle, x_phys)["features"]
        diff = w * (pred - target_vec)
        return float(np.dot(diff, diff))

    lb = bundle.input_bounds[:, 0]
    ub = bundle.input_bounds[:, 1]
    box = list(zip(lb.tolist(), ub.tolist()))

    rng = np.random.default_rng(seed)
    best_x: np.ndarray | None = None
    best_loss = np.inf
    for _ in range(max(1, n_restarts)):
        x0 = lb + rng.random(len(lb)) * (ub - lb)
        try:
            res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=box)
        except Exception:
            continue
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_x = np.asarray(res.x)

    if best_x is None:
        raise RuntimeError("inverse_design_bundle: all optimizer restarts failed")

    return {
        "x": best_x,
        "loss": best_loss,
        "prediction": predict_spectral_features(bundle, best_x),
        "target": target_vec,
        "mask": mask,
    }


# -- Canvas widgets -----------------------------------------------------------


class ResponseCurveCanvas(tk.Canvas):
    """DAW-style response curve display with draggable parameter markers."""

    PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 30, 40
    MARKER_HIT_RADIUS = 14

    def __init__(self, parent, param_colors=None, on_marker_drag=None, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, cursor="crosshair", **kwargs)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_hover)

        self._on_marker_drag = on_marker_drag
        self._param_colors = param_colors or {}
        self._curves = {}
        self._markers = {}
        self._marker_px = {}
        self._selected = None
        self._y_min = 0
        self._y_max = 1
        self._dragging = None
        self._hovering = None
        self._baseline: float | None = None
        self._obs_label = ""
        self._obs_unit = ""

    def _on_resize(self, event):
        self.redraw()

    def set_curves(self, curves, markers, selected, y_range, baseline=None, obs_label="", obs_unit=""):
        self._curves = curves
        self._markers = markers
        self._selected = selected
        self._y_min, self._y_max = y_range
        self._baseline = baseline
        self._obs_label = obs_label
        self._obs_unit = obs_unit
        self.redraw()

    def _plot_geometry(self):
        w = self.winfo_width()
        h = self.winfo_height()
        pl, pr, pt, pb = self.PAD_L, self.PAD_R, self.PAD_T, self.PAD_B
        pw = w - pl - pr
        ph = h - pt - pb
        ys = self._y_max - self._y_min
        if ys == 0:
            ys = 1
        return pl, pr, pt, pb, pw, ph, ys

    def _to_px(self, xn, yv):
        pl, _, pt, _, pw, ph, ys = self._plot_geometry()
        px = pl + int(xn * pw)
        py = pt + int((1 - (yv - self._y_min) / ys) * ph)
        return px, py

    def _from_px(self, px):
        pl, _, _, _, pw, _, _ = self._plot_geometry()
        if pw <= 0:
            return 0.0
        return max(0.0, min(1.0, (px - pl) / pw))

    def _find_nearest_marker(self, mx, my):
        best = None
        best_dist = self.MARKER_HIT_RADIUS + 1
        for pname, (px, py) in self._marker_px.items():
            dist = ((mx - px) ** 2 + (my - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = pname
        return best

    def _on_press(self, event):
        hit = self._find_nearest_marker(event.x, event.y)
        if hit:
            self._dragging = hit
            self.config(cursor="hand2")

    def _on_drag(self, event):
        if self._dragging is None:
            return
        x_norm = self._from_px(event.x)
        if self._on_marker_drag:
            self._on_marker_drag(self._dragging, x_norm)

    def _on_release(self, event):
        if self._dragging is not None:
            self._dragging = None
            self.config(cursor="crosshair")

    def _on_hover(self, event):
        hit = self._find_nearest_marker(event.x, event.y)
        if hit != self._hovering:
            self._hovering = hit
            self.config(cursor="hand2" if hit else "crosshair")

    def _color_for(self, pname, idx):
        return self._param_colors.get(pname, DEFAULT_PARAM_COLORS[idx % len(DEFAULT_PARAM_COLORS)])

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pl, pr, pt, pb, plot_w, plot_h, y_span = self._plot_geometry()

        # Grid
        for i in range(5):
            y = pt + int(plot_h * i / 4)
            self.create_line(pl, y, w - pr, y, fill=C["grid"], width=1)
        for i in range(11):
            x = pl + int(plot_w * i / 10)
            self.create_line(x, pt, x, pt + plot_h, fill=C["grid"], width=1)

        self.create_text(w // 2, h - 8, text="Normalized parameter [0=min, 1=max]", fill=C["text_dim"], font=("Menlo", 9))
        _y_label = self._obs_label or "\u0176"
        _y_unit = f" ({self._obs_unit})" if self._obs_unit else ""
        self.create_text(12, h // 2, text=f"{_y_label}{_y_unit}", fill=C["text_dim"], font=("Menlo", 9), angle=90)

        # Y-axis ticks
        for i in range(5):
            frac = i / 4
            y = pt + int(plot_h * (1 - frac))
            val = self._y_min + y_span * frac
            self.create_text(pl - 5, y, text=f"{val:.2f}", fill=C["text_dim"], font=("Menlo", 8), anchor="e")

        # Baseline reference line (wild-type / midpoint)
        if self._baseline is not None and self._y_min <= self._baseline <= self._y_max:
            _, _bl_py = self._to_px(0, self._baseline)
            self.create_line(pl, _bl_py, w - pr, _bl_py, fill=C["accent2"], width=1, dash=(6, 4))
            self.create_text(
                w - pr + 2, _bl_py, text=f"baseline {self._baseline:.2f}",
                fill=C["accent2"], font=("Menlo", 7), anchor="w",
            )

        # Draw curves
        param_list = list(self._curves.keys())
        for pi, pname in enumerate(param_list):
            xn, yv = self._curves[pname]
            is_sel = pname == self._selected
            is_drag = pname == self._dragging
            color = self._color_for(pname, pi)
            width = 3 if (is_sel or is_drag) else 1

            points = []
            for i in range(len(xn)):
                px, py = self._to_px(xn[i], yv[i])
                points.extend([px, py])

            if len(points) >= 4:
                stipple = "" if (is_sel or is_drag) else "gray50"
                self.create_line(points, fill=color, width=width, smooth=True, stipple=stipple)

        # Draw markers
        self._marker_px.clear()
        for pi, pname in enumerate(param_list):
            if pname not in self._markers:
                continue
            mx, my = self._markers[pname]
            is_sel = pname == self._selected
            is_drag = pname == self._dragging
            is_hover = pname == self._hovering
            color = self._color_for(pname, pi)
            px, py = self._to_px(mx, my)
            self._marker_px[pname] = (px, py)

            size = 10 if is_drag else (8 if (is_sel or is_hover) else 5)
            outline_color = "#ffffff" if (is_sel or is_drag or is_hover) else ""
            outline_width = 2 if (is_sel or is_drag) else (1 if is_hover else 0)
            self.create_oval(px - size, py - size, px + size, py + size, fill=color, outline=outline_color, width=outline_width)

            if is_sel or is_drag:
                self.create_text(px, py - size - 8, text=f"\u0176={my:.3f}", fill=color, font=("Menlo", 9, "bold"))

        drag_hint = "  [drag dots]" if self._markers else ""
        _title_obs = f" // {self._obs_label}" if self._obs_label else ""
        self.create_text(w // 2, 14, text=f"PCE RESPONSE CURVES{_title_obs}{drag_hint}", fill=C["text"], font=("Menlo", 11, "bold"))


class HeatmapCanvas(tk.Canvas):
    """Sensitivity spectrogram: params x growth stages."""

    def __init__(self, parent, param_colors=None, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._param_colors = param_colors or {}
        self._stage_predictions = None
        self._param_positions: dict[str, float] = {}

    def set_data(self, stages, params, selected, stage_predictions=None, param_positions=None):
        self._data = (stages, params, selected)
        self._stage_predictions = stage_predictions
        self._param_positions = param_positions or {}
        self.redraw()

    def _val_to_color(self, v):
        v = max(0, min(v / 0.7, 1.0))
        if v < 0.33:
            t = v / 0.33
            r, g, b = int(13 + t * 29), int(13 + t * 167), int(222)
        elif v < 0.66:
            t = (v - 0.33) / 0.33
            r, g, b = int(42 + t * 213), int(180 + t * -10), int(222 - t * 222)
        else:
            t = (v - 0.66) / 0.34
            r, g, b = 255, int(170 - t * 119), int(t * 102)
        return f"#{r:02x}{g:02x}{b:02x}"

    def redraw(self):
        self.delete("all")
        if self._data is None:
            return
        stages, params, selected = self._data
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pad_l, pad_r, pad_t, pad_b = 90, 45, 30, 20
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        n_stages = len(stages)
        n_params = len(params)
        if n_stages == 0 or n_params == 0:
            return
        cell_w = plot_w / n_stages
        cell_h = plot_h / n_params

        for pi, pname in enumerate(params):
            short = pname.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass")
            is_sel = pname == selected
            color = self._param_colors.get(pname, C["text_dim"]) if is_sel else C["text_dim"]
            self.create_text(pad_l - 5, pad_t + int(cell_h * (pi + 0.5)), text=short, fill=color, font=("Menlo", 9, "bold" if is_sel else ""), anchor="e")

            for si, stage in enumerate(stages):
                v = stage["sobol_total_order"].get(pname, 0)
                x0 = pad_l + int(si * cell_w)
                y0 = pad_t + int(pi * cell_h)
                x1 = x0 + int(cell_w)
                y1 = y0 + int(cell_h)
                self.create_rectangle(x0, y0, x1, y1, fill=self._val_to_color(v), outline=C["border"])

        heatmap_bottom = pad_t + plot_h
        for si in range(n_stages):
            x = pad_l + int(cell_w * (si + 0.5))
            self.create_text(x, heatmap_bottom + 10, text=f"\u03b8{si}", fill=C["text_dim"], font=("Menlo", 7))

        # ── Tracking dots: slider position cursors on each param row ──
        # Each dot shows WHERE in the sensitivity landscape the current
        # knob setting is, moving left/right as the user drags sliders.
        for _pname, _pos in self._param_positions.items():
            if _pname not in params:
                continue
            _pi = params.index(_pname)
            _is_sel = _pname == selected
            _color = self._param_colors.get(_pname, C["text_dim"])
            _dot_x = pad_l + int(_pos * plot_w)
            _dot_y = pad_t + int(cell_h * (_pi + 0.5))
            _r = 6 if _is_sel else 4
            self.create_oval(
                _dot_x - _r, _dot_y - _r, _dot_x + _r, _dot_y + _r,
                fill="#ffffff", outline=_color, width=2,
            )

        # Colorbar
        cb_x = w - pad_r + 8
        cb_top = pad_t
        cb_h = plot_h
        for ci in range(40):
            frac = ci / 40
            clr = self._val_to_color(0.7 * (1 - frac))
            cy0 = cb_top + int(frac * cb_h)
            cy1 = cb_top + int((frac + 1.0 / 40) * cb_h)
            self.create_rectangle(cb_x, cy0, cb_x + 10, cy1, fill=clr, outline="")
        self.create_text(cb_x + 13, cb_top, text="0.7", fill=C["text_dim"], font=("Menlo", 7), anchor="nw")
        self.create_text(cb_x + 13, cb_top + cb_h, text="0.0", fill=C["text_dim"], font=("Menlo", 7), anchor="sw")

        self.create_text(w // 2, 14, text="STRATEGY 4 // SENSITIVITY SPECTROGRAM", fill=C["text"], font=("Menlo", 10, "bold"))


class StrategyBarCanvas(tk.Canvas):
    """Grouped bar chart comparing Sobol indices across groups (generations or seeds)."""

    def __init__(self, parent, title="", bar_color=None, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._title = title
        self._bar_color = bar_color or C["accent_blue"]

    def set_data(self, groups):
        """groups: list of dicts, each with keys 'label' and 'sobol_total_order' (dict param->float)."""
        self._data = groups
        self.redraw()

    def redraw(self):
        self.delete("all")
        if not self._data:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pad_l, pad_r, pad_t, pad_b = 60, 15, 28, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        groups = self._data
        n_groups = len(groups)
        if n_groups == 0:
            return

        all_params = list(groups[0]["sobol_total_order"].keys())
        n_params = len(all_params)
        if n_params == 0:
            return

        group_w = plot_w / n_groups
        bar_w = max(3, (group_w * 0.8) / n_params)
        max_val = max(
            max(g["sobol_total_order"].get(p, 0) for p in all_params)
            for g in groups
        ) or 0.5

        # Grid
        for i in range(5):
            y = pad_t + int(plot_h * i / 4)
            self.create_line(pad_l, y, w - pad_r, y, fill=C["grid"], width=1)
            val = max_val * (1 - i / 4)
            self.create_text(pad_l - 5, y, text=f"{val:.2f}", fill=C["text_dim"], font=("Menlo", 7), anchor="e")

        for gi, group in enumerate(groups):
            gx = pad_l + int(gi * group_w)
            self.create_text(
                gx + int(group_w / 2), h - 8,
                text=group["label"], fill=C["text_dim"], font=("Menlo", 8)
            )

            for pi, pname in enumerate(all_params):
                val = group["sobol_total_order"].get(pname, 0)
                color = DEFAULT_PARAM_COLORS[pi % len(DEFAULT_PARAM_COLORS)]
                bx = gx + int(group_w * 0.1) + int(pi * (bar_w + 1))
                bar_h = int((val / max_val) * plot_h) if max_val > 0 else 0
                by = pad_t + plot_h - bar_h
                self.create_rectangle(bx, by, bx + int(bar_w), pad_t + plot_h, fill=color, outline="")

        # Legend
        for pi, pname in enumerate(all_params):
            color = DEFAULT_PARAM_COLORS[pi % len(DEFAULT_PARAM_COLORS)]
            short = pname.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass")
            lx = pad_l + pi * 120
            self.create_rectangle(lx, pad_t - 16, lx + 8, pad_t - 8, fill=color, outline="")
            self.create_text(lx + 12, pad_t - 12, text=short, fill=color, font=("Menlo", 7), anchor="w")

        self.create_text(w // 2, 10, text=self._title, fill=C["text"], font=("Menlo", 10, "bold"))


# -- Spectral / Koopman-synth canvases ----------------------------------------


class SpectralSynthCanvas(tk.Canvas):
    """Per-partial bars for pitch (Hz), damping (1/s), and amplitude.

    Updated live as the user drags PCE knobs.  Three colour-coded rows;
    each column is one Koopman mode.  Frequency bars are in Hz so users can
    read them exactly the same way as an Ableton spectrum analyser.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._prediction: dict[str, np.ndarray] | None = None
        self._bundle: SpectralBundle | None = None

    def set_prediction(self, bundle: SpectralBundle | None, pred: dict | None):
        self._bundle = bundle
        self._prediction = pred
        self.redraw()

    def _draw_row(self, y0, y1, label, values, color, fmt):
        self.create_text(12, (y0 + y1) // 2, text=label, fill=color, font=("Menlo", 9, "bold"), anchor="w")
        if values is None or len(values) == 0:
            return
        w = self.winfo_width()
        x0 = 110
        x1 = w - 20
        col_w = max(1, (x1 - x0) / len(values))
        vmax = float(max(np.abs(values).max(), 1e-12))
        row_h = y1 - y0 - 14
        for k, v in enumerate(values):
            cx0 = x0 + int(k * col_w) + 2
            cx1 = x0 + int((k + 1) * col_w) - 2
            # Center the bar vertically at the midline so negative σ bars
            # hang below.
            mid_y = (y0 + y1) // 2
            h_px = int((abs(v) / vmax) * row_h / 2)
            bar_top = mid_y - h_px if v >= 0 else mid_y
            bar_bot = mid_y if v >= 0 else mid_y + h_px
            self.create_rectangle(cx0, bar_top, cx1, bar_bot, fill=color, outline="")
            self.create_text((cx0 + cx1) // 2, y1 - 2, text=fmt.format(v), fill=C["text_dim"], font=("Menlo", 7))
            self.create_text((cx0 + cx1) // 2, y0 + 2, text=f"k{k}", fill=C["text_dim"], font=("Menlo", 7))

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        self.create_text(w // 2, 10, text="KOOPMAN SYNTH // per-partial spectrum", fill=C["text"], font=("Menlo", 10, "bold"))
        if self._prediction is None:
            self.create_text(w // 2, h // 2, text="(no spectral bundle loaded)", fill=C["text_dim"], font=("Menlo", 9))
            return

        band_top = 26
        band_h = (h - band_top - 10) / 3
        self._draw_row(
            band_top, int(band_top + band_h),
            "Hz", self._prediction["freq_hz"], C["accent1"], "{:.3e}",
        )
        self._draw_row(
            int(band_top + band_h), int(band_top + 2 * band_h),
            "σ (1/s)", self._prediction["sigma"], C["accent4"], "{:.2e}",
        )
        self._draw_row(
            int(band_top + 2 * band_h), int(band_top + 3 * band_h),
            "|a|", self._prediction["amp"], C["accent3"], "{:.2e}",
        )


class ModulationMatrixCanvas(tk.Canvas):
    """Heatmap of the Sobol total-order matrix (features × parameters).

    This is the *modulation routing* panel of the synth: row = Koopman
    feature, column = physical knob, cell colour = S_Ti.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._bundle: SpectralBundle | None = None

    def set_bundle(self, bundle: SpectralBundle | None):
        self._bundle = bundle
        self.redraw()

    def _val_to_color(self, v):
        v = max(0.0, min(float(v) / 0.7, 1.0))
        if v < 0.33:
            t = v / 0.33
            r, g, b = int(13 + t * 29), int(13 + t * 167), 222
        elif v < 0.66:
            t = (v - 0.33) / 0.33
            r, g, b = int(42 + t * 213), int(180 - t * 10), int(222 - t * 222)
        else:
            t = (v - 0.66) / 0.34
            r, g, b = 255, int(170 - t * 119), int(t * 102)
        return f"#{r:02x}{g:02x}{b:02x}"

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        self.create_text(w // 2, 10, text="MODULATION MATRIX // Sobol S_Ti", fill=C["text"], font=("Menlo", 10, "bold"))
        if self._bundle is None:
            self.create_text(w // 2, h // 2, text="(no spectral bundle loaded)", fill=C["text_dim"], font=("Menlo", 9))
            return

        pad_l, pad_r, pad_t, pad_b = 95, 25, 26, 20
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        mod = self._bundle.modulation_matrix
        n_feat, n_par = mod.shape
        if n_feat == 0 or n_par == 0:
            return
        cell_w = plot_w / n_par
        cell_h = plot_h / n_feat

        for fi, fname in enumerate(self._bundle.feature_names):
            self.create_text(
                pad_l - 5,
                pad_t + int(cell_h * (fi + 0.5)),
                text=fname,
                fill=C["text_dim"],
                font=("Menlo", 7),
                anchor="e",
            )
            for pi in range(n_par):
                x0 = pad_l + int(pi * cell_w)
                y0 = pad_t + int(fi * cell_h)
                x1 = x0 + int(cell_w)
                y1 = y0 + int(cell_h)
                self.create_rectangle(
                    x0, y0, x1, y1, fill=self._val_to_color(mod[fi, pi]), outline=C["border"]
                )
        for pi, pname in enumerate(self._bundle.parameter_names):
            short = pname.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass")
            self.create_text(
                pad_l + int(cell_w * (pi + 0.5)),
                pad_t + plot_h + 10,
                text=short,
                fill=C["text_dim"],
                font=("Menlo", 7),
            )


class ResynthOscilloscope(tk.Canvas):
    """Oscilloscope showing the resynthesized 1-D waveform at current knobs."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._signal: np.ndarray | None = None

    def set_signal(self, signal: np.ndarray | None):
        self._signal = signal
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return
        self.create_text(w // 2, 10, text="RESYNTH // reconstructed waveform", fill=C["text"], font=("Menlo", 10, "bold"))
        if self._signal is None or len(self._signal) == 0:
            self.create_text(w // 2, h // 2, text="(move a knob to synthesize)", fill=C["text_dim"], font=("Menlo", 9))
            return

        sig = np.asarray(self._signal, dtype=float)
        pad = 22
        plot_w = w - 2 * pad
        plot_h = h - pad - 14
        ymin, ymax = float(sig.min()), float(sig.max())
        if ymax - ymin < 1e-9:
            ymax = ymin + 1e-9
        zero_y = pad + int(plot_h * (ymax / (ymax - ymin)))

        # Zero line
        self.create_line(pad, zero_y, pad + plot_w, zero_y, fill=C["grid"], width=1)

        points = []
        n = len(sig)
        for i in range(n):
            px = pad + int((i / max(1, n - 1)) * plot_w)
            py = pad + int((1 - (sig[i] - ymin) / (ymax - ymin)) * plot_h)
            points.extend([px, py])
        if len(points) >= 4:
            self.create_line(points, fill=C["accent3"], width=2, smooth=True)


class PredictedProfileCanvas(tk.Canvas):
    """Per-observable predicted profile across the cell cycle (θ-bins).

    Shows one line per observable, X = θ (growth progress), Y = predicted
    value from the per-stage PCE surrogate.  The selected observable is
    drawn bold with units on the Y-axis; all others are dimmed.  When
    "(aggregate)" is selected, all observables are shown as % deviation
    from baseline on a common axis.

    This is the panel that lets biologists tune parameter knobs and
    directly see the predicted effect on familiar output observables.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._profiles: dict[str, np.ndarray] | None = None  # obs_name → (n_stages,)
        self._baselines: dict[str, float] = {}
        self._selected_obs: str = "(aggregate)"
        self._obs_colors: list[str] = []

    def set_profiles(
        self,
        profiles: dict[str, np.ndarray] | None,
        baselines: dict[str, float],
        selected_obs: str = "(aggregate)",
        compare_profile: np.ndarray | None = None,
    ):
        self._profiles = profiles
        self._baselines = baselines
        self._selected_obs = selected_obs
        self._compare_profile = compare_profile
        self.redraw()

    def redraw(self):
        self.delete("all")
        _w = self.winfo_width()
        _h = self.winfo_height()
        if _w < 10 or _h < 10:
            return

        self.create_text(
            _w // 2, 12,
            text="PREDICTED OBSERVABLE PROFILE // per cell-cycle stage",
            fill=C["text"], font=("Menlo", 10, "bold"),
        )

        if not self._profiles:
            self.create_text(
                _w // 2, _h // 2,
                text="Re-run: uv run uq quantify ... --export-path ./uq_results\n"
                "then reload the uq_results.json to see predicted profiles.\n"
                "(requires growth_stratified_surrogate/coefficients_per_output.npy)",
                fill=C["text_dim"], font=("Menlo", 8), justify="center",
            )
            return

        _pad_l, _pad_r, _pad_t, _pad_b = 65, 20, 28, 30
        _pw = _w - _pad_l - _pad_r
        _ph = _h - _pad_t - _pad_b
        if _pw < 10 or _ph < 10:
            return

        _sel = self._selected_obs
        _is_aggregate = _sel == "(aggregate)"
        _all_names = list(self._profiles.keys())
        _n_stages = len(next(iter(self._profiles.values())))
        _colors = DEFAULT_PARAM_COLORS

        if _is_aggregate:
            # Show all as % deviation from baseline
            _y_label = "Δ from baseline (%)"
            _all_vals: list[float] = []
            _lines: list[tuple[str, np.ndarray, str, int]] = []
            for _i, _name in enumerate(_all_names):
                _prof = self._profiles[_name]
                _bl = self._baselines.get(_name, 0.0)
                if abs(_bl) > 1e-15:
                    _pct = (_prof - _bl) / abs(_bl) * 100.0
                else:
                    _pct = np.zeros_like(_prof)
                _all_vals.extend(_pct.tolist())
                _color = _colors[_i % len(_colors)]
                _lines.append((_name, _pct, _color, 2))

            if not _all_vals:
                return
            _y_min = min(_all_vals) - 0.5
            _y_max = max(_all_vals) + 0.5
            if _y_max - _y_min < 1:
                _y_min, _y_max = -1, 1
        else:
            # Single observable in actual units
            _lbl, _unit, _fmt = _obs_label(_sel)
            _y_label = f"{_lbl} ({_unit})" if _unit else _lbl
            _prof = self._profiles.get(_sel)
            if _prof is None:
                return
            _margin = max(abs(_prof.max() - _prof.min()) * 0.15, 1e-6)
            _y_min = float(_prof.min()) - _margin
            _y_max = float(_prof.max()) + _margin

            _sel_idx = _all_names.index(_sel) if _sel in _all_names else 0
            _lines = []
            # Dim lines for other observables (normalized to this y range for shape)
            for _i, _name in enumerate(_all_names):
                if _name == _sel:
                    _lines.append((_name, _prof, _colors[_i % len(_colors)], 4))
                else:
                    _other = self._profiles[_name]
                    # Rescale other observable to fit in the same Y range (show shape)
                    _o_min, _o_max = float(_other.min()), float(_other.max())
                    _o_span = _o_max - _o_min or 1.0
                    _rescaled = _y_min + (_other - _o_min) / _o_span * (_y_max - _y_min)
                    _lines.append((_name, _rescaled, _colors[_i % len(_colors)], 2))

        _y_span = _y_max - _y_min or 1.0

        # Grid + Y-axis
        for _gi in range(5):
            _gy = _pad_t + int(_ph * _gi / 4)
            self.create_line(_pad_l, _gy, _w - _pad_r, _gy, fill=C["grid"], width=1, dash=(2, 4))
            _val = _y_max - _y_span * _gi / 4
            self.create_text(_pad_l - 5, _gy, text=f"{_val:.2f}", fill=C["text_dim"], font=("Menlo", 7), anchor="e")

        # Y-axis label
        self.create_text(10, (_pad_t + _pad_t + _ph) // 2, text=_y_label, fill=C["text_dim"], font=("Menlo", 8), angle=90)

        # X-axis labels
        for _si in range(_n_stages):
            _x = _pad_l + int(_pw * (_si + 0.5) / _n_stages)
            _lo_pct = int(100 * _si / _n_stages)
            _hi_pct = int(100 * (_si + 1) / _n_stages)
            self.create_text(_x, _h - 10, text=f"{_lo_pct}-{_hi_pct}%", fill=C["text_dim"], font=("Menlo", 7))
        self.create_text(_w // 2, _h - 2, text="\u03b8 (cell cycle progress)", fill=C["text_dim"], font=("Menlo", 7))

        # Baseline reference
        if not _is_aggregate:
            _bl = self._baselines.get(_sel, 0.0)
            if _y_min <= _bl <= _y_max:
                _bl_py = _pad_t + int((1 - (_bl - _y_min) / _y_span) * _ph)
                self.create_line(_pad_l, _bl_py, _w - _pad_r, _bl_py, fill=C["accent2"], width=1, dash=(6, 4))
                self.create_text(_w - _pad_r + 2, _bl_py, text="baseline", fill=C["accent2"], font=("Menlo", 7), anchor="w")

        # Draw lines
        for _name, _vals, _color, _width in _lines:
            _pts: list[int] = []
            for _si in range(_n_stages):
                _px = _pad_l + int(_pw * (_si + 0.5) / _n_stages)
                _py = _pad_t + int((1 - (_vals[_si] - _y_min) / _y_span) * _ph)
                _pts.extend([_px, _py])
            if len(_pts) >= 4:
                _stipple = "" if _width > 1 else "gray50"
                self.create_line(_pts, fill=_color, width=_width, smooth=True, stipple=_stipple)
            # Dots
            for _si in range(_n_stages):
                _px = _pad_l + int(_pw * (_si + 0.5) / _n_stages)
                _py = _pad_t + int((1 - (_vals[_si] - _y_min) / _y_span) * _ph)
                _r = 5 if _width > 2 else 3
                self.create_oval(_px - _r, _py - _r, _px + _r, _py + _r, fill=_color, outline="")

        # Comparison A overlay (dimmed dashed line)
        _cmp = getattr(self, "_compare_profile", None)
        if _cmp is not None and len(_cmp) == _n_stages:
            _cmp_pts: list[int] = []
            for _si in range(_n_stages):
                _px = _pad_l + int(_pw * (_si + 0.5) / _n_stages)
                _cval = float(_cmp[_si])
                if _is_aggregate:
                    _bl = list(self._baselines.values())[0] if self._baselines else 0.0
                    _cval = (_cval - _bl) / abs(_bl) * 100 if abs(_bl) > 1e-15 else 0.0
                _py = _pad_t + int((1 - (_cval - _y_min) / _y_span) * _ph)
                _cmp_pts.extend([_px, _py])
            if len(_cmp_pts) >= 4:
                self.create_line(_cmp_pts, fill=C["accent5"], width=2, dash=(8, 4), smooth=True)
                self.create_text(
                    _cmp_pts[-2], _cmp_pts[-1] - 10,
                    text="Config A", fill=C["accent5"], font=("Menlo", 7),
                )

        # Legend (top right)
        _lx = _w - _pad_r - 10
        _n_names = len(_all_names)
        for _i, _name in enumerate(reversed(_all_names)):
            _ci = (_n_names - 1 - _i) % len(_colors)
            _color = _colors[_ci]
            _short_lbl, _, _ = _obs_label(_name)
            _ly = _pad_t + 2 + _i * 12
            self.create_rectangle(_lx - 50, _ly, _lx - 42, _ly + 8, fill=_color, outline="")
            self.create_text(_lx - 40, _ly + 4, text=_short_lbl, fill=_color, font=("Menlo", 7), anchor="w")


# -- Main Application ---------------------------------------------------------


class UQDawSimpleApp:
    """Tkinter UQ Simple DAW — all 4 RFC006 strategies."""

    def __init__(self, root, data_path=None):
        self.root = root
        self.root.title("UQ DAW Simple // RFC006")
        self.root.geometry("1400x900")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1000, 700)

        self.data = None
        self.surr_data = None
        self.selected_param = tk.StringVar()
        self.selected_observable = tk.StringVar(value="(aggregate)")
        self.param_colors = {}
        self._baselines: dict[str, float] = {}  # obs_short_name → PCE(midpoint)
        self._observable_full_names: list[str] = []
        self._view_mode = tk.StringVar(value="standard")
        self._compare_snapshot: dict[str, Any] | None = None  # frozen config A
        self._compare_profile_a: np.ndarray | None = None
        self._history: list[dict[str, float]] = []
        self._history_idx: int = -1
        self._history_debounce_id: str | None = None

        self._build_ui()

        # Undo/redo keybindings
        self.root.bind_all("<Control-z>", lambda e: self._undo())
        self.root.bind_all("<Control-y>", lambda e: self._redo())
        self.root.bind_all("<Command-z>", lambda e: self._undo())  # macOS
        self.root.bind_all("<Command-Shift-z>", lambda e: self._redo())  # macOS

        if data_path:
            self._load_file(data_path)

    def _build_ui(self):
        # -- Top bar --
        top = tk.Frame(self.root, bg=C["panel"], height=40)
        top.pack(fill="x", padx=4, pady=(4, 0))
        top.pack_propagate(False)

        tk.Label(top, text="UQ DAW Simple // RFC006", bg=C["panel"], fg=C["accent1"], font=("Menlo", 13, "bold")).pack(side="left", padx=10)

        # View mode radio buttons
        _mode_frame = tk.Frame(top, bg=C["panel"])
        _mode_frame.pack(side="left", padx=20)
        for _mode_val, _mode_label in [("simple", "Simple"), ("standard", "Standard"), ("expert", "Expert")]:
            tk.Radiobutton(
                _mode_frame, text=_mode_label, variable=self._view_mode,
                value=_mode_val, bg=C["panel"], fg=C["text"], selectcolor=C["panel_light"],
                activebackground=C["panel"], activeforeground=C["accent1"],
                font=("Menlo", 9), indicatoron=True,
                command=self._apply_view_mode,
            ).pack(side="left", padx=4)

        tk.Button(
            top, text="Load JSON", command=self._open_file,
            bg=C["panel_light"], fg=C["text"], font=("Menlo", 10), relief="flat", cursor="hand2",
        ).pack(side="right", padx=10)

        tk.Button(
            top, text="Inverse Design", command=self._run_inverse_design,
            bg=C["panel_light"], fg=C["accent3"], font=("Menlo", 10), relief="flat", cursor="hand2",
        ).pack(side="right", padx=4)

        self._compare_btn = tk.Button(
            top, text="Compare", command=self._on_compare_toggle,
            bg=C["panel_light"], fg=C["accent5"], font=("Menlo", 10), relief="flat", cursor="hand2",
        )
        self._compare_btn.pack(side="right", padx=4)

        self._clear_compare_btn = tk.Button(
            top, text="Clear A/B", command=self._on_compare_clear,
            bg=C["panel_light"], fg=C["text_dim"], font=("Menlo", 10), relief="flat", cursor="hand2",
        )
        self._clear_compare_btn.pack(side="right", padx=2)

        tk.Button(
            top, text="Save Patch", command=self._save_patch,
            bg=C["panel_light"], fg=C["accent4"], font=("Menlo", 10), relief="flat", cursor="hand2",
        ).pack(side="right", padx=4)

        tk.Button(
            top, text="Load Patch", command=self._load_patch,
            bg=C["panel_light"], fg=C["accent4"], font=("Menlo", 10), relief="flat", cursor="hand2",
        ).pack(side="right", padx=4)

        self.status_label = tk.Label(top, text="No data loaded", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 9))
        self.status_label.pack(side="right", padx=10)

        # -- Top section: left controls + right response curves --
        _top_pane = tk.PanedWindow(self.root, orient="horizontal", bg=C["bg"], sashwidth=4, sashrelief="flat")
        _top_pane.pack(fill="both", expand=True, padx=4, pady=(4, 1))

        self.left_frame = tk.Frame(_top_pane, bg=C["bg"], width=260)
        _top_pane.add(self.left_frame, minsize=230)

        self.right_top_frame = tk.Frame(_top_pane, bg=C["bg"])
        _top_pane.add(self.right_top_frame, minsize=500)

        # -- Bottom section --
        self.bottom_frame = tk.Frame(self.root, bg=C["bg"])
        self.bottom_frame.pack(fill="both", expand=True, padx=4, pady=(1, 4))

        # -- Left: Solo selector --
        solo_frame = tk.LabelFrame(self.left_frame, text="SOLO PARAMETER", bg=C["panel"], fg=C["accent1"], font=("Menlo", 10, "bold"), labelanchor="n")
        solo_frame.pack(fill="x", padx=4, pady=4)

        self.param_menu = ttk.Combobox(solo_frame, textvariable=self.selected_param, state="readonly", font=("Menlo", 10))
        self.param_menu.pack(fill="x", padx=8, pady=8)
        self.selected_param.trace_add("write", lambda *_: self._on_param_change())

        # -- Left: Observable selector (searchable for large sets like transcriptome) --
        obs_frame = tk.LabelFrame(self.left_frame, text="OBSERVABLE", bg=C["panel"], fg=C["accent4"], font=("Menlo", 10, "bold"), labelanchor="n")
        obs_frame.pack(fill="x", padx=4, pady=4)

        self._obs_search_var = tk.StringVar()
        self._obs_search_var.trace_add("write", lambda *_: self._filter_observables())
        self._obs_search_entry = tk.Entry(
            obs_frame, textvariable=self._obs_search_var,
            bg=C["panel_light"], fg=C["accent4"], font=("Menlo", 9),
            insertbackground=C["accent4"],
        )
        self._obs_search_entry.pack(fill="x", padx=8, pady=(8, 2))
        self._obs_search_entry.insert(0, "")

        self._obs_listbox = tk.Listbox(
            obs_frame, bg=C["panel_light"], fg=C["text"], font=("Menlo", 8),
            selectbackground=C["accent4"], selectforeground=C["bg"],
            height=6, activestyle="none", exportselection=False,
        )
        self._obs_listbox.pack(fill="x", padx=8, pady=(0, 8))
        self._obs_listbox.bind("<<ListboxSelect>>", self._on_obs_listbox_select)

        self._obs_all_choices: list[str] = ["(aggregate)"]
        self._obs_count_label = tk.Label(
            obs_frame, text="", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 7),
        )
        self._obs_count_label.pack(fill="x", padx=8)

        # -- Left: PCE sliders --
        slider_frame = tk.LabelFrame(self.left_frame, text="PCE SURROGATE KNOBS", bg=C["panel"], fg=C["accent3"], font=("Menlo", 10, "bold"), labelanchor="n")
        slider_frame.pack(fill="x", padx=4, pady=4)

        self.slider_container = tk.Frame(slider_frame, bg=C["panel"])
        self.slider_container.pack(fill="x", padx=4, pady=4)

        self.sliders = {}
        self.slider_labels = {}

        # -- Left: Prediction readout --
        self.readout_label = tk.Label(self.left_frame, text="\u0176 = ---", bg=C["panel"], fg=C["accent3"], font=("Menlo", 16, "bold"))
        self.readout_label.pack(fill="x", padx=4, pady=(4, 2))

        # -- Left: Target mode --
        target_frame = tk.LabelFrame(self.left_frame, text="TARGET \u0176", bg=C["panel"], fg=C["accent2"], font=("Menlo", 10, "bold"), labelanchor="n")
        target_frame.pack(fill="x", padx=4, pady=4)

        _tf_row = tk.Frame(target_frame, bg=C["panel"])
        _tf_row.pack(fill="x", padx=4, pady=4)

        self.target_entry = tk.Entry(_tf_row, bg=C["panel_light"], fg=C["accent2"], font=("Menlo", 12), width=10, insertbackground=C["accent2"])
        self.target_entry.pack(side="left", padx=4)

        tk.Button(
            _tf_row, text="Find", command=self._find_target,
            bg=C["panel_light"], fg=C["accent2"], font=("Menlo", 10, "bold"), relief="flat", cursor="hand2",
        ).pack(side="left", padx=4)

        # -- Left: Strategy info --
        self.strategy_label = tk.Label(self.left_frame, text="Strategies: ---", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 9), wraplength=240, justify="left")
        self.strategy_label.pack(fill="x", padx=8, pady=(2, 4))

        # -- Left: Condition selector (multi-parca, hidden by default) --
        self._condition_frame = tk.LabelFrame(
            self.left_frame, text="GROWTH CONDITION", bg=C["panel"],
            fg=C["accent_gold"], font=("Menlo", 10, "bold"), labelanchor="n",
        )
        self._selected_condition = tk.StringVar()
        self._condition_menu = ttk.Combobox(
            self._condition_frame, textvariable=self._selected_condition,
            state="readonly", font=("Menlo", 10),
        )
        self._condition_menu.pack(fill="x", padx=8, pady=8)
        self._selected_condition.trace_add("write", lambda *_: self._on_condition_change())
        self._conditions_data: dict[str, Any] | None = None
        # Frame is packed only when multi-condition data is loaded

        # -- Left: Context panel (per-observable detail) --
        ctx_frame = tk.LabelFrame(
            self.left_frame, text="CONTEXT", bg=C["panel"],
            fg=C["accent_blue"], font=("Menlo", 10, "bold"), labelanchor="n",
        )
        ctx_frame.pack(fill="x", padx=4, pady=4)
        self._context_text = tk.Text(
            ctx_frame, bg=C["panel_light"], fg=C["text"], font=("Menlo", 8),
            height=8, wrap="word", state="disabled", borderwidth=0,
            highlightthickness=0, padx=6, pady=4,
        )
        self._context_text.pack(fill="x", padx=4, pady=4)

        # -- Build visualization panels --
        self._build_viz_panels()

    def _build_viz_panels(self):
        # Top-right: Response curves
        self.response_canvas = ResponseCurveCanvas(
            self.right_top_frame, on_marker_drag=self._on_curve_drag, height=280,
        )
        self.response_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom: sensitivity spectrogram (strategy 4)
        self.heatmap_canvas = HeatmapCanvas(self.bottom_frame, height=180)
        self.heatmap_canvas.pack(fill="x", padx=2, pady=(1, 1))

        # Bottom: predicted observable profile (per-stage PCE) — main output panel
        self.profile_canvas = PredictedProfileCanvas(self.bottom_frame, height=280)
        self.profile_canvas.pack(fill="both", expand=True, padx=2, pady=(1, 4))

    def _open_file(self):
        path = filedialog.askopenfilename(title="Load UQ Results", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self._load_file(path)

    def _load_file(self, path):
        path = Path(path)
        if not path.exists():
            self.status_label.config(text=f"Not found: {path.name}")
            return

        with open(path) as f:
            self.data = json.load(f)

        export_dir = path.parent
        self.surr_data = self._load_surrogates(export_dir)

        params = list(self.data["parameters"].keys())
        self.param_colors = {p: DEFAULT_PARAM_COLORS[i % len(DEFAULT_PARAM_COLORS)] for i, p in enumerate(params)}
        self.response_canvas._param_colors = self.param_colors
        self.heatmap_canvas._param_colors = self.param_colors

        self.param_menu["values"] = params
        self.selected_param.set(params[0])

        # Populate observable selector
        obs_names = self.data.get("observable_names", [])
        short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]
        self._obs_all_choices = ["(aggregate)"] + short_names
        self._observable_full_names = obs_names
        self.selected_observable.set("(aggregate)")
        self._obs_search_var.set("")
        self._filter_observables()
        self._obs_count_label.config(text=f"{len(obs_names)} observables")

        # Compute baselines (PCE at parameter midpoints)
        self._baselines = {}
        if self.surr_data and self.surr_data.get("bounds") is not None:
            bounds = self.surr_data["bounds"]
            mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
            mid_norm = normalize_to_germ(mid, bounds)
            self._baselines["(aggregate)"] = float(
                legendre_eval(mid_norm, self.surr_data["pop_coeffs"], self.surr_data["pop_mi"])
            )
            per_out = self.surr_data.get("pop_coeffs_per_output")
            if per_out is not None:
                for _i, _name in enumerate(obs_names):
                    _short = _name.split("__")[-1] if "__" in _name else _name
                    self._baselines[_short] = float(
                        legendre_eval(mid_norm, per_out[_i], self.surr_data["pop_mi"])
                    )

        # Detect multi-condition data
        mc_data = self.data.get("per_condition")
        if mc_data and len(mc_data) > 1:
            self._conditions_data = mc_data
            cond_ids = list(mc_data.keys())
            self._condition_menu["values"] = cond_ids
            self._selected_condition.set(cond_ids[0])
            self._condition_frame.pack(fill="x", padx=4, pady=4, before=self.strategy_label)
        else:
            self._conditions_data = None
            self._condition_frame.pack_forget()

        self._build_sliders(params)
        self._update_strategy_info()
        self._update_all_viz()
        self._update_context_panel()
        self.status_label.config(text=f"Loaded: {path.name}")

    def _load_surrogates(self, export_dir):
        pop_dir = export_dir / "population_surrogate"
        if not pop_dir.exists():
            return None
        try:
            result = {
                "pop_coeffs": np.load(pop_dir / "coefficients.npy"),
                "pop_mi": np.load(pop_dir / "multi_indices.npy"),
            }
            bounds_path = pop_dir / "input_bounds.npy"
            result["bounds"] = np.load(bounds_path) if bounds_path.exists() else None

            # Per-output coefficients (for observable selector)
            per_out_path = pop_dir / "coefficients_per_output.npy"
            result["pop_coeffs_per_output"] = (
                np.load(per_out_path) if per_out_path.exists() else None
            )

            # Per-stage coefficients (for exact growth-stratified prediction)
            gs_dir = export_dir / "growth_stratified_surrogate"
            gs_per_out = gs_dir / "coefficients_per_output.npy"
            gs_mi = gs_dir / "multi_indices.npy"
            if gs_per_out.exists() and gs_mi.exists():
                result["gs_coeffs_per_output"] = np.load(gs_per_out)
                result["gs_mi"] = np.load(gs_mi)
            else:
                result["gs_coeffs_per_output"] = None
                result["gs_mi"] = None

            return result
        except Exception:
            return None

    def _filter_observables(self) -> None:
        """Filter the observable listbox by the search entry text."""
        query = self._obs_search_var.get().strip().lower()
        self._obs_listbox.delete(0, tk.END)
        for _choice in self._obs_all_choices:
            if not query or query in _choice.lower():
                self._obs_listbox.insert(tk.END, _choice)
        # Re-select the current observable if it's still visible
        _cur = self.selected_observable.get()
        for _i in range(self._obs_listbox.size()):
            if self._obs_listbox.get(_i) == _cur:
                self._obs_listbox.selection_set(_i)
                self._obs_listbox.see(_i)
                break

    def _on_obs_listbox_select(self, event) -> None:
        """Handle click on an observable in the listbox."""
        _sel = self._obs_listbox.curselection()
        if not _sel:
            return
        _choice = self._obs_listbox.get(_sel[0])
        if _choice != self.selected_observable.get():
            self.selected_observable.set(_choice)
            self._on_param_change()
        self._update_context_panel()

    def _update_context_panel(self) -> None:
        """Populate the context sidebar with per-observable detail."""
        self._context_text.config(state="normal")
        self._context_text.delete("1.0", "end")

        obs_choice = self.selected_observable.get()
        if not self.data or not self.surr_data:
            self._context_text.insert("end", "Load data first.")
            self._context_text.config(state="disabled")
            return

        lines: list[str] = []
        params = list(self.data["parameters"].keys())
        per_out = self.surr_data.get("pop_coeffs_per_output")
        mi = self.surr_data.get("pop_mi")
        bounds = self.surr_data.get("bounds")
        obs_names = getattr(self, "_observable_full_names", [])
        short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]

        # Per-observable Sobol ranking (top-3 params)
        if per_out is not None and mi is not None and obs_choice in short_names and bounds is not None:
            obs_idx = short_names.index(obs_choice)
            coeffs = per_out[obs_idx]
            # Compute per-param sensitivity as |dŶ/dx_i| at midpoint
            mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
            mid_norm = normalize_to_germ(mid, bounds)
            sensitivities = []
            for pi in range(len(params)):
                delta = (bounds[pi, 1] - bounds[pi, 0]) * 0.005
                x_plus = mid.copy()
                x_plus[pi] = min(mid[pi] + delta, bounds[pi, 1])
                x_minus = mid.copy()
                x_minus[pi] = max(mid[pi] - delta, bounds[pi, 0])
                y_plus = legendre_eval(normalize_to_germ(x_plus, bounds), coeffs, mi)
                y_minus = legendre_eval(normalize_to_germ(x_minus, bounds), coeffs, mi)
                sensitivities.append(abs(y_plus - y_minus) / (2 * delta + 1e-12))
            ranked = sorted(zip(params, sensitivities), key=lambda ps: -ps[1])
            lines.append("── Top-3 drivers ──")
            for name, sens in ranked[:3]:
                short = name.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass")
                lines.append(f"  {short}: {sens:.3e}")
            lines.append("")

        # Observable metadata
        _lbl, _unit, _fmt = _obs_label(obs_choice if obs_choice != "(aggregate)" else "")
        baseline = self._baselines.get(obs_choice)
        if baseline is not None:
            lines.append(f"Baseline: {baseline:{_fmt}} {_unit}")
            # Current deviation
            if self.sliders and bounds is not None:
                x = np.array([float(self.sliders[p].get()) for p in params])
                if per_out is not None and obs_choice in short_names:
                    obs_idx = short_names.index(obs_choice)
                    x_norm = normalize_to_germ(x, bounds)
                    current = legendre_eval(x_norm, per_out[obs_idx], mi)
                    delta_pct = (current - baseline) / abs(baseline) * 100 if baseline != 0 else 0
                    sign = "+" if delta_pct >= 0 else ""
                    lines.append(f"Current:  {current:{_fmt}} {_unit} ({sign}{delta_pct:.1f}%)")
            lines.append("")

        # Surrogate quality
        s1 = self.data.get("phase1_population", {})
        if s1:
            lines.append("── Surrogate quality ──")
            s_ti = s1.get("sobol_total_order", {}).get(obs_choice, None)
            if s_ti is not None:
                lines.append(f"  S_Ti (pop): {s_ti:.3f}")

        if not lines:
            lines.append("Select an observable for context.")

        self._context_text.insert("end", "\n".join(lines))
        self._context_text.config(state="disabled")

    def _build_sliders(self, params):
        for w in self.slider_container.winfo_children():
            w.destroy()
        self.sliders.clear()
        self.slider_labels.clear()

        if self.surr_data is None or self.surr_data.get("bounds") is None:
            return

        bounds = self.surr_data["bounds"]
        for i, pname in enumerate(params):
            color = self.param_colors.get(pname, DEFAULT_PARAM_COLORS[i % len(DEFAULT_PARAM_COLORS)])
            lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
            mid = (lo + hi) / 2

            frame = tk.Frame(self.slider_container, bg=C["panel"])
            frame.pack(fill="x", pady=3)

            short = pname.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass")
            tk.Label(frame, text=short, bg=C["panel"], fg=color, font=("Menlo", 9, "bold"), width=12, anchor="w").pack(side="left", padx=4)

            slider = tk.Scale(
                frame, from_=lo, to=hi, orient="horizontal", resolution=(hi - lo) / 200, length=140,
                bg=C["panel"], fg=color, troughcolor=C["panel_light"], highlightthickness=0,
                font=("Menlo", 8), showvalue=False,
                command=lambda v, p=pname: self._on_slider_change(p, v),
            )
            slider.set(mid)
            slider.pack(side="left", padx=2)

            val_label = tk.Label(frame, text=f"{mid:.3f}", bg=C["panel"], fg=color, font=("Menlo", 9), width=8, anchor="e")
            val_label.pack(side="left", padx=4)

            self.sliders[pname] = slider
            self.slider_labels[pname] = val_label

    def _on_slider_change(self, pname, value):
        value = float(value)
        if pname in self.slider_labels:
            self.slider_labels[pname].config(text=f"{value:.3f}")
        self._update_response_curves()
        # Debounced history push (200ms after last movement)
        if self._history_debounce_id is not None:
            self.root.after_cancel(self._history_debounce_id)
        self._history_debounce_id = self.root.after(200, self._push_history)

    def _push_history(self) -> None:
        """Record current slider state in the undo history."""
        if not self.sliders:
            return
        state = {p: float(s.get()) for p, s in self.sliders.items()}
        # Truncate forward history if we undid
        if self._history_idx < len(self._history) - 1:
            self._history = self._history[: self._history_idx + 1]
        self._history.append(state)
        if len(self._history) > 50:
            self._history = self._history[-50:]
        self._history_idx = len(self._history) - 1

    def _undo(self) -> None:
        if self._history_idx <= 0:
            return
        self._history_idx -= 1
        self._restore_history(self._history[self._history_idx])

    def _redo(self) -> None:
        if self._history_idx >= len(self._history) - 1:
            return
        self._history_idx += 1
        self._restore_history(self._history[self._history_idx])

    def _restore_history(self, state: dict[str, float]) -> None:
        """Set sliders to a historical state without pushing new history."""
        if self._history_debounce_id is not None:
            self.root.after_cancel(self._history_debounce_id)
            self._history_debounce_id = None
        for pname, val in state.items():
            if pname in self.sliders:
                self.sliders[pname].set(val)
        # Update visuals (slider callback already fired for each .set())

    def _on_curve_drag(self, pname, x_normalized):
        if self.surr_data is None:
            return
        bounds = self.surr_data.get("bounds")
        if bounds is None:
            return
        params = list(self.data["parameters"].keys())
        if pname not in params:
            return
        pi = params.index(pname)
        lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
        physical_val = max(lo, min(hi, lo + x_normalized * (hi - lo)))
        if pname in self.sliders:
            self.sliders[pname].set(physical_val)

    def _on_param_change(self):
        if not self.sliders:
            return
        self._update_all_viz()

    def _update_strategy_info(self):
        if self.data is None:
            return
        s2 = self.data.get("strategy2_by_generation", {})
        s3 = self.data.get("strategy3_by_seed", {})
        n_stages = self.data.get("phase2_growth_stratified", {}).get("n_stages", "?")
        s2_text = f"{s2.get('n_generations', 0)} gen" if "generations" in s2 else "N/A"
        s3_text = f"{s3.get('n_seeds', 0)} seeds" if "seeds" in s3 else "N/A"
        self.strategy_label.config(
            text=f"S1: bulk | S2: {s2_text} | S3: {s3_text} | S4: {n_stages} stages"
        )

    def _on_condition_change(self) -> None:
        """Handle condition selector change — reload per-condition data."""
        cond_id = self._selected_condition.get()
        if not cond_id or not self._conditions_data:
            return
        cond_data = self._conditions_data.get(cond_id)
        if cond_data:
            # Update the Sobol data for the selected condition
            self.data["phase1_population"] = cond_data
            self._update_all_viz()
            self.status_label.config(text=f"Condition: {cond_id}")

    def _apply_view_mode(self) -> None:
        """Show/hide panels based on view mode (Simple/Standard/Expert)."""
        mode = self._view_mode.get()

        # Simple: sliders + readout + profile only
        # Standard: + response curves + spectrogram (default)
        # Expert: + context sidebar + patches + target + spectral

        show_response = mode in ("standard", "expert")
        show_spectrogram = mode in ("standard", "expert")
        show_context = mode == "expert"

        if hasattr(self, "response_canvas"):
            if show_response:
                self.response_canvas.pack(fill="both", expand=True, padx=2, pady=2)
            else:
                self.response_canvas.pack_forget()

        if hasattr(self, "heatmap_canvas"):
            if show_spectrogram:
                self.heatmap_canvas.pack(fill="x", padx=2, pady=(1, 1))
            else:
                self.heatmap_canvas.pack_forget()

        if hasattr(self, "_context_text"):
            ctx_parent = self._context_text.master
            if show_context:
                ctx_parent.pack(fill="x", padx=4, pady=4)
            else:
                ctx_parent.pack_forget()

    def _update_all_viz(self):
        if self.data is None:
            return
        selected = self.selected_param.get()
        params = list(self.data["parameters"].keys())

        # Strategy 4: spectrogram
        stages = self.data.get("phase2_growth_stratified", {}).get("stages", [])
        self.heatmap_canvas.set_data(stages, params, selected)

        self._update_response_curves()

    def _update_response_curves(self):
        if self.data is None or self.surr_data is None:
            return

        params = list(self.data["parameters"].keys())
        bounds = self.surr_data.get("bounds")
        if bounds is None:
            return
        if not self.sliders or any(p not in self.sliders for p in params):
            return

        selected = self.selected_param.get()
        x = np.array([float(self.sliders[p].get()) for p in params])
        x_norm = normalize_to_germ(x, bounds)

        # Resolve which coefficients to use based on observable selector
        obs_choice = self.selected_observable.get()
        coeffs = self.surr_data["pop_coeffs"]  # default: aggregate
        mi = self.surr_data["pop_mi"]
        obs_label = "\u0176"

        per_out = self.surr_data.get("pop_coeffs_per_output")
        if per_out is not None and obs_choice != "(aggregate)":
            obs_names = getattr(self, "_observable_full_names", [])
            short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]
            if obs_choice in short_names:
                obs_idx = short_names.index(obs_choice)
                coeffs = per_out[obs_idx]
                obs_label = obs_choice

        pop_y = legendre_eval(x_norm, coeffs, mi)

        # Format readout with units + baseline delta
        _lbl, _unit, _fmt = _obs_label(obs_label if obs_label != "\u0176" else "(aggregate)")
        _val_str = f"{pop_y:{_fmt}}"
        _baseline_key = obs_label if obs_label != "\u0176" else "(aggregate)"
        _baseline = self._baselines.get(_baseline_key)
        if _baseline is not None and _baseline != 0:
            _delta_pct = (pop_y - _baseline) / abs(_baseline) * 100
            _sign = "+" if _delta_pct >= 0 else ""
            _unit_str = f" {_unit}" if _unit else ""
            self.readout_label.config(
                text=f"{_lbl} = {_val_str}{_unit_str}  ({_sign}{_delta_pct:.1f}%)"
            )
        else:
            _unit_str = f" {_unit}" if _unit else ""
            self.readout_label.config(text=f"{_lbl} = {_val_str}{_unit_str}")

        n_sweep = 80
        curves = {}
        markers = {}
        all_y = []
        local_sensitivity = {}
        param_positions = {}  # normalized slider positions for heatmap tracking dots

        for pi, pname in enumerate(params):
            lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
            sv = np.linspace(lo, hi, n_sweep)
            sv_norm = (sv - lo) / (hi - lo + 1e-12)
            sy = []
            for v in sv:
                x_sw = x.copy()
                x_sw[pi] = v
                x_sw_n = normalize_to_germ(x_sw, bounds)
                sy.append(legendre_eval(x_sw_n, coeffs, mi))
            sy = np.array(sy)
            curves[pname] = (sv_norm, sy)
            all_y.extend(sy.tolist())

            cur_norm = (x[pi] - lo) / (hi - lo + 1e-12)
            markers[pname] = (cur_norm, pop_y)
            param_positions[pname] = cur_norm

            delta = (hi - lo) * 0.005
            x_plus = x.copy()
            x_plus[pi] = min(x[pi] + delta, hi)
            x_minus = x.copy()
            x_minus[pi] = max(x[pi] - delta, lo)
            y_plus = legendre_eval(normalize_to_germ(x_plus, bounds), coeffs, mi)
            y_minus = legendre_eval(normalize_to_germ(x_minus, bounds), coeffs, mi)
            local_sensitivity[pname] = abs(y_plus - y_minus) / (2 * delta + 1e-12)

        if all_y:
            y_min = min(all_y) - 0.1 * abs(min(all_y))
            y_max = max(all_y) + 0.1 * abs(max(all_y))
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
        else:
            y_min, y_max = -1, 1

        _baseline_key = obs_label if obs_label != "\u0176" else "(aggregate)"
        _baseline_val = self._baselines.get(_baseline_key)
        _lbl, _unit, _ = _obs_label(_baseline_key)
        self.response_canvas.set_curves(
            curves, markers, selected, (y_min, y_max),
            baseline=_baseline_val,
            obs_label=_lbl,
            obs_unit=_unit,
        )

        # Per-stage predictions for spectrogram
        stage_data = self.data.get("phase2_growth_stratified", {}).get("stages", [])
        stage_predictions = None
        gs_per_out = self.surr_data.get("gs_coeffs_per_output") if self.surr_data else None
        gs_mi = self.surr_data.get("gs_mi") if self.surr_data else None
        n_obs = len(getattr(self, "_observable_full_names", [])) or 1

        if stage_data and gs_per_out is not None and gs_mi is not None:
            # Exact per-stage PCE evaluation using growth_stratified_surrogate
            n_stages = len(stage_data)
            stage_predictions = np.zeros(n_stages)

            # Resolve which observable column to show
            obs_choice = self.selected_observable.get()
            obs_idx = 0  # default: first observable
            if obs_choice != "(aggregate)":
                obs_names = getattr(self, "_observable_full_names", [])
                short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]
                if obs_choice in short_names:
                    obs_idx = short_names.index(obs_choice)

            for si in range(n_stages):
                # The combined PCE output layout is: stage0_obs0, stage0_obs1, ..., stage1_obs0, ...
                col_idx = si * n_obs + obs_idx
                if col_idx < gs_per_out.shape[0]:
                    stage_predictions[si] = legendre_eval(x_norm, gs_per_out[col_idx], gs_mi)

        elif stage_data and local_sensitivity:
            # Fallback: linear approximation (old behavior)
            n_stages = len(stage_data)
            stage_predictions = np.zeros(n_stages)
            param_effects = {}
            for pi, pname in enumerate(params):
                deviation = x_norm[pi]
                param_effects[pname] = local_sensitivity[pname] * deviation

            for si in range(n_stages):
                contrib = sum(
                    param_effects[pname] * stage_data[si]["sobol_total_order"].get(pname, 0)
                    for pname in params
                )
                stage_predictions[si] = pop_y + contrib

        self.heatmap_canvas.set_data(
            stage_data, params, selected,
            stage_predictions=stage_predictions,
            param_positions=param_positions,
        )

        # ── Predicted observable profile (per-stage PCE for all observables) ──
        self._update_predicted_profile(x_norm, stage_data)

    def _update_predicted_profile(self, x_norm: np.ndarray, stage_data: list) -> None:
        """Compute per-observable predicted profiles across θ-bins.

        Uses the per-stage per-output coefficients from
        ``growth_stratified_surrogate/coefficients_per_output.npy``.
        """
        gs_per_out = self.surr_data.get("gs_coeffs_per_output") if self.surr_data else None
        gs_mi = self.surr_data.get("gs_mi") if self.surr_data else None
        obs_names = getattr(self, "_observable_full_names", [])

        if gs_per_out is None or gs_mi is None or not stage_data or not obs_names:
            self.profile_canvas.set_profiles(None, {})
            return

        n_stages = len(stage_data)
        n_obs = len(obs_names)
        profiles: dict[str, np.ndarray] = {}

        for _oi, _full_name in enumerate(obs_names):
            _short = _full_name.split("__")[-1] if "__" in _full_name else _full_name
            _vals = np.zeros(n_stages)
            for _si in range(n_stages):
                _col = _si * n_obs + _oi
                if _col < gs_per_out.shape[0]:
                    _vals[_si] = legendre_eval(x_norm, gs_per_out[_col], gs_mi)
            profiles[_short] = _vals

        _obs_sel = self.selected_observable.get()
        self.profile_canvas.set_profiles(
            profiles, self._baselines, selected_obs=_obs_sel,
            compare_profile=self._compare_profile_a,
        )

    # ── What-If Comparison ─────────────────────────────────────────

    def _on_compare_toggle(self) -> None:
        """Freeze current slider state as 'Configuration A'."""
        if not self.sliders:
            return
        self._compare_snapshot = {p: float(s.get()) for p, s in self.sliders.items()}
        # Capture current profile if available
        if hasattr(self, "profile_canvas") and self.profile_canvas._profiles:
            obs = self.selected_observable.get()
            if obs in self.profile_canvas._profiles:
                self._compare_profile_a = self.profile_canvas._profiles[obs].copy()
            elif obs == "(aggregate)" and self.profile_canvas._profiles:
                # Store first observable as representative
                first_key = next(iter(self.profile_canvas._profiles))
                self._compare_profile_a = self.profile_canvas._profiles[first_key].copy()
        self._compare_btn.config(fg=C["accent3"], text="A frozen")
        self.status_label.config(text="Config A frozen — adjust sliders for B")

    def _on_compare_clear(self) -> None:
        """Clear the comparison snapshot."""
        self._compare_snapshot = None
        self._compare_profile_a = None
        self._compare_btn.config(fg=C["accent5"], text="Compare")
        self.status_label.config(text="Comparison cleared")
        self._update_response_curves()

    # ── Patch save/load ────────────────────────────────────────────

    def _get_patches_dir(self) -> Path:
        d = Path("patches")
        d.mkdir(exist_ok=True)
        return d

    def _save_patch(self) -> None:
        """Save the current slider values as a named JSON patch."""
        if not self.sliders:
            return
        patch = {p: float(s.get()) for p, s in self.sliders.items()}

        patches_dir = self._get_patches_dir()
        existing = sorted(patches_dir.glob("patch_*.json"))
        idx = len(existing)
        path = patches_dir / f"patch_{idx:03d}.json"
        path.write_text(json.dumps(patch, indent=2))
        self.status_label.config(text=f"Saved: {path.name}")

    def _load_patch(self) -> None:
        """Load a previously saved patch and set sliders."""
        patches_dir = self._get_patches_dir()
        files = sorted(patches_dir.glob("patch_*.json"))
        if not files:
            self.status_label.config(text="No patches saved yet")
            return
        path = filedialog.askopenfilename(
            title="Load Patch",
            initialdir=str(patches_dir),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            patch = json.loads(Path(path).read_text())
        except Exception as e:
            self.status_label.config(text=f"Bad patch: {e}")
            return
        for pname, val in patch.items():
            if pname in self.sliders:
                self.sliders[pname].set(float(val))
        self.status_label.config(text=f"Loaded: {Path(path).name}")

    # ── Target mode (inverse design on population PCE) ────────────

    def _find_target(self) -> None:
        """Find parameters that produce a target Ŷ value.

        Uses L-BFGS-B on the population PCE surrogate — no vEcoli calls.
        """
        if self.surr_data is None or not self.sliders:
            return
        raw = self.target_entry.get().strip()
        if not raw:
            self.status_label.config(text="Enter a target Ŷ value first")
            return
        try:
            target_val = float(raw)
        except ValueError:
            self.status_label.config(text="Target must be a number")
            return

        from scipy.optimize import minimize

        bounds = self.surr_data["bounds"]
        coeffs = self.surr_data["pop_coeffs"]
        mi = self.surr_data["pop_mi"]
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
            try:
                res = minimize(loss, x0=x0, method="L-BFGS-B", bounds=box)
            except Exception:
                continue
            if res.fun < best_loss:
                best_loss = float(res.fun)
                best_x = np.asarray(res.x)

        if best_x is None:
            self.status_label.config(text="Target search failed")
            return

        for _i, _pname in enumerate(self.sliders):
            if _i < len(best_x):
                self.sliders[_pname].set(float(best_x[_i]))

        achieved = legendre_eval(normalize_to_germ(best_x, bounds), coeffs, mi)
        self.status_label.config(
            text=f"Target {target_val:.4f} → Ŷ={achieved:.4f} (err={abs(achieved-target_val):.2e})"
        )

    # ── Spectral inverse design (Koopman bundle) ─────────────────

    def _run_inverse_design(self) -> None:
        """Toolbar-triggered inverse design from the current knob setting.

        Uses the current knob values as the *target* and searches for a new
        parameter vector that hits the same spectrum.  The result is
        written back into the sliders.
        """
        bundle = self.spectral_bundle
        if bundle is None or not self.sliders:
            return
        try:
            x_current = np.array([float(self.sliders[p].get()) for p in bundle.parameter_names])
        except Exception:
            return
        current = predict_spectral_features(bundle, x_current)
        try:
            out = inverse_design_bundle(
                bundle,
                target={"freq_hz": current["freq_hz"], "sigma": current["sigma"]},
                n_restarts=10,
                seed=0,
            )
        except Exception as e:
            self.status_label.config(text=f"Inverse design failed: {e}")
            return
        x_new = out["x"]
        for i, pname in enumerate(bundle.parameter_names):
            if pname in self.sliders:
                self.sliders[pname].set(float(x_new[i]))
        self.status_label.config(text=f"Inverse design: loss={out['loss']:.3e}")


# -- Entry point --------------------------------------------------------------


def run_tk_dashboard_simple(data_path=None):
    """Launch the Tkinter UQ Simple DAW."""
    root = tk.Tk()
    UQDawSimpleApp(root, data_path=data_path)
    root.mainloop()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tk_dashboard_simple(_path)
