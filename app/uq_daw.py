"""
UQ Results DAW — Tkinter implementation of the RFC006 pipeline dashboard.

Provides the same visualizations as app/dashboard.py (marimo) but with
native Tk widgets for real-time slider reactivity and DAW-style interaction.

Launch:
    uv run uq dashboard --run-mode tk
    uv run python app/uq_daw.py [path/to/uq_results.json]
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

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
    "grid": "#222244",
}

PARAM_COLORS = {
    "vio_expression": C["accent1"],
    "vio_trl_eff": C["accent4"],
    "mecillinam_concentration": C["accent2"],
}

DEFAULT_PARAM_COLORS = [C["accent1"], C["accent4"], C["accent2"], C["accent3"], C["accent5"]]


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


# -- Canvas widgets -----------------------------------------------------------


class ResponseCurveCanvas(tk.Canvas):
    """DAW-style response curve display with draggable parameter markers.

    Markers can be clicked and dragged horizontally along their curve.
    Dragging a marker updates the corresponding slider value in lock-step.
    """

    PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 30, 40
    MARKER_HIT_RADIUS = 14  # px — how close a click must be to grab a marker

    def __init__(self, parent, on_marker_drag=None, **kwargs):
        """
        Args:
            on_marker_drag: callback(param_name, x_normalized) called during drag.
                x_normalized is in [0, 1] (0 = param min, 1 = param max).
        """
        super().__init__(parent, bg=C["panel"], highlightthickness=0, cursor="crosshair", **kwargs)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_hover)

        self._on_marker_drag = on_marker_drag
        self._curves = {}
        self._markers = {}  # param_name -> (x_norm, y_val)
        self._marker_px = {}  # param_name -> (px, py) — cached pixel positions
        self._selected = None
        self._y_min = 0
        self._y_max = 1
        self._dragging = None  # param_name currently being dragged
        self._hovering = None  # param_name under cursor
        self._param_order = []  # ordered param names for consistent indexing

    def _on_resize(self, event):
        self.redraw()

    def set_curves(self, curves, markers, selected, y_range):
        """
        curves: dict param_name -> (x_norm_array, y_array)
        markers: dict param_name -> (x_norm_pos, y_pos)
        selected: currently soloed param name
        y_range: (y_min, y_max)
        """
        self._curves = curves
        self._markers = markers
        self._selected = selected
        self._y_min, self._y_max = y_range
        self._param_order = list(curves.keys())
        self.redraw()

    def _plot_geometry(self):
        """Return (pad_l, pad_r, pad_t, pad_b, plot_w, plot_h, y_span)."""
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
        """Convert pixel x to normalized parameter value [0, 1], clamped."""
        pl, _, _, _, pw, _, _ = self._plot_geometry()
        if pw <= 0:
            return 0.0
        return max(0.0, min(1.0, (px - pl) / pw))

    def _find_nearest_marker(self, mx, my):
        """Find the marker closest to pixel (mx, my) within hit radius."""
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

        # Axes labels
        self.create_text(
            w // 2, h - 8, text="Normalized parameter value [0=min, 1=max]", fill=C["text_dim"], font=("Menlo", 9)
        )
        self.create_text(12, h // 2, text="Y\u0302", fill=C["text_dim"], font=("Menlo", 10), angle=90)

        # Y-axis ticks
        for i in range(5):
            frac = i / 4
            y = pt + int(plot_h * (1 - frac))
            val = self._y_min + y_span * frac
            self.create_text(pl - 5, y, text=f"{val:.2f}", fill=C["text_dim"], font=("Menlo", 8), anchor="e")

        # Draw curves
        for pname, (xn, yv) in self._curves.items():
            is_sel = pname == self._selected
            is_drag = pname == self._dragging
            color = PARAM_COLORS.get(pname, C["text_dim"])
            width = 3 if (is_sel or is_drag) else 1

            points = []
            for i in range(len(xn)):
                px, py = self._to_px(xn[i], yv[i])
                points.extend([px, py])

            if len(points) >= 4:
                stipple = "" if (is_sel or is_drag) else "gray50"
                self.create_line(points, fill=color, width=width, smooth=True, stipple=stipple)

                if is_sel:
                    fill_points = [pl, pt + plot_h] + points + [pl + plot_w, pt + plot_h]
                    self.create_polygon(fill_points, fill=color, stipple="gray12", outline="")

        # Draw markers (and cache pixel positions for hit-testing)
        self._marker_px.clear()
        for pname, (mx, my) in self._markers.items():
            is_sel = pname == self._selected
            is_drag = pname == self._dragging
            is_hover = pname == self._hovering
            color = PARAM_COLORS.get(pname, C["text_dim"])
            px, py = self._to_px(mx, my)
            self._marker_px[pname] = (px, py)

            # Marker size based on state
            if is_drag:
                size = 10
            elif is_sel or is_hover:
                size = 8
            else:
                size = 5

            outline_color = "#ffffff" if (is_sel or is_drag or is_hover) else ""
            outline_width = 2 if (is_sel or is_drag) else (1 if is_hover else 0)
            self.create_oval(
                px - size, py - size, px + size, py + size, fill=color, outline=outline_color, width=outline_width
            )

            # Value label for selected or dragged marker
            if is_sel or is_drag:
                self.create_text(px, py - size - 8, text=f"Y\u0302={my:.3f}", fill=color, font=("Menlo", 9, "bold"))

            # Show param name near hovered marker
            if is_hover and not is_sel:
                short = (
                    pname.replace("mecillinam_concentration", "mec")
                    .replace("vio_expression", "vio")
                    .replace("vio_trl_eff", "trl")
                )
                self.create_text(px, py - size - 8, text=short, fill=color, font=("Menlo", 8))

        # Title
        drag_hint = "  [drag dots to set values]" if self._markers else ""
        self.create_text(
            w // 2,
            14,
            text=f"PCE RESPONSE CURVES // All Parameters{drag_hint}",
            fill=C["text"],
            font=("Menlo", 11, "bold"),
        )


class SobolEQCanvas(tk.Canvas):
    """Sobol sensitivity view: S_Ti across cell cycle stages with local sensitivity overlay."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._selected = None
        self._local_sens = None  # dict param_name -> slope at current position

    def set_data(self, stages, params, selected, local_sensitivity=None):
        """
        local_sensitivity: dict param_name -> float (|dY/dx_i| at current slider pos).
            If provided, bars are drawn at the bottom showing instantaneous sensitivity
            scaled relative to S_Ti — linking the static Sobol view to the live sliders.
        """
        self._data = (stages, params)
        self._selected = selected
        self._local_sens = local_sensitivity
        self.redraw()

    def redraw(self):
        self.delete("all")
        if self._data is None:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        stages, params = self._data
        pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 55
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        n = len(stages)

        # Grid
        for i in range(5):
            y = pad_t + int(plot_h * i / 4)
            self.create_line(pad_l, y, w - pad_r, y, fill=C["grid"], width=1)

        # X ticks
        for j in range(n):
            x = pad_l + int(plot_w * (j + 0.5) / n)
            self.create_text(x, pad_t + plot_h + 12, text=f"\u03b8{j}", fill=C["text_dim"], font=("Menlo", 8))

        # Draw Sobol curves per param
        for pname in params:
            is_sel = pname == self._selected
            color = PARAM_COLORS.get(pname, C["text_dim"])
            values = [s["total_order"].get(pname, 0) for s in stages]

            points = []
            for j, v in enumerate(values):
                x = pad_l + int(plot_w * (j + 0.5) / n)
                y = pad_t + int(plot_h * (1 - v / 0.7))
                points.extend([x, y])

            if len(points) >= 4:
                stipple = "" if is_sel else "gray50"
                self.create_line(points, fill=color, width=3 if is_sel else 1, smooth=True, stipple=stipple)

        # Y-axis ticks
        for i in range(5):
            frac = i / 4
            y = pad_t + int(plot_h * (1 - frac))
            self.create_text(pad_l - 5, y, text=f"{0.7 * frac:.2f}", fill=C["text_dim"], font=("Menlo", 8), anchor="e")

        # Local sensitivity bars at bottom (live from sliders)
        if self._local_sens:
            bar_y = h - 18
            bar_h = 10
            max_sens = max(abs(v) for v in self._local_sens.values()) or 1
            total_bar_w = plot_w * 0.8
            bar_x_start = pad_l + plot_w * 0.1
            n_params = len(self._local_sens)
            bar_w = total_bar_w / n_params - 4

            self.create_text(
                pad_l, bar_y, text="LOCAL |dY\u0302/dx|:", fill=C["text_dim"], font=("Menlo", 7), anchor="w"
            )

            for pi, (pname, sens_val) in enumerate(self._local_sens.items()):
                is_sel = pname == self._selected
                color = PARAM_COLORS.get(pname, C["text_dim"])
                bx = bar_x_start + pi * (bar_w + 4)
                fill_w = max(2, int(bar_w * abs(sens_val) / max_sens))
                self.create_rectangle(
                    bx,
                    bar_y - bar_h // 2,
                    bx + fill_w,
                    bar_y + bar_h // 2,
                    fill=color,
                    outline="" if not is_sel else "#fff",
                )
                self.create_text(
                    bx + fill_w + 3, bar_y, text=f"{abs(sens_val):.3f}", fill=color, font=("Menlo", 7), anchor="w"
                )

        self.create_text(
            w // 2, 14, text=f"SOBOL S_Ti // {self._selected} soloed", fill=C["text"], font=("Menlo", 11, "bold")
        )


class HeatmapCanvas(tk.Canvas):
    """Sensitivity spectrogram: params x stages heatmap with per-stage prediction overlay."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._param_positions = {}  # param_name -> normalized position [0,1]
        self._stage_predictions = None  # array of n_stages Y-hat values

    def set_data(self, stages, params, selected, param_positions=None, stage_predictions=None):
        """
        param_positions: dict param_name -> float in [0,1] (current slider
            position normalized). Draws a colored dot on each param's row.
        stage_predictions: array-like of n_stages floats — per-stage Y-hat
            from the cell cycle surrogate at current slider values. Drawn
            as a reactive line chart below the heatmap.
        """
        self._data = (stages, params, selected)
        self._param_positions = param_positions or {}
        self._stage_predictions = stage_predictions
        self.redraw()

    def _val_to_color(self, v):
        """Map 0..0.7 to dark-blue-cyan-amber-pink."""
        v = max(0, min(v / 0.7, 1.0))
        if v < 0.33:
            t = v / 0.33
            r, g, b = int(13 + t * 29), int(13 + t * 167), int(46 + t * 176)
        elif v < 0.66:
            t = (v - 0.33) / 0.33
            r, g, b = int(42 + t * (255 - 42)), int(180 + t * (170 - 180)), int(222 - t * 222)
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

        has_preds = self._stage_predictions is not None and len(self._stage_predictions) > 0

        # Reserve bottom space for the prediction curve
        pred_h = 60 if has_preds else 0
        pad_l, pad_r, pad_t, pad_b = 90, 20, 30, 20 + pred_h
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        n_stages = len(stages)
        n_params = len(params)
        if n_stages == 0 or n_params == 0:
            return
        cell_w = plot_w / n_stages
        cell_h = plot_h / n_params

        # Draw heatmap cells
        for pi, pname in enumerate(params):
            short = (
                pname.replace("mecillinam_concentration", "mecillinam")
                .replace("vio_expression", "vio_exp")
                .replace("vio_trl_eff", "vio_trl")
            )
            is_sel = pname == selected
            color = PARAM_COLORS.get(pname, C["text_dim"]) if is_sel else C["text_dim"]
            self.create_text(
                pad_l - 5,
                pad_t + int(cell_h * (pi + 0.5)),
                text=short,
                fill=color,
                font=("Menlo", 9, "bold" if is_sel else ""),
                anchor="e",
            )

            for si, stage in enumerate(stages):
                v = stage["total_order"].get(pname, 0)
                x0 = pad_l + int(si * cell_w)
                y0 = pad_t + int(pi * cell_h)
                x1 = x0 + int(cell_w)
                y1 = y0 + int(cell_h)
                self.create_rectangle(x0, y0, x1, y1, fill=self._val_to_color(v), outline=C["border"])

        # Stage labels between heatmap and prediction curve
        heatmap_bottom = pad_t + plot_h
        for si in range(n_stages):
            x = pad_l + int(cell_w * (si + 0.5))
            self.create_text(x, heatmap_bottom + 10, text=f"\u03b8{si}", fill=C["text_dim"], font=("Menlo", 7))

        # Draw slider position cursors on each param row
        for pname, pos in self._param_positions.items():
            if pname not in params:
                continue
            pi = params.index(pname)
            is_sel = pname == selected
            color = PARAM_COLORS.get(pname, C["text_dim"])
            stage_x = pad_l + int(pos * plot_w)
            cy = pad_t + int(cell_h * (pi + 0.5))
            r = 5 if is_sel else 3
            self.create_oval(stage_x - r, cy - r, stage_x + r, cy + r, fill="#ffffff", outline=color, width=2)

        # -- Per-stage prediction curve (cell cycle surrogate output) --
        if has_preds:
            preds = np.asarray(self._stage_predictions)
            pred_top = heatmap_bottom + 20
            pred_bottom = h - 5
            pred_plot_h = pred_bottom - pred_top
            if pred_plot_h < 10:
                pred_plot_h = 10

            p_min = float(preds.min())
            p_max = float(preds.max())
            p_span = p_max - p_min
            if p_span == 0:
                p_span = 1.0

            # Background for prediction area
            self.create_rectangle(
                pad_l,
                pred_top - 2,
                pad_l + plot_w,
                pred_bottom + 2,
                fill=C["panel_light"] if "panel_light" in C else "#252545",
                outline=C["border"],
            )

            # Grid lines
            for i in range(3):
                gy = pred_top + int(pred_plot_h * i / 2)
                self.create_line(pad_l, gy, pad_l + plot_w, gy, fill=C["grid"], width=1, dash=(2, 4))

            # Y-axis ticks for prediction
            for i in range(3):
                frac = i / 2
                gy = pred_top + int(pred_plot_h * (1 - frac))
                val = p_min + p_span * frac
                self.create_text(pad_l - 5, gy, text=f"{val:.2f}", fill=C["accent3"], font=("Menlo", 7), anchor="e")

            # Draw prediction line
            points = []
            for si in range(n_stages):
                px = pad_l + int(cell_w * (si + 0.5))
                py = pred_top + int((1 - (preds[si] - p_min) / p_span) * pred_plot_h)
                points.extend([px, py])

            if len(points) >= 4:
                self.create_line(points, fill=C["accent3"], width=2, smooth=True)

            # Draw dots at each stage
            for si in range(n_stages):
                px = pad_l + int(cell_w * (si + 0.5))
                py = pred_top + int((1 - (preds[si] - p_min) / p_span) * pred_plot_h)
                self.create_oval(px - 3, py - 3, px + 3, py + 3, fill=C["accent3"], outline="")

            # Find peak stage
            peak_stage = int(np.argmax(preds))
            peak_px = pad_l + int(cell_w * (peak_stage + 0.5))
            peak_py = pred_top + int((1 - (preds[peak_stage] - p_min) / p_span) * pred_plot_h)
            self.create_oval(
                peak_px - 5, peak_py - 5, peak_px + 5, peak_py + 5, fill=C["accent3"], outline="#ffffff", width=2
            )
            self.create_text(
                peak_px,
                peak_py - 10,
                text=f"\u03b8{peak_stage}={preds[peak_stage]:.3f}",
                fill=C["accent3"],
                font=("Menlo", 8, "bold"),
            )

            self.create_text(
                pad_l + 3,
                pred_top - 8,
                text="Y\u0302 per stage (cell cycle surrogate)",
                fill=C["accent3"],
                font=("Menlo", 7, "bold"),
                anchor="w",
            )

        self.create_text(
            w // 2, 14, text="SENSITIVITY SPECTROGRAM + STAGE PREDICTION", fill=C["text"], font=("Menlo", 10, "bold")
        )


class ObservableStageCanvas(tk.Canvas):
    """Observable-domain heatmap: observables x cell cycle stages in physical units.

    Shows per-stage observable values modulated by current slider positions.
    All data comes from pipeline outputs:
    - Baselines from cell_cycle_profile (Step 6c θ-binned aggregation)
    - Modulation weights from per-stage S_Ti (Step 7b Sobol)
    - Modulation magnitude from local |dY/dx_i| (PCE finite difference)
    """

    # Colorscale: dark → blue → green → yellow (physical values, not variance fractions)
    _CMAP = [
        (0.0, (13, 13, 46)),
        (0.25, (0, 100, 180)),
        (0.5, (0, 180, 140)),
        (0.75, (180, 220, 50)),
        (1.0, (255, 255, 100)),
    ]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._obs_names = []
        self._n_stages = 0
        self._values = None  # (n_obs, n_stages) array in physical units
        self._baselines = None  # (n_obs, n_stages) array — unmodulated baselines

    def set_data(self, obs_names, n_stages, values, baselines=None):
        """
        obs_names: list of observable names
        n_stages: number of cell cycle bins
        values: (n_obs, n_stages) array — current predicted values in physical units
        baselines: (n_obs, n_stages) array — unmodulated baseline values (for delta display)
        """
        self._obs_names = obs_names
        self._n_stages = n_stages
        self._values = np.asarray(values) if values is not None else None
        self._baselines = np.asarray(baselines) if baselines is not None else None
        self.redraw()

    def _val_to_color(self, v, v_min, v_max):
        """Map a physical value to a color via the colorscale."""
        if v_max == v_min:
            t = 0.5
        else:
            t = max(0.0, min(1.0, (v - v_min) / (v_max - v_min)))

        # Interpolate in colorscale
        for i in range(len(self._CMAP) - 1):
            t0, c0 = self._CMAP[i]
            t1, c1 = self._CMAP[i + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{r:02x}{g:02x}{b:02x}"
        return C["text_dim"]

    def redraw(self):
        self.delete("all")
        if self._values is None or len(self._obs_names) == 0:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        n_obs = len(self._obs_names)
        n_stages = self._n_stages
        if n_stages == 0:
            return

        pad_l, pad_r, pad_t, pad_b = 80, 40, 30, 25
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        cell_w = plot_w / n_stages
        cell_h = plot_h / n_obs

        # Global min/max for color mapping (across all obs and stages)
        v_min = float(self._values.min())
        v_max = float(self._values.max())

        # Draw cells
        for oi in range(n_obs):
            short = self._obs_names[oi].split("__")[-1] if "__" in self._obs_names[oi] else self._obs_names[oi]
            self.create_text(pad_l - 5, pad_t + int(cell_h * (oi + 0.5)),
                             text=short, fill=C["text"], font=("Menlo", 9, "bold"), anchor="e")

            for si in range(n_stages):
                v = self._values[oi, si]
                x0 = pad_l + int(si * cell_w)
                y0 = pad_t + int(oi * cell_h)
                x1 = x0 + int(cell_w)
                y1 = y0 + int(cell_h)
                color = self._val_to_color(v, v_min, v_max)
                self.create_rectangle(x0, y0, x1, y1, fill=color, outline=C["border"])

                # Show value in cell if cells are large enough
                if cell_w > 35 and cell_h > 16:
                    text_color = "#000000" if v > (v_min + v_max) / 2 else C["text"]
                    self.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                     text=f"{v:.3f}", fill=text_color, font=("Menlo", 7))

                # Show delta from baseline as a small indicator
                if self._baselines is not None:
                    delta = v - self._baselines[oi, si]
                    if abs(delta) > 1e-6:
                        sign = "+" if delta > 0 else ""
                        delta_color = C["accent3"] if delta > 0 else C["accent2"]
                        self.create_text(x1 - 3, y0 + 3, text=f"{sign}{delta:.2f}",
                                         fill=delta_color, font=("Menlo", 6), anchor="ne")

        # Stage labels
        for si in range(n_stages):
            x = pad_l + int(cell_w * (si + 0.5))
            self.create_text(x, h - 8, text=f"\u03b8{si}", fill=C["text_dim"], font=("Menlo", 7))

        # Colorbar on right
        cb_x = w - pad_r + 8
        cb_w = 12
        cb_top = pad_t
        cb_h = plot_h
        n_cb = 50
        for i in range(n_cb):
            frac = i / n_cb
            val = v_max - frac * (v_max - v_min)
            color = self._val_to_color(val, v_min, v_max)
            cy0 = cb_top + int(frac * cb_h)
            cy1 = cb_top + int((frac + 1.0 / n_cb) * cb_h)
            self.create_rectangle(cb_x, cy0, cb_x + cb_w, cy1, fill=color, outline="")

        self.create_text(cb_x + cb_w + 3, cb_top, text=f"{v_max:.2f}",
                         fill=C["text_dim"], font=("Menlo", 7), anchor="nw")
        self.create_text(cb_x + cb_w + 3, cb_top + cb_h, text=f"{v_min:.2f}",
                         fill=C["text_dim"], font=("Menlo", 7), anchor="sw")

        self.create_text(w // 2, 14, text="OBSERVABLE DOMAIN // Y per stage (physical units)",
                         fill=C["text"], font=("Menlo", 10, "bold"))


class VarianceDecompCanvas(tk.Canvas):
    """Donut chart for variance decomposition."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._fracs = None

    def set_data(self, gen_f, seed_f, resid_f):
        self._fracs = (gen_f, seed_f, resid_f)
        self.redraw()

    def redraw(self):
        self.delete("all")
        if self._fracs is None:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        cx, cy = w // 2, h // 2 + 10
        r = min(w, h) // 2 - 30
        if r < 10:
            return

        gen_f, seed_f, resid_f = self._fracs
        total = gen_f + seed_f + resid_f
        if total == 0:
            total = 1

        slices = [
            ("Gen", gen_f / total, C["accent4"]),
            ("Seed", seed_f / total, C["accent5"]),
            ("Residual", resid_f / total, C["accent3"]),
        ]

        start = 90
        for label, frac, color in slices:
            extent = -frac * 360
            self.create_arc(cx - r, cy - r, cx + r, cy + r, start=start, extent=extent, fill=color, outline=C["border"])
            # Label
            mid_angle = np.radians(start + extent / 2)
            lx = cx + int((r * 0.7) * np.cos(mid_angle))
            ly = cy - int((r * 0.7) * np.sin(mid_angle))
            if frac > 0.01:
                self.create_text(lx, ly, text=f"{label}\n{frac * 100:.1f}%", fill="#000", font=("Menlo", 8, "bold"))
            start += extent

        # Donut hole
        ir = r // 2
        self.create_oval(cx - ir, cy - ir, cx + ir, cy + ir, fill=C["panel"], outline=C["panel"])

        self.create_text(w // 2, 12, text="VARIANCE DECOMP", fill=C["text"], font=("Menlo", 10, "bold"))


# -- Main Application ---------------------------------------------------------


class UQDawApp:
    """Tkinter UQ Results DAW."""

    def __init__(self, root, data_path=None):
        self.root = root
        self.root.title("UQ DAW // RFC006")
        self.root.geometry("1400x900")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1000, 700)

        self.data = None
        self.surr_data = None
        self.selected_param = tk.StringVar()

        self._build_ui()

        if data_path:
            self._load_file(data_path)

    def _build_ui(self):
        # -- Top bar --
        top = tk.Frame(self.root, bg=C["panel"], height=40)
        top.pack(fill="x", padx=4, pady=(4, 0))
        top.pack_propagate(False)

        tk.Label(top, text="UQ DAW // RFC006", bg=C["panel"], fg=C["accent1"], font=("Menlo", 13, "bold")).pack(
            side="left", padx=10
        )

        tk.Button(
            top,
            text="Load JSON",
            command=self._open_file,
            bg=C["panel_light"],
            fg=C["text"],
            font=("Menlo", 10),
            relief="flat",
            cursor="hand2",
        ).pack(side="right", padx=10)

        self.status_label = tk.Label(top, text="No data loaded", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 9))
        self.status_label.pack(side="right", padx=10)

        self.header_label = tk.Label(top, text="", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 9))
        self.header_label.pack(side="right", padx=10)

        # -- Main area (left controls + right visualizations) --
        main = tk.PanedWindow(self.root, orient="horizontal", bg=C["bg"], sashwidth=4, sashrelief="flat")
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # Left panel: parameter selector + sliders
        self.left_frame = tk.Frame(main, bg=C["bg"], width=280)
        main.add(self.left_frame, minsize=250)

        # Right panel: visualizations
        self.right_frame = tk.Frame(main, bg=C["bg"])
        main.add(self.right_frame, minsize=600)

        # -- Left: Solo selector --
        solo_frame = tk.LabelFrame(
            self.left_frame,
            text="SOLO PARAMETER",
            bg=C["panel"],
            fg=C["accent1"],
            font=("Menlo", 10, "bold"),
            labelanchor="n",
        )
        solo_frame.pack(fill="x", padx=4, pady=4)

        self.param_menu = ttk.Combobox(
            solo_frame, textvariable=self.selected_param, state="readonly", font=("Menlo", 10)
        )
        self.param_menu.pack(fill="x", padx=8, pady=8)
        self.selected_param.trace_add("write", lambda *_: self._on_param_change())

        # -- Left: PCE sliders --
        slider_frame = tk.LabelFrame(
            self.left_frame,
            text="PCE SURROGATE KNOBS",
            bg=C["panel"],
            fg=C["accent3"],
            font=("Menlo", 10, "bold"),
            labelanchor="n",
        )
        slider_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.slider_container = tk.Frame(slider_frame, bg=C["panel"])
        self.slider_container.pack(fill="both", expand=True, padx=4, pady=4)

        self.sliders = {}
        self.slider_labels = {}

        # -- Left: Prediction readout --
        self.readout_label = tk.Label(
            self.left_frame, text="Y\u0302 = ---", bg=C["panel"], fg=C["accent3"], font=("Menlo", 16, "bold")
        )
        self.readout_label.pack(fill="x", padx=4, pady=4)

        # -- Right: visualization grid --
        self._build_viz_panels()

    def _build_viz_panels(self):
        # Top: Response curves (the EQ — dominant panel)
        self.response_canvas = ResponseCurveCanvas(
            self.right_frame,
            height=260,
            on_marker_drag=self._on_curve_drag,
        )
        self.response_canvas.pack(fill="both", expand=True, padx=2, pady=(2, 1))

        # Middle row: Observable domain heatmap (physical units, reactive)
        self.obs_canvas = ObservableStageCanvas(self.right_frame, height=160)
        self.obs_canvas.pack(fill="both", expand=False, padx=2, pady=1)

        # Bottom row: Sobol + Sensitivity Spectrogram + Variance (linked contextual views)
        bottom_row = tk.Frame(self.right_frame, bg=C["bg"])
        bottom_row.pack(fill="both", expand=True, padx=2, pady=(1, 2))

        self.eq_canvas = SobolEQCanvas(bottom_row, height=170)
        self.eq_canvas.pack(side="left", fill="both", expand=True, padx=(0, 1))

        self.heatmap_canvas = HeatmapCanvas(bottom_row, height=170)
        self.heatmap_canvas.pack(side="left", fill="both", expand=True, padx=1)

        self.decomp_canvas = VarianceDecompCanvas(bottom_row, height=170)
        self.decomp_canvas.pack(side="right", fill="both", expand=False, padx=(1, 0))

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Load UQ Results",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        path = Path(path)
        if not path.exists():
            self.status_label.config(text=f"Not found: {path.name}")
            return

        with open(path) as f:
            self.data = json.load(f)

        # Load surrogates from sibling directory
        export_dir = path.parent
        self.surr_data = self._load_surrogates(export_dir)

        # Populate UI
        params = self.data["parameter_names"]
        self.param_menu["values"] = params
        self.selected_param.set(params[0])

        self._build_sliders()
        self._update_header()
        self._update_all_viz()

        self.status_label.config(text=f"Loaded: {path.name}")

    def _load_surrogates(self, export_dir):
        pop_dir = export_dir / "population_surrogate"
        cc_dir = export_dir / "cell_cycle_surrogate"

        if not pop_dir.exists() or not cc_dir.exists():
            return None

        try:
            result = {
                "pop_coeffs": np.load(pop_dir / "coefficients.npy"),
                "pop_mi": np.load(pop_dir / "multi_indices.npy"),
                "cc_coeffs": np.load(cc_dir / "coefficients.npy"),
                "cc_mi": np.load(cc_dir / "multi_indices.npy"),
            }
            bounds_path = pop_dir / "input_bounds.npy"
            result["bounds"] = np.load(bounds_path) if bounds_path.exists() else None
            return result
        except Exception:
            return None

    def _build_sliders(self):
        # Clear old sliders
        for w in self.slider_container.winfo_children():
            w.destroy()
        self.sliders.clear()
        self.slider_labels.clear()

        if self.data is None or self.surr_data is None:
            return

        params = self.data["parameter_names"]
        bounds = self.surr_data.get("bounds")
        if bounds is None:
            return

        for i, pname in enumerate(params):
            color = PARAM_COLORS.get(pname, DEFAULT_PARAM_COLORS[i % len(DEFAULT_PARAM_COLORS)])
            lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
            mid = (lo + hi) / 2

            frame = tk.Frame(self.slider_container, bg=C["panel"])
            frame.pack(fill="x", pady=3)

            short = (
                pname.replace("mecillinam_concentration", "mecillinam")
                .replace("vio_expression", "vio_exp")
                .replace("vio_trl_eff", "vio_trl")
            )
            tk.Label(frame, text=short, bg=C["panel"], fg=color, font=("Menlo", 9, "bold"), width=10, anchor="w").pack(
                side="left", padx=4
            )

            slider = tk.Scale(
                frame,
                from_=lo,
                to=hi,
                orient="horizontal",
                resolution=(hi - lo) / 200,
                length=140,
                bg=C["panel"],
                fg=color,
                troughcolor=C["panel_light"],
                highlightthickness=0,
                font=("Menlo", 8),
                showvalue=False,
                command=lambda v, p=pname: self._on_slider_change(p, v),
            )
            slider.set(mid)
            slider.pack(side="left", padx=2)

            val_label = tk.Label(
                frame, text=f"{mid:.2f}", bg=C["panel"], fg=color, font=("Menlo", 9), width=7, anchor="e"
            )
            val_label.pack(side="left", padx=4)

            self.sliders[pname] = slider
            self.slider_labels[pname] = val_label

    def _on_slider_change(self, pname, value):
        value = float(value)
        if pname in self.slider_labels:
            self.slider_labels[pname].config(text=f"{value:.2f}")
        self._update_response_curves()

    def _on_curve_drag(self, pname, x_normalized):
        """Called when a marker dot is dragged on the response curve canvas.

        Converts normalized [0,1] position back to physical parameter value
        and sets the corresponding slider — which triggers _on_slider_change
        and redraws everything in lock-step.
        """
        if self.surr_data is None:
            return
        bounds = self.surr_data.get("bounds")
        if bounds is None:
            return

        params = self.data["parameter_names"]
        if pname not in params:
            return
        pi = params.index(pname)

        lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
        physical_val = lo + x_normalized * (hi - lo)
        physical_val = max(lo, min(hi, physical_val))

        # Set the slider — this triggers _on_slider_change automatically
        if pname in self.sliders:
            self.sliders[pname].set(physical_val)

    def _on_param_change(self):
        self._update_all_viz()

    def _update_header(self):
        if self.data is None:
            return
        surr = self.data.get("surrogates", {})
        pop_r2 = surr.get("population", {}).get("r_squared", "?")
        cc_r2 = surr.get("cell_cycle", {}).get("r_squared", "?")
        n_p = self.data.get("n_parameters", "?")
        n_s = self.data.get("n_cell_cycle_stages", "?")
        self.header_label.config(text=f"PARAMS:{n_p}  STAGES:{n_s}  POP R\u00b2:{pop_r2}  CC R\u00b2:{cc_r2}")

    def _update_all_viz(self):
        if self.data is None:
            return
        selected = self.selected_param.get()
        stages = self.data.get("phase2_cell_cycle_sobol_per_stage", [])
        params = self.data["parameter_names"]

        # EQ
        self.eq_canvas.set_data(stages, params, selected)

        # Heatmap
        self.heatmap_canvas.set_data(stages, params, selected)

        # Variance decomp
        decomp = self.data.get("variance_decomposition", {})
        gen_f = decomp.get("generation_fraction", [0])[0]
        seed_f = decomp.get("seed_fraction", [0])[0]
        resid_f = decomp.get("residual_fraction", [1])[0]
        self.decomp_canvas.set_data(gen_f, seed_f, resid_f)

        # Response curves
        self._update_response_curves()

    def _update_response_curves(self):
        if self.data is None or self.surr_data is None:
            return

        params = self.data["parameter_names"]
        bounds = self.surr_data.get("bounds")
        if bounds is None:
            return

        selected = self.selected_param.get()

        # Current slider values
        x = np.array([float(self.sliders[p].get()) for p in params])
        x_norm = normalize_to_germ(x, bounds)

        # Current prediction
        pop_y = legendre_eval(x_norm, self.surr_data["pop_coeffs"], self.surr_data["pop_mi"])
        self.readout_label.config(text=f"Y\u0302 = {pop_y:.4f}")

        # Sweep each parameter and compute local sensitivity (slope at current pos)
        n_sweep = 80
        curves = {}
        markers = {}
        all_y = []
        local_sensitivity = {}
        param_positions = {}

        for pi, pname in enumerate(params):
            lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
            sv = np.linspace(lo, hi, n_sweep)
            sv_norm = (sv - lo) / (hi - lo + 1e-12)
            sy = []
            for v in sv:
                x_sw = x.copy()
                x_sw[pi] = v
                x_sw_n = normalize_to_germ(x_sw, bounds)
                sy.append(legendre_eval(x_sw_n, self.surr_data["pop_coeffs"], self.surr_data["pop_mi"]))
            sy = np.array(sy)
            curves[pname] = (sv_norm, sy)
            all_y.extend(sy.tolist())

            # Marker at current slider position
            cur_norm = (x[pi] - lo) / (hi - lo + 1e-12)
            markers[pname] = (cur_norm, pop_y)
            param_positions[pname] = cur_norm

            # Local sensitivity: |dY/dx_i| at current position via finite difference
            delta = (hi - lo) * 0.005
            x_plus = x.copy()
            x_plus[pi] = min(x[pi] + delta, hi)
            x_minus = x.copy()
            x_minus[pi] = max(x[pi] - delta, lo)
            y_plus = legendre_eval(
                normalize_to_germ(x_plus, bounds), self.surr_data["pop_coeffs"], self.surr_data["pop_mi"]
            )
            y_minus = legendre_eval(
                normalize_to_germ(x_minus, bounds), self.surr_data["pop_coeffs"], self.surr_data["pop_mi"]
            )
            local_sensitivity[pname] = abs(y_plus - y_minus) / (2 * delta + 1e-12)

        if all_y:
            y_min = min(all_y) - 0.1 * abs(min(all_y))
            y_max = max(all_y) + 0.1 * abs(max(all_y))
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
        else:
            y_min, y_max = -1, 1

        # Update all linked panels
        self.response_canvas.set_curves(curves, markers, selected, (y_min, y_max))

        # Per-stage predicted output using pipeline data:
        # Combine the population surrogate prediction with per-stage Sobol
        # indices to estimate how each parameter's contribution distributes
        # across the cell cycle. For each stage k:
        #   Y_hat_k = baseline + sum_i (x_i_effect * S_Ti^(k))
        # where x_i_effect = local sensitivity * (x_i - midpoint).
        # This uses only pipeline outputs (S_Ti per stage + PCE prediction).
        stage_predictions = None
        param_effects = {}
        stage_data = self.data.get("phase2_cell_cycle_sobol_per_stage", [])
        if stage_data and local_sensitivity:
            n_stages = len(stage_data)
            stage_predictions = np.zeros(n_stages)

            # Per-param effect at current slider position (deviation from midpoint)
            param_effects = {}
            for pi, pname in enumerate(params):
                lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
                mid_norm = 0.0  # midpoint in germ space
                deviation = x_norm[pi] - mid_norm
                param_effects[pname] = local_sensitivity[pname] * deviation

            # Distribute across stages weighted by per-stage S_Ti
            baseline = pop_y
            for si in range(n_stages):
                stage_contrib = 0.0
                for pname in params:
                    s_ti = stage_data[si]["total_order"].get(pname, 0)
                    stage_contrib += param_effects[pname] * s_ti
                stage_predictions[si] = baseline + stage_contrib

        # Push local sensitivity to Sobol panel (live link)
        stages = self.data.get("phase2_cell_cycle_sobol_per_stage", [])
        self.eq_canvas.set_data(stages, params, selected, local_sensitivity=local_sensitivity)

        # Push slider positions + stage predictions to Heatmap
        self.heatmap_canvas.set_data(
            stages, params, selected, param_positions=param_positions, stage_predictions=stage_predictions
        )

        # Observable-domain heatmap: per-stage values in physical units
        profile = self.data.get("cell_cycle_profile")
        if profile and stage_data and local_sensitivity and param_effects:
            n_stages_p = len(profile.get("stages", []))
            # Collect observable names and baselines from profile
            obs_names = []
            baselines_list = []
            for key in profile:
                if key == "stages":
                    continue
                obs_names.append(key)
                vals = profile[key]
                baselines_list.append(np.array(vals[:n_stages_p], dtype=float))

            if obs_names and baselines_list:
                baselines = np.array(baselines_list)  # (n_obs, n_stages)
                # Modulate baselines by slider-driven parameter effects
                # Y_obs_k(x) = baseline_obs_k * (1 + sum_i [sens_i * dev_i * S_Ti^(k)])
                modulated = baselines.copy()
                for si in range(min(n_stages_p, len(stage_data))):
                    modulation = 0.0
                    for pname in params:
                        s_ti = stage_data[si]["total_order"].get(pname, 0)
                        modulation += param_effects.get(pname, 0) * s_ti
                    # Scale modulation relative to baseline magnitude
                    for oi in range(len(obs_names)):
                        base_mag = abs(baselines[oi, si]) + 1e-12
                        modulated[oi, si] = baselines[oi, si] * (1 + modulation / base_mag)

                self.obs_canvas.set_data(obs_names, n_stages_p, modulated, baselines=baselines)


# -- Entry point --------------------------------------------------------------


def run_tk_dashboard(data_path=None):
    """Launch the Tkinter UQ DAW."""
    root = tk.Tk()
    UQDawApp(root, data_path=data_path)
    root.mainloop()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tk_dashboard(_path)
