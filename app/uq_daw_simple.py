"""
UQ Simple DAW — Tkinter dashboard for the uq_simple pipeline.

Provides DAW-style interactive visualization of all 4 RFC006 strategies:
  1. Uniform (bulk) — PCE response curves with draggable markers
  2. By generation — per-generation Sobol bar comparison
  3. By lineage seed — per-seed Sobol bar comparison
  4. Growth-stratified — sensitivity spectrogram + per-stage prediction

Launch:
    uv run python app/uq_daw_simple.py [path/to/uq_results.json]
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
    "accent_blue": "#2E86C1",
    "accent_gold": "#D4AC0D",
    "grid": "#222244",
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

    def _on_resize(self, event):
        self.redraw()

    def set_curves(self, curves, markers, selected, y_range):
        self._curves = curves
        self._markers = markers
        self._selected = selected
        self._y_min, self._y_max = y_range
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
        self.create_text(12, h // 2, text="\u0176", fill=C["text_dim"], font=("Menlo", 10), angle=90)

        # Y-axis ticks
        for i in range(5):
            frac = i / 4
            y = pt + int(plot_h * (1 - frac))
            val = self._y_min + y_span * frac
            self.create_text(pl - 5, y, text=f"{val:.2f}", fill=C["text_dim"], font=("Menlo", 8), anchor="e")

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
        self.create_text(w // 2, 14, text=f"PCE RESPONSE CURVES{drag_hint}", fill=C["text"], font=("Menlo", 11, "bold"))


class HeatmapCanvas(tk.Canvas):
    """Sensitivity spectrogram: params x growth stages."""

    def __init__(self, parent, param_colors=None, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._param_colors = param_colors or {}
        self._stage_predictions = None

    def set_data(self, stages, params, selected, stage_predictions=None):
        self._data = (stages, params, selected)
        self._stage_predictions = stage_predictions
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

        has_preds = self._stage_predictions is not None and len(self._stage_predictions) > 0
        pred_h = 55 if has_preds else 0
        pad_l, pad_r, pad_t, pad_b = 90, 45, 30, 20 + pred_h
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

        if has_preds:
            preds = np.asarray(self._stage_predictions)
            pred_top = heatmap_bottom + 20
            pred_bottom = h - 5
            pred_plot_h = max(pred_bottom - pred_top, 10)
            p_min, p_max = float(preds.min()), float(preds.max())
            p_span = p_max - p_min or 1.0

            self.create_rectangle(pad_l, pred_top - 2, pad_l + plot_w, pred_bottom + 2, fill=C["panel_light"], outline=C["border"])

            points = []
            for si in range(n_stages):
                px = pad_l + int(cell_w * (si + 0.5))
                py = pred_top + int((1 - (preds[si] - p_min) / p_span) * pred_plot_h)
                points.extend([px, py])
            if len(points) >= 4:
                self.create_line(points, fill=C["accent3"], width=2, smooth=True)
            for si in range(n_stages):
                px = pad_l + int(cell_w * (si + 0.5))
                py = pred_top + int((1 - (preds[si] - p_min) / p_span) * pred_plot_h)
                self.create_oval(px - 3, py - 3, px + 3, py + 3, fill=C["accent3"], outline="")

            self.create_text(pad_l + 3, pred_top - 8, text="\u0176 per stage", fill=C["accent3"], font=("Menlo", 7, "bold"), anchor="w")

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
        self.param_colors = {}

        self._build_ui()

        if data_path:
            self._load_file(data_path)

    def _build_ui(self):
        # -- Top bar --
        top = tk.Frame(self.root, bg=C["panel"], height=40)
        top.pack(fill="x", padx=4, pady=(4, 0))
        top.pack_propagate(False)

        tk.Label(top, text="UQ DAW Simple // RFC006", bg=C["panel"], fg=C["accent1"], font=("Menlo", 13, "bold")).pack(side="left", padx=10)

        tk.Button(
            top, text="Load JSON", command=self._open_file,
            bg=C["panel_light"], fg=C["text"], font=("Menlo", 10), relief="flat", cursor="hand2",
        ).pack(side="right", padx=10)

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

        # -- Left: Strategy info --
        self.strategy_label = tk.Label(self.left_frame, text="Strategies: ---", bg=C["panel"], fg=C["text_dim"], font=("Menlo", 9), wraplength=240, justify="left")
        self.strategy_label.pack(fill="x", padx=8, pady=(2, 4))

        # -- Build visualization panels --
        self._build_viz_panels()

    def _build_viz_panels(self):
        # Top-right: Response curves
        self.response_canvas = ResponseCurveCanvas(
            self.right_top_frame, on_marker_drag=self._on_curve_drag, height=280,
        )
        self.response_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom: strategies 2-3 side by side, then spectrogram full-width
        _strat_row = tk.Frame(self.bottom_frame, bg=C["bg"])
        _strat_row.pack(fill="x", padx=2, pady=(2, 1))

        self.gen_canvas = StrategyBarCanvas(_strat_row, title="STRATEGY 2 // BY GENERATION", bar_color=C["accent_blue"], height=160)
        self.gen_canvas.pack(side="left", fill="both", expand=True, padx=(0, 1))

        self.seed_canvas = StrategyBarCanvas(_strat_row, title="STRATEGY 3 // BY LINEAGE SEED", bar_color=C["accent_gold"], height=160)
        self.seed_canvas.pack(side="right", fill="both", expand=True, padx=(1, 0))

        # Bottom: sensitivity spectrogram (strategy 4)
        self.heatmap_canvas = HeatmapCanvas(self.bottom_frame, height=200)
        self.heatmap_canvas.pack(fill="both", expand=True, padx=2, pady=(1, 2))

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

        self._build_sliders(params)
        self._update_strategy_info()
        self._update_all_viz()
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
            return result
        except Exception:
            return None

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
        self.strategy_label.config(text=f"S1: bulk | S2: {s2_text} | S3: {s3_text} | S4: {n_stages} stages")

    def _update_all_viz(self):
        if self.data is None:
            return
        selected = self.selected_param.get()
        params = list(self.data["parameters"].keys())

        # Strategy 2: by generation
        s2 = self.data.get("strategy2_by_generation", {})
        if "generations" in s2:
            gen_groups = [
                {"label": f"Gen {g['generation']}", "sobol_total_order": g["sobol_total_order"]}
                for g in s2["generations"]
            ]
            self.gen_canvas.set_data(gen_groups)
        else:
            self.gen_canvas.set_data(None)

        # Strategy 3: by seed
        s3 = self.data.get("strategy3_by_seed", {})
        if "seeds" in s3:
            seed_groups = [
                {"label": f"Seed {s['lineage_seed']}", "sobol_total_order": s["sobol_total_order"]}
                for s in s3["seeds"]
            ]
            self.seed_canvas.set_data(seed_groups)
        else:
            self.seed_canvas.set_data(None)

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
        pop_y = legendre_eval(x_norm, self.surr_data["pop_coeffs"], self.surr_data["pop_mi"])
        self.readout_label.config(text=f"\u0176 = {pop_y:.4f}")

        n_sweep = 80
        curves = {}
        markers = {}
        all_y = []
        local_sensitivity = {}

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

            cur_norm = (x[pi] - lo) / (hi - lo + 1e-12)
            markers[pname] = (cur_norm, pop_y)

            delta = (hi - lo) * 0.005
            x_plus = x.copy()
            x_plus[pi] = min(x[pi] + delta, hi)
            x_minus = x.copy()
            x_minus[pi] = max(x[pi] - delta, lo)
            y_plus = legendre_eval(normalize_to_germ(x_plus, bounds), self.surr_data["pop_coeffs"], self.surr_data["pop_mi"])
            y_minus = legendre_eval(normalize_to_germ(x_minus, bounds), self.surr_data["pop_coeffs"], self.surr_data["pop_mi"])
            local_sensitivity[pname] = abs(y_plus - y_minus) / (2 * delta + 1e-12)

        if all_y:
            y_min = min(all_y) - 0.1 * abs(min(all_y))
            y_max = max(all_y) + 0.1 * abs(max(all_y))
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
        else:
            y_min, y_max = -1, 1

        self.response_canvas.set_curves(curves, markers, selected, (y_min, y_max))

        # Per-stage predictions for spectrogram
        stage_data = self.data.get("phase2_growth_stratified", {}).get("stages", [])
        stage_predictions = None
        if stage_data and local_sensitivity:
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

        self.heatmap_canvas.set_data(stage_data, params, selected, stage_predictions=stage_predictions)


# -- Entry point --------------------------------------------------------------


def run_tk_dashboard_simple(data_path=None):
    """Launch the Tkinter UQ Simple DAW."""
    root = tk.Tk()
    UQDawSimpleApp(root, data_path=data_path)
    root.mainloop()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tk_dashboard_simple(_path)
