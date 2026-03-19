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
    """DAW-style response curve display with interactive parameter markers."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", self._on_resize)
        self._curves = {}
        self._markers = {}
        self._selected = None
        self._y_min = 0
        self._y_max = 1

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
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 10 or h < 10:
            return

        pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 40
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b

        # Grid
        for i in range(5):
            y = pad_t + int(plot_h * i / 4)
            self.create_line(pad_l, y, w - pad_r, y, fill=C["grid"], width=1)
        for i in range(11):
            x = pad_l + int(plot_w * i / 10)
            self.create_line(x, pad_t, x, pad_t + plot_h, fill=C["grid"], width=1)

        # Axes labels
        self.create_text(w // 2, h - 8, text="Normalized parameter value [0=min, 1=max]",
                         fill=C["text_dim"], font=("Menlo", 9))
        self.create_text(12, h // 2, text="Y\u0302", fill=C["text_dim"], font=("Menlo", 10), angle=90)

        # Y-axis ticks
        y_span = self._y_max - self._y_min
        if y_span == 0:
            y_span = 1
        for i in range(5):
            frac = i / 4
            y = pad_t + int(plot_h * (1 - frac))
            val = self._y_min + y_span * frac
            self.create_text(pad_l - 5, y, text=f"{val:.2f}", fill=C["text_dim"],
                             font=("Menlo", 8), anchor="e")

        def to_px(xn, yv):
            px = pad_l + int(xn * plot_w)
            py = pad_t + int((1 - (yv - self._y_min) / y_span) * plot_h)
            return px, py

        # Draw curves
        for pname, (xn, yv) in self._curves.items():
            is_sel = pname == self._selected
            color = PARAM_COLORS.get(pname, C["text_dim"])
            width = 3 if is_sel else 1

            # Build point list
            points = []
            for i in range(len(xn)):
                px, py = to_px(xn[i], yv[i])
                points.extend([px, py])

            if len(points) >= 4:
                stipple = "" if is_sel else "gray50"
                self.create_line(points, fill=color, width=width, smooth=True, stipple=stipple)

                # Fill under selected curve
                if is_sel:
                    fill_points = [pad_l, pad_t + plot_h] + points + [pad_l + plot_w, pad_t + plot_h]
                    self.create_polygon(fill_points, fill=color, stipple="gray12", outline="")

        # Draw markers
        for pname, (mx, my) in self._markers.items():
            is_sel = pname == self._selected
            color = PARAM_COLORS.get(pname, C["text_dim"])
            px, py = to_px(mx, my)
            size = 8 if is_sel else 4
            outline = "#ffffff" if is_sel else ""
            self.create_oval(px - size, py - size, px + size, py + size,
                             fill=color, outline=outline, width=2 if is_sel else 0)
            if is_sel:
                self.create_text(px, py - 14, text=f"Y\u0302={my:.3f}",
                                 fill=color, font=("Menlo", 9, "bold"))

        # Title
        self.create_text(w // 2, 14, text="PCE RESPONSE CURVES // All Parameters",
                         fill=C["text"], font=("Menlo", 11, "bold"))


class SobolEQCanvas(tk.Canvas):
    """Parametric EQ view: Sobol S_Ti across cell cycle stages."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None
        self._selected = None

    def set_data(self, stages, params, selected):
        self._data = (stages, params)
        self._selected = selected
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
        pad_l, pad_r, pad_t, pad_b = 50, 20, 30, 40
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        n = len(stages)

        # Grid
        for i in range(5):
            y = pad_t + int(plot_h * i / 4)
            self.create_line(pad_l, y, w - pad_r, y, fill=C["grid"], width=1)

        # X ticks (stage centers)
        for j in range(n):
            x = pad_l + int(plot_w * (j + 0.5) / n)
            self.create_text(x, h - 12, text=f"\u03b8{j}", fill=C["text_dim"], font=("Menlo", 8))

        # Draw curves per param
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
                self.create_line(points, fill=color, width=3 if is_sel else 1,
                                 smooth=True, stipple=stipple)

        # Y-axis
        for i in range(5):
            frac = i / 4
            y = pad_t + int(plot_h * (1 - frac))
            self.create_text(pad_l - 5, y, text=f"{0.7 * frac:.2f}", fill=C["text_dim"],
                             font=("Menlo", 8), anchor="e")

        self.create_text(w // 2, 14, text=f"PARAMETRIC EQ // S_Ti \u2014 solo: {self._selected}",
                         fill=C["text"], font=("Menlo", 11, "bold"))
        self.create_text(w // 2, h - 25, text="\u03b8 (cell cycle position)",
                         fill=C["text_dim"], font=("Menlo", 9))


class HeatmapCanvas(tk.Canvas):
    """Sensitivity spectrogram: params x stages heatmap."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["panel"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.redraw())
        self._data = None

    def set_data(self, stages, params, selected):
        self._data = (stages, params, selected)
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
            r, g, b = int(255), int(170 - t * 119), int(t * 102)
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

        pad_l, pad_r, pad_t, pad_b = 90, 20, 30, 30
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        n_stages = len(stages)
        n_params = len(params)
        cell_w = plot_w / n_stages
        cell_h = plot_h / n_params

        for pi, pname in enumerate(params):
            short = pname.replace("mecillinam_concentration", "mecillinam").replace("vio_expression", "vio_exp").replace("vio_trl_eff", "vio_trl")
            is_sel = pname == selected
            color = PARAM_COLORS.get(pname, C["text_dim"]) if is_sel else C["text_dim"]
            self.create_text(pad_l - 5, pad_t + int(cell_h * (pi + 0.5)),
                             text=short, fill=color, font=("Menlo", 9, "bold" if is_sel else ""),
                             anchor="e")

            for si, stage in enumerate(stages):
                v = stage["total_order"].get(pname, 0)
                x0 = pad_l + int(si * cell_w)
                y0 = pad_t + int(pi * cell_h)
                x1 = x0 + int(cell_w)
                y1 = y0 + int(cell_h)
                self.create_rectangle(x0, y0, x1, y1, fill=self._val_to_color(v), outline=C["border"])

        for si in range(n_stages):
            x = pad_l + int(cell_w * (si + 0.5))
            self.create_text(x, h - 10, text=f"{si}", fill=C["text_dim"], font=("Menlo", 8))

        self.create_text(w // 2, 14, text="SENSITIVITY SPECTROGRAM (S_Ti)",
                         fill=C["text"], font=("Menlo", 11, "bold"))


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
            self.create_arc(cx - r, cy - r, cx + r, cy + r,
                            start=start, extent=extent, fill=color, outline=C["border"])
            # Label
            mid_angle = np.radians(start + extent / 2)
            lx = cx + int((r * 0.7) * np.cos(mid_angle))
            ly = cy - int((r * 0.7) * np.sin(mid_angle))
            if frac > 0.01:
                self.create_text(lx, ly, text=f"{label}\n{frac * 100:.1f}%",
                                 fill="#000", font=("Menlo", 8, "bold"))
            start += extent

        # Donut hole
        ir = r // 2
        self.create_oval(cx - ir, cy - ir, cx + ir, cy + ir, fill=C["panel"], outline=C["panel"])

        self.create_text(w // 2, 12, text="VARIANCE DECOMP",
                         fill=C["text"], font=("Menlo", 10, "bold"))


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

        tk.Label(top, text="UQ DAW // RFC006", bg=C["panel"], fg=C["accent1"],
                 font=("Menlo", 13, "bold")).pack(side="left", padx=10)

        tk.Button(top, text="Load JSON", command=self._open_file,
                  bg=C["panel_light"], fg=C["text"], font=("Menlo", 10),
                  relief="flat", cursor="hand2").pack(side="right", padx=10)

        self.status_label = tk.Label(top, text="No data loaded", bg=C["panel"],
                                     fg=C["text_dim"], font=("Menlo", 9))
        self.status_label.pack(side="right", padx=10)

        self.header_label = tk.Label(top, text="", bg=C["panel"],
                                     fg=C["text_dim"], font=("Menlo", 9))
        self.header_label.pack(side="right", padx=10)

        # -- Main area (left controls + right visualizations) --
        main = tk.PanedWindow(self.root, orient="horizontal", bg=C["bg"],
                               sashwidth=4, sashrelief="flat")
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # Left panel: parameter selector + sliders
        self.left_frame = tk.Frame(main, bg=C["bg"], width=280)
        main.add(self.left_frame, minsize=250)

        # Right panel: visualizations
        self.right_frame = tk.Frame(main, bg=C["bg"])
        main.add(self.right_frame, minsize=600)

        # -- Left: Solo selector --
        solo_frame = tk.LabelFrame(self.left_frame, text="SOLO PARAMETER",
                                    bg=C["panel"], fg=C["accent1"],
                                    font=("Menlo", 10, "bold"), labelanchor="n")
        solo_frame.pack(fill="x", padx=4, pady=4)

        self.param_menu = ttk.Combobox(solo_frame, textvariable=self.selected_param,
                                        state="readonly", font=("Menlo", 10))
        self.param_menu.pack(fill="x", padx=8, pady=8)
        self.selected_param.trace_add("write", lambda *_: self._on_param_change())

        # -- Left: PCE sliders --
        slider_frame = tk.LabelFrame(self.left_frame, text="PCE SURROGATE KNOBS",
                                      bg=C["panel"], fg=C["accent3"],
                                      font=("Menlo", 10, "bold"), labelanchor="n")
        slider_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.slider_container = tk.Frame(slider_frame, bg=C["panel"])
        self.slider_container.pack(fill="both", expand=True, padx=4, pady=4)

        self.sliders = {}
        self.slider_labels = {}

        # -- Left: Prediction readout --
        self.readout_label = tk.Label(self.left_frame, text="Y\u0302 = ---",
                                      bg=C["panel"], fg=C["accent3"],
                                      font=("Menlo", 16, "bold"))
        self.readout_label.pack(fill="x", padx=4, pady=4)

        # -- Right: visualization grid --
        self._build_viz_panels()

    def _build_viz_panels(self):
        # Top row: EQ + Heatmap
        top_row = tk.Frame(self.right_frame, bg=C["bg"])
        top_row.pack(fill="both", expand=True, padx=2, pady=2)

        self.eq_canvas = SobolEQCanvas(top_row, height=200)
        self.eq_canvas.pack(side="left", fill="both", expand=True, padx=(0, 2))

        self.heatmap_canvas = HeatmapCanvas(top_row, height=200)
        self.heatmap_canvas.pack(side="right", fill="both", expand=True, padx=(2, 0))

        # Middle row: Response curves
        self.response_canvas = ResponseCurveCanvas(self.right_frame, height=280)
        self.response_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom row: Variance decomp (small)
        self.decomp_canvas = VarianceDecompCanvas(self.right_frame, height=150)
        self.decomp_canvas.pack(fill="x", padx=2, pady=2)

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

            short = pname.replace("mecillinam_concentration", "mecillinam").replace("vio_expression", "vio_exp").replace("vio_trl_eff", "vio_trl")
            tk.Label(frame, text=short, bg=C["panel"], fg=color,
                     font=("Menlo", 9, "bold"), width=10, anchor="w").pack(side="left", padx=4)

            slider = tk.Scale(frame, from_=lo, to=hi, orient="horizontal",
                              resolution=(hi - lo) / 200, length=140,
                              bg=C["panel"], fg=color, troughcolor=C["panel_light"],
                              highlightthickness=0, font=("Menlo", 8),
                              showvalue=False,
                              command=lambda v, p=pname: self._on_slider_change(p, v))
            slider.set(mid)
            slider.pack(side="left", padx=2)

            val_label = tk.Label(frame, text=f"{mid:.2f}", bg=C["panel"], fg=color,
                                  font=("Menlo", 9), width=7, anchor="e")
            val_label.pack(side="left", padx=4)

            self.sliders[pname] = slider
            self.slider_labels[pname] = val_label

    def _on_slider_change(self, pname, value):
        value = float(value)
        if pname in self.slider_labels:
            self.slider_labels[pname].config(text=f"{value:.2f}")
        self._update_response_curves()

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
        self.header_label.config(
            text=f"PARAMS:{n_p}  STAGES:{n_s}  POP R\u00b2:{pop_r2}  CC R\u00b2:{cc_r2}"
        )

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

        # Sweep each parameter
        n_sweep = 80
        curves = {}
        markers = {}
        all_y = []

        for pi, pname in enumerate(params):
            sv = np.linspace(float(bounds[pi, 0]), float(bounds[pi, 1]), n_sweep)
            sv_norm = (sv - bounds[pi, 0]) / (bounds[pi, 1] - bounds[pi, 0] + 1e-12)
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
            cur_norm = (x[pi] - bounds[pi, 0]) / (bounds[pi, 1] - bounds[pi, 0] + 1e-12)
            markers[pname] = (cur_norm, pop_y)

        if all_y:
            y_min = min(all_y) - 0.1 * abs(min(all_y))
            y_max = max(all_y) + 0.1 * abs(max(all_y))
            if y_min == y_max:
                y_min -= 0.5
                y_max += 0.5
        else:
            y_min, y_max = -1, 1

        self.response_canvas.set_curves(curves, markers, selected, (y_min, y_max))


# -- Entry point --------------------------------------------------------------


def run_tk_dashboard(data_path=None):
    """Launch the Tkinter UQ DAW."""
    root = tk.Tk()
    UQDawApp(root, data_path=data_path)
    root.mainloop()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tk_dashboard(_path)
