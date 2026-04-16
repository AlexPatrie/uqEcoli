"""
uq interactive dashboard — all 4 RFC006 strategies.

Launch:
    uv run marimo run app/dashboard_simple.py

Loads artifacts from a uq pipeline export (uq_results.json +
population_surrogate/) and provides reactive parameter exploration
across all four aggregation strategies:
  1. Uniform (bulk) — PCE response curves + population Sobol
  2. By generation — per-generation Sobol comparison
  3. By lineage seed — per-seed Sobol comparison
  4. Growth-stratified — sensitivity spectrogram + per-stage prediction

The observable set is determined by the ``--observables`` presets
passed to ``uq sample`` (mass, higher_order, exchange_fluxes,
transcriptome, proteome, fluxome — see ``uq/observables.py``).
The dashboard is observable-agnostic: it reads whatever Sobol indices
and surrogate coefficients ``quantify`` exported.
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    import json
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    COLORS = {
        "bg": "#0a0a0f",
        "panel": "#12121a",
        "grid": "#1a1a2e",
        "text": "#e0e0e0",
        "text_dim": "#888",
        "accent1": "#00f0ff",
        "accent2": "#ff3366",
        "accent3": "#00ff88",
        "accent4": "#ffaa00",
    }
    PCOLORS = ["#00f0ff", "#ff3366", "#00ff88", "#ffaa00", "#aa66ff", "#ff6600"]
    DAW = dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="JetBrains Mono, monospace", color=COLORS["text"], size=11),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        margin=dict(l=50, r=20, t=40, b=40),
    )

    file_input = mo.ui.file(filetypes=[".json"], label="Load uq_results.json from uq export")
    results_dir_input = mo.ui.text(
        value="./uq_results",
        label="Or enter export directory path (containing uq_results.json + population_surrogate/)",
        full_width=True,
    )

    return COLORS, DAW, Path, PCOLORS, file_input, go, json, make_subplots, np, results_dir_input


@app.cell
def _(Path, file_input, json, np, results_dir_input):
    _data = None
    _surr = {"available": False}

    # Try directory path input first (more reliable — finds both JSON + surrogates)
    _dir_path = Path(results_dir_input.value).resolve() if results_dir_input.value else None
    if _dir_path and (_dir_path / "uq_results.json").exists():
        _data = json.loads((_dir_path / "uq_results.json").read_text())
        _pop = _dir_path / "population_surrogate"
        if (_pop / "coefficients.npy").exists():
            try:
                _surr = {
                    "available": True,
                    "pop_coeffs": np.load(_pop / "coefficients.npy"),
                    "pop_mi": np.load(_pop / "multi_indices.npy"),
                    "bounds": np.load(_pop / "input_bounds.npy"),
                }
            except Exception as _e:
                _surr = {"available": False, "error": str(_e)}

    # Fallback: file upload widget
    elif file_input.value:
        _raw = file_input.value[0].contents
        _data = json.loads(_raw)

        for _d in [
            Path("./uq_results"),
            Path("./test_uq_results"),
            Path("examples/expected_output_simple/results"),
            Path("examples/expected_output/results"),
            Path("."),
        ]:
            _pop = _d / "population_surrogate"
            if (_pop / "coefficients.npy").exists():
                try:
                    _surr = {
                        "available": True,
                        "pop_coeffs": np.load(_pop / "coefficients.npy"),
                        "pop_mi": np.load(_pop / "multi_indices.npy"),
                        "bounds": np.load(_pop / "input_bounds.npy"),
                    }
                except Exception as _e:
                    _surr = {"available": False, "error": str(_e)}
                break

    data = _data
    surr = _surr
    return data, surr


@app.cell
def _(data, mo, surr):
    if data is None:
        mo.output.replace(mo.md("## Upload `uq_results.json` from `uq-simple quantify --export-path`"))
        mo.stop(True)
    if not surr.get("available"):
        mo.output.replace(mo.md("## Surrogate .npy files not found — place them next to uq_results.json"))
        mo.stop(True)

    _params = list(data["parameters"].keys())
    _bounds = surr["bounds"]

    _slider_list = []
    for _i, _p in enumerate(_params):
        _lo, _hi = float(_bounds[_i, 0]), float(_bounds[_i, 1])
        _slider_list.append(
            mo.ui.slider(start=_lo, stop=_hi, value=(_lo + _hi) / 2, step=(_hi - _lo) / 100, label=_p, full_width=True)
        )
    param_sliders = mo.ui.array(_slider_list)
    return (param_sliders,)


@app.cell
def _(np):
    def legendre_eval(x_norm, coeffs, mi):
        """Ŷ = Σ c_α · Π P_{α_i}(x_i) via stable Legendre recurrence."""
        _max_ord = int(mi.max()) if mi.size > 0 else 0
        _n_p = mi.shape[1]
        _P = np.zeros((_max_ord + 1, _n_p))
        _P[0, :] = 1.0
        if _max_ord >= 1:
            _P[1, :] = x_norm
        for _nn in range(2, _max_ord + 1):
            _P[_nn, :] = ((2 * _nn - 1) * x_norm * _P[_nn - 1, :] - (_nn - 1) * _P[_nn - 2, :]) / _nn
        _result = 0.0
        for _t in range(len(coeffs)):
            _term = float(coeffs[_t])
            for _pp in range(_n_p):
                _term *= _P[mi[_t, _pp], _pp]
            _result += _term
        return _result
    return (legendre_eval,)


@app.cell
def _(data, legendre_eval, np, param_sliders, surr):
    _params = list(data["parameters"].keys())
    _bounds = surr["bounds"]
    _coeffs = surr["pop_coeffs"]
    _mi = surr["pop_mi"]

    current_x = np.array([float(param_sliders[_i].value) for _i in range(len(_params))])
    current_x_norm = 2.0 * (current_x - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0
    current_y_hat = legendre_eval(current_x_norm, _coeffs, _mi)

    _n_sweep = 80
    sweep_data = {}
    sensitivity_data = {}
    for _pi, _p in enumerate(_params):
        _lo, _hi = float(_bounds[_pi, 0]), float(_bounds[_pi, 1])
        _sv = np.linspace(_lo, _hi, _n_sweep)
        _sy = []
        for _v in _sv:
            _x_sw = current_x.copy()
            _x_sw[_pi] = _v
            _x_sw_n = 2.0 * (_x_sw - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0
            _sy.append(legendre_eval(_x_sw_n, _coeffs, _mi))
        sweep_data[_p] = (_sv, np.array(_sy))

        _delta = (_hi - _lo) * 0.005
        _x_plus = current_x.copy()
        _x_plus[_pi] = min(current_x[_pi] + _delta, _hi)
        _x_minus = current_x.copy()
        _x_minus[_pi] = max(current_x[_pi] - _delta, _lo)
        _y_plus = legendre_eval(
            2.0 * (_x_plus - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0, _coeffs, _mi,
        )
        _y_minus = legendre_eval(
            2.0 * (_x_minus - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0, _coeffs, _mi,
        )
        sensitivity_data[_p] = abs(_y_plus - _y_minus) / (2 * _delta + 1e-12)

    return current_x, current_x_norm, current_y_hat, sensitivity_data, sweep_data


@app.cell
def _(COLORS, DAW, PCOLORS, data, go, np, current_x, current_y_hat, surr, sweep_data):
    _params = list(data["parameters"].keys())
    _bounds = surr["bounds"]

    fig_response = go.Figure()
    for _pi, _p in enumerate(_params):
        _sv, _sy = sweep_data[_p]
        _color = PCOLORS[_pi % len(PCOLORS)]
        _sv_norm = (_sv - _bounds[_pi, 0]) / (_bounds[_pi, 1] - _bounds[_pi, 0] + 1e-12)

        fig_response.add_trace(go.Scatter(
            x=_sv_norm, y=_sy, mode="lines", name=_p,
            line=dict(color=_color, width=2.5, shape="spline"),
            customdata=np.column_stack([_sv, _sy]),
            hovertemplate=f"<b>{_p}</b><br>value=%{{customdata[0]:.3f}}<br>Ŷ=%{{customdata[1]:.4f}}<extra></extra>",
        ))
        _cur_norm = (current_x[_pi] - _bounds[_pi, 0]) / (_bounds[_pi, 1] - _bounds[_pi, 0] + 1e-12)
        fig_response.add_trace(go.Scatter(
            x=[_cur_norm], y=[current_y_hat], mode="markers", showlegend=False,
            marker=dict(size=12, color=_color, symbol="diamond", line=dict(width=2, color="#fff")),
            hovertemplate=f"<b>{_p}</b>={current_x[_pi]:.3f}<br>Ŷ={current_y_hat:.4f}<extra></extra>",
        ))

    fig_response.update_layout(
        **DAW, height=320,
        title=dict(text=f"PCE RESPONSE CURVES — Ŷ = {current_y_hat:.4f}", font=dict(size=13, color=COLORS["accent1"])),
        xaxis_title="Normalized parameter [0 = min, 1 = max]",
        yaxis_title="Ŷ (surrogate prediction)",
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10)),
    )
    return (fig_response,)


@app.cell
def _(COLORS, DAW, data, go, make_subplots, np, current_x_norm, current_y_hat, sensitivity_data):
    _params = list(data["parameters"].keys())
    _stages = data["phase2_growth_stratified"]["stages"]
    _n_stages = len(_stages)
    _theta_labels = [f"θ{_s['stage']}" for _s in _stages]
    _growth_labels = [_s.get("growth_description", f"Stage {_s['stage']}") for _s in _stages]

    fig_spectrogram = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45],
        subplot_titles=("Sensitivity Spectrogram (S_Ti) — by growth stage", "Per-Stage Prediction Ŷ"),
        vertical_spacing=0.18, shared_xaxes=True,
    )

    _z_sens = [[_s["sobol_total_order"].get(_p, 0) for _s in _stages] for _p in _params]
    _short = [_p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac") for _p in _params]

    fig_spectrogram.add_trace(go.Heatmap(
        z=_z_sens, x=_theta_labels, y=_short,
        colorscale=[[0, "#0d0d0d"], [0.15, "#1a1a4e"], [0.3, "#2a2a8e"],
                     [0.5, "#00b4d8"], [0.7, "#00f0ff"], [0.85, "#ffaa00"], [1.0, "#ff3366"]],
        colorbar=dict(title="S_Ti", len=0.4, y=0.82, thickness=10, x=1.02),
        zmin=0, zmax=max(0.5, max(max(_row) for _row in _z_sens)),
        customdata=[[_growth_labels[_j] for _j in range(_n_stages)] for _ in _params],
        hovertemplate="Param: %{y}<br>%{customdata}<br>S_Ti: %{z:.3f}<extra></extra>",
    ), row=1, col=1)

    _stage_preds = np.zeros(_n_stages)
    for _si, _s in enumerate(_stages):
        _contrib = sum(
            sensitivity_data.get(_p, 0) * current_x_norm[_pi] * _s["sobol_total_order"].get(_p, 0)
            for _pi, _p in enumerate(_params)
        )
        _stage_preds[_si] = current_y_hat + _contrib

    fig_spectrogram.add_trace(go.Scatter(
        x=_theta_labels, y=_stage_preds, mode="lines+markers", name="Ŷ per stage",
        line=dict(color=COLORS["accent3"], width=3, shape="spline"),
        marker=dict(size=7, color=COLORS["accent3"]),
        customdata=_growth_labels,
        hovertemplate="%{customdata}<br>Ŷ=%{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig_spectrogram.add_hline(y=current_y_hat, line_dash="dot", line_color=COLORS["text_dim"],
                               annotation_text=f"bulk Ŷ={current_y_hat:.4f}", row=2, col=1)

    fig_spectrogram.update_layout(**DAW, height=420, showlegend=False)
    fig_spectrogram.update_yaxes(title_text="Ŷ", row=2, col=1, title_font=dict(color=COLORS["accent3"]))
    for _ann in fig_spectrogram.layout.annotations:
        _ann.font = dict(color=COLORS["text_dim"], size=11)

    return (fig_spectrogram,)


@app.cell
def _(COLORS, DAW, PCOLORS, data, go, mo):
    # Strategy 2: grouped bar chart (generations × params)
    _s2 = data.get("strategy2_by_generation", {})
    _s2_gens = _s2.get("generations", [])
    _params = list(data["parameters"].keys())

    if _s2_gens:
        fig_s2 = go.Figure()
        for _pi, _p in enumerate(_params):
            _short = _p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
            _vals = [_g["sobol_total_order"].get(_p, 0) for _g in _s2_gens]
            _labels = [f"Gen {_g['generation']}" for _g in _s2_gens]
            fig_s2.add_trace(go.Bar(
                name=_short, x=_labels, y=_vals,
                marker_color=PCOLORS[_pi % len(PCOLORS)],
                hovertemplate=f"<b>{_short}</b><br>S_Ti=%{{y:.3f}}<extra></extra>",
            ))
        fig_s2.update_layout(
            **DAW, height=300, barmode="group",
            title=dict(text="STRATEGY 2 — BY GENERATION (S_Ti)", font=dict(size=13, color=COLORS["accent1"])),
            xaxis_title="Generation", yaxis_title="S_Ti (total Sobol)",
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
        )
        fig_s2_output = mo.ui.plotly(fig_s2)
    else:
        fig_s2_output = mo.md("*Strategy 2 not available (run with `--generations >= 2`)*")
    return (fig_s2_output,)


@app.cell
def _(COLORS, DAW, PCOLORS, data, go, mo):
    # Strategy 3: grouped bar chart (seeds × params)
    _s3 = data.get("strategy3_by_seed", {})
    _s3_seeds = _s3.get("seeds", [])
    _params = list(data["parameters"].keys())

    if _s3_seeds:
        fig_s3 = go.Figure()
        for _pi, _p in enumerate(_params):
            _short = _p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
            _vals = [_s["sobol_total_order"].get(_p, 0) for _s in _s3_seeds]
            _labels = [f"Seed {_s['lineage_seed']}" for _s in _s3_seeds]
            fig_s3.add_trace(go.Bar(
                name=_short, x=_labels, y=_vals,
                marker_color=PCOLORS[_pi % len(PCOLORS)],
                hovertemplate=f"<b>{_short}</b><br>S_Ti=%{{y:.3f}}<extra></extra>",
            ))
        fig_s3.update_layout(
            **DAW, height=300, barmode="group",
            title=dict(text="STRATEGY 3 — BY LINEAGE SEED (S_Ti)", font=dict(size=13, color=COLORS["accent1"])),
            xaxis_title="Lineage Seed", yaxis_title="S_Ti (total Sobol)",
            legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
        )
        fig_s3_output = mo.ui.plotly(fig_s3)
    else:
        fig_s3_output = mo.md("*Strategy 3 not available (run with `--n-init-sims >= 2`)*")
    return (fig_s3_output,)


@app.cell
def _(data, mo):
    _params_info = data["parameters"]
    _sobol = data["phase1_population"]["sobol_total_order"]

    _rows = []
    for _name, _info in _params_info.items():
        _s_ti = _sobol.get(_name, 0)
        _rows.append(f"| `{_name}` | {_s_ti:.1%} | {_info.get('biological_role', '')} |")

    param_table = mo.md(
        "### Strategy 1 — Population-Averaged Sensitivity\n\n"
        "| Parameter | S_Ti | Biological Role |\n"
        "|-----------|------|------------------|\n"
        + "\n".join(_rows)
        + "\n\n*S_Ti = total Sobol index: fraction of output variance attributable to this parameter.*"
    )
    return (param_table,)


@app.cell
def _(data, fig_response, fig_s2_output, fig_s3_output, fig_spectrogram, file_input, mo, param_sliders, param_table, results_dir_input):
    _methods = data.get("methods", {}) if data else {}
    _refs = data.get("references", []) if data else []
    _params = list(data["parameters"].keys()) if data else []
    _n_stages = data["phase2_growth_stratified"]["n_stages"] if data else 0

    _header = mo.md(f"""
# UQ Sensitivity Explorer — vEcoli Whole-Cell Model

**Parameters:** {', '.join(f'`{_p}`' for _p in _params)} |
**Growth stages:** {_n_stages} |
**Surrogate:** {_methods.get('surrogate', 'PCE')} |
**Sensitivity:** {_methods.get('sensitivity', 'Sobol')}

{' · '.join(_refs)}

---
""")

    _slider_panel = mo.vstack([
        mo.md("### Parameter Controls"),
        mo.md("*Drag sliders → instant PCE prediction*"),
        param_sliders,
        mo.md("---"),
        param_table,
    ])

    _plot_panel = mo.vstack([
        mo.ui.plotly(fig_response),
        mo.ui.plotly(fig_spectrogram),
        mo.hstack([fig_s2_output, fig_s3_output], widths=[1, 1]),
    ])

    _s2_status = "available" if "generations" in data.get("strategy2_by_generation", {}) else "not available"
    _s3_status = "available" if "seeds" in data.get("strategy3_by_seed", {}) else "not available"
    _methods_md = mo.md(
        "### Methods\n"
        f"- **Sampling:** {_methods.get('sampling', 'LHS')}\n"
        f"- **Surrogate:** {_methods.get('surrogate', 'PCE')}\n"
        f"- **Sensitivity:** {_methods.get('sensitivity', 'Sobol')}\n"
        f"- **Stratification:** {_methods.get('stratification', 'growth-based')}\n"
        f"- **Strategy 2 (by generation):** {_s2_status}\n"
        f"- **Strategy 3 (by seed):** {_s3_status}\n"
    )

    mo.output.replace(mo.vstack([
        mo.hstack([results_dir_input, file_input]),
        _header,
        mo.hstack([_slider_panel, _plot_panel], widths=[1, 3]),
        _methods_md,
    ]))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
