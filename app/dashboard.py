import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full", app_title="UQ Results // DAW", css_file="")

with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # -- DAW color palette (dark theme, neon accents) --
    C = {
        "bg": "#0d0d0d",
        "panel": "#1a1a2e",
        "panel_border": "#2a2a4a",
        "text": "#e0e0e0",
        "text_dim": "#888",
        "accent1": "#00f0ff",  # cyan
        "accent2": "#ff3366",  # hot pink
        "accent3": "#33ff99",  # neon green
        "accent4": "#ffaa00",  # amber
        "accent5": "#aa66ff",  # purple
        "grid": "#222244",
        "vio": "#00f0ff",
        "trl": "#ffaa00",
        "mec": "#ff3366",
    }

    PARAM_COLORS = {
        "vio_expression": C["vio"],
        "vio_trl_eff": C["trl"],
        "mecillinam_concentration": C["mec"],
    }

    DAW_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["panel"],
        font=dict(family="JetBrains Mono, SF Mono, Fira Code, monospace", color=C["text"], size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
    )


@app.cell
def file_selector():
    _default = str(Path(__file__).parent.parent / "test_export_output" / "uq_results.json")
    file_input = mo.ui.text(
        value=_default,
        label="",
        full_width=True,
    )
    mo.output.replace(
        mo.vstack([
            mo.md("### Load UQ Results"),
            file_input,
        ])
    )
    return (file_input,)


@app.cell
def load_data(file_input):
    _path = Path(file_input.value)
    if not _path.exists():
        mo.output.replace(mo.md(f"**File not found:** `{_path}`"))
        data = None
    else:
        with open(_path) as _f:
            data = json.load(_f)
    return (data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A dropdown to pick one parameter, with color-coded badges for all parameters (cyan = vio_expression, amber = vio_trl_eff, pink =
    mecillinam_concentration).

    This is the master control. Every reactive panel below highlights the selected ("soloed") parameter and dims the others. Like soloing a
    channel in a DAW — you're isolating one parameter's contribution to hear it above the mix. A stakeholder can click through each parameter and instantly see
    how its influence changes across the cell cycle, without the other parameters' curves cluttering the view.
    """)
    return


@app.cell
def param_selector(data):
    if data is None:
        mo.stop(True, mo.md("Waiting for data..."))

    _params = data["parameter_names"]
    param_dropdown = mo.ui.dropdown(
        options={p: p for p in _params},
        value=_params[0],
        label="",
    )

    _badges = " ".join(
        f'<span style="background:{PARAM_COLORS.get(p,"#666")};color:#000;'
        f'padding:2px 8px;border-radius:3px;font-size:11px;font-family:monospace;'
        f'font-weight:bold;margin-right:4px;">{p}</span>'
        for p in _params
    )

    mo.output.replace(
        mo.vstack([
            mo.md(f""" \
            A dropdown to pick one parameter, with color-coded badges for all parameters (cyan = vio_expression, amber = vio_trl_eff, pink =
    mecillinam_concentration).

    This is the master control. Every reactive panel below highlights the selected ("soloed") parameter and dims the others. Like soloing a
    channel in a DAW — you're isolating one parameter's contribution to hear it above the mix. A stakeholder can click through each parameter and instantly see
    how its influence changes across the cell cycle, without the other parameters' curves cluttering the view.
            """),
            mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    <div style="display:flex;align-items:center;gap:16px;">
      <div style="color:#888;font-size:12px;text-transform:uppercase;letter-spacing:2px;">Solo Parameter</div>
      <div>{param_dropdown}</div>
    </div>
    <div style="margin-top:10px;">{_badges}</div>
    </div>
    """),
        ])
    )
    return (param_dropdown,)


@app.cell
def header_strip(data):
    if data is None:
        mo.stop(True)

    _surr = data.get("surrogates", {})
    _pop_r2 = _surr.get("population", {}).get("r_squared", "?")
    _cc_r2 = _surr.get("cell_cycle", {}).get("r_squared", "?")
    _n_stages = data.get("n_cell_cycle_stages", "?")
    _n_params = data.get("n_parameters", "?")
    _morris = data.get("morris_screening")
    _n_traj = _morris.get("n_trajectories", "?") if _morris else "?"

    mo.output.replace(
        mo.vstack([
        mo.md(f""" \
        ### Header Strip (header_strip, line 122)

      What it shows: Five at-a-glance metrics — number of parameters, number of cell cycle stages, Phase 1 surrogate R^2, Phase 2 surrogate R^2, and
      Morris trajectory count.

      Why it's useful: These are the "trust indicators" for the entire analysis. R^2 tells you how well the PCE polynomial approximates the real
      simulation — if POP R^2 = 0.92, the surrogate explains 92% of output variance, and the Sobol indices derived from it are reliable. If R^2 were
      0.3, everything downstream would be suspect. Morris trajectory count tells you how well-sampled the prescreening was. A stakeholder can look at
      this strip and immediately know whether to trust the results below.
        """),
        mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;padding:10px 16px;
            display:flex;gap:32px;align-items:center;font-family:monospace;font-size:12px;">
      <div style="color:{C['accent1']};"><b>PARAMS</b> <span style="color:#e0e0e0;">{_n_params}</span></div>
      <div style="color:{C['accent3']};"><b>STAGES</b> <span style="color:#e0e0e0;">{_n_stages}</span></div>
      <div style="color:{C['accent4']};"><b>POP R\u00b2</b> <span style="color:#e0e0e0;">{_pop_r2}</span></div>
      <div style="color:{C['accent5']};"><b>CC R\u00b2</b> <span style="color:#e0e0e0;">{_cc_r2}</span></div>
      <div style="color:{C['accent2']};"><b>MORRIS r</b> <span style="color:#e0e0e0;">{_n_traj}</span></div>
      <div style="flex:1;"></div>
      <div style="color:#444;font-size:10px;letter-spacing:3px;">RFC006 // UQ DAW</div>
    </div>
    """)]))
    return


@app.cell
def eq_strip(data, param_dropdown):
    """Sobol EQ: per-stage S_i shown as frequency bands, selected param highlighted."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _stages = data["phase2_cell_cycle_sobol_per_stage"]
    _params = data["parameter_names"]
    _n = len(_stages)

    # Build x-axis: theta bin centers
    _x = [(s["theta_range"][0] + s["theta_range"][1]) / 2 for s in _stages]
    _x_labels = [f"\u03b8={s['theta_range'][0]:.1f}" for s in _stages]

    _fig = go.Figure()

    # Background traces (dimmed)
    for _p in _params:
        _y = [s["total_order"].get(_p, 0) for s in _stages]
        _is_selected = _p == _selected
        _fig.add_trace(go.Scatter(
            x=_x, y=_y,
            mode="lines",
            name=_p,
            line=dict(
                color=PARAM_COLORS.get(_p, "#666"),
                width=4 if _is_selected else 1.5,
                shape="spline",
            ),
            opacity=1.0 if _is_selected else 0.25,
            hovertemplate=f"<b>{_p}</b><br>\u03b8=%{{x:.2f}}<br>S_Ti=%{{y:.3f}}<extra></extra>",
        ))

    # Fill under selected param
    _sel_y = [s["total_order"].get(_selected, 0) for s in _stages]
    _sel_color = PARAM_COLORS.get(_selected, C["accent1"])
    _fig.add_trace(go.Scatter(
        x=_x, y=_sel_y,
        fill="tozeroy",
        mode="none",
        # fillcolor=_sel_color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in _sel_color else _sel_color + "1a",
        fillcolor="#ff0000",
        showlegend=False,
        hoverinfo="skip",
    ))

    # First-order as dashed overlay for selected
    _sel_s1 = [s["first_order"].get(_selected, 0) for s in _stages]
    _fig.add_trace(go.Scatter(
        x=_x, y=_sel_s1,
        mode="lines",
        name=f"{_selected} (S\u1d62)",
        line=dict(color=_sel_color, width=2, dash="dot"),
        opacity=0.7,
        hovertemplate=f"<b>{_selected} (main effect)</b><br>\u03b8=%{{x:.2f}}<br>S_i=%{{y:.3f}}<extra></extra>",
    ))

    # Interaction band (S_Ti - S_i)
    _interaction = [t - s for t, s in zip(_sel_y, _sel_s1)]
    if max(_interaction) > 0.01:
        _fig.add_trace(go.Bar(
            x=_x, y=_interaction,
            name="Interactions",
            marker_color=_sel_color,
            opacity=0.15,
            width=0.08,
            hovertemplate="Interaction=%{y:.3f}<extra></extra>",
        ))

    _fig.update_layout(
        **DAW_LAYOUT,
        height=280,
        title=dict(text=f"PARAMETRIC EQ // S_Ti across cell cycle \u2014 solo: {_selected}", font=dict(size=13)),
        xaxis_title="\u03b8 (cell cycle position)",
        yaxis_title="Sobol Index",
        yaxis_range=[0, max(max(_sel_y) * 1.15, 0.6)],
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        bargap=0.3,
    )

    mo.output.replace(mo.vstack([
        mo.md(f""" \
        Parametric EQ (eq_strip, line 149)

      What it shows: Three overlapping spline curves, one per parameter, plotted across θ (cell cycle position, 0 to 1). The soloed parameter is thick
      with a filled area underneath; the others are ghosted. Two lines for the soloed param:
      - Solid line = S_Ti (total-order Sobol index — includes all interactions)
      - Dashed line = S_i (first-order — main effect only, no interactions)
      - Semi-transparent bars = the gap between them (S_Ti - S_i = interaction contribution)

      Why it's useful: This is the core visualization of the entire pipeline. It answers the central RFC006 question: "How does each parameter's
      importance change across the cell cycle?"

      In your data, you can see vio_expression (cyan) starts high at θ=0 (birth, S_Ti ≈ 0.64) and drops to near zero at θ=1 (pre-division). Mecillinam
      (pink) does the opposite — negligible early, dominant late. This tells a biologist: "vio pathway matters most during early growth; mecillinam's
      PBP2 inhibition matters most during septation/division." That's actionable — it tells you when in the cell cycle each drug/perturbation has its
      effect.

      The EQ analogy: each parameter is a frequency band. The curve shape shows where in the cell cycle "spectrum" that parameter's energy concentrates.
       Soloing isolates one band.

        """),mo.ui.plotly(_fig)]))
    return


@app.cell
def channel_strip(data, param_dropdown):
    """Channel strip: population fader + Morris meter + variance decomp pie."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _params = data["parameter_names"]
    _sel_color = PARAM_COLORS.get(_selected, C["accent1"])

    _fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Population Faders (S\u1d62 / S_Ti)", "Morris Screening (\u03bc* vs \u03c3)", "Variance Decomposition"),
        column_widths=[0.35, 0.35, 0.3],
        horizontal_spacing=0.08,
    )

    # -- Panel 1: Population Sobol as vertical faders --
    _p1 = data["phase1_population_sobol"]
    for _i, _p in enumerate(_params):
        _s1 = _p1["first_order"].get(_p, 0)
        _st = _p1["total_order"].get(_p, 0)
        _pc = PARAM_COLORS.get(_p, "#666")
        _alpha = 1.0 if _p == _selected else 0.4
        _fig.add_trace(go.Bar(
            x=[_p], y=[_st],
            name=f"{_p} S_Ti",
            marker_color=_pc,
            opacity=_alpha,
            width=0.35,
            showlegend=False,
            hovertemplate=f"<b>{_p}</b><br>S_Ti={_st:.3f}<extra></extra>",
        ), row=1, col=1)
        _fig.add_trace(go.Bar(
            x=[_p], y=[_s1],
            name=f"{_p} S\u1d62",
            marker_color=_pc,
            opacity=_alpha * 0.5,
            width=0.2,
            showlegend=False,
            hovertemplate=f"<b>{_p}</b><br>S_i={_s1:.3f}<extra></extra>",
        ), row=1, col=1)

    _fig.update_yaxes(range=[0, 1], title_text="Index", row=1, col=1)

    # -- Panel 2: Morris mu* vs sigma scatter --
    _morris = data.get("morris_screening")
    if _morris:
        _m_params = _morris["parameter_names"]
        _mu_star = _morris["mu_star"]
        _sigma = _morris["sigma"]
        for _i, _p in enumerate(_m_params):
            _pc = PARAM_COLORS.get(_p, "#666")
            _is_sel = _p == _selected
            _fig.add_trace(go.Scatter(
                x=[_mu_star[_i]], y=[_sigma[_i]],
                mode="markers+text",
                text=[_p.split("_")[-1]],
                textposition="top center",
                textfont=dict(color=_pc, size=10 if _is_sel else 8),
                marker=dict(
                    size=18 if _is_sel else 10,
                    color=_pc,
                    opacity=1.0 if _is_sel else 0.4,
                    symbol="diamond" if _is_sel else "circle",
                    line=dict(width=2 if _is_sel else 0, color="#fff"),
                ),
                showlegend=False,
                hovertemplate=f"<b>{_p}</b><br>\u03bc*={_mu_star[_i]:.3f}<br>\u03c3={_sigma[_i]:.3f}<extra></extra>",
            ), row=1, col=2)
        # Classification line: sigma = 0.5 * mu*
        _mx = max(_mu_star) * 1.2
        _fig.add_trace(go.Scatter(
            x=[0, _mx], y=[0, 0.5 * _mx],
            mode="lines",
            line=dict(color="#444", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=2)
        _fig.add_annotation(
            x=_mx * 0.7, y=0.5 * _mx * 0.7 + 0.02,
            text="nonlinear/interactive",
            showarrow=False, font=dict(color="#555", size=9),
            xref="x2", yref="y2",
        )

    _fig.update_xaxes(title_text="\u03bc* (importance)", row=1, col=2)
    _fig.update_yaxes(title_text="\u03c3 (nonlinearity)", row=1, col=2)

    # -- Panel 3: Variance decomposition pie --
    _decomp = data.get("variance_decomposition", {})
    _gen_f = _decomp.get("generation_fraction", [0])[0]
    _seed_f = _decomp.get("seed_fraction", [0])[0]
    _resid_f = _decomp.get("residual_fraction", [1])[0]

    _fig.add_trace(go.Pie(
        labels=["Generation", "Seed", "Residual (cell cycle)"],
        values=[_gen_f, _seed_f, _resid_f],
        marker=dict(colors=[C["accent4"], C["accent5"], C["accent3"]]),
        textinfo="label+percent",
        textfont=dict(size=10),
        hole=0.45,
        hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
    ), row=1, col=3)

    _fig.update_layout(
        **DAW_LAYOUT,
        height=300,
        showlegend=False,
    )
    # Fix subplot title colors
    for _ann in _fig.layout.annotations:
        _ann.font = dict(color=C["text_dim"], size=11)

    mo.output.replace(mo.ui.plotly(_fig))
    return


@app.cell
def heatmap_strip(data, param_dropdown):
    """Heatmap: all params x stages as spectrogram + cell cycle profile overlay."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _stages = data["phase2_cell_cycle_sobol_per_stage"]
    _params = data["parameter_names"]
    _n_stages = len(_stages)
    _n_params = len(_params)
    _profile = data.get("cell_cycle_profile")

    _fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=("Sensitivity Spectrogram (S_Ti)", "Cell Cycle Profile"),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # -- Panel 1: Heatmap (params x stages) --
    _z = np.zeros((_n_params, _n_stages))
    for _j, _s in enumerate(_stages):
        for _i, _p in enumerate(_params):
            _z[_i, _j] = _s["total_order"].get(_p, 0)

    _theta_labels = [f"{s['theta_range'][0]:.1f}-{s['theta_range'][1]:.1f}" for s in _stages]
    _short_params = [p.replace("mecillinam_concentration", "mecillinam").replace("vio_expression", "vio_exp").replace("vio_trl_eff", "vio_trl") for p in _params]

    # Highlight row for selected param
    _sel_idx = _params.index(_selected) if _selected in _params else 0

    _fig.add_trace(go.Heatmap(
        z=_z,
        x=_theta_labels,
        y=_short_params,
        colorscale=[
            [0.0, "#0d0d0d"],
            [0.15, "#1a1a4e"],
            [0.3, "#2a2a8e"],
            [0.5, "#00b4d8"],
            [0.7, "#00f0ff"],
            [0.85, "#ffaa00"],
            [1.0, "#ff3366"],
        ],
        colorbar=dict(title="S_Ti", len=0.45, y=0.78, thickness=12),
        hovertemplate="Param: %{y}<br>\u03b8: %{x}<br>S_Ti: %{z:.3f}<extra></extra>",
    ), row=1, col=1)

    # Selection indicator bracket on y-axis
    _fig.add_annotation(
        x=-0.5, y=_sel_idx,
        text="\u25b6",
        showarrow=False,
        font=dict(color=PARAM_COLORS.get(_selected, "#fff"), size=16),
        xref="x", yref="y",
    )

    # -- Panel 2: Cell cycle profile (mass + growth) --
    if _profile:
        _stages_x = _profile["stages"]
        _x_labels_p = [f"{i/len(_stages_x):.1f}-{(i+1)/len(_stages_x):.1f}" for i in _stages_x]

        # Find mass and growth keys dynamically
        _mass_key = next((k for k in _profile if "mass" in k.lower() and "mean" in k.lower()), None)
        _growth_key = next((k for k in _profile if "growth" in k.lower() and "mean" in k.lower()), None)

        if _mass_key:
            _mass = _profile[_mass_key]
            _fig.add_trace(go.Scatter(
                x=_x_labels_p, y=_mass,
                mode="lines+markers",
                name="Mass",
                line=dict(color=C["accent3"], width=3, shape="spline"),
                marker=dict(size=6, color=C["accent3"]),
                hovertemplate="Mass=%{y:.3f}<extra></extra>",
            ), row=2, col=1)
            _fig.update_yaxes(title_text="Dry Mass", row=2, col=1, title_font=dict(color=C["accent3"]))

        if _growth_key:
            _growth = _profile[_growth_key]
            # Normalize growth to mass scale for dual-axis overlay
            _g_min, _g_max = min(_growth), max(_growth)
            if _mass_key:
                _m_min, _m_max = min(_mass), max(_mass)
                _g_norm = [_m_min + (_m_max - _m_min) * (g - _g_min) / (_g_max - _g_min + 1e-12) for g in _growth]
            else:
                _g_norm = _growth

            _fig.add_trace(go.Scatter(
                x=_x_labels_p, y=_g_norm,
                mode="lines+markers",
                name="Growth Rate (scaled)",
                line=dict(color=C["accent4"], width=2, dash="dash", shape="spline"),
                marker=dict(size=5, color=C["accent4"], symbol="diamond"),
                hovertemplate="Growth=%{customdata:.5f}<extra></extra>",
                customdata=_growth,
            ), row=2, col=1)

        # Add B/C/D period annotations
        _fig.add_vrect(x0=_x_labels_p[0], x1=_x_labels_p[1], fillcolor=C["accent1"], opacity=0.06,
                       line_width=0, row=2, col=1, annotation_text="B", annotation_position="top left",
                       annotation=dict(font=dict(color=C["accent1"], size=10)))
        if len(_x_labels_p) > 7:
            _fig.add_vrect(x0=_x_labels_p[2], x1=_x_labels_p[6], fillcolor=C["accent5"], opacity=0.04,
                           line_width=0, row=2, col=1, annotation_text="C", annotation_position="top left",
                           annotation=dict(font=dict(color=C["accent5"], size=10)))
            _fig.add_vrect(x0=_x_labels_p[7], x1=_x_labels_p[-1], fillcolor=C["accent2"], opacity=0.05,
                           line_width=0, row=2, col=1, annotation_text="D", annotation_position="top left",
                           annotation=dict(font=dict(color=C["accent2"], size=10)))

    _fig.update_layout(
        **DAW_LAYOUT,
        height=520,
        legend=dict(orientation="h", y=-0.08, x=0.3, font=dict(size=10)),
    )
    for _ann in _fig.layout.annotations:
        if hasattr(_ann, "font") and _ann.font is not None:
            pass
        else:
            _ann.font = dict(color=C["text_dim"], size=11)

    mo.output.replace(mo.ui.plotly(_fig))
    return


@app.cell
def relevance_strip(data):
    """Relevance meters + surrogate specs."""
    if data is None:
        mo.stop(True)

    _ccr = data.get("cell_cycle_relevance")
    _surr = data.get("surrogates", {})

    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Observable Relevance (cell cycle)", "Surrogate Specs"),
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "table"}]],
    )

    if _ccr:
        _obs = _ccr.get("relevant_observables", [])
        _scores = _ccr.get("relevance_scores", {})
        _names = [o.split("__")[-1] for o in _obs]
        _vals = [_scores.get(o, 0) for o in _obs]

        _fig.add_trace(go.Bar(
            x=_vals, y=_names,
            orientation="h",
            marker=dict(
                color=[C["accent3"] if v > 0.8 else C["accent4"] if v > 0.5 else C["text_dim"] for v in _vals],
            ),
            text=[f"{v:.2f}" for v in _vals],
            textposition="outside",
            textfont=dict(color=C["text"], size=11),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    _fig.update_xaxes(title_text="Relevance Score", range=[0, 1.1], row=1, col=1)

    # Surrogate specs table
    _pop = _surr.get("population", {})
    _cc = _surr.get("cell_cycle", {})
    _fig.add_trace(go.Table(
        header=dict(
            values=["", "Population", "Cell Cycle"],
            fill_color=C["panel"],
            font=dict(color=C["accent1"], size=11),
            line=dict(color=C["panel_border"]),
            align="left",
        ),
        cells=dict(
            values=[
                ["Basis", "Order p", "Dim in", "Dim out", "R\u00b2", "Terms"],
                [_pop.get("basis_type", "?"), _pop.get("polynomial_order", "?"),
                 _pop.get("input_dim", "?"), _pop.get("output_dim", "?"),
                 f'{_pop.get("r_squared", 0):.3f}', _pop.get("n_terms", "?")],
                [_cc.get("basis_type", "?"), _cc.get("polynomial_order", "?"),
                 _cc.get("input_dim", "?"), _cc.get("output_dim", "?"),
                 f'{_cc.get("r_squared", 0):.3f}', _cc.get("n_terms", "?")],
            ],
            fill_color=C["bg"],
            font=dict(color=C["text"], size=11, family="monospace"),
            line=dict(color=C["panel_border"]),
            align="left",
            height=26,
        ),
    ), row=1, col=2)

    _fig.update_layout(
        **DAW_LAYOUT,
        height=250,
    )
    for _ann in _fig.layout.annotations:
        _ann.font = dict(color=C["text_dim"], size=11)

    mo.output.replace(mo.ui.plotly(_fig))
    return


@app.cell
def stage_detail(data, param_dropdown):
    """Per-stage detail: bar chart of all params' S_Ti for selected param highlighted."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _stages = data["phase2_cell_cycle_sobol_per_stage"]
    _params = data["parameter_names"]

    _fig = go.Figure()

    _x_stages = [f"Stage {s['stage']}" for s in _stages]

    for _p in _params:
        _st = [s["total_order"].get(_p, 0) for s in _stages]
        _pc = PARAM_COLORS.get(_p, "#666")
        _is_sel = _p == _selected

        # Total order bars
        _fig.add_trace(go.Bar(
            x=_x_stages, y=_st,
            name=f"{_p} S_Ti",
            marker_color=_pc,
            opacity=1.0 if _is_sel else 0.2,
            hovertemplate=f"<b>{_p}</b> S_Ti=%{{y:.3f}}<extra></extra>",
        ))

    _fig.update_layout(
        **DAW_LAYOUT,
        height=260,
        barmode="group",
        title=dict(text=f"MIXER // Per-Stage Sensitivity \u2014 solo: {_selected}", font=dict(size=13)),
        xaxis_title="Cell Cycle Stage",
        yaxis_title="S_Ti",
        yaxis_range=[0, 0.7],
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
    )

    mo.output.replace(mo.ui.plotly(_fig))
    return


@app.cell
def interaction_analyzer(data, param_dropdown):
    """Interaction gap (S_Ti - S_i) across stages for all params."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _stages = data["phase2_cell_cycle_sobol_per_stage"]
    _params = data["parameter_names"]

    _fig = go.Figure()

    for _p in _params:
        _gaps = [s["total_order"].get(_p, 0) - s["first_order"].get(_p, 0) for s in _stages]
        _pc = PARAM_COLORS.get(_p, "#666")
        _is_sel = _p == _selected

        _fig.add_trace(go.Scatter(
            x=list(range(len(_stages))),
            y=_gaps,
            mode="lines+markers",
            name=_p,
            line=dict(color=_pc, width=3 if _is_sel else 1, shape="spline"),
            marker=dict(size=8 if _is_sel else 4),
            opacity=1.0 if _is_sel else 0.3,
            hovertemplate=f"<b>{_p}</b><br>Stage %{{x}}<br>Interaction=%{{y:.4f}}<extra></extra>",
        ))

    _fig.update_layout(
        **DAW_LAYOUT,
        height=220,
        title=dict(text="SIDECHAIN // Interaction Gap (S_Ti \u2212 S\u1d62)", font=dict(size=13)),
        xaxis_title="Cell Cycle Stage",
        yaxis_title="Interaction",
        legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=10)),
    )

    mo.output.replace(mo.ui.plotly(_fig))
    return


@app.cell
def footer():
    mo.output.replace(mo.md("""
    <div style="text-align:center;color:#333;font-size:10px;font-family:monospace;
            padding:12px;letter-spacing:2px;">
    RFC006 // MILESTONE 08.4.2 // UQ DAW
    </div>
    """))
    return


if __name__ == "__main__":
    app.run()
