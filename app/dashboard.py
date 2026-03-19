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
def file_selector(set_file_input):
    file_area = mo.ui.file(kind="area", on_change=lambda v: set_file_input(v[0].name))
    return (file_area,)


@app.cell
def _():
    from uq.common.utils import get_repo_root

    _default = str(get_repo_root() / "examples" / "uq_artifacts" / "test_export_output" / "uq_results.json")
    get_file_input, set_file_input = mo.state(_default)
    return get_file_input, set_file_input


@app.cell
def _(file_area, get_file_input, header, set_file_input):
    val = get_file_input()
    if val is None:
        set_file_input(_default)

    file_input = mo.ui.text(
        value=get_file_input(),
        # value=file_val,
        label="",
        full_width=True,
    )
    _header = header("""
    ### 0. File Loader:

    Upload the `uq_results.json` file generated from a given `uq` pipeline workflow.
    """)

    mo.output.replace(
        mo.vstack([
            # _header,
            mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
        <div> {_header} </div>
        <div style="color:#888;font-size:12px;text-transform:uppercase;letter-spacing:2px;padding-top:22px;padding-bottom:5px;">Load UQ Results</div>
        <div>{file_input}</div>
        <div>{file_area}</div>
    </div>
    """)
        ])
    )
    return (file_input,)


@app.cell
def load_data(file_area, file_input):
    _path = Path(file_input.value)
    if not _path.exists():
        #
        #
        if file_area.value:
            data = json.loads(file_area.value[0].contents)
        else:
            mo.output.replace(mo.md(f"**File not found:** `{_path}`"))
            data = None
    else:
        with open(_path) as _f:
            data = json.load(_f)

    if file_area.value:
        data = json.loads(file_area.value[0].contents)
    return (data,)


@app.cell
def _():
    def header(text: str) -> mo.accordion:
        key = f"{mo.icon('streamline-stickies-color:online-information-duo', size=30)}"
        return mo.accordion({key: mo.md(text)})

    def render_info_panel(v, title):
        return mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    <div style="color:#888;font-size:12px;text-transform:uppercase;letter-spacing:2px;">{title}</div>
    <div style="display:flex;align-items:center;gap:16px;">
      <div>{v}</div>
    </div>
    </div>
    """)

    def render_panel(_header, value):
        return mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    <div style="padding-bottom:22px;"> {_header} </div>
    <div>{value}</div>
    </div>""")

    return header, render_panel


@app.cell
def param_selector(data, header):
    if data is None:
        mo.stop(True, mo.md("Waiting for data..."))

    _params = data["parameter_names"]
    param_dropdown = mo.ui.dropdown(
        options={p: p for p in _params},
        value=_params[0],
        label="",
    )

    _badges = " ".join(
        f'<span style="background:{PARAM_COLORS.get(p, "#666")};color:#000;'
        f"padding:2px 8px;border-radius:3px;font-size:11px;font-family:monospace;"
        f'font-weight:bold;margin-right:4px;">{p}</span>'
        for p in _params
    )

    _header = header(""" \
    ### 1. Parameter Selector
    >> A dropdown to pick one parameter, with color-coded badges for all parameters (cyan = vio_expression, amber = vio_trl_eff, pink =
    mecillinam_concentration).

    >> This is the master control. Every reactive panel below highlights the selected ("soloed") parameter and dims the others. Like soloing a
    channel in a DAW — you're isolating one parameter's contribution to hear it above the mix. A stakeholder can click through each parameter and instantly see
    how its influence changes across the cell cycle, without the other parameters' curves cluttering the view.
            """)
    mo.output.replace(
        mo.vstack([
            mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    <div style="padding-bottom:22px;"> {_header} </div>
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
def header_strip(data, header):
    if data is None:
        mo.stop(True)

    _surr = data.get("surrogates", {})
    _pop_r2 = _surr.get("population", {}).get("r_squared", "?")
    _cc_r2 = _surr.get("cell_cycle", {}).get("r_squared", "?")
    _n_stages = data.get("n_cell_cycle_stages", "?")
    _n_params = data.get("n_parameters", "?")
    _morris = data.get("morris_screening")
    _n_traj = _morris.get("n_trajectories", "?") if _morris else "?"
    _header = header(""" \
    ### 2. Header Strip (header_strip, line 122)

    #### What it shows:

    >> Five at-a-glance metrics — number of parameters, number of cell cycle stages, Phase 1 surrogate R^2, Phase 2 surrogate R^2, and Morris trajectory count.

    #### Why it's useful:

    >> These are the "trust indicators" for the entire analysis. R^2 tells you how well the PCE polynomial approximates the real simulation — if ```POP R^2 = 0.92```, the surrogate explains 92% of output variance, and the Sobol indices derived from it are reliable. If R^2 were 0.3, everything downstream would be suspect. Morris trajectory count tells you how well-sampled the prescreening was. A stakeholder can look at this strip and immediately know whether to trust the results below.
    """)
    mo.output.replace(
        mo.vstack([
            mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
        <div style="padding-bottom:22px;"> {_header} </div>

    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;padding:10px 16px;
            display:flex;gap:32px;align-items:center;font-family:monospace;font-size:12px;">

      <div style="color:{C["accent1"]};"><b>PARAMS</b> <span style="color:#e0e0e0;">{_n_params}</span></div>
      <div style="color:{C["accent3"]};"><b>STAGES</b> <span style="color:#e0e0e0;">{_n_stages}</span></div>
      <div style="color:{C["accent4"]};"><b>POP R\u00b2</b> <span style="color:#e0e0e0;">{_pop_r2}</span></div>
      <div style="color:{C["accent5"]};"><b>CC R\u00b2</b> <span style="color:#e0e0e0;">{_cc_r2}</span></div>
      <div style="color:{C["accent2"]};"><b>MORRIS r</b> <span style="color:#e0e0e0;">{_n_traj}</span></div>
      <div style="flex:1;"></div>
      <div style="color:#444;font-size:10px;letter-spacing:3px;">RFC006 // UQ DAW</div>
    </div>
    </div>
    """)
        ])
    )

    def render_header_strip():
        if data is None:
            mo.stop(True)

        _surr = data.get("surrogates", {})
        _pop_r2 = _surr.get("population", {}).get("r_squared", "?")
        _cc_r2 = _surr.get("cell_cycle", {}).get("r_squared", "?")
        _n_stages = data.get("n_cell_cycle_stages", "?")
        _n_params = data.get("n_parameters", "?")
        _morris = data.get("morris_screening")
        _n_traj = _morris.get("n_trajectories", "?") if _morris else "?"
        _header = header(""" \
        ### 2. Header Strip (header_strip, line 122)

        #### What it shows:

        >> Five at-a-glance metrics — number of parameters, number of cell cycle stages, Phase 1 surrogate R^2, Phase 2 surrogate R^2, and Morris trajectory count.

        #### Why it's useful:

        >> These are the "trust indicators" for the entire analysis. R^2 tells you how well the PCE polynomial approximates the real simulation — if ```POP R^2 = 0.92```, the surrogate explains 92% of output variance, and the Sobol indices derived from it are reliable. If R^2 were 0.3, everything downstream would be suspect. Morris trajectory count tells you how well-sampled the prescreening was. A stakeholder can look at this strip and immediately know whether to trust the results below.
        """)
        return mo.vstack([
            mo.md(f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
            <div style="padding-bottom:22px;"> {_header} </div>

        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;padding:10px 16px;
                display:flex;gap:32px;align-items:center;font-family:monospace;font-size:12px;">

          <div style="color:{C["accent1"]};"><b>PARAMS</b> <span style="color:#e0e0e0;">{_n_params}</span></div>
          <div style="color:{C["accent3"]};"><b>STAGES</b> <span style="color:#e0e0e0;">{_n_stages}</span></div>
          <div style="color:{C["accent4"]};"><b>POP R\u00b2</b> <span style="color:#e0e0e0;">{_pop_r2}</span></div>
          <div style="color:{C["accent5"]};"><b>CC R\u00b2</b> <span style="color:#e0e0e0;">{_cc_r2}</span></div>
          <div style="color:{C["accent2"]};"><b>MORRIS r</b> <span style="color:#e0e0e0;">{_n_traj}</span></div>
          <div style="flex:1;"></div>
          <div style="color:#444;font-size:10px;letter-spacing:3px;">RFC006 // UQ DAW</div>
        </div>
        </div>
        """)
        ])

    return


@app.cell
def eq_strip(data, header, param_dropdown):
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
        _fig.add_trace(
            go.Scatter(
                x=_x,
                y=_y,
                mode="lines",
                name=_p,
                line=dict(
                    color=PARAM_COLORS.get(_p, "#666"),
                    width=4 if _is_selected else 1.5,
                    shape="spline",
                ),
                opacity=1.0 if _is_selected else 0.25,
                hovertemplate=f"<b>{_p}</b><br>\u03b8=%{{x:.2f}}<br>S_Ti=%{{y:.3f}}<extra></extra>",
            )
        )

    # Fill under selected param
    _sel_y = [s["total_order"].get(_selected, 0) for s in _stages]
    _sel_color = PARAM_COLORS.get(_selected, C["accent1"])
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=_sel_y,
            fill="tozeroy",
            mode="none",
            # fillcolor=_sel_color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in _sel_color else _sel_color + "1a",
            fillcolor="#ff0000",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # First-order as dashed overlay for selected
    _sel_s1 = [s["first_order"].get(_selected, 0) for s in _stages]
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=_sel_s1,
            mode="lines",
            name=f"{_selected} (S\u1d62)",
            line=dict(color=_sel_color, width=2, dash="dot"),
            opacity=0.7,
            hovertemplate=f"<b>{_selected} (main effect)</b><br>\u03b8=%{{x:.2f}}<br>S_i=%{{y:.3f}}<extra></extra>",
        )
    )

    # Interaction band (S_Ti - S_i)
    _interaction = [t - s for t, s in zip(_sel_y, _sel_s1)]
    if max(_interaction) > 0.01:
        _fig.add_trace(
            go.Bar(
                x=_x,
                y=_interaction,
                name="Interactions",
                marker_color=_sel_color,
                opacity=0.15,
                width=0.08,
                hovertemplate="Interaction=%{y:.3f}<extra></extra>",
            )
        )

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
    _header = header(""" \
    ### 3. Parametric EQ (eq_strip, line 149)

    #### What it shows: Three overlapping spline curves, one per parameter, plotted across `θ` (cell cycle position, 0 to 1).

    >> The soloed parameter is thick with a filled area underneath; the others are ghosted.

    Two lines for the soloed param:
    - Solid line = S_Ti (total-order Sobol index — includes all interactions)
    - Dashed line = S_i (first-order — main effect only, no interactions)
    - Semi-transparent bars = the gap between them (S_Ti - S_i = interaction contribution)

    #### Why it's useful: This is the core visualization of the entire pipeline. It answers the central RFC006 question: "How does each parameter's importance change across the cell cycle?"

    In vecoli data, you can see vio_expression (cyan) starts high at θ=0 (birth, `S_Ti ≈ 0.64`) and drops to near zero at `θ=1` (pre-division). Mecillinam
    (pink) does the opposite — negligible early, dominant late. This tells a biologist: "vio pathway matters most during early growth; mecillinam's
    PBP2 inhibition matters most during septation/division." That's actionable — it tells you when in the cell cycle each drug/perturbation has its
    effect.

    The EQ analogy: each parameter is a frequency band. The curve shape shows where in the cell cycle "spectrum" that parameter's energy concentrates; Soloing isolates one band.

    #### Governing equations: Per-stage Sobol indices from PCE coefficients (Phase 2, Strategy 4):

    ```
    S_i^(k) = Var[E(Y_bar_k | X_i)] / Var(Y_bar_k)
            = sum_{alpha: alpha_i>0, alpha_j=0 for j!=i} c_alpha^2 / sum_{alpha!=0} c_alpha^2

    S_Ti^(k) = 1 - Var[E(Y_bar_k | X_~i)] / Var(Y_bar_k)
    ```

    ...where `Y_bar_k = mean(Y | theta in stage k)` is the stage-binned output and the sums run over PCE multi-indices alpha. `S_Ti - S_i` is the interaction contribution.

    _RFC006 call to action (§3)_: "A second type of analysis will need to be generated for the aggregation strategy (4), and will involve the definition of a low-dimensional (possibly scalar) 'cell cycle variable' computed from omics variables. This variable or 'coordinate' will be used for deterministically binning simulation data into cell stages, in order to then perform a 'phenotypic' sensitivity analysis across the physiological time dimension."
    """)
    mo.output.replace(
        mo.vstack([
            mo.md(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    <div style="padding-bottom:22px;"> {_header} </div>
    <div> {mo.ui.plotly(_fig)}</div>
    </div>""")
        ])
    )
    return


@app.cell
def channel_strip(data, header, param_dropdown, render_panel):
    """Channel strip: population fader + Morris meter + variance decomp pie."""
    if data is None:
        mo.stop(True)

    _selected = param_dropdown.value
    _params = data["parameter_names"]
    _sel_color = PARAM_COLORS.get(_selected, C["accent1"])

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Population Faders (S\u1d62 / S_Ti)",
            "Morris Screening (\u03bc* vs \u03c3)",
            "Variance Decomposition",
        ),
        column_widths=[0.35, 0.35, 0.3],
        horizontal_spacing=0.08,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]],
    )

    # -- Panel 1: Population Sobol as vertical faders --
    _p1 = data["phase1_population_sobol"]
    for _i, _p in enumerate(_params):
        _s1 = _p1["first_order"].get(_p, 0)
        _st = _p1["total_order"].get(_p, 0)
        _pc = PARAM_COLORS.get(_p, "#666")
        _alpha = 1.0 if _p == _selected else 0.4
        _fig.add_trace(
            go.Bar(
                x=[_p],
                y=[_st],
                name=f"{_p} S_Ti",
                marker_color=_pc,
                opacity=_alpha,
                width=0.35,
                showlegend=False,
                hovertemplate=f"<b>{_p}</b><br>S_Ti={_st:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        _fig.add_trace(
            go.Bar(
                x=[_p],
                y=[_s1],
                name=f"{_p} S\u1d62",
                marker_color=_pc,
                opacity=_alpha * 0.5,
                width=0.2,
                showlegend=False,
                hovertemplate=f"<b>{_p}</b><br>S_i={_s1:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

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
            _fig.add_trace(
                go.Scatter(
                    x=[_mu_star[_i]],
                    y=[_sigma[_i]],
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
                ),
                row=1,
                col=2,
            )
        # Classification line: sigma = 0.5 * mu*
        _mx = max(_mu_star) * 1.2
        _fig.add_trace(
            go.Scatter(
                x=[0, _mx],
                y=[0, 0.5 * _mx],
                mode="lines",
                line=dict(color="#444", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        _fig.add_annotation(
            x=_mx * 0.7,
            y=0.5 * _mx * 0.7 + 0.02,
            text="nonlinear/interactive",
            showarrow=False,
            font=dict(color="#555", size=9),
            xref="x2",
            yref="y2",
        )

    _fig.update_xaxes(title_text="\u03bc* (importance)", row=1, col=2)
    _fig.update_yaxes(title_text="\u03c3 (nonlinearity)", row=1, col=2)

    # -- Panel 3: Variance decomposition pie --
    _decomp = data.get("variance_decomposition", {})
    _gen_f = _decomp.get("generation_fraction", [0])[0]
    _seed_f = _decomp.get("seed_fraction", [0])[0]
    _resid_f = _decomp.get("residual_fraction", [1])[0]

    _fig.add_trace(
        go.Pie(
            labels=["Generation", "Seed", "Residual (cell cycle)"],
            values=[_gen_f, _seed_f, _resid_f],
            marker=dict(colors=[C["accent4"], C["accent5"], C["accent3"]]),
            textinfo="label+percent",
            textfont=dict(size=10),
            hole=0.45,
            hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        **DAW_LAYOUT,
        height=300,
        showlegend=False,
    )
    # Fix subplot title colors
    for _ann in _fig.layout.annotations:
        _ann.font = dict(color=C["text_dim"], size=11)
    _header = header(""" \
    Channel Strip (channel_strip, line 232)

      Three sub-panels side by side:

      Panel 1 — Population Faders (S_i / S_Ti): Vertical bars for each parameter's bulk Sobol indices
      (Phase 1). Wide bar = S_Ti, narrow inner bar = S_i. Soloed param is full opacity; others dimmed.

      Why it's useful: This is the "summary view" — one number per parameter for the entire population. It
      tells you the overall importance ranking without cell-cycle resolution. A stakeholder can see at a
      glance: "vio_expression and mecillinam are roughly equal in total importance (0.55 and 0.50),
      vio_trl_eff is negligible (0.08)." The gap between the wide and narrow bars is the interaction effect
       — if S_Ti is much larger than S_i, that parameter interacts with others.

      Governing equations: Population-level Sobol decomposition (Phase 1, Strategies 1-3):

    ```
      Var(Y) = sum_i V_i + sum_{i<j} V_ij + ... + V_{1,2,...,n}
      S_i  = V_i / Var(Y)
      S_Ti = 1 - Var[E(Y|X_~i)] / Var(Y)
      where Y = (1/N) sum Y_cell is the uniformly aggregated output (Strategy 1) and V_i = Var[E(Y|X_i)] is the variance due to parameter i alone.
    ```

    >> RFC006 call to action (§4, Activity 4): "Implement well established global sensitivity analysis
      methods based on the aggregation strategies (1-3). Expected to use PCE surrogate method for the
      stochastic function (sim_data -> SIM output)."

      ---
      Panel 2 — Morris Screening (mu vs sigma): Scatter plot of each parameter's Morris mu (x-axis,
      importance) vs sigma (y-axis, nonlinearity). A dashed diagonal line at sigma = 0.5 * mu* separates
      "linear effect" (below) from "nonlinear/interactive" (above).

      Why it's useful: Morris is the cheap prescreening step (O(n) evaluations vs O(n^2) for Sobol). It
      tells you which parameters to bother with before running the expensive PCE. Parameters in the
      lower-right (high mu*, low sigma) have strong, predictable effects. Parameters above the line have
      nonlinear or interactive effects — they may amplify or cancel depending on other parameter values. In
       your data, mecillinam has sigma = 0.22 vs mu* = 0.40, putting it closer to the "interactive" zone —
      consistent with the fact that mecillinam's cell wall effects interact with vio pathway resource
      competition.

      Governing equations: Morris Elementary Effects:

    ```
      EE_i(x) = [f(x + delta * e_i) - f(x)] / delta
      mu*_i = mean(|EE_i|)   (robust importance)
      sigma_i = std(EE_i)    (nonlinearity/interaction indicator)
      Cost = r x (n+1) evaluations, where r = n_trajectories, n = n_params.
    ```

    >> RFC006 call to action (§4, footnote): "using UQPy or PyTUQ libraries" — Morris is the prescreening
      method from these libraries that reduces n params to K params (K << n) before the expensive PCE
      fitting.

      ---
      Panel 3 — Variance Decomposition Pie: Donut chart showing how total output variance splits into
      generation effects, lineage seed effects, and residual (cell cycle + parameter sensitivity).

      Why it's useful: This answers "Where does the uncertainty come from?" If generation_fraction were
      large, it would mean the simulation hasn't converged — you'd need more generations. If seed_fraction
      were large, stochastic noise dominates and you'd need more lineage seeds. In your data, residual ≈
      99.99%, meaning almost all variance is from parameter sensitivity and cell cycle dynamics, not from
      convergence artifacts or random seed effects. That's exactly what you want to see — it means the
      Sobol indices are measuring real parameter effects, not noise.

      Governing equations: ANOVA-style variance decomposition across Strategies 1-3:

      ```
      Var(Y) = Var_gen + Var_seed + Var_residual
      gen_fraction  = Var_between_generations / Var_total
      seed_fraction = Var_between_seeds / Var_total
      residual      = 1 - gen_fraction - seed_fraction
      ```

    >> RFC006 call to action (§1): "This will enable us to deconvolve different types of uncertainty, and to model the relationship between 'bulk' and 'single-cell' attributes more accurately."

    """)
    mo.output.replace(mo.vstack([render_panel(_header, mo.ui.plotly(_fig))]))
    return


@app.cell
def heatmap_strip(data, header, param_dropdown, render_panel):
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
        rows=2,
        cols=1,
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
    _short_params = [
        p.replace("mecillinam_concentration", "mecillinam")
        .replace("vio_expression", "vio_exp")
        .replace("vio_trl_eff", "vio_trl")
        for p in _params
    ]

    # Highlight row for selected param
    _sel_idx = _params.index(_selected) if _selected in _params else 0

    _fig.add_trace(
        go.Heatmap(
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
        ),
        row=1,
        col=1,
    )

    # Selection indicator bracket on y-axis
    _fig.add_annotation(
        x=-0.5,
        y=_sel_idx,
        text="\u25b6",
        showarrow=False,
        font=dict(color=PARAM_COLORS.get(_selected, "#fff"), size=16),
        xref="x",
        yref="y",
    )

    # -- Panel 2: Cell cycle profile (mass + growth) --
    if _profile:
        _stages_x = _profile["stages"]
        _x_labels_p = [f"{i / len(_stages_x):.1f}-{(i + 1) / len(_stages_x):.1f}" for i in _stages_x]

        # Find mass and growth keys dynamically
        _mass_key = next((k for k in _profile if "mass" in k.lower() and "mean" in k.lower()), None)
        _growth_key = next((k for k in _profile if "growth" in k.lower() and "mean" in k.lower()), None)

        if _mass_key:
            _mass = _profile[_mass_key]
            _fig.add_trace(
                go.Scatter(
                    x=_x_labels_p,
                    y=_mass,
                    mode="lines+markers",
                    name="Mass",
                    line=dict(color=C["accent3"], width=3, shape="spline"),
                    marker=dict(size=6, color=C["accent3"]),
                    hovertemplate="Mass=%{y:.3f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
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

            _fig.add_trace(
                go.Scatter(
                    x=_x_labels_p,
                    y=_g_norm,
                    mode="lines+markers",
                    name="Growth Rate (scaled)",
                    line=dict(color=C["accent4"], width=2, dash="dash", shape="spline"),
                    marker=dict(size=5, color=C["accent4"], symbol="diamond"),
                    hovertemplate="Growth=%{customdata:.5f}<extra></extra>",
                    customdata=_growth,
                ),
                row=2,
                col=1,
            )

        # Add B/C/D period annotations
        _fig.add_vrect(
            x0=_x_labels_p[0],
            x1=_x_labels_p[1],
            fillcolor=C["accent1"],
            opacity=0.06,
            line_width=0,
            row=2,
            col=1,
            annotation_text="B",
            annotation_position="top left",
            annotation=dict(font=dict(color=C["accent1"], size=10)),
        )
        if len(_x_labels_p) > 7:
            _fig.add_vrect(
                x0=_x_labels_p[2],
                x1=_x_labels_p[6],
                fillcolor=C["accent5"],
                opacity=0.04,
                line_width=0,
                row=2,
                col=1,
                annotation_text="C",
                annotation_position="top left",
                annotation=dict(font=dict(color=C["accent5"], size=10)),
            )
            _fig.add_vrect(
                x0=_x_labels_p[7],
                x1=_x_labels_p[-1],
                fillcolor=C["accent2"],
                opacity=0.05,
                line_width=0,
                row=2,
                col=1,
                annotation_text="D",
                annotation_position="top left",
                annotation=dict(font=dict(color=C["accent2"], size=10)),
            )

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
    _header = header("""
    ### 5. Sensitivity Spectrogram + Cell Cycle Profile (heatmap_strip, line 349)

      Top panel — Sensitivity Spectrogram: A heatmap with parameters on the y-axis, θ-bins on the x-axis,
      and S_Ti as color intensity (dark = low, cyan/amber/pink = high). A ▶ arrow marks the soloed
      parameter's row.

      Why it's useful: This is the "bird's eye view" of Phase 2. Instead of reading 10 separate Sobol
      tables, you see the entire parameter-vs-stage landscape at once. Hot colors reveal where and which
      parameters drive variance. In your data, you'd see the upper-left corner lit up (vio_expression at
      early stages) and the lower-right lit up (mecillinam at late stages), with vio_trl_eff as a dim band
      across the middle. This pattern tells a biologist that the two perturbations affect the cell at
      different lifecycle phases — they're temporally orthogonal.

      The DAW analogy: this is the spectrogram view in an audio editor — frequency (parameters) vs time
      (cell cycle), with intensity as color.

      Bottom panel — Cell Cycle Profile: Two overlaid curves — dry mass (green, solid) and growth rate
      (amber, dashed, scaled to mass axis). Shaded vertical bands mark approximate B, C, and D periods.

      Why it's useful: This is the biological ground truth that the spectrogram above explains. Mass
      increases monotonically from ~1.03 to ~1.91 (the cell doubles before dividing). Growth rate peaks in
      early/mid cycle then dips during D-period as resources shift to division machinery. The B/C/D shading
       lets a biologist correlate the spectrogram patterns with known cell biology — "vio_expression
      dominates during B/C period (active growth), mecillinam dominates during D period (cell wall
      synthesis for septation)."

      Having the spectrogram directly above the profile with shared x-axes means you can visually trace:
      "stage 7 is where mass ≈ 1.65 and growth rate hits its minimum — and that's exactly where
      mecillinam's S_Ti peaks."

      Governing equations: Cell cycle variable θ and Strategy 4 binning:

    ```
      lambda = |lambda| * e^(i*omega)       (Koopman eigenvalue from DMD)

      theta(x) = arg(phi(x)) / 2*pi         (eigenfunction phase, theta in [0,1])

      Stage k: theta in [k/n_bins, (k+1)/n_bins)

      Y_bar_k = mean(Y | theta in stage k)  (per-stage aggregated output)

      Profile values: mass_mean_k = mean(M | theta in stage k), growth_mean_k = mean(dM/dt | theta in stage k).
    ```

    #### RFC006 call to action (§3):

    "The choice of the 'cell cycle variable' will be informed by the
      sensitivity analyses (1-3), will be explored through dedicated visualisations, and will be discussed
      with all subteams. An established example for such a variable is the 'cell angle', and in principle,
      any deterministic function of relevant process variables inside vEcoli may be considered if it has
      approximately cyclic behaviour."

    """)
    mo.output.replace(mo.vstack([render_panel(_header, mo.ui.plotly(_fig))]))
    return


@app.cell
def relevance_strip(data, header, render_panel):
    """Relevance meters + surrogate specs."""
    if data is None:
        mo.stop(True)

    _ccr = data.get("cell_cycle_relevance")
    _surr = data.get("surrogates", {})

    _fig = make_subplots(
        rows=1,
        cols=2,
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

        _fig.add_trace(
            go.Bar(
                x=_vals,
                y=_names,
                orientation="h",
                marker=dict(
                    color=[C["accent3"] if v > 0.8 else C["accent4"] if v > 0.5 else C["text_dim"] for v in _vals],
                ),
                text=[f"{v:.2f}" for v in _vals],
                textposition="outside",
                textfont=dict(color=C["text"], size=11),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    _fig.update_xaxes(title_text="Relevance Score", range=[0, 1.1], row=1, col=1)

    # Surrogate specs table
    _pop = _surr.get("population", {})
    _cc = _surr.get("cell_cycle", {})
    _fig.add_trace(
        go.Table(
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
                    [
                        _pop.get("basis_type", "?"),
                        _pop.get("polynomial_order", "?"),
                        _pop.get("input_dim", "?"),
                        _pop.get("output_dim", "?"),
                        f"{_pop.get('r_squared', 0):.3f}",
                        _pop.get("n_terms", "?"),
                    ],
                    [
                        _cc.get("basis_type", "?"),
                        _cc.get("polynomial_order", "?"),
                        _cc.get("input_dim", "?"),
                        _cc.get("output_dim", "?"),
                        f"{_cc.get('r_squared', 0):.3f}",
                        _cc.get("n_terms", "?"),
                    ],
                ],
                fill_color=C["bg"],
                font=dict(color=C["text"], size=11, family="monospace"),
                line=dict(color=C["panel_border"]),
                align="left",
                height=26,
            ),
        ),
        row=1,
        col=2,
    )

    _fig.update_layout(
        **DAW_LAYOUT,
        height=250,
    )
    for _ann in _fig.layout.annotations:
        _ann.font = dict(color=C["text_dim"], size=11)
    _header = header("""
    ### 6. Observable Relevance + Surrogate Specs (relevance_strip, line 475)

    #### Left panel — Observable Relevance:

    Horizontal bar chart showing which simulation observables were
      identified as cell-cycle-relevant by the GSA-informed selection (Step 5b). Color-coded by score
      (green > 0.8, amber > 0.5, grey otherwise).

    #### Why it's useful:

    This answers "Which outputs did we give to the Koopman DMD to define θ?" Not all
      simulation outputs carry cell-cycle information — some vary due to generation convergence or seed
      noise. The relevance score is the residual variance fraction (1 - gen_frac - seed_frac): observables
      with high residual are the ones whose variance is cell-cycle-driven. In your data, dry_mass (0.92)
      and growth (0.78) are both highly relevant. If an observable had a score of 0.02, it would mean
      generation/seed effects explain almost all its variance and it's useless for defining the cell cycle
      coordinate.

    #### Right panel — Surrogate Specs:

    A table comparing the two PCE surrogates (Population and Cell Cycle) —
       basis type, polynomial order, input/output dimensions, R^2, and number of terms.

    #### Why it's useful:

    ### This is the model card. A reviewer can check: "The population surrogate uses
      `Legendre basis order 3` with `R^2=0.92` — that's a good fit." If R^2 were low, you'd know to increase
      polynomial_order or n_samples. The output_dim difference (2 for population vs 10 for cell cycle)
      reflects that Phase 2 fits one surrogate across all 10 stages simultaneously.

    #### Governing equations:

    Observable relevance (GSA-informed selection, Step 5b):

    ```
      relevance_score_j = residual_j = 1 - gen_fraction_j - seed_fraction_j

      ...where j indexes observables.
    ```

    High residual means the observable's variance is NOT explained by
      generation or seed effects — it's cell-cycle or parameter-driven, making it informative input for
      Koopman DMD.

    #### PCE surrogate structure:

    ```
      f_hat(x) = sum_alpha c_alpha * prod_i P_{alpha_i}(x_i)

      ...where P_n are Legendre polynomials, |alpha| <= p (polynomial order), and the number of terms = C(n+p,
       p).
    ```

    >> RFC006 call to action (§3): "The choice of the 'cell cycle variable' will be informed by the
      sensitivity analyses (1-3)" — this is Step 5b, where Strategies 1-3 determine which observables are
      relevant for defining the cell cycle variable.


        """)
    mo.output.replace(mo.vstack([render_panel(_header, mo.ui.plotly(_fig))]))
    return


@app.cell
def stage_detail(data, header, param_dropdown, render_panel):
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
        _fig.add_trace(
            go.Bar(
                x=_x_stages,
                y=_st,
                name=f"{_p} S_Ti",
                marker_color=_pc,
                opacity=1.0 if _is_sel else 0.2,
                hovertemplate=f"<b>{_p}</b> S_Ti=%{{y:.3f}}<extra></extra>",
            )
        )

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
    _header = header("""
    ### 7. Mixer (stage_detail, line 552)

    #### What it shows:

    >> Grouped bar chart — one cluster per cell cycle stage, with one bar per parameter colored by its identity. The soloed parameter is full opacity; others are ghosted at 20%.

    #### Why it's useful:

    >> This is the discrete, stage-by-stage version of the EQ curve. While the EQ shows smooth splines (good for seeing trends), the mixer shows the exact S_Ti value at each stage as a bar you can compare across parameters. It answers "At stage 5 specifically, is mecillinam or vio_expression more important?" You can read the bar heights directly. The grouping makes cross-parameter comparison at a single stage easy, whereas the EQ makes cross-stage comparison for a single parameter easy — complementary views of the same data.

    #### Governing equation: Same per-stage total-order Sobol as the EQ, displayed differently:

    ```
      S_Ti^(k) for all params i, at each stage k
    ```

    >> The mixer transposes the EQ's perspective: EQ sweeps across `k` for fixed `i`, the mixer compares across `i` at each fixed `k`.

    #### RFC006 call to action (§4, Activity 5):

    >> "Apply the sensitivity analysis for aggregation strategies (1-3) for representative simulations from each use case. Report (slides) containing explanation of methods and results on representative simulations." — the mixer is the per-stage view that would go into such a report.
    """)
    mo.output.replace(mo.vstack([render_panel(_header, mo.ui.plotly(_fig))]))
    return


@app.cell
def interaction_analyzer(data, header, param_dropdown, render_panel):
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

        _fig.add_trace(
            go.Scatter(
                x=list(range(len(_stages))),
                y=_gaps,
                mode="lines+markers",
                name=_p,
                line=dict(color=_pc, width=3 if _is_sel else 1, shape="spline"),
                marker=dict(size=8 if _is_sel else 4),
                opacity=1.0 if _is_sel else 0.3,
                hovertemplate=f"<b>{_p}</b><br>Stage %{{x}}<br>Interaction=%{{y:.4f}}<extra></extra>",
            )
        )

    _fig.update_layout(
        **DAW_LAYOUT,
        height=220,
        title=dict(text="SIDECHAIN // Interaction Gap (S_Ti \u2212 S\u1d62)", font=dict(size=13)),
        xaxis_title="Cell Cycle Stage",
        yaxis_title="Interaction",
        legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=10)),
    )
    _header = header("""
    ### 8. Sidechain / Interaction Analyzer (interaction_analyzer, line 594)

    #### What it shows:

    Spline curves of (`S_Ti - S_i`) for each parameter across stages. This is the
    interaction gap — the fraction of variance that comes from joint effects between parameters rather
    than individual effects.

    #### Why it's useful:

    This is the most subtle and potentially most important panel. If the interaction gap
    is near zero, parameters act independently — you can study them in isolation. If it's large,
    parameters interact — changing vio_expression and mecillinam together produces effects that neither
    alone would predict.

    In your data, vio_expression has a larger interaction gap (~0.06) at early stages that shrinks to
    near zero at late stages. Mecillinam's gap grows from ~0.03 to ~0.07 across the cycle. This means: at
    early stages, vio interacts with other parameters (probably mecillinam — they compete for cellular
    resources during active growth). At late stages, mecillinam's interaction grows (PBP2 inhibition
    interacts with vio-induced metabolic load during septation).

    A biologist designing a combination treatment would look at this panel to know: "If I change both
    vio_expression and mecillinam_concentration simultaneously, stages 7-9 are where the interaction is
    strongest — that's where synergistic or antagonistic drug effects would manifest." That's not visible
     in the main-effect S_i alone.

    #### Governing equation: Interaction decomposition from the Sobol ANOVA:

    ```
      Interaction_i^(k) = S_Ti^(k) - S_i^(k) = sum_{j!=i} V_ij^(k) / Var(Y_bar_k) + higher-order terms
    ```

    >> This captures all variance involving parameter i jointly with at least one other parameter. It equals
       zero if and only if parameter i acts independently of all other parameters at stage k.

    #### RFC006 call to action (§3):

    >> "We will extend this by directly applying well established global sensitivity analysis methods for the stochastic function (sim_data -> SIM output), based on the aggregation strategies (1-3)." — Sobol total-order indices inherently decompose into main effects and  interactions; the sidechain panel makes that decomposition explicit and visible across the cell cycle.
    """)
    mo.output.replace(mo.vstack([render_panel(_header, mo.ui.plotly(_fig))]))
    return


@app.cell
def load_surrogates(data, file_input):
    """Load PCE surrogates from the export directory (sibling files to uq_results.json)."""
    if data is None:
        mo.stop(True)

    _export_dir = Path(file_input.value).parent
    _pop_dir = _export_dir / "population_surrogate"
    _cc_dir = _export_dir / "cell_cycle_surrogate"

    surr_data = {"available": False}

    if _pop_dir.exists() and _cc_dir.exists():
        try:
            surr_data = {
                "available": True,
                "pop_coeffs": np.load(_pop_dir / "coefficients.npy"),
                "pop_mi": np.load(_pop_dir / "multi_indices.npy"),
                "cc_coeffs": np.load(_cc_dir / "coefficients.npy"),
                "cc_mi": np.load(_cc_dir / "multi_indices.npy"),
            }
            # Bounds from surrogate metadata or from JSON
            if (_pop_dir / "input_bounds.npy").exists():
                surr_data["bounds"] = np.load(_pop_dir / "input_bounds.npy")
            else:
                surr_data["bounds"] = np.array([[0, 5], [0, 2], [0, 10]], dtype=float)
        except Exception as _e:
            surr_data = {"available": False, "error": str(_e)}
    return (surr_data,)


@app.cell
def _():
    return


@app.function
def get_sliders(_params, _bounds):
    return mo.ui.array([
        mo.ui.slider(
            orientation="vertical",
            start=float(_bounds[_i, 0]),
            stop=float(_bounds[_i, 1]),
            step=float((_bounds[_i, 1] - _bounds[_i, 0]) / 100),
            value=float((_bounds[_i, 0] + _bounds[_i, 1]) / 2),
            label=_p,
            show_value=True,
            full_width=True,
        )
        for _i, _p in enumerate(_params)
    ])


@app.cell
def pce_eq_sliders(data, surr_data):
    """Parameter sliders for the PCE prediction EQ."""
    if data is None or not surr_data.get("available"):
        mo.stop(True)

    _params = data["parameter_names"]
    _bounds = surr_data["bounds"]

    _sliders = get_sliders(_params, _bounds)
    param_sliders = _sliders

    # mo.output.replace(
    #     mo.vstack([
    #
    #         mo.md(f"""
    # <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;">
    #   <div style="color:{C['accent3']};font-size:12px;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">
    # PCE Surrogate Knobs
    #   </div>
    #   {mo.hstack([_sliders[_i] for _i in range(len(_sliders))])}
    # </div>
    # """),
    #     ])
    # )
    param_sliders = _sliders
    return (param_sliders,)


@app.cell
def pce_eq_plot(data, header, param_dropdown, param_sliders, surr_data):
    """PCE prediction EQ with dual heatmaps — matches tk dashboard functionality."""
    if data is None or not surr_data.get("available"):
        mo.stop(True)

    _n_stages = data.get("n_cell_cycle_stages", 10)
    _selected = param_dropdown.value
    _params = data["parameter_names"]
    _bounds = surr_data["bounds"]
    _stage_data = data.get("phase2_cell_cycle_sobol_per_stage", [])
    _profile = data.get("cell_cycle_profile")

    # Current parameter values from sliders
    _x = np.array([float(param_sliders[_i].value) for _i in range(len(_params))])
    _x_norm = 2.0 * (_x - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0

    # Direct Legendre evaluation
    def _legendre_eval(_x_n, _coeffs, _mi):
        _max_ord = int(_mi.max())
        _n_p = _mi.shape[1]
        _P = np.zeros((_max_ord + 1, _n_p))
        _P[0, :] = 1.0
        if _max_ord >= 1:
            _P[1, :] = _x_n
        for _nn in range(2, _max_ord + 1):
            _P[_nn, :] = ((2 * _nn - 1) * _x_n * _P[_nn - 1, :] - (_nn - 1) * _P[_nn - 2, :]) / _nn
        _result = 0.0
        for _t in range(len(_coeffs)):
            _term = _coeffs[_t]
            for _pp in range(_n_p):
                _term *= _P[_mi[_t, _pp], _pp]
            _result += _term
        return _result

    # Population surrogate prediction
    _pop_y = _legendre_eval(_x_norm, surr_data["pop_coeffs"], surr_data["pop_mi"])

    # Sweep all parameters + compute local sensitivity
    _n_sweep = 60
    _sweeps = {}
    _local_sens = {}
    _param_effects = {}
    for _pi, _p in enumerate(_params):
        _lo, _hi = float(_bounds[_pi, 0]), float(_bounds[_pi, 1])
        _sv = np.linspace(_lo, _hi, _n_sweep)
        _sy = []
        for _v in _sv:
            _x_sw = _x.copy()
            _x_sw[_pi] = _v
            _x_sw_n = 2.0 * (_x_sw - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0
            _sy.append(_legendre_eval(_x_sw_n, surr_data["pop_coeffs"], surr_data["pop_mi"]))
        _sweeps[_p] = (_sv, np.array(_sy))

        # Local sensitivity via finite difference
        _delta = (_hi - _lo) * 0.005
        _x_plus = _x.copy()
        _x_plus[_pi] = min(_x[_pi] + _delta, _hi)
        _x_minus = _x.copy()
        _x_minus[_pi] = max(_x[_pi] - _delta, _lo)
        _y_plus = _legendre_eval(
            2.0 * (_x_plus - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0,
            surr_data["pop_coeffs"],
            surr_data["pop_mi"],
        )
        _y_minus = _legendre_eval(
            2.0 * (_x_minus - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-12) - 1.0,
            surr_data["pop_coeffs"],
            surr_data["pop_mi"],
        )
        _local_sens[_p] = abs(_y_plus - _y_minus) / (2 * _delta + 1e-12)
        _param_effects[_p] = _local_sens[_p] * _x_norm[_pi]

    _sel_idx = _params.index(_selected) if _selected in _params else 0
    _sel_color = PARAM_COLORS.get(_selected, C["accent1"])

    # ========== FIGURE 1: Response Curves ==========
    _fig1 = go.Figure()
    for _pi, _p in enumerate(_params):
        _sv, _sy = _sweeps[_p]
        _pc = PARAM_COLORS.get(_p, "#666")
        _is_sel = _p == _selected
        _sv_norm = (_sv - _bounds[_pi, 0]) / (_bounds[_pi, 1] - _bounds[_pi, 0] + 1e-12)
        _fig1.add_trace(
            go.Scatter(
                x=_sv_norm,
                y=_sy,
                mode="lines",
                name=_p,
                line=dict(color=_pc, width=4 if _is_sel else 1.5, shape="spline"),
                opacity=1.0 if _is_sel else 0.25,
                customdata=np.column_stack([_sv, _sy]),
                hovertemplate=f"<b>{_p}</b><br>{_p}=%{{customdata[0]:.2f}}<br>Y\u0302=%{{customdata[1]:.4f}}<extra></extra>",
            )
        )
        _cur_norm = (_x[_pi] - _bounds[_pi, 0]) / (_bounds[_pi, 1] - _bounds[_pi, 0] + 1e-12)
        _fig1.add_trace(
            go.Scatter(
                x=[_cur_norm],
                y=[_pop_y],
                mode="markers",
                name=f"{_p} (current)" if _is_sel else "",
                showlegend=_is_sel,
                marker=dict(
                    size=14 if _is_sel else 8,
                    color=_pc,
                    symbol="diamond" if _is_sel else "circle",
                    opacity=1.0 if _is_sel else 0.35,
                    line=dict(width=2 if _is_sel else 0, color="#fff"),
                ),
                hovertemplate=f"<b>{_p}</b>={_x[_pi]:.2f}<br>Y\u0302={_pop_y:.4f}<extra></extra>",
            )
        )

    # Fill under soloed curve
    _sel_sv_norm = (_sweeps[_selected][0] - _bounds[_sel_idx, 0]) / (
        _bounds[_sel_idx, 1] - _bounds[_sel_idx, 0] + 1e-12
    )
    _fig1.add_trace(
        go.Scatter(
            x=_sel_sv_norm,
            y=_sweeps[_selected][1],
            fill="tozeroy",
            mode="none",
            fillcolor=f"rgba({int(_sel_color[1:3], 16)},{int(_sel_color[3:5], 16)},{int(_sel_color[5:7], 16)},0.1)",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    _fig1.update_layout(
        **DAW_LAYOUT,
        height=300,
        title=dict(text=f"PCE RESPONSE CURVES // Y\u0302 = {_pop_y:.4f}", font=dict(size=13)),
        xaxis_title="Normalized parameter value [0=min, 1=max]",
        yaxis_title="Y\u0302 (predicted output)",
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=10)),
    )

    # ========== FIGURE 2: Sensitivity Spectrogram + Per-Stage Prediction ==========
    _theta_labels = [f"\u03b8{s['stage']}" for s in _stage_data] if _stage_data else []

    _fig2 = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "Sensitivity Spectrogram (S_Ti) — variance fractions",
            "Per-Stage Prediction (Y\u0302) — physical units",
        ),
        vertical_spacing=0.15,
        shared_xaxes=True,
    )

    if _stage_data:
        # Heatmap: params x stages (S_Ti)
        _z_sens = [[s["total_order"].get(_p, 0) for s in _stage_data] for _p in _params]
        _short_params = [
            _p.replace("mecillinam_concentration", "mecillinam")
            .replace("vio_expression", "vio_exp")
            .replace("vio_trl_eff", "vio_trl")
            for _p in _params
        ]
        _fig2.add_trace(
            go.Heatmap(
                z=_z_sens,
                x=_theta_labels,
                y=_short_params,
                colorscale=[
                    [0, "#0d0d0d"],
                    [0.15, "#1a1a4e"],
                    [0.3, "#2a2a8e"],
                    [0.5, "#00b4d8"],
                    [0.7, "#00f0ff"],
                    [0.85, "#ffaa00"],
                    [1.0, "#ff3366"],
                ],
                colorbar=dict(title="S_Ti", len=0.4, y=0.82, thickness=10, x=1.02),
                zmin=0,
                zmax=0.7,
                hovertemplate="Param: %{y}<br>\u03b8: %{x}<br>S_Ti: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Per-stage prediction using Sobol weights
        _n_s = len(_stage_data)
        _stage_preds = np.zeros(_n_s)
        for _si in range(_n_s):
            _contrib = sum(_param_effects.get(_p, 0) * _stage_data[_si]["total_order"].get(_p, 0) for _p in _params)
            _stage_preds[_si] = _pop_y + _contrib

        _fig2.add_trace(
            go.Scatter(
                x=_theta_labels,
                y=_stage_preds,
                mode="lines+markers",
                name="Y\u0302 per stage",
                line=dict(color=C["accent3"], width=3, shape="spline"),
                marker=dict(size=7, color=C["accent3"]),
                hovertemplate="\u03b8%{x}<br>Y\u0302=%{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Highlight peak stage
        _peak = int(np.argmax(_stage_preds))
        _fig2.add_trace(
            go.Scatter(
                x=[_theta_labels[_peak]],
                y=[_stage_preds[_peak]],
                mode="markers+text",
                text=[f"{_stage_preds[_peak]:.3f}"],
                textposition="top center",
                textfont=dict(color=C["accent3"], size=10),
                marker=dict(size=12, color=C["accent3"], line=dict(width=2, color="#fff")),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    _fig2.update_layout(**DAW_LAYOUT, height=380, showlegend=False)
    _fig2.update_yaxes(title_text="Y\u0302", row=2, col=1, title_font=dict(color=C["accent3"]))
    for _ann in _fig2.layout.annotations:
        _ann.font = dict(color=C["text_dim"], size=11)

    # ========== FIGURE 3: Observable-Domain Heatmap ==========
    _fig3 = None
    if _profile and _stage_data:
        _obs_names = [_k for _k in _profile if _k != "stages"]
        _n_obs = len(_obs_names)
        _n_s = len(_profile.get("stages", []))

        if _n_obs > 0 and _n_s > 0:
            _baselines = np.array([_profile[_k][:_n_s] for _k in _obs_names], dtype=float)
            _modulated = _baselines.copy()

            for _si in range(min(_n_s, len(_stage_data))):
                _mod = sum(_param_effects.get(_p, 0) * _stage_data[_si]["total_order"].get(_p, 0) for _p in _params)
                for _oi in range(_n_obs):
                    _base_mag = abs(_baselines[_oi, _si]) + 1e-12
                    _modulated[_oi, _si] = _baselines[_oi, _si] * (1 + _mod / _base_mag)

            _obs_short = [_k.split("__")[-1] if "__" in _k else _k for _k in _obs_names]
            _stage_labels = [f"\u03b8{_si}" for _si in range(_n_s)]

            # Delta annotation text
            _delta_text = [["" for _ in range(_n_s)] for _ in range(_n_obs)]
            for _oi in range(_n_obs):
                for _si in range(_n_s):
                    _d = _modulated[_oi, _si] - _baselines[_oi, _si]
                    if abs(_d) > 1e-6:
                        _delta_text[_oi][_si] = f"{'+' if _d > 0 else ''}{_d:.3f}"

            _fig3 = go.Figure()
            _fig3.add_trace(
                go.Heatmap(
                    z=_modulated.tolist(),
                    x=_stage_labels,
                    y=_obs_short,
                    colorscale=[
                        [0, "#0d0d2e"],
                        [0.25, "#006490"],
                        [0.5, "#00b48c"],
                        [0.75, "#b4dc32"],
                        [1.0, "#ffff64"],
                    ],
                    colorbar=dict(title="Value", thickness=10, x=1.02),
                    customdata=np.array(_delta_text),
                    hovertemplate="Observable: %{y}<br>\u03b8: %{x}<br>Value: %{z:.4f}<br>\u0394: %{customdata}<extra></extra>",
                )
            )
            _fig3.update_layout(
                **DAW_LAYOUT,
                height=220,
                title=dict(text="OBSERVABLE DOMAIN // Y per stage (physical units)", font=dict(size=13)),
            )

    # ========== Local sensitivity readout ==========
    _sens_max = max(_local_sens.values()) if _local_sens else 1
    _sens_bars_html = (
        "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="color:{PARAM_COLORS.get(_p, "#666")};font-family:monospace;font-size:10px;width:80px;">{_p.split("_")[-1]}</span>'
            f'<div style="background:{PARAM_COLORS.get(_p, "#666")};height:8px;width:{max(2, int(120 * _local_sens[_p] / _sens_max))}px;border-radius:2px;"></div>'
            f'<span style="color:{C["text_dim"]};font-family:monospace;font-size:9px;">{_local_sens[_p]:.4f}</span>'
            f"</div>"
            for _p in _params
        )
        if _local_sens
        else ""
    )

    # ========== Assemble output ==========
    _header_text = header("""\
    ### 9. PCE Prediction EQ (Surrogate Knobs)

    >> Sliders control PCE surrogate evaluation. All response curves, the sensitivity spectrogram,
    the observable-domain heatmap, and local sensitivity bars update in lock-step.
    The observable heatmap shows per-stage values in physical units, modulated by slider-driven
    parameter effects weighted by per-stage Sobol indices — no synthetic data.

    ```
    Y_obs_k(x) = baseline_obs_k * (1 + sum_i [|dY/dx_i| * (x_i - mid) * S_Ti^(k)] / |baseline|)
    ```
    """)

    _panels = [
        mo.md(f"""<div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:12px;">
            <div>{_header_text}</div>
            <div style="display:flex;align-items:center;gap:16px;margin:8px 0;">
                <div style="color:{C["accent3"]};font-family:monospace;font-size:18px;font-weight:bold;">Y\u0302 = {_pop_y:.4f}</div>
                <div style="flex:1;border-top:1px solid {C["border"]};"></div>
                <div style="color:{C["text_dim"]};font-size:10px;">LOCAL |dY\u0302/dx|</div>
            </div>
            <div style="margin-bottom:8px;">{_sens_bars_html}</div>
            <div style="background:#1a1a2e;border:1px solid {C["border"]};border-radius:6px;padding:10px;margin-bottom:8px;">
                <div style="color:{C["accent3"]};font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">PCE Surrogate Knobs</div>
                {mo.hstack([param_sliders[_i] for _i in range(len(param_sliders))], justify="start")}
            </div>
        </div>"""),
        mo.ui.plotly(_fig1),
        mo.ui.plotly(_fig2),
    ]
    if _fig3 is not None:
        _panels.append(mo.ui.plotly(_fig3))

    mo.output.replace(mo.vstack(_panels))
    return


@app.cell
def footer():
    mo.output.replace(
        mo.md(f"""
    <div style="text-align:center;color:#333;font-size:10px;font-family:monospace;
            padding:12px;letter-spacing:2px;">
    {mo.icon("streamline-color:bacteria-virus-cells-biology", size=16)} RFC006 // MILESTONE 08.4.2 // UQ DAW {mo.icon("streamline-color:bacteria-virus-cells-biology", size=16)}
    </div>
    """)
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
