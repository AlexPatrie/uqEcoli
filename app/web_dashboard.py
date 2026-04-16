"""Standalone web dashboard for UQ results.

Serves a Plotly Dash application that provides interactive
parameter exploration from ``uq_results.json`` + surrogate artifacts.
Shareable via URL — no Python environment needed on the viewer side.

Launch:
    uv run uq dashboard --run-mode web
    uv run python app/web_dashboard.py [path/to/uq_results]

Requires:
    uv pip install dash
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ── Legendre PCE evaluation (self-contained) ──


def legendre_eval(x_norm, coeffs, mi):
    """Evaluate PCE: sum c_alpha * prod P_{alpha_i}(x_i)."""
    max_ord = int(mi.max()) if mi.size else 0
    n_p = mi.shape[1]
    P = np.zeros((max_ord + 1, n_p))
    P[0, :] = 1.0
    if max_ord >= 1:
        P[1, :] = x_norm
    for n in range(2, max_ord + 1):
        P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
    result = 0.0
    for t in range(len(coeffs)):
        term = float(coeffs[t])
        for p in range(n_p):
            term *= P[mi[t, p], p]
        result += term
    return result


def normalize_to_germ(x, bounds):
    return 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0


# ── Load data ──


def load_results(results_dir: str | Path):
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "uq_results.json").read_text())

    surr = None
    pop_dir = results_dir / "population_surrogate"
    if (pop_dir / "coefficients.npy").exists():
        surr = {
            "pop_coeffs": np.load(pop_dir / "coefficients.npy"),
            "pop_mi": np.load(pop_dir / "multi_indices.npy"),
            "bounds": np.load(pop_dir / "input_bounds.npy"),
        }
    return data, surr


# ── Dash app factory ──


def create_app(results_dir: str | Path):
    """Create and return a Dash app with interactive UQ panels."""
    import dash
    from dash import Input, Output, dcc, html

    import plotly.graph_objects as go

    data, surr = load_results(results_dir)
    params = list(data["parameters"].keys())
    bounds = surr["bounds"] if surr else np.zeros((len(params), 2))
    coeffs = surr["pop_coeffs"] if surr else np.zeros(1)
    mi = surr["pop_mi"] if surr else np.zeros((1, len(params)), dtype=int)

    app = dash.Dash(__name__, title="UQ Sensitivity Explorer")

    # Build slider components
    sliders = []
    for i, p in enumerate(params):
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        short = p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
        sliders.append(html.Div([
            html.Label(short, style={"color": "#00f0ff", "fontFamily": "monospace", "fontSize": "12px"}),
            dcc.Slider(
                id=f"slider-{i}",
                min=lo, max=hi,
                value=(lo + hi) / 2,
                step=(hi - lo) / 200,
                marks={lo: f"{lo:.3f}", hi: f"{hi:.3f}"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"marginBottom": "10px"}))

    # Spectrogram data
    stages = data.get("phase2_growth_stratified", {}).get("stages", [])
    z_sens = [[s["sobol_total_order"].get(p, 0) for s in stages] for p in params]
    x_labels = [f"θ{s['stage']}" for s in stages]
    y_labels = [p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass") for p in params]

    spectrogram_fig = go.Figure(data=go.Heatmap(
        z=z_sens, x=x_labels, y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title="S_Ti"),
    ))
    spectrogram_fig.update_layout(
        title="Sensitivity Spectrogram — S_Ti by Growth Stage",
        paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
        font=dict(family="monospace", color="#e0e0e0", size=11),
        height=350, margin=dict(l=120, r=20, t=40, b=40),
    )

    # Sobol bar chart
    s1 = data["phase1_population"]["sobol_total_order"]
    sobol_fig = go.Figure(data=go.Bar(
        x=y_labels,
        y=[s1.get(p, 0) for p in params],
        marker_color="#00f0ff",
    ))
    sobol_fig.update_layout(
        title="Population Sobol Indices (S_Ti)",
        paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
        font=dict(family="monospace", color="#e0e0e0", size=11),
        height=300, margin=dict(l=60, r=20, t=40, b=60),
    )

    app.layout = html.Div(style={
        "backgroundColor": "#0a0a0f", "color": "#e0e0e0",
        "fontFamily": "monospace", "padding": "20px",
        "minHeight": "100vh",
    }, children=[
        html.H1("UQ Sensitivity Explorer", style={"color": "#00f0ff", "textAlign": "center"}),
        html.P(
            f"Parameters: {len(params)} | Growth stages: {len(stages)} | Surrogate: PCE (Legendre)",
            style={"textAlign": "center", "color": "#888"},
        ),
        html.Hr(style={"borderColor": "#2a2a4a"}),
        html.Div(style={"display": "flex", "gap": "30px"}, children=[
            # Left: sliders
            html.Div(style={"flex": "1", "minWidth": "250px"}, children=[
                html.H3("Parameter Controls", style={"color": "#00f0ff"}),
                html.Div(id="prediction-readout", style={
                    "fontSize": "20px", "fontWeight": "bold", "color": "#33ff99",
                    "marginBottom": "15px",
                }),
                *sliders,
            ]),
            # Right: plots
            html.Div(style={"flex": "3"}, children=[
                dcc.Graph(id="response-curves", style={"height": "350px"}),
                dcc.Graph(figure=spectrogram_fig),
                dcc.Graph(figure=sobol_fig),
            ]),
        ]),
    ])

    # Callback: update response curves when sliders change
    slider_inputs = [Input(f"slider-{i}", "value") for i in range(len(params))]

    @app.callback(
        [Output("response-curves", "figure"), Output("prediction-readout", "children")],
        slider_inputs,
    )
    def update_response(*slider_values):
        x = np.array([float(v) for v in slider_values])
        x_norm = normalize_to_germ(x, bounds)
        y_hat = legendre_eval(x_norm, coeffs, mi)

        fig = go.Figure()
        n_sweep = 80
        pcolors = ["#00f0ff", "#ff3366", "#00ff88", "#ffaa00", "#aa66ff", "#ff6600"]

        for pi, p in enumerate(params):
            lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
            sv = np.linspace(lo, hi, n_sweep)
            sv_norm = (sv - lo) / (hi - lo + 1e-12)
            sy = []
            for v in sv:
                x_sw = x.copy()
                x_sw[pi] = v
                x_sw_n = normalize_to_germ(x_sw, bounds)
                sy.append(legendre_eval(x_sw_n, coeffs, mi))

            short = p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
            color = pcolors[pi % len(pcolors)]
            fig.add_trace(go.Scatter(
                x=sv_norm, y=sy, mode="lines", name=short,
                line=dict(color=color, width=2.5),
            ))
            cur_norm = (x[pi] - lo) / (hi - lo + 1e-12)
            fig.add_trace(go.Scatter(
                x=[cur_norm], y=[y_hat], mode="markers", showlegend=False,
                marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color="#fff")),
            ))

        fig.update_layout(
            title=f"PCE Response Curves — Ŷ = {y_hat:.4f}",
            paper_bgcolor="#0a0a0f", plot_bgcolor="#12121a",
            font=dict(family="monospace", color="#e0e0e0", size=11),
            xaxis_title="Normalized parameter [0=min, 1=max]",
            yaxis_title="Ŷ",
            height=350, margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(gridcolor="#1a1a2e"), yaxis=dict(gridcolor="#1a1a2e"),
        )

        return fig, f"Ŷ = {y_hat:.4f}"

    return app


def run_web_dashboard(results_path: str | None = None, port: int = 8050) -> None:
    """Launch the web dashboard server."""
    if results_path is None:
        results_path = "./uq_results"
    app = create_app(results_path)
    print(f"Starting UQ dashboard at http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    run_web_dashboard(_path)
