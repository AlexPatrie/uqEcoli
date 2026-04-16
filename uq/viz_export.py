"""Publication-ready figure exports from UQ results.

Generates print-quality PDFs and a LaTeX Sobol table from
``uq_results.json`` + surrogate artifacts.

Usage (via CLI):
    uv run uq export-figures --results-path ./uq_results

Requires ``kaleido`` for Plotly PDF export:
    uv pip install kaleido
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Lazy Plotly imports (only needed when functions are called) ──


def _load_results(results_dir: str | Path) -> tuple[dict, dict | None]:
    """Load uq_results.json and optional surrogate .npy files."""
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


def export_sobol_bar_chart(
    data: dict,
    output_path: str | Path,
    width: int = 900,
    height: int = 500,
) -> Path:
    """Grouped bar chart: S_Ti per strategy, all params → PDF."""
    import plotly.graph_objects as go

    output_path = Path(output_path)
    params = list(data["parameters"].keys())
    short = [p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac") for p in params]

    fig = go.Figure()

    # Strategy 1: population
    s1 = data["phase1_population"]["sobol_total_order"]
    fig.add_trace(go.Bar(name="S1 Population", x=short, y=[s1.get(p, 0) for p in params]))

    # Strategy 2: by generation (average across gens)
    s2_gens = data.get("strategy2_by_generation", {}).get("generations", [])
    if s2_gens:
        avg = {}
        for p in params:
            avg[p] = np.mean([g["sobol_total_order"].get(p, 0) for g in s2_gens])
        fig.add_trace(go.Bar(name="S2 By Gen (avg)", x=short, y=[avg[p] for p in params]))

    # Strategy 3: by seed (average across seeds)
    s3_seeds = data.get("strategy3_by_seed", {}).get("seeds", [])
    if s3_seeds:
        avg = {}
        for p in params:
            avg[p] = np.mean([s["sobol_total_order"].get(p, 0) for s in s3_seeds])
        fig.add_trace(go.Bar(name="S3 By Seed (avg)", x=short, y=[avg[p] for p in params]))

    # Strategy 4: growth-stratified (average across stages)
    s4_stages = data.get("phase2_growth_stratified", {}).get("stages", [])
    if s4_stages:
        avg = {}
        for p in params:
            avg[p] = np.mean([s["sobol_total_order"].get(p, 0) for s in s4_stages])
        fig.add_trace(go.Bar(name="S4 Growth (avg)", x=short, y=[avg[p] for p in params]))

    fig.update_layout(
        barmode="group",
        title="Total Sobol Indices (S_Ti) by Strategy",
        yaxis_title="S_Ti",
        xaxis_title="Parameter",
        font=dict(family="serif", size=12),
        width=width,
        height=height,
        margin=dict(l=60, r=20, t=50, b=80),
    )
    fig.write_image(str(output_path))
    return output_path


def export_spectrogram(
    data: dict,
    output_path: str | Path,
    width: int = 900,
    height: int = 400,
) -> Path:
    """Sensitivity spectrogram heatmap (params × growth stages) → PDF."""
    import plotly.graph_objects as go

    output_path = Path(output_path)
    params = list(data["parameters"].keys())
    short = [p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac") for p in params]
    stages = data.get("phase2_growth_stratified", {}).get("stages", [])
    if not stages:
        return output_path

    z = [[s["sobol_total_order"].get(p, 0) for s in stages] for p in params]
    x_labels = [f"θ{s['stage']}" for s in stages]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=short,
        colorscale="Viridis",
        colorbar=dict(title="S_Ti"),
    ))
    fig.update_layout(
        title="Sensitivity Spectrogram — S_Ti by Growth Stage",
        xaxis_title="Cell Cycle Stage (θ)",
        font=dict(family="serif", size=12),
        width=width,
        height=height,
        margin=dict(l=120, r=20, t=50, b=40),
    )
    fig.write_image(str(output_path))
    return output_path


def export_response_curves(
    data: dict,
    surr: dict,
    output_path: str | Path,
    width: int = 800,
    height: int = 400,
    n_sweep: int = 100,
) -> Path:
    """PCE response curves at midpoint → PDF."""
    import plotly.graph_objects as go

    output_path = Path(output_path)
    params = list(data["parameters"].keys())
    bounds = surr["bounds"]
    coeffs = surr["pop_coeffs"]
    mi = surr["pop_mi"]
    n_params = len(params)

    # Midpoint
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])

    def _legendre_eval(x_norm, c, m):
        max_ord = int(m.max()) if m.size else 0
        P = np.zeros((max_ord + 1, m.shape[1]))
        P[0, :] = 1.0
        if max_ord >= 1:
            P[1, :] = x_norm
        for n in range(2, max_ord + 1):
            P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
        result = 0.0
        for t in range(len(c)):
            term = float(c[t])
            for p in range(m.shape[1]):
                term *= P[m[t, p], p]
            result += term
        return result

    fig = go.Figure()
    for pi, pname in enumerate(params):
        lo, hi = float(bounds[pi, 0]), float(bounds[pi, 1])
        sv = np.linspace(lo, hi, n_sweep)
        sv_norm = (sv - lo) / (hi - lo + 1e-12)
        sy = []
        for v in sv:
            x = mid.copy()
            x[pi] = v
            xn = 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0
            sy.append(_legendre_eval(xn, coeffs, mi))

        short = pname.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry_mass_frac")
        fig.add_trace(go.Scatter(x=sv_norm, y=sy, mode="lines", name=short))

    fig.update_layout(
        title="PCE Response Curves (at midpoint)",
        xaxis_title="Normalized parameter [0=min, 1=max]",
        yaxis_title="Ŷ",
        font=dict(family="serif", size=12),
        width=width,
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.write_image(str(output_path))
    return output_path


def export_sobol_latex(
    data: dict,
    output_path: str | Path,
    top_k: int = 10,
) -> Path:
    """LaTeX table of top-K params per strategy → .tex file."""
    output_path = Path(output_path)
    params = list(data["parameters"].keys())
    s1 = data["phase1_population"]["sobol_total_order"]

    # Sort by S1 total order
    ranked = sorted(params, key=lambda p: -s1.get(p, 0))[:top_k]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Total Sobol indices ($S_{T_i}$) — top parameters by strategy.}",
        r"\label{tab:sobol}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Parameter & S1 (Pop) & S2 (Gen) & S3 (Seed) & S4 (Growth) \\",
        r"\midrule",
    ]

    s2_gens = data.get("strategy2_by_generation", {}).get("generations", [])
    s3_seeds = data.get("strategy3_by_seed", {}).get("seeds", [])
    s4_stages = data.get("phase2_growth_stratified", {}).get("stages", [])

    for p in ranked:
        short = p.replace("fraction_active_", "").replace("cell_dry_mass_fraction", "dry\\_mass\\_frac")
        s1_val = s1.get(p, 0)
        s2_val = np.mean([g["sobol_total_order"].get(p, 0) for g in s2_gens]) if s2_gens else float("nan")
        s3_val = np.mean([s["sobol_total_order"].get(p, 0) for s in s3_seeds]) if s3_seeds else float("nan")
        s4_val = np.mean([s["sobol_total_order"].get(p, 0) for s in s4_stages]) if s4_stages else float("nan")

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if np.isfinite(v) else "---"

        lines.append(
            f"  \\texttt{{{short}}} & {_fmt(s1_val)} & {_fmt(s2_val)} & {_fmt(s3_val)} & {_fmt(s4_val)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines))
    return output_path


def export_all_figures(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all publication figures from a UQ export directory.

    Returns list of generated file paths.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, surr = _load_results(results_dir)
    generated: list[Path] = []

    generated.append(export_sobol_bar_chart(data, output_dir / "sobol_bar_chart.pdf"))
    generated.append(export_spectrogram(data, output_dir / "spectrogram.pdf"))

    if surr is not None:
        generated.append(export_response_curves(data, surr, output_dir / "response_curves.pdf"))

    generated.append(export_sobol_latex(data, output_dir / "sobol_table.tex"))

    return generated
