"""Self-contained HTML report from UQ results.

Reads ``uq_results.json`` and ``manifest.json`` from an export directory
and produces a single, dependency-free HTML file with inline SVG charts
and modern styling.  All four RFC006 aggregation strategies are visualized
so biologists can compare parameter sensitivity at the population,
generation, lineage, and cell-cycle levels.

Usage (via CLI):
    uv run uq report --results-path ./uq_results
    uv run uq report --results-path ./uq_results --output report.html
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ── Color palette ──

_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#84cc16",  # lime
]

_VIRIDIS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
]


def _viridis(t: float) -> str:
    """Map *t* in [0, 1] to a Viridis hex color."""
    t = max(0.0, min(1.0, t))
    idx = t * (len(_VIRIDIS) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(_VIRIDIS) - 1)
    frac = idx - lo
    c1, c2 = _VIRIDIS[lo], _VIRIDIS[hi]
    r = int(int(c1[1:3], 16) * (1 - frac) + int(c2[1:3], 16) * frac)
    g = int(int(c1[3:5], 16) * (1 - frac) + int(c2[3:5], 16) * frac)
    b = int(int(c1[5:7], 16) * (1 - frac) + int(c2[5:7], 16) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def _short(name: str) -> str:
    """Abbreviate parameter/observable names for display."""
    return (
        name.replace("fraction_active_", "")
        .replace("cell_dry_mass_fraction", "dry_mass_frac")
        .replace("listeners__mass__", "")
        .replace("instantaneous_growth_rate", "growth_rate")
    )


def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


# ── Generic SVG builders ──


def _svg_bar_chart(
    total: dict[str, float],
    first: dict[str, float] | None = None,
    *,
    width: int = 700,
    height: int = 320,
    color_total: str = _COLORS[0],
    color_first: str = _COLORS[1],
) -> str:
    """Grouped bar chart of Sobol indices as inline SVG.

    *total* and *first* are ``{param_name: value}`` dicts.
    """
    params = list(total.keys())
    n = len(params)
    if n == 0:
        return ""
    has_first = first is not None and bool(first)

    vals_t = [total.get(p, 0) for p in params]
    vals_f = [first.get(p, 0) for p in params] if has_first else []
    max_val = max(max(vals_t), max(vals_f) if vals_f else 0, 0.01)
    max_val = math.ceil(max_val * 10) / 10

    ml, mr, mt, mb = 55, 20, 30, 90
    pw = width - ml - mr
    ph = height - mt - mb
    gw = pw / n
    bw = gw * 0.35 if has_first else gw * 0.6
    gap = gw * 0.05

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'style="width:100%;max-width:{width}px;height:auto;">']

    # Grid
    for i in range(6):
        yv = max_val * i / 5
        yp = mt + ph - (yv / max_val) * ph
        L.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{width - mr}" y2="{yp:.1f}" '
                 f'stroke="#333355" stroke-width="1"/>')
        L.append(f'<text x="{ml - 8}" y="{yp + 4:.1f}" text-anchor="end" '
                 f'font-size="11" fill="#94a3b8">{yv:.2f}</text>')

    # Bars
    for i, p in enumerate(params):
        xb = ml + i * gw + gap
        ht = (vals_t[i] / max_val) * ph
        yt = mt + ph - ht
        xt = xb + (0 if has_first else (gw - bw) / 2 - gap)
        L.append(f'<rect x="{xt:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{ht:.1f}" '
                 f'fill="{color_total}" rx="3"><title>{p}\nS_Ti = {vals_t[i]:.4f}</title></rect>')
        if has_first:
            hf = (vals_f[i] / max_val) * ph
            yf = mt + ph - hf
            xf = xb + bw + gap
            L.append(f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bw:.1f}" height="{hf:.1f}" '
                     f'fill="{color_first}" rx="3"><title>{p}\nS_i = {vals_f[i]:.4f}</title></rect>')
        # X label
        xc = ml + i * gw + gw / 2
        L.append(f'<text x="{xc:.1f}" y="{mt + ph + 16}" text-anchor="end" font-size="11" '
                 f'fill="#e2e8f0" transform="rotate(-35 {xc:.1f} {mt + ph + 16})">'
                 f'{html.escape(_short(p))}</text>')

    # Legend
    lx, ly = ml + 10, mt + 8
    L.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color_total}" rx="2"/>')
    L.append(f'<text x="{lx + 16}" y="{ly + 10}" font-size="11" fill="#e2e8f0">Total (S_Ti)</text>')
    if has_first:
        L.append(f'<rect x="{lx + 100}" y="{ly}" width="12" height="12" fill="{color_first}" rx="2"/>')
        L.append(f'<text x="{lx + 116}" y="{ly + 10}" font-size="11" fill="#e2e8f0">First (S_i)</text>')

    L.append('</svg>')
    return '\n'.join(L)


def _svg_heatmap(
    params: list[str],
    groups: list[dict[str, float]],
    group_labels: list[str],
    *,
    width: int = 700,
    height: int = 300,
) -> str:
    """Generic heatmap: rows = params, columns = groups."""
    n_p = len(params)
    n_g = len(groups)
    if n_g == 0 or n_p == 0:
        return '<p style="color:#94a3b8;">No data available.</p>'

    ml, mr, mt, mb = 130, 70, 30, 50
    pw = width - ml - mr
    ph = height - mt - mb
    cw = pw / n_g
    ch = ph / n_p

    all_v = [g.get(p, 0) for p in params for g in groups]
    vmin, vmax = min(all_v), max(all_v)
    if vmax <= vmin:
        vmax = vmin + 0.01

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'style="width:100%;max-width:{width}px;height:auto;">']

    for pi, p in enumerate(params):
        yc = mt + pi * ch + ch / 2
        L.append(f'<text x="{ml - 8}" y="{yc + 4}" text-anchor="end" font-size="11" '
                 f'fill="#e2e8f0">{html.escape(_short(p))}</text>')
        for gi, g in enumerate(groups):
            val = g.get(p, 0)
            t = (val - vmin) / (vmax - vmin)
            col = _viridis(t)
            x = ml + gi * cw
            y = mt + pi * ch
            L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                     f'fill="{col}"><title>{_short(p)} @ {group_labels[gi]}: {val:.4f}</title></rect>')

    # X labels
    for gi, lbl in enumerate(group_labels):
        xc = ml + gi * cw + cw / 2
        L.append(f'<text x="{xc:.1f}" y="{mt + ph + 16}" text-anchor="middle" font-size="10" '
                 f'fill="#94a3b8">{html.escape(lbl)}</text>')

    # Color bar
    cb_x = width - mr + 15
    cb_w, cb_h = 14, ph
    for i in range(20):
        t = i / 19
        cy = mt + cb_h - t * cb_h
        L.append(f'<rect x="{cb_x}" y="{cy:.1f}" width="{cb_w}" height="{cb_h / 19 + 1:.1f}" '
                 f'fill="{_viridis(t)}"/>')
    L.append(f'<text x="{cb_x + cb_w + 4}" y="{mt + 10}" font-size="10" fill="#94a3b8">{vmax:.3f}</text>')
    L.append(f'<text x="{cb_x + cb_w + 4}" y="{mt + cb_h}" font-size="10" fill="#94a3b8">{vmin:.3f}</text>')
    L.append(f'<text x="{cb_x + cb_w / 2}" y="{mt - 10}" text-anchor="middle" font-size="10" '
             f'fill="#94a3b8">S_Ti</text>')

    L.append('</svg>')
    return '\n'.join(L)


def _svg_cross_strategy_bars(
    params: list[str],
    strategy_data: list[tuple[str, dict[str, float], str]],
    *,
    width: int = 700,
    height: int = 360,
) -> str:
    """Grouped bar chart comparing S_Ti across strategies.

    *strategy_data* is a list of ``(label, {param: value}, color)``.
    """
    n_p = len(params)
    n_s = len(strategy_data)
    if n_p == 0 or n_s == 0:
        return ""

    all_v = [d.get(p, 0) for _, d, _ in strategy_data for p in params]
    max_val = max(max(all_v), 0.01)
    max_val = math.ceil(max_val * 10) / 10

    ml, mr, mt, mb = 55, 20, 45, 95
    pw = width - ml - mr
    ph = height - mt - mb
    gw = pw / n_p
    bw = gw * 0.8 / n_s
    gap = gw * 0.1 / (n_s + 1)

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'style="width:100%;max-width:{width}px;height:auto;">']

    # Grid
    for i in range(6):
        yv = max_val * i / 5
        yp = mt + ph - (yv / max_val) * ph
        L.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{width - mr}" y2="{yp:.1f}" '
                 f'stroke="#333355" stroke-width="1"/>')
        L.append(f'<text x="{ml - 8}" y="{yp + 4:.1f}" text-anchor="end" '
                 f'font-size="11" fill="#94a3b8">{yv:.2f}</text>')

    # Grouped bars
    for pi, p in enumerate(params):
        for si, (lbl, vals, col) in enumerate(strategy_data):
            v = vals.get(p, 0)
            h = (v / max_val) * ph
            x = ml + pi * gw + gap + si * (bw + gap)
            y = mt + ph - h
            L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'fill="{col}" rx="2"><title>{lbl}: {_short(p)} = {v:.4f}</title></rect>')

        xc = ml + pi * gw + gw / 2
        L.append(f'<text x="{xc:.1f}" y="{mt + ph + 16}" text-anchor="end" font-size="11" '
                 f'fill="#e2e8f0" transform="rotate(-35 {xc:.1f} {mt + ph + 16})">'
                 f'{html.escape(_short(p))}</text>')

    # Legend
    lx, ly = ml + 10, mt - 28
    for si, (lbl, _, col) in enumerate(strategy_data):
        ox = lx + si * 140
        L.append(f'<rect x="{ox}" y="{ly}" width="12" height="12" fill="{col}" rx="2"/>')
        L.append(f'<text x="{ox + 16}" y="{ly + 10}" font-size="11" fill="#e2e8f0">{html.escape(lbl)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


# ── Table builders ──


def _ranking_table(
    total: dict[str, float],
    first: dict[str, float] | None = None,
) -> str:
    """Parameter ranking table with inline bar + interaction column."""
    has_first = first is not None and bool(first)
    rows = []
    for p, t in total.items():
        f = first.get(p, 0) if has_first else 0.0
        inter = t - f if has_first else float("nan")
        rows.append((p, t, f, inter))
    rows.sort(key=lambda r: -r[1])

    L = ['<table>', '<thead><tr>',
         '<th>Rank</th><th>Parameter</th><th>S_Ti (Total)</th>']
    if has_first:
        L.append('<th>S_i (First)</th><th>Interaction</th>')
    L.extend(['</tr></thead>', '<tbody>'])

    for rank, (p, t, f, inter) in enumerate(rows, 1):
        pct = min(t * 100, 100)
        bar = f'<div class="bar" style="width:{pct:.0f}%"></div>'
        L.append(f'<tr><td class="rank">{rank}</td>'
                 f'<td><code>{html.escape(_short(p))}</code></td>'
                 f'<td><div class="bar-cell">{bar}<span>{_fmt(t)}</span></div></td>')
        if has_first:
            inter_s = _fmt(inter) if math.isfinite(inter) else "-"
            L.append(f'<td>{_fmt(f)}</td><td>{inter_s}</td>')
        L.append('</tr>')

    L.extend(['</tbody>', '</table>'])
    return '\n'.join(L)


def _evolution_table(
    params: list[str],
    groups: list[dict[str, float]],
    group_labels: list[str],
) -> str:
    """Color-coded table showing S_Ti across groups with trend arrows."""
    if not groups:
        return '<p style="color:#94a3b8;">No data available.</p>'

    L = ['<table class="stage-table">', '<thead><tr><th>Parameter</th>']
    for lbl in group_labels:
        L.append(f'<th>{html.escape(lbl)}</th>')
    L.append('<th>Trend</th></tr></thead><tbody>')

    for p in params:
        vals = [g.get(p, 0) for g in groups]
        vmin = min(vals)
        vmax = max(vals) if max(vals) > vmin else vmin + 0.01
        trend = vals[-1] - vals[0]
        arrow = "&#x2197;" if trend > 0.01 else ("&#x2198;" if trend < -0.01 else "&#x2192;")
        cls = "trend-up" if trend > 0.01 else ("trend-down" if trend < -0.01 else "trend-flat")

        L.append(f'<tr><td><code>{html.escape(_short(p))}</code></td>')
        for v in vals:
            t = (v - vmin) / (vmax - vmin)
            bg = _viridis(t)
            fg = "#fff" if t < 0.6 else "#1a1a2e"
            L.append(f'<td style="background:{bg};color:{fg};text-align:center">{v:.3f}</td>')
        L.append(f'<td class="{cls}">{arrow} {trend:+.3f}</td></tr>')

    L.extend(['</tbody></table>'])
    return '\n'.join(L)


def _cross_strategy_insight(
    params: list[str],
    s1_total: dict[str, float],
    s2_avg: dict[str, float] | None,
    s3_avg: dict[str, float] | None,
) -> str:
    """Generate a callout when top-ranked parameters differ across strategies."""
    top1 = max(s1_total, key=s1_total.get) if s1_total else None

    findings = []
    if s2_avg:
        top2 = max(s2_avg, key=s2_avg.get)
        if top2 != top1:
            findings.append(
                f'At the <strong>generation level</strong>, '
                f'<code>{html.escape(_short(top2))}</code> '
                f'(S_Ti = {_fmt(s2_avg[top2])}) overtakes '
                f'<code>{html.escape(_short(top1))}</code> '
                f'(S_Ti = {_fmt(s2_avg.get(top1, 0))}) as the top driver.'
            )
    if s3_avg:
        top3 = max(s3_avg, key=s3_avg.get)
        if top3 != top1:
            findings.append(
                f'At the <strong>lineage level</strong>, '
                f'<code>{html.escape(_short(top3))}</code> '
                f'(S_Ti = {_fmt(s3_avg[top3])}) dominates instead of '
                f'<code>{html.escape(_short(top1))}</code> '
                f'(S_Ti = {_fmt(s3_avg.get(top1, 0))}).'
            )

    if not findings:
        return (
            '<div class="callout callout-ok">'
            '<div class="icon">&#x2714;&#xFE0F;</div>'
            '<div class="detail">'
            '<div class="title">Consistent Rankings</div>'
            '<div class="desc">The top driver is the same across all aggregation strategies '
            '&mdash; parameter importance is robust to how cells are grouped.</div>'
            '</div></div>'
        )

    return (
        '<div class="callout callout-warn">'
        '<div class="icon">&#x26A0;&#xFE0F;</div>'
        '<div class="detail">'
        '<div class="title">Strategy-Dependent Rankings</div>'
        '<div class="desc">' + ' '.join(findings) +
        ' The population-level top driver '
        f'(<code>{html.escape(_short(top1))}</code>) does not dominate under '
        'all aggregation strategies &mdash; consider these additional parameters.</div>'
        '</div></div>'
    )


# ── Surrogate data loading ──


def _load_surrogate_json(results_dir: Path) -> dict[str, Any] | None:
    """Load PCE surrogate .npy files and return JSON-serializable dict.

    Returns None if the required files are missing.
    """
    pop_dir = results_dir / "population_surrogate"
    gs_dir = results_dir / "growth_stratified_surrogate"

    required = [
        pop_dir / "coefficients_per_output.npy",
        pop_dir / "multi_indices.npy",
        pop_dir / "input_bounds.npy",
    ]
    if not all(f.exists() for f in required):
        return None

    pop_cpo = np.load(pop_dir / "coefficients_per_output.npy")
    mi = np.load(pop_dir / "multi_indices.npy")
    bounds = np.load(pop_dir / "input_bounds.npy")

    surr: dict[str, Any] = {
        "pop_coeffs": pop_cpo.tolist(),
        "multi_indices": mi.tolist(),
        "bounds": bounds.tolist(),
    }

    gs_cpo_path = gs_dir / "coefficients_per_output.npy"
    if gs_cpo_path.exists():
        surr["gs_coeffs"] = np.load(gs_cpo_path).tolist()

    return surr


def _interactive_css() -> str:
    """CSS for the interactive PCE explorer section."""
    # Use raw string to avoid f-string brace issues
    return """
/* Interactive explorer */
.explorer-grid {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 24px;
  margin-top: 16px;
}
@media (max-width: 700px) {
  .explorer-grid { grid-template-columns: 1fr; }
}
.slider-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.slider-group label {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 4px;
  color: var(--text-dim);
}
.slider-group label code {
  color: var(--text);
}
.slider-group input[type=range] {
  width: 100%;
  accent-color: var(--accent);
  height: 6px;
  cursor: pointer;
}
.slider-group .bounds {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--text-dim);
  margin-top: 2px;
}
.pred-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.pred-readout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
}
.pred-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.pred-card .pred-label {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.pred-card .pred-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--accent-light);
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}
.profile-chart {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.profile-chart h3 {
  font-size: 13px;
  color: var(--text-dim);
  margin-bottom: 8px;
  font-weight: 500;
}
.explorer-controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.explorer-controls button {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 12px;
  cursor: pointer;
  font-weight: 500;
}
.explorer-controls button:hover { opacity: 0.85; }
.obs-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 14px;
}
.obs-toggle {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: var(--text-dim);
  cursor: pointer;
  user-select: none;
  transition: all 0.15s ease;
}
.obs-toggle:hover { border-color: var(--accent); }
.obs-toggle.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.obs-toggle .color-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
}
"""


def _interactive_js(surr_json: str, param_names_json: str, obs_names_json: str, n_stages: int) -> str:
    """JavaScript for interactive PCE evaluation."""
    return f"""
<script>
(function() {{
  const S = {surr_json};
  const PARAMS = {param_names_json};
  const OBS = {obs_names_json};
  const N_STAGES = {n_stages};
  const nP = PARAMS.length;
  const nObs = OBS.length;
  const bounds = S.bounds;
  const mi = S.multi_indices;
  const popC = S.pop_coeffs;
  const gsC = S.gs_coeffs || null;
  const nBasis = mi.length;

  function legendreEval(xNorm, coeffs) {{
    let maxOrd = 0;
    for (let t = 0; t < nBasis; t++)
      for (let p = 0; p < nP; p++)
        if (mi[t][p] > maxOrd) maxOrd = mi[t][p];
    const P = [];
    for (let p = 0; p < nP; p++) {{
      P[p] = new Float64Array(maxOrd + 1);
      P[p][0] = 1.0;
      if (maxOrd >= 1) P[p][1] = xNorm[p];
      for (let n = 2; n <= maxOrd; n++)
        P[p][n] = ((2*n - 1) * xNorm[p] * P[p][n-1] - (n-1) * P[p][n-2]) / n;
    }}
    let result = 0;
    for (let t = 0; t < coeffs.length; t++) {{
      let term = coeffs[t];
      for (let p = 0; p < nP; p++) term *= P[p][mi[t][p]];
      result += term;
    }}
    return result;
  }}

  function getXNorm() {{
    const xPhys = [];
    for (let i = 0; i < nP; i++) {{
      const sl = document.getElementById('pce-slider-' + i);
      xPhys.push(parseFloat(sl.value));
    }}
    const xNorm = [];
    for (let i = 0; i < nP; i++) {{
      const lo = bounds[i][0], hi = bounds[i][1];
      xNorm.push(2 * (xPhys[i] - lo) / (hi - lo + 1e-15) - 1);
    }}
    return {{ xPhys, xNorm }};
  }}

  function shortName(s) {{
    return s.replace('fraction_active_', '')
            .replace('cell_dry_mass_fraction', 'dry_mass_frac')
            .replace('listeners__mass__', '')
            .replace('instantaneous_growth_rate', 'growth_rate');
  }}

  // Track selected observables as a Set of indices
  const selected = new Set();

  function getSelected() {{
    return selected.size > 0 ? Array.from(selected).sort() : Array.from({{length: nObs}}, (_, i) => i);
  }}

  function update() {{
    const {{ xPhys, xNorm }} = getXNorm();
    const sel = getSelected();

    // Update slider value displays
    for (let i = 0; i < nP; i++) {{
      document.getElementById('pce-val-' + i).textContent = xPhys[i].toFixed(4);
    }}

    // Population predictions
    const popPreds = [];
    for (let o = 0; o < nObs; o++) {{
      popPreds.push(legendreEval(xNorm, popC[o]));
    }}

    // Update readout cards — highlight selected
    for (let o = 0; o < nObs; o++) {{
      const el = document.getElementById('pce-pred-' + o);
      if (el) {{
        el.textContent = popPreds[o].toFixed(4);
        el.parentElement.style.borderColor = selected.has(o) ? 'var(--accent)' : 'var(--border)';
      }}
    }}

    // Growth-stratified profile chart
    if (gsC && N_STAGES > 0) {{
      drawProfile(xNorm, sel);
    }}
  }}

  const CHART_COLORS = ['#6366f1','#f59e0b','#10b981','#ef4444','#3b82f6',
                         '#8b5cf6','#ec4899','#14b8a6','#f97316','#84cc16'];

  function drawProfile(xNorm, selList) {{
    const svg = document.getElementById('pce-profile-svg');
    if (!svg) return;
    const W = 560, H = 200;
    const ml = 60, mr = 20, mt = 10, mb = 40;
    const pw = W - ml - mr, ph = H - mt - mb;
    const selSet = new Set(selList);

    // Compute all per-stage predictions
    const allPreds = []; // [obs][stage]
    for (let o = 0; o < nObs; o++) {{
      allPreds[o] = [];
      for (let s = 0; s < N_STAGES; s++) {{
        const colIdx = s * nObs + o;
        if (colIdx < gsC.length) {{
          allPreds[o].push(legendreEval(xNorm, gsC[colIdx]));
        }}
      }}
    }}

    // Find Y range across selected obs
    let yMin = Infinity, yMax = -Infinity;
    for (const o of selList) {{
      for (const v of allPreds[o]) {{
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }}
    }}
    if (yMax <= yMin) {{ yMax = yMin + 1; }}
    const yPad = (yMax - yMin) * 0.1;
    yMin -= yPad; yMax += yPad;

    let parts = [];
    // Background
    parts.push('<rect width="' + W + '" height="' + H + '" fill="#222240" rx="4"/>');
    // Grid
    for (let i = 0; i <= 4; i++) {{
      const yv = yMin + (yMax - yMin) * i / 4;
      const yp = mt + ph - (i / 4) * ph;
      parts.push('<line x1="' + ml + '" y1="' + yp.toFixed(1) + '" x2="' + (W-mr)
        + '" y2="' + yp.toFixed(1) + '" stroke="#333355" stroke-width="1"/>');
      parts.push('<text x="' + (ml-6) + '" y="' + (yp+4).toFixed(1)
        + '" text-anchor="end" font-size="10" fill="#94a3b8">' + yv.toFixed(2) + '</text>');
    }}
    // X-axis labels
    for (let s = 0; s < N_STAGES; s++) {{
      const xc = ml + (s + 0.5) * pw / N_STAGES;
      const lo = Math.round(s / N_STAGES * 100);
      const hi = Math.round((s+1) / N_STAGES * 100);
      parts.push('<text x="' + xc.toFixed(1) + '" y="' + (H - mb + 16)
        + '" text-anchor="middle" font-size="9" fill="#94a3b8">'
        + lo + '-' + hi + '%</text>');
    }}
    // X-axis title
    parts.push('<text x="' + (ml + pw/2) + '" y="' + (H - 4)
      + '" text-anchor="middle" font-size="11" fill="#94a3b8">'
      + 'Cell Cycle Progress (\\u03b8)</text>');

    // Draw lines — selected observables bold, rest dimmed
    for (let o = 0; o < nObs; o++) {{
      if (allPreds[o].length === 0) continue;
      const isSel = selSet.has(o);
      // Skip unselected entirely when specific selection exists
      if (selected.size > 0 && !isSel) continue;
      const pts = [];
      for (let s = 0; s < allPreds[o].length; s++) {{
        const x = ml + (s + 0.5) * pw / N_STAGES;
        const y = mt + ph - ((allPreds[o][s] - yMin) / (yMax - yMin)) * ph;
        pts.push(x.toFixed(1) + ',' + y.toFixed(1));
      }}
      const col = CHART_COLORS[o % CHART_COLORS.length];
      parts.push('<polyline points="' + pts.join(' ')
        + '" fill="none" stroke="' + col + '" stroke-width="2.5"'
        + ' stroke-linejoin="round"/>');
      // Dots
      for (let s = 0; s < allPreds[o].length; s++) {{
        const x = ml + (s + 0.5) * pw / N_STAGES;
        const y = mt + ph - ((allPreds[o][s] - yMin) / (yMax - yMin)) * ph;
        parts.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1)
          + '" r="3" fill="' + col + '"/>');
      }}
    }}

    // Legend (only selected)
    let lx = ml + 4, ly = mt + 4;
    for (const o of selList) {{
      const col = CHART_COLORS[o % CHART_COLORS.length];
      const name = shortName(OBS[o]);
      if (name.length > 25) continue;
      parts.push('<rect x="' + lx + '" y="' + ly + '" width="10" height="10" fill="' + col + '" rx="2"/>');
      parts.push('<text x="' + (lx+14) + '" y="' + (ly+9) + '" font-size="10" fill="#e2e8f0">' + name + '</text>');
      ly += 14;
      if (ly > mt + ph - 10) break;
    }}

    svg.innerHTML = parts.join('');
  }}

  function syncToggleStyles() {{
    const pills = document.querySelectorAll('.obs-toggle');
    pills.forEach(function(pill) {{
      const idx = parseInt(pill.dataset.idx);
      if (selected.has(idx)) {{
        pill.classList.add('active');
      }} else {{
        pill.classList.remove('active');
      }}
    }});
  }}

  // Build slider panel
  function init() {{
    const panel = document.getElementById('pce-sliders');
    if (!panel) return;
    for (let i = 0; i < nP; i++) {{
      const lo = bounds[i][0], hi = bounds[i][1];
      const mid = (lo + hi) / 2;
      const div = document.createElement('div');
      div.className = 'slider-group';
      div.innerHTML =
        '<label><code>' + shortName(PARAMS[i]) + '</code><span id="pce-val-' + i + '">'
        + mid.toFixed(4) + '</span></label>'
        + '<input type="range" id="pce-slider-' + i + '" min="' + lo + '" max="' + hi
        + '" step="' + ((hi - lo) / 200) + '" value="' + mid + '">'
        + '<div class="bounds"><span>' + lo.toFixed(3) + '</span><span>' + hi.toFixed(3) + '</span></div>';
      panel.appendChild(div);
      document.getElementById('pce-slider-' + i).addEventListener('input', update);
    }}

    // Build observable toggle pills
    const toggles = document.getElementById('pce-obs-toggles');
    for (let o = 0; o < nObs; o++) {{
      const pill = document.createElement('span');
      pill.className = 'obs-toggle active';
      pill.dataset.idx = o;
      const col = CHART_COLORS[o % CHART_COLORS.length];
      pill.innerHTML = '<span class="color-dot" style="background:' + col + '"></span>'
        + shortName(OBS[o]);
      selected.add(o);
      pill.addEventListener('click', function() {{
        const idx = parseInt(this.dataset.idx);
        if (selected.has(idx)) {{
          selected.delete(idx);
        }} else {{
          selected.add(idx);
        }}
        syncToggleStyles();
        update();
      }});
      toggles.appendChild(pill);
    }}

    // Build prediction readout cards
    const readout = document.getElementById('pce-readout');
    for (let o = 0; o < nObs; o++) {{
      const card = document.createElement('div');
      card.className = 'pred-card';
      card.innerHTML = '<div class="pred-label">' + shortName(OBS[o])
        + '</div><div class="pred-value" id="pce-pred-' + o + '">-</div>';
      card.style.cursor = 'pointer';
      card.dataset.idx = o;
      card.addEventListener('click', function() {{
        const idx = parseInt(this.dataset.idx);
        if (selected.has(idx)) {{
          selected.delete(idx);
        }} else {{
          selected.add(idx);
        }}
        syncToggleStyles();
        update();
      }});
      readout.appendChild(card);
    }}

    // Reset sliders button
    document.getElementById('pce-reset').addEventListener('click', function() {{
      for (let i = 0; i < nP; i++) {{
        const lo = bounds[i][0], hi = bounds[i][1];
        document.getElementById('pce-slider-' + i).value = (lo + hi) / 2;
      }}
      update();
    }});

    // Select All button
    document.getElementById('pce-select-all').addEventListener('click', function() {{
      for (let o = 0; o < nObs; o++) selected.add(o);
      syncToggleStyles();
      update();
    }});

    // Clear selection button
    document.getElementById('pce-select-none').addEventListener('click', function() {{
      selected.clear();
      syncToggleStyles();
      update();
    }});

    update();
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', init);
  }} else {{
    init();
  }}
}})();
</script>"""


def _interactive_section_html(has_gs: bool) -> str:
    """HTML skeleton for the interactive explorer (JS fills in the dynamic parts)."""
    profile_block = ""
    if has_gs:
        profile_block = (
            '<div class="profile-chart">'
            '<h3>Predicted Observable Profile Across Cell Cycle</h3>'
            '<svg id="pce-profile-svg" viewBox="0 0 560 200" '
            'style="width:100%;max-width:560px;height:auto;"></svg>'
            '</div>'
        )
    return f"""
<div class="explorer-controls">
  <button id="pce-reset">Reset to Midpoint</button>
  <button id="pce-select-all">Select All</button>
  <button id="pce-select-none" style="background:var(--surface2);color:var(--text-dim);border:1px solid var(--border);">Clear</button>
</div>
<div class="obs-toggles" id="pce-obs-toggles"></div>
<div class="explorer-grid">
  <div class="slider-panel" id="pce-sliders"></div>
  <div class="pred-panel">
    <div class="pred-readout" id="pce-readout"></div>
    {profile_block}
  </div>
</div>"""


# ── Main report generator ──


def generate_html_report(
    results_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML report from UQ export artifacts.

    Parameters
    ----------
    results_dir
        Path to the UQ export directory (must contain ``uq_results.json``).
    output_path
        Where to write the HTML file. Defaults to ``<results_dir>/report.html``.

    Returns
    -------
    Path
        The written HTML file path.
    """
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "uq_results.json").read_text())

    manifest = None
    manifest_path = results_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    output_path = Path(output_path) if output_path else results_dir / "report.html"

    # ── Load surrogate data for interactive explorer ──
    surr = _load_surrogate_json(results_dir)

    # ── Extract structured data ──
    params = list(data["parameters"].keys())
    obs = data.get("observable_names", [])
    n_params = data.get("n_parameters", len(params))
    framework = data.get("framework", "UQ Pipeline")

    # Strategy 1 — population
    s1_total = data["phase1_population"]["sobol_total_order"]
    s1_first = data["phase1_population"].get("sobol_first_order", {})

    # Strategy 2 — by generation
    s2_data = data.get("strategy2_by_generation", {})
    s2_gens = s2_data.get("generations", [])
    n_gens = s2_data.get("n_generations", 0)

    # Strategy 3 — by lineage seed
    s3_data = data.get("strategy3_by_seed", {})
    s3_seeds = s3_data.get("seeds", [])
    n_seeds = s3_data.get("n_seeds", 0)

    # Strategy 4 — growth-stratified
    s4_data = data.get("phase2_growth_stratified", {})
    s4_stages = s4_data.get("stages", [])
    n_stages = s4_data.get("n_stages", 0)

    # Averages for cross-strategy comparison
    def _avg_sobol(entries: list[dict], key: str = "sobol_total_order") -> dict[str, float]:
        if not entries:
            return {}
        out = {p: 0.0 for p in params}
        for e in entries:
            for p in params:
                out[p] += e[key].get(p, 0)
        return {p: v / len(entries) for p, v in out.items()}

    s2_avg = _avg_sobol(s2_gens) if s2_gens else None
    s3_avg = _avg_sobol(s3_seeds) if s3_seeds else None

    # Manifest info
    timestamp = hostname = cli_cmd = n_samples = ""
    pkg_versions: dict[str, str] = {}
    git_shas: dict[str, str] = {}
    if manifest:
        timestamp = manifest.get("timestamp", "")
        hostname = manifest.get("hostname", "")
        cli_cmd = " ".join(manifest.get("cli_command", []))
        n_samples = str(manifest.get("n_samples", ""))
        pkg_versions = manifest.get("package_versions", {})
        git_shas = manifest.get("git_sha", {})

    ts_display = timestamp
    if timestamp:
        try:
            ts_display = datetime.fromisoformat(timestamp).strftime("%B %d, %Y at %H:%M UTC")
        except Exception:
            pass

    top_param = max(s1_total, key=s1_total.get)
    top_val = s1_total[top_param]

    # ── Build all chart/table fragments ──

    # S1: population bar chart + ranking
    s1_bar = _svg_bar_chart(s1_total, s1_first if s1_first else None)
    s1_rank = _ranking_table(s1_total, s1_first if s1_first else None)

    # S2: per-generation
    s2_groups = [g["sobol_total_order"] for g in s2_gens]
    s2_labels = [f"Gen {g['generation']}" for g in s2_gens]
    s2_heatmap = _svg_heatmap(params, s2_groups, s2_labels) if n_gens > 1 else ""
    s2_table = _evolution_table(params, s2_groups, s2_labels) if n_gens > 1 else ""
    s2_single_bar = ""
    s2_single_rank = ""
    if n_gens == 1 and s2_gens:
        g0_total = s2_gens[0]["sobol_total_order"]
        g0_first = s2_gens[0].get("sobol_first_order")
        s2_single_bar = _svg_bar_chart(g0_total, g0_first)
        s2_single_rank = _ranking_table(g0_total, g0_first)

    # S3: per-seed
    s3_groups = [s["sobol_total_order"] for s in s3_seeds]
    s3_labels = [f"Seed {s['lineage_seed']}" for s in s3_seeds]
    s3_heatmap = _svg_heatmap(params, s3_groups, s3_labels) if n_seeds > 1 else ""
    s3_table = _evolution_table(params, s3_groups, s3_labels) if n_seeds > 1 else ""
    s3_single_bar = ""
    s3_single_rank = ""
    if n_seeds == 1 and s3_seeds:
        sd0_total = s3_seeds[0]["sobol_total_order"]
        sd0_first = s3_seeds[0].get("sobol_first_order")
        s3_single_bar = _svg_bar_chart(sd0_total, sd0_first)
        s3_single_rank = _ranking_table(sd0_total, sd0_first)

    # S4: growth-stratified
    s4_groups = [s["sobol_total_order"] for s in s4_stages]
    s4_labels = [f"{s['theta_range'][0]:.0%}-{s['theta_range'][1]:.0%}" for s in s4_stages]
    s4_heatmap = _svg_heatmap(params, s4_groups, s4_labels)
    s4_table = _evolution_table(params, s4_groups, s4_labels)

    # ── Experimental design section ──
    param_specs = manifest.get("parameter_specs", []) if manifest else []
    surr_meta_path = results_dir / "population_surrogate" / "metadata.json"
    surr_meta = json.loads(surr_meta_path.read_text()) if surr_meta_path.exists() else {}
    bounds = surr.get("bounds", []) if surr else []
    obs_columns = manifest.get("observable_names", obs) if manifest else obs

    def _param_spec_table() -> str:
        """Parameter specification table with biological context."""
        if not param_specs and not bounds:
            return ""
        rows = []
        for i, name in enumerate(params):
            spec = next((s for s in param_specs if s["name"] == name), None)
            attr_path = spec["attr_path"] if spec and spec.get("attr_path") else ""
            desc = spec["description"] if spec and spec.get("description") else ""
            lo = bounds[i][0] if i < len(bounds) else ""
            hi = bounds[i][1] if i < len(bounds) else ""
            mid = (lo + hi) / 2 if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else ""
            # Format numbers — use scientific for very small values
            def _fnum(v: float | str) -> str:
                if isinstance(v, str):
                    return v
                if abs(v) < 0.01 and v != 0:
                    return f"{v:.2e}"
                return f"{v:.4f}"
            rows.append((name, attr_path, _fnum(lo), _fnum(hi), _fnum(mid), desc))

        L = ['<table>', '<thead><tr>',
             '<th>Parameter</th><th>SimData Path</th>',
             '<th>Min</th><th>Max</th><th>Midpoint</th>',
             '<th>Biological Role</th>',
             '</tr></thead>', '<tbody>']
        for name, attr, lo, hi, mid, desc in rows:
            L.append(
                f'<tr><td><code>{html.escape(_short(name))}</code></td>'
                f'<td><code style="font-size:10px;color:var(--text-dim)">{html.escape(attr)}</code></td>'
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">{lo}</td>'
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">{hi}</td>'
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">{mid}</td>'
                f'<td style="font-size:12px;color:var(--text-dim);max-width:260px">{html.escape(desc)}</td></tr>'
            )
        L.extend(['</tbody>', '</table>'])
        return '\n'.join(L)

    param_spec_html = _param_spec_table()

    # Simulation design summary
    poly_order = surr_meta.get("polynomial_order", "")
    basis_type = surr_meta.get("basis_type", "")
    n_basis_terms = surr_meta.get("input_dim", "")
    r_squared = surr_meta.get("r_squared", "")

    # Cross-strategy comparison
    cross_strategies: list[tuple[str, dict[str, float], str]] = [
        ("Population", s1_total, _COLORS[0]),
    ]
    if s2_avg:
        cross_strategies.append(("By Generation", s2_avg, _COLORS[1]))
    if s3_avg:
        cross_strategies.append(("By Lineage", s3_avg, _COLORS[2]))
    if s4_stages:
        s4_avg_dict: dict[str, float] = {}
        for p in params:
            s4_avg_dict[p] = sum(s["sobol_total_order"].get(p, 0) for s in s4_stages) / len(s4_stages)
        cross_strategies.append(("Growth-Stratified", s4_avg_dict, _COLORS[3]))
    cross_bar = _svg_cross_strategy_bars(params, cross_strategies)
    cross_insight = _cross_strategy_insight(params, s1_total, s2_avg, s3_avg)

    # ── Strategy 2/3 section builders ──

    def _s2_section() -> str:
        if n_gens == 0:
            return ""
        desc = (
            f"Sensitivity analysis performed on data aggregated by generation. "
            f"<strong>{n_gens} generation{'s' if n_gens != 1 else ''}</strong> analyzed."
        )
        if n_gens == 1:
            note = ('<p style="color:var(--text-dim);font-size:13px;margin:12px 0 0;">'
                    'Only 1 generation available. Run with more generations to observe '
                    'how parameter importance evolves across cell divisions.</p>')
            return (
                f'<details class="accordion">'
                f'<summary>By Generation <span class="badge">Strategy 2</span></summary>'
                f'<div class="accordion-body">'
                f'<p class="section-desc">{desc}</p>'
                f'<div class="chart-container">{s2_single_bar}</div>'
                f'{s2_single_rank}'
                f'{note}</div></details>'
            )
        return (
            f'<details class="accordion">'
            f'<summary>By Generation <span class="badge">Strategy 2</span></summary>'
            f'<div class="accordion-body">'
            f'<p class="section-desc">{desc} '
            f'Does the dominant parameter change as cells age through divisions?</p>'
            f'<div class="chart-container">{s2_heatmap}</div>'
            f'<div style="overflow-x:auto;margin-top:16px;">{s2_table}</div>'
            f'</div></details>'
        )

    def _s3_section() -> str:
        if n_seeds == 0:
            return ""
        desc = (
            f"Sensitivity analysis performed on data aggregated by lineage seed. "
            f"<strong>{n_seeds} lineage{'s' if n_seeds != 1 else ''}</strong> analyzed."
        )
        if n_seeds == 1:
            note = ('<p style="color:var(--text-dim);font-size:13px;margin:12px 0 0;">'
                    'Only 1 lineage seed available. Run with more seeds to assess '
                    'whether sensitivity depends on stochastic lineage history.</p>')
            return (
                f'<details class="accordion">'
                f'<summary>By Lineage <span class="badge">Strategy 3</span></summary>'
                f'<div class="accordion-body">'
                f'<p class="section-desc">{desc}</p>'
                f'<div class="chart-container">{s3_single_bar}</div>'
                f'{s3_single_rank}'
                f'{note}</div></details>'
            )
        return (
            f'<details class="accordion">'
            f'<summary>By Lineage <span class="badge">Strategy 3</span></summary>'
            f'<div class="accordion-body">'
            f'<p class="section-desc">{desc} '
            f'Is the sensitivity landscape consistent across independent lineages, '
            f'or do some lineages have unique dominant drivers?</p>'
            f'<div class="chart-container">{s3_heatmap}</div>'
            f'<div style="overflow-x:auto;margin-top:16px;">{s3_table}</div>'
            f'</div></details>'
        )

    # ── Assemble HTML ──

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UQ Report</title>
<style>
:root {{
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --surface2: #222240;
  --border: #2d2d50;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --accent: #6366f1;
  --accent-light: #818cf8;
  --success: #10b981;
  --warning: #f59e0b;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}

/* Header */
.header {{
  text-align: center;
  padding: 48px 0 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
}}
.header h1 {{
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
}}
.header .subtitle {{ color: var(--text-dim); font-size: 14px; }}

/* Metrics */
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px;
  margin-bottom: 36px;
}}
.metric {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  text-align: center;
}}
.metric .value {{ font-size: 26px; font-weight: 700; color: var(--accent-light); }}
.metric .label {{
  font-size: 11px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;
}}

/* Sections */
.section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 28px;
  margin-bottom: 24px;
}}
.section h2 {{
  font-size: 18px; font-weight: 600;
  margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}}
.section h2 .badge {{
  font-size: 11px; background: var(--accent); color: #fff;
  padding: 2px 8px; border-radius: 10px; font-weight: 500;
}}
.section-desc {{
  color: var(--text-dim); font-size: 13px;
  margin-bottom: 16px; line-height: 1.5;
}}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
  text-align: left; padding: 10px 12px;
  border-bottom: 2px solid var(--border);
  color: var(--text-dim); font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
}}
td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
tr:last-child td {{ border-bottom: none; }}
tr:hover {{ background: var(--surface2); }}
code {{
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px; background: var(--surface2);
  padding: 2px 6px; border-radius: 4px;
}}
.rank {{ color: var(--text-dim); font-weight: 600; width: 40px; }}
.bar-cell {{ display: flex; align-items: center; gap: 8px; min-width: 140px; }}
.bar {{
  height: 8px; background: var(--accent);
  border-radius: 4px; min-width: 2px;
}}
.bar-cell span {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}

/* Stage/evolution tables */
.stage-table td {{
  padding: 6px 8px; font-size: 12px; font-variant-numeric: tabular-nums;
}}
.stage-table th {{ font-size: 10px; padding: 6px 8px; }}
.trend-up {{ color: #ef4444; font-weight: 600; }}
.trend-down {{ color: #10b981; font-weight: 600; }}
.trend-flat {{ color: var(--text-dim); }}

/* Charts */
.chart-container {{
  display: flex; justify-content: center;
  padding: 8px 0; overflow-x: auto;
}}
svg text {{ font-family: 'Inter', -apple-system, sans-serif; }}

/* Observable pills */
.obs-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
.obs-pill {{
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 14px;
  font-size: 12px; font-family: monospace; color: var(--text-dim);
}}

/* Info grid */
.info-grid {{
  display: grid; grid-template-columns: 140px 1fr;
  gap: 8px 16px; font-size: 13px;
}}
.info-grid .key {{ color: var(--text-dim); font-weight: 500; }}
.info-grid .val {{ color: var(--text); word-break: break-all; }}
.info-grid .val code {{ font-size: 11px; }}

/* Callouts */
.callout {{
  border-radius: 12px; padding: 20px 24px;
  margin-bottom: 24px;
  display: flex; align-items: center; gap: 16px;
}}
.callout .icon {{ font-size: 28px; line-height: 1; }}
.callout .detail {{ flex: 1; }}
.callout .detail .title {{ font-weight: 600; font-size: 15px; margin-bottom: 2px; }}
.callout .detail .desc {{ font-size: 13px; color: var(--text-dim); }}
.callout-warn {{
  background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
  border: 1px solid rgba(245,158,11,0.3);
}}
.callout-ok {{
  background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
  border: 1px solid rgba(16,185,129,0.3);
}}
.callout-info {{
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
  border: 1px solid rgba(99,102,241,0.3);
}}

/* Accordion (strategy cards) */
details.accordion {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 24px;
  overflow: hidden;
}}
details.accordion > summary {{
  padding: 20px 28px;
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  user-select: none;
}}
details.accordion > summary::-webkit-details-marker {{ display: none; }}
details.accordion > summary::before {{
  content: "\\25B6";
  font-size: 12px;
  color: var(--accent-light);
  transition: transform 0.2s ease;
  display: inline-block;
}}
details.accordion[open] > summary::before {{
  transform: rotate(90deg);
}}
details.accordion > summary:hover {{
  background: var(--surface2);
}}
details.accordion > .accordion-body {{
  padding: 0 28px 28px;
}}

/* Divider */
.strategy-divider {{
  text-align: center; color: var(--text-dim);
  font-size: 12px; text-transform: uppercase;
  letter-spacing: 1px; margin: 36px 0 24px;
  display: flex; align-items: center; gap: 16px;
}}
.strategy-divider::before, .strategy-divider::after {{
  content: ""; flex: 1; height: 1px; background: var(--border);
}}

/* Footer */
.footer {{
  text-align: center; padding: 32px 0 16px;
  color: var(--text-dim); font-size: 12px;
  border-top: 1px solid var(--border); margin-top: 24px;
}}

@media (max-width: 600px) {{
  .container {{ padding: 20px 12px; }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .header h1 {{ font-size: 22px; }}
  .section {{ padding: 16px; }}
}}
{_interactive_css() if surr else ""}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div style="display:flex;justify-content:center;margin-bottom:20px;">
    <div style="display:inline-flex;align-items:center;gap:0;">
      <div style="
        border:2px solid #ff3366;
        border-radius:50px;
        padding:0.7rem 2rem;
        background:linear-gradient(135deg, rgba(233,30,144,0.08), rgba(0,229,255,0.05));
        text-align:center;
        min-width:320px;
      ">
        <div style="
          font-family:'Courier New',monospace;
          font-weight:900;
          font-size:1.4rem;
          letter-spacing:0.15em;
          background:linear-gradient(90deg, #ff3366, #00f0ff);
          -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;
        ">&#x1F9EC; ATLANTIS &#x1F9EC;</div>
        <div style="
          font-family:'Courier New',monospace;
          font-size:0.7rem;
          color:#00f0ff;
          letter-spacing:0.1em;
          margin-top:0.2rem;
        ">whole-cell simulation platform</div>
      </div>
      <svg width="90" height="60" viewBox="0 0 90 60" style="margin-left:-2px;">
        <path d="M0,30 C15,15 25,45 40,30 S60,15 75,30 S85,42 90,35" stroke="#ff3366" fill="none" stroke-width="2" opacity="0.7"/>
        <path d="M0,25 C12,10 22,40 35,25 S55,10 70,25 S82,38 88,30" stroke="#00f0ff" fill="none" stroke-width="1.5" opacity="0.5"/>
        <path d="M0,35 C18,20 28,50 45,35 S65,20 80,35 S88,48 90,40" stroke="#ffd700" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>
    </div>
  </div>
  <div style="padding:0.7rem 2rem">
    <h1>Uncertainty Quantification Report</h1>
    <div class="subtitle">{ts_display or "Generated " + datetime.now(tz=timezone.utc).strftime("%B %d, %Y")}</div>
  </div>
</div>

<!-- Key metrics -->
<div class="metrics">
  <div class="metric"><div class="value">{n_params}</div><div class="label">Parameters</div></div>
  <div class="metric"><div class="value">{len(obs)}</div><div class="label">Observables</div></div>
  <div class="metric"><div class="value">{n_samples or '-'}</div><div class="label">Samples</div></div>
  <div class="metric"><div class="value">{n_gens}</div><div class="label">Generations</div></div>
  <div class="metric"><div class="value">{n_seeds}</div><div class="label">Lineages</div></div>
  <div class="metric"><div class="value">{n_stages}</div><div class="label">Cell Cycle Stages</div></div>
</div>

<!-- Top driver -->
<div class="callout callout-info">
  <div class="icon">&#x1F3AF;</div>
  <div class="detail">
    <div class="title">Population-Level Top Driver: <code>{html.escape(_short(top_param))}</code></div>
    <div class="desc">Explains {top_val:.1%} of total output variance at the population level
      (S_Ti&nbsp;=&nbsp;{_fmt(top_val)}). See below for whether this holds across strategies.</div>
  </div>
</div>

<!-- ═══════════════ EXPERIMENTAL DESIGN ═══════════════ -->

<div class="strategy-divider">Experimental Design</div>

<div class="section">
  <h2>Simulation Design</h2>
  <p class="section-desc">
    Each sample corresponds to a full vEcoli whole-cell simulation with perturbed
    <code>SimulationDataEcoli</code> attributes. Parameters are varied simultaneously
    via Latin Hypercube Sampling within the bounds below, and the simulation is
    executed using the <code>sim_data_setattr</code> variant mechanism.
  </p>
  <div class="info-grid" style="margin-bottom:16px;">
    <div class="key">Samples (N)</div><div class="val">{n_samples or '-'}</div>
    <div class="key">Polynomial Order</div><div class="val">{poly_order or '-'}</div>
    <div class="key">Basis</div><div class="val">{(basis_type or '-').capitalize()}</div>
    <div class="key">Generations</div><div class="val">{n_gens}</div>
    <div class="key">Lineage Seeds</div><div class="val">{n_seeds}</div>
  </div>
</div>

{"" if not param_spec_html else '''
<div class="section">
  <h2>Parameter Specifications</h2>
  <p class="section-desc">
    Each parameter targets a specific attribute in the vEcoli
    <code>SimulationDataEcoli</code> object. Bounds define the perturbation range
    used during sampling (&plusmn; around the calibrated baseline value).
  </p>
  <div style="overflow-x:auto;">
    ''' + param_spec_html + '''
  </div>
</div>
'''}

<div class="section">
  <h2>Tracked Observables</h2>
  <p class="section-desc">
    Output variables extracted from each simulation and used to compute
    sensitivity indices. These correspond to time-averaged quantities
    from vEcoli's listener outputs.
  </p>
  <div class="obs-list">
    {''.join(f'<span class="obs-pill">{html.escape(_short(o))}</span>' for o in obs)}
  </div>
</div>

<!-- ═══════════════ SENSITIVITY ANALYSIS ═══════════════ -->

<div class="strategy-divider">Sensitivity Analysis</div>

{cross_insight}

<div class="section">
  <h2>S_Ti Comparison Across Strategies</h2>
  <p class="section-desc">
    Total Sobol indices side by side. Differences reveal that parameter importance
    depends on how cells are grouped &mdash; population-level conclusions alone
    may miss important drivers visible at the generation, lineage, or cell-cycle level.
  </p>
  <div class="chart-container">{cross_bar}</div>
</div>

<!-- Strategy 1: Population -->
<details class="accordion">
  <summary>Population <span class="badge">Strategy 1</span></summary>
  <div class="accordion-body">
    <p class="section-desc">
      All cells pooled uniformly across every seed and generation.
      Which parameters drive the most variance across the entire population?
    </p>
    <div class="chart-container">{s1_bar}</div>
    {s1_rank}
  </div>
</details>

<!-- Strategy 2: By Generation -->
{_s2_section()}

<!-- Strategy 3: By Lineage -->
{_s3_section()}

<!-- Strategy 4: Growth-Stratified -->
<details class="accordion">
  <summary>Growth-Stratified <span class="badge">Strategy 4</span></summary>
  <div class="accordion-body">
    <p class="section-desc">
      S_Ti resolved across {n_stages} cell cycle stages
      (&theta; = growth progress from birth to division).
      How does parameter importance evolve within a single cell cycle?
    </p>
    <div class="chart-container">{s4_heatmap}</div>
    <div style="overflow-x:auto;margin-top:16px;">{s4_table}</div>
  </div>
</details>

{"" if not surr else '''
<div class="strategy-divider">Interactive PCE Explorer</div>

<div class="section">
  <h2>Surrogate Predictor <span class="badge">Interactive</span></h2>
  <p class="section-desc">
    Drag parameter sliders to evaluate the fitted PCE surrogate in real time.
    Predicted observable values update instantly &mdash; no simulation required.
    Use this to explore "what if" scenarios: tune parameters until the predicted
    outputs match a desired cell state.
  </p>
  ''' + _interactive_section_html(has_gs=bool(surr.get("gs_coeffs"))) + '''
</div>
'''}

<!-- Provenance -->
<div class="section">
  <h2>Provenance</h2>
  <div class="info-grid">
    {"".join(f'<div class="key">{k}</div><div class="val">{v}</div>' for k, v in [
        ("Timestamp", ts_display),
        ("Host", html.escape(hostname)),
        ("Samples", n_samples),
        ("CLI Command", f"<code>{html.escape(cli_cmd)}</code>" if cli_cmd else "-"),
    ] + [
        (f"Git ({html.escape(repo)})", f"<code>{html.escape(sha[:12])}</code>")
        for repo, sha in git_shas.items()
    ] + [
        (html.escape(pkg), f"<code>{html.escape(ver)}</code>")
        for pkg, ver in pkg_versions.items()
    ] if v and v != "-")}
  </div>
</div>

<div class="footer">
  Generated from {html.escape(str(results_dir.resolve()))}
</div>

</div>
{"" if not surr else _interactive_js(
    json.dumps(surr),
    json.dumps(params),
    json.dumps(obs),
    n_stages,
)}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html)
    return output_path
