"""
UQPC Interactive GUI — marimo-powered interface for the full UQ workflow.

Launch:
    uv run marimo run app/gui.py

Covers the complete pipeline:
  1. Configure parameters (toggle, edit bounds)
  2. Run sampling (PCRV.sampleGerm → vEcoli workflow.py)
  3. Run quantify (PCE + Sobol, all 4 strategies)
  4. Explore results interactively (response curves, Sobol bars, spectrogram)
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from wigglystuff import (
        ParallelCoordinates,
        TangleSlider,
    )
    from wigglystuff import (
        ProgressBar as WProgressBar,
    )

    return Path, TangleSlider, go, json, make_subplots, np, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # UQPC · Global Sensitivity Analysis

    **RFC006** · PyTUQ UQPC workflow · vEcoli whole-cell model

    ---
    """)
    return


@app.cell
def _(mo):
    _sim_data_input = mo.ui.text(
        value="/Users/alexanderpatrie/sms/uqEcoli/sim_data/baseline/kb/simData.cPickle",
        label="simData.cPickle path",
        full_width=True,
    )
    _cache_dir_input = mo.ui.text(
        value="./uq_cache",
        label="Cache directory",
        full_width=True,
    )
    _export_dir_input = mo.ui.text(
        value="./uq_results",
        label="Export directory",
        full_width=True,
    )
    _n_samples_slider = mo.ui.slider(
        start=2, stop=200, step=1, value=20,
        label="Number of samples (variants)",
        show_value=True,
        full_width=True,
    )
    _generations_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=1,
        label="Generations",
        show_value=True,
    )
    _seeds_slider = mo.ui.slider(
        start=1, stop=8, step=1, value=1,
        label="Seeds (n_init_sims)",
        show_value=True,
    )
    _order_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=2,
        label="PCE polynomial order",
        show_value=True,
    )
    _bins_slider = mo.ui.slider(
        start=2, stop=20, step=1, value=10,
        label="Growth bins (Strategy 4)",
        show_value=True,
    )
    _regression_select = mo.ui.dropdown(
        options={
            "lsq — Least Squares": "lsq",
            "bcs — Bayesian Compressed Sensing": "bcs",
            "anl — Analytical Bayesian": "anl",
        },
        value="lsq — Least Squares",
        label="Regression method",
    )

    mo.md("## Configuration")
    mo.hstack(
        [
            mo.vstack([
                mo.md("### Paths"),
                _sim_data_input,
                _cache_dir_input,
                _export_dir_input,
            ]),
            mo.vstack([
                mo.md("### Sampling"),
                _n_samples_slider,
                _generations_slider,
                _seeds_slider,
            ]),
            mo.vstack([
                mo.md("### Quantify"),
                _order_slider,
                _bins_slider,
                _regression_select,
            ]),
        ],
        justify="start",
        gap=2,
    )

    sim_data_path = _sim_data_input
    cache_dir = _cache_dir_input
    export_dir = _export_dir_input
    n_samples = _n_samples_slider
    n_generations = _generations_slider
    n_seeds = _seeds_slider
    pce_order = _order_slider
    growth_bins = _bins_slider
    regression = _regression_select
    return (
        cache_dir,
        export_dir,
        growth_bins,
        n_generations,
        n_samples,
        n_seeds,
        pce_order,
        regression,
        sim_data_path,
    )


@app.cell
def _(mo):
    _param_defs = [
        ("fraction_active_rnap_free", "process.transcription.fraction_active_rnap_free", 0.25, 0.47),
        ("fraction_active_rnap_bound", "process.transcription.fraction_active_rnap_bound", 0.12, 0.22),
        ("basal_elongation_rate", "process.translation.basal_elongation_rate", 15.0, 28.0),
        ("kinetic_objective_weight", "process.metabolism.kinetic_objective_weight", 5e-8, 5e-7),
        ("secretion_penalty_coeff", "process.metabolism.secretion_penalty_coeff", 5e-4, 5e-3),
        ("cell_dry_mass_fraction", "mass.cell_dry_mass_fraction", 0.25, 0.35),
    ]

    _toggles = {}
    _elements = []
    for _name, _path, _lo, _hi in _param_defs:
        _chk = mo.ui.checkbox(value=True, label=f"**{_name}**")
        _lo_num = mo.ui.number(value=_lo, start=0, stop=1e6, step=_lo * 0.01, label="lo")
        _hi_num = mo.ui.number(value=_hi, start=0, stop=1e6, step=_hi * 0.01, label="hi")
        _toggles[_name] = {"check": _chk, "lo": _lo_num, "hi": _hi_num, "path": _path}
        _elements.append(
            mo.hstack([_chk, _lo_num, _hi_num, mo.md(f"`{_path}`")], justify="start", gap=1)
        )

    mo.md("### Variant Parameters")
    mo.vstack(_elements)

    param_toggles = _toggles
    return


@app.cell
def _(mo, n_generations, n_samples, n_seeds):
    _total = (n_samples.value + 1) * n_seeds.value * n_generations.value
    mo.md(
        f"""
        ### Sampling Summary

        | Metric | Value |
        |--------|-------|
        | PCRV samples (variants) | **{n_samples.value}** |
        | Generations per variant | **{n_generations.value}** |
        | Seeds per variant | **{n_seeds.value}** |
        | **Total vEcoli simulations** | **{_total}** |
        | Strategy 2 (by generation) | {"Enabled" if n_generations.value >= 2 else "Disabled (need generations >= 2)"} |
        | Strategy 3 (by seed) | {"Enabled" if n_seeds.value >= 2 else "Disabled (need seeds >= 2)"} |
        """
    )
    return


@app.cell
def _(mo):
    _run_sample_btn = mo.ui.run_button(label="Run Sampling (Steps 1-3)")
    _run_sample_btn

    run_sample_btn = _run_sample_btn
    run_sample_btn
    return (run_sample_btn,)


@app.cell
def _(
    cache_dir,
    mo,
    n_generations,
    n_samples,
    n_seeds,
    run_sample_btn,
    sim_data_path,
    subprocess,
    sys,
):
    mo.stop(not run_sample_btn.value, mo.md("*Click 'Run Sampling' to start*"))

    _cmd = [
        sys.executable, "-m", "uq.cli", "sample",
        sim_data_path.value,
        "--cache-dir", cache_dir.value,
        "--n-samples", str(n_samples.value),
        "--generations", str(n_generations.value),
        "--n-init-sims", str(n_seeds.value),
    ]

    with mo.status.spinner("Running vEcoli workflow.py..."):
        _result = subprocess.run(_cmd, capture_output=True, text=True, timeout=7200)

    if _result.returncode == 0:
        mo.md(
            f"""
            ### Sampling Complete

            ```
            {_result.stdout[-2000:]}
            ```
            """
        )
    else:
        mo.md(
            f"""
            ### Sampling Error (exit code {_result.returncode})

            ```
            {_result.stderr[-2000:]}
            ```
            """
        )

    sample_done = _result.returncode == 0
    return


@app.cell
def _(mo):
    _run_quantify_btn = mo.ui.run_button(label="Run Quantify (Steps 4-5)")
    _run_quantify_btn

    run_quantify_btn = _run_quantify_btn
    return (run_quantify_btn,)


@app.cell
def _(run_quantify_btn):
    run_quantify_btn
    return


@app.cell
def _(
    cache_dir,
    export_dir,
    growth_bins,
    mo,
    pce_order,
    regression,
    run_quantify_btn,
    sim_data_path,
    subprocess,
    sys,
):
    mo.stop(not run_quantify_btn.value, mo.md("*Click 'Run Quantify' after sampling*"))

    _cmd = [
        sys.executable, "-m", "uq.cli", "quantify",
        sim_data_path.value,
        "--cache-dir", cache_dir.value,
        "--export-path", export_dir.value,
        "--polynomial-order", str(pce_order.value),
        "--n-bins", str(growth_bins.value),
        "--regression", regression.value,
    ]

    with mo.status.spinner("Fitting PCE surrogates + computing Sobol indices..."):
        _result = subprocess.run(_cmd, capture_output=True, text=True, timeout=300)

    if _result.returncode == 0:
        mo.md(
            f"""
            ### Quantify Complete

            ```
            {_result.stdout[-3000:]}
            ```
            """
        )
    else:
        mo.md(
            f"""
            ### Quantify Error (exit code {_result.returncode})

            ```
            {_result.stderr[-2000:]}
            ```
            """
        )

    quantify_done = _result.returncode == 0
    return


@app.cell
def _(Path, export_dir, json, mo, np):
    _export = Path(export_dir.value)
    _json_path = _export / "uq_results.json"

    if not _json_path.exists():
        mo.stop(True, mo.md("*Run quantify first — no results found*"))

    with open(_json_path) as _f:
        _data = json.load(_f)

    # Load surrogate
    _pop_surr = _export / "population_surrogate"
    _surr = {"available": False}
    if (_pop_surr / "coefficients.npy").exists():
        _surr = {
            "available": True,
            "coeffs": np.load(_pop_surr / "coefficients.npy"),
            "mi": np.load(_pop_surr / "multi_indices.npy"),
            "bounds": np.load(_pop_surr / "input_bounds.npy"),
        }

    data = _data
    surr = _surr
    return data, surr


@app.cell
def _(data, go, mo):
    if data is None:
        mo.stop(True)

    _s1 = data.get("phase1_population", {})
    _sobol = _s1.get("sobol_total_order", {})

    if not _sobol:
        mo.stop(True, mo.md("*No Strategy 1 results*"))

    _params = list(_sobol.keys())
    _values = [_sobol[p] for p in _params]

    _fig = go.Figure(
        go.Bar(
            x=_values,
            y=_params,
            orientation="h",
            marker=dict(
                color=_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="S_T"),
            ),
            text=[f"{v:.1%}" for v in _values],
            textposition="auto",
        )
    )
    _fig.update_layout(
        title="Strategy 1 — Population Sobol (S_T)",
        xaxis_title="Total-order Sobol index",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=400,
    )

    mo.md("## Results")
    _fig
    return


@app.cell
def _(data, go, mo, np):
    if data is None:
        mo.stop(True)

    _s4 = data.get("phase2_growth_stratified", {})
    _stages = _s4.get("stages", [])
    if not _stages:
        mo.stop(True, mo.md("*No Strategy 4 results*"))

    _params = list(_stages[0].get("sobol_total_order", {}).keys())
    _n_stages = len(_stages)
    _n_params = len(_params)

    _heatmap = np.zeros((_n_params, _n_stages))
    for _j, _stage in enumerate(_stages):
        _st = _stage.get("sobol_total_order", {})
        for _i, _p in enumerate(_params):
            _heatmap[_i, _j] = _st.get(_p, 0)

    _theta_labels = [
        f"{_stages[j].get('theta_range', [0, 0])[0]:.0%}–{_stages[j].get('theta_range', [0, 0])[1]:.0%}"
        for j in range(_n_stages)
    ]

    _fig = go.Figure(
        go.Heatmap(
            z=_heatmap,
            x=_theta_labels,
            y=_params,
            colorscale="Viridis",
            colorbar=dict(title="S_T"),
            text=np.round(_heatmap * 100, 1).astype(str),
            texttemplate="%{text}%",
        )
    )
    _fig.update_layout(
        title="Strategy 4 — Growth-Stratified Sensitivity Spectrogram",
        xaxis_title="Growth progress θ (0 = birth, 1 = division)",
        yaxis_title="Parameter",
        template="plotly_dark",
        height=450,
    )
    _fig
    return


@app.cell
def _(TangleSlider, data, mo, surr):
    if data is None or not surr.get("available"):
        mo.stop(True, mo.md("*Surrogate not available — run quantify with export*"))

    _params = list(data["parameters"].keys())
    _bounds = surr["bounds"]
    _coeffs = surr["coeffs"]
    _mi = surr["mi"]

    # Build TangleSliders for each parameter
    _sliders = {}
    _slider_elements = []
    for _i, _p in enumerate(_params):
        _lo, _hi = float(_bounds[_i, 0]), float(_bounds[_i, 1])
        _mid = (_lo + _hi) / 2
        _step = (_hi - _lo) / 100
        _s = TangleSlider(
            amount=_mid,
            min_value=_lo,
            max_value=_hi,
            step=_step,
            suffix=f"  ({_p})",
            digits=4,
        )
        _sliders[_p] = _s
        _slider_elements.append(_s)

    mo.md("### PCE Response Explorer")
    mo.md("Drag values inline to see how the surrogate response changes:")
    mo.vstack(_slider_elements)

    param_sliders = _sliders
    return (param_sliders,)


@app.cell
def _(data, go, make_subplots, mo, np, param_sliders, surr):
    if not surr.get("available"):
        mo.stop(True)

    _params = list(data["parameters"].keys())
    _bounds = surr["bounds"]
    _coeffs = surr["coeffs"]
    _mi = surr["mi"]

    # Get current slider values
    _x_current = np.array([param_sliders[p].value for p in _params])

    # Evaluate PCE at sweep points for each parameter
    _n_sweep = 100
    _figs = make_subplots(
        rows=2, cols=3,
        subplot_titles=_params[:6],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for _idx, _p in enumerate(_params[:6]):
        _row = _idx // 3 + 1
        _col = _idx % 3 + 1
        _lo, _hi = float(_bounds[_idx, 0]), float(_bounds[_idx, 1])

        _sweep_vals = np.linspace(_lo, _hi, _n_sweep)
        _y_sweep = []

        for _sv in _sweep_vals:
            _x_eval = _x_current.copy()
            _x_eval[_idx] = _sv
            # Scale to germ space [-1, 1]
            _x_germ = 2.0 * (_x_eval - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0] + 1e-30) - 1.0
            # Evaluate polynomial: sum(coeff * prod(x^mi))
            _y = 0.0
            for _k in range(len(_coeffs)):
                _term = _coeffs[_k]
                for _d in range(len(_params)):
                    if _mi[_k, _d] > 0:
                        _term *= _x_germ[_d] ** _mi[_k, _d]
                _y += _term
            _y_sweep.append(_y)

        _figs.add_trace(
            go.Scatter(
                x=_sweep_vals, y=_y_sweep,
                mode="lines",
                line=dict(width=2),
                name=_p,
                showlegend=False,
            ),
            row=_row, col=_col,
        )
        # Mark current value
        _curr_val = param_sliders[_p].value
        _curr_idx = int((_curr_val - _lo) / (_hi - _lo + 1e-30) * (_n_sweep - 1))
        _curr_idx = max(0, min(_curr_idx, _n_sweep - 1))
        _figs.add_trace(
            go.Scatter(
                x=[_curr_val], y=[_y_sweep[_curr_idx]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="diamond"),
                showlegend=False,
            ),
            row=_row, col=_col,
        )

    _figs.update_layout(
        title="PCE Response Curves (drag sliders above to explore)",
        template="plotly_dark",
        height=600,
    )
    _figs
    return


@app.cell
def _(data, go, make_subplots, mo):
    if data is None:
        mo.stop(True)

    _s2 = data.get("strategy2_by_generation", {})
    _s3 = data.get("strategy3_by_seed", {})
    _s2_gens = _s2.get("generations", [])
    _s3_seeds = _s3.get("seeds", [])

    if not _s2_gens and not _s3_seeds:
        mo.stop(True, mo.md("*No Strategy 2/3 data (need generations >= 2 or seeds >= 2)*"))

    _n_plots = (1 if _s2_gens else 0) + (1 if _s3_seeds else 0)
    _titles = []
    if _s2_gens:
        _titles.append("Strategy 2 — By Generation")
    if _s3_seeds:
        _titles.append("Strategy 3 — By Seed")

    _fig = make_subplots(rows=1, cols=_n_plots, subplot_titles=_titles)
    _col = 1

    if _s2_gens:
        for _gen_data in _s2_gens:
            _gen = _gen_data["generation"]
            _sobol = _gen_data["sobol_total_order"]
            _params = list(_sobol.keys())
            _vals = [_sobol[p] for p in _params]
            _fig.add_trace(
                go.Bar(x=_params, y=_vals, name=f"Gen {_gen}"),
                row=1, col=_col,
            )
        _col += 1

    if _s3_seeds:
        for _seed_data in _s3_seeds:
            _seed = _seed_data["lineage_seed"]
            _sobol = _seed_data["sobol_total_order"]
            _params = list(_sobol.keys())
            _vals = [_sobol[p] for p in _params]
            _fig.add_trace(
                go.Bar(x=_params, y=_vals, name=f"Seed {_seed}"),
                row=1, col=_col,
            )

    _fig.update_layout(
        template="plotly_dark",
        height=400,
        barmode="group",
    )
    _fig
    return


@app.cell
def _(data, json, mo):
    if data is None:
        mo.stop(True)

    _json_str = json.dumps(data, indent=2)

    mo.md("### Exported Results (uq_results.json)")
    mo.ui.code_editor(
        value=_json_str,
        language="json",
    )
    return


if __name__ == "__main__":
    app.run()
