"""Koopman Eigenmode Visualization

Generates synthetic timeseries data matching the structure of real vEcoli
simulation outputs (X parameter samples, Y observable timeseries), runs
the Phase 2 DMD/Koopman decomposition, and visualizes the resulting
eigenmodes, frequency spectrum, mode shapes, and power spectrum.

Run with: uv run marimo run examples/koopman_eigenmodes.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


# ── Imports ──────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    import polars as pl

    from libuq import (
        DynamicModeDecomposition,
        ExtendedDMD,
        KoopmanSpectrum,
        KoopmanMode,
        KoopmanCellCycleVariable,
        CellCycleKoopmanAnalyzer,
        XSpaceVecoli,
    )
    from libuq.wrappers import DataDrivenWrapper
    from libuq.viz import plot_koopman_spectrum

    return (
        DynamicModeDecomposition,
        ExtendedDMD,
        KoopmanCellCycleVariable,
        CellCycleKoopmanAnalyzer,
        XSpaceVecoli,
        DataDrivenWrapper,
        plot_koopman_spectrum,
        alt,
        mo,
        np,
        pd,
        pl,
    )


# ── Title ────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    # Koopman Eigenmode Visualization

    This notebook generates **synthetic timeseries data** that matches the
    structure produced by real vEcoli simulations, runs DMD/Koopman
    decomposition (Phase 2 of the UQ pipeline), and visualizes the
    resulting eigenmodes.

    **What you'll see:**

    1. Raw synthetic timeseries from `DataDrivenWrapper`
    2. Eigenvalue spectrum in the complex plane (unit circle = stability boundary)
    3. Frequency spectrum with cell cycle harmonic identification
    4. Mode shapes — which observables participate in each eigenmode
    5. Power spectrum — spectral energy distribution
    6. Koopman cell cycle variable extraction
    7. The full 4-panel `plot_koopman_spectrum()` figure from the pipeline
    """)
    return


# ── Configuration sliders ────────────────────────────────────────────────
@app.cell
def _(mo):
    config_sliders = mo.ui.array([
        mo.ui.slider(50, 500, value=200, step=10, label="n_timesteps"),
        mo.ui.slider(2, 12, value=8, step=1, label="DMD rank"),
        mo.ui.slider(10.0, 200.0, value=60.0, step=5.0, label="Expected cycle time"),
    ])
    mo.md(f"""
    ## Configuration

    Adjust the parameters below to see how they affect the spectral decomposition.

    {config_sliders}
    """)
    return (config_sliders,)


# ── Build parameter space & synthetic wrapper ────────────────────────────
@app.cell
def _(DataDrivenWrapper, XSpaceVecoli, np):
    # Observable names matching real vEcoli output columns
    observable_names = [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__protein_mass",
        "listeners__fba__exchange_fluxes",
        "listeners__rna_counts__mRNA_cistron",
        "listeners__monomer_counts",
    ]

    # Build parameter space (3 continuous input parameters)
    param_space = XSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
    )

    # Synthetic observable statistics (realistic scale)
    _obs_means = np.array([1.2e-12, 3.6e-12, 8.0e-13, 0.015, 450.0, 1200.0])
    _obs_stds = np.array([2.0e-13, 5.0e-13, 1.5e-13, 0.003, 80.0, 200.0])

    wrapper = DataDrivenWrapper(
        parameter_space=param_space,
        observable_means=_obs_means,
        observable_stds=_obs_stds,
        seed=42,
        n_timesteps=200,
    )

    return observable_names, param_space, wrapper


# ── Generate synthetic timeseries ────────────────────────────────────────
@app.cell
def _(observable_names, config_sliders, alt, mo, np, param_space, pd, wrapper):
    n_timesteps = int(config_sliders.value[0])
    dmd_rank = int(config_sliders.value[1])
    expected_cycle_time = float(config_sliders.value[2])

    # Generate a synthetic trajectory with explicit cell-cycle-like dynamics.
    # This mirrors how real vEcoli outputs look: exponential growth with
    # periodic modulation at the cell cycle frequency plus harmonics, and
    # per-observable phase offsets.
    _rng = np.random.default_rng(42)
    _t = np.arange(n_timesteps, dtype=float)
    _T_cycle = expected_cycle_time  # cell cycle period in timesteps
    _omega = 2 * np.pi / _T_cycle

    # Base signals: mean + slow growth + cell cycle oscillation + 2nd harmonic + noise.
    # Each observable has a random phase offset, mimicking how different cellular
    # processes (mass, mRNA, protein, fluxes) oscillate with the cell cycle but
    # with different phase relationships.
    _obs_means = np.array([1.2e-12, 3.6e-12, 8.0e-13, 0.015, 450.0, 1200.0])
    _obs_stds = np.array([2.0e-13, 5.0e-13, 1.5e-13, 0.003, 80.0, 200.0])
    _n_obs = len(observable_names)
    _phases = _rng.uniform(0, 2 * np.pi, _n_obs)

    _ts_cols = []
    for _i in range(_n_obs):
        # Slow exponential growth (weak relative to oscillation)
        _growth = _obs_means[_i] * np.exp(0.001 * _t)
        # Fundamental cell cycle oscillation (strong)
        _cc_fund = _obs_stds[_i] * 1.0 * np.sin(_omega * _t + _phases[_i])
        # 2nd harmonic (moderate)
        _cc_harm2 = _obs_stds[_i] * 0.4 * np.sin(2 * _omega * _t + _phases[_i] * 1.5)
        # Measurement noise
        _noise = _obs_stds[_i] * 0.02 * _rng.normal(size=n_timesteps)
        _ts_cols.append(_growth + _cc_fund + _cc_harm2 + _noise)

    timeseries = np.column_stack(_ts_cols)  # (n_timesteps, n_obs)

    # Also generate X samples and Y batch via the DataDrivenWrapper for context
    wrapper.n_timesteps = n_timesteps
    _rng2 = np.random.default_rng(99)
    n_batch = 30
    X_samples = np.column_stack([_rng2.uniform(lo, hi, n_batch) for lo, hi in param_space.bounds_array])
    Y_batch = wrapper.evaluate_batch(X_samples)

    # Normalize timeseries for visualization
    _t = np.arange(n_timesteps)
    _rows = []
    for _i, _name in enumerate(observable_names):
        _vals = timeseries[:, _i]
        _norm = (_vals - _vals.mean()) / (_vals.std() + 1e-30)
        for _j in range(n_timesteps):
            _rows.append({
                "time": int(_t[_j]),
                "value": float(_norm[_j]),
                "observable": _name.split("__")[-1],
            })

    _traj_df = pd.DataFrame(_rows)

    traj_chart = (
        alt.Chart(_traj_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="Timestep"),
            y=alt.Y("value:Q", title="Normalized value"),
            color=alt.Color(
                "observable:N",
                scale=alt.Scale(scheme="category10"),
                legend=alt.Legend(title="Observable"),
            ),
            strokeDash=alt.StrokeDash("observable:N"),
        )
        .properties(width=700, height=300, title="Synthetic Timeseries (DataDrivenWrapper)")
    )

    mo.md(f"""
    ## Synthetic Timeseries

    Generated via `DataDrivenWrapper` with sinusoidal cell-cycle-like
    modulation. Shape: **({n_timesteps}, {len(observable_names)})** —
    matches what a real simulation would produce.

    - **X** (input parameters): {param_space.parameter_names} — {X_samples.shape[0]} LHS samples
    - **Y** (observable means): shape {Y_batch.shape}
    """)
    return (
        X_samples,
        Y_batch,
        dmd_rank,
        expected_cycle_time,
        n_batch,
        n_timesteps,
        timeseries,
        traj_chart,
    )


@app.cell
def _(traj_chart):
    traj_chart
    return


# ── DMD decomposition ───────────────────────────────────────────────────
@app.cell
def _(
    DynamicModeDecomposition,
    observable_names,
    dmd_rank,
    mo,
    np,
    timeseries,
):
    dmd = DynamicModeDecomposition(rank=dmd_rank, dt=1.0)
    dmd.fit(timeseries)
    spectrum = dmd.get_spectrum(observable_names=observable_names)

    _osc_modes = spectrum.get_oscillatory_modes()
    _dom_modes = spectrum.get_dominant_modes(3)

    mo.md(f"""
    ## DMD Decomposition

    Fitted DMD with **rank = {dmd_rank}** to the synthetic timeseries.

    | Property | Value |
    |----------|-------|
    | Eigenvalues | {len(spectrum.eigenvalues)} |
    | Oscillatory modes | {len(_osc_modes)} |
    | Dominant mode frequency | {_dom_modes[0].frequency:.6f} Hz |
    | Dominant mode period | {_dom_modes[0].period:.2f} timesteps |
    | Dominant mode \\|amplitude\\| | {np.abs(_dom_modes[0].amplitude):.4f} |
    """)
    return dmd, spectrum


# ── Panel 1: Eigenvalues in the complex plane ────────────────────────────
@app.cell
def _(
    CellCycleKoopmanAnalyzer,
    alt,
    expected_cycle_time,
    mo,
    np,
    pl,
    spectrum,
):
    _analyzer = CellCycleKoopmanAnalyzer(
        expected_cycle_time=expected_cycle_time,
        frequency_tolerance=0.3,
    )
    cc_modes = _analyzer.identify_cell_cycle_modes(spectrum)
    _cc_eigs = {m.eigenvalue for m in cc_modes}

    _eigs = spectrum.eigenvalues
    _is_cc = np.array([e in _cc_eigs for e in _eigs])

    # Unit circle
    _theta = np.linspace(0, 2 * np.pi, 200)
    _circle_df = pl.DataFrame({"x": np.cos(_theta), "y": np.sin(_theta)})

    _unit_circle = (
        alt.Chart(_circle_df).mark_line(strokeDash=[5, 5], color="gray", opacity=0.5).encode(x="x:Q", y="y:Q")
    )

    # Eigenvalue points
    _eig_records = []
    for _i, _e in enumerate(_eigs):
        _eig_records.append({
            "Re": float(_e.real),
            "Im": float(_e.imag),
            "magnitude": float(np.abs(_e)),
            "is_cell_cycle": bool(_is_cc[_i]),
            "label": f"Mode {_i + 1}",
        })
    _eig_df = pl.DataFrame(_eig_records)

    _other_pts = (
        alt.Chart(_eig_df.filter(pl.col("is_cell_cycle").not_()))
        .mark_circle(size=120, opacity=0.7)
        .encode(
            x=alt.X("Re:Q", title="Re(lambda)", scale=alt.Scale(domain=[-1.5, 1.5])),
            y=alt.Y("Im:Q", title="Im(lambda)", scale=alt.Scale(domain=[-1.5, 1.5])),
            color=alt.Color(
                "magnitude:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="|lambda|"),
            ),
            tooltip=["label", "Re", "Im", "magnitude"],
        )
    )

    _cc_pts = (
        alt.Chart(_eig_df.filter(pl.col("is_cell_cycle")))
        .mark_point(size=250, shape="diamond", filled=True, color="crimson")
        .encode(
            x="Re:Q",
            y="Im:Q",
            tooltip=["label", "Re", "Im", "magnitude"],
        )
    )

    _labels = alt.Chart(_eig_df).mark_text(dx=12, fontSize=10).encode(x="Re:Q", y="Im:Q", text="label:N")

    eigenvalue_chart = (_unit_circle + _other_pts + _cc_pts + _labels).properties(
        width=450,
        height=450,
        title="Eigenvalues in the Complex Plane",
    )

    mo.md(f"""
    ## Eigenvalue Spectrum

    - **{len(cc_modes)}** cell cycle mode(s) identified (red diamonds)
    - Dashed circle = unit circle (stability boundary)
    - Inside unit circle = decaying; outside = growing; on = persistent
    """)
    return cc_modes, eigenvalue_chart


@app.cell
def _(eigenvalue_chart):
    eigenvalue_chart
    return


# ── Panel 2: Frequency spectrum with harmonics ───────────────────────────
@app.cell
def _(alt, cc_modes, expected_cycle_time, mo, np, pl, spectrum):
    _expected_freq = 1.0 / expected_cycle_time
    _freqs = np.array([m.frequency for m in spectrum.modes])
    _amps = np.array([np.abs(m.amplitude) for m in spectrum.modes])
    _is_cc = np.array([m in cc_modes for m in spectrum.modes])

    _sort = np.argsort(_freqs)
    _freq_df = pl.DataFrame({
        "frequency": _freqs[_sort],
        "amplitude": _amps[_sort],
        "is_cell_cycle": _is_cc[_sort].tolist(),
    })

    _bars = (
        alt.Chart(_freq_df)
        .mark_bar(width=8)
        .encode(
            x=alt.X("frequency:Q", title="Frequency (Hz)"),
            y=alt.Y("amplitude:Q", title="|Amplitude|"),
            color=alt.condition(
                alt.datum.is_cell_cycle,
                alt.value("crimson"),
                alt.value("steelblue"),
            ),
            tooltip=["frequency", "amplitude", "is_cell_cycle"],
        )
    )

    # Harmonic reference lines
    _harm_df = pl.DataFrame({
        "freq": [_expected_freq * h for h in range(1, 5)],
        "label": ["f0", "2f0", "3f0", "4f0"],
    })

    _rules = alt.Chart(_harm_df).mark_rule(strokeDash=[4, 4], color="crimson", opacity=0.6).encode(x="freq:Q")

    _harm_labels = (
        alt.Chart(_harm_df).mark_text(dy=-10, color="crimson", fontSize=11).encode(x="freq:Q", text="label:N")
    )

    freq_chart = (_bars + _rules + _harm_labels).properties(
        width=600,
        height=300,
        title="Frequency Spectrum with Cell Cycle Harmonics",
    )

    mo.md(f"""
    ## Frequency Spectrum

    Expected fundamental: **f0 = {_expected_freq:.6f} Hz**
    (cycle time = {expected_cycle_time:.0f} timesteps).
    Red bars = cell cycle harmonics; blue = other modes.
    """)
    return (freq_chart,)


@app.cell
def _(freq_chart):
    freq_chart
    return


# ── Panel 3: Mode shapes (Chladni patterns) ─────────────────────────────
@app.cell
def _(observable_names, alt, cc_modes, mo, np, pl, spectrum):
    _short_names = [n.split("__")[-1] for n in observable_names]
    _n_display = min(len(spectrum.modes), 8)
    _top_modes = sorted(spectrum.modes, key=lambda m: np.abs(m.amplitude), reverse=True)[:_n_display]

    _rows = []
    for _j, _mode in enumerate(_top_modes):
        _vec = np.abs(_mode.mode[: len(_short_names)])
        _max_val = np.max(_vec) if np.max(_vec) > 0 else 1.0
        _vec_norm = _vec / _max_val

        _tag = " *" if _mode in cc_modes else ""
        if _mode.is_oscillatory and _mode.period is not None:
            _mlabel = f"f={_mode.frequency:.5f}{_tag}"
        else:
            _mlabel = f"decay{_tag}"

        for _i, _obs in enumerate(_short_names):
            _rows.append({
                "observable": _obs,
                "mode": _mlabel,
                "mode_idx": _j,
                "amplitude": float(_vec_norm[_i]),
            })

    _mode_df = pl.DataFrame(_rows)
    _mode_order = sorted(
        _mode_df["mode"].unique().to_list(),
        key=lambda m: next(r["mode_idx"] for r in _rows if r["mode"] == m),
    )

    _heatmap = (
        alt.Chart(_mode_df)
        .mark_rect()
        .encode(
            x=alt.X("mode:N", title="Mode (frequency)", sort=_mode_order),
            y=alt.Y("observable:N", title="Observable", sort=_short_names),
            color=alt.Color(
                "amplitude:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                legend=alt.Legend(title="|phi|"),
            ),
            tooltip=[
                "observable",
                "mode",
                alt.Tooltip("amplitude:Q", format=".3f"),
            ],
        )
    )

    _text = (
        alt.Chart(_mode_df)
        .mark_text(fontSize=11, color="white")
        .encode(
            x=alt.X("mode:N", sort=_mode_order),
            y=alt.Y("observable:N", sort=_short_names),
            text=alt.Text("amplitude:Q", format=".2f"),
        )
    )

    mode_shape_chart = (_heatmap + _text).properties(
        width=500,
        height=300,
        title="Mode Shapes: Observable Participation in Each Eigenmode",
    )

    mo.md("""
    ## Mode Shapes (Chladni Patterns)

    Each column is a Koopman eigenmode; each row is an observable.
    The value indicates how strongly that observable participates in
    the mode. Modes marked with `*` are identified cell cycle harmonics.

    High values (yellow) = strong participation. This reveals which
    observables oscillate together at each frequency.
    """)
    return (mode_shape_chart,)


@app.cell
def _(mode_shape_chart):
    mode_shape_chart
    return


# ── Panel 4: Power spectrum ─────────────────────────────────────────────
@app.cell
def _(alt, expected_cycle_time, mo, np, pl, spectrum):
    _ps_freqs, _ps_powers = spectrum.get_power_spectrum()
    _ps_df = pl.DataFrame({
        "frequency": _ps_freqs,
        "power": _ps_powers,
    })

    _power_line = (
        alt.Chart(_ps_df)
        .mark_line(color="steelblue", strokeWidth=2, point=True)
        .encode(
            x=alt.X("frequency:Q", title="Frequency (Hz)"),
            y=alt.Y("power:Q", title="Power (|a|^2)", scale=alt.Scale(type="log")),
            tooltip=["frequency", alt.Tooltip("power:Q", format=".4e")],
        )
    )

    _expected_freq = 1.0 / expected_cycle_time
    _cc_lines_df = pl.DataFrame({
        "freq": [_expected_freq * h for h in range(1, 5)],
    })
    _cc_rules = alt.Chart(_cc_lines_df).mark_rule(strokeDash=[4, 4], color="crimson", opacity=0.6).encode(x="freq:Q")

    power_chart = (_power_line + _cc_rules).properties(
        width=600,
        height=300,
        title="Power Spectrum (log scale)",
    )

    mo.md("""
    ## Power Spectrum

    Log-scale power vs frequency. Peaks show where the system's
    "energy" concentrates spectrally. Dashed red lines mark expected
    cell cycle harmonics.
    """)
    return (power_chart,)


@app.cell
def _(power_chart):
    power_chart
    return


# ── Mode properties: energy and dynamics ─────────────────────────────────
@app.cell
def _(observable_names, alt, mo, np, pl, spectrum):
    _short = [n.split("__")[-1] for n in observable_names]
    _props = []
    for _idx, _m in enumerate(spectrum.modes[:8]):
        _energy = float(np.abs(_m.amplitude) ** 2)
        _mode_abs = np.abs(_m.mode[: len(_short)])
        _dom_idx = int(np.argmax(_mode_abs)) if len(_mode_abs) > 0 else 0
        _dom_name = _short[_dom_idx] if _dom_idx < len(_short) else "N/A"
        _props.append({
            "mode": f"Mode {_idx + 1}",
            "frequency": abs(_m.frequency),
            "energy": _energy,
            "growth_rate": _m.growth_rate,
            "dominant_observable": _dom_name,
            "stable": _m.is_stable,
        })

    _props_df = pl.DataFrame(_props)

    energy_bars = (
        alt.Chart(_props_df)
        .mark_bar()
        .encode(
            x=alt.X("mode:N", title="Mode", sort=[f"Mode {i + 1}" for i in range(8)]),
            y=alt.Y("energy:Q", title="Energy (|a|^2)"),
            color=alt.Color("dominant_observable:N", legend=alt.Legend(title="Dominant\nObservable")),
            tooltip=["mode", "energy", "frequency", "dominant_observable"],
        )
        .properties(width=350, height=250, title="Mode Energy Distribution")
    )

    freq_decay = (
        alt.Chart(_props_df)
        .mark_circle(size=200)
        .encode(
            x=alt.X("frequency:Q", title="Frequency"),
            y=alt.Y("growth_rate:Q", title="Growth Rate (neg = decay)"),
            color=alt.Color("energy:Q", scale=alt.Scale(scheme="viridis")),
            size=alt.Size("energy:Q", scale=alt.Scale(range=[50, 400])),
            tooltip=["mode", "frequency", "growth_rate", "energy", "stable"],
        )
        .properties(width=350, height=250, title="Frequency vs Growth Rate")
    )

    mo.md("""
    ## Mode Properties

    **Left**: Energy per mode — higher = more variance explained.
    **Right**: Frequency vs growth rate — stable modes decay (negative growth),
    persistent oscillations sit near zero.
    """)
    return energy_bars, freq_decay


@app.cell
def _(energy_bars, freq_decay):
    energy_bars | freq_decay
    return


# ── Koopman cell cycle variable ─────────────────────────────────────────
@app.cell
def _(
    KoopmanCellCycleVariable,
    observable_names,
    alt,
    expected_cycle_time,
    mo,
    np,
    pd,
    pl,
    timeseries,
):
    # Build a polars DataFrame matching what KoopmanCellCycleVariable expects
    _n_t = timeseries.shape[0]
    _cc_data = {col: timeseries[:, i] for i, col in enumerate(observable_names)}
    _cc_data["time"] = np.arange(_n_t)
    _cc_data["generation"] = np.zeros(_n_t, dtype=int)
    _cc_data["agent_id"] = ["cell_0"] * _n_t
    _cc_df = pl.DataFrame(_cc_data)

    koopman_cc = KoopmanCellCycleVariable(
        expected_cycle_time=expected_cycle_time,
        frequency_tolerance=0.3,
        use_edmd=True,
        observable_columns=observable_names[:2],  # mass columns
    )

    cc_result = koopman_cc.compute(_cc_df)

    _cc_vals = cc_result.values if hasattr(cc_result, "values") else cc_result.parameters
    _phase_labels = (
        cc_result.phase_labels
        if hasattr(cc_result, "phase_labels") and cc_result.phase_labels is not None
        else np.full(len(_cc_vals), "unknown")
    )

    _cc_vis_df = pd.DataFrame({
        "time": np.arange(len(_cc_vals)),
        "theta": _cc_vals,
        "phase": _phase_labels,
    })

    theta_chart = (
        alt.Chart(_cc_vis_df)
        .mark_circle(size=40, opacity=0.8)
        .encode(
            x=alt.X("time:Q", title="Timestep"),
            y=alt.Y("theta:Q", title="Cell Cycle Variable (theta)"),
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["B_period", "C_period", "D_period", "unknown"],
                    range=["#3498db", "#e74c3c", "#2ecc71", "#999"],
                ),
                legend=alt.Legend(title="Phase"),
            ),
            tooltip=["time", "theta", "phase"],
        )
        .properties(
            width=700,
            height=250,
            title="Koopman Cell Cycle Variable (theta) over Time",
        )
    )

    theta_hist = (
        alt.Chart(_cc_vis_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("theta:Q", bin=alt.Bin(maxbins=25), title="theta"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["B_period", "C_period", "D_period", "unknown"],
                    range=["#3498db", "#e74c3c", "#2ecc71", "#999"],
                ),
            ),
        )
        .properties(width=700, height=200, title="Distribution of theta by Phase")
    )

    _mode_info = ""
    if koopman_cc.cell_cycle_mode is not None:
        _m = koopman_cc.cell_cycle_mode
        _mode_info = f"""
    | Property | Value |
    |----------|-------|
    | Frequency | {_m.frequency:.6f} Hz |
    | Period | {_m.period:.2f} timesteps |
    | Growth rate | {_m.growth_rate:.6f} |
    | Is oscillatory | {_m.is_oscillatory} |
    | Is stable | {_m.is_stable} |
    """

    mo.md(f"""
    ## Koopman Cell Cycle Variable

    `KoopmanCellCycleVariable` projects the timeseries onto the
    identified cell cycle eigenmode and extracts the eigenfunction
    phase as a cell cycle coordinate in [0, 1].

    - **theta range**: [{_cc_vals.min():.3f}, {_cc_vals.max():.3f}]
    - **theta mean**: {_cc_vals.mean():.3f}

    ### Identified Cell Cycle Mode
    {_mode_info}
    """)
    return theta_chart, theta_hist, cc_result, koopman_cc


@app.cell
def _(theta_chart):
    theta_chart
    return


@app.cell
def _(theta_hist):
    theta_hist
    return


# ── Reconstruction quality ───────────────────────────────────────────────
@app.cell
def _(observable_names, alt, mo, np, pd, spectrum, timeseries):
    _t_recon = np.arange(timeseries.shape[0]).astype(float)
    _reconstructed = spectrum.reconstruct(_t_recon)

    _short = [n.split("__")[-1] for n in observable_names]
    _recon_rows = []
    for _i in range(min(3, len(_short))):
        _orig = timeseries[:, _i]
        _rec = _reconstructed[:, _i]
        # Normalize both for comparison
        _o_norm = (_orig - _orig.mean()) / (_orig.std() + 1e-30)
        _r_norm = (_rec - _rec.mean()) / (_rec.std() + 1e-30)
        for _j in range(len(_t_recon)):
            _recon_rows.append({
                "time": int(_j),
                "value": float(_o_norm[_j]),
                "series": f"{_short[_i]} (original)",
            })
            _recon_rows.append({
                "time": int(_j),
                "value": float(_r_norm[_j]),
                "series": f"{_short[_i]} (reconstructed)",
            })

    _recon_df = pd.DataFrame(_recon_rows)

    recon_chart = (
        alt.Chart(_recon_df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("time:Q", title="Timestep"),
            y=alt.Y("value:Q", title="Normalized value"),
            color=alt.Color("series:N", legend=alt.Legend(title="Series")),
            strokeDash=alt.StrokeDash(
                "series:N",
                legend=None,
            ),
        )
        .properties(
            width=700,
            height=250,
            title="DMD Reconstruction vs Original (first 3 observables)",
        )
    )

    mo.md("""
    ## Reconstruction Quality

    How well do the extracted Koopman eigenmodes reconstruct the
    original signal? Solid vs dashed lines show original vs
    reconstructed timeseries.
    """)
    return (recon_chart,)


@app.cell
def _(recon_chart):
    recon_chart
    return


# ── Full pipeline 4-panel Plotly figure ──────────────────────────────────
@app.cell
def _(
    observable_names,
    expected_cycle_time,
    koopman_cc,
    mo,
    plot_koopman_spectrum,
):
    _spec = koopman_cc.spectrum
    if _spec is None:
        mo.md(
            "**No spectrum available from KoopmanCellCycleVariable** — "
            "the cell cycle mode may not have been identified."
        )
        plotly_fig = None
    else:
        plotly_fig = plot_koopman_spectrum(
            spectrum=_spec,
            expected_cycle_time=expected_cycle_time,
            frequency_tolerance=0.3,
            observable_names=observable_names[:2],
        )
        mo.md("""
        ## Pipeline 4-Panel Figure

        This is the same `plot_koopman_spectrum()` output that the full UQ
        pipeline exports as `koopman_spectrum.pdf`. It shows:

        1. **Eigenvalues** — complex plane with unit circle
        2. **Frequency spectrum** — amplitude bars with harmonic markers
        3. **Mode shapes** — Chladni-pattern heatmap
        4. **Power spectrum** — log-scale energy distribution
        """)
    return (plotly_fig,)


@app.cell
def _(plotly_fig):
    plotly_fig
    return


# ── Summary ──────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ## Summary

    This notebook demonstrated the **Phase 2 Koopman/DMD decomposition**
    from the UQ pipeline using synthetic data:

    1. **`DataDrivenWrapper`** generates timeseries matching real vEcoli
       output structure (sinusoidal cell-cycle modulation)
    2. **`DynamicModeDecomposition`** extracts eigenvalues, eigenvectors,
       and amplitudes from the timeseries
    3. **`CellCycleKoopmanAnalyzer`** identifies which modes correspond
       to cell cycle harmonics
    4. **`KoopmanCellCycleVariable`** projects onto the cell cycle mode
       and extracts the eigenfunction phase as theta in [0, 1]
    5. **`plot_koopman_spectrum()`** produces the pipeline's 4-panel
       spectral visualization

    ### Key Takeaway

    The Koopman approach is **data-driven** — it finds periodic structure
    without assumptions about growth mechanisms. The eigenmode with
    frequency closest to 1/T_cycle captures the cell division rhythm,
    and its phase angle serves as the cell cycle coordinate for Phase 2
    stratified sensitivity analysis.
    """)
    return


if __name__ == "__main__":
    app.run()
