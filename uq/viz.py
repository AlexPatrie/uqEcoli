"""
Visualization utilities for Koopman spectral analysis.

The Chladni plate analogy: just as vibrating a metal plate at different
frequencies reveals spatial nodal patterns (where sand collects), DMD
decomposes cellular dynamics into modes — each with a frequency (how fast
it oscillates) and a spatial pattern (which observables participate).

Identifying cell cycle modes is like finding the fundamental resonance of
cell division amid all the other "vibrations" in the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from uq.koopman import CellCycleKoopmanAnalyzer, KoopmanMode, KoopmanSpectrum


def plot_koopman_spectrum(
    spectrum: KoopmanSpectrum,
    expected_cycle_time: float = 3600.0,
    frequency_tolerance: float = 0.3,
    observable_names: list[str] | None = None,
) -> go.Figure:
    """
    Visualize Koopman eigenvalue spectrum with cell cycle mode identification.

    Produces a 4-panel figure:
      1. **Eigenvalues in the complex plane** — unit circle shows stability
         boundary; cell cycle modes highlighted.
      2. **Frequency spectrum** — bar chart of mode amplitudes vs frequency,
         with expected cell cycle frequency and harmonics marked.
      3. **Mode shapes** ("Chladni patterns") — heatmap of how each observable
         participates in each mode, analogous to nodal patterns on a vibrating plate.
      4. **Power spectrum** — log-scale power vs frequency, showing where the
         cell's "energy" concentrates spectrally.

    Args:
        spectrum: KoopmanSpectrum from DMD analysis.
        expected_cycle_time: Expected cell division period in seconds.
        frequency_tolerance: Fractional tolerance for matching cell cycle
            frequency (0.3 = 30%).
        observable_names: Names for the observable axes. Falls back to
            spectrum.observable_names or generic labels.

    Returns:
        Plotly Figure with 4 subplots.
    """
    from uq.koopman import CellCycleKoopmanAnalyzer

    # Identify cell cycle modes
    analyzer = CellCycleKoopmanAnalyzer(
        expected_cycle_time=expected_cycle_time,
        frequency_tolerance=frequency_tolerance,
    )
    cc_modes = analyzer.identify_cell_cycle_modes(spectrum)
    cc_eigenvalues = {m.eigenvalue for m in cc_modes}

    obs_names = observable_names or spectrum.observable_names
    if not obs_names:
        n_obs = spectrum.eigenvectors.shape[0] if spectrum.eigenvectors.ndim == 2 else 1
        obs_names = [f"obs_{i}" for i in range(n_obs)]

    # Shorten long listener-style column names for display
    short_names = [n.split("__")[-1] if "__" in n else n for n in obs_names]

    expected_freq = 1.0 / expected_cycle_time

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Eigenvalues (complex plane)",
            "Frequency spectrum",
            "Mode shapes (Chladni patterns)",
            "Power spectrum",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    # ── Panel 1: Eigenvalues in the complex plane ──

    # Unit circle
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta_circle),
            y=np.sin(theta_circle),
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Unit circle",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # All eigenvalues
    eigs = spectrum.eigenvalues
    is_cc = np.array([e in cc_eigenvalues for e in eigs])

    # Non-cell-cycle modes
    fig.add_trace(
        go.Scatter(
            x=np.real(eigs[~is_cc]),
            y=np.imag(eigs[~is_cc]),
            mode="markers",
            marker=dict(size=8, color="steelblue", opacity=0.6),
            name="Other modes",
            hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Cell cycle modes
    if np.any(is_cc):
        fig.add_trace(
            go.Scatter(
                x=np.real(eigs[is_cc]),
                y=np.imag(eigs[is_cc]),
                mode="markers",
                marker=dict(
                    size=14,
                    color="crimson",
                    symbol="star",
                    line=dict(width=1, color="darkred"),
                ),
                name="Cell cycle modes",
                hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<br>CELL CYCLE<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Re(λ)", row=1, col=1)
    fig.update_yaxes(title_text="Im(λ)", scaleanchor="x", scaleratio=1, row=1, col=1)

    # ── Panel 2: Frequency spectrum (amplitude vs frequency) ──

    freqs = np.array([m.frequency for m in spectrum.modes])
    amps = np.array([np.abs(m.amplitude) for m in spectrum.modes])
    is_cc_mode = np.array([m in cc_modes for m in spectrum.modes])

    # Sort by frequency for display
    sort_idx = np.argsort(freqs)
    freqs_sorted = freqs[sort_idx]
    amps_sorted = amps[sort_idx]
    is_cc_sorted = is_cc_mode[sort_idx]

    colors = ["crimson" if c else "steelblue" for c in is_cc_sorted]

    fig.add_trace(
        go.Bar(
            x=freqs_sorted,
            y=amps_sorted,
            marker_color=colors,
            name="Mode amplitudes",
            showlegend=False,
            hovertemplate="f = %{x:.6f} Hz<br>|a| = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Mark expected cell cycle frequency and harmonics
    for harmonic in range(1, 5):
        f_h = harmonic * expected_freq
        label = "f₀" if harmonic == 1 else f"{harmonic}f₀"
        fig.add_vline(
            x=f_h,
            line=dict(color="crimson", dash="dot", width=1.5),
            row=1,
            col=2,
        )
        fig.add_annotation(
            x=f_h,
            y=max(amps) * 1.05,
            text=label,
            showarrow=False,
            font=dict(color="crimson", size=10),
            xref="x2",
            yref="y2",
        )

    # Shade tolerance bands around expected frequency
    for harmonic in range(1, 5):
        f_h = harmonic * expected_freq
        f_lo = f_h * (1 - frequency_tolerance)
        f_hi = f_h * (1 + frequency_tolerance)
        fig.add_vrect(
            x0=f_lo,
            x1=f_hi,
            fillcolor="crimson",
            opacity=0.08,
            line_width=0,
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="|Amplitude|", row=1, col=2)

    # ── Panel 3: Mode shapes — the "Chladni patterns" ──
    #
    # Each column is a mode, each row is an observable. The value is how
    # strongly that observable participates in that mode — analogous to the
    # displacement pattern on a vibrating plate at a given resonant frequency.

    n_display = min(len(spectrum.modes), 12)
    top_modes = sorted(spectrum.modes, key=lambda m: np.abs(m.amplitude), reverse=True)[:n_display]

    mode_matrix = np.zeros((len(short_names), n_display))
    mode_labels = []
    for j, mode in enumerate(top_modes):
        mode_vec = np.abs(mode.mode[: len(short_names)])
        if np.max(mode_vec) > 0:
            mode_vec = mode_vec / np.max(mode_vec)  # normalize per mode
        mode_matrix[:, j] = mode_vec

        tag = " *" if mode in cc_modes else ""
        if mode.is_oscillatory and mode.period is not None:
            mode_labels.append(f"f={mode.frequency:.5f}{tag}")
        else:
            mode_labels.append(f"decay{tag}")

    fig.add_trace(
        go.Heatmap(
            z=mode_matrix,
            x=mode_labels,
            y=short_names,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="|φ|", x=1.02, len=0.45, y=0.22),
            hovertemplate="Mode: %{x}<br>Observable: %{y}<br>|φ| = %{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Mode (frequency)", tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text="Observable", row=2, col=1)

    # ── Panel 4: Power spectrum (log scale) ──

    ps_freqs, ps_powers = spectrum.get_power_spectrum()

    fig.add_trace(
        go.Scatter(
            x=ps_freqs,
            y=ps_powers,
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
            name="Power",
            showlegend=False,
            hovertemplate="f = %{x:.6f} Hz<br>Power = %{y:.4e}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Mark cell cycle frequencies on power spectrum too
    for harmonic in range(1, 5):
        f_h = harmonic * expected_freq
        fig.add_vline(
            x=f_h,
            line=dict(color="crimson", dash="dot", width=1.5),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Power (|a|²)", type="log", row=2, col=2)

    # ── Layout ──

    period_str = f"{expected_cycle_time:.0f}s"
    n_cc = len(cc_modes)
    title = (
        f"Koopman Spectral Decomposition — "
        f"{n_cc} cell cycle mode{'s' if n_cc != 1 else ''} identified "
        f"(T₀ = {period_str}, tol = {frequency_tolerance:.0%})"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=750,
        width=1100,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )

    return fig
