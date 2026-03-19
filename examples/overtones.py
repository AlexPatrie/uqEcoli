import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from itertools import product

    import numpy as np
    import plotly.graph_objects as go

    def compute_dissonance(
        note1_freq: float,
        note2_freq: float,
        n_overtones: int = 8,
        n_points: int = 800,
        freq_range: tuple[float, float] = None,
    ) -> dict:
        """
        Computes the summed dissonance curve across all pairs of
        (note1 fundamentals + overtones) x (note2 fundamentals + overtones).

        Uses the Sethares roughness model:
          roughness peaks when beat frequency ≈ 0.25 * critical bandwidth (Bark scale)

        Parameters
        ----------
        note1_freq, note2_freq : float  — fundamental frequencies in Hz
        n_overtones            : int    — number of overtones per note (including fundamental)
        n_points               : int    — resolution of the frequency axis
        freq_range             : tuple  — (min_hz, max_hz), defaults to span both notes

        Returns
        -------
        dict with keys:
            freqs           : np.ndarray (n_points,)  — frequency axis
            roughness       : np.ndarray (n_points,)  — summed dissonance curve
            partials        : list[dict]              — each partial pair's contribution
            note1_partials  : list[(freq, amp)]
            note2_partials  : list[(freq, amp)]
            coincidences    : set[float]              — shared overtone frequencies
        """

        def overtone_series(fundamental: float, n: int) -> list[tuple[float, float]]:
            return [(fundamental * k, 1.0 / k) for k in range(1, n + 1)]

        def critical_bandwidth(f: float) -> float:
            return 25 + 75 * (1 + 1.4 * (f / 1000) ** 2) ** 0.69

        def sethares_roughness(f1: float, f2: float, a1: float, a2: float, freqs: np.ndarray) -> np.ndarray:
            f_low = min(f1, f2)
            cbw = critical_bandwidth(f_low)
            peak = 0.25 * cbw
            beat = np.abs(freqs - f2)
            roughness = (
                a1
                * a2
                * (np.exp(-3.5 * beat / peak) - np.exp(-5.75 * beat / peak))
                * np.where(beat < cbw * 1.2, 1.0, 0.0)
            )
            return np.clip(roughness, 0, None)

        note1_partials = overtone_series(note1_freq, n_overtones)
        note2_partials = overtone_series(note2_freq, n_overtones)

        all_freqs = [f for f, _ in note1_partials + note2_partials]
        lo = freq_range[0] if freq_range else max(20, min(all_freqs) * 0.5)
        hi = freq_range[1] if freq_range else max(all_freqs) * 1.5
        freqs = np.linspace(lo, hi, n_points)

        summed = np.zeros(n_points)
        partials = []

        for (f1, a1), (f2, a2) in product(note1_partials, note2_partials):
            curve = sethares_roughness(f1, f2, a1, a2, freqs)
            summed += curve
            partials.append(dict(f1=f1, a1=a1, f2=f2, a2=a2, curve=curve))

        # --- detect coincidences ---
        n1_freqs = {round(f, 2) for f, _ in note1_partials}
        n2_freqs = {round(f, 2) for f, _ in note2_partials}
        coincidences = n1_freqs & n2_freqs

        return dict(
            freqs=freqs,
            roughness=summed,
            partials=partials,
            note1_partials=note1_partials,
            note2_partials=note2_partials,
            note1_freq=note1_freq,
            note2_freq=note2_freq,
            coincidences=coincidences,
        )

    def plot_dissonance(dissonance_data: dict, show_partials: bool = False, partial_points: int = 200) -> go.Figure:
        """
        Plots the summed dissonance curve with individual partial-pair
        contributions, vertical markers for all overtone positions,
        and highlighted coincidence points where overtones align.

        Parameters
        ----------
        dissonance_data : dict  — output of compute_dissonance()
        show_partials   : bool  — overlay individual partial-pair curves
        partial_points  : int   — downsample resolution for partial curves
        """
        freqs = dissonance_data["freqs"]
        roughness = dissonance_data["roughness"]
        n1 = dissonance_data["note1_freq"]
        n2 = dissonance_data["note2_freq"]
        coincidences = dissonance_data["coincidences"]

        step = max(1, len(freqs) // partial_points)
        f_ds = freqs[::step]

        fig = go.Figure()

        # --- individual partial pair curves (faint, downsampled) ---
        if show_partials:
            for p in dissonance_data["partials"]:
                if p["curve"].max() < 1e-6:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=f_ds,
                        y=p["curve"][::step],
                        mode="lines",
                        line=dict(width=0.8, color="rgba(255,100,255,0.18)"),
                        hovertemplate=f"f1={p['f1']:.1f}Hz × f2={p['f2']:.1f}Hz<extra></extra>",
                        showlegend=False,
                    )
                )

        # --- summed roughness curve ---
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=roughness,
                mode="lines",
                name="Total roughness",
                line=dict(color="magenta", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255,0,255,0.12)",
            )
        )

        # --- note1 overtone vlines (cyan) ---
        for i, (f, a) in enumerate(dissonance_data["note1_partials"]):
            if round(f, 2) in coincidences:
                continue  # skip — will be drawn as coincidence instead
            fig.add_vline(
                x=f,
                line=dict(color="cyan", width=1.2, dash="dot"),
                annotation=dict(
                    text=f"n1·{i + 1}<br>{f:.0f}Hz",
                    font=dict(color="cyan", size=10),
                    yref="paper",
                    y=0.98 - (i % 3) * 0.08,
                ),
            )

        # --- note2 overtone vlines (yellow) ---
        for i, (f, a) in enumerate(dissonance_data["note2_partials"]):
            if round(f, 2) in coincidences:
                continue  # skip — will be drawn as coincidence instead
            fig.add_vline(
                x=f,
                line=dict(color="yellow", width=1.2, dash="dot"),
                annotation=dict(
                    text=f"n2·{i + 1}<br>{f:.0f}Hz",
                    font=dict(color="yellow", size=10),
                    yref="paper",
                    y=0.88 - (i % 3) * 0.08,
                ),
            )

        # --- coincidence vlines (white, solid, prominent) ---
        for f in sorted(coincidences):
            fig.add_vline(
                x=f,
                line=dict(color="white", width=2.5, dash="solid"),
                annotation=dict(
                    text=f"⚡ {f:.0f}Hz<br>(shared)",
                    font=dict(color="white", size=11),
                    yref="paper",
                    y=0.5,
                ),
            )

        fig.update_layout(
            title=dict(
                text=f"Dissonance curve — {n1}Hz vs {n2}Hz  |  {len(coincidences)} shared overtone(s)",
                font=dict(color="white", size=16),
            ),
            plot_bgcolor="#0a0a0a",
            paper_bgcolor="#0a0a0a",
            xaxis=dict(
                title="Frequency (Hz)",
                color="white",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.07)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Roughness (a.u.)",
                color="white",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.07)",
                zeroline=False,
            ),
            legend=dict(font=dict(color="white")),
            margin=dict(l=60, r=20, t=60, b=60),
            hovermode="x unified",
        )

        return fig

    return compute_dissonance, plot_dissonance


@app.cell
def _():
    # perfect fifth: A4 vs E5
    # data = compute_dissonance(440, 660, n_overtones=8, n_points=800)  # ← drop from 4000
    # plot_dissonance(data, show_partials=False).show()                  # ← drop individual curves
    return


@app.cell
def _(compute_dissonance, plot_dissonance):
    data = compute_dissonance(440, 660, n_overtones=16, n_points=800)
    plot_dissonance(data, show_partials=False).show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
