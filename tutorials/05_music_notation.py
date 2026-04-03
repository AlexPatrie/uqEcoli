"""
Tutorial 05: Musical Notation for Koopman Spectral Analysis

This tutorial demonstrates the music package, which provides a bijective mapping
between Western musical notation and Koopman spectral analysis of cellular dynamics.

Key concepts:
- Pitch ↔ Frequency (logarithmic mapping)
- Duration ↔ Mode stability (inverse growth rate)
- Dynamics ↔ Amplitude (normalized 0-1)
- Articulation ↔ Growth/decay sign
- Lossless vs Lossy encoding
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Musical Notation for Koopman Spectral Analysis

    This tutorial explores the **music** package, which implements a formal mapping
    between Western musical notation and Koopman spectral analysis.

    ## The Core Insight

    Koopman modes are like musical harmonics:
    - **Frequency** → Pitch (logarithmic, like musical intervals)
    - **Amplitude** → Dynamics (loudness)
    - **Growth/Decay Rate** → Note duration and articulation
    - **Mode Shape** → Orchestration (which observables participate)

    This mapping enables us to:
    1. Visualize cellular dynamics as musical scores
    2. Leverage human musical cognition for pattern recognition
    3. Create a human-readable "program" for UQ analysis
    """)
    return


@app.cell
def _():
    import numpy as np
    import altair as alt
    import polars as pl

    return alt, np, pl


@app.cell
def _():
    # Import the music package
    from apollo.types import (
        CellularNote,
        CellularScore,
        CellularStaff,
        Clef,
        Dynamic,
        NoteValue,
        Articulation,
        ScoreMetadata,
        LosslessMode,
        LosslessScore,
        ObservableClass,
        Tempo,
        KeySignature,
    )
    from apollo.mappings import (
        frequency_to_pitch,
        pitch_to_frequency,
        amplitude_to_dynamic,
        dynamic_to_amplitude,
        growth_rate_to_duration,
        growth_rate_to_articulation,
        cycle_time_to_tempo,
    )
    from apollo.encoding import (
        encode_spectrum,
        encode_spectrum_lossless,
        encode_mode,
    )
    from apollo.decoding import (
        decode_score,
        decode_score_lossless,
        verify_reconstruction,
    )

    return (
        Dynamic,
        NoteValue,
        amplitude_to_dynamic,
        decode_score,
        decode_score_lossless,
        dynamic_to_amplitude,
        encode_spectrum,
        encode_spectrum_lossless,
        frequency_to_pitch,
        growth_rate_to_articulation,
        growth_rate_to_duration,
        pitch_to_frequency,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1: Frequency ↔ Pitch Mapping

    The fundamental mapping uses the logarithmic relationship between frequency
    and musical pitch. Just as musical intervals are perceived logarithmically,
    Koopman mode frequencies map to pitches via:

    ```
    MIDI = 12 × log₂(f / f₀) + 69
    ```

    where f₀ is the fundamental frequency (maps to A4 = 440 Hz in music,
    or the cell cycle frequency in cellular dynamics).
    """)
    return


@app.cell
def _(frequency_to_pitch, pitch_to_frequency):
    # Demonstrate frequency to pitch mapping
    _cell_cycle_time = 3600.0  # 1 hour cell cycle
    _fundamental_freq = 1.0 / _cell_cycle_time

    # Create harmonic series
    _harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
    _frequencies = [n * _fundamental_freq for n in _harmonics]
    _pitches = [frequency_to_pitch(f, _fundamental_freq) for f in _frequencies]

    print("Harmonic Series (Cell Cycle Time = 1 hour)")
    print("=" * 50)
    for h, f, p in zip(_harmonics, _frequencies, _pitches):
        print(f"Harmonic {h}×: f = {f * 1e6:.2f} µHz → Pitch: {p}")

    # Verify roundtrip
    print("\n" + "=" * 50)
    print("Roundtrip verification:")
    for p in _pitches:
        f_recovered = pitch_to_frequency(p, _fundamental_freq)
        print(f"  {p} → {f_recovered * 1e6:.2f} µHz")
    return


@app.cell
def _(alt, np, pl):
    # Visualize the frequency-pitch mapping
    _fundamental = 1.0 / 3600.0
    _freq_range = np.linspace(0.1 * _fundamental, 10 * _fundamental, 100)
    _midi = 12 * np.log2(_freq_range / _fundamental) + 69

    _df = pl.DataFrame({
        "Frequency (relative to fundamental)": _freq_range / _fundamental,
        "MIDI Number": _midi,
        "Octave": (_midi / 12).astype(int) - 1,
    })

    _chart = (
        alt.Chart(_df.to_pandas())
        .mark_line(color="#2563eb")
        .encode(
            x=alt.X("Frequency (relative to fundamental):Q", scale=alt.Scale(type="log")),
            y=alt.Y("MIDI Number:Q"),
            tooltip=["Frequency (relative to fundamental)", "MIDI Number"],
        )
        .properties(
            title="Frequency → MIDI Mapping (Logarithmic)",
            width=500,
            height=300,
        )
    )

    # Add horizontal lines for octaves
    _octave_lines = (
        alt.Chart(
            pl.DataFrame({"y": [57, 69, 81, 93]}).to_pandas()  # A3, A4, A5, A6
        )
        .mark_rule(strokeDash=[5, 5], color="gray")
        .encode(y="y:Q")
    )

    _chart + _octave_lines
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2: Amplitude ↔ Dynamics Mapping

    Musical dynamics (ppp to fff) map to normalized amplitude values.
    This provides a human-intuitive scale for mode importance.

    | Dynamic | Amplitude Range | Interpretation |
    |---------|-----------------|----------------|
    | ppp     | [0.01, 0.025]   | Barely audible |
    | pp      | [0.025, 0.05]   | Very soft |
    | p       | [0.05, 0.10]    | Soft |
    | mp      | [0.10, 0.20]    | Moderately soft |
    | mf      | [0.20, 0.40]    | Moderately loud |
    | f       | [0.40, 0.60]    | Loud |
    | ff      | [0.60, 0.80]    | Very loud |
    | fff     | [0.80, 0.90]    | Extremely loud |
    """)
    return


@app.cell
def _(Dynamic, alt, amplitude_to_dynamic, dynamic_to_amplitude, np, pl):
    # Demonstrate amplitude-dynamic mapping
    _amplitudes = np.linspace(0, 1, 20)
    _dynamics = [amplitude_to_dynamic(a) for a in _amplitudes]
    _recovered = [dynamic_to_amplitude(d) for d in _dynamics]

    _df = pl.DataFrame({
        "Original Amplitude": _amplitudes,
        "Dynamic": [d.value for d in _dynamics],
        "Recovered Amplitude": _recovered,
    })

    # Create step chart showing dynamic regions
    _dynamic_values = list(Dynamic)
    _dynamic_data = []
    for d in _dynamic_values:
        _dynamic_data.append({
            "Dynamic": d.value,
            "Midpoint": dynamic_to_amplitude(d),
        })

    _step_chart = (
        alt.Chart(pl.DataFrame(_dynamic_data).to_pandas())
        .mark_bar(color="#10b981")
        .encode(
            x=alt.X("Dynamic:N", sort=[d.value for d in _dynamic_values]),
            y=alt.Y("Midpoint:Q", title="Amplitude"),
            tooltip=["Dynamic", "Midpoint"],
        )
        .properties(
            title="Dynamic Markings → Amplitude",
            width=500,
            height=250,
        )
    )

    _step_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3: Duration & Articulation ↔ Stability

    Mode stability (inverse of |growth rate|) maps to note duration:
    - **Long notes** (breve, whole) = Very stable modes (small |γ|)
    - **Short notes** (eighth, sixteenth) = Fast-decaying modes (large |γ|)

    The **sign** of the growth rate maps to articulation:
    - **Staccato** (•) = Decaying mode (γ < 0)
    - **Tenuto** (−) = Sustained mode (γ ≈ 0)
    - **Accent** (>) = Growing mode (γ > 0)
    """)
    return


@app.cell
def _(
    NoteValue,
    alt,
    growth_rate_to_articulation,
    growth_rate_to_duration,
    np,
    pl,
):
    # Demonstrate growth rate mapping
    _growth_rates = np.array([-0.3, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.3])
    _durations = [growth_rate_to_duration(g) for g in _growth_rates]
    _articulations = [growth_rate_to_articulation(g) for g in _growth_rates]

    # Duration order for sorting
    _duration_order = {
        NoteValue.BREVE: 0,
        NoteValue.WHOLE: 1,
        NoteValue.HALF: 2,
        NoteValue.QUARTER: 3,
        NoteValue.EIGHTH: 4,
        NoteValue.SIXTEENTH: 5,
        NoteValue.THIRTY_SECOND: 6,
        NoteValue.SIXTY_FOURTH: 7,
    }

    _df = pl.DataFrame({
        "Growth Rate (γ)": _growth_rates,
        "Duration": [d.value for d in _durations],
        "Duration Order": [_duration_order[d] for d in _durations],
        "Articulation": [a.value for a in _articulations],
        "Stability": ["Decaying" if g < -0.001 else "Growing" if g > 0.001 else "Stable" for g in _growth_rates],
    })

    _chart = (
        alt.Chart(_df.to_pandas())
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("Growth Rate (γ):Q", scale=alt.Scale(domain=[-0.35, 0.35])),
            y=alt.Y("Duration Order:Q", title="Note Duration", scale=alt.Scale(reverse=True)),
            color=alt.Color(
                "Stability:N",
                scale=alt.Scale(domain=["Decaying", "Stable", "Growing"], range=["#3b82f6", "#10b981", "#ef4444"]),
            ),
            shape=alt.Shape("Articulation:N"),
            tooltip=["Growth Rate (γ)", "Duration", "Articulation", "Stability"],
        )
        .properties(
            title="Growth Rate → Duration & Articulation",
            width=500,
            height=300,
        )
    )

    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 4: Creating a Synthetic Koopman Spectrum

    Let's create a synthetic Koopman spectrum representing a cell with:
    - A growth mode (non-oscillatory)
    - A fundamental cell cycle mode
    - Several harmonics
    """)
    return


@app.cell
def _(np):
    from libuq.koopman import KoopmanMode, KoopmanSpectrum

    # Create synthetic modes
    _cell_cycle_time = 3600.0  # 1 hour
    _fundamental_freq = 1.0 / _cell_cycle_time
    _dt = 1.0

    _synthetic_modes = []

    # Mode 0: Growth mode (non-oscillatory, slightly growing)
    _growth_mode = KoopmanMode(
        eigenvalue=complex(1.0003, 0.0),  # Slightly > 1 for growth
        mode=np.array([0.9, 0.4, 0.3, 0.5]),
        amplitude=complex(0.85, 0.0),
    )
    _synthetic_modes.append(_growth_mode)

    # Mode 1: Fundamental (1× cell cycle frequency)
    _omega1 = 2 * np.pi * _fundamental_freq
    _fundamental_mode = KoopmanMode(
        eigenvalue=np.exp(-0.005 + 1j * _omega1 * _dt),
        mode=np.array([0.8, 0.7, 0.4, 0.6]),
        amplitude=complex(0.7, 0.1),
    )
    _synthetic_modes.append(_fundamental_mode)

    # Mode 2: Second harmonic (2× frequency)
    _omega2 = 4 * np.pi * _fundamental_freq
    _second_harmonic = KoopmanMode(
        eigenvalue=np.exp(-0.02 + 1j * _omega2 * _dt),
        mode=np.array([0.5, 0.9, 0.3, 0.2]),
        amplitude=complex(0.4, -0.1),
    )
    _synthetic_modes.append(_second_harmonic)

    # Mode 3: Third harmonic (3× frequency)
    _omega3 = 6 * np.pi * _fundamental_freq
    _third_harmonic = KoopmanMode(
        eigenvalue=np.exp(-0.05 + 1j * _omega3 * _dt),
        mode=np.array([0.3, 0.4, 0.8, 0.2]),
        amplitude=complex(0.2, 0.05),
    )
    _synthetic_modes.append(_third_harmonic)

    # Mode 4: Fourth harmonic (fast decay)
    _omega4 = 8 * np.pi * _fundamental_freq
    _fourth_harmonic = KoopmanMode(
        eigenvalue=np.exp(-0.15 + 1j * _omega4 * _dt),
        mode=np.array([0.2, 0.3, 0.4, 0.1]),
        amplitude=complex(0.1, 0.02),
    )
    _synthetic_modes.append(_fourth_harmonic)

    # Create spectrum
    synthetic_spectrum = KoopmanSpectrum(
        modes=_synthetic_modes,
        eigenvalues=np.array([m.eigenvalue for m in _synthetic_modes]),
        eigenvectors=np.column_stack([m.mode for m in _synthetic_modes]),
        amplitudes=np.array([m.amplitude for m in _synthetic_modes]),
        dt=_dt,
        observable_names=["mass", "mRNA", "protein", "flux"],
    )

    print("Synthetic Koopman Spectrum:")
    print("=" * 60)
    for _i, _mode in enumerate(synthetic_spectrum.modes):
        print(f"Mode {_i}: f={_mode.frequency:.6f} Hz, |b|={abs(_mode.amplitude):.3f}, γ={_mode.growth_rate:.4f}")
    return (synthetic_spectrum,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 5: Encoding the Spectrum as a Musical Score

    Now we'll convert the Koopman spectrum to a musical score using
    the **lossy** encoding (standard notation).
    """)
    return


@app.cell
def _(encode_spectrum, synthetic_spectrum):
    # Encode spectrum as musical score
    cellular_score = encode_spectrum(
        synthetic_spectrum,
        cell_cycle_time=3600.0,
        title="E. coli K-12 MG1655 Cell Cycle",
        strain="E. coli K-12 MG1655",
        condition="LB media, 37°C, aerobic",
    )

    # Display ASCII representation
    print(cellular_score.to_ascii())
    return (cellular_score,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Piano Roll Visualization

    A **piano roll** view shows each Koopman mode as a note on a virtual keyboard.
    The vertical axis represents pitch (frequency), and note width represents duration (stability).
    """)
    return


@app.cell
def _(alt, cellular_score, pl):
    # Piano Roll Visualization
    _notes_data = []
    _note_symbols = {
        "breve": "𝅜",
        "whole": "𝅝",
        "half": "𝅗𝅥",
        "quarter": "♩",
        "eighth": "♪",
        "16th": "𝅘𝅥𝅯",
        "32nd": "𝅘𝅥𝅰",
        "64th": "𝅘𝅥𝅱",
    }
    _duration_widths = {
        "breve": 4.0,
        "whole": 2.0,
        "half": 1.0,
        "quarter": 0.5,
        "eighth": 0.25,
        "16th": 0.125,
        "32nd": 0.0625,
        "64th": 0.03125,
    }

    _x_pos = 0.0
    for _i, _staff in enumerate(cellular_score.staves):
        _x_pos = 0.0
        for _j, _note in enumerate(_staff.notes):
            _width = _duration_widths.get(_note.note_value.value, 0.5)
            _notes_data.append({
                "Staff": _staff.observable_class.value,
                "x_start": _x_pos,
                "x_end": _x_pos + _width,
                "x_mid": _x_pos + _width / 2,
                "Pitch": _note.pitch,
                "MIDI": _note.midi_number,
                "Dynamic": _note.dynamic.value,
                "Duration": _note.note_value.value,
                "Symbol": _note_symbols.get(_note.note_value.value, "♩"),
                "Amplitude": _note.amplitude,
                "Harmonic": f"H{_note.harmonic_number}" if _note.harmonic_number > 0 else "—",
            })
            _x_pos += _width + 0.1

    _df = pl.DataFrame(_notes_data)

    # Piano roll bars
    _bars = (
        alt.Chart(_df.to_pandas())
        .mark_bar(cornerRadius=3, height=15)
        .encode(
            x=alt.X("x_start:Q", title="Time (beats)"),
            x2="x_end:Q",
            y=alt.Y("MIDI:Q", title="Pitch (MIDI)", scale=alt.Scale(zero=False, domain=[60, 100])),
            color=alt.Color(
                "Dynamic:N",
                scale=alt.Scale(
                    domain=["ppppp", "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff", "fffff"],
                    scheme="blues",
                ),
            ),
            opacity=alt.value(0.8),
            tooltip=["Pitch", "Dynamic", "Duration", "Harmonic", "Amplitude"],
        )
    )

    # Add note symbols as text
    _symbols = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x="x_mid:Q",
            y=alt.Y("MIDI:Q"),
            text="Symbol:N",
            color=alt.value("white"),
        )
    )

    # Piano keyboard reference lines (white keys)
    _white_keys = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96]
    _key_lines = (
        alt.Chart(
            pl.DataFrame({
                "midi": _white_keys,
                "note": [
                    "C4",
                    "D4",
                    "E4",
                    "F4",
                    "G4",
                    "A4",
                    "B4",
                    "C5",
                    "D5",
                    "E5",
                    "F5",
                    "G5",
                    "A5",
                    "B5",
                    "C6",
                    "D6",
                    "E6",
                    "F6",
                    "G6",
                    "A6",
                    "B6",
                    "C7",
                ],
            }).to_pandas()
        )
        .mark_rule(strokeDash=[2, 2], color="#ddd")
        .encode(y="midi:Q")
    )

    _piano_roll = (_key_lines + _bars + _symbols).properties(
        title="Piano Roll: Koopman Modes as Musical Notes",
        width=600,
        height=350,
    )

    _piano_roll
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Staff Notation View

    A visual representation approximating standard musical notation,
    showing notes on a staff with pitch positions and note values.
    """)
    return


@app.cell
def _(alt, cellular_score, pl):
    # Staff Notation Visualization
    _staff_data = []
    _articulation_symbols = {"tenuto": "—", "staccato": "•", "accent": ">", "fermata": "𝄐", "marcato": "^"}
    _note_heads = {
        "breve": "𝅜",
        "whole": "𝅝",
        "half": "𝅗𝅥",
        "quarter": "●",
        "eighth": "●",
        "16th": "●",
        "32nd": "●",
        "64th": "●",
    }

    _x = 1.0
    for _note in cellular_score.staves[0].notes if cellular_score.staves else []:
        # Convert MIDI to staff position (0 = middle C = C4 = MIDI 60)
        _staff_pos = (_note.midi_number - 60) / 2  # 2 semitones per staff line
        _staff_data.append({
            "x": _x,
            "staff_position": _staff_pos,
            "midi": _note.midi_number,
            "pitch": _note.pitch,
            "note_head": _note_heads.get(_note.note_value.value, "●"),
            "articulation": _articulation_symbols.get(_note.articulation.value, ""),
            "dynamic": _note.dynamic.value,
            "duration": _note.note_value.value,
        })
        _x += 1.5

    _df = (
        pl.DataFrame(_staff_data)
        if _staff_data
        else pl.DataFrame({
            "x": [],
            "staff_position": [],
            "midi": [],
            "pitch": [],
            "note_head": [],
            "articulation": [],
            "dynamic": [],
            "duration": [],
        })
    )

    # Staff lines (5 lines for treble clef, centered around B4=71)
    _line_positions = [-2, -1, 0, 1, 2]  # E4, G4, B4, D5, F5
    _staff_lines = (
        alt.Chart(pl.DataFrame({"y": _line_positions}).to_pandas())
        .mark_rule(color="black", strokeWidth=1)
        .encode(y="y:Q")
    )

    # Note heads
    _notes = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=28, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("staff_position:Q", scale=alt.Scale(domain=[-5, 10]), axis=None),
            text="note_head:N",
            color=alt.value("black"),
            tooltip=["pitch", "dynamic", "duration"],
        )
    )

    # Articulation marks (above notes)
    _artics = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=16, dy=-20)
        .encode(
            x="x:Q",
            y="staff_position:Q",
            text="articulation:N",
            color=alt.value("#666"),
        )
    )

    # Dynamics (below notes)
    _dynamics = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=12, dy=25, fontStyle="italic")
        .encode(
            x="x:Q",
            y="staff_position:Q",
            text="dynamic:N",
            color=alt.value("#2563eb"),
        )
    )

    # Treble clef
    _clef = (
        alt.Chart(pl.DataFrame({"x": [0.3], "y": [0], "text": ["𝄞"]}).to_pandas())
        .mark_text(fontSize=60, fontWeight="bold")
        .encode(
            x="x:Q",
            y="y:Q",
            text="text:N",
            color=alt.value("black"),
        )
    )

    _staff_viz = (
        alt.layer(_staff_lines, _notes, _artics, _dynamics, _clef)
        .properties(
            title="Staff Notation: Cellular Dynamics Score",
            width=600,
            height=250,
        )
        .configure_view(strokeWidth=0)
    )

    _staff_viz
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Power Spectrum Visualization (Audio Equalizer Style)

    This visualization shows the Koopman spectrum as an audio equalizer,
    with bars representing mode amplitudes at different frequencies/harmonics.
    """)
    return


@app.cell
def _(alt, cellular_score, pl):
    # Equalizer-style power spectrum
    _eq_data = []
    for _i, _staff in enumerate(cellular_score.staves):
        for _j, _note in enumerate(_staff.notes):
            _eq_data.append({
                "Harmonic": f"H{_note.harmonic_number}" if _note.harmonic_number > 0 else "DC",
                "harmonic_num": _note.harmonic_number,
                "Pitch": _note.pitch,
                "Amplitude": _note.amplitude,
                "Dynamic": _note.dynamic.value,
                "Frequency": _note.frequency,
            })

    _df = pl.DataFrame(_eq_data)

    # Sort by harmonic number
    _df = _df.sort("harmonic_num")

    # Main bars
    _bars = (
        alt.Chart(_df.to_pandas())
        .mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
        )
        .encode(
            x=alt.X("Harmonic:N", sort=alt.SortField("harmonic_num"), title="Harmonic"),
            y=alt.Y("Amplitude:Q", title="Amplitude"),
            color=alt.Color("Amplitude:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["Harmonic", "Pitch", "Amplitude", "Dynamic"],
        )
    )

    # Add pitch labels on top
    _labels = (
        alt.Chart(_df.to_pandas())
        .mark_text(dy=-10, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("Harmonic:N", sort=alt.SortField("harmonic_num")),
            y="Amplitude:Q",
            text="Pitch:N",
            color=alt.value("white"),
        )
    )

    # Add glow effect with additional lighter bars
    _glow = (
        alt.Chart(_df.to_pandas())
        .mark_bar(
            opacity=0.3,
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X("Harmonic:N", sort=alt.SortField("harmonic_num")),
            y=alt.Y("Amplitude:Q"),
            color=alt.value("#00ff88"),
        )
    )

    _equalizer = (
        (_glow + _bars + _labels)
        .properties(
            title="Power Spectrum Equalizer",
            width=500,
            height=300,
        )
        .configure_view(fill="#1a1a2e")
        .configure_axis(labelColor="white", titleColor="white", gridColor="#333")
        .configure_title(color="white")
    )

    _equalizer
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Grand Staff View (Multi-Observable Orchestration)

    This view shows how different observables (transcriptome, proteome, etc.)
    participate in each Koopman mode, like an orchestral score with multiple instruments.
    """)
    return


@app.cell
def _(alt, cellular_score, pl):
    # Grand staff / orchestration view
    _orch_data = []

    for _i, _staff in enumerate(cellular_score.staves):
        _staff_name = _staff.observable_class.value.replace("_", " ").title()
        for _j, _note in enumerate(_staff.notes):
            _orch_data.append({
                "Observable": _staff_name,
                "Mode": f"Mode {_j}",
                "mode_idx": _j,
                "Pitch": _note.pitch,
                "MIDI": _note.midi_number,
                "Amplitude": _note.amplitude,
                "Dynamic": _note.dynamic.value,
                "Harmonic": _note.harmonic_number,
            })

    _df = pl.DataFrame(_orch_data)

    # Create heatmap-style orchestration view
    _heatmap = (
        alt.Chart(_df.to_pandas())
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Mode:N", sort=alt.SortField("mode_idx"), title="Koopman Mode"),
            y=alt.Y("Observable:N", title="Observable Class"),
            color=alt.Color("Amplitude:Q", scale=alt.Scale(scheme="blues"), title="Amplitude"),
            tooltip=["Observable", "Mode", "Pitch", "Dynamic", "Amplitude"],
        )
    )

    # Add pitch labels
    _pitch_labels = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("Mode:N", sort=alt.SortField("mode_idx")),
            y="Observable:N",
            text="Pitch:N",
            color=alt.condition(alt.datum.Amplitude > 0.4, alt.value("white"), alt.value("black")),
        )
    )

    _orchestration = (_heatmap + _pitch_labels).properties(
        title="Orchestration View: Observable Participation in Koopman Modes",
        width=500,
        height=200,
    )

    _orchestration
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Circular Harmonic Visualization

    A radial view showing harmonics arranged around a circle,
    where angle represents frequency ratio and radius represents amplitude.
    """)
    return


@app.cell
def _(alt, cellular_score, np, pl):
    # Circular/radial harmonic visualization
    _radial_data = []

    for _staff in cellular_score.staves:
        for _j, _note in enumerate(_staff.notes):
            _harmonic = _note.harmonic_number if _note.harmonic_number > 0 else 0
            # Angle based on harmonic (fundamental at top, harmonics spread around)
            _angle = (_harmonic * 360 / 8) % 360  # Spread 8 harmonics around circle
            _radius = _note.amplitude * 100  # Scale for visibility

            # Convert to cartesian for plotting
            _theta = np.radians(_angle - 90)  # -90 to start at top
            _x = _radius * np.cos(_theta)
            _y = _radius * np.sin(_theta)

            _radial_data.append({
                "x": _x,
                "y": _y,
                "angle": _angle,
                "radius": _radius,
                "Harmonic": f"H{_harmonic}" if _harmonic > 0 else "DC",
                "Pitch": _note.pitch,
                "Amplitude": _note.amplitude,
                "Dynamic": _note.dynamic.value,
            })

    _df = pl.DataFrame(_radial_data)

    # Background circles for reference
    _circles_data = []
    for _r in [25, 50, 75, 100]:
        for _theta in np.linspace(0, 2 * np.pi, 50):
            _circles_data.append({"x": _r * np.cos(_theta), "y": _r * np.sin(_theta), "r": _r})
    _circles_df = pl.DataFrame(_circles_data)

    _bg_circles = (
        alt.Chart(_circles_df.to_pandas())
        .mark_line(color="#ddd", strokeWidth=1, opacity=0.5)
        .encode(x="x:Q", y="y:Q", detail="r:N")
    )

    # Radial lines from center to each mode
    _lines = (
        alt.Chart(_df.to_pandas())
        .mark_rule(strokeWidth=3, opacity=0.7)
        .encode(
            x=alt.value(300),  # Center x
            y=alt.value(200),  # Center y
            x2=alt.X2("x:Q"),
            y2=alt.Y2("y:Q"),
            color=alt.Color("Amplitude:Q", scale=alt.Scale(scheme="plasma")),
        )
    )

    # Points at each mode
    _points = (
        alt.Chart(_df.to_pandas())
        .mark_circle(size=300, opacity=0.9)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-120, 120]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-120, 120]), axis=None),
            color=alt.Color("Amplitude:Q", scale=alt.Scale(scheme="plasma")),
            tooltip=["Harmonic", "Pitch", "Amplitude", "Dynamic"],
        )
    )

    # Labels
    _labels = (
        alt.Chart(_df.to_pandas())
        .mark_text(fontSize=12, fontWeight="bold", dy=-20)
        .encode(
            x="x:Q",
            y="y:Q",
            text="Pitch:N",
            color=alt.value("black"),
        )
    )

    # Center label
    _center = (
        alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["f₀"]}).to_pandas())
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="text:N", color=alt.value("#666"))
    )

    _radial = (_bg_circles + _points + _labels + _center).properties(
        title="Harmonic Wheel: Koopman Modes in Circular Arrangement",
        width=400,
        height=400,
    )

    _radial
    return


@app.cell
def _(alt, cellular_score, pl):
    # Original simple visualization (kept for reference)
    _notes_data = []
    for _i, _staff in enumerate(cellular_score.staves):
        for _j, _note in enumerate(_staff.notes):
            _notes_data.append({
                "Staff": _staff.observable_class.value,
                "Note Index": _j,
                "Pitch": _note.pitch,
                "MIDI": _note.midi_number,
                "Dynamic": _note.dynamic.value,
                "Duration": _note.note_value.value,
                "Articulation": _note.articulation.value,
                "Harmonic": _note.harmonic_number,
                "Frequency": _note.frequency,
                "Amplitude": _note.amplitude,
            })

    _df = pl.DataFrame(_notes_data)

    # Create visualization
    _chart = (
        alt.Chart(_df.to_pandas())
        .mark_point(size=300, filled=True)
        .encode(
            x=alt.X("Note Index:O", title="Mode Index"),
            y=alt.Y("MIDI:Q", title="Pitch (MIDI)", scale=alt.Scale(zero=False)),
            color=alt.Color("Dynamic:N", scale=alt.Scale(scheme="blues")),
            shape=alt.Shape("Articulation:N"),
            size=alt.Size("Amplitude:Q", scale=alt.Scale(range=[100, 500])),
            tooltip=["Pitch", "Dynamic", "Duration", "Articulation", "Frequency", "Amplitude"],
        )
        .properties(
            title="Cellular Score Visualization",
            width=500,
            height=300,
        )
    )

    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 6: Lossless Encoding

    The lossy encoding discretizes values. For exact reconstruction,
    we use **lossless encoding** which preserves:
    - Exact complex eigenvalues
    - Exact complex amplitudes with phase
    - Full mode shape vectors
    """)
    return


@app.cell
def _(encode_spectrum_lossless, synthetic_spectrum):
    # Create lossless encoding
    lossless_score = encode_spectrum_lossless(
        synthetic_spectrum,
        duration=1000,  # Number of time steps
        cell_cycle_time=3600.0,
        title="Lossless E. coli Score",
    )

    # Display lossless notation
    print(lossless_score.to_notation())
    return (lossless_score,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 7: Reconstruction from Lossless Score

    The key property of lossless encoding is that we can **exactly reconstruct**
    the original time series using:

    $$X(t) = \sum_i \text{Re}[\phi_i \cdot b_i \cdot \lambda_i^t]$$
    """)
    return


@app.cell
def _(alt, lossless_score, np, pl):
    # Reconstruct time series from lossless score
    reconstructed = lossless_score.reconstruct()

    # Create original from spectrum reconstruction
    _t = np.arange(lossless_score.duration)

    # Visualize reconstruction
    _df = pl.DataFrame({
        "Time": _t[:500],
        "mass": reconstructed[:500, 0],
        "mRNA": reconstructed[:500, 1],
        "protein": reconstructed[:500, 2],
        "flux": reconstructed[:500, 3],
    }).unpivot(index="Time", variable_name="Observable", value_name="Value")

    _chart = (
        alt.Chart(_df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time:Q"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Observable:N"),
        )
        .properties(
            title="Reconstructed Time Series from Lossless Score",
            width=600,
            height=300,
        )
    )

    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 8: Decoding - Score to Spectrum

    We can also decode a musical score back to a Koopman spectrum.
    The **lossy** decoding is approximate (due to discretization),
    while **lossless** decoding is exact.
    """)
    return


@app.cell
def _(cellular_score, decode_score, decode_score_lossless, lossless_score, np):
    # Decode lossy score
    decoded_spectrum_lossy = decode_score(cellular_score)

    print("Lossy Decoding Results:")
    print("=" * 60)
    for _i, _mode in enumerate(decoded_spectrum_lossy.modes):
        print(f"Mode {_i}: f={_mode.frequency:.6f} Hz, |b|={abs(_mode.amplitude):.3f}")

    # Decode lossless score
    decoded_spectrum_lossless = decode_score_lossless(lossless_score)

    print("\nLossless Decoding Results:")
    print("=" * 60)
    for _i, _mode in enumerate(decoded_spectrum_lossless.modes):
        print(f"Mode {_i}: f={_mode.frequency:.6f} Hz, |b|={abs(_mode.amplitude):.3f}, λ={_mode.eigenvalue:.6f}")

    # Verify the eigenvalues match exactly
    print("\nEigenvalue Match Verification:")
    print("=" * 60)
    for _i, (_orig, _decoded) in enumerate(zip(lossless_score.modes, decoded_spectrum_lossless.modes)):
        _match = np.isclose(_orig.eigenvalue, _decoded.eigenvalue)
        print(f"Mode {_i}: {'EXACT MATCH' if _match else 'MISMATCH'}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 9: MusicXML Export

    The music package integrates with **music21** for professional
    music notation export. This enables:
    - Export to MusicXML for notation software
    - MIDI export for audio playback
    - PDF rendering via external tools
    """)
    return


@app.cell
def _(cellular_score):
    # Try to import music21 and export
    try:
        from apollo.m21 import score_to_music21, score_to_musicxml

        # Convert to music21 score
        m21_score = score_to_music21(cellular_score)
        print("music21 Score created successfully!")
        print(f"Parts: {len(m21_score.parts)}")

        # Show text representation
        print("\nText representation:")
        for _part in m21_score.parts:
            print(f"  Part: {_part.partName}")
            for _note in _part.recurse().notes[:5]:
                print(f"    {_note.nameWithOctave} - {_note.duration.type}")

    except ImportError:
        print("music21 not installed. To enable MusicXML export:")
        print("  uv sync --group music")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 10: Mapping Summary

    | Koopman Domain | Musical Domain | Mapping |
    |----------------|----------------|---------|
    | Frequency f | Pitch | MIDI = 12·log₂(f/f₀) + 69 |
    | Amplitude |b| | Dynamic (ppp-fff) | 12 discrete levels |
    | \|Growth rate\| | Note duration | Longer = more stable |
    | Sign of γ | Articulation | •=decay, >=growth |
    | Mode shape φ | Orchestration | Per-staff dynamics |
    | Cell cycle T | Tempo | BPM = 240/T_min |
    | Harmonic structure | Key signature | # of sharps/flats |
    | Aggregation strata | Movement structure | I, II, III... |

    This mapping is **complete** (all Koopman components represented),
    **reversible** (scores can be decoded), and **RFC006-compliant**
    (supports all four aggregation strategies).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 11: UQ Pipeline → Musical Score (Layer 2)

    The existing tutorial covers Layer 1: encoding individual Koopman modes as notes.
    The `apollo.uq_score` module provides Layer 2: encoding the *entire UQ pipeline output*
    (Sobol indices, variance decomposition, per-stage sensitivity) as a musical score.

    This is not metaphor — PCE coefficients are a spectral decomposition (orthogonal polynomial basis),
    and Sobol indices partition variance exactly as a power spectrum partitions energy.
    """)
    return


@app.cell
def _(np):
    from apollo.uq_score import encode_sensitivity

    # Synthetic Sobol indices for 5 parameters
    param_names = ["vio_expression", "mecA_kcat", "ppGpp_hill_n", "ftsZ_threshold", "murG_expression"]
    sobol_first = np.array([0.35, 0.12, 0.06, 0.03, 0.01])
    sobol_total = np.array([0.42, 0.15, 0.08, 0.035, 0.015])

    # Synthetic variance decomposition
    variance_decomp = {
        "generation_fraction": np.array([0.60]),
        "seed_fraction": np.array([0.30]),
    }

    score = encode_sensitivity(
        sobol_first_order=sobol_first,
        sobol_total_order=sobol_total,
        parameter_names=param_names,
        variance_decomposition=variance_decomp,
        cell_cycle_time=3600.0,
        n_cell_cycle_bins=10,
        output_name="listeners__mass__dry_mass",
    )

    print(score.to_ascii())
    return (score,)


@app.cell
def _(score):
    print(score.read_aloud())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Layer 1 vs Layer 2

    | Layer | Input | Output | Granularity |
    |-------|-------|--------|-------------|
    | Layer 1 | Koopman eigenvalues | Individual notes | One mode = one note |
    | Layer 2 | Sobol indices + variance decomp | Full score | Entire pipeline = one score |

    Both share `apollo.types` and `apollo.mappings` primitives (Dynamic, Tempo, etc.).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion

    The **music** package provides a novel way to represent and visualize
    Koopman spectral analysis of cellular dynamics using Western musical notation.

    Key features:
    - **Lossy encoding**: Human-readable standard notation
    - **Lossless encoding**: Exact numerical annotations for reconstruction
    - **MusicXML export**: Integration with professional music software
    - **Bidirectional**: Encode spectra as scores, decode scores as spectra

    This creates a new "language" for discussing cellular dynamics that
    leverages centuries of musical theory and human auditory cognition.
    """)
    return


@app.cell
def _():
    from music21 import corpus

    return (corpus,)


@app.cell
def _(corpus):
    bach = corpus.parse("bach/bwv269")
    bach.id = "bwv269"
    bach.measures(0, 4).show()
    return (bach,)


@app.cell
def _(bach):
    tsTree = bach.asTimespans(flatten=True)
    tsTree
    return (tsTree,)


@app.cell
def _():
    63 / 21
    return


@app.cell
def _(tsTree):
    tsTree[0]
    return


@app.cell
def _(tsTree):
    for ts in tsTree[20:32]:
        print(ts)
    return


@app.cell
def _(tsTree):
    v = tsTree.getVerticalityAt(8.0)
    v
    return (v,)


@app.cell
def _(v):
    # The Verticality object knows which elements are just starting:
    v.startTimespans
    return


@app.cell
def _(v):
    # ...and which are continuing:
    v.overlapTimespans
    return


@app.cell
def _(v):
    # ...It also knows which elements are have just stopped before the Verticality:
    v.stopTimespans
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
