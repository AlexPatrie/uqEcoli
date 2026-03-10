"""Tutorial: Musical Notation for Koopman Spectral Analysis

This tutorial implements the concepts from readmes/MUSICAL.md, demonstrating
how standard Western musical notation can efficiently describe Koopman spectra
from cellular dynamics.

Run with: marimo run tutorials/music.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import numpy as np
    from dataclasses import dataclass, field
    from typing import Optional
    from enum import Enum

    return Enum, alt, dataclass, field, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # Musical Notation for Koopman Spectral Analysis

    This tutorial demonstrates how **standard Western musical notation** can
    efficiently describe Koopman spectra from cellular dynamics.

    ## Why Musical Notation?

    Musical notation is a highly evolved system for representing:
    - **Spectral content** (pitch = frequency)
    - **Temporal dynamics** (duration = stability)
    - **Amplitude** (dynamics markings)
    - **Relationships** (harmony, phase)

    These are exactly what Koopman analysis produces!

    ## What You'll Learn

    1. The mapping between musical notation and Koopman concepts
    2. How to generate "cellular scores" from simulation data
    3. Visual representations of cellular dynamics as music
    4. Comparing spectra using musical intuition
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: The Notation System

    ### Mapping Musical Elements to Koopman Analysis
    """)
    return


@app.cell
def _(Enum):
    # Define enums for musical notation elements

    class Clef(Enum):
        """Clefs represent observable classes."""
        TREBLE = ("transcriptome", "mRNA counts")
        BASS = ("proteome", "protein levels")
        ALTO = ("metabolome", "metabolic fluxes")
        TENOR = ("properties", "higher-order properties")

    class NoteValue(Enum):
        """Note values represent mode stability (inverse decay rate)."""
        WHOLE = (1.0, 0.001, "very stable")
        HALF = (0.5, 0.01, "stable")
        QUARTER = (0.25, 0.1, "moderate decay")
        EIGHTH = (0.125, 1.0, "fast decay")
        SIXTEENTH = (0.0625, float('inf'), "transient")

    class Dynamic(Enum):
        """Dynamics represent mode amplitude."""
        PPP = ("ppp", 0.01, "pianississimo")
        PP = ("pp", 0.05, "pianissimo")
        P = ("p", 0.1, "piano")
        MP = ("mp", 0.2, "mezzo-piano")
        MF = ("mf", 0.4, "mezzo-forte")
        F = ("f", 0.6, "forte")
        FF = ("ff", 0.8, "fortissimo")
        FFF = ("fff", 1.0, "fortississimo")

    class Articulation(Enum):
        """Articulations represent growth/decay character."""
        STACCATO = ("staccato", "sharply decaying")
        TENUTO = ("tenuto", "sustained")
        ACCENT = ("accent", "growing")
        FERMATA = ("fermata", "persistent indefinitely")

    class Tempo(Enum):
        """Tempo markings represent cell cycle duration."""
        GRAVE = ("Grave", 120, "stationary phase")
        LARGO = ("Largo", 90, "minimal media")
        ADAGIO = ("Adagio", 70, "moderate growth")
        ANDANTE = ("Andante", 55, "standard conditions")
        MODERATO = ("Moderato", 45, "good growth")
        ALLEGRO = ("Allegro", 35, "fast growth")
        PRESTO = ("Presto", 25, "maximal growth")

    return Articulation, Clef, Dynamic, NoteValue, Tempo


@app.cell
def _(mo, pd):
    # Create visualization of the mapping
    mapping_data = [
        {"Musical Element": "Pitch (vertical)", "Koopman Equivalent": "Mode frequency", "Example": "Higher = faster oscillation"},
        {"Musical Element": "Note duration", "Koopman Equivalent": "Mode stability", "Example": "Whole note = persistent mode"},
        {"Musical Element": "Dynamics (pp-ff)", "Koopman Equivalent": "Mode amplitude", "Example": "fff = dominant mode"},
        {"Musical Element": "Time signature", "Koopman Equivalent": "Cell cycle period", "Example": "4/4 = standard cycle"},
        {"Musical Element": "Key signature", "Koopman Equivalent": "Harmonic structure", "Example": "C major = integer harmonics"},
        {"Musical Element": "Articulation", "Koopman Equivalent": "Growth/decay rate", "Example": "Tenuto = sustained"},
        {"Musical Element": "Tempo marking", "Koopman Equivalent": "Cycle duration", "Example": "Allegro = fast growth"},
        {"Musical Element": "Clef", "Koopman Equivalent": "Observable class", "Example": "Treble = transcriptome"},
    ]

    mapping_df = pd.DataFrame(mapping_data)

    mo.md("""
    ### The Fundamental Mapping

    | Musical Element | Koopman Equivalent | Example |
    |-----------------|-------------------|---------|
    | Pitch (vertical) | Mode frequency | Higher = faster oscillation |
    | Note duration | Mode stability | Whole note = persistent mode |
    | Dynamics (pp-ff) | Mode amplitude | fff = dominant mode |
    | Time signature | Cell cycle period | 4/4 = standard cycle |
    | Key signature | Harmonic structure | C major = integer harmonics |
    | Articulation | Growth/decay rate | Tenuto = sustained |
    | Tempo marking | Cycle duration | Allegro = fast growth |
    | Clef | Observable class | Treble = transcriptome |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: The CellularScore Class

    We implement a class that converts Koopman spectra to musical notation.
    """)
    return


@app.cell
def _(Articulation, Clef, Dynamic, NoteValue, Tempo, dataclass, field, np):
    @dataclass
    class CellularNote:
        """A single note in the cellular score, representing one Koopman mode."""
        harmonic: float          # Harmonic number (1.0 = fundamental)
        amplitude: float         # Mode amplitude (0-1)
        growth_rate: float       # Growth/decay rate
        frequency: float         # Actual frequency in Hz
        observable_class: str    # Which observable class

        def get_note_value(self) -> NoteValue:
            """Convert growth rate to note duration."""
            abs_rate = abs(self.growth_rate)
            if abs_rate < 0.001:
                return NoteValue.WHOLE
            elif abs_rate < 0.01:
                return NoteValue.HALF
            elif abs_rate < 0.1:
                return NoteValue.QUARTER
            elif abs_rate < 1.0:
                return NoteValue.EIGHTH
            else:
                return NoteValue.SIXTEENTH

        def get_dynamic(self) -> Dynamic:
            """Convert amplitude to dynamic marking."""
            if self.amplitude < 0.01:
                return Dynamic.PPP
            elif self.amplitude < 0.05:
                return Dynamic.PP
            elif self.amplitude < 0.1:
                return Dynamic.P
            elif self.amplitude < 0.2:
                return Dynamic.MP
            elif self.amplitude < 0.4:
                return Dynamic.MF
            elif self.amplitude < 0.6:
                return Dynamic.F
            elif self.amplitude < 0.8:
                return Dynamic.FF
            else:
                return Dynamic.FFF

        def get_articulation(self) -> Articulation:
            """Convert growth rate to articulation."""
            if self.growth_rate < -0.01:
                return Articulation.STACCATO
            elif self.growth_rate > 0.01:
                return Articulation.ACCENT
            elif abs(self.growth_rate) < 0.0001:
                return Articulation.FERMATA
            else:
                return Articulation.TENUTO

        def to_ascii(self) -> str:
            """Generate ASCII representation of this note."""
            note_symbols = {
                NoteValue.WHOLE: "",
                NoteValue.HALF: "",
                NoteValue.QUARTER: "",
                NoteValue.EIGHTH: "",
                NoteValue.SIXTEENTH: "",
            }
            articulation_symbols = {
                Articulation.STACCATO: "",
                Articulation.TENUTO: "",
                Articulation.ACCENT: ">",
                Articulation.FERMATA: "",
            }

            note = note_symbols[self.get_note_value()]
            dyn = self.get_dynamic().value[0]
            art = articulation_symbols[self.get_articulation()]
            harm = f"({self.harmonic:.0f}x)" if self.harmonic >= 1 else "(0x)"

            return f"{note} {dyn} {art}  {harm}"


    @dataclass
    class CellularStaff:
        """A staff in the cellular score, representing one observable class."""
        clef: Clef
        notes: list = field(default_factory=list)
        time_signature: str = "4/4"

        def add_note(self, note: CellularNote):
            """Add a note to this staff."""
            self.notes.append(note)

        def to_ascii(self, width: int = 60) -> str:
            """Generate ASCII representation of this staff."""
            clef_symbols = {
                Clef.TREBLE: "",
                Clef.BASS: "",
                Clef.ALTO: "",
                Clef.TENOR: "",
            }

            lines = []
            lines.append(f"{self.clef.value[0].capitalize()} {clef_symbols[self.clef]} {self.time_signature}")
            lines.append("" + "" * width + "")

            # Sort notes by harmonic number
            sorted_notes = sorted(self.notes, key=lambda n: n.harmonic)

            # Create note line
            note_line = ""
            for note in sorted_notes[:6]:  # Max 6 notes per staff
                note_line += note.to_ascii() + "   "

            lines.append(f" {note_line}")
            lines.append("" + "" * width + "")

            return "\n".join(lines)


    @dataclass
    class CellularScore:
        """
        A complete cellular score representing a Koopman spectrum.

        Converts Koopman spectral analysis results into musical notation,
        enabling intuitive understanding of cellular dynamics.
        """
        title: str = "Cellular Score"
        tempo: Tempo = Tempo.ANDANTE
        key: str = "C major"
        staves: list = field(default_factory=list)
        cell_cycle_time: float = 60.0  # minutes

        def add_staff(self, staff: CellularStaff):
            """Add a staff to the score."""
            self.staves.append(staff)

        def get_tempo_marking(self) -> str:
            """Get tempo marking based on cell cycle time."""
            for tempo in Tempo:
                if self.cell_cycle_time >= tempo.value[1]:
                    return f"{tempo.value[0]} ( = {tempo.value[1]} min)"
            return f"Prestissimo ( = {self.cell_cycle_time:.0f} min)"

        def to_ascii(self) -> str:
            """Generate complete ASCII score."""
            lines = []
            lines.append("=" * 70)
            lines.append(f"  {self.title.upper()}")
            lines.append("")
            lines.append(f"  {self.get_tempo_marking()}  |  Key: {self.key}")
            lines.append("=" * 70)
            lines.append("")

            for staff in self.staves:
                lines.append(staff.to_ascii())
                lines.append("")

            lines.append("=" * 70)
            lines.append("LEGEND:")
            lines.append("   = stable (whole)    = quarter (moderate decay)")
            lines.append("   = stable (half)     = eighth (fast decay)")
            lines.append("   = tenuto (sustained)    = staccato (decaying)")
            lines.append("  fff/ff/f/mf/mp/p/pp = amplitude (loud  soft)")
            lines.append("  (nx) = harmonic number relative to cell cycle")
            lines.append("=" * 70)

            return "\n".join(lines)

        @classmethod
        def from_spectrum(
            cls,
            modes: list,
            observable_names: list,
            cell_cycle_time: float = 60.0,
            title: str = "Koopman Spectrum Score",
        ) -> "CellularScore":
            """
            Create a CellularScore from Koopman spectrum data.

            Args:
                modes: List of dicts with 'frequency', 'amplitude', 'growth_rate', 'mode_shape'
                observable_names: Names of observables
                cell_cycle_time: Expected cell cycle time in minutes
                title: Score title

            Returns:
                CellularScore instance
            """
            fundamental_freq = 1.0 / (cell_cycle_time * 60)  # Convert to Hz

            # Determine key based on harmonic structure
            harmonics = [m['frequency'] / fundamental_freq for m in modes if m['frequency'] > 0]
            harmonic_deviation = np.mean([abs(h - round(h)) for h in harmonics]) if harmonics else 0

            if harmonic_deviation < 0.05:
                key = "C major"
            elif harmonic_deviation < 0.1:
                key = "G major"
            elif harmonic_deviation < 0.15:
                key = "D major"
            else:
                key = "Chromatic"

            score = cls(
                title=title,
                key=key,
                cell_cycle_time=cell_cycle_time,
            )

            # Group observables by class
            obs_classes = {
                'transcriptome': [o for o in observable_names if 'mRNA' in o or 'rna' in o.lower()],
                'proteome': [o for o in observable_names if 'protein' in o.lower()],
                'metabolome': [o for o in observable_names if 'flux' in o.lower()],
                'properties': [o for o in observable_names if 'mass' in o.lower() or 'growth' in o.lower()],
            }

            clef_map = {
                'transcriptome': Clef.TREBLE,
                'proteome': Clef.BASS,
                'metabolome': Clef.ALTO,
                'properties': Clef.TENOR,
            }

            # Create staves for each observable class
            for obs_class, obs_list in obs_classes.items():
                if not obs_list:
                    continue

                staff = CellularStaff(clef=clef_map[obs_class])

                for mode in modes:
                    freq = mode['frequency']
                    if freq <= 0:
                        harmonic = 0  # Growth mode
                    else:
                        harmonic = freq / fundamental_freq

                    # Get average amplitude for this observable class
                    mode_shape = mode.get('mode_shape', {})
                    class_amps = [mode_shape.get(o, 0) for o in obs_list]
                    avg_amp = np.mean(class_amps) if class_amps else mode['amplitude']

                    note = CellularNote(
                        harmonic=harmonic,
                        amplitude=avg_amp,
                        growth_rate=mode['growth_rate'],
                        frequency=freq,
                        observable_class=obs_class,
                    )
                    staff.add_note(note)

                score.add_staff(staff)

            return score

    return CellularNote, CellularScore, CellularStaff


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: Generating Synthetic Koopman Spectra

    Let's create synthetic Koopman spectra for demonstration.
    """)
    return


@app.cell
def _(np):
    def generate_synthetic_spectrum(
        cell_cycle_time: float = 40.0,  # minutes
        n_harmonics: int = 4,
        noise_level: float = 0.1,
        condition: str = "healthy",
        seed: int = 42,
    ) -> list:
        """
        Generate a synthetic Koopman spectrum.

        Args:
            cell_cycle_time: Cell cycle duration in minutes
            n_harmonics: Number of harmonics to include
            noise_level: Amount of noise in harmonic frequencies
            condition: 'healthy', 'stressed', or 'mutant'
            seed: Random seed

        Returns:
            List of mode dictionaries
        """
        rng = np.random.default_rng(seed)
        fundamental_freq = 1.0 / (cell_cycle_time * 60)  # Hz

        modes = []

        # Condition-specific parameters
        if condition == "healthy":
            amp_decay = 0.5  # Harmonics decay normally
            stability = 0.001  # Very stable
            freq_noise = 0.02  # Nearly integer harmonics
        elif condition == "stressed":
            amp_decay = 0.3  # Harmonics decay faster
            stability = 0.05  # Less stable
            freq_noise = 0.1  # More deviation
        else:  # mutant
            amp_decay = 0.7  # Harmonics persist more
            stability = 0.02  # Moderate stability
            freq_noise = 0.15  # Irregular harmonics

        # Growth mode (non-oscillatory)
        modes.append({
            'frequency': 0.0,
            'amplitude': 0.8 if condition == "healthy" else 0.5,
            'growth_rate': 0.001,  # Slight positive growth
            'mode_shape': {
                'mass': 0.9, 'mRNA_total': 0.3, 'protein_total': 0.4, 'flux_total': 0.2
            },
        })

        # Oscillatory modes (harmonics of cell cycle)
        for n in range(1, n_harmonics + 1):
            freq_deviation = rng.normal(0, freq_noise)
            freq = fundamental_freq * n * (1 + freq_deviation)

            amplitude = (1.0 / (n ** amp_decay)) * (1 + rng.normal(0, noise_level))
            amplitude = max(0.01, min(1.0, amplitude))

            growth_rate = -stability * n + rng.normal(0, stability * 0.5)

            # Mode shape varies by harmonic
            mode_shape = {
                'mass': 0.8 / n + rng.random() * 0.2,
                'mRNA_total': 0.6 + 0.2 * np.sin(n * np.pi / 3) + rng.random() * 0.2,
                'protein_total': 0.5 + 0.1 * np.cos(n * np.pi / 4) + rng.random() * 0.2,
                'flux_total': 0.3 + 0.4 * (n % 2) + rng.random() * 0.2,
            }

            modes.append({
                'frequency': freq,
                'amplitude': amplitude,
                'growth_rate': growth_rate,
                'mode_shape': mode_shape,
            })

        return modes

    # Generate spectra for different conditions
    healthy_spectrum = generate_synthetic_spectrum(condition="healthy", seed=42)
    stressed_spectrum = generate_synthetic_spectrum(condition="stressed", seed=43)
    mutant_spectrum = generate_synthetic_spectrum(condition="mutant", seed=44)
    return healthy_spectrum, mutant_spectrum, stressed_spectrum


@app.cell
def _(CellularScore, healthy_spectrum, mo):
    # Create and display the healthy cell score
    observable_names = ['mass', 'mRNA_total', 'protein_total', 'flux_total']

    healthy_score = CellularScore.from_spectrum(
        modes=healthy_spectrum,
        observable_names=observable_names,
        cell_cycle_time=40.0,
        title="Wild-Type E. coli in LB Media",
    )

    mo.md(f"""
    ### Example: Wild-Type E. coli Score

    ```
    {healthy_score.to_ascii()}
    ```
    """)
    return (observable_names,)


@app.cell
def _(mo):
    mo.md("""
    ## Part 4: Visual Representation

    Let's create Altair visualizations that capture musical concepts.
    """)
    return


@app.cell
def _(healthy_spectrum, mo, mutant_spectrum, pd, stressed_spectrum):
    def spectrum_to_df(spectrum: list, condition: str, cell_cycle_time: float = 40.0) -> pd.DataFrame:
        """Convert spectrum to DataFrame for visualization."""
        fundamental_freq = 1.0 / (cell_cycle_time * 60)
        rows = []
        for _i, mode in enumerate(spectrum):
            freq = mode['frequency']
            harmonic = freq / fundamental_freq if freq > 0 else 0
            rows.append({
                'condition': condition,
                'mode_idx': _i,
                'frequency': freq,
                'harmonic': harmonic,
                'amplitude': mode['amplitude'],
                'growth_rate': mode['growth_rate'],
                'is_oscillatory': freq > 0,
                'note_size': 100 + mode['amplitude'] * 400,
            })
        return pd.DataFrame(rows)

    # Combine all spectra
    all_spectra = pd.concat([
        spectrum_to_df(healthy_spectrum, "Wild-Type"),
        spectrum_to_df(stressed_spectrum, "Stressed"),
        spectrum_to_df(mutant_spectrum, "Mutant"),
    ])

    mo.md("""
    ### Spectrum Visualization as "Musical Score"

    This visualization shows Koopman modes arranged like a musical score:
    - **Y-axis**: Harmonic number (like pitch on a staff)
    - **Size**: Mode amplitude (like dynamics ff/pp)
    - **Color**: Growth rate (blue=stable, red=decaying)
    - **Columns**: Different cellular conditions
    """)
    return (all_spectra,)


@app.cell
def _(all_spectra, alt, pd):
    # Create the "score" visualization
    score_chart = alt.Chart(all_spectra).mark_circle().encode(
        x=alt.X('condition:N', title='Condition', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('harmonic:Q', title='Harmonic Number (Pitch)',
                scale=alt.Scale(domain=[-0.5, 5])),
        size=alt.Size('amplitude:Q', title='Amplitude (Dynamics)',
                      scale=alt.Scale(range=[50, 500]),
                      legend=alt.Legend(title="Amplitude")),
        color=alt.Color('growth_rate:Q', title='Growth Rate',
                        scale=alt.Scale(scheme='redblue', domain=[-0.1, 0.01]),
                        legend=alt.Legend(title="Stability")),
        tooltip=['condition', 'harmonic', 'amplitude', 'growth_rate', 'frequency']
    ).properties(
        width=400,
        height=400,
        title='Koopman Spectrum as Musical Score'
    )

    # Add reference lines for integer harmonics (like staff lines)
    harmonic_lines = alt.Chart(
        pd.DataFrame({'harmonic': [0, 1, 2, 3, 4]})
    ).mark_rule(strokeDash=[5, 5], opacity=0.3).encode(
        y='harmonic:Q'
    )

    # Add harmonic labels
    harmonic_labels = alt.Chart(
        pd.DataFrame({
            'harmonic': [0, 1, 2, 3, 4],
            'label': ['Growth', 'Fund.', '2nd', '3rd', '4th']
        })
    ).mark_text(align='right', dx=-10, fontSize=10).encode(
        y='harmonic:Q',
        text='label:N'
    )

    combined_score = (harmonic_lines + score_chart + harmonic_labels)
    combined_score
    return


@app.cell
def _(mo):
    mo.md("""
    ### Dynamics Comparison

    This shows how mode amplitudes (musical "dynamics") differ across conditions.
    """)
    return


@app.cell
def _(all_spectra, alt, pd):
    # Create dynamics comparison chart
    dynamics_chart = alt.Chart(all_spectra[all_spectra['harmonic'] > 0]).mark_bar().encode(
        x=alt.X('harmonic:O', title='Harmonic'),
        y=alt.Y('amplitude:Q', title='Amplitude'),
        color=alt.Color('condition:N',
                        scale=alt.Scale(domain=['Wild-Type', 'Stressed', 'Mutant'],
                                       range=['#2ecc71', '#e74c3c', '#9b59b6'])),
        xOffset='condition:N',
        tooltip=['condition', 'harmonic', 'amplitude']
    ).properties(
        width=500,
        height=300,
        title='Mode Amplitudes by Condition (Musical Dynamics)'
    )

    # Add dynamic markings
    dynamic_markers = pd.DataFrame({
        'amplitude': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
        'label': ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff']
    })

    dynamic_lines = alt.Chart(dynamic_markers).mark_rule(
        strokeDash=[2, 2], opacity=0.5
    ).encode(y='amplitude:Q')

    dynamic_labels_chart = alt.Chart(dynamic_markers).mark_text(
        align='left', dx=5, fontSize=9, color='gray'
    ).encode(
        y='amplitude:Q',
        text='label:N'
    )

    (dynamics_chart + dynamic_lines + dynamic_labels_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 5: Mode Shape as Orchestration

    In music, "orchestration" describes which instruments play each part.
    In cellular dynamics, the "mode shape" shows which observables participate.
    """)
    return


@app.cell
def _(alt, healthy_spectrum, mo, pd):
    # Extract mode shapes for visualization
    mode_shape_data = []
    for _i, mode in enumerate(healthy_spectrum):
        freq = mode['frequency']
        fundamental_freq = 1.0 / (40.0 * 60)
        harmonic = freq / fundamental_freq if freq > 0 else 0

        for obs, amp in mode['mode_shape'].items():
            mode_shape_data.append({
                'harmonic': f"{harmonic:.0f}x" if harmonic > 0 else "Growth",
                'observable': obs.replace('_', ' ').title(),
                'participation': amp,
                'harmonic_num': harmonic,
            })

    mode_shape_df = pd.DataFrame(mode_shape_data)

    # Create heatmap (like an orchestration chart)
    orchestration_chart = alt.Chart(mode_shape_df).mark_rect().encode(
        x=alt.X('harmonic:N', title='Harmonic (Mode)', sort=['Growth', '1x', '2x', '3x', '4x']),
        y=alt.Y('observable:N', title='Observable (Instrument)'),
        color=alt.Color('participation:Q', title='Participation',
                        scale=alt.Scale(scheme='viridis')),
        tooltip=['harmonic', 'observable', alt.Tooltip('participation:Q', format='.2f')]
    ).properties(
        width=400,
        height=250,
        title='Mode Shape as Orchestration Chart'
    )

    # Add text labels
    orchestration_text = alt.Chart(mode_shape_df).mark_text(
        color='white', fontSize=11
    ).encode(
        x=alt.X('harmonic:N', sort=['Growth', '1x', '2x', '3x', '4x']),
        y='observable:N',
        text=alt.Text('participation:Q', format='.2f')
    )

    mo.md("""
    ### Orchestration Chart

    Like a musical score shows which instruments play each passage,
    this shows which observables participate in each Koopman mode.

    - **Rows**: Observables ("instruments")
    - **Columns**: Modes ("passages")
    - **Color/Value**: Participation amplitude
    """)
    return orchestration_chart, orchestration_text


@app.cell
def _(orchestration_chart, orchestration_text):
    (orchestration_chart + orchestration_text)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 6: Eigenvalue Spectrum as Complex Plane Score

    Musical pitch exists on a 1D axis. Koopman eigenvalues exist in the
    2D complex plane, encoding both frequency (angle) and stability (radius).
    """)
    return


@app.cell
def _(alt, healthy_spectrum, np, pd):
    # Convert spectrum to eigenvalue representation
    def spectrum_to_eigenvalues(spectrum: list, dt: float = 1.0) -> pd.DataFrame:
        """Convert spectrum to eigenvalue complex plane representation."""
        rows = []
        fundamental_freq = 1.0 / (40.0 * 60)

        for _i, mode in enumerate(spectrum):
            freq = mode['frequency']
            growth = mode['growth_rate']

            # Eigenvalue: lambda = e^{(gamma + i*omega)*dt}
            omega = 2 * np.pi * freq
            gamma = growth

            eigenvalue = np.exp((gamma + 1j * omega) * dt)

            harmonic = freq / fundamental_freq if freq > 0 else 0

            rows.append({
                'mode_idx': _i,
                'real': eigenvalue.real,
                'imag': eigenvalue.imag,
                'magnitude': np.abs(eigenvalue),
                'angle': np.angle(eigenvalue),
                'amplitude': mode['amplitude'],
                'harmonic': f"{harmonic:.0f}x" if harmonic > 0 else "Growth",
                'is_stable': np.abs(eigenvalue) <= 1.0,
            })

        return pd.DataFrame(rows)

    eigenvalue_df = spectrum_to_eigenvalues(healthy_spectrum)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle_df = pd.DataFrame({
        'x': np.cos(theta),
        'y': np.sin(theta)
    })

    unit_circle = alt.Chart(unit_circle_df).mark_line(
        strokeDash=[5, 5], color='gray', opacity=0.5
    ).encode(x='x:Q', y='y:Q')

    # Eigenvalue points
    eigenvalue_points = alt.Chart(eigenvalue_df).mark_circle().encode(
        x=alt.X('real:Q', title='Real Part', scale=alt.Scale(domain=[-1.5, 1.5])),
        y=alt.Y('imag:Q', title='Imaginary Part', scale=alt.Scale(domain=[-1.5, 1.5])),
        size=alt.Size('amplitude:Q', scale=alt.Scale(range=[50, 400]),
                      legend=alt.Legend(title='Amplitude')),
        color=alt.Color('harmonic:N', legend=alt.Legend(title='Mode')),
        tooltip=['harmonic', 'real', 'imag', 'magnitude', 'amplitude']
    )

    eigenvalue_labels = alt.Chart(eigenvalue_df).mark_text(
        dx=15, fontSize=10
    ).encode(
        x='real:Q',
        y='imag:Q',
        text='harmonic:N'
    )

    eigenvalue_chart = (unit_circle + eigenvalue_points + eigenvalue_labels).properties(
        width=400,
        height=400,
        title='Eigenvalue Spectrum in Complex Plane'
    )

    eigenvalue_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ### Interpreting the Complex Plane

    - **Unit circle (dashed)**: Boundary of stability
    - **Inside circle**: Decaying modes (fading notes)
    - **On circle**: Persistent modes (sustained notes)
    - **Outside circle**: Growing modes (crescendo)
    - **Angle**: Frequency (pitch)
    - **Size**: Amplitude (dynamics)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 7: Tempo and Time Signature

    The cell cycle period determines the "tempo" of cellular music.
    """)
    return


@app.cell
def _(Tempo, alt, pd):
    # Tempo visualization
    tempo_data = []
    for tempo in Tempo:
        tempo_data.append({
            'marking': tempo.value[0],
            'cycle_time': tempo.value[1],
            'description': tempo.value[2],
            'order': list(Tempo).index(tempo),
        })

    tempo_df = pd.DataFrame(tempo_data)

    # Current cell cycle marker
    current_cycle = 40  # minutes

    tempo_chart = alt.Chart(tempo_df).mark_bar(opacity=0.7).encode(
        x=alt.X('cycle_time:Q', title='Cell Cycle Time (minutes)'),
        y=alt.Y('marking:N', title='Tempo Marking',
                sort=alt.EncodingSortField(field='order', order='descending')),
        color=alt.Color('cycle_time:Q', scale=alt.Scale(scheme='viridis'),
                        legend=None),
        tooltip=['marking', 'cycle_time', 'description']
    ).properties(
        width=400,
        height=300,
        title='Tempo Markings for Cell Cycle Duration'
    )

    # Add current tempo marker
    current_tempo_line = alt.Chart(
        pd.DataFrame({'x': [current_cycle]})
    ).mark_rule(color='red', strokeWidth=2).encode(x='x:Q')

    current_tempo_label = alt.Chart(
        pd.DataFrame({'x': [current_cycle], 'label': ['Current: 40 min']})
    ).mark_text(color='red', dx=5, dy=-10, align='left').encode(
        x='x:Q', text='label:N'
    )

    (tempo_chart + current_tempo_line + current_tempo_label)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 8: Comparative Analysis

    Let's compare the three conditions side-by-side as a "triptych" score.
    """)
    return


@app.cell
def _(
    CellularScore,
    healthy_spectrum,
    mo,
    mutant_spectrum,
    observable_names,
    stressed_spectrum,
):
    # Generate scores for all conditions
    scores = {
        "Wild-Type": CellularScore.from_spectrum(
            healthy_spectrum, observable_names, 40.0, "Wild-Type E. coli"
        ),
        "Stressed": CellularScore.from_spectrum(
            stressed_spectrum, observable_names, 40.0, "Stressed E. coli"
        ),
        "Mutant": CellularScore.from_spectrum(
            mutant_spectrum, observable_names, 40.0, "Mutant E. coli"
        ),
    }

    mo.md(f"""
    ### Comparative Cellular Scores

    **Wild-Type (Healthy)**
    ```
    {scores["Wild-Type"].to_ascii()}
    ```

    **Stressed Cell**
    ```
    {scores["Stressed"].to_ascii()}
    ```

    **Mutant Cell**
    ```
    {scores["Mutant"].to_ascii()}
    ```

    ### Visual Comparison
    """)
    return


@app.cell
def _(all_spectra, alt):
    # Create faceted comparison
    faceted_score = alt.Chart(all_spectra).mark_circle().encode(
        x=alt.X('growth_rate:Q', title='Stability (Growth Rate)',
                scale=alt.Scale(domain=[-0.15, 0.02])),
        y=alt.Y('harmonic:Q', title='Harmonic (Pitch)',
                scale=alt.Scale(domain=[-0.5, 5])),
        size=alt.Size('amplitude:Q', scale=alt.Scale(range=[30, 300])),
        color=alt.Color('amplitude:Q', scale=alt.Scale(scheme='plasma')),
        tooltip=['harmonic', 'amplitude', 'growth_rate']
    ).properties(
        width=200,
        height=300,
    ).facet(
        column=alt.Column('condition:N', title='Cellular Condition')
    ).properties(
        title='Koopman Spectra Comparison: Stability vs Pitch'
    )

    faceted_score
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 9: Phase Relationships

    Like musicians who must stay "in sync", cellular processes have
    phase relationships that determine coordination.
    """)
    return


@app.cell
def _(alt, healthy_spectrum, np, pd):
    # Calculate phase relationships between modes

    def phase_relationships():
        phase_data = []
        fundamental_freq = 1.0 / (40.0 * 60)

        oscillatory_modes = [m for m in healthy_spectrum if m['frequency'] > 0]

        for _i, mode_i in enumerate(oscillatory_modes):
            for _j, mode_j in enumerate(oscillatory_modes):
                if _i < _j:
                    # Calculate phase difference
                    omega_i = 2 * np.pi * mode_i['frequency']
                    omega_j = 2 * np.pi * mode_j['frequency']

                    # Phase is complex argument of eigenvalue
                    phase_i = omega_i  # Simplified
                    phase_j = omega_j

                    phase_diff = np.abs(phase_i - phase_j) % (2 * np.pi)
                    if phase_diff > np.pi:
                        phase_diff = 2 * np.pi - phase_diff

                    harm_i = mode_i['frequency'] / fundamental_freq
                    harm_j = mode_j['frequency'] / fundamental_freq

                    phase_data.append({
                        'mode_i': f"{harm_i:.0f}x",
                        'mode_j': f"{harm_j:.0f}x",
                        'phase_diff': np.degrees(phase_diff),
                        'relationship': 'In Phase' if phase_diff < 30 else ('Quadrature' if 60 < phase_diff < 120 else 'Out of Phase'),
                    })

        phase_df = pd.DataFrame(phase_data)

        phase_chart = alt.Chart(phase_df).mark_rect().encode(
            x=alt.X('mode_i:N', title='Mode A'),
            y=alt.Y('mode_j:N', title='Mode B'),
            color=alt.Color('phase_diff:Q', title='Phase Difference ()',
                            scale=alt.Scale(scheme='viridis', domain=[0, 180])),
            tooltip=['mode_i', 'mode_j', alt.Tooltip('phase_diff:Q', format='.1f'), 'relationship']
        ).properties(
            width=300,
            height=300,
            title='Phase Relationships Between Modes'
        )

        return phase_chart

    phase_relationships()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 10: The Cellular Score Class - Full Implementation

    Here's a demonstration of using the CellularScore class with real
    Koopman analysis from the UQ package.
    """)
    return


@app.cell
def _(CellularNote, CellularScore, CellularStaff, Clef, mo):
    # Demonstrate manual score construction
    demo_staff = CellularStaff(clef=Clef.TREBLE)

    # Add notes manually
    demo_staff.add_note(CellularNote(
        harmonic=1.0, amplitude=0.85, growth_rate=-0.001,
        frequency=0.000417, observable_class='transcriptome'
    ))
    demo_staff.add_note(CellularNote(
        harmonic=2.0, amplitude=0.42, growth_rate=-0.003,
        frequency=0.000834, observable_class='transcriptome'
    ))
    demo_staff.add_note(CellularNote(
        harmonic=3.0, amplitude=0.18, growth_rate=-0.008,
        frequency=0.001251, observable_class='transcriptome'
    ))

    demo_score = CellularScore(
        title="Manual Score Construction Demo",
        cell_cycle_time=40.0,
        key="C major",
    )
    demo_score.add_staff(demo_staff)

    mo.md(f"""
    ### Manual Score Construction

    You can build scores manually for custom analysis:

    ```python
    from tutorials.music import CellularNote, CellularStaff, CellularScore, Clef

    staff = CellularStaff(clef=Clef.TREBLE)
    staff.add_note(CellularNote(
        harmonic=1.0,        # Fundamental
        amplitude=0.85,      # fff
        growth_rate=-0.001,  # Very stable
        frequency=0.000417,  # Hz
        observable_class='transcriptome'
    ))

    score = CellularScore(title="My Analysis")
    score.add_staff(staff)
    print(score.to_ascii())
    ```

    **Output:**
    ```
    {demo_score.to_ascii()}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    ### What We've Learned

    1. **Musical notation efficiently represents Koopman spectra**
       - Pitch = frequency, duration = stability, dynamics = amplitude

    2. **The CellularScore class converts spectra to notation**
       - Automatic mapping of modes to notes
       - Observable classes as different staves/clefs

    3. **Visual representations capture musical intuition**
       - Score-like plots for quick pattern recognition
       - Orchestration charts for mode shapes
       - Complex plane for eigenvalues

    4. **Comparison is intuitive**
       - Side-by-side scores reveal condition differences
       - "Louder" modes (higher amplitude) are visually prominent

    ### Key Mappings

    | Music | Koopman | Code |
    |-------|---------|------|
    | Pitch | Frequency | `mode.frequency` |
    | Duration | Stability | `mode.growth_rate` |
    | Dynamics | Amplitude | `mode.amplitude` |
    | Instrument | Observable | `mode_shape[obs]` |
    | Tempo | Cycle time | `cell_cycle_time` |
    | Key | Harmonic structure | Frequency ratios |

    ### Next Steps

    - Try with real Koopman spectra from `uq.DynamicModeDecomposition`
    - Extend notation for uncertainty representation
    - Implement sonification (convert to actual audio)
    - Build pattern libraries for common spectral signatures
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Appendix: Integration with UQ Package

    ```python
    from uq import DynamicModeDecomposition, CellCycleKoopmanAnalyzer
    from tutorials.music import CellularScore

    # Run Koopman analysis
    dmd = DynamicModeDecomposition(rank=10)
    dmd.fit(trajectory_data)
    spectrum = dmd.get_spectrum(observable_names=['mass', 'mRNA', 'protein', 'flux'])

    # Convert to musical notation
    modes = [
        {
            'frequency': m.frequency,
            'amplitude': np.abs(m.amplitude),
            'growth_rate': m.growth_rate,
            'mode_shape': dict(zip(observable_names, np.abs(m.mode))),
        }
        for m in spectrum.modes
    ]

    score = CellularScore.from_spectrum(
        modes=modes,
        observable_names=observable_names,
        cell_cycle_time=40.0,
        title="E. coli K-12 MG1655"
    )

    print(score.to_ascii())
    ```
    """)
    return


@app.cell
def _():
    from music21 import corpus 

    s = corpus.parse('bach/bwv65.2.xml')
    return (s,)


@app.cell
def _(s):
    s.analyze('key')
    return


@app.cell
def _():
    import music21 as mc 

    # mc.configure.run()
    return (mc,)


@app.cell
def _(s):
    s.show()
    return


@app.cell
def _():
    import partitura as pt
    my_xml_file = pt.EXAMPLE_MUSICXML
    score = pt.load_score(my_xml_file)
    return pt, score


@app.cell
def _(score):
    part = score.parts[0]
    print(part.pretty())
    return (part,)


@app.cell
def _(part, pt):
    pt.render(part)
    return


@app.cell
def _(mc):

    us = mc.environment.UserSettings()
    us['musescoreDirectPNGPath'].exists()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
