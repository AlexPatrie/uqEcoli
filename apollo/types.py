"""
Type definitions for the music notation system.

This module defines the core data structures for representing cellular dynamics
as musical notation, including both lossy (standard notation) and lossless
(with exact numerical annotations) representations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np


class Clef(str, Enum):
    """Musical clef representing observable class."""

    TREBLE = "treble"  # Transcriptome
    BASS = "bass"  # Proteome
    ALTO = "alto"  # Metabolome (fluxes)
    TENOR = "tenor"  # Exchange fluxes
    PERCUSSION = "percussion"  # Higher-order properties


class ObservableClass(str, Enum):
    """Observable class per RFC006 OutputType."""

    TRANSCRIPTOME = "transcriptome"
    PROTEOME = "proteome"
    METABOLIC_FLUXES = "metabolic_fluxes"
    EXCHANGE_FLUXES = "exchange_fluxes"
    HIGHER_ORDER_PROPERTIES = "higher_order_properties"


class NoteValue(str, Enum):
    """Musical note duration values."""

    BREVE = "breve"  # Double whole note (2.0)
    WHOLE = "whole"  # 1.0
    HALF = "half"  # 0.5
    QUARTER = "quarter"  # 0.25
    EIGHTH = "eighth"  # 0.125
    SIXTEENTH = "16th"  # 0.0625
    THIRTY_SECOND = "32nd"  # 0.03125
    SIXTY_FOURTH = "64th"  # 0.015625


class Dynamic(str, Enum):
    """Musical dynamic markings."""

    PPPPP = "ppppp"  # [0.000, 0.005]
    PPPP = "pppp"  # [0.005, 0.010]
    PPP = "ppp"  # [0.010, 0.025]
    PP = "pp"  # [0.025, 0.050]
    P = "p"  # [0.050, 0.100]
    MP = "mp"  # [0.100, 0.200]
    MF = "mf"  # [0.200, 0.400]
    F = "f"  # [0.400, 0.600]
    FF = "ff"  # [0.600, 0.800]
    FFF = "fff"  # [0.800, 0.900]
    FFFF = "ffff"  # [0.900, 0.950]
    FFFFF = "fffff"  # [0.950, 1.000]


class Articulation(str, Enum):
    """Musical articulation marking growth/decay sign."""

    TENUTO = "tenuto"  # - : gamma approximately 0 (sustained)
    STACCATO = "staccato"  # . : gamma < 0 (decaying/stable)
    ACCENT = "accent"  # > : gamma > 0 (growing/unstable)
    FERMATA = "fermata"  # U : gamma = 0 exactly (on unit circle)
    MARCATO = "marcato"  # ^ : gamma >> 0 (strongly growing)


class Tempo(str, Enum):
    """Musical tempo markings."""

    GRAVE = "grave"  # 20-40 BPM, T_cycle: 120-60 min
    LARGO = "largo"  # 40-60 BPM, T_cycle: 60-40 min
    LARGHETTO = "larghetto"  # 60-66 BPM, T_cycle: 40-36 min
    ADAGIO = "adagio"  # 66-76 BPM, T_cycle: 36-32 min
    ANDANTE = "andante"  # 76-108 BPM, T_cycle: 32-22 min
    MODERATO = "moderato"  # 108-120 BPM, T_cycle: 22-20 min
    ALLEGRETTO = "allegretto"  # 112-120 BPM, T_cycle: 21-20 min
    ALLEGRO = "allegro"  # 120-156 BPM, T_cycle: 20-15 min
    VIVACE = "vivace"  # 156-176 BPM, T_cycle: 15-14 min
    PRESTO = "presto"  # 168-200 BPM, T_cycle: 14-12 min
    PRESTISSIMO = "prestissimo"  # >200 BPM, T_cycle: <12 min


class KeySignature(str, Enum):
    """Musical key signatures encoding harmonic structure."""

    C_MAJOR = "C"  # No sharps/flats: perfect integer harmonics
    G_MAJOR = "G"  # 1 sharp: harmonics ~2% high
    D_MAJOR = "D"  # 2 sharps: harmonics ~4% high
    A_MAJOR = "A"  # 3 sharps: harmonics ~6% high
    E_MAJOR = "E"  # 4 sharps: harmonics ~8% high
    F_MAJOR = "F"  # 1 flat: harmonics ~2% low
    Bb_MAJOR = "Bb"  # 2 flats: harmonics ~4% low
    Eb_MAJOR = "Eb"  # 3 flats: harmonics ~6% low
    A_MINOR = "Am"  # Minor key: altered harmonic relationships
    CHROMATIC = "chromatic"  # No stable harmonic structure


class TimeSignature(str, Enum):
    """Time signatures encoding cell cycle period."""

    CUT_TIME = "2/2"  # 120 min: stationary phase
    THREE_TWO = "3/2"  # 90 min: minimal media
    FOUR_FOUR = "4/4"  # 60 min: standard LB (reference)
    SIX_EIGHT = "6/8"  # 40 min: compound meter
    FIVE_FOUR = "5/4"  # 48 min: irregular (stressed)
    SEVEN_EIGHT = "7/8"  # 34 min: complex/perturbed


@dataclass
class ScoreMetadata:
    """Metadata for a cellular score."""

    title: str = "Cellular Score"
    strain: str = "E. coli K-12 MG1655"
    condition: str = "LB media, 37C, aerobic"
    experiment_id: str = ""

    tempo: Tempo = Tempo.MODERATO
    tempo_bpm: int = 120
    key: KeySignature = KeySignature.C_MAJOR
    time_signature: TimeSignature = TimeSignature.FOUR_FOUR

    cell_cycle_time: float = 3600.0  # seconds
    dt: float = 1.0  # timestep
    n_generations: int = 1
    n_lineages: int = 1

    fundamental_frequency: float = field(default=0.0)

    def __post_init__(self):
        if self.fundamental_frequency == 0.0:
            self.fundamental_frequency = 1.0 / self.cell_cycle_time


@dataclass
class CellularNote:
    """
    A single note representing a Koopman mode in lossy encoding.

    Attributes:
        pitch: Pitch class with octave (e.g., "A4", "E6")
        midi_number: MIDI note number
        note_value: Duration value
        dynamic: Dynamic marking
        articulation: Articulation marking
        harmonic_number: Which harmonic of the fundamental (1, 2, 3, ...)

        # Original Koopman values (for reference, not used in lossy decoding)
        frequency: Original frequency in Hz
        amplitude: Original amplitude
        growth_rate: Original growth rate
    """

    pitch: str
    midi_number: int
    note_value: NoteValue
    dynamic: Dynamic
    articulation: Articulation
    harmonic_number: int = 1

    # Original values (informational)
    frequency: float = 0.0
    amplitude: float = 0.0
    growth_rate: float = 0.0

    dots: int = 0  # Number of augmentation dots

    def __str__(self) -> str:
        dot_str = "." * self.dots
        return f"{self.pitch} {self.note_value.value}{dot_str} {self.dynamic.value} {self.articulation.value}"


@dataclass
class CellularStaff:
    """
    A staff representing one observable class.

    Attributes:
        clef: Clef determining observable class
        observable_class: The observable class this staff represents
        observable_name: Specific observable name (e.g., "lacZ mRNA")
        notes: List of notes on this staff
    """

    clef: Clef
    observable_class: ObservableClass
    observable_name: str = ""
    notes: list[CellularNote] = field(default_factory=list)

    def add_note(self, note: CellularNote) -> None:
        """Add a note to this staff."""
        self.notes.append(note)


@dataclass
class CellularScore:
    """
    A complete cellular score (lossy encoding).

    Attributes:
        metadata: Score metadata
        staves: List of staves (one per observable class)
    """

    metadata: ScoreMetadata
    staves: list[CellularStaff] = field(default_factory=list)

    def add_staff(self, staff: CellularStaff) -> None:
        """Add a staff to the score."""
        self.staves.append(staff)

    def get_staff_by_clef(self, clef: Clef) -> Optional[CellularStaff]:
        """Get staff by clef type."""
        for staff in self.staves:
            if staff.clef == clef:
                return staff
        return None

    def get_all_notes(self) -> list[CellularNote]:
        """Get all notes across all staves."""
        notes = []
        for staff in self.staves:
            notes.extend(staff.notes)
        return notes

    def to_ascii(self) -> str:
        """Generate ASCII representation of the score."""
        lines = []
        lines.append("=" * 72)
        lines.append(f"  {self.metadata.title}")
        lines.append(f'  "{self.metadata.strain}"')
        lines.append("")
        lines.append(f"  Tempo: {self.metadata.tempo.value} ({self.metadata.tempo_bpm} BPM)")
        lines.append(f"  Key: {self.metadata.key.value}")
        lines.append(f"  Time: {self.metadata.time_signature.value}")
        lines.append(f"  Cell Cycle: {self.metadata.cell_cycle_time:.0f}s")
        lines.append("=" * 72)
        lines.append("")

        for staff in self.staves:
            lines.append(f"{staff.clef.value.upper()} ({staff.observable_class.value}):")
            note_strs = [str(n) for n in staff.notes]
            lines.append("  | " + " | ".join(note_strs) + " |")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)


# =============================================================================
# Lossless Types
# =============================================================================


@dataclass
class LosslessMode:
    """
    Lossless encoding of a single Koopman mode.

    Contains exact numerical values for perfect reconstruction.

    Attributes:
        eigenvalue: Complex eigenvalue (lambda)
        amplitude: Complex amplitude coefficient (b)
        mode_shape: Complex mode shape vector (phi)
        pitch_hint: Human-readable pitch reference (optional)
        observable_names: Names of observables in mode_shape
    """

    eigenvalue: complex
    amplitude: complex
    mode_shape: np.ndarray  # Complex vector
    observable_names: list[str] = field(default_factory=list)
    pitch_hint: str = ""

    @property
    def frequency(self) -> float:
        """Compute frequency from eigenvalue."""
        if np.abs(self.eigenvalue) > 1e-10:
            return float(np.abs(np.imag(np.log(self.eigenvalue))) / (2 * np.pi))
        return 0.0

    @property
    def growth_rate(self) -> float:
        """Compute growth rate from eigenvalue."""
        if np.abs(self.eigenvalue) > 1e-10:
            return float(np.real(np.log(self.eigenvalue)))
        return 0.0

    @property
    def amplitude_magnitude(self) -> float:
        """Amplitude magnitude."""
        return float(np.abs(self.amplitude))

    @property
    def amplitude_phase(self) -> float:
        """Amplitude phase in degrees."""
        return float(np.angle(self.amplitude, deg=True))

    def to_notation(self) -> str:
        """Convert to lossless musical notation string."""
        mag = self.amplitude_magnitude
        phase = self.amplitude_phase

        # Format mode shape
        phi_strs = []
        for i, phi_i in enumerate(self.mode_shape):
            name = self.observable_names[i] if i < len(self.observable_names) else f"obs_{i}"
            phi_strs.append(f"{name}:{np.abs(phi_i):.4f}@{np.angle(phi_i, deg=True):.1f}")
        phi_str = "[" + ", ".join(phi_strs) + "]"

        return (
            f"{self.pitch_hint} "
            f"lambda=({self.eigenvalue.real:.6f},{self.eigenvalue.imag:.6f}) "
            f"b=({mag:.4f}@{phase:.1f}) "
            f"phi={phi_str}"
        )

    @classmethod
    def from_koopman_mode(cls, mode: "KoopmanMode", observable_names: list[str] | None = None) -> "LosslessMode":
        """Create from a KoopmanMode object."""
        from apollo.mappings import frequency_to_pitch

        pitch_hint = frequency_to_pitch(mode.frequency, 1.0 / 3600.0) if mode.frequency > 0 else "--"

        return cls(
            eigenvalue=mode.eigenvalue,
            amplitude=mode.amplitude,
            mode_shape=mode.mode,
            observable_names=observable_names or [],
            pitch_hint=pitch_hint,
        )


@dataclass
class LosslessScore:
    """
    Complete lossless score for exact time series reconstruction.

    Contains all information needed to perfectly reconstruct the
    original time series (within the rank approximation).

    Reconstruction formula:
        X(t) = sum_i Re[phi_i * b_i * lambda_i^t]
    """

    modes: list[LosslessMode]
    dt: float  # Timestep
    duration: int  # Number of time steps
    observable_names: list[str]

    metadata: ScoreMetadata = field(default_factory=ScoreMetadata)

    @property
    def rank(self) -> int:
        """Number of modes (rank of approximation)."""
        return len(self.modes)

    @property
    def n_observables(self) -> int:
        """Number of observables."""
        return len(self.observable_names)

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct original time series from lossless encoding.

        Returns:
            X: Reconstructed time series, shape (duration, n_observables)
        """
        n_obs = self.n_observables
        X = np.zeros((self.duration, n_obs), dtype=complex)

        for mode in self.modes:
            phi = mode.mode_shape
            b = mode.amplitude
            lam = mode.eigenvalue

            for t in range(self.duration):
                X[t] += phi * b * (lam**t)

        return np.real(X)

    def reconstruction_at_time(self, t: int) -> np.ndarray:
        """Reconstruct state at a single time point."""
        state = np.zeros(self.n_observables, dtype=complex)

        for mode in self.modes:
            state += mode.mode_shape * mode.amplitude * (mode.eigenvalue**t)

        return np.real(state)

    def to_notation(self) -> str:
        """Convert to lossless notation string."""
        lines = []
        lines.append("=" * 72)
        lines.append("  LOSSLESS CELLULAR SCORE")
        lines.append(f'  "{self.metadata.title}"')
        lines.append("")
        lines.append("  METADATA:")
        lines.append(f"    dt = {self.dt} s")
        lines.append(f"    rank = {self.rank}")
        lines.append(f"    duration = {self.duration} steps")
        lines.append(f"    observables = {self.observable_names}")
        lines.append("=" * 72)
        lines.append("")

        for i, mode in enumerate(self.modes):
            lines.append(f"MODE {i + 1}:")
            lines.append(f"    {mode.to_notation()}")
            lines.append("")

        lines.append("=" * 72)
        lines.append("RECONSTRUCTION:")
        lines.append("    X(t) = sum_i Re[phi_i * b_i * lambda_i^t]")
        lines.append("=" * 72)

        return "\n".join(lines)

    @classmethod
    def from_spectrum(
        cls,
        spectrum: "KoopmanSpectrum",
        duration: int,
        metadata: ScoreMetadata | None = None,
    ) -> "LosslessScore":
        """
        Create lossless score from Koopman spectrum.

        Args:
            spectrum: KoopmanSpectrum from DMD analysis
            duration: Number of time steps for reconstruction
            metadata: Optional score metadata
        """
        modes = []
        for m in spectrum.modes:
            modes.append(LosslessMode.from_koopman_mode(m, spectrum.observable_names))

        return cls(
            modes=modes,
            dt=spectrum.dt,
            duration=duration,
            observable_names=spectrum.observable_names,
            metadata=metadata or ScoreMetadata(cell_cycle_time=1.0 / (spectrum.dt * len(spectrum.modes))),
        )


# Type alias for external imports
if TYPE_CHECKING:
    from uq.koopman import KoopmanMode, KoopmanSpectrum
