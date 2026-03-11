"""
Bidirectional mappings between musical notation and Koopman spectral parameters.

This module implements the precise mathematical mappings specified in MUSIC.md,
enabling conversion between cellular dynamics and musical representation.

All mappings are designed to be invertible where mathematically possible.
"""

import math
from typing import Optional

import numpy as np

from apollo.types import (
    Articulation,
    Clef,
    Dynamic,
    KeySignature,
    NoteValue,
    ObservableClass,
    Tempo,
)

# =============================================================================
# Pitch <-> Frequency Mapping
# =============================================================================

# Reference: A4 = 440 Hz (concert pitch)
# In cellular context: A4 = fundamental frequency f0
A4_MIDI = 69
A4_FREQ = 440.0

# Note names for MIDI conversion
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def frequency_to_pitch(frequency: float, fundamental_freq: float, use_sharps: bool = True) -> str:
    """
    Convert frequency to pitch name.

    Uses logarithmic mapping:
        MIDI = 12 * log2(f / f0) + 69

    Args:
        frequency: Frequency in Hz
        fundamental_freq: Fundamental frequency (maps to A4)
        use_sharps: Use sharps (True) or flats (False) for accidentals

    Returns:
        Pitch name (e.g., "A4", "E6", "C#5")
    """
    if frequency <= 0 or fundamental_freq <= 0:
        return "--"  # Non-oscillatory

    # Calculate MIDI number relative to fundamental
    # A4 (MIDI 69) = fundamental_freq
    ratio = frequency / fundamental_freq
    midi = 12 * math.log2(ratio) + A4_MIDI

    # Clamp to valid MIDI range
    midi = max(0, min(127, round(midi)))

    return midi_to_pitch_name(midi, use_sharps)


def pitch_to_frequency(pitch: str, fundamental_freq: float) -> float:
    """
    Convert pitch name to frequency.

    Args:
        pitch: Pitch name (e.g., "A4", "E6")
        fundamental_freq: Fundamental frequency (A4 maps to this)

    Returns:
        Frequency in Hz
    """
    if pitch == "--" or not pitch:
        return 0.0

    midi = pitch_name_to_midi(pitch)
    # A4 (MIDI 69) = fundamental_freq
    ratio = 2 ** ((midi - A4_MIDI) / 12)
    return fundamental_freq * ratio


def frequency_to_midi(frequency: float, fundamental_freq: float) -> int:
    """Convert frequency to MIDI note number."""
    if frequency <= 0 or fundamental_freq <= 0:
        return -1

    ratio = frequency / fundamental_freq
    midi = 12 * math.log2(ratio) + A4_MIDI
    return max(0, min(127, round(midi)))


def midi_to_frequency(midi: int, fundamental_freq: float) -> float:
    """Convert MIDI note number to frequency."""
    ratio = 2 ** ((midi - A4_MIDI) / 12)
    return fundamental_freq * ratio


def midi_to_pitch_name(midi: int, use_sharps: bool = True) -> str:
    """
    Convert MIDI number to pitch name.

    Args:
        midi: MIDI note number (0-127)
        use_sharps: Use sharps (True) or flats (False)

    Returns:
        Pitch name (e.g., "A4", "C#5")
    """
    octave = (midi // 12) - 1
    note_index = midi % 12

    note_name = NOTE_NAMES[note_index]

    if not use_sharps and "#" in note_name:
        # Convert sharp to equivalent flat
        flat_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        note_name = flat_names[note_index]

    return f"{note_name}{octave}"


def pitch_name_to_midi(pitch: str) -> int:
    """
    Convert pitch name to MIDI number.

    Args:
        pitch: Pitch name (e.g., "A4", "C#5", "Bb3")

    Returns:
        MIDI note number
    """
    # Parse pitch string
    i = 0
    note_part = ""

    # Extract note name
    while i < len(pitch) and pitch[i].isalpha():
        note_part += pitch[i]
        i += 1

    # Extract accidental
    accidental = 0
    while i < len(pitch) and pitch[i] in "#b":
        if pitch[i] == "#":
            accidental += 1
        else:
            accidental -= 1
        i += 1

    # Extract octave
    octave_str = pitch[i:]
    try:
        octave = int(octave_str)
    except ValueError:
        octave = 4

    # Find base note index
    note_upper = note_part[0].upper()
    base_notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    note_index = base_notes.get(note_upper, 9)  # Default to A

    # Calculate MIDI number
    midi = (octave + 1) * 12 + note_index + accidental
    return max(0, min(127, midi))


def frequency_to_harmonic_number(frequency: float, fundamental_freq: float) -> int:
    """
    Determine which harmonic of the fundamental a frequency corresponds to.

    Args:
        frequency: Frequency to check
        fundamental_freq: Fundamental frequency

    Returns:
        Harmonic number (1 = fundamental, 2 = first overtone, etc.)
        Returns 0 if not close to any harmonic
    """
    if frequency <= 0 or fundamental_freq <= 0:
        return 0

    ratio = frequency / fundamental_freq

    # Check against integer harmonics with 10% tolerance
    for n in range(1, 16):
        if abs(ratio - n) < 0.1 * n:
            return n

    return 0


# =============================================================================
# Duration <-> Mode Stability Mapping
# =============================================================================

# Duration values (in beats, where quarter = 1)
DURATION_VALUES = {
    NoteValue.BREVE: 8.0,
    NoteValue.WHOLE: 4.0,
    NoteValue.HALF: 2.0,
    NoteValue.QUARTER: 1.0,
    NoteValue.EIGHTH: 0.5,
    NoteValue.SIXTEENTH: 0.25,
    NoteValue.THIRTY_SECOND: 0.125,
    NoteValue.SIXTY_FOURTH: 0.0625,
}

# Growth rate thresholds for duration mapping
# |gamma| = k / D where k = 0.01 (default scaling)
GROWTH_RATE_THRESHOLDS = {
    NoteValue.BREVE: 0.005,  # Extremely stable
    NoteValue.WHOLE: 0.01,  # Very stable
    NoteValue.HALF: 0.02,  # Stable
    NoteValue.QUARTER: 0.04,  # Moderately stable
    NoteValue.EIGHTH: 0.08,  # Moderate decay
    NoteValue.SIXTEENTH: 0.16,  # Fast decay
    NoteValue.THIRTY_SECOND: 0.32,  # Very fast decay
    NoteValue.SIXTY_FOURTH: float("inf"),  # Transient
}


def growth_rate_to_duration(growth_rate: float) -> NoteValue:
    """
    Convert growth rate magnitude to note duration.

    Longer notes = more stable modes (smaller |gamma|).

    Args:
        growth_rate: Growth rate (can be positive or negative)

    Returns:
        NoteValue representing stability
    """
    abs_gamma = abs(growth_rate)

    for note_value, threshold in GROWTH_RATE_THRESHOLDS.items():
        if abs_gamma < threshold:
            return note_value

    return NoteValue.SIXTY_FOURTH


def duration_to_growth_rate(note_value: NoteValue, k: float = 0.01) -> float:
    """
    Convert note duration to growth rate magnitude.

    Uses formula: |gamma| = k / D

    Args:
        note_value: Note duration
        k: Scaling constant (default 0.01)

    Returns:
        Growth rate magnitude (positive)
    """
    duration = DURATION_VALUES.get(note_value, 1.0)
    return k / duration


def note_value_to_duration(note_value: NoteValue, dots: int = 0) -> float:
    """
    Convert note value to duration in beats.

    Args:
        note_value: Base note value
        dots: Number of augmentation dots

    Returns:
        Duration in beats (quarter = 1.0)
    """
    base_duration = DURATION_VALUES.get(note_value, 1.0)

    # Each dot adds half the previous value
    total = base_duration
    dot_value = base_duration / 2
    for _ in range(dots):
        total += dot_value
        dot_value /= 2

    return total


# =============================================================================
# Dynamics <-> Amplitude Mapping
# =============================================================================

# Dynamic ranges (amplitude normalized to 0-1)
DYNAMIC_RANGES = {
    Dynamic.PPPPP: (0.000, 0.005),
    Dynamic.PPPP: (0.005, 0.010),
    Dynamic.PPP: (0.010, 0.025),
    Dynamic.PP: (0.025, 0.050),
    Dynamic.P: (0.050, 0.100),
    Dynamic.MP: (0.100, 0.200),
    Dynamic.MF: (0.200, 0.400),
    Dynamic.F: (0.400, 0.600),
    Dynamic.FF: (0.600, 0.800),
    Dynamic.FFF: (0.800, 0.900),
    Dynamic.FFFF: (0.900, 0.950),
    Dynamic.FFFFF: (0.950, 1.000),
}

# Midpoints for decoding
DYNAMIC_MIDPOINTS = {
    Dynamic.PPPPP: 0.0025,
    Dynamic.PPPP: 0.0075,
    Dynamic.PPP: 0.0175,
    Dynamic.PP: 0.0375,
    Dynamic.P: 0.075,
    Dynamic.MP: 0.15,
    Dynamic.MF: 0.30,
    Dynamic.F: 0.50,
    Dynamic.FF: 0.70,
    Dynamic.FFF: 0.85,
    Dynamic.FFFF: 0.925,
    Dynamic.FFFFF: 0.975,
}


def amplitude_to_dynamic(amplitude: float) -> Dynamic:
    """
    Convert amplitude to dynamic marking.

    Args:
        amplitude: Amplitude (assumed normalized 0-1, or will be clamped)

    Returns:
        Dynamic marking
    """
    # Clamp to valid range
    amp = max(0.0, min(1.0, amplitude))

    for dynamic, (low, high) in DYNAMIC_RANGES.items():
        if low <= amp < high:
            return dynamic

    return Dynamic.FFFFF


def dynamic_to_amplitude(dynamic: Dynamic) -> float:
    """
    Convert dynamic marking to amplitude.

    Returns midpoint of the dynamic's range.

    Args:
        dynamic: Dynamic marking

    Returns:
        Amplitude value (0-1)
    """
    return DYNAMIC_MIDPOINTS.get(dynamic, 0.5)


# =============================================================================
# Articulation <-> Growth Rate Sign Mapping
# =============================================================================


def growth_rate_to_articulation(growth_rate: float, threshold: float = 0.001) -> Articulation:
    """
    Convert growth rate to articulation marking.

    Articulation encodes the SIGN of gamma:
    - Staccato (.) : gamma < 0 (decaying/stable)
    - Tenuto (-) : gamma ~ 0 (sustained)
    - Accent (>) : gamma > 0 (growing/unstable)
    - Marcato (^) : gamma >> 0 (strongly growing)

    Args:
        growth_rate: Growth rate
        threshold: Threshold for "near zero"

    Returns:
        Articulation marking
    """
    if growth_rate > 10 * threshold:
        return Articulation.MARCATO
    elif growth_rate > threshold:
        return Articulation.ACCENT
    elif growth_rate < -threshold:
        return Articulation.STACCATO
    else:
        return Articulation.TENUTO


def articulation_to_growth_sign(articulation: Articulation) -> int:
    """
    Convert articulation to growth rate sign.

    Args:
        articulation: Articulation marking

    Returns:
        -1 for decaying, 0 for sustained, +1 for growing
    """
    sign_map = {
        Articulation.STACCATO: -1,
        Articulation.TENUTO: 0,
        Articulation.FERMATA: 0,
        Articulation.ACCENT: 1,
        Articulation.MARCATO: 1,
    }
    return sign_map.get(articulation, 0)


# =============================================================================
# Tempo <-> Cell Cycle Time Mapping
# =============================================================================

# Tempo BPM ranges
TEMPO_BPM_RANGES = {
    Tempo.GRAVE: (20, 40),
    Tempo.LARGO: (40, 60),
    Tempo.LARGHETTO: (60, 66),
    Tempo.ADAGIO: (66, 76),
    Tempo.ANDANTE: (76, 108),
    Tempo.MODERATO: (108, 120),
    Tempo.ALLEGRETTO: (112, 120),
    Tempo.ALLEGRO: (120, 156),
    Tempo.VIVACE: (156, 176),
    Tempo.PRESTO: (168, 200),
    Tempo.PRESTISSIMO: (200, 300),
}


def cycle_time_to_tempo(cycle_time_seconds: float) -> tuple[Tempo, int]:
    """
    Convert cell cycle time to tempo marking and BPM.

    Uses formula: T_cycle = (4 / BPM) * 60 minutes
    Therefore: BPM = 240 / (T_cycle_minutes)

    Args:
        cycle_time_seconds: Cell cycle time in seconds

    Returns:
        Tuple of (Tempo marking, exact BPM)
    """
    cycle_time_minutes = cycle_time_seconds / 60.0
    bpm = int(240.0 / cycle_time_minutes)

    # Clamp BPM to valid range
    bpm = max(20, min(300, bpm))

    # Find appropriate tempo marking
    tempo = Tempo.MODERATO
    for t, (low, high) in TEMPO_BPM_RANGES.items():
        if low <= bpm < high:
            tempo = t
            break

    return tempo, bpm


def tempo_to_cycle_time(tempo: Tempo, bpm: Optional[int] = None) -> float:
    """
    Convert tempo to cell cycle time.

    Args:
        tempo: Tempo marking
        bpm: Exact BPM (if None, uses midpoint of tempo's range)

    Returns:
        Cell cycle time in seconds
    """
    if bpm is None:
        low, high = TEMPO_BPM_RANGES.get(tempo, (108, 120))
        bpm = (low + high) // 2

    # T_cycle = (4 / BPM) * 60 minutes = 240 / BPM minutes
    cycle_time_minutes = 240.0 / bpm
    return cycle_time_minutes * 60.0


# =============================================================================
# Clef <-> Observable Class Mapping
# =============================================================================

CLEF_TO_OBSERVABLE = {
    Clef.TREBLE: ObservableClass.TRANSCRIPTOME,
    Clef.BASS: ObservableClass.PROTEOME,
    Clef.ALTO: ObservableClass.METABOLIC_FLUXES,
    Clef.TENOR: ObservableClass.EXCHANGE_FLUXES,
    Clef.PERCUSSION: ObservableClass.HIGHER_ORDER_PROPERTIES,
}

OBSERVABLE_TO_CLEF = {v: k for k, v in CLEF_TO_OBSERVABLE.items()}


def clef_to_observable_class(clef: Clef) -> ObservableClass:
    """Convert clef to observable class."""
    return CLEF_TO_OBSERVABLE.get(clef, ObservableClass.HIGHER_ORDER_PROPERTIES)


def observable_class_to_clef(obs_class: ObservableClass) -> Clef:
    """Convert observable class to clef."""
    return OBSERVABLE_TO_CLEF.get(obs_class, Clef.PERCUSSION)


def classify_observable(name: str) -> ObservableClass:
    """
    Classify an observable by its name.

    Args:
        name: Observable name (e.g., "mRNA_lacZ", "protein_DnaA", "mass")

    Returns:
        ObservableClass
    """
    name_lower = name.lower()

    if any(kw in name_lower for kw in ["mrna", "rna", "transcript"]):
        return ObservableClass.TRANSCRIPTOME
    elif any(kw in name_lower for kw in ["protein", "enzyme"]):
        return ObservableClass.PROTEOME
    elif any(kw in name_lower for kw in ["flux", "reaction"]):
        return ObservableClass.METABOLIC_FLUXES
    elif any(kw in name_lower for kw in ["exchange", "uptake", "secretion"]):
        return ObservableClass.EXCHANGE_FLUXES
    else:
        return ObservableClass.HIGHER_ORDER_PROPERTIES


# =============================================================================
# Key Signature <-> Harmonic Structure Mapping
# =============================================================================


def harmonic_deviation_to_key(mean_deviation: float) -> KeySignature:
    """
    Convert mean harmonic deviation to key signature.

    Deviation = (actual_freq - expected_freq) / expected_freq

    Args:
        mean_deviation: Mean fractional deviation from perfect harmonics

    Returns:
        KeySignature
    """
    if mean_deviation > 0.06:
        return KeySignature.E_MAJOR  # 4 sharps
    elif mean_deviation > 0.04:
        return KeySignature.A_MAJOR  # 3 sharps
    elif mean_deviation > 0.02:
        return KeySignature.D_MAJOR  # 2 sharps
    elif mean_deviation > 0.01:
        return KeySignature.G_MAJOR  # 1 sharp
    elif mean_deviation < -0.06:
        return KeySignature.Eb_MAJOR  # 3 flats
    elif mean_deviation < -0.04:
        return KeySignature.Bb_MAJOR  # 2 flats
    elif mean_deviation < -0.02:
        return KeySignature.F_MAJOR  # 1 flat
    else:
        return KeySignature.C_MAJOR  # No accidentals


def analyze_harmonic_structure(
    frequencies: list[float],
    fundamental_freq: float,
) -> tuple[KeySignature, float]:
    """
    Analyze frequencies to determine key signature.

    Args:
        frequencies: List of mode frequencies
        fundamental_freq: Expected fundamental frequency

    Returns:
        Tuple of (KeySignature, mean_deviation)
    """
    if not frequencies or fundamental_freq <= 0:
        return KeySignature.C_MAJOR, 0.0

    deviations = []
    for freq in frequencies:
        if freq > 0:
            # Find nearest integer harmonic
            ratio = freq / fundamental_freq
            nearest_harmonic = round(ratio)
            if nearest_harmonic > 0:
                expected = nearest_harmonic * fundamental_freq
                deviation = (freq - expected) / expected
                deviations.append(deviation)

    if not deviations:
        return KeySignature.C_MAJOR, 0.0

    mean_dev = sum(deviations) / len(deviations)
    return harmonic_deviation_to_key(mean_dev), mean_dev
