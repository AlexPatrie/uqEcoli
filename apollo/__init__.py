"""
Music Package: Musical Notation for Koopman Spectral Analysis.

This package implements a bijective mapping between Western musical notation
and Koopman spectral analysis, as specified in MUSIC.md.

The mapping enables:
1. Encoding Koopman spectra as musical scores (lossy and lossless)
2. Decoding musical scores back to Koopman spectra
3. Export to MusicXML via music21
4. Human-readable visualization of cellular dynamics

Key concepts:
- Pitch ↔ Frequency (logarithmic mapping)
- Duration ↔ Mode stability (inverse growth rate)
- Dynamics ↔ Amplitude (normalized 0-1)
- Articulation ↔ Growth/decay sign
- Time signature ↔ Cell cycle period
- Key signature ↔ Harmonic structure
"""

from apollo.types import (
    Articulation,
    CellularNote,
    CellularScore,
    CellularStaff,
    Clef,
    Dynamic,
    KeySignature,
    LosslessMode,
    LosslessScore,
    NoteValue,
    ObservableClass,
    ScoreMetadata,
    Tempo,
    TimeSignature,
)
from apollo.mappings import (
    amplitude_to_dynamic,
    clef_to_observable_class,
    dynamic_to_amplitude,
    duration_to_growth_rate,
    frequency_to_pitch,
    growth_rate_to_articulation,
    growth_rate_to_duration,
    note_value_to_duration,
    observable_class_to_clef,
    pitch_to_frequency,
    tempo_to_cycle_time,
    cycle_time_to_tempo,
)
from apollo.encoding import (
    encode_spectrum,
    encode_spectrum_lossless,
    encode_mode,
    encode_mode_lossless,
)
from apollo.decoding import (
    decode_score,
    decode_score_lossless,
    decode_note,
)
from apollo.m21 import (
    score_to_music21,
    score_to_musicxml,
    lossless_score_to_music21,
)

__all__ = [
    # Types
    "Articulation",
    "CellularNote",
    "CellularScore",
    "CellularStaff",
    "Clef",
    "Dynamic",
    "KeySignature",
    "LosslessMode",
    "LosslessScore",
    "NoteValue",
    "ObservableClass",
    "ScoreMetadata",
    "Tempo",
    "TimeSignature",
    # Mappings
    "amplitude_to_dynamic",
    "clef_to_observable_class",
    "dynamic_to_amplitude",
    "duration_to_growth_rate",
    "frequency_to_pitch",
    "growth_rate_to_articulation",
    "growth_rate_to_duration",
    "note_value_to_duration",
    "observable_class_to_clef",
    "pitch_to_frequency",
    "tempo_to_cycle_time",
    "cycle_time_to_tempo",
    # Encoding
    "encode_spectrum",
    "encode_spectrum_lossless",
    "encode_mode",
    "encode_mode_lossless",
    # Decoding
    "decode_score",
    "decode_score_lossless",
    "decode_note",
    # music21 integration
    "score_to_music21",
    "score_to_musicxml",
    "lossless_score_to_music21",
]
