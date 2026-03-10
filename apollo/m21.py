"""
music21 integration for MusicXML export and visualization.

This module provides integration with the music21 library for:
- Exporting cellular scores to MusicXML format
- Creating music21 Score objects for visualization
- Adding lossless annotations as lyrics/text expressions
"""

from typing import Optional

from apollo.types import (
    Articulation,
    CellularNote,
    CellularScore,
    CellularStaff,
    Clef,
    Dynamic,
    LosslessMode,
    LosslessScore,
    NoteValue,
)


# Mapping from our types to music21 types
NOTE_VALUE_TO_M21 = {
    NoteValue.BREVE: "breve",
    NoteValue.WHOLE: "whole",
    NoteValue.HALF: "half",
    NoteValue.QUARTER: "quarter",
    NoteValue.EIGHTH: "eighth",
    NoteValue.SIXTEENTH: "16th",
    NoteValue.THIRTY_SECOND: "32nd",
    NoteValue.SIXTY_FOURTH: "64th",
}

CLEF_TO_M21 = {
    Clef.TREBLE: "treble",
    Clef.BASS: "bass",
    Clef.ALTO: "alto",
    Clef.TENOR: "tenor",
    Clef.PERCUSSION: "percussion",
}

DYNAMIC_TO_M21 = {
    Dynamic.PPPPP: "ppppp",
    Dynamic.PPPP: "pppp",
    Dynamic.PPP: "ppp",
    Dynamic.PP: "pp",
    Dynamic.P: "p",
    Dynamic.MP: "mp",
    Dynamic.MF: "mf",
    Dynamic.F: "f",
    Dynamic.FF: "ff",
    Dynamic.FFF: "fff",
    Dynamic.FFFF: "ffff",
    Dynamic.FFFFF: "fffff",
}

ARTICULATION_TO_M21 = {
    Articulation.TENUTO: "tenuto",
    Articulation.STACCATO: "staccato",
    Articulation.ACCENT: "accent",
    Articulation.FERMATA: "fermata",
    Articulation.MARCATO: "strongAccent",
}


def _import_music21():
    """Import music21, raising helpful error if not installed."""
    try:
        import music21
        return music21
    except ImportError as e:
        raise ImportError(
            "music21 is required for MusicXML export. "
            "Install with: uv sync --group music"
        ) from e


def note_to_music21(note: CellularNote):
    """
    Convert CellularNote to music21 Note.

    Args:
        note: CellularNote to convert

    Returns:
        music21.note.Note
    """
    m21 = _import_music21()

    # Create note from pitch
    if note.pitch == "--":
        # Non-oscillatory mode: use rest
        m21_note = m21.note.Rest()
    else:
        m21_note = m21.note.Note(note.pitch)

    # Set duration
    m21_duration = NOTE_VALUE_TO_M21.get(note.note_value, "quarter")
    m21_note.duration = m21.duration.Duration(m21_duration)

    # Add dots
    for _ in range(note.dots):
        m21_note.duration.dots += 1

    # Add articulation
    artic_name = ARTICULATION_TO_M21.get(note.articulation)
    if artic_name:
        if artic_name == "fermata":
            m21_note.expressions.append(m21.expressions.Fermata())
        else:
            artic_class = getattr(m21.articulations, artic_name.capitalize(), None)
            if artic_class:
                m21_note.articulations.append(artic_class())

    # Add lyric with harmonic info
    if note.harmonic_number > 0:
        m21_note.lyric = f"H{note.harmonic_number}"

    return m21_note


def staff_to_music21(staff: CellularStaff, tempo_bpm: int = 120):
    """
    Convert CellularStaff to music21 Part.

    Args:
        staff: CellularStaff to convert
        tempo_bpm: Tempo in BPM

    Returns:
        music21.stream.Part
    """
    m21 = _import_music21()

    part = m21.stream.Part()
    part.partName = staff.observable_name or staff.observable_class.value

    # Add clef
    clef_name = CLEF_TO_M21.get(staff.clef, "treble")
    if clef_name == "percussion":
        part.append(m21.clef.PercussionClef())
    else:
        clef_class = getattr(m21.clef, f"{clef_name.capitalize()}Clef", m21.clef.TrebleClef)
        part.append(clef_class())

    # Create measure
    measure = m21.stream.Measure(number=1)

    # Add time signature
    measure.append(m21.meter.TimeSignature("4/4"))

    # Add tempo
    measure.append(m21.tempo.MetronomeMark(number=tempo_bpm))

    # Track current dynamic for grouping
    current_dynamic = None

    for note in staff.notes:
        # Add dynamic if changed
        dyn_name = DYNAMIC_TO_M21.get(note.dynamic)
        if dyn_name and dyn_name != current_dynamic:
            dyn = m21.dynamics.Dynamic(dyn_name)
            measure.append(dyn)
            current_dynamic = dyn_name

        # Add note
        m21_note = note_to_music21(note)
        measure.append(m21_note)

    part.append(measure)
    return part


def score_to_music21(score: CellularScore):
    """
    Convert CellularScore to music21 Score.

    Args:
        score: CellularScore to convert

    Returns:
        music21.stream.Score
    """
    m21 = _import_music21()

    m21_score = m21.stream.Score()

    # Add metadata
    m21_score.metadata = m21.metadata.Metadata()
    m21_score.metadata.title = score.metadata.title
    m21_score.metadata.composer = f"Strain: {score.metadata.strain}"

    # Add key signature to first part
    key_str = score.metadata.key.value
    if key_str == "chromatic":
        key = m21.key.Key("C")  # Default for chromatic
    elif "m" in key_str.lower():
        # Minor key
        key = m21.key.Key(key_str.replace("m", "").replace("M", ""), "minor")
    else:
        key = m21.key.Key(key_str, "major")

    # Convert each staff to a part
    for i, staff in enumerate(score.staves):
        part = staff_to_music21(staff, score.metadata.tempo_bpm)

        # Add key to first part
        if i == 0:
            part.insert(0, key)

        m21_score.append(part)

    return m21_score


def score_to_musicxml(score: CellularScore, filepath: Optional[str] = None) -> str:
    """
    Export CellularScore to MusicXML format.

    Args:
        score: CellularScore to export
        filepath: Optional file path to write (if None, returns XML string)

    Returns:
        MusicXML string
    """
    m21 = _import_music21()

    m21_score = score_to_music21(score)

    if filepath:
        m21_score.write("musicxml", fp=filepath)
        with open(filepath) as f:
            return f.read()
    else:
        return m21_score.write("musicxml")


def lossless_mode_to_music21(mode: LosslessMode, measure_offset: float = 0.0):
    """
    Convert LosslessMode to music21 Note with annotations.

    Adds exact eigenvalue/amplitude/mode_shape as text expressions.

    Args:
        mode: LosslessMode to convert
        measure_offset: Offset within measure

    Returns:
        music21.note.Note with annotations
    """
    m21 = _import_music21()

    # Create note from pitch hint
    if mode.pitch_hint == "--":
        m21_note = m21.note.Rest()
    else:
        m21_note = m21.note.Note(mode.pitch_hint)

    # Set duration based on growth rate
    abs_gamma = abs(mode.growth_rate)
    if abs_gamma < 0.005:
        duration_type = "breve"
    elif abs_gamma < 0.01:
        duration_type = "whole"
    elif abs_gamma < 0.02:
        duration_type = "half"
    elif abs_gamma < 0.04:
        duration_type = "quarter"
    else:
        duration_type = "eighth"

    m21_note.duration = m21.duration.Duration(duration_type)

    # Add lossless annotation as lyric
    # Format: λ=(re,im) b=(mag,phase)
    annotation = (
        f"λ=({mode.eigenvalue.real:.4f},{mode.eigenvalue.imag:.4f}) "
        f"b=({mode.amplitude_magnitude:.3f}∠{mode.amplitude_phase:.1f}°)"
    )
    m21_note.lyric = annotation

    # Add mode shape as text expression
    phi_str = " ".join([f"{abs(p):.2f}" for p in mode.mode_shape[:4]])
    text_expr = m21.expressions.TextExpression(f"φ=[{phi_str}...]")
    m21_note.expressions.append(text_expr)

    return m21_note


def lossless_score_to_music21(score: LosslessScore):
    """
    Convert LosslessScore to music21 Score with annotations.

    Includes exact numerical values as lyrics and text expressions
    for lossless roundtrip (when parsed).

    Args:
        score: LosslessScore to convert

    Returns:
        music21.stream.Score
    """
    m21 = _import_music21()

    m21_score = m21.stream.Score()

    # Add metadata
    m21_score.metadata = m21.metadata.Metadata()
    m21_score.metadata.title = f"LOSSLESS: {score.metadata.title}"
    m21_score.metadata.composer = f"dt={score.dt}s rank={score.rank}"

    # Add header text with reconstruction info
    header = m21.text.TextBox(
        f"LOSSLESS CELLULAR SCORE\n"
        f"dt = {score.dt}s\n"
        f"rank = {score.rank}\n"
        f"duration = {score.duration} steps\n"
        f"observables = {', '.join(score.observable_names[:5])}"
    )
    m21_score.insert(0, header)

    # Create single part with all modes
    part = m21.stream.Part()
    part.partName = "Koopman Modes"
    part.append(m21.clef.TrebleClef())

    measure = m21.stream.Measure(number=1)
    measure.append(m21.meter.TimeSignature("4/4"))

    for mode in score.modes:
        m21_note = lossless_mode_to_music21(mode)
        measure.append(m21_note)

    part.append(measure)
    m21_score.append(part)

    return m21_score


def show_score(score: CellularScore, fmt: str = "musicxml"):
    """
    Display a score using music21's show method.

    Args:
        score: CellularScore to display
        fmt: Format for display (musicxml, text, lily, etc.)
    """
    m21_score = score_to_music21(score)
    m21_score.show(fmt)


def show_lossless_score(score: LosslessScore, fmt: str = "text"):
    """
    Display a lossless score.

    Args:
        score: LosslessScore to display
        fmt: Format for display
    """
    m21_score = lossless_score_to_music21(score)
    m21_score.show(fmt)
