"""
Encoding module: Convert Koopman spectra to musical scores.

This module provides both lossy (standard notation) and lossless
(with exact numerical annotations) encoding of Koopman spectral data.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from apollo.mappings import (
    amplitude_to_dynamic,
    analyze_harmonic_structure,
    classify_observable,
    cycle_time_to_tempo,
    frequency_to_harmonic_number,
    frequency_to_midi,
    frequency_to_pitch,
    growth_rate_to_articulation,
    growth_rate_to_duration,
    observable_class_to_clef,
)
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

if TYPE_CHECKING:
    from uq.koopman import KoopmanMode, KoopmanSpectrum


def encode_mode(
    mode: "KoopmanMode",
    fundamental_freq: float,
    normalize_amplitude: float = 1.0,
) -> CellularNote:
    """
    Encode a single Koopman mode as a musical note (lossy).

    Args:
        mode: KoopmanMode to encode
        fundamental_freq: Fundamental frequency for pitch reference
        normalize_amplitude: Max amplitude for normalization

    Returns:
        CellularNote representing the mode
    """
    # Frequency -> Pitch
    pitch = frequency_to_pitch(mode.frequency, fundamental_freq)
    midi = frequency_to_midi(mode.frequency, fundamental_freq)

    # Growth rate -> Duration + Articulation
    note_value = growth_rate_to_duration(mode.growth_rate)
    articulation = growth_rate_to_articulation(mode.growth_rate)

    # Amplitude -> Dynamic
    normalized_amp = abs(mode.amplitude) / normalize_amplitude if normalize_amplitude > 0 else 0
    dynamic = amplitude_to_dynamic(min(1.0, normalized_amp))

    # Determine harmonic number
    harmonic = frequency_to_harmonic_number(mode.frequency, fundamental_freq)

    return CellularNote(
        pitch=pitch,
        midi_number=midi,
        note_value=note_value,
        dynamic=dynamic,
        articulation=articulation,
        harmonic_number=harmonic,
        frequency=mode.frequency,
        amplitude=abs(mode.amplitude),
        growth_rate=mode.growth_rate,
    )


def encode_mode_lossless(
    mode: "KoopmanMode",
    observable_names: list[str],
    fundamental_freq: float = 0.0,
) -> LosslessMode:
    """
    Encode a Koopman mode with full precision (lossless).

    Args:
        mode: KoopmanMode to encode
        observable_names: Names of observables
        fundamental_freq: Fundamental frequency for pitch hint

    Returns:
        LosslessMode with exact values
    """
    # Calculate pitch hint for human readability
    if fundamental_freq > 0 and mode.frequency > 0:
        pitch_hint = frequency_to_pitch(mode.frequency, fundamental_freq)
    else:
        pitch_hint = "--"

    return LosslessMode(
        eigenvalue=mode.eigenvalue,
        amplitude=mode.amplitude,
        mode_shape=mode.mode.copy(),
        observable_names=observable_names,
        pitch_hint=pitch_hint,
    )


def encode_spectrum(
    spectrum: "KoopmanSpectrum",
    cell_cycle_time: float = 3600.0,
    title: str = "Cellular Score",
    strain: str = "E. coli K-12 MG1655",
    condition: str = "LB media, 37C, aerobic",
    n_modes: Optional[int] = None,
) -> CellularScore:
    """
    Encode a Koopman spectrum as a musical score (lossy).

    Groups modes by observable class onto different staves.

    Args:
        spectrum: KoopmanSpectrum from DMD analysis
        cell_cycle_time: Cell cycle time in seconds
        title: Score title
        strain: Strain identifier
        condition: Experimental condition
        n_modes: Number of modes to include (None = all)

    Returns:
        CellularScore representing the spectrum
    """
    # Calculate fundamental frequency
    fundamental_freq = 1.0 / cell_cycle_time

    # Determine tempo
    tempo, bpm = cycle_time_to_tempo(cell_cycle_time)

    # Analyze harmonic structure for key signature
    frequencies = [m.frequency for m in spectrum.modes if m.frequency > 0]
    key, _ = analyze_harmonic_structure(frequencies, fundamental_freq)

    # Create metadata
    metadata = ScoreMetadata(
        title=title,
        strain=strain,
        condition=condition,
        tempo=tempo,
        tempo_bpm=bpm,
        key=key,
        time_signature=TimeSignature.FOUR_FOUR,
        cell_cycle_time=cell_cycle_time,
        dt=spectrum.dt,
        fundamental_frequency=fundamental_freq,
    )

    # Create score
    score = CellularScore(metadata=metadata)

    # Get modes to encode
    modes = spectrum.get_dominant_modes(n_modes) if n_modes else spectrum.modes

    # Find max amplitude for normalization
    max_amp = max((abs(m.amplitude) for m in modes), default=1.0)

    # Group observables by class
    observable_classes: dict[ObservableClass, list[tuple[int, str]]] = {}
    for i, name in enumerate(spectrum.observable_names):
        obs_class = classify_observable(name)
        if obs_class not in observable_classes:
            observable_classes[obs_class] = []
        observable_classes[obs_class].append((i, name))

    # Create staves for each observable class
    for obs_class in ObservableClass:
        if obs_class not in observable_classes:
            continue

        clef = observable_class_to_clef(obs_class)
        staff = CellularStaff(
            clef=clef,
            observable_class=obs_class,
            observable_name=obs_class.value,
        )

        # Add notes for each mode
        for mode in modes:
            note = encode_mode(mode, fundamental_freq, max_amp)
            staff.add_note(note)

        score.add_staff(staff)

    # If no staves were created (no observables classified), create default staff
    if not score.staves:
        staff = CellularStaff(
            clef=Clef.TREBLE,
            observable_class=ObservableClass.HIGHER_ORDER_PROPERTIES,
            observable_name="observables",
        )
        for mode in modes:
            note = encode_mode(mode, fundamental_freq, max_amp)
            staff.add_note(note)
        score.add_staff(staff)

    return score


def encode_spectrum_lossless(
    spectrum: "KoopmanSpectrum",
    duration: int,
    cell_cycle_time: float = 3600.0,
    title: str = "Lossless Cellular Score",
    strain: str = "E. coli K-12 MG1655",
    n_modes: Optional[int] = None,
) -> LosslessScore:
    """
    Encode a Koopman spectrum with full precision (lossless).

    Args:
        spectrum: KoopmanSpectrum from DMD analysis
        duration: Number of time steps for reconstruction
        cell_cycle_time: Cell cycle time in seconds
        title: Score title
        strain: Strain identifier
        n_modes: Number of modes to include (None = all)

    Returns:
        LosslessScore with exact values for reconstruction
    """
    fundamental_freq = 1.0 / cell_cycle_time
    tempo, bpm = cycle_time_to_tempo(cell_cycle_time)

    # Get modes to encode
    modes_to_encode = spectrum.get_dominant_modes(n_modes) if n_modes else spectrum.modes

    # Create metadata
    metadata = ScoreMetadata(
        title=title,
        strain=strain,
        tempo=tempo,
        tempo_bpm=bpm,
        cell_cycle_time=cell_cycle_time,
        dt=spectrum.dt,
        fundamental_frequency=fundamental_freq,
    )

    # Encode each mode losslessly
    lossless_modes = []
    for mode in modes_to_encode:
        lossless_mode = encode_mode_lossless(
            mode,
            spectrum.observable_names,
            fundamental_freq,
        )
        lossless_modes.append(lossless_mode)

    return LosslessScore(
        modes=lossless_modes,
        dt=spectrum.dt,
        duration=duration,
        observable_names=spectrum.observable_names,
        metadata=metadata,
    )


def encode_trajectory(
    X: np.ndarray,
    observable_names: list[str],
    cell_cycle_time: float = 3600.0,
    dt: float = 1.0,
    rank: Optional[int] = None,
    use_edmd: bool = True,
    title: str = "Cellular Score",
) -> CellularScore:
    """
    Encode a trajectory directly as a musical score.

    Performs Koopman analysis internally and converts to score.

    Args:
        X: Trajectory data of shape (n_timesteps, n_observables)
        observable_names: Names of observables
        cell_cycle_time: Cell cycle time in seconds
        dt: Timestep
        rank: DMD rank (None = auto)
        use_edmd: Use Extended DMD
        title: Score title

    Returns:
        CellularScore representing the trajectory's Koopman spectrum
    """
    from uq.koopman import DynamicModeDecomposition, ExtendedDMD

    # Perform DMD analysis
    if use_edmd:
        dmd = ExtendedDMD(rank=rank, dt=dt)
    else:
        dmd = DynamicModeDecomposition(rank=rank, dt=dt)

    dmd.fit(X)
    spectrum = dmd.get_spectrum(observable_names)

    return encode_spectrum(
        spectrum,
        cell_cycle_time=cell_cycle_time,
        title=title,
    )


def encode_trajectory_lossless(
    X: np.ndarray,
    observable_names: list[str],
    cell_cycle_time: float = 3600.0,
    dt: float = 1.0,
    rank: Optional[int] = None,
    use_edmd: bool = True,
    title: str = "Lossless Cellular Score",
) -> LosslessScore:
    """
    Encode a trajectory as a lossless score.

    Args:
        X: Trajectory data of shape (n_timesteps, n_observables)
        observable_names: Names of observables
        cell_cycle_time: Cell cycle time in seconds
        dt: Timestep
        rank: DMD rank (None = auto)
        use_edmd: Use Extended DMD
        title: Score title

    Returns:
        LosslessScore for exact reconstruction
    """
    from uq.koopman import DynamicModeDecomposition, ExtendedDMD

    # Perform DMD analysis
    if use_edmd:
        dmd = ExtendedDMD(rank=rank, dt=dt)
    else:
        dmd = DynamicModeDecomposition(rank=rank, dt=dt)

    dmd.fit(X)
    spectrum = dmd.get_spectrum(observable_names)

    return encode_spectrum_lossless(
        spectrum,
        duration=X.shape[0],
        cell_cycle_time=cell_cycle_time,
        title=title,
    )
