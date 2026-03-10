"""
Decoding module: Convert musical scores back to Koopman spectra.

This module provides both lossy (standard notation) and lossless
decoding of musical scores to recover Koopman spectral data.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from apollo.types import (
    CellularNote,
    CellularScore,
    CellularStaff,
    LosslessMode,
    LosslessScore,
)
from apollo.mappings import (
    articulation_to_growth_sign,
    duration_to_growth_rate,
    dynamic_to_amplitude,
    pitch_to_frequency,
)

if TYPE_CHECKING:
    from uq.koopman import KoopmanMode, KoopmanSpectrum


def decode_note(
    note: CellularNote,
    fundamental_freq: float,
    n_observables: int = 1,
) -> "KoopmanMode":
    """
    Decode a musical note to a Koopman mode (lossy).

    Note: This is an approximate reconstruction since standard notation
    loses precision in frequency, amplitude, and phase.

    Args:
        note: CellularNote to decode
        fundamental_freq: Fundamental frequency for pitch reference
        n_observables: Number of observables (for mode shape)

    Returns:
        KoopmanMode (approximate)
    """
    from uq.koopman import KoopmanMode

    # Pitch -> Frequency
    frequency = pitch_to_frequency(note.pitch, fundamental_freq)

    # Dynamic -> Amplitude
    amplitude = dynamic_to_amplitude(note.dynamic)

    # Duration + Articulation -> Growth rate
    growth_magnitude = duration_to_growth_rate(note.note_value)
    growth_sign = articulation_to_growth_sign(note.articulation)
    growth_rate = growth_sign * growth_magnitude

    # Compute eigenvalue from frequency and growth rate
    # lambda = exp(gamma + i*omega) where omega = 2*pi*f
    omega = 2 * np.pi * frequency
    eigenvalue = np.exp(growth_rate + 1j * omega)

    # Mode shape: uniform distribution (no phase info in lossy encoding)
    mode_shape = np.ones(n_observables, dtype=complex) / np.sqrt(n_observables)

    return KoopmanMode(
        eigenvalue=eigenvalue,
        mode=mode_shape,
        amplitude=complex(amplitude),
    )


def decode_staff(
    staff: CellularStaff,
    fundamental_freq: float,
    n_observables: int = 1,
) -> list["KoopmanMode"]:
    """
    Decode a staff to a list of Koopman modes.

    Args:
        staff: CellularStaff to decode
        fundamental_freq: Fundamental frequency
        n_observables: Number of observables

    Returns:
        List of KoopmanMode
    """
    modes = []
    for note in staff.notes:
        mode = decode_note(note, fundamental_freq, n_observables)
        modes.append(mode)
    return modes


def decode_score(
    score: CellularScore,
    n_observables: Optional[int] = None,
) -> "KoopmanSpectrum":
    """
    Decode a musical score to a Koopman spectrum (lossy).

    Note: This reconstruction is approximate due to:
    - Frequency discretization (12 semitones/octave)
    - Amplitude discretization (10 dynamic levels)
    - Loss of phase information
    - Loss of precise mode shapes

    Args:
        score: CellularScore to decode
        n_observables: Number of observables (auto-detected if None)

    Returns:
        KoopmanSpectrum (approximate)
    """
    from uq.koopman import KoopmanSpectrum

    fundamental_freq = score.metadata.fundamental_frequency

    # Determine number of observables
    if n_observables is None:
        n_observables = len(score.staves) if score.staves else 1

    # Collect all modes from all staves
    all_modes = []

    for staff in score.staves:
        staff_modes = decode_staff(staff, fundamental_freq, n_observables)
        all_modes.extend(staff_modes)

    # Remove duplicates (same pitch across staves represents same mode)
    unique_modes = _merge_duplicate_modes(all_modes)

    # Create spectrum
    eigenvalues = np.array([m.eigenvalue for m in unique_modes])
    amplitudes = np.array([m.amplitude for m in unique_modes])

    # Stack mode shapes
    if unique_modes:
        eigenvectors = np.column_stack([m.mode for m in unique_modes])
    else:
        eigenvectors = np.array([])

    return KoopmanSpectrum(
        modes=unique_modes,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        amplitudes=amplitudes,
        dt=score.metadata.dt,
        observable_names=[staff.observable_name for staff in score.staves],
    )


def _merge_duplicate_modes(modes: list["KoopmanMode"]) -> list["KoopmanMode"]:
    """
    Merge modes that likely represent the same eigenvalue.

    Modes are considered duplicates if their frequencies are within 5%.
    """
    if not modes:
        return []

    # Sort by frequency
    sorted_modes = sorted(modes, key=lambda m: m.frequency)

    merged = [sorted_modes[0]]
    for mode in sorted_modes[1:]:
        # Check if similar to last merged mode
        last = merged[-1]
        if last.frequency > 0 and mode.frequency > 0:
            rel_diff = abs(mode.frequency - last.frequency) / last.frequency
            if rel_diff < 0.05:
                # Merge: keep higher amplitude mode
                if abs(mode.amplitude) > abs(last.amplitude):
                    merged[-1] = mode
                continue
        merged.append(mode)

    return merged


def decode_score_lossless(score: LosslessScore) -> "KoopmanSpectrum":
    """
    Decode a lossless score to a Koopman spectrum (exact).

    This is an exact reconstruction since all eigenvalues, amplitudes,
    and mode shapes are preserved.

    Args:
        score: LosslessScore to decode

    Returns:
        KoopmanSpectrum (exact)
    """
    from uq.koopman import KoopmanMode, KoopmanSpectrum

    modes = []
    for lossless_mode in score.modes:
        mode = KoopmanMode(
            eigenvalue=lossless_mode.eigenvalue,
            mode=lossless_mode.mode_shape,
            amplitude=lossless_mode.amplitude,
        )
        modes.append(mode)

    eigenvalues = np.array([m.eigenvalue for m in modes])
    amplitudes = np.array([m.amplitude for m in modes])

    if modes:
        eigenvectors = np.column_stack([m.mode for m in modes])
    else:
        eigenvectors = np.array([])

    return KoopmanSpectrum(
        modes=modes,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        amplitudes=amplitudes,
        dt=score.dt,
        observable_names=score.observable_names,
    )


def reconstruct_from_score(score: LosslessScore) -> np.ndarray:
    """
    Reconstruct time series from lossless score.

    This is a convenience function that directly returns the
    reconstructed trajectory without creating a KoopmanSpectrum.

    Args:
        score: LosslessScore

    Returns:
        Reconstructed time series, shape (duration, n_observables)
    """
    return score.reconstruct()


def verify_reconstruction(
    original: np.ndarray,
    score: LosslessScore,
) -> dict:
    """
    Verify lossless reconstruction accuracy.

    Args:
        original: Original time series
        score: LosslessScore

    Returns:
        Dictionary with verification metrics
    """
    reconstructed = score.reconstruct()

    # Truncate to same length
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len]
    recon = reconstructed[:min_len]

    # Compute errors
    abs_error = np.abs(orig - recon)
    rel_error = abs_error / (np.abs(orig) + 1e-10)

    norm_orig = np.linalg.norm(orig)
    norm_error = np.linalg.norm(orig - recon)
    relative_l2_error = norm_error / norm_orig if norm_orig > 0 else 0

    return {
        "mean_absolute_error": float(np.mean(abs_error)),
        "max_absolute_error": float(np.max(abs_error)),
        "mean_relative_error": float(np.mean(rel_error)),
        "relative_l2_error": relative_l2_error,
        "is_lossless": relative_l2_error < 1e-10,
    }


def score_to_trajectory(
    score: CellularScore,
    duration: int,
    n_observables: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a lossy score to an approximate trajectory.

    Args:
        score: CellularScore
        duration: Number of time steps
        n_observables: Number of observables

    Returns:
        Approximate trajectory, shape (duration, n_observables)
    """
    spectrum = decode_score(score, n_observables)

    # Use spectrum's reconstruct method
    t = np.arange(duration)
    return spectrum.reconstruct(t)
