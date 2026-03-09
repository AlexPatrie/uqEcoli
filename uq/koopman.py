"""
Koopman Spectral Analysis for Uncertainty Quantification.

This module implements Koopman operator-based analysis for vEcoli simulations,
providing a "spectral" or "harmonic" decomposition of cellular dynamics.

The Koopman operator is an infinite-dimensional linear operator that describes
how observables (functions of state) evolve in time. Even for nonlinear systems,
the Koopman operator is linear, allowing spectral analysis techniques.

Key concepts:
- **Koopman Modes**: Like harmonics in music, these are spatial patterns that
  evolve with simple exponential/oscillatory dynamics
- **Koopman Eigenvalues**: Determine the frequency and growth/decay rate of each mode
- **Mode Amplitudes**: Like the loudness of each harmonic, these determine
  how much each mode contributes to the signal

This provides a complementary view to PCE-based sensitivity analysis:
- PCE: Static variance-based sensitivity indices
- Koopman: Dynamic spectral decomposition revealing temporal structure

Methods implemented:
- Dynamic Mode Decomposition (DMD)
- Extended DMD (EDMD) with dictionary functions
- Spectral sensitivity analysis
- Cell cycle mode identification
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
from scipy import linalg
from scipy.signal import find_peaks

if TYPE_CHECKING:
    import polars as pl
    from duckdb import DuckDBPyConnection


class KoopmanDictionary(str, Enum):
    """Available dictionary functions for Extended DMD."""

    IDENTITY = "identity"  # Just the observables themselves
    POLYNOMIAL = "polynomial"  # Polynomial lifting
    FOURIER = "fourier"  # Fourier basis functions
    RBF = "rbf"  # Radial basis functions
    CUSTOM = "custom"


@dataclass
class KoopmanMode:
    """
    A single Koopman mode representing a harmonic of the system dynamics.

    Attributes:
        eigenvalue: Complex eigenvalue (frequency and growth rate)
        mode: Spatial pattern (how observables participate in this mode)
        amplitude: Mode amplitude (contribution to the signal)
        frequency: Oscillation frequency in Hz (derived from eigenvalue)
        growth_rate: Exponential growth/decay rate (derived from eigenvalue)
        period: Oscillation period in seconds (if oscillatory)
        damping_time: Time constant for decay (if damped)
    """

    eigenvalue: complex
    mode: np.ndarray
    amplitude: complex
    frequency: float = 0.0
    growth_rate: float = 0.0
    period: Optional[float] = None
    damping_time: Optional[float] = None

    def __post_init__(self):
        """Compute derived quantities from eigenvalue."""
        # For discrete-time systems with timestep dt:
        # eigenvalue = exp(lambda * dt) where lambda = growth_rate + i*omega
        if np.abs(self.eigenvalue) > 1e-10:
            log_eig = np.log(self.eigenvalue + 1e-10)
            self.growth_rate = float(np.real(log_eig))
            self.frequency = float(np.abs(np.imag(log_eig)) / (2 * np.pi))

            if self.frequency > 1e-10:
                self.period = 1.0 / self.frequency

            if self.growth_rate < 0:
                self.damping_time = -1.0 / self.growth_rate

    @property
    def is_oscillatory(self) -> bool:
        """Whether this mode oscillates (has imaginary component)."""
        return np.abs(np.imag(self.eigenvalue)) > 1e-10

    @property
    def is_stable(self) -> bool:
        """Whether this mode decays (|eigenvalue| < 1)."""
        return np.abs(self.eigenvalue) < 1.0

    @property
    def is_growing(self) -> bool:
        """Whether this mode grows (|eigenvalue| > 1)."""
        return np.abs(self.eigenvalue) > 1.0


@dataclass
class KoopmanSpectrum:
    """
    Complete Koopman spectral decomposition of a trajectory.

    Attributes:
        modes: List of Koopman modes sorted by amplitude
        eigenvalues: All eigenvalues
        eigenvectors: Mode matrix (columns are modes)
        amplitudes: Mode amplitudes
        reconstruction_error: Error in reconstructing original data
        dt: Timestep used for frequency calculations
        observable_names: Names of the observables
    """

    modes: list[KoopmanMode] = field(default_factory=list)
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    reconstruction_error: float = 0.0
    dt: float = 1.0
    observable_names: list[str] = field(default_factory=list)

    @property
    def n_modes(self) -> int:
        """Number of modes in the spectrum."""
        return len(self.modes)

    def get_dominant_modes(self, n: int = 5) -> list[KoopmanMode]:
        """Get the n modes with largest amplitude."""
        return sorted(self.modes, key=lambda m: np.abs(m.amplitude), reverse=True)[:n]

    def get_oscillatory_modes(self) -> list[KoopmanMode]:
        """Get modes that oscillate (have imaginary eigenvalue component)."""
        return [m for m in self.modes if m.is_oscillatory]

    def get_stable_modes(self) -> list[KoopmanMode]:
        """Get modes that decay over time."""
        return [m for m in self.modes if m.is_stable]

    def get_modes_in_frequency_range(
        self, f_min: float, f_max: float
    ) -> list[KoopmanMode]:
        """Get modes with frequency in the specified range."""
        return [m for m in self.modes if f_min <= m.frequency <= f_max]

    def reconstruct(self, t: np.ndarray) -> np.ndarray:
        """
        Reconstruct the signal at given time points using Koopman modes.

        Args:
            t: Time points (in same units as dt)

        Returns:
            Reconstructed signal of shape (len(t), n_observables)
        """
        n_obs = self.eigenvectors.shape[0]
        result = np.zeros((len(t), n_obs), dtype=complex)

        for i, ti in enumerate(t):
            for mode in self.modes:
                # x(t) = sum_j amplitude_j * mode_j * exp(eigenvalue_j * t / dt)
                result[i] += mode.amplitude * mode.mode * (mode.eigenvalue ** (ti / self.dt))

        return np.real(result)

    def get_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the power spectrum (frequency vs amplitude).

        Returns:
            Tuple of (frequencies, powers)
        """
        freqs = np.array([m.frequency for m in self.modes])
        powers = np.array([np.abs(m.amplitude) ** 2 for m in self.modes])

        # Sort by frequency
        idx = np.argsort(freqs)
        return freqs[idx], powers[idx]


class DynamicModeDecomposition:
    """
    Dynamic Mode Decomposition (DMD) for Koopman spectral analysis.

    DMD extracts spatiotemporal coherent structures from time-series data,
    approximating the Koopman modes of the underlying dynamical system.

    The algorithm finds a best-fit linear operator A such that X' ≈ A @ X,
    where X and X' are time-shifted snapshot matrices.

    References:
        - Schmid, P.J. (2010). "Dynamic mode decomposition of numerical and
          experimental data"
        - Kutz, J.N. et al. (2016). "Dynamic Mode Decomposition: Data-Driven
          Modeling of Complex Systems"
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        svd_threshold: float = 1e-10,
        dt: float = 1.0,
    ):
        """
        Initialize DMD.

        Args:
            rank: Truncation rank for SVD (None for automatic)
            svd_threshold: Threshold for singular value truncation
            dt: Timestep between snapshots (for frequency calculation)
        """
        self.rank = rank
        self.svd_threshold = svd_threshold
        self.dt = dt

        # Results stored after fit
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._Vh: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DynamicModeDecomposition":
        """
        Fit DMD to trajectory data.

        Args:
            X: Data matrix of shape (n_snapshots, n_observables)
               Each row is a snapshot at one time point

        Returns:
            self
        """
        # Transpose to (n_observables, n_snapshots) for standard DMD formulation
        X = X.T

        # Split into X and X' (time-shifted)
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # SVD of X1
        U, S, Vh = linalg.svd(X1, full_matrices=False)

        # Determine rank
        if self.rank is None:
            rank = np.sum(S > self.svd_threshold * S[0])
        else:
            rank = min(self.rank, len(S))

        # Truncate
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vhr = Vh[:rank, :]

        # Build reduced Koopman operator
        # A_tilde = Ur.T @ X2 @ Vhr.T @ diag(1/Sr)
        A_tilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / Sr)

        # Eigendecomposition of A_tilde
        eigenvalues, W = linalg.eig(A_tilde)

        # Recover full eigenvectors (DMD modes)
        # Phi = X2 @ Vhr.T @ diag(1/Sr) @ W
        eigenvectors = X2 @ Vhr.conj().T @ np.diag(1.0 / Sr) @ W

        # Compute amplitudes from initial condition
        # x0 = sum_j amplitude_j * mode_j
        # amplitudes = pinv(eigenvectors) @ x0
        x0 = X1[:, 0]
        amplitudes = linalg.lstsq(eigenvectors, x0)[0]

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._amplitudes = amplitudes
        self._U = Ur
        self._S = Sr
        self._Vh = Vhr

        return self

    def get_spectrum(
        self, observable_names: Optional[list[str]] = None
    ) -> KoopmanSpectrum:
        """
        Get the Koopman spectrum from fitted DMD.

        Args:
            observable_names: Optional names for the observables

        Returns:
            KoopmanSpectrum containing all modes
        """
        if self._eigenvalues is None:
            raise ValueError("Must call fit() before get_spectrum()")

        modes = []
        for i in range(len(self._eigenvalues)):
            mode = KoopmanMode(
                eigenvalue=self._eigenvalues[i],
                mode=self._eigenvectors[:, i],
                amplitude=self._amplitudes[i],
            )
            modes.append(mode)

        # Sort by amplitude
        modes.sort(key=lambda m: np.abs(m.amplitude), reverse=True)

        # Compute reconstruction error
        # ... (simplified for now)

        return KoopmanSpectrum(
            modes=modes,
            eigenvalues=self._eigenvalues,
            eigenvectors=self._eigenvectors,
            amplitudes=self._amplitudes,
            dt=self.dt,
            observable_names=observable_names or [],
        )


class ExtendedDMD:
    """
    Extended Dynamic Mode Decomposition (EDMD) with dictionary functions.

    EDMD lifts the observables into a higher-dimensional space using
    dictionary functions, enabling better approximation of nonlinear
    Koopman eigenfunctions.

    This is like adding "overtones" to capture nonlinear dynamics.
    """

    def __init__(
        self,
        dictionary: KoopmanDictionary = KoopmanDictionary.POLYNOMIAL,
        dictionary_order: int = 2,
        rank: Optional[int] = None,
        dt: float = 1.0,
        custom_dict_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize EDMD.

        Args:
            dictionary: Type of dictionary functions to use
            dictionary_order: Order/degree for polynomial/Fourier dictionaries
            rank: Truncation rank for SVD
            dt: Timestep between snapshots
            custom_dict_func: Custom dictionary function (if dictionary=CUSTOM)
        """
        self.dictionary = dictionary
        self.dictionary_order = dictionary_order
        self.rank = rank
        self.dt = dt
        self.custom_dict_func = custom_dict_func

        self._dmd = DynamicModeDecomposition(rank=rank, dt=dt)
        self._n_original_obs: int = 0

    def _lift(self, X: np.ndarray) -> np.ndarray:
        """
        Lift observables using dictionary functions.

        Args:
            X: Data of shape (n_snapshots, n_observables)

        Returns:
            Lifted data of shape (n_snapshots, n_lifted)
        """
        self._n_original_obs = X.shape[1]

        if self.dictionary == KoopmanDictionary.IDENTITY:
            return X

        elif self.dictionary == KoopmanDictionary.POLYNOMIAL:
            # Include polynomial terms up to given order
            lifted = [X]
            for order in range(2, self.dictionary_order + 1):
                # Add all monomials of this order
                for i in range(X.shape[1]):
                    lifted.append(X[:, i:i+1] ** order)
            return np.hstack(lifted)

        elif self.dictionary == KoopmanDictionary.FOURIER:
            # Add Fourier features
            lifted = [X]
            for k in range(1, self.dictionary_order + 1):
                for i in range(X.shape[1]):
                    lifted.append(np.sin(k * X[:, i:i+1]))
                    lifted.append(np.cos(k * X[:, i:i+1]))
            return np.hstack(lifted)

        elif self.dictionary == KoopmanDictionary.RBF:
            # Radial basis functions centered at data points
            # Use subset of points as centers
            n_centers = min(self.dictionary_order * X.shape[1], X.shape[0] // 2)
            idx = np.linspace(0, X.shape[0] - 1, n_centers, dtype=int)
            centers = X[idx]

            # Compute RBF features
            lifted = [X]
            sigma = np.std(X) + 1e-6
            for center in centers:
                dist_sq = np.sum((X - center) ** 2, axis=1, keepdims=True)
                lifted.append(np.exp(-dist_sq / (2 * sigma ** 2)))
            return np.hstack(lifted)

        elif self.dictionary == KoopmanDictionary.CUSTOM:
            if self.custom_dict_func is None:
                raise ValueError("custom_dict_func required for CUSTOM dictionary")
            return self.custom_dict_func(X)

        else:
            raise ValueError(f"Unknown dictionary: {self.dictionary}")

    def fit(self, X: np.ndarray) -> "ExtendedDMD":
        """
        Fit EDMD to trajectory data.

        Args:
            X: Data matrix of shape (n_snapshots, n_observables)

        Returns:
            self
        """
        # Lift observables
        X_lifted = self._lift(X)

        # Apply standard DMD to lifted data
        self._dmd.fit(X_lifted)

        return self

    def get_spectrum(
        self, observable_names: Optional[list[str]] = None
    ) -> KoopmanSpectrum:
        """Get the Koopman spectrum from fitted EDMD."""
        spectrum = self._dmd.get_spectrum(observable_names)

        # Project modes back to original observable space
        for mode in spectrum.modes:
            mode.mode = mode.mode[:self._n_original_obs]

        spectrum.eigenvectors = spectrum.eigenvectors[:self._n_original_obs, :]

        return spectrum


class KoopmanSensitivityAnalyzer:
    """
    Sensitivity analysis using Koopman spectral decomposition.

    This provides a "harmonic" view of how input parameters affect
    simulation dynamics:

    - **Mode Sensitivity**: How do Koopman modes change with inputs?
    - **Frequency Sensitivity**: How do oscillation frequencies shift?
    - **Amplitude Sensitivity**: How do mode strengths change?
    - **Spectral Variance Decomposition**: What fraction of variance is
      in each mode, and how does this depend on inputs?
    """

    def __init__(
        self,
        dmd_rank: Optional[int] = None,
        use_edmd: bool = False,
        dictionary: KoopmanDictionary = KoopmanDictionary.POLYNOMIAL,
        dictionary_order: int = 2,
        dt: float = 1.0,
    ):
        """
        Initialize Koopman sensitivity analyzer.

        Args:
            dmd_rank: Truncation rank for DMD
            use_edmd: Whether to use Extended DMD
            dictionary: Dictionary type for EDMD
            dictionary_order: Dictionary order for EDMD
            dt: Timestep for frequency calculations
        """
        self.dmd_rank = dmd_rank
        self.use_edmd = use_edmd
        self.dictionary = dictionary
        self.dictionary_order = dictionary_order
        self.dt = dt

    def analyze_trajectory(
        self,
        X: np.ndarray,
        observable_names: Optional[list[str]] = None,
    ) -> KoopmanSpectrum:
        """
        Analyze a single trajectory.

        Args:
            X: Trajectory data of shape (n_timesteps, n_observables)
            observable_names: Names of observables

        Returns:
            KoopmanSpectrum for this trajectory
        """
        if self.use_edmd:
            dmd = ExtendedDMD(
                dictionary=self.dictionary,
                dictionary_order=self.dictionary_order,
                rank=self.dmd_rank,
                dt=self.dt,
            )
        else:
            dmd = DynamicModeDecomposition(rank=self.dmd_rank, dt=self.dt)

        dmd.fit(X)
        return dmd.get_spectrum(observable_names)

    def compare_spectra(
        self,
        trajectories: list[np.ndarray],
        labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compare Koopman spectra across multiple trajectories/conditions.

        Args:
            trajectories: List of trajectory arrays
            labels: Labels for each trajectory

        Returns:
            Dictionary with comparison results
        """
        spectra = [self.analyze_trajectory(X) for X in trajectories]

        if labels is None:
            labels = [f"trajectory_{i}" for i in range(len(trajectories))]

        # Extract dominant frequencies
        dominant_freqs = []
        dominant_amps = []

        for spectrum in spectra:
            top_modes = spectrum.get_dominant_modes(5)
            dominant_freqs.append([m.frequency for m in top_modes])
            dominant_amps.append([np.abs(m.amplitude) for m in top_modes])

        # Compute spectral distance between trajectories
        # Using simple amplitude-weighted frequency difference
        n = len(spectra)
        spectral_distance = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Compare power spectra
                f1, p1 = spectra[i].get_power_spectrum()
                f2, p2 = spectra[j].get_power_spectrum()

                # Interpolate to common frequency grid
                f_common = np.union1d(f1, f2)
                p1_interp = np.interp(f_common, f1, p1, left=0, right=0)
                p2_interp = np.interp(f_common, f2, p2, left=0, right=0)

                # Spectral distance (L2 norm of power difference)
                dist = np.sqrt(np.sum((p1_interp - p2_interp) ** 2))
                spectral_distance[i, j] = dist
                spectral_distance[j, i] = dist

        return {
            "spectra": spectra,
            "labels": labels,
            "dominant_frequencies": dominant_freqs,
            "dominant_amplitudes": dominant_amps,
            "spectral_distance": spectral_distance,
        }

    def spectral_sensitivity(
        self,
        X_baseline: np.ndarray,
        X_perturbed: list[np.ndarray],
        parameter_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute spectral sensitivity indices.

        Measures how the Koopman spectrum changes with parameter perturbations.

        Args:
            X_baseline: Baseline trajectory
            X_perturbed: List of perturbed trajectories (one per parameter)
            parameter_names: Names of perturbed parameters

        Returns:
            Dictionary with spectral sensitivity measures
        """
        spectrum_baseline = self.analyze_trajectory(X_baseline)
        spectra_perturbed = [self.analyze_trajectory(X) for X in X_perturbed]

        # Frequency sensitivity: how much do frequencies shift?
        freq_sensitivity = []
        baseline_freqs = np.array([m.frequency for m in spectrum_baseline.get_dominant_modes(10)])

        for spectrum in spectra_perturbed:
            perturbed_freqs = np.array([m.frequency for m in spectrum.get_dominant_modes(10)])
            # Match modes by closest frequency
            freq_shift = np.mean(np.abs(baseline_freqs - perturbed_freqs))
            freq_sensitivity.append(freq_shift)

        # Amplitude sensitivity: how much do amplitudes change?
        amp_sensitivity = []
        baseline_amps = np.array([np.abs(m.amplitude) for m in spectrum_baseline.modes])
        baseline_amps = baseline_amps / (np.sum(baseline_amps) + 1e-10)

        for spectrum in spectra_perturbed:
            perturbed_amps = np.array([np.abs(m.amplitude) for m in spectrum.modes])
            perturbed_amps = perturbed_amps / (np.sum(perturbed_amps) + 1e-10)

            # Pad to same length
            max_len = max(len(baseline_amps), len(perturbed_amps))
            b_padded = np.pad(baseline_amps, (0, max_len - len(baseline_amps)))
            p_padded = np.pad(perturbed_amps, (0, max_len - len(perturbed_amps)))

            amp_shift = np.sum(np.abs(b_padded - p_padded))
            amp_sensitivity.append(amp_shift)

        # Normalize to get sensitivity indices
        freq_sensitivity = np.array(freq_sensitivity)
        amp_sensitivity = np.array(amp_sensitivity)

        total_freq = np.sum(freq_sensitivity) + 1e-10
        total_amp = np.sum(amp_sensitivity) + 1e-10

        return {
            "frequency_sensitivity": freq_sensitivity / total_freq,
            "amplitude_sensitivity": amp_sensitivity / total_amp,
            "parameter_names": parameter_names,
            "baseline_spectrum": spectrum_baseline,
            "perturbed_spectra": spectra_perturbed,
        }


class CellCycleKoopmanAnalyzer:
    """
    Specialized Koopman analysis for cell cycle dynamics.

    The cell cycle should appear as dominant oscillatory Koopman modes.
    This analyzer identifies these "cell cycle harmonics" and tracks how
    they change across different conditions.
    """

    def __init__(
        self,
        expected_cycle_time: float = 3600.0,  # Expected cell cycle time in seconds
        frequency_tolerance: float = 0.3,  # Tolerance for matching cell cycle frequency
        dt: float = 1.0,
    ):
        """
        Initialize cell cycle Koopman analyzer.

        Args:
            expected_cycle_time: Expected cell division time in seconds
            frequency_tolerance: Fractional tolerance for matching frequencies
            dt: Timestep of data
        """
        self.expected_cycle_time = expected_cycle_time
        self.expected_frequency = 1.0 / expected_cycle_time
        self.frequency_tolerance = frequency_tolerance
        self.dt = dt

        self._analyzer = KoopmanSensitivityAnalyzer(dt=dt, use_edmd=True)

    def identify_cell_cycle_modes(
        self, spectrum: KoopmanSpectrum
    ) -> list[KoopmanMode]:
        """
        Identify Koopman modes that correspond to cell cycle dynamics.

        Looks for modes with frequency near the expected cell cycle frequency
        and its harmonics.

        Args:
            spectrum: Koopman spectrum from DMD analysis

        Returns:
            List of cell cycle-related modes
        """
        cell_cycle_modes = []

        for mode in spectrum.modes:
            if not mode.is_oscillatory:
                continue

            # Check if frequency matches cell cycle or its harmonics
            for harmonic in range(1, 5):  # Check up to 4th harmonic
                expected_freq = harmonic * self.expected_frequency
                if np.abs(mode.frequency - expected_freq) < self.frequency_tolerance * expected_freq:
                    cell_cycle_modes.append(mode)
                    break

        return cell_cycle_modes

    def analyze_cell_cycle_spectrum(
        self, X: np.ndarray, observable_names: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Analyze cell cycle dynamics using Koopman decomposition.

        Args:
            X: Trajectory data of shape (n_timesteps, n_observables)
            observable_names: Names of observables

        Returns:
            Dictionary with cell cycle analysis results
        """
        spectrum = self._analyzer.analyze_trajectory(X, observable_names)
        cc_modes = self.identify_cell_cycle_modes(spectrum)

        # Compute what fraction of variance is in cell cycle modes
        total_power = sum(np.abs(m.amplitude) ** 2 for m in spectrum.modes)
        cc_power = sum(np.abs(m.amplitude) ** 2 for m in cc_modes)
        cc_fraction = cc_power / (total_power + 1e-10)

        # Find the fundamental cell cycle mode
        fundamental_mode = None
        if cc_modes:
            # Mode with frequency closest to expected
            fundamental_mode = min(
                cc_modes,
                key=lambda m: np.abs(m.frequency - self.expected_frequency)
            )

        return {
            "spectrum": spectrum,
            "cell_cycle_modes": cc_modes,
            "cell_cycle_variance_fraction": cc_fraction,
            "fundamental_mode": fundamental_mode,
            "estimated_cycle_time": 1.0 / fundamental_mode.frequency if fundamental_mode else None,
            "n_cell_cycle_harmonics": len(cc_modes),
        }


def extract_koopman_features(
    conn: "DuckDBPyConnection",
    history_sql: str,
    output_columns: list[str],
    dt: float = 1.0,
    use_edmd: bool = True,
    n_modes: int = 10,
) -> dict[str, Any]:
    """
    Extract Koopman features from simulation data for UQ analysis.

    This is a convenience function that extracts Koopman spectral features
    from vEcoli simulation outputs stored in Parquet format.

    Args:
        conn: DuckDB connection
        history_sql: SQL query for history data
        output_columns: Columns to analyze
        dt: Timestep for frequency calculation
        use_edmd: Whether to use Extended DMD
        n_modes: Number of dominant modes to extract

    Returns:
        Dictionary with Koopman features suitable for sensitivity analysis
    """
    from ecoli.library.parquet_emitter import read_stacked_columns

    # Read data
    columns_sql = ", ".join(output_columns)
    query = f"""
        SELECT {columns_sql}, time, generation, agent_id, lineage_seed
        FROM ({read_stacked_columns(history_sql, output_columns, order_results=True)})
        ORDER BY lineage_seed, generation, agent_id, time
    """

    data = conn.sql(query).pl()

    # Group by cell and analyze each trajectory
    all_spectra = []

    for (seed, gen, agent), group in data.group_by(["lineage_seed", "generation", "agent_id"]):
        # Extract observables as numpy array
        X = group.select(output_columns).to_numpy()

        if len(X) < 10:  # Skip short trajectories
            continue

        # Analyze
        if use_edmd:
            dmd = ExtendedDMD(rank=n_modes, dt=dt)
        else:
            dmd = DynamicModeDecomposition(rank=n_modes, dt=dt)

        try:
            dmd.fit(X)
            spectrum = dmd.get_spectrum(output_columns)
            all_spectra.append({
                "lineage_seed": seed,
                "generation": gen,
                "agent_id": agent,
                "spectrum": spectrum,
            })
        except Exception:
            continue

    # Aggregate features across cells
    if not all_spectra:
        return {"error": "No valid trajectories found"}

    # Extract summary statistics
    all_dom_freqs = []
    all_dom_amps = []

    for item in all_spectra:
        spectrum = item["spectrum"]
        top_modes = spectrum.get_dominant_modes(n_modes)
        all_dom_freqs.append([m.frequency for m in top_modes])
        all_dom_amps.append([np.abs(m.amplitude) for m in top_modes])

    return {
        "spectra": all_spectra,
        "mean_dominant_frequencies": np.mean(all_dom_freqs, axis=0),
        "std_dominant_frequencies": np.std(all_dom_freqs, axis=0),
        "mean_dominant_amplitudes": np.mean(all_dom_amps, axis=0),
        "std_dominant_amplitudes": np.std(all_dom_amps, axis=0),
        "n_trajectories": len(all_spectra),
    }
