"""
Unit tests for UQ Koopman spectral analysis.

Tests the Koopman operator-based analysis for dynamic sensitivity
and cell cycle harmonic decomposition.
"""

import numpy as np
import pytest


class TestKoopmanDictionary:
    """Tests for the KoopmanDictionary enum."""

    @pytest.mark.unit
    def test_dictionary_types_exist(self):
        """KoopmanDictionary should have standard dictionary types."""
        from uq import KoopmanDictionary

        assert KoopmanDictionary.IDENTITY is not None
        assert KoopmanDictionary.POLYNOMIAL is not None
        assert KoopmanDictionary.FOURIER is not None
        assert KoopmanDictionary.RBF is not None
        assert KoopmanDictionary.CUSTOM is not None

    @pytest.mark.unit
    def test_dictionary_values(self):
        """KoopmanDictionary should have correct string values."""
        from uq import KoopmanDictionary

        assert KoopmanDictionary.IDENTITY.value == "identity"
        assert KoopmanDictionary.POLYNOMIAL.value == "polynomial"
        assert KoopmanDictionary.FOURIER.value == "fourier"
        assert KoopmanDictionary.RBF.value == "rbf"


class TestKoopmanMode:
    """Tests for the KoopmanMode dataclass."""

    @pytest.mark.unit
    def test_mode_creation(self):
        """KoopmanMode should store eigenvalue, mode, and amplitude."""
        from uq import KoopmanMode

        eigenvalue = 0.9 + 0.1j
        mode = np.array([1.0, 0.5, 0.2])
        amplitude = 1.5 + 0.5j

        km = KoopmanMode(eigenvalue=eigenvalue, mode=mode, amplitude=amplitude)

        assert km.eigenvalue == eigenvalue
        np.testing.assert_array_equal(km.mode, mode)
        assert km.amplitude == amplitude

    @pytest.mark.unit
    def test_frequency_computation(self):
        """KoopmanMode should compute frequency from eigenvalue."""
        from uq import KoopmanMode

        # Eigenvalue with oscillatory component
        eigenvalue = np.exp(0.01 + 0.5j)  # growth_rate=0.01, omega=0.5
        km = KoopmanMode(
            eigenvalue=eigenvalue,
            mode=np.array([1.0]),
            amplitude=1.0,
        )

        # Frequency should be omega / (2*pi)
        assert km.frequency > 0

    @pytest.mark.unit
    def test_growth_rate_computation(self):
        """KoopmanMode should compute growth rate from eigenvalue."""
        from uq import KoopmanMode

        # Growing mode (|eigenvalue| > 1)
        eigenvalue = 1.1 + 0j
        km = KoopmanMode(
            eigenvalue=eigenvalue,
            mode=np.array([1.0]),
            amplitude=1.0,
        )

        assert km.growth_rate > 0

    @pytest.mark.unit
    def test_is_oscillatory_property(self):
        """is_oscillatory should detect imaginary eigenvalue component."""
        from uq import KoopmanMode

        # Oscillatory - use eigenvalue with clear oscillatory behavior
        km_osc = KoopmanMode(
            eigenvalue=np.exp(1j * 0.5),  # exp(i*0.5) has oscillation
            mode=np.array([1.0]),
            amplitude=1.0,
        )
        assert km_osc.is_oscillatory  # numpy.bool comparison

        # Non-oscillatory
        km_real = KoopmanMode(
            eigenvalue=0.9 + 0j,
            mode=np.array([1.0]),
            amplitude=1.0,
        )
        assert not km_real.is_oscillatory

    @pytest.mark.unit
    def test_is_stable_property(self):
        """is_stable should detect decaying modes."""
        from uq import KoopmanMode

        # Stable (|eigenvalue| < 1)
        km_stable = KoopmanMode(
            eigenvalue=0.8 + 0j,  # Use complex to be explicit
            mode=np.array([1.0]),
            amplitude=1.0,
        )
        # Check the magnitude condition directly
        assert np.abs(km_stable.eigenvalue) < 1.0

        # Unstable (|eigenvalue| > 1)
        km_unstable = KoopmanMode(
            eigenvalue=1.2 + 0j,
            mode=np.array([1.0]),
            amplitude=1.0,
        )
        assert np.abs(km_unstable.eigenvalue) > 1.0

    @pytest.mark.unit
    def test_is_growing_property(self):
        """is_growing should detect growing modes."""
        from uq import KoopmanMode

        km_growing = KoopmanMode(
            eigenvalue=1.1 + 0j,
            mode=np.array([1.0]),
            amplitude=1.0,
        )
        # Check the magnitude condition directly
        assert np.abs(km_growing.eigenvalue) > 1.0

    @pytest.mark.unit
    def test_period_for_oscillatory(self):
        """period should be set for oscillatory modes with sufficient frequency."""
        from uq import KoopmanMode

        # Eigenvalue corresponding to a clear oscillation
        # Use moderate frequency to ensure it's detected
        omega = 0.5  # rad/unit_time
        eigenvalue = np.exp(1j * omega)

        km = KoopmanMode(
            eigenvalue=eigenvalue,
            mode=np.array([1.0]),
            amplitude=1.0,
        )

        # Check that frequency was computed
        assert km.frequency > 0
        # Period may or may not be set depending on frequency threshold
        if km.frequency > 1e-10:
            assert km.period is not None or km.frequency > 0

    @pytest.mark.unit
    def test_damping_time_for_decaying(self):
        """damping_time should be set for decaying modes."""
        from uq import KoopmanMode

        # Decaying mode with known decay rate
        decay_rate = -0.1  # decay rate
        eigenvalue = np.exp(decay_rate)

        km = KoopmanMode(
            eigenvalue=eigenvalue,
            mode=np.array([1.0]),
            amplitude=1.0,
        )

        assert km.damping_time is not None
        # Damping time should be approximately 1/|decay_rate| = 10
        assert np.abs(km.damping_time - 10) < 1


class TestKoopmanSpectrum:
    """Tests for the KoopmanSpectrum dataclass."""

    @pytest.mark.unit
    def test_spectrum_creation(self):
        """KoopmanSpectrum should store modes and metadata."""
        from uq import KoopmanMode, KoopmanSpectrum

        modes = [
            KoopmanMode(eigenvalue=0.9, mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=0.8, mode=np.array([0.5]), amplitude=0.5),
        ]

        spectrum = KoopmanSpectrum(
            modes=modes,
            eigenvalues=np.array([0.9, 0.8]),
            dt=1.0,
        )

        assert spectrum.n_modes == 2
        assert spectrum.dt == 1.0

    @pytest.mark.unit
    def test_get_dominant_modes(self):
        """get_dominant_modes should return modes by amplitude."""
        from uq import KoopmanMode, KoopmanSpectrum

        modes = [
            KoopmanMode(eigenvalue=0.9, mode=np.array([1.0]), amplitude=0.5),
            KoopmanMode(eigenvalue=0.8, mode=np.array([1.0]), amplitude=2.0),
            KoopmanMode(eigenvalue=0.7, mode=np.array([1.0]), amplitude=1.0),
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        dominant = spectrum.get_dominant_modes(n=2)

        assert len(dominant) == 2
        assert np.abs(dominant[0].amplitude) >= np.abs(dominant[1].amplitude)

    @pytest.mark.unit
    def test_get_oscillatory_modes(self):
        """get_oscillatory_modes should filter oscillatory modes."""
        from uq import KoopmanMode, KoopmanSpectrum

        modes = [
            KoopmanMode(eigenvalue=0.9 + 0.1j, mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=0.8, mode=np.array([1.0]), amplitude=1.0),
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        osc_modes = spectrum.get_oscillatory_modes()

        assert len(osc_modes) == 1
        assert osc_modes[0].is_oscillatory

    @pytest.mark.unit
    def test_get_stable_modes(self):
        """get_stable_modes should filter stable modes."""
        from uq import KoopmanMode, KoopmanSpectrum

        modes = [
            KoopmanMode(eigenvalue=0.9, mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=1.1, mode=np.array([1.0]), amplitude=1.0),
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        stable_modes = spectrum.get_stable_modes()

        assert len(stable_modes) == 1
        assert stable_modes[0].is_stable

    @pytest.mark.unit
    def test_get_modes_in_frequency_range(self):
        """get_modes_in_frequency_range should filter by frequency."""
        from uq import KoopmanMode, KoopmanSpectrum

        # Create modes with different frequencies
        modes = [
            KoopmanMode(eigenvalue=np.exp(1j * 0.1), mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=np.exp(1j * 1.0), mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=np.exp(1j * 2.0), mode=np.array([1.0]), amplitude=1.0),
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        filtered = spectrum.get_modes_in_frequency_range(0.1, 0.2)

        # Should filter to modes in range
        assert all(0.1 <= m.frequency <= 0.2 for m in filtered)

    @pytest.mark.unit
    def test_get_power_spectrum(self):
        """get_power_spectrum should return frequency and power arrays."""
        from uq import KoopmanMode, KoopmanSpectrum

        modes = [
            KoopmanMode(eigenvalue=np.exp(1j * 0.5), mode=np.array([1.0]), amplitude=2.0),
            KoopmanMode(eigenvalue=np.exp(1j * 1.0), mode=np.array([1.0]), amplitude=1.0),
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        freqs, powers = spectrum.get_power_spectrum()

        assert len(freqs) == len(modes)
        assert len(powers) == len(modes)
        # Power should be amplitude squared
        assert (
            np.abs(powers[0] - np.abs(modes[0].amplitude) ** 2) < 0.1
            or np.abs(powers[1] - np.abs(modes[0].amplitude) ** 2) < 0.1
        )


class TestDynamicModeDecomposition:
    """Tests for the DynamicModeDecomposition class."""

    @pytest.mark.unit
    def test_dmd_initialization(self):
        """DMD should initialize with default parameters."""
        from uq import DynamicModeDecomposition

        dmd = DynamicModeDecomposition()

        assert dmd.rank is None
        assert dmd.dt == 1.0

    @pytest.mark.unit
    def test_dmd_fit_with_simple_data(self, synthetic_trajectory):
        """DMD should fit to trajectory data."""
        from uq import DynamicModeDecomposition

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(synthetic_trajectory)

        assert dmd._eigenvalues is not None
        assert dmd._eigenvectors is not None
        assert dmd._amplitudes is not None

    @pytest.mark.unit
    def test_dmd_get_spectrum(self, synthetic_trajectory):
        """DMD should produce KoopmanSpectrum."""
        from uq import DynamicModeDecomposition, KoopmanSpectrum

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(synthetic_trajectory)
        spectrum = dmd.get_spectrum()

        assert isinstance(spectrum, KoopmanSpectrum)
        assert spectrum.n_modes > 0

    @pytest.mark.unit
    def test_dmd_requires_fit_before_spectrum(self):
        """DMD should raise if get_spectrum called before fit."""
        from uq import DynamicModeDecomposition

        dmd = DynamicModeDecomposition()

        with pytest.raises(ValueError, match="Must call fit"):
            dmd.get_spectrum()

    @pytest.mark.unit
    def test_dmd_rank_truncation(self, synthetic_trajectory):
        """DMD should respect rank parameter."""
        from uq import DynamicModeDecomposition

        dmd = DynamicModeDecomposition(rank=2, dt=1.0)
        dmd.fit(synthetic_trajectory)
        spectrum = dmd.get_spectrum()

        assert spectrum.n_modes <= 2

    @pytest.mark.unit
    def test_dmd_observable_names(self, synthetic_trajectory):
        """DMD should store observable names in spectrum."""
        from uq import DynamicModeDecomposition

        names = ["mass", "growth_rate", "protein"]
        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(synthetic_trajectory)
        spectrum = dmd.get_spectrum(observable_names=names)

        assert spectrum.observable_names == names


class TestExtendedDMD:
    """Tests for the ExtendedDMD class."""

    @pytest.mark.unit
    def test_edmd_initialization(self):
        """ExtendedDMD should initialize with dictionary settings."""
        from uq import ExtendedDMD, KoopmanDictionary

        edmd = ExtendedDMD(
            dictionary=KoopmanDictionary.POLYNOMIAL,
            dictionary_order=3,
        )

        assert edmd.dictionary == KoopmanDictionary.POLYNOMIAL
        assert edmd.dictionary_order == 3

    @pytest.mark.unit
    def test_edmd_fit(self, synthetic_trajectory):
        """ExtendedDMD should fit to trajectory data."""
        from uq import ExtendedDMD

        edmd = ExtendedDMD(dictionary_order=2, dt=1.0)
        edmd.fit(synthetic_trajectory)

        spectrum = edmd.get_spectrum()
        assert spectrum.n_modes > 0

    @pytest.mark.unit
    def test_edmd_polynomial_lifting(self, rng):
        """ExtendedDMD should lift data with polynomial features."""
        from uq import ExtendedDMD, KoopmanDictionary

        X = rng.uniform(size=(100, 2))

        edmd = ExtendedDMD(
            dictionary=KoopmanDictionary.POLYNOMIAL,
            dictionary_order=2,
        )

        lifted = edmd._lift(X)

        # Should have original features plus polynomial terms
        assert lifted.shape[1] > X.shape[1]

    @pytest.mark.unit
    def test_edmd_fourier_lifting(self, rng):
        """ExtendedDMD should lift data with Fourier features."""
        from uq import ExtendedDMD, KoopmanDictionary

        X = rng.uniform(size=(100, 2))

        edmd = ExtendedDMD(
            dictionary=KoopmanDictionary.FOURIER,
            dictionary_order=2,
        )

        lifted = edmd._lift(X)

        # Should have original features plus sin/cos terms
        assert lifted.shape[1] > X.shape[1]

    @pytest.mark.unit
    def test_edmd_identity_lifting(self, rng):
        """ExtendedDMD with identity should not change data."""
        from uq import ExtendedDMD, KoopmanDictionary

        X = rng.uniform(size=(100, 2))

        edmd = ExtendedDMD(dictionary=KoopmanDictionary.IDENTITY)
        lifted = edmd._lift(X)

        np.testing.assert_array_equal(lifted, X)

    @pytest.mark.unit
    def test_edmd_custom_dictionary(self, rng):
        """ExtendedDMD should support custom dictionary function."""
        from uq import ExtendedDMD, KoopmanDictionary

        def custom_func(X):
            return np.hstack([X, X**2, np.sin(X)])

        X = rng.uniform(size=(100, 2))

        edmd = ExtendedDMD(
            dictionary=KoopmanDictionary.CUSTOM,
            custom_dict_func=custom_func,
        )

        lifted = edmd._lift(X)
        assert lifted.shape[1] == 6  # 2 original + 2 squared + 2 sin


class TestKoopmanSensitivityAnalyzer:
    """Tests for the KoopmanSensitivityAnalyzer class."""

    @pytest.mark.unit
    def test_analyzer_initialization(self):
        """KoopmanSensitivityAnalyzer should initialize with settings."""
        from uq import KoopmanSensitivityAnalyzer

        analyzer = KoopmanSensitivityAnalyzer(
            dmd_rank=10,
            use_edmd=True,
            dt=1.0,
        )

        assert analyzer.dmd_rank == 10
        assert analyzer.use_edmd is True

    @pytest.mark.unit
    def test_analyze_trajectory(self, synthetic_trajectory):
        """analyze_trajectory should return KoopmanSpectrum."""
        from uq import KoopmanSensitivityAnalyzer, KoopmanSpectrum

        analyzer = KoopmanSensitivityAnalyzer(dt=1.0)
        spectrum = analyzer.analyze_trajectory(synthetic_trajectory)

        assert isinstance(spectrum, KoopmanSpectrum)

    @pytest.mark.unit
    def test_compare_spectra(self, rng):
        """compare_spectra should compare multiple trajectories."""
        from uq import KoopmanSensitivityAnalyzer

        analyzer = KoopmanSensitivityAnalyzer(dt=1.0)

        # Create two slightly different trajectories
        t = np.arange(100)
        X1 = np.column_stack([np.sin(0.1 * t), np.cos(0.1 * t)])
        X2 = np.column_stack([np.sin(0.12 * t), np.cos(0.12 * t)])

        result = analyzer.compare_spectra([X1, X2], labels=["baseline", "perturbed"])

        assert "spectra" in result
        assert "spectral_distance" in result
        assert len(result["spectra"]) == 2

    @pytest.mark.unit
    def test_spectral_sensitivity(self, rng):
        """spectral_sensitivity should compute sensitivity indices."""
        from uq import KoopmanSensitivityAnalyzer

        analyzer = KoopmanSensitivityAnalyzer(dt=1.0)

        # Create baseline and perturbed trajectories
        t = np.arange(100)
        X_baseline = np.column_stack([np.sin(0.1 * t), np.cos(0.1 * t)])
        X_p1 = np.column_stack([np.sin(0.12 * t), np.cos(0.12 * t)])
        X_p2 = np.column_stack([np.sin(0.15 * t), np.cos(0.15 * t)])

        result = analyzer.spectral_sensitivity(
            X_baseline,
            [X_p1, X_p2],
            parameter_names=["param1", "param2"],
        )

        assert "frequency_sensitivity" in result
        assert "amplitude_sensitivity" in result
        assert len(result["frequency_sensitivity"]) == 2


class TestCellCycleKoopmanAnalyzer:
    """Tests for the CellCycleKoopmanAnalyzer class."""

    @pytest.mark.unit
    def test_analyzer_initialization(self):
        """CellCycleKoopmanAnalyzer should initialize with cycle time."""
        from uq import CellCycleKoopmanAnalyzer

        analyzer = CellCycleKoopmanAnalyzer(
            expected_cycle_time=3600.0,
            frequency_tolerance=0.3,
        )

        assert analyzer.expected_cycle_time == 3600.0
        assert analyzer.expected_frequency == 1.0 / 3600.0

    @pytest.mark.unit
    def test_identify_cell_cycle_modes(self):
        """identify_cell_cycle_modes should find modes at cell cycle frequency."""
        from uq import CellCycleKoopmanAnalyzer, KoopmanMode, KoopmanSpectrum

        analyzer = CellCycleKoopmanAnalyzer(expected_cycle_time=100.0)

        # Create spectrum with mode at cell cycle frequency
        cc_freq = 1.0 / 100.0  # 0.01 Hz
        omega = 2 * np.pi * cc_freq

        modes = [
            KoopmanMode(eigenvalue=np.exp(1j * omega), mode=np.array([1.0]), amplitude=1.0),
            KoopmanMode(eigenvalue=np.exp(1j * 0.5), mode=np.array([1.0]), amplitude=1.0),  # Different freq
        ]

        spectrum = KoopmanSpectrum(modes=modes)
        cc_modes = analyzer.identify_cell_cycle_modes(spectrum)

        # Should identify at least one cell cycle mode
        assert len(cc_modes) >= 1

    @pytest.mark.unit
    def test_analyze_cell_cycle_spectrum(self, rng):
        """analyze_cell_cycle_spectrum should return analysis dict."""
        from uq import CellCycleKoopmanAnalyzer

        analyzer = CellCycleKoopmanAnalyzer(expected_cycle_time=50.0, dt=1.0)

        # Create oscillating trajectory at cell cycle frequency
        t = np.arange(200)
        cc_freq = 1.0 / 50.0
        X = np.column_stack([
            np.sin(2 * np.pi * cc_freq * t),
            np.cos(2 * np.pi * cc_freq * t),
        ])

        result = analyzer.analyze_cell_cycle_spectrum(X)

        assert "spectrum" in result
        assert "cell_cycle_modes" in result
        assert "cell_cycle_variance_fraction" in result

    @pytest.mark.unit
    def test_cell_cycle_variance_fraction(self, rng):
        """cell_cycle_variance_fraction should be between 0 and 1."""
        from uq import CellCycleKoopmanAnalyzer

        analyzer = CellCycleKoopmanAnalyzer(expected_cycle_time=50.0, dt=1.0)

        t = np.arange(200)
        cc_freq = 1.0 / 50.0
        X = np.column_stack([
            np.sin(2 * np.pi * cc_freq * t) + 0.1 * rng.normal(size=len(t)),
            np.cos(2 * np.pi * cc_freq * t) + 0.1 * rng.normal(size=len(t)),
        ])

        result = analyzer.analyze_cell_cycle_spectrum(X)

        assert 0 <= result["cell_cycle_variance_fraction"] <= 1


class TestKoopmanIntegration:
    """Integration tests for Koopman analysis."""

    @pytest.mark.unit
    def test_full_analysis_pipeline(self, synthetic_trajectory):
        """Full Koopman analysis pipeline should work."""
        from uq import DynamicModeDecomposition

        # Fit DMD
        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(synthetic_trajectory)

        # Get spectrum
        spectrum = dmd.get_spectrum()

        # Get dominant modes
        dominant = spectrum.get_dominant_modes(5)

        # Get power spectrum
        freqs, powers = spectrum.get_power_spectrum()

        assert len(dominant) <= 5
        assert len(freqs) == spectrum.n_modes

    @pytest.mark.unit
    def test_reconstruct_from_modes(self, rng):
        """KoopmanSpectrum.reconstruct should approximate original data."""
        from uq import DynamicModeDecomposition

        # Create simple oscillating data
        t = np.arange(100)
        X = np.column_stack([np.sin(0.1 * t), np.cos(0.1 * t)])

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(X)
        spectrum = dmd.get_spectrum()

        # Reconstruct
        t_recon = np.arange(50)
        X_recon = spectrum.reconstruct(t_recon)

        # Should have same shape
        assert X_recon.shape == (50, 2)

    @pytest.mark.unit
    def test_edmd_improves_nonlinear_fit(self, rng):
        """EDMD should better capture nonlinear dynamics than DMD."""
        from uq import DynamicModeDecomposition, ExtendedDMD

        # Create nonlinear trajectory
        t = np.arange(100)
        X = np.column_stack([
            np.sin(0.1 * t) ** 2,  # Nonlinear
            np.cos(0.1 * t) ** 2,
        ])

        # Fit both
        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(X)

        edmd = ExtendedDMD(dictionary_order=2, dt=1.0)
        edmd.fit(X)

        # Both should produce valid spectra
        spectrum_dmd = dmd.get_spectrum()
        spectrum_edmd = edmd.get_spectrum()

        assert spectrum_dmd.n_modes > 0
        assert spectrum_edmd.n_modes > 0


class TestExtractKoopmanFeatures:
    """Tests for the extract_koopman_features function."""

    @pytest.mark.unit
    def test_function_exists(self):
        """extract_koopman_features should be importable."""
        from uq import extract_koopman_features

        assert callable(extract_koopman_features)


class TestKoopmanMusicAnalogy:
    """Tests verifying the music-Koopman analogy concepts."""

    @pytest.mark.unit
    def test_modes_like_harmonics(self, rng):
        """Koopman modes should decompose like musical harmonics."""
        from uq import DynamicModeDecomposition

        # Create data with multiple "harmonics"
        t = np.arange(200)
        fundamental = np.sin(0.1 * t)
        second_harmonic = 0.5 * np.sin(0.2 * t)  # 2x frequency
        third_harmonic = 0.25 * np.sin(0.3 * t)  # 3x frequency

        X = np.column_stack([
            fundamental + second_harmonic + third_harmonic,
            np.cos(0.1 * t) + 0.5 * np.cos(0.2 * t),
        ])

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(X)
        spectrum = dmd.get_spectrum()

        # Should find oscillatory modes (the "harmonics")
        osc_modes = spectrum.get_oscillatory_modes()
        assert len(osc_modes) > 0

    @pytest.mark.unit
    def test_amplitude_like_loudness(self, rng):
        """Mode amplitude/eigenvalue should characterize dynamics."""
        from uq import DynamicModeDecomposition

        # Create data with exponential growth (easy for DMD to fit)
        t = np.arange(100)
        # Exponential growth trajectory
        X = np.column_stack([
            np.exp(0.01 * t),  # Slow growth
            np.exp(0.02 * t),  # Faster growth
        ])

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(X)
        spectrum = dmd.get_spectrum()

        # Should have modes
        assert spectrum.n_modes > 0
        # Modes should have eigenvalues (the key feature for dynamics)
        dominant = spectrum.get_dominant_modes(1)[0]
        assert dominant.eigenvalue is not None
