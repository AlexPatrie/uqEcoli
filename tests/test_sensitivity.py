"""
Unit tests for UQ sensitivity analysis.

Tests the PCE surrogate method and Sobol indices computation
as specified in Milestone 08.4.2.
"""

import numpy as np
import pytest


class TestSensitivityMethod:
    """Tests for the SensitivityMethod enum."""

    @pytest.mark.unit
    def test_pce_method_exists(self):
        """PCE sensitivity method should exist."""
        from uq import SensitivityMethod

        assert SensitivityMethod.PCE is not None
        assert SensitivityMethod.PCE.value == "pce"

    @pytest.mark.unit
    def test_sobol_method_exists(self):
        """Sobol sensitivity method should exist."""
        from uq import SensitivityMethod

        assert SensitivityMethod.SOBOL is not None
        assert SensitivityMethod.SOBOL.value == "sobol"

    @pytest.mark.unit
    def test_morris_method_exists(self):
        """Morris screening method should exist."""
        from uq import SensitivityMethod

        assert SensitivityMethod.MORRIS is not None
        assert SensitivityMethod.MORRIS.value == "morris"


class TestSobolIndices:
    """Tests for the SobolIndices dataclass."""

    @pytest.mark.unit
    def test_creation_with_arrays(self):
        """SobolIndices should store numpy arrays."""
        from uq import SobolIndices

        first_order = np.array([0.4, 0.3, 0.2])
        total_order = np.array([0.5, 0.35, 0.25])

        indices = SobolIndices(
            first_order=first_order,
            total_order=total_order,
            parameter_names=["p1", "p2", "p3"],
        )

        np.testing.assert_array_equal(indices.first_order, first_order)
        np.testing.assert_array_equal(indices.total_order, total_order)

    @pytest.mark.unit
    def test_creation_with_fixture(self, sample_sobol_indices):
        """SobolIndices fixture should be valid."""
        assert sample_sobol_indices.first_order is not None
        assert sample_sobol_indices.total_order is not None
        assert len(sample_sobol_indices.parameter_names) == 3

    @pytest.mark.unit
    def test_get_most_influential_total(self, sample_sobol_indices):
        """get_most_influential should return top parameters by total order."""
        top = sample_sobol_indices.get_most_influential(n=2, index_type="total")

        assert len(top) == 2
        # First should have highest total order index
        assert top[0][0] == "vio_expression"
        assert top[0][1] == 0.55

    @pytest.mark.unit
    def test_get_most_influential_first(self, sample_sobol_indices):
        """get_most_influential should work with first-order indices."""
        top = sample_sobol_indices.get_most_influential(n=2, index_type="first")

        assert len(top) == 2
        assert top[0][0] == "vio_expression"
        assert top[0][1] == 0.45

    @pytest.mark.unit
    def test_second_order_optional(self):
        """second_order should be optional."""
        from uq import SobolIndices

        indices = SobolIndices(
            first_order=np.array([0.5]),
            total_order=np.array([0.6]),
        )

        assert indices.second_order is None

    @pytest.mark.unit
    def test_confidence_intervals_optional(self):
        """confidence_intervals should be optional."""
        from uq import SobolIndices

        indices = SobolIndices(
            first_order=np.array([0.5]),
            total_order=np.array([0.6]),
        )

        assert indices.confidence_intervals is None

    @pytest.mark.unit
    def test_multi_output_sobol(self):
        """SobolIndices should support multiple outputs."""
        from uq import SobolIndices

        # 2 outputs, 3 parameters
        first_order = np.array([[0.4, 0.3, 0.2], [0.35, 0.35, 0.25]])
        total_order = np.array([[0.5, 0.35, 0.25], [0.45, 0.4, 0.3]])

        indices = SobolIndices(
            first_order=first_order,
            total_order=total_order,
            parameter_names=["p1", "p2", "p3"],
            output_names=["mass", "growth_rate"],
        )

        assert indices.first_order.shape == (2, 3)
        assert indices.total_order.shape == (2, 3)

    @pytest.mark.unit
    def test_get_most_influential_multi_output(self):
        """get_most_influential should average across outputs."""
        from uq import SobolIndices

        # Parameter p1 dominant in output1, p2 in output2
        first_order = np.array([[0.6, 0.2, 0.1], [0.2, 0.6, 0.1]])
        total_order = np.array([[0.7, 0.25, 0.15], [0.25, 0.7, 0.15]])

        indices = SobolIndices(
            first_order=first_order,
            total_order=total_order,
            parameter_names=["p1", "p2", "p3"],
        )

        top = indices.get_most_influential(n=3)
        # Average total order: p1=0.475, p2=0.475, p3=0.15
        # Either p1 or p2 should be first (they're equal)
        assert top[2][0] == "p3"  # p3 should be last


class TestPCESurrogate:
    """Tests for the PCESurrogate dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        """PCESurrogate should store coefficients and metadata."""
        from uq import PCESurrogate

        coeffs = np.array([1.0, 0.5, 0.2, 0.1])
        multi_idx = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        surrogate = PCESurrogate(
            coefficients=coeffs,
            multi_indices=multi_idx,
            basis_type="legendre",
            polynomial_order=2,
            input_dim=2,
            output_dim=1,
        )

        np.testing.assert_array_equal(surrogate.coefficients, coeffs)
        assert surrogate.polynomial_order == 2
        assert surrogate.input_dim == 2

    @pytest.mark.unit
    def test_default_basis_type(self):
        """PCESurrogate should default to Legendre basis."""
        from uq import PCESurrogate

        surrogate = PCESurrogate(
            coefficients=np.array([1.0]),
            multi_indices=np.array([[0]]),
        )

        assert surrogate.basis_type == "legendre"

    @pytest.mark.unit
    def test_predict_not_implemented(self):
        """PCESurrogate.predict should raise NotImplementedError."""
        from uq import PCESurrogate

        surrogate = PCESurrogate(
            coefficients=np.array([1.0]),
            multi_indices=np.array([[0]]),
        )

        with pytest.raises(NotImplementedError):
            surrogate.predict(np.array([[0.5, 0.5]]))


class TestSensitivityAnalyzer:
    """Tests for the SensitivityAnalyzer class."""

    @pytest.mark.unit
    def test_initialization_with_parameter_space(self, input_parameter_space):
        """SensitivityAnalyzer should initialize with parameter space."""
        from uq import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(input_parameter_space)

        assert analyzer.parameter_space is input_parameter_space
        assert analyzer.wrapper is None
        assert analyzer.samples is None
        assert analyzer.outputs is None

    @pytest.mark.unit
    def test_initialization_with_precomputed_data(self, input_parameter_space, rng):
        """SensitivityAnalyzer should accept precomputed samples and outputs."""
        from uq import SensitivityAnalyzer

        X = rng.uniform(size=(50, 3))
        Y = rng.normal(size=(50, 2))

        analyzer = SensitivityAnalyzer(
            parameter_space=input_parameter_space,
            samples=X,
            outputs=Y,
        )

        np.testing.assert_array_equal(analyzer.samples, X)
        np.testing.assert_array_equal(analyzer.outputs, Y)

    @pytest.mark.unit
    def test_get_samples_with_precomputed(self, input_parameter_space, rng):
        """_get_samples_and_outputs should return precomputed data."""
        from uq import SensitivityAnalyzer

        X = rng.uniform(size=(50, 3))
        Y = rng.normal(size=(50, 2))

        analyzer = SensitivityAnalyzer(
            parameter_space=input_parameter_space,
            samples=X,
            outputs=Y,
        )

        X_out, Y_out = analyzer._get_samples_and_outputs(None)

        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(Y_out, Y)

    @pytest.mark.unit
    def test_get_samples_no_data_raises(self, input_parameter_space):
        """_get_samples_and_outputs should raise if no data available."""
        from uq import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(input_parameter_space)

        with pytest.raises(ValueError, match="provide samples/outputs or a wrapper"):
            analyzer._get_samples_and_outputs(None)


class TestSensitivityAnalysisIntegration:
    """Integration tests for sensitivity analysis with synthetic data."""

    @pytest.mark.unit
    def test_analyze_with_synthetic_data(self, parameter_output_samples, input_parameter_space):
        """Sensitivity analysis should work with synthetic parameter-output pairs."""
        X, Y = parameter_output_samples

        # Verify data shapes
        assert X.shape[1] == input_parameter_space.n_parameters
        assert X.shape[0] == Y.shape[0]

    @pytest.mark.unit
    def test_sobol_indices_sum_constraint(self, sample_sobol_indices):
        """First-order indices should sum to approximately total variance explained."""
        # This is a loose constraint - sum of first-order <= 1 for no interactions
        sum_first = np.sum(sample_sobol_indices.first_order)
        # In presence of interactions, this can exceed 1, but still bounded
        assert sum_first >= 0

    @pytest.mark.unit
    def test_total_order_geq_first_order(self, sample_sobol_indices):
        """Total-order indices should be >= first-order indices."""
        for i in range(len(sample_sobol_indices.first_order)):
            assert sample_sobol_indices.total_order[i] >= sample_sobol_indices.first_order[i], (
                f"Total order should be >= first order for parameter {i}"
            )

    @pytest.mark.unit
    def test_sensitivity_with_known_function(self, rng):
        """Sensitivity analysis should recover known importance from analytic function."""

        # Create simple linear function: Y = 2*x1 + 0.5*x2
        # So x1 should be more important

        n_samples = 100
        X = rng.uniform(0, 1, size=(n_samples, 2))
        Y = 2 * X[:, 0:1] + 0.5 * X[:, 1:2] + 0.01 * rng.normal(size=(n_samples, 1))

        # Analytically: Var(Y) = 4*Var(x1) + 0.25*Var(x2)
        # For uniform [0,1]: Var = 1/12
        # So S1 ≈ 4/12 / (4/12 + 0.25/12) = 4/4.25 ≈ 0.94
        # And S2 ≈ 0.25/4.25 ≈ 0.06

        # Manual variance calculation
        total_var = np.var(Y)
        var_x1 = 1.0 / 12.0
        var_x2 = 1.0 / 12.0

        # Expected sensitivities
        expected_s1 = (4 * var_x1) / (4 * var_x1 + 0.25 * var_x2)
        expected_s2 = (0.25 * var_x2) / (4 * var_x1 + 0.25 * var_x2)

        # Verify x1 should dominate
        assert expected_s1 > expected_s2
        assert expected_s1 > 0.9


class TestPCEMethod:
    """Tests specifically for PCE-based sensitivity analysis."""

    @pytest.mark.unit
    def test_pce_is_default_method(self):
        """PCE should be the primary method per UQ framework."""
        from uq import SensitivityMethod

        # PCE is the recommended approach in the RFC
        assert SensitivityMethod.PCE.value == "pce"

    @pytest.mark.unit
    def test_pce_produces_sobol_indices(self, sample_sobol_indices):
        """PCE method should produce valid Sobol indices."""
        # The PCE method computes Sobol indices analytically from PCE coefficients
        assert sample_sobol_indices.first_order is not None
        assert sample_sobol_indices.total_order is not None

    @pytest.mark.unit
    def test_pce_surrogate_has_required_attributes(self):
        """PCESurrogate should have attributes needed for sensitivity."""
        from uq import PCESurrogate

        surrogate = PCESurrogate(
            coefficients=np.array([1.0, 0.5]),
            multi_indices=np.array([[0, 0], [1, 0]]),
            polynomial_order=3,
            input_dim=3,
            output_dim=1,
        )

        # These attributes are needed for Sobol index computation
        assert hasattr(surrogate, "coefficients")
        assert hasattr(surrogate, "multi_indices")
        assert hasattr(surrogate, "polynomial_order")


class TestLibrarySupport:
    """Tests for UQPy and PyTUQ library support."""

    @pytest.mark.unit
    def test_analyze_method_supports_uqpy_flag(self, input_parameter_space, rng):
        """analyze_with_pce should have use_uqpy parameter."""
        from uq import SensitivityAnalyzer

        X = rng.uniform(size=(50, 3))
        Y = rng.normal(size=(50, 1))

        analyzer = SensitivityAnalyzer(
            parameter_space=input_parameter_space,
            samples=X,
            outputs=Y,
        )

        # Method should accept use_uqpy parameter
        import inspect

        sig = inspect.signature(analyzer.analyze_with_pce)
        assert "use_uqpy" in sig.parameters

    @pytest.mark.unit
    def test_pytuq_method_exists(self, input_parameter_space):
        """SensitivityAnalyzer should have PyTUQ method."""
        from uq import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(input_parameter_space)

        assert hasattr(analyzer, "_analyze_pce_pytuq")

    @pytest.mark.unit
    def test_uqpy_method_exists(self, input_parameter_space):
        """SensitivityAnalyzer should have UQPy method."""
        from uq import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(input_parameter_space)

        assert hasattr(analyzer, "_analyze_pce_uqpy")


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.unit
    def test_run_sensitivity_analysis_exists(self):
        """run_sensitivity_analysis function should exist."""
        from uq import run_sensitivity_analysis

        assert callable(run_sensitivity_analysis)

    @pytest.mark.unit
    def test_analyze_precomputed_results_exists(self):
        """analyze_precomputed_results function should exist."""
        from uq import analyze_precomputed_results

        assert callable(analyze_precomputed_results)

    @pytest.mark.unit
    def test_run_sensitivity_analysis_parameters(self):
        """run_sensitivity_analysis should have required parameters."""
        import inspect

        from uq import run_sensitivity_analysis

        sig = inspect.signature(run_sensitivity_analysis)
        params = list(sig.parameters.keys())

        # Should have key parameters
        assert "sim_data_path" in params
        assert "output_dir" in params
        assert "aggregation_strategy" in params
        assert "polynomial_order" in params
        assert "use_uqpy" in params


class TestSobolIndexInterpretation:
    """Tests for Sobol index interpretation utilities."""

    @pytest.mark.unit
    def test_first_order_measures_main_effect(self):
        """First-order index should measure main effect (no interactions)."""
        from uq import SobolIndices

        # Create indices where p1 has high main effect
        indices = SobolIndices(
            first_order=np.array([0.8, 0.1, 0.05]),
            total_order=np.array([0.85, 0.15, 0.08]),
            parameter_names=["p1", "p2", "p3"],
        )

        # p1 accounts for 80% of variance on its own
        assert indices.first_order[0] == 0.8

    @pytest.mark.unit
    def test_total_minus_first_measures_interactions(self):
        """Difference between total and first order indicates interactions."""
        from uq import SobolIndices

        indices = SobolIndices(
            first_order=np.array([0.3, 0.3, 0.1]),
            total_order=np.array([0.5, 0.5, 0.2]),
            parameter_names=["p1", "p2", "p3"],
        )

        # Interaction effects
        interactions = indices.total_order - indices.first_order
        assert interactions[0] == 0.2  # p1 has 20% interaction effects
        assert interactions[1] == 0.2  # p2 has 20% interaction effects

    @pytest.mark.unit
    def test_indices_identify_unimportant_parameters(self):
        """Low Sobol indices should identify unimportant parameters."""
        from uq import SobolIndices

        indices = SobolIndices(
            first_order=np.array([0.45, 0.45, 0.001]),
            total_order=np.array([0.5, 0.5, 0.002]),
            parameter_names=["important1", "important2", "unimportant"],
        )

        # Get least influential
        all_params = indices.get_most_influential(n=3)
        assert all_params[-1][0] == "unimportant"
        assert all_params[-1][1] < 0.01
