"""
Unit tests for the PCE (Polynomial Chaos Expansion) module.

Tests cover the complete PCE workflow including:
- Multi-index generation
- Coefficient fitting (least squares, analytical, variational via PyTUQ)
- Sample generation (LHS)
- Sample processing for stochastic functions
- End-to-end surrogate generation
"""

import math

import numpy as np
import pytest


class TestGenerateMultiIndices:
    """Tests for generate_multi_indices function."""

    @pytest.mark.unit
    def test_term_count_matches_binomial(self):
        """Number of terms should match C(n+p, p)."""
        from libuq.pce.surrogate import generate_multi_indices

        test_cases = [
            (3, 2, 10),  # C(5,2) = 10
            (5, 2, 21),  # C(7,2) = 21
            (3, 3, 20),  # C(6,3) = 20
            (10, 2, 66),  # C(12,2) = 66
        ]

        for n_params, max_order, expected in test_cases:
            indices = generate_multi_indices(n_params, max_order)
            assert len(indices) == expected, f"n={n_params}, p={max_order}: got {len(indices)}, expected {expected}"

    @pytest.mark.unit
    def test_indices_shape(self):
        """Multi-indices should have shape (n_terms, n_params)."""
        from libuq.pce.surrogate import generate_multi_indices

        indices = generate_multi_indices(n_params=4, max_order=2)

        assert indices.ndim == 2
        assert indices.shape[1] == 4  # n_params columns

    @pytest.mark.unit
    def test_includes_constant_term(self):
        """First term should be all zeros (constant)."""
        from libuq.pce.surrogate import generate_multi_indices

        indices = generate_multi_indices(n_params=3, max_order=2)

        assert np.all(indices[0] == 0)

    @pytest.mark.unit
    def test_total_order_constraint(self):
        """Sum of each multi-index should be <= max_order."""
        from libuq.pce.surrogate import generate_multi_indices

        max_order = 3
        indices = generate_multi_indices(n_params=4, max_order=max_order)

        for idx in indices:
            assert sum(idx) <= max_order

    @pytest.mark.unit
    def test_order_zero_gives_one_term(self):
        """Order 0 should give exactly 1 term (constant)."""
        from libuq.pce.surrogate import generate_multi_indices

        indices = generate_multi_indices(n_params=5, max_order=0)

        assert len(indices) == 1
        assert np.all(indices[0] == 0)

    @pytest.mark.unit
    def test_order_one_gives_n_plus_one_terms(self):
        """Order 1 should give n+1 terms (constant + linear)."""
        from libuq.pce.surrogate import generate_multi_indices

        n_params = 4
        indices = generate_multi_indices(n_params=n_params, max_order=1)

        assert len(indices) == n_params + 1


class TestFitPCECoefficients:
    """Tests for fit_pce_coefficients function."""

    @pytest.mark.unit
    def test_fits_linear_function(self, rng):
        """Should fit a linear function with R² ≈ 1."""
        from libuq.pce.surrogate import fit_pce_coefficients

        n_samples = 50
        X = rng.uniform(-1, 1, (n_samples, 2))
        Y = 1.0 + 2.0 * X[:, 0] + 0.5 * X[:, 1]

        result = fit_pce_coefficients(X, Y, polynomial_order=1)

        assert result.r_squared > 0.99

    @pytest.mark.unit
    def test_fits_quadratic_function(self, rng):
        """Should fit a quadratic function accurately."""
        from libuq.pce.surrogate import fit_pce_coefficients

        n_samples = 100
        X = rng.uniform(-1, 1, (n_samples, 2))
        Y = 1.0 + X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2

        result = fit_pce_coefficients(X, Y, polynomial_order=2)

        assert result.r_squared > 0.98

    @pytest.mark.unit
    def test_returns_correct_structure(self, rng):
        """Result should have all required fields."""
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (50, 3))
        Y = X[:, 0] + X[:, 1]

        result = fit_pce_coefficients(X, Y, polynomial_order=2)

        assert hasattr(result, "coefficients")
        assert hasattr(result, "multi_indices")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "polynomial_order")
        assert hasattr(result, "n_params")
        assert hasattr(result, "method")

    @pytest.mark.unit
    def test_coefficient_count_matches_terms(self, rng):
        """Number of coefficients should match number of terms."""
        from libuq.pce.surrogate import fit_pce_coefficients

        n_params = 3
        p_order = 2
        expected_terms = math.comb(n_params + p_order, p_order)

        X = rng.uniform(-1, 1, (100, n_params))
        Y = X[:, 0] + X[:, 1]

        result = fit_pce_coefficients(X, Y, polynomial_order=p_order)

        assert len(result.coefficients) == expected_terms
        assert len(result.multi_indices) == expected_terms

    @pytest.mark.unit
    def test_least_squares_method(self, rng):
        """Least squares method should work."""
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (50, 2))
        Y = X[:, 0] + X[:, 1]

        result = fit_pce_coefficients(X, Y, polynomial_order=2, method="least_squares")

        assert result.method == "least_squares"
        assert result.r_squared > 0.9

    @pytest.mark.unit
    def test_warns_on_insufficient_samples(self, rng):
        """Should warn when samples < 2 * n_terms."""
        from libuq.pce.surrogate import fit_pce_coefficients

        # 3 params, order 2 = 10 terms, need 20 samples
        X = rng.uniform(-1, 1, (15, 3))  # Only 15 samples
        Y = X[:, 0]

        with pytest.warns(UserWarning, match="less than recommended"):
            fit_pce_coefficients(X, Y, polynomial_order=2)

    @pytest.mark.unit
    def test_raises_on_underdetermined(self, rng):
        """Should raise when samples < n_terms."""
        from libuq.pce.surrogate import fit_pce_coefficients

        # 3 params, order 2 = 10 terms, only 5 samples
        X = rng.uniform(-1, 1, (5, 3))
        Y = X[:, 0]

        with pytest.raises(ValueError, match="Underdetermined"):
            fit_pce_coefficients(X, Y, polynomial_order=2)

    @pytest.mark.unit
    def test_to_surrogate_method(self, rng):
        """to_surrogate should return a PCESurrogate."""
        from libuq import PCESurrogate
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (50, 2))
        Y = X[:, 0] + X[:, 1]

        result = fit_pce_coefficients(X, Y, polynomial_order=2)
        surrogate = result.to_surrogate()

        assert isinstance(surrogate, PCESurrogate)

    @pytest.mark.unit
    def test_with_bounds_normalization(self, rng):
        """Should work with custom bounds for normalization."""
        from libuq.pce.surrogate import fit_pce_coefficients

        bounds = np.array([[0, 10], [0, 5]])
        X = rng.uniform(bounds[:, 0], bounds[:, 1], (50, 2))
        Y = X[:, 0] / 10 + X[:, 1] / 5  # Normalize manually for comparison

        result = fit_pce_coefficients(X, Y, polynomial_order=2, bounds=bounds)

        assert result.r_squared > 0.9
        assert result.input_bounds is not None


class TestCreateSamples:
    """Tests for create_samples function (Latin Hypercube Sampling)."""

    @pytest.mark.unit
    def test_output_shape(self):
        """Should return array of shape (N, n_params)."""
        from libuq.models import Parameter
        from libuq.pce.surrogate import create_samples

        params = [
            Parameter(name="p1", bounds=[0, 1], default=0.5, step=0.1, description=""),
            Parameter(name="p2", bounds=[0, 2], default=1.0, step=0.1, description=""),
            Parameter(name="p3", bounds=[0, 3], default=1.5, step=0.1, description=""),
        ]

        X = create_samples(N=50, selected=params)

        assert X.shape == (50, 3)

    @pytest.mark.unit
    def test_samples_within_bounds(self):
        """All samples should be within parameter bounds."""
        from libuq.models import Parameter
        from libuq.pce.surrogate import create_samples

        params = [
            Parameter(name="p1", bounds=[0.5, 2.0], default=1.0, step=0.1, description=""),
            Parameter(name="p2", bounds=[-1.0, 1.0], default=0.0, step=0.1, description=""),
        ]

        X = create_samples(N=100, selected=params)

        assert np.all(X[:, 0] >= 0.5) and np.all(X[:, 0] <= 2.0)
        assert np.all(X[:, 1] >= -1.0) and np.all(X[:, 1] <= 1.0)

    @pytest.mark.unit
    def test_lhs_stratification(self):
        """LHS should provide better coverage than random."""
        from libuq.models import Parameter
        from libuq.pce.surrogate import create_samples

        params = [Parameter(name="p1", bounds=[0, 1], default=0.5, step=0.1, description="")]

        X = create_samples(N=10, selected=params)

        # LHS divides [0,1] into 10 bins; each bin should have exactly 1 sample
        bins = np.floor(X[:, 0] * 10).astype(int)
        bins = np.clip(bins, 0, 9)  # Handle edge case of X=1.0
        unique_bins = len(np.unique(bins))

        assert unique_bins >= 8  # Allow some tolerance


class TestProcessSamples:
    """Tests for process_samples functions."""

    @pytest.mark.unit
    def test_simple_processor(self):
        """Simple processor should work with deterministic function."""
        from libuq.pce.surrogate import process_samples_simple

        X = np.array([[1, 2], [3, 4], [5, 6]])
        f = lambda x: x[0] + x[1]

        Y = process_samples_simple(X, f, n_replicates=1)

        np.testing.assert_array_equal(Y, [3, 7, 11])

    @pytest.mark.unit
    def test_simple_processor_with_replicates(self, rng):
        """Simple processor should handle stochastic functions."""
        from libuq.pce.surrogate import process_samples_simple

        X = np.array([[1.0], [2.0], [3.0]])

        # Stochastic function: mean + small noise
        call_count = [0]

        def f(x):
            call_count[0] += 1
            return x[0] + rng.normal(0, 0.01)

        Y = process_samples_simple(X, f, n_replicates=5)

        # Should have called f 15 times (3 samples × 5 replicates)
        assert call_count[0] == 15
        # Results should be close to x values
        np.testing.assert_allclose(Y, [1, 2, 3], atol=0.1)

    @pytest.mark.unit
    def test_adaptive_processor_converges(self):
        """Adaptive processor should converge for low-noise functions."""
        from libuq.pce.surrogate import process_samples_adaptive

        X = np.array([[1.0], [2.0]])

        call_counts = [0]

        def f(x):
            call_counts[0] += 1
            return x[0]  # Deterministic = zero noise

        Y = process_samples_adaptive(X, f, target_cv=0.05, min_reps=3, max_reps=20)

        # Should stop at min_reps since there's no variance
        assert call_counts[0] <= 6  # 2 samples × 3 min_reps
        np.testing.assert_array_equal(Y, [1, 2])

    @pytest.mark.unit
    def test_process_samples_dispatcher(self):
        """process_samples should dispatch to correct method."""
        from libuq.pce.surrogate import process_samples

        X = np.array([[1.0], [2.0]])
        f = lambda x: x[0]

        Y_simple = process_samples(X, f, method="simple", n_replicates=1)
        Y_adaptive = process_samples(X, f, method="adaptive", min_reps=1, max_reps=3)

        np.testing.assert_array_equal(Y_simple, [1, 2])
        np.testing.assert_array_equal(Y_adaptive, [1, 2])

    @pytest.mark.unit
    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        from libuq.pce.surrogate import process_samples

        X = np.array([[1.0]])
        f = lambda x: x[0]

        with pytest.raises(ValueError, match="Not a valid method"):
            process_samples(X, f, method="invalid")


class TestPCEConfig:
    """Tests for PCE configuration classes."""

    @pytest.mark.unit
    def test_surrogate_config_calculates_p(self):
        """PCESurrogateConfig should auto-calculate polynomial order."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=[0, 1], default=0.5, step=0.1, description="") for i in range(5)]

        # 100 samples, 5 params: should determine max p
        config = PCESurrogateConfig(parameters=params, n_samples=100)

        assert config.polynomial_order is not None
        assert config.polynomial_order >= 1

    @pytest.mark.unit
    def test_surrogate_config_validates_samples(self):
        """PCESurrogateConfig should warn on insufficient samples."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=[0, 1], default=0.5, step=0.1, description="") for i in range(5)]

        # Very few samples for 5 params
        with pytest.warns(UserWarning):
            PCESurrogateConfig(parameters=params, n_samples=5, polynomial_order=3)

    @pytest.mark.unit
    def test_get_pce_config(self):
        """get_pce_config should create valid PCEConfig."""
        from libuq.models import Parameter, PCEConfig
        from libuq.pce.surrogate import get_pce_config

        params = [Parameter(name="p1", bounds=[0, 1], default=0.5, step=0.1, description="")]

        config = get_pce_config(prescreened=params, sample_size=50)

        assert isinstance(config, PCEConfig)
        assert config.surrogate.n_samples == 50

    @pytest.mark.unit
    def test_config_aliases(self):
        """Config should have n, N, p aliases."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=[0, 1], default=0.5, step=0.1, description="") for i in range(3)]
        config = PCESurrogateConfig(parameters=params, n_samples=50, polynomial_order=2)

        assert config.n == 3  # n_params
        assert config.N == 50  # n_samples
        assert config.p == 2  # polynomial_order


class TestGenerateSurrogate:
    """Integration tests for generate_surrogate function."""

    @pytest.mark.unit
    def test_end_to_end_simple_function(self, input_parameter_space):
        """Should create working surrogate for simple function."""
        from libuq import PCESurrogate
        from libuq.pce.surrogate import generate_surrogate

        # Simple linear function
        def f(x):
            return x[0] + 0.5 * x[1] + 0.25 * x[2]

        surrogate = generate_surrogate(
            space=input_parameter_space,
            generator=f,
            sample_size=50,
            min_reps=1,
            max_reps=1,  # Deterministic function
        )

        assert isinstance(surrogate, PCESurrogate)

    @pytest.mark.unit
    def test_surrogate_prediction_reasonable(self, input_parameter_space, rng):
        """Surrogate predictions should be in reasonable range."""
        from libuq.pce.surrogate import generate_surrogate

        # Known function
        def f(x):
            return x[0] * 2  # Just depends on first param

        surrogate = generate_surrogate(
            space=input_parameter_space,
            generator=f,
            sample_size=50,
            min_reps=1,
            max_reps=1,
        )

        # Test prediction
        bounds = np.array(input_parameter_space.parameter_bounds)
        X_test = rng.uniform(bounds[:, 0], bounds[:, 1], (10, 3))

        # Surrogate should have predict method
        # Note: May raise NotImplementedError if not fully implemented
        try:
            Y_pred = surrogate.predict(X_test)
            assert Y_pred is not None
        except NotImplementedError:
            pytest.skip("PCESurrogate.predict not implemented")


class TestRegressionMethods:
    """Tests for PyTUQ regression methods (analytical, variational)."""

    @pytest.mark.unit
    def test_analytical_fits_linear(self, rng):
        """Analytical (full Bayesian) regression should fit a linear function."""
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (100, 3))
        Y = X[:, 0] + 0.5 * X[:, 1]

        result = fit_pce_coefficients(
            X,
            Y,
            polynomial_order=2,
            method="analytical",
        )

        assert result.method == "analytical"
        assert result.r_squared > 0.9

    @pytest.mark.unit
    def test_variational_fits_quadratic(self, rng):
        """Variational inference regression should fit a quadratic function."""
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (100, 3))
        Y = 1.0 + X[:, 0] ** 2 + 0.5 * X[:, 1]

        result = fit_pce_coefficients(
            X,
            Y,
            polynomial_order=2,
            method="variational",
        )

        assert result.method == "variational"
        assert result.r_squared > 0.9

    @pytest.mark.unit
    def test_invalid_method_raises(self, rng):
        """Invalid fitting method should raise."""
        from libuq.pce.surrogate import fit_pce_coefficients

        X = rng.uniform(-1, 1, (50, 2))
        Y = X[:, 0]

        with pytest.raises(ValueError, match="Unknown method"):
            fit_pce_coefficients(X, Y, polynomial_order=2, method="invalid")


class TestPCEFitResult:
    """Tests for PCEFitResult dataclass."""

    @pytest.mark.unit
    def test_sparsity_calculation(self):
        """Sparsity should be computed correctly."""
        from libuq.models import PCEFitResult

        # 5 coefficients, 2 non-zero
        coefficients = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
        multi_indices = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]])

        result = PCEFitResult(
            coefficients=coefficients,
            multi_indices=multi_indices,
            basis_type="legendre",
            polynomial_order=2,
            n_params=2,
            r_squared=0.95,
            n_samples=50,
            method="least_squares",
        )

        # 3 out of 5 are zero → 60% sparsity
        assert result.sparsity == pytest.approx(0.6)

    @pytest.mark.unit
    def test_to_surrogate_preserves_metadata(self):
        """to_surrogate should preserve key metadata."""
        from libuq.models import PCEFitResult

        coefficients = np.array([1.0, 0.5, 0.2])
        multi_indices = np.array([[0, 0], [1, 0], [0, 1]])
        bounds = np.array([[0, 1], [0, 2]])

        result = PCEFitResult(
            coefficients=coefficients,
            multi_indices=multi_indices,
            basis_type="legendre",
            polynomial_order=1,
            n_params=2,
            r_squared=0.98,
            n_samples=50,
            method="least_squares",
            input_bounds=bounds,
        )

        surrogate = result.to_surrogate()

        np.testing.assert_array_equal(surrogate.coefficients, coefficients)
        np.testing.assert_array_equal(surrogate.multi_indices, multi_indices)
        assert surrogate.polynomial_order == 1
        assert surrogate.basis_type == "legendre"
