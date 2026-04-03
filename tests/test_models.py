"""
Unit tests for UQ data models.

Tests cover all dataclasses in uq/models.py including:
- Parameter and related simulation config models
- PCE configuration classes
- PCEFitResult
"""

import math
import warnings

import numpy as np
import pytest


class TestParameter:
    """Tests for Parameter dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        """Parameter should store all attributes."""
        from libuq.models import Parameter

        param = Parameter(
            name="temperature",
            bounds=(200, 400),
            default=300,
            step=10,
            description="Temperature in Kelvin",
        )

        assert param.name == "temperature"
        assert param.bounds == (200, 400)
        assert param.default == 300
        assert param.step == 10

    @pytest.mark.unit
    def test_model_dump(self):
        """model_dump should return dict with tuple bounds."""
        from libuq.models import Parameter

        param = Parameter(
            name="pressure",
            bounds=[1, 10],  # list input
            default=5,
            step=0.5,
            description="Pressure",
        )

        dump = param.model_dump()

        assert isinstance(dump, dict)
        assert dump["name"] == "pressure"
        assert isinstance(dump["bounds"], tuple)  # Should be converted to tuple


class TestPCEParameterSelectionConfig:
    """Tests for PCEParameterSelectionConfig."""

    @pytest.mark.unit
    def test_default_values(self):
        """Should have sensible defaults."""
        from libuq.models import PCEParameterSelectionConfig

        config = PCEParameterSelectionConfig()

        assert config.n_trajectories == 20
        assert config.n_top == 5

    @pytest.mark.unit
    def test_custom_values(self):
        """Should accept custom values."""
        from libuq.models import PCEParameterSelectionConfig

        config = PCEParameterSelectionConfig(n_trajectories=30, n_top=10)

        assert config.n_trajectories == 30
        assert config.n_top == 10


class TestPCEPreprocessingConfig:
    """Tests for PCEPreprocessingConfig."""

    @pytest.mark.unit
    def test_default_values(self):
        """Should have sensible defaults."""
        from libuq.models import PCEPreprocessingConfig

        config = PCEPreprocessingConfig()

        assert config.target_cv == 0.05
        assert config.min_reps == 3
        assert config.max_reps == 20

    @pytest.mark.unit
    def test_custom_values(self):
        """Should accept custom values."""
        from libuq.models import PCEPreprocessingConfig

        config = PCEPreprocessingConfig(target_cv=0.1, min_reps=5, max_reps=50)

        assert config.target_cv == 0.1
        assert config.min_reps == 5
        assert config.max_reps == 50


class TestPCESolverConfig:
    """Tests for PCESolverConfig."""

    @pytest.mark.unit
    def test_default_values(self):
        """Should have sensible defaults."""
        from libuq.models import PCESolverConfig

        config = PCESolverConfig()

        assert config.basis_type == "legendre"
        assert config.method == "least_squares"

    @pytest.mark.unit
    def test_analytical_config(self):
        """Should accept analytical (full Bayesian) configuration."""
        from libuq.models import PCESolverConfig

        config = PCESolverConfig(method="analytical")

        assert config.method == "analytical"

    @pytest.mark.unit
    def test_variational_config(self):
        """Should accept variational inference configuration."""
        from libuq.models import PCESolverConfig

        config = PCESolverConfig(method="variational")

        assert config.method == "variational"

    @pytest.mark.unit
    def test_hermite_basis(self):
        """Should accept Hermite basis."""
        from libuq.models import PCESolverConfig

        config = PCESolverConfig(basis_type="hermite")

        assert config.basis_type == "hermite"


class TestPCESurrogateConfig:
    """Tests for PCESurrogateConfig."""

    @pytest.mark.unit
    def test_creation_with_parameters(self):
        """Should store parameters and sample size."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [
            Parameter(name="p1", bounds=(0, 1), default=0.5, step=0.1, description="Param 1"),
            Parameter(name="p2", bounds=(0, 2), default=1.0, step=0.2, description="Param 2"),
        ]

        config = PCESurrogateConfig(parameters=params, n_samples=100, polynomial_order=2)

        assert len(config.parameters) == 2
        assert config.n_samples == 100
        assert config.polynomial_order == 2

    @pytest.mark.unit
    def test_convenience_aliases(self):
        """n, N, p should be aliases."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=(0, 1), default=0.5, step=0.1, description="") for i in range(5)]

        config = PCESurrogateConfig(parameters=params, n_samples=100, polynomial_order=3)

        assert config.n == 5  # len(parameters)
        assert config.N == 100  # n_samples
        assert config.p == 3  # polynomial_order

    @pytest.mark.unit
    def test_auto_calculate_polynomial_order(self):
        """Should auto-calculate polynomial order from sample budget."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=(0, 1), default=0.5, step=0.1, description="") for i in range(3)]

        # With 100 samples and 3 params, should calculate a reasonable order
        config = PCESurrogateConfig(parameters=params, n_samples=100)

        assert config.polynomial_order is not None
        assert config.polynomial_order >= 1

    @pytest.mark.unit
    def test_warns_on_insufficient_samples(self):
        """Should warn when samples are insufficient."""
        from libuq.models import Parameter, PCESurrogateConfig

        params = [Parameter(name=f"p{i}", bounds=(0, 1), default=0.5, step=0.1, description="") for i in range(5)]

        # Very few samples for 5 params with order 3
        with pytest.warns(UserWarning):
            PCESurrogateConfig(parameters=params, n_samples=10, polynomial_order=3)


class TestPCEConfig:
    """Tests for PCEConfig (nested configuration)."""

    @pytest.mark.unit
    def test_creation_with_all_subconfigs(self):
        """Should combine all sub-configurations."""
        from libuq.models import (
            Parameter,
            PCEConfig,
            PCEParameterSelectionConfig,
            PCEPreprocessingConfig,
            PCESolverConfig,
            PCESurrogateConfig,
        )

        params = [Parameter(name="p1", bounds=(0, 1), default=0.5, step=0.1, description="")]

        config = PCEConfig(
            selection=PCEParameterSelectionConfig(n_trajectories=15),
            preprocessing=PCEPreprocessingConfig(target_cv=0.03),
            solver=PCESolverConfig(method="lasso"),
            surrogate=PCESurrogateConfig(parameters=params, n_samples=50),
        )

        assert config.selection.n_trajectories == 15
        assert config.preprocessing.target_cv == 0.03
        assert config.solver.method == "lasso"
        assert config.surrogate.n_samples == 50


class TestPCEFitResult:
    """Tests for PCEFitResult dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        """Should store all fit results."""
        from libuq.models import PCEFitResult

        coeffs = np.array([1.0, 0.5, 0.2])
        indices = np.array([[0, 0], [1, 0], [0, 1]])

        result = PCEFitResult(
            coefficients=coeffs,
            multi_indices=indices,
            basis_type="legendre",
            polynomial_order=1,
            n_params=2,
            r_squared=0.95,
            n_samples=50,
            method="least_squares",
        )

        np.testing.assert_array_equal(result.coefficients, coeffs)
        assert result.r_squared == 0.95
        assert result.method == "least_squares"

    @pytest.mark.unit
    def test_sparsity_computed(self):
        """Sparsity should be computed automatically."""
        from libuq.models import PCEFitResult

        # 3 coefficients, 1 is zero
        coeffs = np.array([1.0, 0.0, 0.5])
        indices = np.array([[0], [1], [2]])

        result = PCEFitResult(
            coefficients=coeffs,
            multi_indices=indices,
            basis_type="legendre",
            polynomial_order=2,
            n_params=1,
            r_squared=0.9,
            n_samples=50,
            method="least_squares",
        )

        # 1 out of 3 is zero → ~33% sparsity
        assert result.sparsity == pytest.approx(1 / 3)

    @pytest.mark.unit
    def test_sparsity_all_nonzero(self):
        """Sparsity should be 0 when all coefficients are non-zero."""
        from libuq.models import PCEFitResult

        coeffs = np.array([1.0, 0.5, 0.2, 0.1])
        indices = np.array([[0], [1], [2], [3]])

        result = PCEFitResult(
            coefficients=coeffs,
            multi_indices=indices,
            basis_type="legendre",
            polynomial_order=3,
            n_params=1,
            r_squared=0.99,
            n_samples=100,
            method="least_squares",
        )

        assert result.sparsity == 0.0

    @pytest.mark.unit
    def test_sparsity_all_zero(self):
        """Sparsity should be 1 when all coefficients are zero."""
        from libuq.models import PCEFitResult

        coeffs = np.array([0.0, 0.0, 0.0])
        indices = np.array([[0], [1], [2]])

        result = PCEFitResult(
            coefficients=coeffs,
            multi_indices=indices,
            basis_type="legendre",
            polynomial_order=2,
            n_params=1,
            r_squared=0.0,
            n_samples=50,
            method="least_squares",
        )

        assert result.sparsity == 1.0

    @pytest.mark.unit
    def test_to_surrogate(self):
        """to_surrogate should return valid PCESurrogate."""
        from libuq import PCESurrogate
        from libuq.models import PCEFitResult

        coeffs = np.array([1.0, 0.5])
        indices = np.array([[0, 0], [1, 0]])
        bounds = np.array([[0, 1], [0, 2]])

        result = PCEFitResult(
            coefficients=coeffs,
            multi_indices=indices,
            basis_type="legendre",
            polynomial_order=1,
            n_params=2,
            r_squared=0.98,
            n_samples=50,
            method="least_squares",
            input_bounds=bounds,
        )

        surrogate = result.to_surrogate()

        assert isinstance(surrogate, PCESurrogate)
        np.testing.assert_array_equal(surrogate.coefficients, coeffs)
        assert surrogate.polynomial_order == 1
        assert surrogate.input_dim == 2

    @pytest.mark.unit
    def test_to_surrogate_preserves_bounds(self):
        """to_surrogate should preserve input bounds."""
        from libuq.models import PCEFitResult

        bounds = np.array([[0, 10], [5, 15]])

        result = PCEFitResult(
            coefficients=np.array([1.0]),
            multi_indices=np.array([[0, 0]]),
            basis_type="legendre",
            polynomial_order=0,
            n_params=2,
            r_squared=1.0,
            n_samples=10,
            method="least_squares",
            input_bounds=bounds,
        )

        surrogate = result.to_surrogate()

        np.testing.assert_array_equal(surrogate.input_bounds, bounds)


class TestCellCycleVariable:
    """Tests for CellCycleVariable dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        """Should store values and metadata."""
        from libuq.models import CellCycleVariable

        values = np.linspace(0, 1, 100)

        ccv = CellCycleVariable(
            values=values,
            normalized=True,
            variable_name="test_variable",
        )

        np.testing.assert_array_equal(ccv.values, values)
        assert ccv.normalized is True
        assert ccv.variable_name == "test_variable"

    @pytest.mark.unit
    def test_to_stage_bins(self):
        """to_stage_bins should discretize correctly."""
        from libuq.models import CellCycleVariable

        values = np.array([0.05, 0.15, 0.25, 0.95])

        ccv = CellCycleVariable(values=values, normalized=True)
        bins = ccv.to_stage_bins(n_bins=10)

        # 0.05 → bin 0, 0.15 → bin 1, 0.25 → bin 2, 0.95 → bin 9
        assert bins[0] == 0
        assert bins[1] == 1
        assert bins[2] == 2
        assert bins[3] == 9

    @pytest.mark.unit
    def test_phase_labels_optional(self):
        """phase_labels should be optional."""
        from libuq.models import CellCycleVariable

        ccv = CellCycleVariable(values=np.array([0.5]))

        assert ccv.phase_labels is None


class TestMediaCondition:
    """Tests for MediaCondition enum."""

    @pytest.mark.unit
    def test_all_conditions_exist(self):
        """All media conditions should be defined."""
        from libuq.models import MediaCondition

        assert MediaCondition.BASAL.value == "basal"
        assert MediaCondition.WITH_AA.value == "with_aa"
        assert MediaCondition.ACETATE.value == "acetate"
        assert MediaCondition.SUCCINATE.value == "succinate"
        assert MediaCondition.NO_OXYGEN.value == "no_oxygen"


class TestCellCyclePhase:
    """Tests for CellCyclePhase enum."""

    @pytest.mark.unit
    def test_all_phases_exist(self):
        """All cell cycle phases should be defined."""
        from libuq.models import CellCyclePhase

        assert CellCyclePhase.B_PERIOD.value == "B_period"
        assert CellCyclePhase.C_PERIOD.value == "C_period"
        assert CellCyclePhase.D_PERIOD.value == "D_period"
        assert CellCyclePhase.UNKNOWN.value == "unknown"
