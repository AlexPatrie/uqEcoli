"""
Unit tests for UQ input parameter definitions.

Tests the input parameter classes that define scientifically relevant
variables for sensitivity analysis.
"""

import numpy as np
import pytest


class TestVioPathwayParams:
    """Tests for violacein pathway parameter definitions."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        """VioPathwayParams should have sensible defaults."""
        from uq import VioPathwayParams

        params = VioPathwayParams()
        assert params.enabled is True
        assert params.expression >= 0

    @pytest.mark.unit
    def test_creation_with_custom_values(self, vio_params):
        """VioPathwayParams should accept custom values."""
        assert vio_params.expression == 2.5
        assert vio_params.translation_efficiency == 1.2
        assert vio_params.induction_gen == 1
        assert vio_params.condition == "basal"

    @pytest.mark.unit
    def test_expression_bounds(self):
        """Expression should be non-negative."""
        from uq import VioPathwayParams

        params = VioPathwayParams(expression=0.0)
        assert params.expression >= 0

    @pytest.mark.unit
    def test_disabled_state(self):
        """VioPathwayParams can be disabled."""
        from uq import VioPathwayParams

        params = VioPathwayParams(enabled=False)
        assert params.enabled is False


class TestMecillinamParams:
    """Tests for mecillinam antibiotic parameter definitions."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        """MecillinamParams should have sensible defaults."""
        from uq import MecillinamParams

        params = MecillinamParams()
        assert isinstance(params.times, list)
        assert isinstance(params.concentrations, list)

    @pytest.mark.unit
    def test_creation_with_custom_values(self, mecillinam_params):
        """MecillinamParams should accept custom values."""
        assert mecillinam_params.times == [0.0, 3600.0]
        assert mecillinam_params.concentrations == [0.0, 5.0]
        assert mecillinam_params.knockouts == ["murG"]

    @pytest.mark.unit
    def test_times_and_concentrations_match(self):
        """Times and concentrations should have matching lengths."""
        from uq import MecillinamParams

        params = MecillinamParams(
            times=[0.0, 100.0, 200.0],
            concentrations=[0.0, 1.0, 2.0],
        )
        assert len(params.times) == len(params.concentrations)


class TestGeneKnockoutParams:
    """Tests for gene knockout parameter definitions."""

    @pytest.mark.unit
    def test_creation_with_defaults(self):
        """GeneKnockoutParams should have empty defaults."""
        from uq import GeneKnockoutParams

        params = GeneKnockoutParams()
        assert isinstance(params.gene_deletions, list)
        assert isinstance(params.translation_knockouts, list)

    @pytest.mark.unit
    def test_creation_with_custom_values(self, knockout_params):
        """GeneKnockoutParams should accept custom values."""
        assert "lacZ" in knockout_params.gene_deletions
        assert "galK" in knockout_params.gene_deletions
        assert "murG" in knockout_params.translation_knockouts


class TestUQInputParameters:
    """Tests for the combined UQ input parameter container."""

    @pytest.mark.unit
    def test_creation_with_subparams(self, vio_params, mecillinam_params, knockout_params):
        """UQInputParameters should combine all parameter types."""
        from uq import UQInputParameters

        params = UQInputParameters(
            vio=vio_params,
            mecillinam=mecillinam_params,
            knockouts=knockout_params,
            seed=42,
            generations=8,
        )

        assert params.vio is vio_params
        assert params.mecillinam is mecillinam_params
        assert params.knockouts is knockout_params
        assert params.seed == 42
        assert params.generations == 8

    @pytest.mark.unit
    def test_default_subparams(self):
        """UQInputParameters should create defaults for subparams."""
        from uq import UQInputParameters

        params = UQInputParameters()
        assert params.vio is not None
        assert params.mecillinam is not None
        assert params.knockouts is not None


class TestInputParameterSpace:
    """Tests for InputParameterSpace used in sensitivity analysis."""

    @pytest.mark.unit
    def test_creation_with_vio_and_mecillinam(self, input_parameter_space):
        """InputParameterSpace should include both parameter sets."""
        assert input_parameter_space.n_parameters == 3
        names = input_parameter_space.parameter_names
        assert "vio_expression" in names
        assert "vio_trl_eff" in names
        assert "mecillinam_concentration" in names

    @pytest.mark.unit
    def test_bounds_are_valid(self, input_parameter_space):
        """Parameter bounds should be valid (lower < upper)."""
        for lb, ub in input_parameter_space.parameter_bounds:
            assert lb < ub, f"Invalid bounds: {lb} >= {ub}"

    @pytest.mark.unit
    def test_bounds_array(self, input_parameter_space):
        """bounds_array should return numpy array."""
        bounds = input_parameter_space.bounds_array
        assert isinstance(bounds, np.ndarray)
        assert bounds.shape == (3, 2)

    @pytest.mark.unit
    def test_vio_only(self):
        """InputParameterSpace can include only vio parameters."""
        from uq import InputParameterSpace

        space = InputParameterSpace(include_vio=True, include_mecillinam=False)
        assert space.n_parameters == 2
        assert all("vio" in name for name in space.parameter_names)

    @pytest.mark.unit
    def test_mecillinam_only(self):
        """InputParameterSpace can include only mecillinam parameters."""
        from uq import InputParameterSpace

        space = InputParameterSpace(include_vio=False, include_mecillinam=True)
        assert space.n_parameters == 1
        assert "mecillinam" in space.parameter_names[0]

    @pytest.mark.unit
    def test_custom_bounds(self):
        """InputParameterSpace accepts custom bounds."""
        from uq import InputParameterSpace

        space = InputParameterSpace(
            vio_expression_bounds=(1.0, 3.0),
            vio_trl_eff_bounds=(0.5, 1.5),
            mecillinam_conc_bounds=(0.0, 5.0),
        )

        bounds = space.parameter_bounds
        assert bounds[0] == (1.0, 3.0)
        assert bounds[1] == (0.5, 1.5)
        assert bounds[2] == (0.0, 5.0)

    @pytest.mark.unit
    def test_get_pytuq_bounds(self, input_parameter_space):
        """get_pytuq_bounds returns numpy arrays."""
        lb, ub = input_parameter_space.get_pytuq_bounds()

        assert isinstance(lb, np.ndarray)
        assert isinstance(ub, np.ndarray)
        assert len(lb) == input_parameter_space.n_parameters
        assert np.all(lb < ub)

    @pytest.mark.unit
    def test_sample_to_params_conversion(self, input_parameter_space):
        """sample_to_params converts array to UQInputParameters."""
        sample = np.array([2.0, 1.0, 5.0])

        params = input_parameter_space.sample_to_params(sample)

        assert params.vio.expression == 2.0
        assert params.vio.translation_efficiency == 1.0
