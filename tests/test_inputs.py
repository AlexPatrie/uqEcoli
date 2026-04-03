"""
Unit tests for UQ input parameter definitions.

Tests the input parameter classes that define scientifically relevant
variables for sensitivity analysis.
"""

import numpy as np
import pytest


class TestGenericSimDataParams:
    """Tests for generic sim_data parameter definitions."""

    @pytest.mark.unit
    def test_creation(self):
        """GenericSimDataParams should store parameter specs and values."""
        from libuq.pipeline.models import GenericSimDataParams, SimDataParameter

        specs = [
            SimDataParameter(
                name="kinetic_weight",
                attr_path="process.metabolism.kinetic_objective_weight",
                bounds=(0.0, 1.0),
            ),
        ]
        params = GenericSimDataParams(
            parameter_specs=specs,
            values={"kinetic_weight": 0.5},
        )
        assert len(params.values) == 1
        assert params.values["kinetic_weight"] == 0.5

    @pytest.mark.unit
    def test_from_sample(self):
        """XSpaceVecoli.sample_to_params should produce GenericSimDataParams."""
        from libuq.inputs import XSpaceVecoli
        from libuq.pipeline.models import GenericSimDataParams, SimDataParameter

        space = XSpaceVecoli(
            parameters=[
                SimDataParameter(name="param_a", attr_path="a.b", bounds=(0.0, 1.0)),
                SimDataParameter(name="param_b", attr_path="c.d", bounds=(0.0, 2.0)),
            ]
        )
        sample = np.array([0.5, 1.0])
        params = space.sample_to_params(sample)
        assert isinstance(params, GenericSimDataParams)
        assert params.values["param_a"] == 0.5
        assert params.values["param_b"] == 1.0

    @pytest.mark.unit
    def test_to_simulation_config(self):
        """to_simulation_config should produce a mutations dict."""
        from libuq.pipeline.models import GenericSimDataParams, SimDataParameter

        specs = [
            SimDataParameter(name="p1", attr_path="a.b.c", bounds=(0.0, 1.0)),
            SimDataParameter(name="p2", attr_path="x.y.z", bounds=(0.0, 2.0)),
        ]
        params = GenericSimDataParams(
            parameter_specs=specs,
            values={"p1": 0.3, "p2": 1.5},
        )
        config = params.to_simulation_config()
        assert config["sim_data_mutations"]["a.b.c"] == 0.3
        assert config["sim_data_mutations"]["x.y.z"] == 1.5


class TestInputParameterSpace:
    """Tests for InputParameterSpace used in sensitivity analysis."""

    @pytest.mark.unit
    def test_creation_with_parameters(self, input_parameter_space):
        """InputParameterSpace should include all configured parameters."""
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
    def test_subset_parameters(self):
        """InputParameterSpace works with a subset of parameters."""
        from libuq.inputs import XSpaceVecoli
        from libuq.pipeline.models import SimDataParameter

        space = XSpaceVecoli(
            parameters=[
                SimDataParameter(name="param_a", attr_path="a.b", bounds=(0.0, 1.0)),
                SimDataParameter(name="param_b", attr_path="c.d", bounds=(0.0, 2.0)),
            ]
        )
        assert space.n_parameters == 2
        assert space.parameter_names == ["param_a", "param_b"]

    @pytest.mark.unit
    def test_custom_bounds(self):
        """InputParameterSpace accepts custom bounds via SimDataParameter."""
        from libuq.inputs import XSpaceVecoli
        from libuq.pipeline.models import SimDataParameter

        space = XSpaceVecoli(
            parameters=[
                SimDataParameter(name="p1", attr_path="a.b", bounds=(1.0, 3.0)),
                SimDataParameter(name="p2", attr_path="c.d", bounds=(0.5, 1.5)),
                SimDataParameter(name="p3", attr_path="e.f", bounds=(0.0, 5.0)),
            ]
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
        """sample_to_params converts array to GenericSimDataParams."""
        from libuq.pipeline.models import GenericSimDataParams

        sample = np.array([2.0, 1.0, 5.0])
        params = input_parameter_space.sample_to_params(sample)

        assert isinstance(params, GenericSimDataParams)
        assert params.values["vio_expression"] == 2.0
        assert params.values["vio_trl_eff"] == 1.0
        assert params.values["mecillinam_concentration"] == 5.0
