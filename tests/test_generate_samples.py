"""Tests for the generate_samples workflow via vEcoli's workflow.py.

These tests verify the full chain:
  LHS samples -> workflow config -> workflow.py -> Parquet -> PrecomputedCache
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

SIM_DATA_PATH = "/Users/alexanderpatrie/sms/vecoli_data/outputs/single/parca/kb/simData.cPickle"

# Skip all tests if sim_data not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(SIM_DATA_PATH),
    reason=f"simData.cPickle not found at {SIM_DATA_PATH}",
)


def test_build_variants_section_generic():
    """Test that _build_variants_section_generic produces correct variant config."""
    from libuq.generators.vecoli import _build_variants_section_generic
    from libuq.pipeline.models import SimDataParameter

    params = [
        SimDataParameter("p1", "process.transcription.fraction_active_rnap_free", (0.2, 0.5)),
        SimDataParameter("p2", "mass.cell_dry_mass_fraction", (0.25, 0.35)),
    ]
    X = np.array([[0.3, 0.28], [0.4, 0.32]])

    result = _build_variants_section_generic(X, params)

    assert "sim_data_setattr" in result
    mutations_list = result["sim_data_setattr"]["mutations"]["value"]
    assert len(mutations_list) == 2
    assert mutations_list[0]["process.transcription.fraction_active_rnap_free"] == 0.3
    assert mutations_list[0]["mass.cell_dry_mass_fraction"] == 0.28
    assert mutations_list[1]["process.transcription.fraction_active_rnap_free"] == 0.4


def test_build_workflow_config():
    """Test that _build_workflow_config produces valid vEcoli config."""
    from libuq.generators.vecoli import _build_workflow_config

    variants = {"sim_data_setattr": {"mutations": {"value": [{"a": 1}]}}}
    config = _build_workflow_config(
        sim_data_path="/tmp/kb/simData.cPickle",
        experiment_id="test_exp",
        output_dir="/tmp/output",
        max_duration=10.0,
        variants_section=variants,
    )

    assert config["sim_data_path"] == "/tmp/kb/simData.cPickle"
    assert config["experiment_id"] == "test_exp"
    assert config["emitter"] == "parquet"
    assert config["max_duration"] == 10.0
    assert config["n_init_sims"] == 1
    assert config["generations"] == 1
    assert config["suffix_time"] is False
    assert config["variants"] == variants
    assert config["emitter_arg"]["batch_size"] == 10


def test_sim_data_setattr_variant():
    """Test that the sim_data_setattr variant function works correctly."""
    from ecoli.library.sim_data import LoadSimData
    from ecoli.variants.sim_data_setattr import apply_variant

    sd = LoadSimData(SIM_DATA_PATH).sim_data
    original_val = sd.process.transcription.fraction_active_rnap_free

    params = {
        "mutations": {
            "process.transcription.fraction_active_rnap_free": 0.42,
        }
    }
    sd = apply_variant(sd, params)
    assert sd.process.transcription.fraction_active_rnap_free == 0.42
    assert sd.process.transcription.fraction_active_rnap_free != original_val


def test_xspace_generic_mode():
    """Test XSpaceVecoli in generic mode with SimDataParameter specs."""
    from libuq.inputs import XSpaceVecoli
    from libuq.pipeline.models import GenericSimDataParams, SimDataParameter

    params = [
        SimDataParameter("p1", "some.path", (0.0, 1.0)),
        SimDataParameter("p2", "other.path", (0.5, 1.5)),
    ]
    space = XSpaceVecoli(parameters=params)

    assert space.n_parameters == 2
    assert space.parameter_names == ["p1", "p2"]
    assert space.n_parameters == 2  # all generic now

    sample = np.array([0.3, 0.8])
    result = space.sample_to_params(sample)
    assert isinstance(result, GenericSimDataParams)
    assert result.values["p1"] == 0.3
    assert result.values["p2"] == 0.8

    config = result.to_simulation_config()
    assert "sim_data_mutations" in config
    assert config["sim_data_mutations"]["some.path"] == 0.3
    assert config["sim_data_mutations"]["other.path"] == 0.8


def test_param_loader_validates_paths():
    """Test that ParameterDataset validates sim_data attribute paths."""
    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import ParameterDataset

    ds = ParameterDataset(sim_data_path=SIM_DATA_PATH)

    # Valid path should work
    valid = [SimDataParameter("test", "process.transcription.fraction_active_rnap_free", (0.1, 0.5))]
    space = ds.to_parameter_space(parameters=valid)
    assert space.n_parameters == 1

    # Invalid path should raise
    invalid = [SimDataParameter("bad", "process.nonexistent.attribute", (0.0, 1.0))]
    with pytest.raises(ValueError, match="not found on sim_data"):
        ds.to_parameter_space(parameters=invalid)


@pytest.mark.slow
def test_run_batch_workflow(tmp_path):
    """End-to-end test: LHS samples -> workflow.py -> Parquet -> cache.

    This test runs actual vEcoli simulations (takes ~60s).
    """
    from libuq.generators.vecoli import TimeseriesGeneratorVecoli
    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import ParameterDataset

    ds = ParameterDataset(sim_data_path=SIM_DATA_PATH)
    params = [
        SimDataParameter(
            "rnap_free",
            "process.transcription.fraction_active_rnap_free",
            (0.30, 0.42),
        ),
    ]
    space = ds.to_parameter_space(parameters=params)

    sim_func = TimeseriesGeneratorVecoli(
        baseline_sim_data=ds.sim_data,
        param_space=space,
        max_duration=22.0,
    )

    X = np.array([[0.33], [0.39]])
    Y_agg, Y_ts, _Y_meta = sim_func._run_batch(X, batch_dir=tmp_path / "batch")

    assert Y_agg.shape == (2, 4)  # 2 samples, 4 default observables
    assert Y_ts is not None
    assert len(Y_ts) == 2
    assert all(ts.shape[0] > 0 for ts in Y_ts)
    assert all(ts.shape[1] == 4 for ts in Y_ts)


@pytest.mark.slow
def test_generate_samples_handler(tmp_path):
    """Test the full handlers.generate_samples() function."""
    from libuq.handlers import generate_samples

    cache = generate_samples(
        experiment_ids=["single"],
        sim_base_path="/Users/alexanderpatrie/sms/vecoli_data/outputs",
        cache_dir=str(tmp_path / "cache"),
        n_samples=2,
        seed=42,
        max_duration=22.0,
        live=True,
        params_file="examples/uq_artifacts/params/params_demo.json",
    )

    assert cache.X.shape[0] == 2
    assert cache.Y.shape[0] == 2
    assert cache.Y.shape[1] == 4
    assert len(cache.parameter_names) == 3
    assert cache.Y_timeseries is not None
