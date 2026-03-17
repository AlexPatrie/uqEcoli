"""
vEcoli simulation function for UQ pipeline: f(x) -> y.

This module provides a parameter-agnostic simulation wrapper that:
1. Accepts x as a numpy array positionally aligned with XSpace parameters
2. Converts x -> UQInputParametersVecoli -> variant param dicts via XSpace
3. Applies vEcoli variant functions (apply_variant) to a deep-copied sim_data
4. Runs a single-cell EcoliSim with the timeseries emitter
5. Extracts outputs (y) matching the format expected by the ./uq workflow

The simulation function does NOT know which parameters are being varied --
that is controlled by the XSpace / parameter space layer. Morris prescreening
can reduce the parameter set between phases without changing f.

Variant application uses the real vEcoli apply_variant() functions, which
handle complex multi-step sim_data mutations (e.g., new gene expression
adjustment, internal_shift_dict setup, condition/timeline changes) that
cannot be reduced to simple setattr calls.
"""

import copy
import importlib
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from ecoli.experiments.ecoli_master_sim import EcoliSim
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from uq.common import BaseClass
from uq.io import get_bucket
from uq.pipeline.models import SimulationConfig


# -- Helpers ----------------------------------------------------------------


def _get_nested_dict(d: dict, keys: list[str]) -> Any:
    """Traverse a nested dict by a list of keys."""
    for k in keys:
        d = d[k]
    return d


def _fix_new_gene_rel_adj(
    sim_data: SimulationDataEcoli,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Broadcast rel_adj lists to match the actual new gene counts in sim_data.

    The ``new_gene_internal_shift_variable_strength`` variant asserts::

        len(new_gene_rna_indices) == len(rel_exp_adj_list)
        len(new_monomer_indices)  == len(rel_trl_eff_adj_list)

    ``VioPathwayParams`` defaults both to ``[1.0]``, but violacein sim_data
    may have N monomers in a single operon (1 RNA, N monomers).  This
    helper broadcasts short lists to the correct length.
    """
    if "rel_adj" not in params:
        return params

    from ecoli.variants.new_gene_internal_shift_variable_strength import (
        get_new_gene_ids_and_indices,
    )

    _, rna_indices, _, monomer_indices = get_new_gene_ids_and_indices(sim_data)
    n_rna = len(rna_indices)
    n_mono = len(monomer_indices)

    rel_adj = params["rel_adj"]
    exp_list = rel_adj.get("rel_exp_adj_list", [1.0])
    trl_list = rel_adj.get("rel_trl_eff_adj_list", [1.0])

    # Broadcast length-1 lists to match actual gene counts
    if len(exp_list) == 1 and n_rna > 1:
        exp_list = exp_list * n_rna
    if len(trl_list) == 1 and n_mono > 1:
        trl_list = trl_list * n_mono

    params = {**params, "rel_adj": {
        "rel_exp_adj_list": exp_list,
        "rel_trl_eff_adj_list": trl_list,
    }}
    return params


def _apply_variants(
    sim_data: SimulationDataEcoli,
    variants_config: dict[str, list[dict[str, Any]]],
) -> SimulationDataEcoli:
    """Apply vEcoli variant functions to sim_data.

    Uses the real ``ecoli.variants.<name>.apply_variant()`` functions,
    which correctly handle complex multi-step mutations (expression
    adjustment, internal_shift_dict, condition/timeline changes, etc.).

    Args:
        sim_data: The SimulationDataEcoli to mutate (already deep-copied).
        variants_config: Dict mapping variant function names to lists of
            param dicts, as produced by
            ``UQInputParametersVecoli.to_simulation_config()["variants"]``.
            Example::

                {
                    "new_gene_internal_shift_variable_strength": [{
                        "condition": "basal",
                        "induction_gen": 1,
                        "exp_trl_eff": {"exp": 1.5, "trl_eff": 0.8},
                    }],
                    "mecillinam_timeline": [{
                        "times": [0.0],
                        "concentrations": [5.0],
                    }],
                }

    Returns:
        The mutated sim_data (same object, mutated in place).
    """
    for variant_name, param_dicts in variants_config.items():
        variant_mod = importlib.import_module(f"ecoli.variants.{variant_name}")
        for params in param_dicts:
            # Fix rel_adj list lengths for new gene variants
            if variant_name == "new_gene_internal_shift_variable_strength":
                params = _fix_new_gene_rel_adj(sim_data, params)
            sim_data = variant_mod.apply_variant(sim_data, params)
    return sim_data


# -- RAMEmitter -> numpy extraction ----------------------------------------

# Parquet columns use "__" separators; the RAMEmitter stores nested dicts.
# e.g. "listeners__mass__dry_mass" -> ["listeners"]["mass"]["dry_mass"]

# Higher-order scalar properties (always extracted)
_LISTENER_SCALARS: dict[str, list[str]] = {
    "cell_mass": ["listeners", "mass", "cell_mass"],
    "dry_mass": ["listeners", "mass", "dry_mass"],
    "volume": ["listeners", "mass", "volume"],
}

# Array-valued listeners (extracted when requested via output_keys)
_LISTENER_ARRAYS: dict[str, list[str]] = {
    "mRNA_cistron_counts": ["listeners", "rna_counts", "mRNA_cistron_counts"],
    "monomer_counts": ["listeners", "monomer_counts"],
    "base_reaction_fluxes": ["listeners", "fba_results", "base_reaction_fluxes"],
}


def _extract_timeseries(
    raw_data: dict[float, dict],
    agent_id: str = "0",
    output_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract a (n_timesteps, n_obs) array from RAMEmitter data.

    Args:
        raw_data: Output of ``sim.query()`` with ``raw_output=True``.
            Structure: ``{time: {"agents": {"0": {nested state}}}, ...}``
        agent_id: Agent ID to extract (default "0" for single-cell).
        output_keys: Which outputs to include.  If *None*, extracts all
            scalar listeners (cell_mass, dry_mass, volume) plus growth_rate.

    Returns:
        Tuple of:
        - timeseries array of shape ``(n_timesteps, n_obs)``
        - list of observable column names (length ``n_obs``)
        - 1-D array of time values
    """
    sorted_times = sorted(raw_data.keys())
    if not sorted_times:
        return np.empty((0, 0)), [], np.array([])

    # Determine which listeners to extract
    if output_keys is None:
        extract_scalars = dict(_LISTENER_SCALARS)
        extract_arrays: dict[str, list[str]] = {}
    else:
        extract_scalars = {k: v for k, v in _LISTENER_SCALARS.items() if k in output_keys}
        extract_arrays = {k: v for k, v in _LISTENER_ARRAYS.items() if k in output_keys}

    # Collect raw values per timestep
    rows: list[list[float]] = []
    obs_names: list[str] | None = None

    for t in sorted_times:
        state = raw_data[t]
        try:
            agent_state = state["agents"][agent_id]
        except (KeyError, TypeError):
            continue

        row: list[float] = []
        names: list[str] = []

        # Scalar listeners
        for name, path in extract_scalars.items():
            val = _get_nested_dict(agent_state, path)
            row.append(float(val))
            names.append(name)

        # Array listeners (flattened)
        for name, path in extract_arrays.items():
            val = _get_nested_dict(agent_state, path)
            arr = np.asarray(val, dtype=float).ravel()
            row.extend(arr.tolist())
            names.extend([f"{name}_{i}" for i in range(len(arr))])

        rows.append(row)
        if obs_names is None:
            obs_names = names

    if not rows:
        return np.empty((0, 0)), [], np.array([])

    timeseries = np.array(rows)

    # Derive growth_rate from dry_mass if present
    if obs_names and "dry_mass" in obs_names:
        dm_idx = obs_names.index("dry_mass")
        dm = timeseries[:, dm_idx]
        gr = np.zeros_like(dm)
        gr[1:] = np.diff(dm) / np.where(dm[1:] != 0, dm[1:], 1.0)
        timeseries = np.column_stack([timeseries, gr])
        obs_names.append("growth_rate")

    return timeseries, obs_names or [], np.array(sorted_times)


# -- Simulation wrapper class -----------------------------------------------


@dataclass
class VecoliSimulationFunc:
    """Parameter-agnostic vEcoli simulation wrapper for the UQ pipeline.

    Implements the dual interface required by the ``./uq`` workflow:

    - ``__call__(x)`` -> raw timeseries ``(n_timesteps, n_obs)``
      (used by ``Strategy4Wrapper`` in Phase 2)
    - ``evaluate_batch(X)`` -> aggregated outputs ``(n_samples, n_outputs)``
      (used by ``SensitivityAnalyzer`` in Phase 1)

    The parameter vector ``x`` is a 1-D numpy array whose elements
    correspond positionally to ``param_space.parameter_names``.  The
    ``param_space`` (XSpace) converts ``x`` into
    ``UQInputParametersVecoli`` via ``sample_to_params(x)``, which is
    then converted to vEcoli variant param dicts.  The real
    ``ecoli.variants.<name>.apply_variant()`` functions handle the
    complex sim_data mutations (expression adjustment, internal shifts,
    condition/timeline changes).

    Args:
        baseline_sim_data: The unperturbed SimulationDataEcoli object.
        param_space: XSpace instance that converts x -> UQInputParametersVecoli.
            Provides parameter_names for Sobol labeling and sample_to_params()
            for converting numpy arrays to variant-compatible param dicts.
        sim_config_path: Path to EcoliSim JSON config file (optional,
            uses default if None).
        max_duration: Simulation duration in seconds.
        output_keys: Which outputs to extract from the timeseries.
            If None, extracts scalar listeners (cell_mass, dry_mass,
            volume) plus derived growth_rate.
    """

    baseline_sim_data: SimulationDataEcoli
    param_space: Any  # XSpace — avoid circular import
    sim_config_path: str | None = None
    max_duration: float = 10800.0
    output_keys: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    @property
    def parameter_names(self) -> list[str]:
        """UQ parameter names (for Sobol index labeling)."""
        return self.param_space.parameter_names

    @property
    def obs_names(self) -> list[str]:
        """Observable names determined after the first simulation run."""
        if self._obs_names is None:
            raise RuntimeError(
                "Observable names not yet known. Run at least one "
                "simulation to populate obs_names."
            )
        return self._obs_names

    def _mutate_sim_data(self, x: np.ndarray) -> SimulationDataEcoli:
        """Deep-copy baseline sim_data and apply variant mutations from x.

        Flow:
            x (numpy array)
            -> param_space.sample_to_params(x) -> UQInputParametersVecoli
            -> .to_simulation_config() -> {"variants": {name: [params]}, ...}
            -> _apply_variants(sim_data, variants) using real apply_variant()
        """
        sd = copy.deepcopy(self.baseline_sim_data)

        # x -> UQInputParametersVecoli -> simulation config with variants
        uq_params = self.param_space.sample_to_params(x)
        sim_config = uq_params.to_simulation_config()
        variants = sim_config.get("variants", {})

        if variants:
            sd = _apply_variants(sd, variants)

        return sd

    def _run_single(self, x: np.ndarray) -> dict[float, dict]:
        """Run a single-cell EcoliSim and return RAMEmitter raw data."""
        sd = self._mutate_sim_data(x)

        # Write mutated sim_data to a temp pickle
        fd, variant_path = tempfile.mkstemp(suffix=".cPickle")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(sd, f)

            # Build and run EcoliSim
            if self.sim_config_path is not None:
                sim = EcoliSim.from_file(self.sim_config_path)
            else:
                sim = EcoliSim.from_file()

            sim.config["sim_data_path"] = variant_path
            sim.emitter = "timeseries"
            sim.raw_output = True
            sim.max_duration = self.max_duration

            sim.build_ecoli()
            sim.run()
            data = sim.query()
        finally:
            os.unlink(variant_path)

        return data

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run simulation for a single parameter vector.

        Args:
            x: Parameter values, shape ``(n_params,)``.

        Returns:
            Raw timeseries array of shape ``(n_timesteps, n_obs)``.
            This is the format expected by ``Strategy4Wrapper`` in Phase 2.
        """
        raw_data = self._run_single(x)
        timeseries, obs_names, _ = _extract_timeseries(
            raw_data,
            output_keys=self.output_keys,
        )
        self._obs_names = obs_names
        return timeseries

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate simulation for a batch of parameter vectors.

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.

        Returns:
            Aggregated (time-mean) output array of shape
            ``(n_samples, n_outputs)``.  This is the format expected by
            ``SensitivityAnalyzer.analyze_with_pce()`` in Phase 1.
        """
        results = []
        for x in X:
            ts = self(x)  # (n_timesteps, n_obs)
            results.append(ts.mean(axis=0))  # aggregate -> (n_obs,)
        return np.vstack(results)


# -- Legacy config classes (kept for backward compatibility) -----------------


class NextflowProfile(StrEnum):
    STANDARD = "standard"
    AWS = "aws"
    CCAM = "ccam"


@dataclass
class OutputEmitterConfig(BaseClass):
    type: Literal["parquet", "timeseries", "xarray"] = "parquet"
    out_dir: str | None = None
    out_uri: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.out_uri is None and self.out_dir is None:
            self.out_uri = get_bucket()


@dataclass
class VariantVecoli(BaseClass):
    id: Literal["new_gene_internal_shift_variable_strength", "condition", "mecillinam_timeline"]
    config: dict[str, Any]


@dataclass
class SimulationConfigVecoli(SimulationConfig):
    """Vecoli simulation config (JSON), 1:1"""

    experiment_id: str
    sim_data_path: str | None = None
    n_init_sims: int = field(default=1)
    generations: int = field(default=1)
    variants: list[VariantVecoli] = field(default_factory=list)
    emitter_arg: OutputEmitterConfig = field(default=OutputEmitterConfig)

    def validate(self) -> bool:
        return True

    def model_dump(self) -> dict[str, Any]:
        attrs = ['experiment_id', 'sim_data_path', 'n_init_sims', 'generations']
        config = dict(zip(attrs, [getattr(self, attr) for attr in attrs]))
        config.update(self._format_emitter())
        config.update({"variants": {variant.id: variant.config for variant in self.variants}})
        return config

    def _format_emitter(self) -> dict[str, Any]:
        outdir_key = "out_dir"
        path = self.emitter_arg.out_dir or self.emitter_arg.out_uri
        if path == self.emitter_arg.out_uri:
            outdir_key = "out_uri"

        return {"emitter": self.emitter_arg.type, "emitter_arg": {outdir_key: path}}


# -- Example / smoke test ---------------------------------------------------


def example_simulation_func(
    sim_data_path: str,
    max_duration: float = 10.0,
    include_vio: bool = False,
    include_mecillinam: bool = False,
) -> dict[str, Any]:
    """End-to-end example: load sim_data, build f(x)->y, run one sample.

    Demonstrates the full mutation -> simulate -> extract chain and
    returns everything the downstream pipeline needs.

    Args:
        sim_data_path: Path to a simData.cPickle file.
        max_duration: Short duration for testing (default 10s).
        include_vio: Include vio pathway parameters.
        include_mecillinam: Include mecillinam parameters.

    Returns:
        Dict with keys:
            sim_func: VecoliSimulationFunc instance (pass to pipeline)
            param_space: XSpaceVecoli
            x_sample: the parameter vector used
            uq_params: UQInputParametersVecoli from sample_to_params
            sim_config: variant config dict sent to apply_variant
            timeseries: (n_timesteps, n_obs) raw output from __call__
            aggregated: (n_obs,) time-mean output from evaluate_batch
            obs_names: list of observable column names
            times: 1-D array of simulation time values

    Example::

        from uq.generators.vecoli import example_simulation_func

        result = example_simulation_func(
            sim_data_path="/path/to/vEcoli/out/parca/kb/simData.cPickle",
            max_duration=10.0,
        )

        # What the pipeline receives:
        sim_func = result["sim_func"]       # pass to run_phase1 / run_phase2
        param_space = result["param_space"] # pass to SensitivityAnalyzer

        # Phase 1 interface (SensitivityAnalyzer calls this):
        X = np.random.uniform(size=(5, param_space.n_parameters))
        Y = sim_func.evaluate_batch(X)   # (5, n_obs)

        # Phase 2 interface (Strategy4Wrapper calls this):
        ts = sim_func(X[0])              # (n_timesteps, n_obs)
    """
    from uq.pipeline.param_loader import ParameterDataset

    # --- 1. Load baseline sim_data and build parameter space ---
    ds = ParameterDataset(sim_data_path=sim_data_path)
    param_space = ds.to_parameter_space(
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )

    print(f"Parameter space: {param_space.parameter_names}")
    print(f"Bounds: {param_space.parameter_bounds}")

    if param_space.n_parameters == 0:
        raise ValueError(
            "Parameter space is empty. Set include_vio=True and/or "
            "include_mecillinam=True, or use sim_data that contains "
            "new gene entries (violacein)."
        )

    # --- 2. Instantiate VecoliSimulationFunc ---
    sim_func = VecoliSimulationFunc(
        baseline_sim_data=ds.sim_data,
        param_space=param_space,
        max_duration=max_duration,
    )

    # --- 3. Create a sample x vector (midpoint of bounds) ---
    bounds = param_space.bounds_array
    x_sample = (bounds[:, 0] + bounds[:, 1]) / 2.0
    print(f"Sample x: {dict(zip(param_space.parameter_names, x_sample))}")

    # --- 4. Trace the mutation chain (without running sim) ---
    uq_params = param_space.sample_to_params(x_sample)
    sim_config = uq_params.to_simulation_config()
    print(f"Variant config: {sim_config.get('variants', {})}")

    # --- 5. Run single-cell simulation via __call__ (Phase 2 interface) ---
    print(f"Running EcoliSim (max_duration={max_duration}s)...")
    timeseries = sim_func(x_sample)  # (n_timesteps, n_obs)
    print(f"Timeseries shape: {timeseries.shape}")
    print(f"Observable names: {sim_func.obs_names}")

    # --- 6. Run evaluate_batch (Phase 1 interface) ---
    X_batch = x_sample.reshape(1, -1)  # (1, n_params)
    Y_aggregated = sim_func.evaluate_batch(X_batch)  # (1, n_obs)
    print(f"Aggregated output shape: {Y_aggregated.shape}")
    print(f"Aggregated values: {dict(zip(sim_func.obs_names, Y_aggregated[0]))}")

    # --- 7. Also extract with full timeseries metadata ---
    raw_data = sim_func._run_single(x_sample)
    _, obs_names, times = _extract_timeseries(raw_data)

    return {
        "sim_func": sim_func,
        "param_space": param_space,
        "x_sample": x_sample,
        "uq_params": uq_params,
        "sim_config": sim_config,
        "timeseries": timeseries,
        "aggregated": Y_aggregated[0],
        "obs_names": sim_func.obs_names,
        "times": times,
    }
