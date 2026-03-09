"""
Input-to-output wrapper functions for UQ sensitivity analysis.

This module provides wrapper functions that can be called from numerical
libraries (UQPy, PyTUQ) to map input parameters to simulation outputs.

The wrappers handle:
1. Converting numpy arrays to UQInputParameters
2. Running simulations (or loading cached results)
3. Extracting and aggregating outputs
4. Returning numpy arrays suitable for sensitivity analysis
"""

import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from uq.aggregation import AggregatedOutput, AggregationStrategy, Aggregator
from uq.inputs import InputParameterSpace, UQInputParameters
from uq.outputs import OutputType

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


@dataclass
class WrapperConfig:
    """
    Configuration for the simulation wrapper.

    Attributes:
        sim_data_path: Path to the sim_data pickle file
        output_dir: Directory for simulation outputs
        cache_dir: Directory for caching simulation results
        generations: Number of generations to simulate
        aggregation_strategy: Strategy for aggregating outputs
        output_types: Types of outputs to extract
        generation_lower_bound: Filter for minimum generation
        time_lower_bound: Filter for minimum time
        use_cache: Whether to cache and reuse simulation results
        n_cpus: Number of CPUs for simulation
    """

    sim_data_path: str
    output_dir: str = "./uq_outputs"
    cache_dir: str = "./uq_cache"
    generations: int = 8
    aggregation_strategy: AggregationStrategy = AggregationStrategy.UNIFORM
    output_types: list[OutputType] = field(
        default_factory=lambda: [
            OutputType.TRANSCRIPTOME,
            OutputType.PROTEOME,
            OutputType.EXCHANGE_FLUXES,
            OutputType.HIGHER_ORDER_PROPERTIES,
        ]
    )
    generation_lower_bound: Optional[int] = None
    time_lower_bound: Optional[float] = None
    use_cache: bool = True
    n_cpus: int = 1


class SimulationWrapper:
    """
    Wrapper that maps input parameters to simulation outputs.

    This class provides the interface expected by UQPy and PyTUQ libraries
    for running sensitivity analysis. It handles:
    - Running vEcoli simulations with specified parameters
    - Caching results to avoid redundant simulations
    - Extracting and aggregating outputs
    - Converting to numpy arrays for UQ analysis
    """

    def __init__(
        self,
        config: WrapperConfig,
        parameter_space: InputParameterSpace,
    ):
        """
        Initialize the simulation wrapper.

        Args:
            config: Wrapper configuration
            parameter_space: Definition of the input parameter space
        """
        self.config = config
        self.parameter_space = parameter_space

        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        if config.use_cache:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load sim_data for metadata
        self._sim_data: Optional[SimulationDataEcoli] = None

    @property
    def sim_data(self) -> "SimulationDataEcoli":
        """Lazy-load sim_data."""
        if self._sim_data is None:
            with open(self.config.sim_data_path, "rb") as f:
                self._sim_data = pickle.load(f)
        return self._sim_data

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the simulation for a single input sample.

        This is the main interface for UQPy/PyTUQ.

        Args:
            x: Input parameter array of shape (n_parameters,)

        Returns:
            Output array of shape (n_outputs,)
        """
        # Convert sample to parameters
        params = self.parameter_space.sample_to_params(x, generations=self.config.generations)

        # Check cache
        cache_key = self._get_cache_key(params)
        if self.config.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Run simulation
        output_path = self._run_simulation(params, cache_key)

        # Extract and aggregate outputs
        outputs = self._extract_outputs(output_path)

        # Convert to flat array
        result = self._outputs_to_array(outputs)

        # Cache result
        if self.config.use_cache:
            self._save_to_cache(cache_key, result)

        return result

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the simulation for a batch of input samples.

        Args:
            X: Input parameter array of shape (n_samples, n_parameters)

        Returns:
            Output array of shape (n_samples, n_outputs)
        """
        results = []
        for x in X:
            results.append(self(x))
        return np.vstack(results)

    def _get_cache_key(self, params: UQInputParameters) -> str:
        """Generate a unique cache key for the parameters."""
        config_dict = params.to_config_dict()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load cached result if available."""
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_to_cache(self, cache_key: str, result: np.ndarray) -> None:
        """Save result to cache."""
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.npy"
        np.save(cache_path, result)

    def _run_simulation(self, params: UQInputParameters, run_id: str) -> str:
        """
        Run a vEcoli simulation with the specified parameters.

        Args:
            params: Input parameters
            run_id: Unique identifier for this run

        Returns:
            Path to the output directory
        """
        output_path = Path(self.config.output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Create config file
        config_dict = params.to_config_dict()
        config_dict["sim_data_path"] = self.config.sim_data_path
        config_dict["emitter"] = "parquet"
        config_dict["out_dir"] = str(output_path)
        config_dict["experiment_id"] = run_id

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Run simulation using runscripts/workflow.py
        cmd = [
            "python",
            "-m",
            "runscripts.workflow",
            "--config",
            str(config_path),
        ]

        if self.config.n_cpus > 1:
            cmd.extend(["--cpus", str(self.config.n_cpus)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        return str(output_path)

    def _extract_outputs(self, output_path: str) -> dict[str, AggregatedOutput]:
        """
        Extract and aggregate outputs from simulation results.

        Args:
            output_path: Path to simulation output directory

        Returns:
            Dictionary mapping output names to AggregatedOutput
        """
        from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

        conn = create_duckdb_conn()

        # Get SQL queries for this dataset
        history_sql, config_sql, _ = dataset_sql(output_path, ["*"])

        aggregator = Aggregator(
            conn=conn,
            history_sql=history_sql,
            config_sql=config_sql,
            sim_data=self.sim_data,
        )

        outputs: dict[str, AggregatedOutput] = {}

        if OutputType.TRANSCRIPTOME in self.config.output_types:
            agg, ids = aggregator.aggregate_transcriptome(
                strategy=self.config.aggregation_strategy,
                generation_lower_bound=self.config.generation_lower_bound,
                time_lower_bound=self.config.time_lower_bound,
            )
            outputs["transcriptome"] = agg

        if OutputType.PROTEOME in self.config.output_types:
            agg, ids = aggregator.aggregate_proteome(
                strategy=self.config.aggregation_strategy,
                generation_lower_bound=self.config.generation_lower_bound,
                time_lower_bound=self.config.time_lower_bound,
            )
            outputs["proteome"] = agg

        if OutputType.EXCHANGE_FLUXES in self.config.output_types:
            agg, ids = aggregator.aggregate_fluxes(
                strategy=self.config.aggregation_strategy,
                generation_lower_bound=self.config.generation_lower_bound,
                time_lower_bound=self.config.time_lower_bound,
                exchange_only=True,
            )
            outputs["exchange_fluxes"] = agg

        if OutputType.METABOLIC_FLUXES in self.config.output_types:
            agg, ids = aggregator.aggregate_fluxes(
                strategy=self.config.aggregation_strategy,
                generation_lower_bound=self.config.generation_lower_bound,
                time_lower_bound=self.config.time_lower_bound,
                exchange_only=False,
            )
            outputs["metabolic_fluxes"] = agg

        if OutputType.HIGHER_ORDER_PROPERTIES in self.config.output_types:
            props = aggregator.aggregate_higher_order_properties(
                strategy=self.config.aggregation_strategy,
                generation_lower_bound=self.config.generation_lower_bound,
                time_lower_bound=self.config.time_lower_bound,
            )
            for name, agg in props.items():
                outputs[f"property_{name}"] = agg

        conn.close()
        return outputs

    def _outputs_to_array(self, outputs: dict[str, AggregatedOutput]) -> np.ndarray:
        """
        Convert aggregated outputs to a flat numpy array.

        For uniform aggregation, returns the mean values.
        For stratified aggregation, returns flattened means across groups.

        Args:
            outputs: Dictionary of aggregated outputs

        Returns:
            1D numpy array of output values
        """
        arrays = []
        for name, agg in outputs.items():
            if agg.mean.size > 0:
                arrays.append(agg.mean.flatten())

        if not arrays:
            return np.array([])

        return np.concatenate(arrays)

    def get_output_dimension(self) -> int:
        """
        Get the dimension of the output vector.

        Returns:
            Number of output values
        """
        # This requires running a dummy simulation or loading metadata
        # For now, estimate based on sim_data
        n_outputs = 0

        if OutputType.TRANSCRIPTOME in self.config.output_types:
            n_outputs += len(self.sim_data.process.transcription.cistron_data.struct_array)

        if OutputType.PROTEOME in self.config.output_types:
            n_outputs += len(self.sim_data.process.translation.monomer_data.struct_array)

        if OutputType.EXCHANGE_FLUXES in self.config.output_types:
            rxn_ids = list(self.sim_data.process.metabolism.reaction_stoich().keys())
            n_outputs += sum(1 for rxn_id in rxn_ids if "EX_" in rxn_id)

        if OutputType.METABOLIC_FLUXES in self.config.output_types:
            n_outputs += len(self.sim_data.process.metabolism.reaction_stoich())

        if OutputType.HIGHER_ORDER_PROPERTIES in self.config.output_types:
            n_outputs += 3  # cell_mass, dry_mass, volume

        return n_outputs


class PrecomputedWrapper:
    """
    Wrapper that uses precomputed simulation results.

    This is useful when simulations have already been run and stored,
    allowing sensitivity analysis without re-running simulations.
    """

    def __init__(
        self,
        data_dir: str,
        parameter_space: InputParameterSpace,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.UNIFORM,
        output_types: Optional[list[OutputType]] = None,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ):
        """
        Initialize with directory of precomputed results.

        Args:
            data_dir: Directory containing simulation outputs
            parameter_space: Definition of the input parameter space
            aggregation_strategy: Strategy for aggregating outputs
            output_types: Types of outputs to extract
            generation_lower_bound: Filter for minimum generation
            time_lower_bound: Filter for minimum time
        """
        self.data_dir = data_dir
        self.parameter_space = parameter_space
        self.aggregation_strategy = aggregation_strategy
        self.output_types = output_types or [
            OutputType.TRANSCRIPTOME,
            OutputType.PROTEOME,
            OutputType.EXCHANGE_FLUXES,
            OutputType.HIGHER_ORDER_PROPERTIES,
        ]
        self.generation_lower_bound = generation_lower_bound
        self.time_lower_bound = time_lower_bound

        # Index available simulations
        self._index_simulations()

    def _index_simulations(self) -> None:
        """Index available simulation results and their parameters."""
        self.simulation_index: dict[str, dict[str, Any]] = {}

        data_path = Path(self.data_dir)
        for config_file in data_path.glob("**/config.json"):
            with open(config_file) as f:
                config = json.load(f)

            sim_dir = config_file.parent
            sim_id = sim_dir.name
            self.simulation_index[sim_id] = {
                "path": str(sim_dir),
                "config": config,
            }

    def get_samples_and_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get input samples and corresponding outputs for all indexed simulations.

        Returns:
            Tuple of (X, Y) where:
            - X: Input samples of shape (n_samples, n_parameters)
            - Y: Outputs of shape (n_samples, n_outputs)
        """
        from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

        X_list = []
        Y_list = []

        for sim_id, sim_info in self.simulation_index.items():
            # Extract input parameters
            config = sim_info["config"]
            params = self._config_to_params(config)
            x = self.parameter_space.params_to_sample(params)
            X_list.append(x)

            # Extract outputs
            conn = create_duckdb_conn()
            history_sql, config_sql, _ = dataset_sql(sim_info["path"], ["*"])

            aggregator = Aggregator(
                conn=conn,
                history_sql=history_sql,
                config_sql=config_sql,
            )

            outputs = self._extract_outputs(aggregator)
            y = self._outputs_to_array(outputs)
            Y_list.append(y)

            conn.close()

        return np.vstack(X_list), np.vstack(Y_list)

    def _config_to_params(self, config: dict[str, Any]) -> UQInputParameters:
        """Convert a config dictionary back to UQInputParameters."""
        from uq.inputs import (
            MecillinamParams,
            MediaCondition,
            VioPathwayParams,
        )

        params = UQInputParameters()

        if "variants" in config:
            variants = config["variants"]

            # Extract vio parameters
            if "new_gene_internal_shift_variable_strength" in variants:
                vio_config = variants["new_gene_internal_shift_variable_strength"][0]
                params.vio = VioPathwayParams(
                    enabled=True,
                    induction_gen=vio_config.get("induction_gen", 1),
                    knockout_gen=vio_config.get("knockout_gen"),
                    expression=vio_config.get("exp_trl_eff", {}).get("exp", 1.0),
                    translation_efficiency=vio_config.get("exp_trl_eff", {}).get("trl_eff", 1.0),
                    condition=MediaCondition(vio_config.get("condition", "basal")),
                )
            else:
                params.vio.enabled = False

            # Extract mecillinam parameters
            if "mecillinam_timeline" in variants:
                mec_config = variants["mecillinam_timeline"][0]
                params.mecillinam = MecillinamParams(
                    times=mec_config.get("times", [0.0]),
                    concentrations=mec_config.get("concentrations", [0.0]),
                    knockouts=mec_config.get("knockouts", []),
                )

        params.seed = config.get("seed", 0)
        params.generations = config.get("generations", 8)

        return params

    def _extract_outputs(self, aggregator: Aggregator) -> dict[str, AggregatedOutput]:
        """Extract and aggregate outputs using the aggregator."""
        outputs: dict[str, AggregatedOutput] = {}

        if OutputType.TRANSCRIPTOME in self.output_types:
            agg, _ = aggregator.aggregate_transcriptome(
                strategy=self.aggregation_strategy,
                generation_lower_bound=self.generation_lower_bound,
                time_lower_bound=self.time_lower_bound,
            )
            outputs["transcriptome"] = agg

        if OutputType.PROTEOME in self.output_types:
            agg, _ = aggregator.aggregate_proteome(
                strategy=self.aggregation_strategy,
                generation_lower_bound=self.generation_lower_bound,
                time_lower_bound=self.time_lower_bound,
            )
            outputs["proteome"] = agg

        if OutputType.EXCHANGE_FLUXES in self.output_types:
            agg, _ = aggregator.aggregate_fluxes(
                strategy=self.aggregation_strategy,
                generation_lower_bound=self.generation_lower_bound,
                time_lower_bound=self.time_lower_bound,
                exchange_only=True,
            )
            outputs["exchange_fluxes"] = agg

        if OutputType.HIGHER_ORDER_PROPERTIES in self.output_types:
            props = aggregator.aggregate_higher_order_properties(
                strategy=self.aggregation_strategy,
                generation_lower_bound=self.generation_lower_bound,
                time_lower_bound=self.time_lower_bound,
            )
            for name, agg in props.items():
                outputs[f"property_{name}"] = agg

        return outputs

    def _outputs_to_array(self, outputs: dict[str, AggregatedOutput]) -> np.ndarray:
        """Convert aggregated outputs to a flat numpy array."""
        arrays = []
        for name, agg in outputs.items():
            if agg.mean.size > 0:
                arrays.append(agg.mean.flatten())

        if not arrays:
            return np.array([])

        return np.concatenate(arrays)


def create_uqpy_model(
    config: WrapperConfig,
    parameter_space: InputParameterSpace,
) -> Any:
    """
    Create a UQPy-compatible model wrapper.

    Args:
        config: Wrapper configuration
        parameter_space: Definition of the input parameter space

    Returns:
        UQPy RunModel object
    """
    try:
        from UQpy.run_model.model_execution.PythonModel import PythonModel
        from UQpy.run_model.RunModel import RunModel
    except ImportError:
        raise ImportError("UQPy is required for this function. Install it with: pip install UQpy")

    wrapper = SimulationWrapper(config, parameter_space)

    class VEcoliModel:
        def __init__(self):
            self.wrapper = wrapper

        def run(self, samples: np.ndarray) -> np.ndarray:
            return self.wrapper.evaluate_batch(samples)

    model = PythonModel(
        model_script="uq.wrappers",
        model_object_name="SimulationWrapper",
        var_names=parameter_space.parameter_names,
    )

    return RunModel(model=model)


def create_pytuq_model(
    config: WrapperConfig,
    parameter_space: InputParameterSpace,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a PyTUQ-compatible model function.

    PyTUQ expects a simple function that takes parameter array and returns output array.

    Args:
        config: Wrapper configuration
        parameter_space: Definition of the input parameter space

    Returns:
        Function mapping parameters to outputs
    """
    wrapper = SimulationWrapper(config, parameter_space)

    def model_func(X: np.ndarray) -> np.ndarray:
        """
        Evaluate model at given parameter values.

        Args:
            X: Parameter array of shape (n_samples, n_params) or (n_params,)

        Returns:
            Output array of shape (n_samples, n_outputs) or (n_outputs,)
        """
        if X.ndim == 1:
            return wrapper(X)
        else:
            return wrapper.evaluate_batch(X)

    return model_func
