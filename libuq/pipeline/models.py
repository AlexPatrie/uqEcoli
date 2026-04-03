import abc
import json
import math
import random
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import numpy as np
import polars
import pytest
from ecoli.library.sim_data import LoadSimData
from pydantic import ConfigDict
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from libuq.common.models import BaseClass
from libuq.io import get_bucket
from libuq.pce.models import Parameter
from libuq.synthetic import generate_synthetic_simulation_data

if TYPE_CHECKING:
    from libuq.sensitivity import CellCycleRelevanceResult, MorrisIndices, PCESurrogate, SobolIndices


# === Simulator: where and how to obtain the function that will be fit by UQ === #


@dataclass
class SimulatorSource(BaseClass):
    type: Literal["repo", "pypi", "conda"]
    value: str  # if pypi, whatever pip install <PACKAGE> where <PACKAGE> is value


@classmethod
class SimulatorInstallationConfig:
    command: Literal["uv add", "pip install", "git clone"]
    source: SimulatorSource  # GitRepoUrl(sms_api.Simulator), PyPIPackageName, CondaForgeID


@dataclass
class SimulatorConfig:
    simulator_name: str
    installation: SimulatorInstallationConfig

    def install(self) -> subprocess.CompletedProcess[str]:
        cmd = f"{self.installation.command} {self.installation.source.value}"
        return subprocess.run(cmd.split(" "), check=True)


# === Simulations: simulation config, metadata, computes, API requests, etc ===

# These classes should parameterize the stochastic timeseries generator (f(x) -> y), where
#     SimulationConfig is an attribute of x. x should consist of both simulation, and model-specific params
# Vecoli-specific Simulator param classes


class MediaCondition(str, Enum):
    """Available media conditions for simulations."""

    BASAL = "basal"
    WITH_AA = "with_aa"
    ACETATE = "acetate"
    SUCCINATE = "succinate"
    NO_OXYGEN = "no_oxygen"


class InvalidParamDefinition(Exception):
    pass


@dataclass
class SimDataParameter:
    """A single scalar sim_data parameter for UQ sensitivity analysis.

    Specifies a mutable attribute in ``SimulationDataEcoli`` by dot-path,
    enabling arbitrary parameter sweeps without vEcoli variant functions.

    Attributes:
        name: Human-readable label (used in Sobol index labeling).
        attr_path: Dot-separated path into the sim_data object tree
            (e.g. ``"process.metabolism.kinetic_objective_weight"``).
        bounds: ``(lower, upper)`` bounds for LHS sampling.
        index: For array-valued attributes, which element to perturb.
            If None, the attribute must be a scalar.
        description: Optional human-readable description.
    """

    name: str
    attr_path: str
    bounds: tuple[float, float]
    index: int | None = None
    description: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attr_path": self.attr_path,
            "bounds": list(self.bounds),
            "index": self.index,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimDataParameter":
        return cls(
            name=d["name"],
            attr_path=d["attr_path"],
            bounds=tuple(d["bounds"]),
            index=d.get("index"),
            description=d.get("description", ""),
        )


@dataclass
class GenericSimDataParams:
    """Container for generic sim_data parameter values.

    Maps ``SimDataParameter`` specs to concrete float values from an
    LHS sample.  Used by the direct-mutation path (no variant functions).
    """

    parameter_specs: list[SimDataParameter]
    values: dict[str, float]  # name -> value
    seed: int = 0
    generations: int = 8

    def to_simulation_config(self) -> dict[str, Any]:
        """Convert to a config dict with direct sim_data mutations.

        Returns:
            Dict with ``"sim_data_mutations"`` mapping attr_paths to values.
        """
        config: dict[str, Any] = {
            "seed": self.seed,
            "generations": self.generations,
        }
        mutations: dict[str, Any] = {}
        for spec in self.parameter_specs:
            val = self.values[spec.name]
            if spec.index is not None:
                mutations[spec.attr_path] = {
                    "__index__": spec.index,
                    "__value__": float(val),
                }
            else:
                mutations[spec.attr_path] = float(val)
        config["sim_data_mutations"] = mutations
        return config


@dataclass
class Param(BaseClass):
    """
    Input parameter for UQ pipeline extracted from sim_data.

    Attributes:
        name: str
        bounds: tuple[float, float] (low, high)
        granularity_step: float - granularity of control allowed for ui element
            range control (knob, slider, etc). TODO: move this.
        description: str
    """

    name: str
    bounds: tuple[float, float] | tuple[complex, complex] | None = None
    default: float | int | complex | None = None
    value: float | int | complex | None = None
    granularity_step: float = 0.25
    description: str | None = None

    def __post_init__(self) -> None:
        if self.bounds is None:
            self.bounds = [-1.0, 1.0]
        if self.value is None:
            if self.default is None:
                self.default = self.bounds[1] - self.bounds[0]
            self.set(self.default)

    @property
    def dtype(self):
        return type(self.value)

    def set(self, value: float | int | complex):
        self.value = value

    def model_dump(self):
        d = asdict(self)
        bounds = tuple(self.bounds)
        d["bounds"] = bounds
        d["dtype"] = self.dtype.name
        return d


@dataclass
class BaseClass:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def export(self, f: Path) -> None:
        with open(f.__str__(), "w") as fp:
            json.dump(self.model_dump(), fp, indent=3)


@dataclass
class VariantConfig(BaseClass):
    attribute_id: str  # if SimData were a dataframe, this is one of the cols
    value: float | int | complex = None  # not implicitly delta, rather manually set val
    delta: float | int | complex | None = None

    @classmethod
    def from_delta(cls, param: Param, delta: float | int | complex) -> "VariantConfig":
        variant = VariantConfig(attribute_id=param.name, value=param.value - delta)
        variant.delta = delta
        return variant


# TODO: make a to_param() -> Param method!


@dataclass
class VecoliParams(BaseClass, abc.ABC):
    variants: list[VariantConfig] = field(default_factory=list)

    def apply_perturbations(
        self, selected: dict[str, float | int | complex], factor: float
    ) -> dict[str, float | complex | int]:
        return {key: val * type(factor) for key, val in selected.items()}

    @abc.abstractmethod
    def get_default_perturbations(self):
        pass

    @property
    def default_perturbations(self):
        dp = self.get_default_perturbations()
        dp.update({variant.attribute_id: variant.value for variant in self.variants})
        return dp

    def to_variant_params(self) -> dict[str, Any]:
        """
        Convert to variant parameter dictionary for mecillinam_timeline.

        Returns:
            Dictionary compatible with apply_variant function
        """
        return self.default_perturbations


class NextflowProfile(StrEnum):
    STANDARD = "standard"
    AWS = "aws"
    CCAM = "ccam"


@dataclass
class OutputEmitterConfig(BaseClass):
    type: Literal["parquet", "timeseries", "xarray"]
    out_dir: str | None = None
    out_uri: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig(BaseClass, abc.ABC):
    def __post_init__(self) -> None:
        if not self.validate():
            raise ValueError(f"The requested config payload does not match expected structure:\n{self.model_dump()}")

    @abc.abstractmethod
    def validate(self) -> bool:
        """
        The purpose of this method is to validate timeseries generator
            request payloads/workloads and safely onboard new configs/workloads
            by preventing arbitrary configs or arbitrary funcs.

        This method should be implemented such that:

        Inputs:
            Implementation attributes/fields and vals
        Outputs:
            Whether the serialized representation produced by
                the `model_dump()` method has values
                that are at least in some way directly consumable
                by the corresponding Simulator
        """
        pass


@dataclass
class VariantVecoli(BaseClass):
    """
    Attributes:
        id: str variant module name in vecoli
        config: dict[str, Any] kwarg values to be evaluated as the `params` parameter of ecoli.variants.apply_variants()
    """

    # id: Literal["new_gene_internal_shift_variable_strength", "condition", "mecillinam_timeline"]
    id: str
    config: dict[str, Any]


@dataclass
class SimulationConfigVecoli(SimulationConfig):
    """
    Attributes:
        experiment_id: str
        sim_data_path: str | None = None
        n_init_sims: int = field(default=1)
        generations: int = field(default=1)
        variants: list[VariantVecoli] = field(default_factory=list)
        emitter_arg: OutputEmitterConfig | dict = field(
            default_factory=dict
        )  # OutputEmitterConfig(type="parquet", out_uri=get_bucket())
    """

    experiment_id: str
    sim_data_path: str | None = None
    n_init_sims: int = field(default=1)
    generations: int = field(default=1)
    max_duration: float = field(default=10800.0)
    variants: list[VariantVecoli] = field(default_factory=list)
    emitter_arg: OutputEmitterConfig | dict = field(
        default_factory=dict
    )  # OutputEmitterConfig(type="parquet", out_uri=get_bucket())

    def validate(self) -> bool:
        # TODO: add full vEcoli simulation config JSON attributes exposed by SMS API
        # seed, generations, "new_gene_internal_shift_variable_strength" condition mecillinam_timeline, variants
        # expected_keys = ['emitter', 'emitter_arg', 'experiment_id', 'sim_data_path']
        # expected_types = [str, dict, str, str]
        # payload = self.model_dump()
        # payload_keys = sorted(list(payload.keys()))
        # return payload_keys == expected_keys \
        #     and [type(payload[key]) for key in payload_keys] == expected_types
        return True

    def model_dump(self) -> dict[str, Any]:
        attrs = [self.experiment_id, self.sim_data_path, self.n_init_sims, self.generations, self.max_duration]
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


@dataclass
class Simulation(BaseClass):
    """
    Attributes:
        database_id: int
        config: SimulationConfigVecoli
    """

    database_id: int
    config: SimulationConfigVecoli


@dataclass
class TimeseriesDataset(BaseClass):
    """
    Attributes:
        database_id: int
        simulation: Simulation
        metadata_included: bool
        selections: list[str] | None (observables to select)
    """

    database_id: int  # TODO: used to perform loookup dataset from db/s3 (cross-reference with simulation attr)
    simulation: Simulation  # or simulation_id
    metadata_included: bool = True
    selections: list[str] | None = None

    @property
    def outdir_root(self):
        emitter = self.simulation.config.emitter_arg
        return emitter.out_dir or emitter.out_uri

    @property
    def x(self):
        # TODO: get this from stanford (simdata as df)
        return self.load_simdata()

    @property
    def y(self):
        return self.load_timeseries()

    def _get_repo_root(self) -> Path:
        """Get the repository root directory."""
        # Try to find repo root by looking for pyproject.toml
        current = Path(__file__).resolve().parent
        for _ in range(10):  # Max 10 levels up
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        # Fallback to cwd
        return Path.cwd()

    def load_dataset(
        self,
        experiment_id: str,
        outdir_root: Path | None = None,
        observables: list[str] | None = None,
        include_metadata: bool = True,
    ) -> polars.DataFrame:
        """
        Load simulation dataset from parquet files.

        Args:
            experiment_id: The experiment identifier (e.g., "api_simulation_default")
            outdir_root: Root directory for simulation outputs. Defaults to
                         {repo_root}/api_integration/sims
            observables: Optional list of column names to select
            include_metadata: If True, include hive partition columns (variant, lineage_seed,
                             generation, agent_id) from the directory structure

        Returns:
            Polars DataFrame with the simulation data
        """
        if outdir_root is None:
            outdir_root = self._get_repo_root() / "api_integration" / "sims"

        base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"

        # Scan nested parquet files with hive partitioning to extract metadata columns
        # (variant, lineage_seed, generation, agent_id) from directory structure
        lf = polars.scan_parquet(str(base_path / "**/*.pq"), hive_partitioning=include_metadata)

        if observables is not None:
            # Filter to only existing columns, but always include metadata if requested
            available = lf.collect_schema().names()
            valid_observables = [col for col in observables if col in available]

            # Add metadata columns if they exist and include_metadata is True
            if include_metadata:
                metadata_cols = ["variant", "lineage_seed", "generation", "agent_id"]
                for col in metadata_cols:
                    if col in available and col not in valid_observables:
                        valid_observables.append(col)

            if valid_observables:
                lf = lf.select(valid_observables)

        return lf.collect()

    def load_timeseries(self) -> polars.DataFrame:
        return self.load_dataset(
            self.simulation.config.experiment_id,
            Path(self.outdir_root),
            observables=self.selections,
            include_metadata=self.metadata_included,
        )

    def load_simdata(self) -> polars.DataFrame:  # SimulationDataEcoli:
        # TODO: be able to turn into pl.DataFrame!
        # return LoadSimData(sim_data_path=self.simulation.config.sim_data_path).sim_data
        return generate_synthetic_simulation_data()


# === UQ Pipeline (e2e workflow params and outputs): x, y, theta, etc ===


@dataclass
class UQInputParametersInterface(BaseClass, abc.ABC):
    """
    Implementations of this interface should implement the `to_simulator_config()` method,
    in which the parameters defined in this class are formatted for the given
    simulator (library/repo/api). In this case, the aforementioned "simulator" provides
    the simulation executor function, or at least that which is required to generate
    the timeseries outputs. This simulation function (f) is the callable that
    parameterizes the PCE phase(s) in creation and fitting of `PCESurrogate` output
    instances.

    The naming scheme should be:
        `UQInputParameters<SIMULATOR NAME>`
    """

    @abc.abstractmethod
    def to_simulation_config(self, *args, **kwargs) -> SimulationConfig:
        pass


@dataclass
class UQInputParameters(UQInputParametersInterface):
    """ """

    # params:

    @abc.abstractmethod
    def to_simulation_config(self, *args, **kwargs) -> SimulationConfig:
        pass


class CellCyclePhase(str, Enum):
    """Standard cell cycle phases for E. coli."""

    B_PERIOD = "B_period"  # Pre-initiation (birth to replication initiation)
    C_PERIOD = "C_period"  # DNA replication
    D_PERIOD = "D_period"  # Post-replication to division
    UNKNOWN = "unknown"


@dataclass
class CellCycleVariable(BaseClass):
    """
    Container for cell cycle variable values.

    Attributes:
        values: The computed cell cycle variable values, shape (n_timepoints,)
        phase_labels: Cell cycle phase labels for each timepoint
        normalized: Whether values are normalized to [0, 1]
        variable_name: Name of the cell cycle variable
        metadata: Additional metadata about the computation
    """

    values: np.ndarray
    phase_labels: Optional[np.ndarray] = None
    normalized: bool = True
    variable_name: str = "cell_cycle_variable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_stage_bins(self, n_bins: int = 10) -> np.ndarray:
        """
        Bin the cell cycle variable into discrete stages.

        Args:
            n_bins: Number of bins/stages

        Returns:
            Array of bin indices (0 to n_bins-1)
        """
        if self.normalized:
            bins = np.linspace(0, 1, n_bins + 1)
        else:
            bins = np.linspace(self.values.min(), self.values.max(), n_bins + 1)

        return np.digitize(self.values, bins) - 1


# === PCE: Surrogate creation, surrogate outputs ===


@dataclass
class PCESurrogateProfile:
    population: "PCESurrogate"
    cell_cycle: "PCESurrogate"


@dataclass
class SobolIndexProfile(BaseClass):
    """
    each attribute of size 2 => one for fist order, one for total order
    """

    population: "SobolIndices"
    cell_cycle: list["SobolIndices"]  # (of clen(n_cell_cycle_bins))


# === UQ e2e workflow/pipeline results/outputs ===


class StratificationLens(StrEnum):
    POPULATION = "population"
    CELL_CYCLE = "cell_cycle"


@dataclass
class UqProfile:
    """
    Uncertainty Quantification Profile for a given simulation/dataset/parameter-set.
    The atomic output of `uq` workflows and the primary report for `uq` cli/api.

    Attributes:
        stratification: (StratificationLens) "population" (bulk, phase1) or "cell_cycle" (tempo is theta, phase2)
        sobol_indices: (list[SobolIndices]) For population stratification, must be len == 1, otherwise len == n_theta_bins
    """

    stratification: StratificationLens
    sobol_indices: list["SobolIndices"]
    surrogate: "PCESurrogate"  # TODO: or, surrogate_id --> hydrate pickled instances!

    def __post_init__(self) -> None:
        if len(self.sobol_indices) > 1 and self.stratification == StratificationLens.POPULATION:
            raise ValueError(
                "A population level stratification "
                "is only expected to have 1 set of SobolIndices. "
                "As such, a cell cycle level stratification is"
                "expected to have n_bin sets of SobolIndices!"
            )


@dataclass
class PipelineResult:
    """
    Complete output of the RFC006 UQ pipeline.

    Phase 1 gives the population-level view (bulk). Phase 2 gives the
    within-cell-lifecycle view (phenotypic). They decompose the same total
    variance into different components — like decomposing the total variance
    of human height into "between countries" vs "within countries."

    Attributes:
        population: Phase 1 UqProfile (bulk Sobol + PCE surrogate).
        cell_cycle: Phase 2 UqProfile (per-stage Sobol + PCE surrogate).
        variance_decomposition: Step 4 output — generation/seed/residual
            fractions per observable. Keys: 'generation_fraction',
            'seed_fraction', 'between_generation_variance',
            'between_seed_variance', 'within_group_variance'.
        aggregation: Step 3 output — AggregatedOutput for strategies 1-3.
            None if not retained (e.g., loaded from export).
        morris_indices: Step 5a output — Morris prescreening results.
            None if prescreening was skipped.
        cell_cycle_relevance: Step 5b output — GSA-informed observable
            selection for Koopman cell cycle variable.
            None if not retained (e.g., loaded from export).
    """

    population: UqProfile
    cell_cycle: UqProfile
    variance_decomposition: dict[str, Any] = field(default_factory=dict)
    aggregation: Any = None  # AggregationResult from workflow, not serialized
    morris_indices: Optional["MorrisIndices"] = None
    cell_cycle_relevance: Optional["CellCycleRelevanceResult"] = None
    cell_cycle_profile: Optional[dict[str, Any]] = None  # per-stage observable means/stds

    def export(self, path: str | Path) -> None:
        """Serialize the full pipeline result to disk.

        Creates:
            path/population_surrogate/        — PCESurrogate export
            path/cell_cycle_surrogate/        — PCESurrogate export
            path/population_sobol/            — DataclassIO export
            path/cell_cycle_sobol_stage_N/    — DataclassIO export per stage
            path/variance_decomposition.json  — Step 4 fractions
            path/morris_indices/              — DataclassIO export (if available)
            path/metadata.json                — Pipeline metadata
            path/uq_results.json              — Comprehensive human-readable summary
            path/koopman_spectrum.pdf          — Koopman eigenmode visualization (if available)
        """
        from libuq.io import DataclassIO

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Export surrogates
        self.population.surrogate.export(path / "population_surrogate")
        self.cell_cycle.surrogate.export(path / "cell_cycle_surrogate")

        # Export Sobol indices
        DataclassIO.save(self.population.sobol_indices[0], path / "population_sobol")
        for i, sobol in enumerate(self.cell_cycle.sobol_indices):
            DataclassIO.save(sobol, path / f"cell_cycle_sobol_stage_{i}")

        # Export variance decomposition (Step 4)
        if self.variance_decomposition:
            decomp_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.variance_decomposition.items()
            }
            (path / "variance_decomposition.json").write_text(json.dumps(decomp_serializable, indent=2))

        # Export Morris indices (Step 5a)
        if self.morris_indices is not None:
            DataclassIO.save(self.morris_indices, path / "morris_indices")

        # Metadata
        meta = {
            "n_cell_cycle_stages": len(self.cell_cycle.sobol_indices),
            "population_params": self.population.sobol_indices[0].parameter_names,
            "population_stratification": self.population.stratification.value,
            "cell_cycle_stratification": self.cell_cycle.stratification.value,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Export cell cycle profile (Step 6c per-stage means)
        if self.cell_cycle_profile is not None:
            profile_serializable = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.cell_cycle_profile.items()
            }
            (path / "cell_cycle_profile.json").write_text(json.dumps(profile_serializable, indent=2))

        # Comprehensive human-readable summary
        self._write_uq_results_json(path / "uq_results.json")

        # Koopman spectrum visualization
        self._write_koopman_pdf(path / "koopman_spectrum.pdf")

    def _write_uq_results_json(self, filepath: Path) -> None:
        """Write a comprehensive, lossless JSON summary of all pipeline outputs."""
        pop_sobol = self.population.sobol_indices[0]
        param_names = pop_sobol.parameter_names

        # Phase 1: population Sobol
        phase1_sobol = {
            "first_order": {name: float(pop_sobol.first_order[i]) for i, name in enumerate(param_names)},
            "total_order": {name: float(pop_sobol.total_order[i]) for i, name in enumerate(param_names)},
        }
        if pop_sobol.second_order is not None:
            phase1_sobol["second_order"] = pop_sobol.second_order.tolist()

        # Phase 2: per-stage Sobol
        n_stages = len(self.cell_cycle.sobol_indices)
        phase2_sobol_per_stage = []
        for k, sobol in enumerate(self.cell_cycle.sobol_indices):
            stage_entry = {
                "stage": k,
                "theta_range": [k / n_stages, (k + 1) / n_stages],
                "first_order": {
                    name: float(sobol.first_order[i]) if i < len(sobol.first_order) else 0.0
                    for i, name in enumerate(param_names)
                },
                "total_order": {
                    name: float(sobol.total_order[i]) if i < len(sobol.total_order) else 0.0
                    for i, name in enumerate(param_names)
                },
            }
            phase2_sobol_per_stage.append(stage_entry)

        # Variance decomposition
        decomp = {}
        if self.variance_decomposition:
            decomp = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.variance_decomposition.items()}

        # Morris indices
        morris = None
        if self.morris_indices is not None:
            mi = self.morris_indices
            morris = {
                "parameter_names": mi.parameter_names,
                "mu": mi.mu.tolist(),
                "mu_star": mi.mu_star.tolist(),
                "sigma": mi.sigma.tolist(),
                "n_trajectories": int(mi.n_trajectories) if hasattr(mi, "n_trajectories") else None,
            }

        # Cell cycle relevance
        cc_relevance = None
        if self.cell_cycle_relevance is not None:
            ccr = self.cell_cycle_relevance
            cc_relevance = {
                "relevant_observables": ccr.relevant_observables,
                "relevance_scores": {k: float(v) for k, v in ccr.relevance_scores.items()},
            }
            if ccr.residual_variance_fraction is not None:
                cc_relevance["residual_variance_fraction"] = ccr.residual_variance_fraction.tolist()

        # Surrogate summary
        pop_surr = self.population.surrogate
        cc_surr = self.cell_cycle.surrogate
        surrogates = {
            "population": {
                "basis_type": pop_surr.basis_type,
                "polynomial_order": pop_surr.polynomial_order,
                "input_dim": pop_surr.input_dim,
                "output_dim": pop_surr.output_dim,
                "r_squared": float(pop_surr.r_squared),
                "n_terms": len(pop_surr.coefficients),
            },
            "cell_cycle": {
                "basis_type": cc_surr.basis_type,
                "polynomial_order": cc_surr.polynomial_order,
                "input_dim": cc_surr.input_dim,
                "output_dim": cc_surr.output_dim,
                "r_squared": float(cc_surr.r_squared),
                "n_terms": len(cc_surr.coefficients),
            },
        }

        # Cell cycle profile (per-stage observable means)
        cc_profile = None
        if self.cell_cycle_profile is not None:
            cc_profile = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.cell_cycle_profile.items()}

        result = {
            "parameter_names": param_names,
            "n_parameters": len(param_names),
            "n_cell_cycle_stages": n_stages,
            "variance_decomposition": decomp,
            "phase1_population_sobol": phase1_sobol,
            "phase2_cell_cycle_sobol_per_stage": phase2_sobol_per_stage,
            "cell_cycle_profile": cc_profile,
            "morris_screening": morris,
            "cell_cycle_relevance": cc_relevance,
            "surrogates": surrogates,
        }

        filepath.write_text(json.dumps(result, indent=2))

    def _write_koopman_pdf(self, filepath: Path) -> None:
        """Write Koopman spectrum 4-panel figure as PDF if spectrum data is available."""
        try:
            # The cell cycle relevance result may have a koopman spectrum
            # attached, or we can check the cell cycle surrogate metadata.
            # The spectrum is available when GSAInformedCellCycleVariable was used.
            spectrum = None

            if self.cell_cycle_relevance is not None:
                # Check if relevance result has a koopman reference
                ccr = self.cell_cycle_relevance
                if hasattr(ccr, "koopman_spectrum") and ccr.koopman_spectrum is not None:
                    spectrum = ccr.koopman_spectrum

            if spectrum is None:
                return

            from libuq.viz import plot_koopman_spectrum

            fig = plot_koopman_spectrum(spectrum)
            fig.write_image(str(filepath))
        except Exception:
            # Don't fail the export if PDF generation fails (missing kaleido, etc.)
            pass

    @classmethod
    def from_export(cls, path: str | Path) -> "PipelineResult":
        """Load a PipelineResult from a previously exported directory."""
        from libuq.io import DataclassIO
        from libuq.sensitivity import MorrisIndices, PCESurrogate, SobolIndices

        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())

        pop_surrogate = PCESurrogate.from_export(path / "population_surrogate")
        pop_sobol = DataclassIO.load(path / "population_sobol", SobolIndices)

        cc_surrogate = PCESurrogate.from_export(path / "cell_cycle_surrogate")
        cc_sobols = [
            DataclassIO.load(path / f"cell_cycle_sobol_stage_{i}", SobolIndices)
            for i in range(meta["n_cell_cycle_stages"])
        ]

        # Load variance decomposition if available
        decomp_path = path / "variance_decomposition.json"
        variance_decomposition = {}
        if decomp_path.exists():
            raw = json.loads(decomp_path.read_text())
            variance_decomposition = {k: np.array(v) if isinstance(v, list) else v for k, v in raw.items()}

        # Load Morris indices if available
        morris_indices = None
        if (path / "morris_indices").exists():
            morris_indices = DataclassIO.load(path / "morris_indices", MorrisIndices)

        # Load cell cycle profile if available
        cell_cycle_profile = None
        profile_path = path / "cell_cycle_profile.json"
        if profile_path.exists():
            raw = json.loads(profile_path.read_text())
            cell_cycle_profile = {k: np.array(v) if isinstance(v, list) else v for k, v in raw.items()}

        return cls(
            population=UqProfile(
                stratification=StratificationLens.POPULATION,
                sobol_indices=[pop_sobol],
                surrogate=pop_surrogate,
            ),
            cell_cycle=UqProfile(
                stratification=StratificationLens.CELL_CYCLE,
                sobol_indices=cc_sobols,
                surrogate=cc_surrogate,
            ),
            variance_decomposition=variance_decomposition,
            morris_indices=morris_indices,
            cell_cycle_profile=cell_cycle_profile,
        )


@dataclass
class PipelineConfig:
    name: str
    dataset_id: int  # used to look up TimeseriesDataset in DB for SMS API :)


@dataclass
class Pipeline:
    """
    Attributes:
        database_id: int
        config: PipelineConfig consisting of pipeline name and dataset_id.
        dataset: Timeseries dataset containing the following attributes: timeseries data(y), parameter dataset (x), database_id, and simulation. Simulation itself
            has a database_id, and a config (vecoli config/api request config?)
        result: PipelineResult object containing 2 `UqProfile` instances, one for
            each stratification type (population(bulk), cell_cycle(cell, [i][j])
    """

    database_id: int
    config: PipelineConfig
    dataset: TimeseriesDataset
    result: Optional[PipelineResult] = None

    def build(
        self,
        simulation_func,
        sim_data_path: str | Path | list[str | Path] | None = None,
        observable_columns: list[str] | None = None,
        output_types: list[str] | None = None,
        generation_lower_bound: int | None = 2,
        time_lower_bound: float | None = 100.0,
        n_bins: int = 10,
        polynomial_order: int = 3,
        n_samples: int = 200,
        expected_cycle_time: float = 3600.0,
        export_path: Optional[Path] = None,
    ) -> "Pipeline":
        """Execute the full UQ pipeline and populate self.result.

        Uses the dataset's simulation config to resolve experiment_id and
        sim_base_path, then delegates to ``execute_pipeline``.

        Args:
            simulation_func: Callable with evaluate_batch(X) → Y.
            sim_data_path: Path(s) to ``simData.cPickle`` file(s). Falls back
                to the dataset's simulation config sim_data_path if not provided.
            observable_columns: Column names of observables to analyze.
            output_types: List of OutputType values to extract.
            generation_lower_bound: Skip initial generations (default: 2).
            time_lower_bound: Skip transient period in seconds (default: 100.0).
            n_bins: Number of cell cycle stage bins for Phase 2.
            polynomial_order: PCE polynomial order for both phases.
            n_samples: Number of LHS samples for PCE fitting.
            expected_cycle_time: Expected cell cycle period in seconds.
            export_path: If provided, export surrogates and results here.

        Returns:
            self, with self.result populated.
        """
        from libuq.pipeline.workflow import execute_pipeline

        if sim_data_path is None:
            sim_data_path = self.dataset.simulation.config.sim_data_path

        self.result = execute_pipeline(
            sim_data_path=sim_data_path,
            simulation_func=simulation_func,
            experiment_ids=self.dataset.simulation.config.experiment_id,
            sim_base_path=self.dataset.outdir_root,
            observable_columns=observable_columns,
            output_types=output_types,
            generation_lower_bound=generation_lower_bound,
            time_lower_bound=time_lower_bound,
            n_bins=n_bins,
            polynomial_order=polynomial_order,
            n_samples=n_samples,
            expected_cycle_time=expected_cycle_time,
            export_path=export_path,
        )
        return self
