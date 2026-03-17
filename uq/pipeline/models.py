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
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from uq.common.models import BaseClass
from uq.io import get_bucket
from uq.pce.models import Parameter
from uq.synthetic import generate_synthetic_simulation_data

if TYPE_CHECKING:
    from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices, PCESurrogate, SobolIndices


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


@dataclass
class VioPathwayParams(VecoliParams):
    """
    Parameters for violacein (vio) pathway presence.

    The vio pathway is a new gene that can be induced at specific generations
    with controllable expression and translation efficiency.

    Attributes:
        enabled: Whether the vio pathway is present
        induction_gen: Generation at which to induce new gene expression
        knockout_gen: Generation to knock out new gene expression (optional)
        expression: Factor by which to multiply new gene expression once induced
        translation_efficiency: Translation efficiency for new gene once induced
        rel_exp_adj_list: List of relative expression adjustments per gene
        rel_trl_eff_adj_list: List of relative translation efficiency adjustments
        condition: Environmental condition (basal, with_aa, etc.)
        variants: list[VariantConfig]
    """

    enabled: bool = True
    induction_gen: int = 1
    knockout_gen: Optional[int] = None
    expression: float = 1.0
    translation_efficiency: float = 1.0
    rel_exp_adj_list: list[float] = field(default_factory=lambda: [1.0])
    rel_trl_eff_adj_list: list[float] = field(default_factory=lambda: [1.0])
    condition: MediaCondition | str = MediaCondition.BASAL

    def __post_init__(self):
        pass

    def get_default_perturbations(self):
        d = {
            "condition": self.condition,
            "induction_gen": self.induction_gen,
            "exp_trl_eff": {
                "exp": self.expression,
                "trl_eff": self.translation_efficiency,
            },
            "rel_adj": {
                "rel_exp_adj_list": self.rel_exp_adj_list,
                "rel_trl_eff_adj_list": self.rel_trl_eff_adj_list,
            },
        }
        return d

    def to_variant_params(self, pert_factor: float = 0.22) -> dict[str, Any]:
        """
        Convert to variant parameter dictionary for new_gene_internal_shift_variable_strength.

        Returns:
            Dictionary compatible with apply_variant function
        """
        # params = self._parse_variants(**deltas) or self.default_perturbation()
        # params = {}
        # if deltas is not None:
        #     for attr, d_v in deltas.items():
        #         original = getattr(self, attr)
        #         params[attr] = original - d_v if not isinstance(d_v, Callable) else d_v(original)
        params = self.default_perturbations
        if self.knockout_gen is not None:
            params["knockout_gen"] = self.knockout_gen
        return params


@dataclass
class MecillinamParams(VecoliParams):
    """
    Parameters for mecillinam antibiotic condition.

    Mecillinam is a beta-lactam antibiotic that inhibits PBP2 (penicillin-binding
    protein 2), affecting cell wall synthesis and cell shape.

    Attributes:
        times: Times at which to change mecillinam concentration (seconds)
        concentrations: Mecillinam concentrations at each time point (mM)
        knockouts: Gene IDs for which to knock out translation
    """

    times: list[float] = field(default_factory=lambda: [0.0])
    concentrations: list[float] = field(default_factory=lambda: [0.0])
    knockouts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.times) != len(self.concentrations):
            raise ValueError("times and concentrations must have the same length")

    def get_default_perturbations(self) -> dict[str, Any]:
        """
        Convert to variant parameter dictionary for mecillinam_timeline.

        Returns:
            Dictionary compatible with apply_variant function
        """
        return {
            "times": self.times,
            "concentrations": self.concentrations,
            "knockouts": self.knockouts,
        }


@dataclass
class GeneKnockoutParams(BaseClass):
    """
    Parameters for gene knockout conditions.

    Gene knockouts can be applied at the ParCa level (gene_deletions) or
    at the translation level (translation efficiency = 0).

    Attributes:
        gene_deletions: List of gene IDs to delete at ParCa level
        translation_knockouts: List of gene IDs to knock out at translation level
    """

    gene_deletions: list[str] = field(default_factory=list)
    translation_knockouts: list[str] = field(default_factory=list)


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
    id: Literal["new_gene_internal_shift_variable_strength", "condition", "mecillinam_timeline"]
    config: dict[str, Any]


@dataclass
class SimulationConfigVecoli(SimulationConfig):
    """Vecoli simulation config (JSON), 1:1"""

    experiment_id: str
    sim_data_path: str
    n_init_sims: int = field(default=1)
    generations: int = field(default=1)
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
        attrs = [self.experiment_id, self.sim_data_path]
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


@dataclass
class UQInputParametersVecoli(UQInputParameters):
    """
    Complete set of vEcoli input parameters for UQ analysis.

    This combines all input parameter types into a single container that can
    be used to parametrize simulation runs for sensitivity analysis.

    Attributes:
        vio: Violacein pathway parameters
        mecillinam: Mecillinam antibiotic parameters
        knockouts: Gene knockout parameters
        seed: Random seed for the simulation
        generations: Number of generations to simulate
        condition: Base media condition (if not set by vio)
    """

    vio: VioPathwayParams = field(default_factory=VioPathwayParams)
    mecillinam: MecillinamParams = field(default_factory=MecillinamParams)
    knockouts: GeneKnockoutParams = field(default_factory=GeneKnockoutParams)
    seed: int = 0
    generations: int = 8
    condition: MediaCondition = MediaCondition.BASAL

    def to_simulation_config(self) -> SimulationConfig:
        """
        Convert to configuration dictionary for EcoliSim.

        Returns:
            Dictionary that can be used to configure a simulation
        """
        config: dict[str, Any] = {
            "seed": self.seed,
            "generations": self.generations,
        }

        # Add variants based on which parameters are active
        variants: dict[str, list[dict[str, Any]]] = {}

        if self.vio.enabled:
            variants["new_gene_internal_shift_variable_strength"] = [self.vio.to_variant_params()]
        else:
            # Just set the condition if vio is not enabled
            variants["condition"] = [{"condition": self.condition.value}]

        if any(self.mecillinam.concentrations):
            variants["mecillinam_timeline"] = [self.mecillinam.to_variant_params()]

        if variants:
            config["variants"] = variants

        return config


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

    def export(self, path: str | Path) -> None:
        """Serialize the full pipeline result to disk.

        Creates:
            path/population_surrogate/   — PCESurrogate export
            path/cell_cycle_surrogate/   — PCESurrogate export
            path/population_sobol/       — DataclassIO export
            path/cell_cycle_sobol_stage_N/ — DataclassIO export per stage
            path/variance_decomposition.json — Step 4 fractions
            path/metadata.json           — Pipeline metadata
        """
        from uq.io import DataclassIO

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

    @classmethod
    def from_export(cls, path: str | Path) -> "PipelineResult":
        """Load a PipelineResult from a previously exported directory."""
        from uq.io import DataclassIO
        from uq.sensitivity import MorrisIndices, PCESurrogate, SobolIndices

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
        from uq.pipeline.workflow import execute_pipeline

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
