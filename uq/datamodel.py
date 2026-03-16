"""
Input parameter definitions for uncertainty quantification.

This module defines the scientifically most relevant input variables for UQ:
- Violacein (vio) pathway presence
- Mecillinam condition
- Gene knockouts

These inputs are parametrized for use with UQPy/PyTUQ sensitivity analysis libraries.
"""

import abc
import pprint
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, override

import numpy as np
import polars
from bs4.element import XMLAttributeDict
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq.common import BaseClass

if TYPE_CHECKING:
    pass


console = Console()


@dataclass
class GeneratorParameter(BaseClass):
    """
    Stochastic timeseries generator parameter, (attribute of x in f(x) -> y dataset).
    """
    name: str
    bounds: tuple[float, float] | tuple[complex, complex] = field(default_factory=list)
    default: float | int | complex | None = None
    value: float | int | complex | None = None
    granularity_step: float = 0.25
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.bounds:
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
class VariantConfig(BaseClass):
    attribute_id: str  # if SimData were a dataframe, this is one of the cols
    value: float | int | complex = None  # not implicitly delta, rather manually set val
    delta: float | int | complex | None = None

    @classmethod
    def from_delta(cls, p: GeneratorParameter, delta: float | int | complex) -> "VariantConfig":
        variant = VariantConfig(attribute_id=p.name, value=p.value - delta)
        variant.delta = delta
        return variant


# TODO: make a to_param() -> Param method!


@dataclass
class GeneratorParameters(BaseClass):
    values: list[GeneratorParameter] = field(default_factory=list)
    variants: list[VariantConfig] = field(default_factory=list)

    def apply_perturbations(
        self, selected: dict[str, float | int | complex], factor: float
    ) -> dict[str, float | complex | int]:
        return {key: val * type(factor) for key, val in selected.items()}

    # @abc.abstractmethod
    def get_default_perturbations(self):
        return {}

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


# TODO: convert to XAttribute[]
@dataclass
class VioPathway(BaseClass):
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

    def to_generator_params(self, variants: list[VariantConfig] | None = None):
        params = list(map(
            lambda p: GeneratorParameter(name=p, value=getattr(self, p)),
            ['enabled', etc]
        ))
        return GeneratorParameters(
            values=params, variants=variants or []
        )

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


@dataclass
class UQInputs:
    """

    """
    # params:

    @abc.abstractmethod
    def to_simulation_config(self, *args, **kwargs) -> SimulationConfig:
        pass


@dataclass
class UQInputParametersVecoli(UQInputs):
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

        return SimulationConfigVecoli


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



class XSpaceInterface(abc.ABC):
    """
    Interface whose implementations fulfill:
        - sample_to_params() -> UQInputParameters: Convert a sample from the model/simulation/function-specific parameter space to UQInputParameters.
        - params_to_sample() -> np.ndarray[ParameterValue]: Convert domain-specific input space-mapped UQInputParameters to an array of sample(perturbation) values 
            when N sample values (sample size) == N perturbations.
    """

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs) -> UQInputs:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            **kwargs: implementation-specific

        Returns:
            UQInputParameters instance
        """
        pass

    @abc.abstractmethod
    def params_to_sample(self, params: UQInputs) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        pass


class XSpace(XSpaceInterface):
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]
    experiment_id: str | None
    """
    Defines the parameter space for UQ sensitivity analysis.

    This class provides methods to sample input parameters and to convert
    between the UQ library format (numpy arrays) and UQInputParameters.

    Attributes:
        parameter_names: Names of the parameters being varied
        parameter_bounds: Lower and upper bounds for each parameter
        parameter_types: Type of each parameter ('continuous', 'discrete', 'categorical')
    """

    def __init__(
        self,
        *args,
        parameter_names: list[str] | None = None,
        parameter_bounds: list[tuple[float, float]] | None = None,
        parameter_types: list[Literal["continuous", "discrete", "categorical"]] | None = None,
        experiment_id: str | None = None,
        **kwargs,
    ) -> None:
        self.parameter_names = parameter_names or []
        self.parameter_bounds = parameter_bounds or []
        self.parameter_types = parameter_types or []
        self.experiment_id = experiment_id

        self.implementation_init(*args, **kwargs)

    def implementation_init(self, *args, **kwargs) -> None:
        # self.define_parameters()
        return None

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the space."""
        return len(self.parameter_names)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Parameter bounds as numpy array for UQPy.

        Returns:
            Array of shape (n_parameters, 2) with [lower, upper] bounds
        """
        return np.array(self.parameter_bounds)

    # @abc.abstractmethod
    # def define_parameters(self, *args, **kwargs) -> list[Parameter]:
    #     pass

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs) -> UQInputs:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            **kwargs: implementation-specific

        Returns:
            UQInputParameters instance
        """
        pass

    @abc.abstractmethod
    def params_to_sample(self, params: UQInputParametersVecoli) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        pass

    def get_uqpy_distributions(self) -> list[Any]:
        """
        Get UQPy distribution objects for this parameter space.

        Returns:
            List of UQPy Distribution objects (Uniform distributions)
        """
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get PyTUQ-compatible bounds arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]

    @property
    def parameters(self) -> dict[str, dict[list[float], str]]:
        params = {}
        # TODO: enable ragged shape, for now symmetry required for construction
        if all(
            list(
                map(
                    lambda c: len(c) > 0 and len(c) == len(self.parameter_names),
                    [self.parameter_bounds, self.parameter_names, self.parameter_types],
                )
            )
        ):
            for i, name in enumerate(self.parameter_names):
                params[name] = {"bounds": self.parameter_bounds[i], "type": self.parameter_types[i]}
        return params

    def __repr__(self) -> str:
        self.show()
        return f"<ParameterSpace '{self.experiment_id}' n={self.n_parameters}>"

    def show(self):
        # Neon 90s header
        title = Text()
        title.append("⚡ ", style="bold yellow")
        title.append("INPUT PARAMETER SPACE", style="bold magenta")
        title.append("  //  ", style="dim cyan")
        title.append(self.experiment_id, style="bold cyan")

        # Parameter table
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            pad_edge=False,
        )
        table.add_column("PARAM", style="bold yellow", no_wrap=True)
        table.add_column("VALUE", style="bright_white")
        table.add_column("TYPE", style="dim cyan")

        for name, val in self.parameters.items():
            table.add_row(name, str(val), type(val).__name__)

        panel = Panel(
            table,
            title=title,
            subtitle=Text(f"n = {self.n_parameters} parameters", style="bold green"),
            border_style="magenta",
            box=box.DOUBLE_EDGE,
            padding=(0, 1),
        )

        console.print(panel)


class XSpaceVecoli(XSpaceInterface):
    """
    Defines the parameter space for UQ sensitivity analysis pipeline
    on vEcoli datasets.

    This class provides methods to sample input parameters and to convert
    between the UQ library format (numpy arrays) and UQInputParameters.

    Kwargs:
        `vio_expression_bounds: tuple[float, float] = (low, high)`
        `vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0)`
        `mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0)`
        `include_vio: bool = True`
        `include_mecillinam: bool = True`
        `knockout_genes: Optional[list[str]] = None`
    Attributes:
        parameter_names: Names of the parameters being varied
        parameter_bounds: Lower and upper bounds for each parameter
        parameter_types: Type of each parameter ('continuous', 'discrete', 'categorical')
    """

    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]

    def implementation_init(self, *args, **kwargs) -> None:
        if kwargs.get("include_vio"):
            self.parameter_names.extend(["vio_expression", "vio_trl_eff"])
            self.parameter_bounds.extend([
                kwargs.get("vio_expression_bounds", (0.0, 5.0)),
                kwargs.get("vio_trl_eff_bounds", (0.0, 2.0)),
            ])
            self.parameter_types.extend(["continuous", "continuous"])

        if kwargs.get("include_mecillinam"):
            self.parameter_names.append("mecillinam_concentration")
            self.parameter_bounds.append(kwargs.get("mecillinam_conc_bounds", (0.0, 10.0)))
            self.parameter_types.append("continuous")

        self.knockout_genes = kwargs.get("knockout_genes", [])
        self._include_vio = kwargs.get("include_vio", False)
        self._include_mecillinam = kwargs.get("include_mecillinam", False)

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the space."""
        return len(self.parameter_names)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Parameter bounds as numpy array for UQPy.

        Returns:
            Array of shape (n_parameters, 2) with [lower, upper] bounds
        """
        return np.array(self.parameter_bounds)

    @override
    def sample_to_params(
        self,
        sample: np.ndarray,
        seed: int = 0,
        generations: int = 8,
        knockouts: Optional[list[str]] = None,
    ) -> UQInputParametersVecoli:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            seed: Random seed for the simulation
            generations: Number of generations to simulate
            knockouts: List of genes to knock out (optional)

        Returns:
            UQInputParameters instance
        """
        params = UQInputParametersVecoli(seed=seed, generations=generations)

        idx = 0
        if self._include_vio:
            params.vio.expression = float(sample[idx])
            params.vio.translation_efficiency = float(sample[idx + 1])
            params.vio.enabled = True
            idx += 2
        else:
            params.vio.enabled = False

        if self._include_mecillinam:
            # Apply concentration at time 0
            params.mecillinam.times = [0.0]
            params.mecillinam.concentrations = [float(sample[idx])]
            idx += 1

        if knockouts:
            params.knockouts.translation_knockouts = knockouts

        return params

    @override
    def params_to_sample(self, params: UQInputParametersVecoli) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        sample: list[float] = []

        if self._include_vio:
            sample.append(params.vio.expression)
            sample.append(params.vio.translation_efficiency)

        if self._include_mecillinam:
            # Use the first concentration value
            conc = params.mecillinam.concentrations[0] if params.mecillinam.concentrations else 0.0
            sample.append(conc)

        return np.array(sample)

    def get_uqpy_distributions(self) -> list[Any]:
        """
        Get UQPy distribution objects for this parameter space.

        Returns:
            List of UQPy Distribution objects (Uniform distributions)
        """
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get PyTUQ-compatible bounds arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]



