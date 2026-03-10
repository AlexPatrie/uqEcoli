"""
Input parameter definitions for uncertainty quantification.

This module defines the scientifically most relevant input variables for UQ:
- Violacein (vio) pathway presence
- Mecillinam condition
- Gene knockouts

These inputs are parametrized for use with UQPy/PyTUQ sensitivity analysis libraries.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import polars

if TYPE_CHECKING:
    pass


class MediaCondition(str, Enum):
    """Available media conditions for simulations."""

    BASAL = "basal"
    WITH_AA = "with_aa"
    ACETATE = "acetate"
    SUCCINATE = "succinate"
    NO_OXYGEN = "no_oxygen"


@dataclass
class VioPathwayParams:
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
    """

    enabled: bool = True
    induction_gen: int = 1
    knockout_gen: Optional[int] = None
    expression: float = 1.0
    translation_efficiency: float = 1.0
    rel_exp_adj_list: list[float] = field(default_factory=lambda: [1.0])
    rel_trl_eff_adj_list: list[float] = field(default_factory=lambda: [1.0])
    condition: MediaCondition = MediaCondition.BASAL

    def to_variant_params(self) -> dict[str, Any]:
        """
        Convert to variant parameter dictionary for new_gene_internal_shift_variable_strength.

        Returns:
            Dictionary compatible with apply_variant function
        """
        params: dict[str, Any] = {
            "condition": self.condition.value,
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
        if self.knockout_gen is not None:
            params["knockout_gen"] = self.knockout_gen
        return params


@dataclass
class MecillinamParams:
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

    def to_variant_params(self) -> dict[str, Any]:
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
class GeneKnockoutParams:
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
class UQInputParameters:
    """
    Complete set of input parameters for UQ analysis.

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

    def to_config_dict(self) -> dict[str, Any]:
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


class InputParameterSpace:
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
        vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
        vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
        mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
        include_vio: bool = True,
        include_mecillinam: bool = True,
        knockout_genes: Optional[list[str]] = None,
    ):
        """
        Initialize the input parameter space.

        Args:
            vio_expression_bounds: (min, max) bounds for vio expression factor
            vio_trl_eff_bounds: (min, max) bounds for vio translation efficiency
            mecillinam_conc_bounds: (min, max) bounds for mecillinam concentration
            include_vio: Whether to include vio parameters in the space
            include_mecillinam: Whether to include mecillinam parameters
            knockout_genes: List of genes that can be knocked out (discrete parameter)
        """
        self.parameter_names: list[str] = []
        self.parameter_bounds: list[tuple[float, float]] = []
        self.parameter_types: list[Literal["continuous", "discrete", "categorical"]] = []

        if include_vio:
            self.parameter_names.extend(["vio_expression", "vio_trl_eff"])
            self.parameter_bounds.extend([vio_expression_bounds, vio_trl_eff_bounds])
            self.parameter_types.extend(["continuous", "continuous"])

        if include_mecillinam:
            self.parameter_names.append("mecillinam_concentration")
            self.parameter_bounds.append(mecillinam_conc_bounds)
            self.parameter_types.append("continuous")

        self.knockout_genes = knockout_genes or []
        self._include_vio = include_vio
        self._include_mecillinam = include_mecillinam

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

    def sample_to_params(
        self,
        sample: np.ndarray,
        seed: int = 0,
        generations: int = 8,
        knockouts: Optional[list[str]] = None,
    ) -> UQInputParameters:
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
        params = UQInputParameters(seed=seed, generations=generations)

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

    def params_to_sample(self, params: UQInputParameters) -> np.ndarray:
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


def _get_repo_root() -> Path:
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
    experiment_id: str,
    outdir_root: Path | None = None,
    observables: list[str] | None = None,
) -> polars.DataFrame:
    """
    Load simulation dataset from parquet files.

    Args:
        experiment_id: The experiment identifier (e.g., "api_simulation_default")
        outdir_root: Root directory for simulation outputs. Defaults to
                     {repo_root}/api_integration/sims
        observables: Optional list of column names to select

    Returns:
        Polars DataFrame with the simulation data
    """
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"

    # Scan nested parquet files (variant/lineage_seed/generation/agent_id/*.pq)
    lf = polars.scan_parquet(str(base_path))
    if observables is not None:
        # Filter to only existing columns
        available = lf.collect_schema().names()
        valid_observables = [col for col in observables if col in available]
        if valid_observables:
            lf = lf.select(valid_observables)
    return lf.collect()


def get_available_columns(
    experiment_id: str,
    outdir_root: Path | None = None,
) -> list[str]:
    """
    Get list of available columns in the dataset.

    Args:
        experiment_id: The experiment identifier
        outdir_root: Root directory for simulation outputs. Defaults to
                     {repo_root}/api_integration/sims

    Returns:
        List of column names
    """
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"
    lf = polars.scan_parquet(str(base_path) + "/**/*.pq")
    return lf.collect_schema().names()
