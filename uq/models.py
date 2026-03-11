from dataclasses import asdict, dataclass

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

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


class CellCyclePhase(str, Enum):
    """Standard cell cycle phases for E. coli."""

    B_PERIOD = "B_period"  # Pre-initiation (birth to replication initiation)
    C_PERIOD = "C_period"  # DNA replication
    D_PERIOD = "D_period"  # Post-replication to division
    UNKNOWN = "unknown"


@dataclass
class CellCycleVariable:
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


@dataclass
class PrescreeningConfig:
    n_trajectories: int = 20
    n_top: int = 5  # how many params to keep


@dataclass
class Parameter:
    name: str
    bounds: tuple[float, float]
    default: float | int | complex
    step: float
    description: str

    def model_dump(self):
        d = asdict(self)
        bounds = tuple(self.bounds)
        d["bounds"] = bounds
        return d
