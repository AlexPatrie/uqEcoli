import abc
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from uq.sensitivity import PCESurrogate


class MediaCondition(str, Enum):
    """Available media conditions for simulations."""

    BASAL = "basal"
    WITH_AA = "with_aa"
    ACETATE = "acetate"
    SUCCINATE = "succinate"
    NO_OXYGEN = "no_oxygen"


class CellCyclePhase(str, Enum):
    """Standard cell cycle phases for E. coli."""

    B_PERIOD = "B_period"  # Pre-initiation (birth to replication initiation)
    C_PERIOD = "C_period"  # DNA replication
    D_PERIOD = "D_period"  # Post-replication to division
    UNKNOWN = "unknown"


@dataclass
class BaseClass:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def export(self, f: Path) -> None:
        with open(f.__str__(), "w") as fp:
            json.dump(self.model_dump(), fp, indent=3)


@dataclass
class VioPathwayParams(BaseClass):
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
class MecillinamParams(BaseClass):
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
class UQInputParameters(BaseClass):
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


@dataclass
class Parameter(BaseClass):
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


# === PCE === #


@dataclass
class PCEWorkflowConfig(BaseClass, abc.ABC):
    @abc.abstractmethod
    def init_defaults(self) -> None:
        pass

    def __post_init__(self) -> None:
        self.init_defaults()


@dataclass
class PCEParameterSelectionConfig(PCEWorkflowConfig):
    """
    Used to configure logic that selects most relevant params from x,
    particularly configuring Morris.

    Attributes:
        n_trajectories: n trajectories used in morris prescreening for param selection.
        n_top: number of most relevant parameters to select and return
    """

    n_trajectories: int | None = None
    n_top: int | None = None  # how many params to keep

    def init_defaults(self) -> None:
        if self.n_trajectories is None:
            self.n_trajectories = 20
        if self.n_top is None:
            self.n_top = 5


@dataclass
class PCEPreprocessingConfig(PCEWorkflowConfig):
    """
    Used to configure sample preprocessing logic (X) for noisy sample execution.
    More reps ~ more noise offset

    Attributes:
        target_cv: target coefficient of variation (std/mean)
        min_reps: floor for n replicates (noise level accounting)
        max_reps: ceiling for n replicates (noise level accounting)
    """

    target_cv: float | None = None
    min_reps: int | None = None
    max_reps: int | None = None

    def init_defaults(self) -> None:
        if self.target_cv is None:
            self.target_cv: float = 0.05
        if self.min_reps is None:
            self.min_reps: int = 3
        if self.max_reps is None:
            self.max_reps: int = 20


@dataclass
class PCESolverConfig(PCEWorkflowConfig):
    """
    Used to configure the fitting process via PyTUQ.

    Attributes:
        basis_type: Polynomial basis type: 'legendre' (uniform inputs) or 'hermite' (Gaussian).
        method: Fitting method (PyTUQ regression):
            - 'least_squares': Standard least squares (default, PyTUQ 'lsq')
            - 'analytical': Full analytical solution (PyTUQ 'anl' with method='full')
            - 'variational': Variational inference (PyTUQ 'anl' with method='vi')
    """

    basis_type: Literal["legendre", "hermite"] | None = None
    method: Literal["least_squares", "analytical", "variational"] | None = None

    def init_defaults(self) -> None:
        if self.basis_type is None:
            self.basis_type = "legendre"
        if self.method is None:
            self.method = "least_squares"


@dataclass
class PCESurrogateConfig(PCEWorkflowConfig):
    """
    Used to parameterize surrogate instantiation workflow.

    Attributes:
        parameters: (`list[Parameter]`) parameters selected and returned from prescreening.
        n_samples: (`int`) Number of samples(perturbations/combos of vals for selected attributes of x), as
            afforded by fixed compute budget/resources.
        polynomial_order: (`int | None`) polynomial order used in calculation of n pce terms. Defaults to 2.
        n: len(parameters) convenience attr.
        N: alias for `n_samples` - consistent with literature.
        p: alias for `polynomial_order` - consistent with literature.
    """

    parameters: list[Parameter]  # selected in prescreening
    n_samples: int  # samples that can be afforded by fixed compute budget/resources
    polynomial_order: int | None = None

    def init_defaults(self) -> None:
        if self.polynomial_order is None:
            self.polynomial_order = 2

    @property
    def n(self) -> int:
        # n selected params from prescreen
        return len(self.parameters)

    @property
    def N(self) -> int:
        # alias for n_samples
        # # Given a parameterized function f(x) = y with n input parameters, the sample
        # size N (number of input configurations evaluated) must satisfy N ≥ 2 * C(n+p, p),
        # where p is the chosen polynomial order.
        return self.n_samples

    @property
    def p(self) -> int:
        # alias for poly order
        return self.polynomial_order

    def __post_init__(self):
        if self.polynomial_order is None:
            self.polynomial_order = self._calculate_p()
        if not self._validate_sample_size():
            warnings.warn(
                f"WARNING: Given n_parameters and polynomial order, "
                f"n_samples is too small with a value of {self.n_samples}"
            )

    def _calculate_p(self) -> int:
        n = self.n
        N = self.n_samples
        p = 1
        while 2 * math.comb(n + p + 1, p + 1) <= N:
            p += 1
        print(f"SUGGESTED POLYNOMIAL ORDER: {p}")
        return p

    def _validate_sample_size(self) -> int:
        """
        For PCE, given timeseries generator f(x) = y:

        :param n: number of selected attributes of x
        :param p: polynomial order
        :return: 2C(n+p, p) where C is the binomial coefficient function
        """
        return self.n_samples >= (2 * math.comb((self.n_samples + self.polynomial_order), self.polynomial_order))


@dataclass
class PCEConfig(BaseClass):
    """
    High-level, nested PCE configuration, attributed by domain/phase of pce workflows.

    Attributes:
        selection: PCEParameterSelectionConfig
        preprocessing: PCEPreprocessingConfig
        solver: PCESolverConfig
        surrogate: PCESurrogateConfig

    `>> selection (PCEParameterSelectionConfig)`:
        n_trajectories:;
        n_top: number of most relevant parameters to return

    `>> preprocessing (PCEPreprocessingConfig)`:
        target_cv: target coefficient of variation (std/mean);
        min_reps: floor for n replicates (noise level accounting);
        max_reps: ceiling for n replicates (noise level accounting)

    `solver (PCESolverConfig)`:
        basis_type: Polynomial basis type: 'legendre' (uniform inputs) or 'hermite' (Gaussian).;
        method: Fitting method:
            - 'least_squares': Standard least squares (default)
            - 'lasso': L1-regularized (sparse) via sklearn
            - 'omp': Orthogonal Matching Pursuit (sparse) via sklearn;
        lasso_alpha: Regularization strength for LASSO (only used if method='lasso').;
        omp_n_nonzero: Number of non-zero coefficients for OMP. If None, uses n_terms // 4.

    `surrogate (PCESurrogateConfig)`:
        parameters: (`list[Parameter]`) parameters selected and returned from prescreening.;
        n_samples: (`int`) Number of samples(perturbations/combos of vals for selected attributes of x), as
            afforded by fixed compute budget/resources.;
        polynomial_order: (`int | None`) polynomial order used in calculation of n pce terms. Defaults to 2.
        n: len(parameters) convenience attr.;
        N: alias for `n_samples` - consistent with literature.;
        p: alias for `polynomial_order` - consistent with literature.

    """

    selection: PCEParameterSelectionConfig
    preprocessing: PCEPreprocessingConfig
    solver: PCESolverConfig
    surrogate: PCESurrogateConfig


@dataclass
class PCEFitResult(BaseClass):
    """Result of fitting PCE coefficients from data.

    Carries both the serializable numpy arrays (for export) and
    an optional live PyTUQ PCE object (for efficient prediction).
    """

    coefficients: np.ndarray
    multi_indices: np.ndarray
    basis_type: str
    polynomial_order: int
    n_params: int
    r_squared: float
    n_samples: int
    method: str
    input_bounds: np.ndarray | None = None
    sparsity: float = field(init=False)

    def __post_init__(self):
        n_nonzero = np.sum(np.abs(self.coefficients) > 1e-10)
        self.sparsity = 1.0 - (n_nonzero / len(self.coefficients))
        # Live PyTUQ PCE object — not serialized, set via set_pytuq_pce()
        self._pytuq_pce = None

    def set_pytuq_pce(self, pce) -> None:
        """Attach the fitted PyTUQ PCE object for use in to_surrogate()."""
        self._pytuq_pce = pce

    def to_surrogate(self) -> "PCESurrogate":
        """Convert fit result to a PCESurrogate for prediction."""
        from uq.sensitivity import PCESurrogate

        surrogate = PCESurrogate(
            coefficients=self.coefficients,
            multi_indices=self.multi_indices,
            basis_type=self.basis_type,
            polynomial_order=self.polynomial_order,
            input_dim=self.n_params,
            output_dim=1,
            r_squared=self.r_squared,
            input_bounds=self.input_bounds,
        )
        if self._pytuq_pce is not None:
            surrogate.set_pytuq_pce(self._pytuq_pce)
        return surrogate
