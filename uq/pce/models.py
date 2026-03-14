# === PCE === #

import abc
import math
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from uq.common.models import BaseClass


@dataclass
class Parameter(BaseClass):
    """
    Input parameter for UQ pipeline extracted from sim_data.

    Attributes:
        name: str
        bounds: tuple[float, float] (low, high)
        step: float - granularity of control allowed for ui element
            range control (knob, slider, etc). TODO: move this.
        description: str
    """
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