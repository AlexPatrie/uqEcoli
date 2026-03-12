"""
PCE is a method where you strategically sample a tiny fraction of the input space to reconstruct (approximate)
    the function's behavior across the entire space — exploiting the
    structure provided by orthogonal polynomial bases in the underlying Hilbert space.

The Decision Framework
  ┌─────────────────────────────────────────────────┬──────────────────────────────────┐
  │         What You Know About Your System         │        Recommended Order         │
  ├─────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Nothing (default)                               │ 2                                │
  ├─────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Linear relationships dominate                   │ 1                                │
  ├─────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Quadratic effects / pairwise interactions       │ 2                                │
  ├─────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Strong nonlinearity / higher-order interactions │ 3                                │
  ├─────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Threshold effects / sharp transitions           │ 3-4 (but consider other methods) │
  └─────────────────────────────────────────────────┴──────────────────────────────────┘
  The Key Constraint: Samples vs. Order

  PCE requires enough samples to fit the polynomial. The number of terms grows combinatorially:

  Number of PCE terms = C(n + p, p) = (n + p)! / (n! × p!)

  where n = number of parameters, p = polynomial
  ┌──────────┬─────────┬─────────┬─────────┬─────────┐
  │ n params │ Order 1 │ Order 2 │ Order 3 │ Order 4 │
  ├──────────┼─────────┼─────────┼─────────┼─────────┤
  │ 3        │ 4       │ 10      │ 20      │ 35      │
  ├──────────┼─────────┼─────────┼─────────┼─────────┤
  │ 5        │ 6       │ 21      │ 56      │ 126     │
  ├──────────┼─────────┼─────────┼─────────┼─────────┤
  │ 10       │ 11      │ 66      │ 286     │ 1001    │
  ├──────────┼─────────┼─────────┼─────────┼─────────┤
  │ 20       │ 21      │ 231     │ 1771    │ 10626   │
  └──────────┴─────────┴─────────┴─────────┴─────────┘
  Rule of thumb: You need at least 2× the number of terms in samples.

  # For 3 params
  # Order 2: 10 terms → need ~20+ samples for stable fit
  # Order 3: 20 terms → need ~40+ samples
"""

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, Callable, Literal

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from scipy.linalg import lstsq
from scipy.stats import qmc

from uq.inputs import InputParameterSpace, InputParameterSpaceVecoli
from uq.models import (
    Parameter,
    PCEConfig,
    PCEFitResult,
    PCEParameterSelectionConfig,
    PCEPreprocessingConfig,
    PCESolverConfig,
    PCESurrogateConfig,
)
from uq.sensitivity import PCESurrogate, SensitivityAnalyzer


class FunctionWrapper:
    """Simple wrapper that adapts a callable to the wrapper interface expected by SensitivityAnalyzer."""

    def __init__(self, f: Callable[[np.ndarray], float | np.ndarray]):
        """
        Args:
            f: A function that takes a 1D parameter array and returns a scalar or array output.
        """
        self.f = f

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function for a batch of input samples.

        Args:
            X: Input parameter array of shape (n_samples, n_parameters)

        Returns:
            Output array of shape (n_samples,) or (n_samples, n_outputs)
        """
        results = []
        for x in X:
            result = self.f(x)
            results.append(result)
        return np.array(results)


def prescreen_parameters(
    full_space: InputParameterSpace,
    config: PCEParameterSelectionConfig | None = None,
    f: Callable[[np.ndarray], float | np.ndarray] | None = None,
) -> list[Parameter]:
    """
    Prescreen parameters using Morris sensitivity analysis.

    Args:
        full_space: The full input parameter space to screen.
        config: Configuration for prescreening (n_trajectories, n_top).
        f: Optional function to evaluate during Morris screening. If not provided,
           Morris analysis cannot be performed and will raise an error.

    Returns:
        List of the most influential parameters as Parameter objects.

    Prescreening attrs related to morris: n_trajectories, n_top (num selections)
    """
    wrapper = FunctionWrapper(f) if f is not None else None
    analyzer = SensitivityAnalyzer(full_space, wrapper=wrapper)
    conf = config or PCEParameterSelectionConfig()
    screening = analyzer.analyze_with_morris(n_trajectories=conf.n_trajectories)
    return screening.to_parameter_config(parameter_bounds=full_space.parameter_bounds, top_n=conf.n_top)


def prescreen_parameters_vecoli(
    vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
    vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
    mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    include_vio: bool = True,
    include_mecillinam: bool = True,
    knockout_genes: list[str] | None = None,
    prescreen_config: PCEParameterSelectionConfig | None = None,
):
    """
    Prescreening attrs related to morris: n_trajectories, n_top (num selections)
    """
    full_space = InputParameterSpaceVecoli(
        vio_expression_bounds=vio_expression_bounds,
        vio_trl_eff_bounds=vio_trl_eff_bounds,
        mecillinam_conc_bounds=mecillinam_conc_bounds,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
        knockout_genes=knockout_genes,
    )
    return prescreen_parameters(full_space=full_space, config=prescreen_config)


def generate_multi_indices(n_params: int, max_order: int) -> np.ndarray:
    """
    Generate multi-indices for PCE basis up to given order.

    Returns array where each row is [order_x1, order_x2, ...] and
    sum of each row <= max_order.
    """
    indices = []
    for total_order in range(max_order + 1):
        for combo in combinations_with_replacement(range(n_params), total_order):
            idx = [0] * n_params
            for i in combo:
                idx[i] += 1
            if idx not in indices:
                indices.append(idx)
    return np.array(indices)


def _evaluate_legendre(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate normalized Legendre polynomial of given order at x in [-1, 1]."""
    coeffs = [0] * order + [1]  # Coefficient vector for P_order
    return legval(x, coeffs)


def _evaluate_hermite(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate probabilist's Hermite polynomial of given order."""
    coeffs = [0] * order + [1]
    return hermeval(x, coeffs)


def _normalize_inputs(
    X: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Normalize inputs from [lb, ub] to [-1, 1] for Legendre basis."""
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return 2 * (X - lb) / (ub - lb) - 1


def _build_basis_matrix(
    X: np.ndarray,
    multi_indices: np.ndarray,
    basis_type: Literal["legendre", "hermite"] = "legendre",
) -> np.ndarray:
    """
    Build the design matrix (basis matrix) for PCE regression.

    Parameters
    ----------
    X : np.ndarray
        Input samples, shape (n_samples, n_params). Should be normalized to
        [-1, 1] for Legendre or standardized for Hermite.
    multi_indices : np.ndarray
        Multi-index array, shape (n_terms, n_params).
    basis_type : str
        Type of polynomial basis ('legendre' or 'hermite').

    Returns
    -------
    np.ndarray
        Basis matrix of shape (n_samples, n_terms).
    """
    n_samples = X.shape[0]
    n_terms = multi_indices.shape[0]
    n_params = X.shape[1]

    eval_func = _evaluate_legendre if basis_type == "legendre" else _evaluate_hermite

    # Precompute univariate polynomials for all orders we need
    max_order = int(multi_indices.max())
    univariate = np.zeros((n_samples, n_params, max_order + 1))
    for p in range(n_params):
        for order in range(max_order + 1):
            univariate[:, p, order] = eval_func(X[:, p], order)

    # Build multivariate basis by taking products
    basis_matrix = np.ones((n_samples, n_terms))
    for t, idx in enumerate(multi_indices):
        for p, order in enumerate(idx):
            if order > 0:
                basis_matrix[:, t] *= univariate[:, p, order]

    return basis_matrix


def fit_pce_coefficients(
    X: np.ndarray,
    Y: np.ndarray,
    polynomial_order: int,
    bounds: np.ndarray | None = None,
    basis_type: Literal["legendre", "hermite"] = "legendre",
    method: Literal["least_squares", "lasso", "omp"] = "least_squares",
    lasso_alpha: float = 0.01,
    omp_n_nonzero: int | None = None,
) -> PCEFitResult:
    """
    Fit PCE coefficients from sample data.

    Given N input-output pairs (X, Y), fit the polynomial chaos expansion
    coefficients using regression.

    Parameters
    ----------
    X : np.ndarray
        Input samples, shape (n_samples, n_params).
    Y : np.ndarray
        Output values, shape (n_samples,) or (n_samples, 1).
    polynomial_order : int
        Maximum total polynomial order (p).
    bounds : np.ndarray, optional
        Parameter bounds, shape (n_params, 2). Required for Legendre basis
        to normalize inputs to [-1, 1]. If None, assumes X is already normalized.
    basis_type : str
        Polynomial basis type: 'legendre' (uniform inputs) or 'hermite' (Gaussian).
    method : str
        Fitting method:
        - 'least_squares': Standard least squares (default)
        - 'lasso': L1-regularized (sparse) via sklearn
        - 'omp': Orthogonal Matching Pursuit (sparse) via sklearn
    lasso_alpha : float
        Regularization strength for LASSO (only used if method='lasso').
    omp_n_nonzero : int, optional
        Number of non-zero coefficients for OMP. If None, uses n_terms // 4.

    Returns
    -------
    PCEFitResult
        Fitted coefficients, multi-indices, and metadata.

    Examples
    --------
    >>> # Generate sample data
    >>> X = np.random.uniform(-1, 1, (100, 3))  # 100 samples, 3 params
    >>> Y = X[:, 0] + 0.5 * X[:, 1]**2 + 0.1 * X[:, 0] * X[:, 2]  # true function
    >>>
    >>> # Fit PCE
    >>> result = fit_pce_coefficients(X, Y, polynomial_order=2)
    >>> print(f"R² = {result.r_squared:.4f}")
    >>> print(f"Sparsity = {result.sparsity:.1%}")
    >>>
    >>> # Use for prediction
    >>> surrogate = result.to_surrogate()
    >>> Y_pred = surrogate.predict(X_new)
    """
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y).ravel()

    n_samples, n_params = X.shape

    if len(Y) != n_samples:
        raise ValueError(f"X has {n_samples} samples but Y has {len(Y)}")

    # Generate multi-indices
    multi_indices = generate_multi_indices(n_params, polynomial_order)
    n_terms = len(multi_indices)

    # Check sample size
    min_samples = 2 * n_terms
    if n_samples < n_terms:
        raise ValueError(
            f"Underdetermined system: {n_samples} samples < {n_terms} terms. "
            f"Need at least {n_terms} samples (recommend {min_samples})."
        )
    if n_samples < min_samples:
        warnings.warn(
            f"Sample size {n_samples} is less than recommended {min_samples} "
            f"(2× the {n_terms} PCE terms). Fit may be unstable."
        )

    # Normalize inputs if bounds provided
    if bounds is not None:
        bounds = np.atleast_2d(bounds)
        X_norm = _normalize_inputs(X, bounds)
    else:
        X_norm = X

    # Build basis matrix
    Phi = _build_basis_matrix(X_norm, multi_indices, basis_type)

    # Fit coefficients based on method
    if method == "least_squares":
        coefficients, residuals, rank, s = lstsq(Phi, Y)
    elif method == "lasso":
        try:
            from sklearn.linear_model import Lasso
        except ImportError:
            raise ImportError("sklearn required for LASSO. Install with: pip install scikit-learn")
        lasso = Lasso(alpha=lasso_alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(Phi, Y)
        coefficients = lasso.coef_
    elif method == "omp":
        try:
            from sklearn.linear_model import OrthogonalMatchingPursuit
        except ImportError:
            raise ImportError("sklearn required for OMP. Install with: pip install scikit-learn")
        n_nonzero = omp_n_nonzero or max(1, n_terms // 4)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
        omp.fit(Phi, Y)
        coefficients = omp.coef_
    else:
        raise ValueError(f"Unknown method: {method}. Use 'least_squares', 'lasso', or 'omp'.")

    # Compute R²
    Y_pred = Phi @ coefficients
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return PCEFitResult(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type=basis_type,
        polynomial_order=polynomial_order,
        n_params=n_params,
        r_squared=r_squared,
        n_samples=n_samples,
        method=method,
        input_bounds=bounds,
    )


def generate_synthetic_coefficients(
    multi_indices: np.ndarray,
    seed: int = 123,
) -> np.ndarray:
    """
    Generate synthetic PCE coefficients with realistic structure.

    - Constant term: ~1.5 (baseline)
    - Linear terms: larger coefficients (main effects)
    - Higher-order terms: smaller coefficients
    - Some negative coefficients for variety
    """
    np.random.seed(seed)
    n_terms = len(multi_indices)
    coefficients = np.zeros(n_terms)

    for i, idx in enumerate(multi_indices):
        total_order = sum(idx)
        if total_order == 0:
            # Constant term
            coefficients[i] = 1.5
        elif total_order == 1:
            # Linear terms: larger, some positive, some negative
            coefficients[i] = np.random.uniform(0.2, 0.8) * np.random.choice([1, -1], p=[0.7, 0.3])
        else:
            # Higher-order terms: smaller
            coefficients[i] = np.random.uniform(0.05, 0.2) / total_order * np.random.choice([1, -1])

    return coefficients


def compute_slider_step(bounds: list[float], n_steps: int = 50) -> float:
    """Compute a reasonable step size for a slider given bounds."""
    range_size = bounds[1] - bounds[0]
    step = range_size / n_steps
    # Round to nice values
    magnitude = 10 ** np.floor(np.log10(step))
    normalized = step / magnitude
    if normalized < 1.5:
        nice_step = 1
    elif normalized < 3.5:
        nice_step = 2
    elif normalized < 7.5:
        nice_step = 5
    else:
        nice_step = 10
    return nice_step * magnitude


# def get_pce_config(
#     prescreened: list[Parameter], sample_size: int, pce_poly_order: int | None = None
# ) -> PCESurrogateConfig:
#     return PCESurrogateConfig(parameters=prescreened, n_samples=sample_size, polynomial_order=pce_poly_order)


def get_pce_config(prescreened: list[Parameter], sample_size: int, **kwargs) -> PCEConfig:
    """
    :param prescreened: selected params from prescreening phase (Morris)
    :param sample_size: number of samples/perturbations/variations of `prescreened` to create.
    :param kwargs: Dict whose outermost keys are: selection, preprocessing, solver, and surrogate, and
        whose values correspond to kwargs accepted by that relevant PCE config. See `PCEConfig` for
        full details.
    """
    selection, preproc, solver, surrogate = {}, {}, {}, {}
    for keyword in kwargs:
        if keyword in ["n_trajectories", "n_top"]:
            selection[keyword] = kwargs[keyword]
        elif keyword in ["target_cv", "min_reps", "max_reps"]:
            preproc[keyword] = kwargs[keyword]
        elif keyword in ["basis_type", "method", "lasso_alpha", "omp_n_nonzero"]:
            solver[keyword] = kwargs[keyword]
        elif keyword in ["polynomial_order", "p"]:
            surrogate["polynomial_order"] = kwargs[keyword]
    return PCEConfig(
        selection=PCEParameterSelectionConfig(**selection),
        preprocessing=PCEPreprocessingConfig(**preproc),
        solver=PCESolverConfig(**solver),
        surrogate=PCESurrogateConfig(parameters=prescreened, n_samples=sample_size, **surrogate),
    )


# def get_pce_config(
#     prescreened: list[Parameter],
#     sample_size: int,
#     pce_poly_order: int | None = None
# ) -> PCEConfig:
#     PCEConfig(
#         selection=PCEParameterSelectionConfig(),
#
#     )
#     return PCESurrogateConfig(parameters=prescreened, n_samples=sample_size, polynomial_order=pce_poly_order)


def create_samples(N: int, selected: list[Parameter]) -> np.ndarray:
    """
    Generate N perturbations/combinations of selected using Latin Hypercube Sampling.

    Why LHS is better:

      1. Stratified — divides each dimension into N equal bins, one sample per bin
      2. No gaps — guarantees coverage across full range
      3. Efficient — better accuracy with same N vs random
      4. Standard — widely used in UQ, well-understood properties

      ---
      Rule of thumb:
      ┌───────────────┬────────────────────────────────┐
      │ Sample budget │            Strategy            │
      ├───────────────┼────────────────────────────────┤
      │ N < 50        │ Sobol sequence (best coverage) │
      ├───────────────┼────────────────────────────────┤
      │ N ≥ 50        │ LHS (good balance)             │
      ├───────────────┼────────────────────────────────┤
      │ Don't care    │ Random uniform (simplest)      │
      └───────────────┴────────────────────────────────┘
      For PCE fitting, LHS is the pragmatic default.
    """
    # Generate LHS samples in [0, 1]^n
    n = len(selected)
    sampler = qmc.LatinHypercube(d=n)
    X_unit = sampler.random(n=N)
    # Scale to parameter bounds
    bounds = np.array([p.bounds for p in selected])
    return qmc.scale(X_unit, bounds[:, 0], bounds[:, 1])


def process_samples_simple(X: np.ndarray, f: Callable[[np.ndarray], np.ndarray], n_replicates: int = 5):
    """
    Why median over mean:
      - Robust to outliers / rare extreme realizations
      - Better for heavy-tailed noise
      - Works well even if noise is small (converges to mean)

    :param X: array of samples (perturbation combos of selected attributes of x) with shape (n_samples, n_params)
    :param f: stochastic timeseries generator (think of this as the simulation func)
    :param n_replicates: noise level for which stochasticity is accounted.
    """
    return np.array([np.median([f(x) for _ in range(n_replicates)]) for x in X])


def process_samples_adaptive(
    X: np.ndarray, f: Callable[[np.ndarray], np.ndarray], target_cv: float = 0.05, min_reps=3, max_reps=20
) -> np.ndarray:
    """
    Adaptively choose replicates based on noise level.

    :param X: array of samples (perturbation combos of selected attributes of x) with shape (n_samples, n_params)
    :param f: stochastic timeseries generator (think of this as the simulation func)
    :param target_cv: target coefficient of variation (std/mean)
    :param min_reps: floor for n replicates (noise level accounting)
    :param max_reps: ceiling for n replicates (noise level accounting)
    """
    Y = np.zeros(len(X))

    for i, x in enumerate(X):
        reps = []
        for r in range(max_reps):
            reps.append(f(x))
            if r >= min_reps - 1:
                cv = np.std(reps) / (np.abs(np.mean(reps)) + 1e-10)
                if cv < target_cv:
                    break
        Y[i] = np.median(reps)

    return Y


def process_samples(
    X: np.ndarray, f: Callable[[np.ndarray], np.ndarray], method: Literal["simple", "adaptive"] = "adaptive", **kwargs
) -> np.ndarray:
    """
    :param X: array of samples (perturbation combos of selected attributes of x) with shape (n_samples, n_params)
    :param f: stochastic timeseries generator (think of this as the simulation func)
    :param method: One of simple; adaptive(Adaptively choose replicates based on noise level). Defaults to adaptive.
    :param kwargs: if using method='adaptive', kwargs are:
        target_cv: float = 0.05,
        min_reps=3,
        max_reps=20, if method='simple', kwargs are n_replicates
    """
    processor = (
        process_samples_adaptive if method == "adaptive" else process_samples_simple if method == "simple" else None
    )
    if processor is None:
        raise ValueError(f"Not a valid method type. Expected one of adaptive;simple, got: {method}")
    return processor(X, f, **kwargs)


def generate_surrogate(
    space: InputParameterSpace,
    generator: Callable[[np.ndarray], np.ndarray],
    sample_size: int,
    prescreening_config: PCEParameterSelectionConfig | None = None,
    **kwargs,
    # basis_type: Literal["legendre", "hermite"] = "legendre",
    # method: Literal["least_squares", "lasso", "omp"] = "least_squares",
    # lasso_alpha: float = 0.01,
    # omp_n_nonzero: int | None = None,
    # target_cv: float = 0.05,
    # min_reps: int = 3,
    # max_reps: int = 20,
) -> PCESurrogate:
    """
    Generate a PCE surrogate. You can then use it like:
            ```
            surrogate = generate_pce_surrogate(...)
            Y_pred = surrogate.predict(X_new)
            ```
    :param space: (InputParameterSpace) Descriptive definition of input parameter space.
    :param generator: (Callable) stochastic timeseries generator (think of this as the simulation func)
    :param sample_size: number of perturbations (as afforded by your computational env/budget).
    :param prescreening_config:
    :param kwargs: Flat definition of kwargs consisting of atomic items from:
        `>> selection (PCEParameterSelectionConfig)`:
            n_trajectories:;
            n_top: number of most relevant parameters to return

        `>> preprocessing (PCEPreprocessingConfig)`:
            target_cv: target coefficient of variation (std/mean);
            min_reps: floor for n replicates (noise level accounting);
            max_reps: ceiling for n replicates (noise level accounting)

        `solver (PCESolverConfig)`:
            basis_type: Polynomial basis type: 'legendre' (uniform inputs) or 'hermite' (Gaussian).;
            method: Fitting method: - 'least_squares': Standard least squares (default)
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
    # prescreen to find most relevant params
    selected = prescreen_parameters(full_space=space, config=prescreening_config, f=generator)
    n_params = len(selected)
    if kwargs.get("n_top") is None:
        kwargs["n_top"] = min(20, max(3, int(math.ceil(math.sqrt(n_params) * 1.5))))  # noqa: RUF046

    # extract/set up/configure for PCE
    config: PCEConfig = get_pce_config(prescreened=selected, sample_size=sample_size, **kwargs)
    param_bounds = np.array([p.bounds for p in selected])
    param_defaults = np.array([
        p.get("default", (p["bounds"][0] + p["bounds"][1]) / 2)
        for p in [param.model_dump() for param in config.surrogate.parameters]
    ])

    # generate sample_size perturbations/combos of selected (X) and run them through f (Y)
    X = create_samples(N=sample_size, selected=selected)
    Y = process_samples(
        X=X,
        f=generator,
        target_cv=config.preprocessing.target_cv,
        min_reps=config.preprocessing.min_reps,
        max_reps=config.preprocessing.max_reps,
    )

    # generate pce coeffs from fitting
    fitting = fit_pce_coefficients(
        X=X,
        Y=Y,
        polynomial_order=config.surrogate.p,
        bounds=param_bounds,
        basis_type=config.solver.basis_type,
        method=config.solver.method,
        lasso_alpha=config.solver.lasso_alpha,
        omp_n_nonzero=config.solver.omp_n_nonzero,
    )
    pce = fitting.to_surrogate()
    print("PCE Surrogate created from config:")
    print(f"  - Parameters: {space.parameter_names}")
    print(f"  - Bounds: {space.parameter_bounds}")
    print(f"  - Defaults: {param_defaults}")
    print(f"  - Polynomial order: {config.surrogate.p}")
    print(f"  - Number of PCE terms: {len(fitting.coefficients)}")

    return pce
