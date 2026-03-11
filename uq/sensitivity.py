"""
Global sensitivity analysis for uncertainty quantification.

This module implements global sensitivity analysis methods using PCE
(Polynomial Chaos Expansion) surrogate models, as specified in the UQ framework.

The implementation supports both UQPy and PyTUQ libraries for:
- Building PCE surrogate models from simulation data
- Computing Sobol sensitivity indices
- Analyzing parameter importance across different outputs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from uq.aggregation import AggregatedOutput, AggregationStrategy
from uq.inputs import InputParameterSpace
from uq.wrappers import PrecomputedWrapper, SimulationWrapper, WrapperConfig

if TYPE_CHECKING:
    import polars as pl

    from uq.cell_cycle import CellCycleVariable


class SensitivityMethod(str, Enum):
    """Available sensitivity analysis methods."""

    PCE = "pce"  # Polynomial Chaos Expansion
    SOBOL = "sobol"  # Sobol indices via Monte Carlo
    MORRIS = "morris"  # Morris screening


@dataclass
class SobolIndices:
    """
    Container for Sobol sensitivity indices.

    Attributes:
        first_order: First-order (main effect) indices, shape (n_params,) or (n_outputs, n_params)
        total_order: Total-order indices, shape (n_params,) or (n_outputs, n_params)
        second_order: Second-order interaction indices, shape (n_params, n_params) or (n_outputs, n_params, n_params)
        parameter_names: Names of the input parameters
        output_names: Names of the outputs (if multiple)
        confidence_intervals: Optional confidence intervals for indices
    """

    first_order: np.ndarray
    total_order: np.ndarray
    second_order: Optional[np.ndarray] = None
    parameter_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    confidence_intervals: Optional[dict[str, np.ndarray]] = None

    def get_most_influential(self, n: int = 5, index_type: str = "total") -> list[tuple[str, float]]:
        """
        Get the most influential parameters.

        Args:
            n: Number of top parameters to return
            index_type: Type of index to use ('first' or 'total')

        Returns:
            List of (parameter_name, index_value) tuples
        """
        if index_type == "first":
            indices = self.first_order
        else:
            indices = self.total_order

        # Handle multi-output case by averaging
        if indices.ndim > 1:
            indices = np.mean(indices, axis=0)

        sorted_idx = np.argsort(indices)[::-1][:n]
        return [(self.parameter_names[i], indices[i]) for i in sorted_idx]


@dataclass
class MorrisIndices:
    """
    Container for Morris screening results (elementary effects).

    Morris screening is a computationally efficient method for identifying
    the most influential parameters. It requires O(n) model evaluations
    (vs O(n²) for Sobol), making it ideal for screening before detailed GSA.

    Interpretation:
        - mu (μ): Mean elementary effect - overall influence (can cancel if non-monotonic)
        - mu_star (μ*): Mean of |elementary effect| - robust measure of influence
        - sigma (σ): Std dev of elementary effects - indicates interactions/nonlinearity

    Classification:
        - High μ*, low σ: Linear effect (no interactions)
        - High μ*, high σ: Nonlinear effect or interactions with other params
        - Low μ*: Parameter has little influence

    Attributes:
        mu: Mean elementary effects, shape (n_params,)
        mu_star: Mean absolute elementary effects, shape (n_params,)
        sigma: Standard deviation of elementary effects, shape (n_params,)
        parameter_names: Names of the input parameters
        elementary_effects: Raw elementary effects, shape (n_trajectories, n_params)
        n_trajectories: Number of Morris trajectories used
        n_levels: Number of grid levels used
    """

    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    parameter_names: list[str] = field(default_factory=list)
    elementary_effects: Optional[np.ndarray] = None
    n_trajectories: int = 0
    n_levels: int = 4

    def get_most_influential(self, n: int = 5) -> list[tuple[str, float]]:
        """
        Get the most influential parameters based on μ* (mu_star).

        μ* is preferred over μ because it handles non-monotonic effects
        where positive and negative effects might cancel out.

        Args:
            n: Number of top parameters to return

        Returns:
            List of (parameter_name, mu_star_value) tuples, sorted by influence
        """
        sorted_idx = np.argsort(self.mu_star)[::-1][:n]
        return [(self.parameter_names[i], float(self.mu_star[i])) for i in sorted_idx]

    def get_screening_candidates(
        self,
        mu_star_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> list[str]:
        """
        Get parameters that pass the screening threshold.

        Use this to identify which parameters to include in detailed
        PCE/Sobol analysis.

        Args:
            mu_star_threshold: Minimum μ* value to be considered influential.
                              If None, uses 10% of max μ* as threshold.
            top_n: Alternatively, just return the top N parameters.

        Returns:
            List of parameter names that pass screening
        """
        if top_n is not None:
            sorted_idx = np.argsort(self.mu_star)[::-1][:top_n]
            return [self.parameter_names[i] for i in sorted_idx]

        if mu_star_threshold is None:
            mu_star_threshold = 0.1 * np.max(self.mu_star)

        passing_idx = np.where(self.mu_star >= mu_star_threshold)[0]
        # Sort by influence
        passing_idx = passing_idx[np.argsort(self.mu_star[passing_idx])[::-1]]
        return [self.parameter_names[i] for i in passing_idx]

    def classify_parameters(self) -> dict[str, list[str]]:
        """
        Classify parameters into categories based on Morris indices.

        Returns:
            Dictionary with keys:
                - 'negligible': Low influence (can likely be fixed)
                - 'linear': High influence, low interactions
                - 'nonlinear': High influence, significant interactions/nonlinearity
        """
        # Normalize to [0, 1] for classification
        mu_star_norm = self.mu_star / (np.max(self.mu_star) + 1e-10)
        sigma_norm = self.sigma / (np.max(self.sigma) + 1e-10)

        negligible = []
        linear = []
        nonlinear = []

        for i, name in enumerate(self.parameter_names):
            if mu_star_norm[i] < 0.1:
                negligible.append(name)
            elif sigma_norm[i] < 0.5:
                linear.append(name)
            else:
                nonlinear.append(name)

        return {
            "negligible": negligible,
            "linear": linear,
            "nonlinear": nonlinear,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of Morris screening results."""
        lines = [
            "Morris Screening Results",
            "=" * 50,
            f"Trajectories: {self.n_trajectories}, Levels: {self.n_levels}",
            "",
            f"{'Parameter':<25} {'μ*':>10} {'σ':>10} {'μ':>10}",
            "-" * 55,
        ]

        # Sort by mu_star descending
        sorted_idx = np.argsort(self.mu_star)[::-1]
        for i in sorted_idx:
            lines.append(
                f"{self.parameter_names[i]:<25} {self.mu_star[i]:>10.4f} {self.sigma[i]:>10.4f} {self.mu[i]:>10.4f}"
            )

        classification = self.classify_parameters()
        lines.extend([
            "",
            "Classification:",
            f"  Negligible: {classification['negligible']}",
            f"  Linear effects: {classification['linear']}",
            f"  Nonlinear/interactions: {classification['nonlinear']}",
        ])

        return "\n".join(lines)

    def to_parameter_config(
        self,
        parameter_bounds: Optional[list[tuple[float, float]]] = None,
        top_n: Optional[int] = None,
        mu_star_threshold: Optional[float] = None,
        include_descriptions: bool = True,
    ) -> list[dict]:
        """
        Convert screening results to PARAMETER_CONFIG format for tutorials.

        This bridges the gap between Morris screening and the reactive
        sensitivity tutorial (03c_reactive_sensitivity_generalized.py),
        allowing you to automatically populate PARAMETER_CONFIG with
        the most influential parameters identified by screening.

        Args:
            parameter_bounds: List of (min, max) bounds for each parameter.
                             If None, uses [0, 1] for all parameters.
                             Can also pass InputParameterSpace.parameter_bounds.
            top_n: Select top N most influential parameters.
            mu_star_threshold: Alternatively, select params above this μ* threshold.
                              If neither is specified, uses top_n=5.
            include_descriptions: Whether to include Morris stats in descriptions.

        Returns:
            List of parameter config dicts compatible with tutorial format:
            [{"name": str, "bounds": [min, max], "default": float,
              "step": float, "description": str}, ...]

        Example:
            >>> # After Morris screening
            >>> morris = analyzer.analyze_with_morris(n_trajectories=20)
            >>>
            >>> # Convert to tutorial format
            >>> PARAMETER_CONFIG = morris.to_parameter_config(
            ...     parameter_bounds=param_space.parameter_bounds,
            ...     top_n=5,
            ... )
            >>>
            >>> # Now use in reactive sensitivity tutorial
            >>> # (copy to 03c_reactive_sensitivity_generalized.py)
        """
        # Determine which parameters to include
        if top_n is None and mu_star_threshold is None:
            top_n = min(5, len(self.parameter_names))

        selected_names = self.get_screening_candidates(
            top_n=top_n,
            mu_star_threshold=mu_star_threshold,
        )

        # Build parameter config
        config = []
        for i, name in enumerate(self.parameter_names):
            if name not in selected_names:
                continue

            # Get bounds
            if parameter_bounds is not None:
                bounds = list(parameter_bounds[i])
            else:
                bounds = [0.0, 1.0]

            # Calculate default (midpoint) and step
            default = (bounds[0] + bounds[1]) / 2
            step = (bounds[1] - bounds[0]) / 50  # 50 steps across range

            # Round step to nice value
            if step > 0:
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
                step = nice_step * magnitude

            # Build description
            if include_descriptions:
                classification = self.classify_parameters()
                if name in classification["linear"]:
                    effect_type = "linear effect"
                elif name in classification["nonlinear"]:
                    effect_type = "nonlinear/interactions"
                else:
                    effect_type = "low influence"

                description = f"Morris screening: μ*={self.mu_star[i]:.4f}, σ={self.sigma[i]:.4f} ({effect_type})"
            else:
                description = ""

            config.append({
                "name": name,
                "bounds": bounds,
                "default": default,
                "step": step,
                "description": description,
            })

        # Sort by mu_star (most influential first)
        name_to_mu_star = {name: self.mu_star[i] for i, name in enumerate(self.parameter_names)}
        config.sort(key=lambda x: name_to_mu_star.get(x["name"], 0), reverse=True)

        return config


def _legendre_polynomial(x: np.ndarray, order: int) -> np.ndarray:
    """
    Evaluate Legendre polynomial of given order at points x.

    Uses the recurrence relation for numerical stability:
    P_0(x) = 1
    P_1(x) = x
    P_n(x) = ((2n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x)) / n

    Args:
        x: Points at which to evaluate, in [-1, 1]
        order: Polynomial order (non-negative integer)

    Returns:
        P_order(x) evaluated at each point
    """
    if order == 0:
        return np.ones_like(x)
    elif order == 1:
        return x.copy()
    else:
        p_prev2 = np.ones_like(x)  # P_0
        p_prev1 = x.copy()  # P_1
        for n in range(2, order + 1):
            p_curr = ((2 * n - 1) * x * p_prev1 - (n - 1) * p_prev2) / n
            p_prev2 = p_prev1
            p_prev1 = p_curr
        return p_prev1


def _hermite_polynomial(x: np.ndarray, order: int) -> np.ndarray:
    """
    Evaluate probabilist's Hermite polynomial (He_n) at points x.

    Uses recurrence: He_0(x) = 1, He_1(x) = x, He_n(x) = x*He_{n-1}(x) - (n-1)*He_{n-2}(x)

    Args:
        x: Points at which to evaluate
        order: Polynomial order

    Returns:
        He_order(x) evaluated at each point
    """
    if order == 0:
        return np.ones_like(x)
    elif order == 1:
        return x.copy()
    else:
        h_prev2 = np.ones_like(x)
        h_prev1 = x.copy()
        for n in range(2, order + 1):
            h_curr = x * h_prev1 - (n - 1) * h_prev2
            h_prev2 = h_prev1
            h_prev1 = h_curr
        return h_prev1


@dataclass
class PCESurrogate:
    """
    Polynomial Chaos Expansion surrogate model.

    Attributes:
        coefficients: PCE coefficients, shape (n_terms,) or (n_terms, n_outputs)
        multi_indices: Multi-index matrix for polynomial terms, shape (n_terms, n_params)
        basis_type: Type of polynomial basis used ('legendre' or 'hermite')
        polynomial_order: Maximum polynomial order
        input_dim: Number of input parameters
        output_dim: Number of outputs
        r_squared: Coefficient of determination
        input_bounds: Optional bounds for input normalization, shape (n_params, 2)
    """

    coefficients: np.ndarray
    multi_indices: np.ndarray
    basis_type: str = "legendre"
    polynomial_order: int = 3
    input_dim: int = 0
    output_dim: int = 0
    r_squared: float = 0.0
    input_bounds: Optional[np.ndarray] = None

    def _normalize_inputs(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize inputs to the domain expected by the polynomial basis.

        For Legendre: map to [-1, 1]
        For Hermite: standardize to mean=0, std=1

        Args:
            X: Input array of shape (n_samples, n_params)

        Returns:
            Normalized inputs
        """
        if self.input_bounds is None:
            # Assume inputs are already normalized
            return X

        if self.basis_type == "legendre":
            # Map [lb, ub] -> [-1, 1]
            lb = self.input_bounds[:, 0]
            ub = self.input_bounds[:, 1]
            return 2.0 * (X - lb) / (ub - lb) - 1.0
        elif self.basis_type == "hermite":
            # Standardize assuming uniform distribution
            lb = self.input_bounds[:, 0]
            ub = self.input_bounds[:, 1]
            mean = (lb + ub) / 2
            std = (ub - lb) / np.sqrt(12)  # std of uniform distribution
            return (X - mean) / std
        else:
            return X

    def _evaluate_basis(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Evaluate all polynomial basis functions at normalized inputs.

        Args:
            X_norm: Normalized inputs, shape (n_samples, n_params)

        Returns:
            Basis matrix of shape (n_samples, n_terms)
        """
        n_samples = X_norm.shape[0]
        n_terms = self.multi_indices.shape[0]

        # Select polynomial function based on basis type
        if self.basis_type == "legendre":
            poly_func = _legendre_polynomial
        elif self.basis_type == "hermite":
            poly_func = _hermite_polynomial
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

        # Evaluate basis functions
        basis_matrix = np.ones((n_samples, n_terms))

        for term_idx in range(n_terms):
            for param_idx in range(self.input_dim):
                order = int(self.multi_indices[term_idx, param_idx])
                if order > 0:
                    # Multiply by univariate polynomial
                    basis_matrix[:, term_idx] *= poly_func(X_norm[:, param_idx], order)

        return basis_matrix

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs using the PCE surrogate.

        The PCE prediction is computed as:
            Y = sum_i(c_i * Psi_i(X))

        where c_i are the coefficients and Psi_i are multivariate
        polynomial basis functions constructed as products of
        univariate polynomials according to the multi-index.

        Args:
            X: Input array of shape (n_samples, n_params) or (n_params,)

        Returns:
            Predicted outputs of shape (n_samples, n_outputs) or (n_outputs,)
        """
        # Handle 1D input
        squeeze_output = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze_output = True

        # Validate input dimension
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input has {X.shape[1]} features, expected {self.input_dim}")

        # Normalize inputs
        X_norm = self._normalize_inputs(X)

        # Evaluate basis functions
        basis_matrix = self._evaluate_basis(X_norm)  # (n_samples, n_terms)

        # Compute predictions: Y = Phi @ coefficients
        coeffs = self.coefficients
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)

        predictions = basis_matrix @ coeffs  # (n_samples, n_outputs)

        # Squeeze if single sample input
        if squeeze_output:
            predictions = predictions.squeeze(0)

        return predictions

    def predict_with_uncertainty(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict outputs with uncertainty estimate.

        For PCE, the uncertainty comes from the variance of the expansion.
        This is a simplified estimate based on coefficient magnitudes.

        Args:
            X: Input array of shape (n_samples, n_params)
            return_std: If True, return std; otherwise return variance

        Returns:
            Tuple of (predictions, uncertainty)
        """
        predictions = self.predict(X)

        # Estimate variance from non-constant coefficients
        # (This is approximate - full uncertainty would need coefficient covariance)
        coeffs = self.coefficients
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)

        # Variance from non-constant terms (skip first coefficient = mean)
        variance = np.sum(coeffs[1:] ** 2, axis=0)

        if return_std:
            uncertainty = np.sqrt(variance)
        else:
            uncertainty = variance

        # Broadcast uncertainty to match prediction shape
        if predictions.ndim == 1:
            return predictions, uncertainty
        else:
            return predictions, np.broadcast_to(uncertainty, predictions.shape)


class SensitivityAnalyzer:
    """
    Performs global sensitivity analysis on vEcoli simulations.

    This class supports multiple sensitivity analysis methods and can work
    with either a simulation wrapper (for running new simulations) or
    precomputed results.
    """

    def __init__(
        self,
        parameter_space: InputParameterSpace,
        wrapper: Optional[Union[SimulationWrapper, PrecomputedWrapper]] = None,
        samples: Optional[np.ndarray] = None,
        outputs: Optional[np.ndarray] = None,
    ):
        """
        Initialize the sensitivity analyzer.

        Args:
            parameter_space: Definition of the input parameter space
            wrapper: Optional simulation wrapper for running simulations
            samples: Optional precomputed input samples of shape (n_samples, n_params)
            outputs: Optional precomputed outputs of shape (n_samples, n_outputs)
        """
        self.parameter_space = parameter_space
        self.wrapper = wrapper
        self.samples = samples
        self.outputs = outputs

    def analyze_with_pce(
        self,
        polynomial_order: int = 3,
        n_samples: Optional[int] = None,
        use_uqpy: bool = True,
    ) -> tuple[SobolIndices, PCESurrogate]:
        """
        Perform sensitivity analysis using PCE surrogate.

        This is the recommended method from the UQ framework RFC.

        Args:
            polynomial_order: Maximum polynomial order for PCE
            n_samples: Number of samples to use (if generating new samples)
            use_uqpy: If True, use UQPy; otherwise use PyTUQ

        Returns:
            Tuple of (SobolIndices, PCESurrogate)
        """
        if use_uqpy:
            return self._analyze_pce_uqpy(polynomial_order, n_samples)
        else:
            return self._analyze_pce_pytuq(polynomial_order, n_samples)

    def _analyze_pce_uqpy(
        self,
        polynomial_order: int,
        n_samples: Optional[int],
    ) -> tuple[SobolIndices, PCESurrogate]:
        """
        Perform PCE-based sensitivity analysis using UQPy.

        Args:
            polynomial_order: Maximum polynomial order
            n_samples: Number of samples

        Returns:
            Tuple of (SobolIndices, PCESurrogate)
        """
        try:
            from UQpy.distributions import JointIndependent, Uniform
            from UQpy.sampling import LatinHypercubeSampling
            from UQpy.sensitivity import PceSensitivity
            from UQpy.surrogates.polynomial_chaos import (
                PolynomialChaosExpansion,
                Polynomials,
                TotalDegreeBasis,
            )
        except ImportError:
            raise ImportError("UQPy is required for PCE sensitivity analysis. Install it with: pip install UQpy")

        # Get or generate samples and outputs
        X, Y = self._get_samples_and_outputs(n_samples)

        # Create distributions for each parameter
        distributions = []
        for lb, ub in self.parameter_space.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))

        joint_dist = JointIndependent(marginals=distributions)

        # Create polynomial basis
        polynomials = Polynomials(
            distributions=joint_dist,
            polynomial_type="legendre",
        )
        basis = TotalDegreeBasis(
            distributions=joint_dist,
            max_degree=polynomial_order,
        )

        # Fit PCE
        pce = PolynomialChaosExpansion(
            polynomial_basis=basis,
            polynomials=polynomials,
        )
        pce.fit(X, Y)

        # Compute Sobol indices
        pce_sa = PceSensitivity(pce)

        # Get first and total order indices
        first_order = pce_sa.calculate_first_order_indices()
        total_order = pce_sa.calculate_total_order_indices()

        # Build result objects
        sobol = SobolIndices(
            first_order=np.array(first_order),
            total_order=np.array(total_order),
            parameter_names=self.parameter_space.parameter_names,
        )

        # Get input bounds for prediction normalization
        bounds = np.array(self.parameter_space.parameter_bounds)

        surrogate = PCESurrogate(
            coefficients=pce.coefficients,
            multi_indices=basis.multi_index_set,
            basis_type="legendre",
            polynomial_order=polynomial_order,
            input_dim=self.parameter_space.n_parameters,
            output_dim=Y.shape[1] if Y.ndim > 1 else 1,
            input_bounds=bounds,
        )

        return sobol, surrogate

    def _analyze_pce_pytuq(
        self,
        polynomial_order: int,
        n_samples: Optional[int],
    ) -> tuple[SobolIndices, PCESurrogate]:
        """
        Perform PCE-based sensitivity analysis using PyTUQ.

        Args:
            polynomial_order: Maximum polynomial order
            n_samples: Number of samples

        Returns:
            Tuple of (SobolIndices, PCESurrogate)
        """
        try:
            from pytuq.gsa import PCESobol
            from pytuq.surrogates import PCE
        except ImportError:
            raise ImportError("PyTUQ is required for PCE sensitivity analysis. Install it with: pip install pytuq")

        # Get or generate samples and outputs
        X, Y = self._get_samples_and_outputs(n_samples)

        # Get bounds
        lb, ub = self.parameter_space.get_pytuq_bounds()

        # Fit PCE surrogate
        pce = PCE(
            order=polynomial_order,
            bounds=(lb, ub),
        )
        pce.fit(X, Y)

        # Compute Sobol indices
        sa = PCESobol(pce)
        first_order = sa.first_order()
        total_order = sa.total_order()

        sobol = SobolIndices(
            first_order=first_order,
            total_order=total_order,
            parameter_names=self.parameter_space.parameter_names,
        )

        # Get input bounds for prediction normalization
        bounds = np.array(self.parameter_space.parameter_bounds)

        surrogate = PCESurrogate(
            coefficients=pce.coefficients,
            multi_indices=pce.multi_indices,
            basis_type="legendre",
            polynomial_order=polynomial_order,
            input_dim=self.parameter_space.n_parameters,
            output_dim=Y.shape[1] if Y.ndim > 1 else 1,
            input_bounds=bounds,
        )

        return sobol, surrogate

    def analyze_with_sobol(
        self,
        n_samples: int = 1024,
        calc_second_order: bool = False,
    ) -> SobolIndices:
        """
        Perform Sobol sensitivity analysis via Monte Carlo sampling.

        This method uses Saltelli's sampling scheme for efficient
        computation of Sobol indices.

        Args:
            n_samples: Number of base samples (total evaluations will be n_samples * (2 * n_params + 2))
            calc_second_order: Whether to calculate second-order indices

        Returns:
            SobolIndices
        """
        try:
            from UQpy.distributions import JointIndependent, Uniform
            from UQpy.sensitivity import SobolSensitivity
        except ImportError:
            raise ImportError("UQPy is required for Sobol sensitivity analysis. Install it with: pip install UQpy")

        # Create distributions
        distributions = []
        for lb, ub in self.parameter_space.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))

        joint_dist = JointIndependent(marginals=distributions)

        # Create model function
        def model_func(X: np.ndarray) -> np.ndarray:
            if self.wrapper is not None:
                return self.wrapper.evaluate_batch(X)
            else:
                raise ValueError("Wrapper required for Sobol analysis")

        # Run Sobol analysis
        sobol = SobolSensitivity(
            runmodel_object=model_func,
            distributions=joint_dist,
            n_samples=n_samples,
            calculate_second_order=calc_second_order,
        )

        result = SobolIndices(
            first_order=sobol.first_order_indices,
            total_order=sobol.total_order_indices,
            second_order=sobol.second_order_indices if calc_second_order else None,
            parameter_names=self.parameter_space.parameter_names,
        )

        return result

    def analyze_with_morris(
        self,
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: Optional[int] = None,
    ) -> MorrisIndices:
        """
        Perform Morris screening for efficient parameter importance ranking.

        Morris screening (Elementary Effects method) is computationally cheap
        compared to Sobol analysis, requiring only O(n_trajectories * (n_params + 1))
        model evaluations. Use this for initial screening to identify which
        parameters to include in detailed PCE/Sobol analysis.

        The method works by computing "elementary effects" - the change in output
        when one parameter is perturbed while others are held fixed. Statistics
        of these effects (mean, std) reveal parameter importance and interactions.

        Args:
            n_trajectories: Number of Morris trajectories. More trajectories give
                           more stable estimates but require more evaluations.
                           Typical values: 10-50. Total evaluations = n_trajectories * (n_params + 1).
            n_levels: Number of grid levels for the parameter space. Higher values
                     give finer resolution but may miss local effects. Typical: 4-8.
            seed: Random seed for reproducibility.

        Returns:
            MorrisIndices containing μ, μ*, σ for each parameter

        Example:
            >>> # Screen 20 parameters with only ~200 evaluations
            >>> morris = analyzer.analyze_with_morris(n_trajectories=10)
            >>> print(morris.summary())
            >>>
            >>> # Get top 5 most influential parameters for detailed analysis
            >>> important_params = morris.get_screening_candidates(top_n=5)
            >>> print(f"Focus detailed analysis on: {important_params}")

        References:
            Morris, M.D. (1991). "Factorial Sampling Plans for Preliminary
            Computational Experiments". Technometrics, 33(2), 161-174.
        """
        # Try UQPy first (preferred)
        try:
            return self._analyze_morris_uqpy(n_trajectories, n_levels, seed)
        except ImportError:
            pass

        # Fallback to manual implementation
        return self._analyze_morris_manual(n_trajectories, n_levels, seed)

    def _analyze_morris_uqpy(
        self,
        n_trajectories: int,
        n_levels: int,
        seed: Optional[int],
    ) -> MorrisIndices:
        """Morris screening using UQPy's MorrisSensitivity."""
        try:
            from UQpy.distributions import JointIndependent, Uniform
            from UQpy.sensitivity import MorrisSensitivity
        except ImportError:
            raise ImportError("UQPy is required for Morris sensitivity analysis. Install it with: pip install UQpy")

        # Create distributions for each parameter
        distributions = []
        for lb, ub in self.parameter_space.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))

        joint_dist = JointIndependent(marginals=distributions)

        # Create model function
        def model_func(X: np.ndarray) -> np.ndarray:
            if self.wrapper is not None:
                return self.wrapper.evaluate_batch(X)
            elif self.samples is not None and self.outputs is not None:
                # For precomputed data, we need to interpolate or raise error
                raise ValueError(
                    "Morris analysis requires a wrapper to evaluate new points. "
                    "Precomputed samples/outputs cannot be used."
                )
            else:
                raise ValueError("Wrapper required for Morris analysis")

        # Set random state if provided
        if seed is not None:
            np.random.seed(seed)

        # Run Morris analysis
        morris = MorrisSensitivity(
            runmodel_object=model_func,
            distributions=joint_dist,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
        )

        # Extract results
        # UQPy returns elementary_effects of shape (n_trajectories, n_params)
        elementary_effects = np.array(morris.elementary_effects)

        # Compute statistics
        mu = np.mean(elementary_effects, axis=0)
        mu_star = np.mean(np.abs(elementary_effects), axis=0)
        sigma = np.std(elementary_effects, axis=0)

        return MorrisIndices(
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            parameter_names=self.parameter_space.parameter_names,
            elementary_effects=elementary_effects,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
        )

    def _analyze_morris_manual(
        self,
        n_trajectories: int,
        n_levels: int,
        seed: Optional[int],
    ) -> MorrisIndices:
        """
        Manual implementation of Morris screening.

        Used as fallback when UQPy is not available.
        """
        if seed is not None:
            np.random.seed(seed)

        n_params = self.parameter_space.n_parameters
        bounds = np.array(self.parameter_space.parameter_bounds)
        lb, ub = bounds[:, 0], bounds[:, 1]

        # Grid step size
        delta = n_levels / (2 * (n_levels - 1))

        # Generate Morris trajectories
        elementary_effects = np.zeros((n_trajectories, n_params))

        for traj in range(n_trajectories):
            # Random starting point on grid
            x_base = np.random.randint(0, n_levels - 1, n_params) / (n_levels - 1)

            # Random permutation of parameters
            perm = np.random.permutation(n_params)

            # Build trajectory: start point + n_params perturbations
            trajectory = np.zeros((n_params + 1, n_params))
            trajectory[0] = x_base.copy()

            for i, param_idx in enumerate(perm):
                trajectory[i + 1] = trajectory[i].copy()
                # Perturb this parameter by +/- delta
                if trajectory[i, param_idx] + delta <= 1.0:
                    trajectory[i + 1, param_idx] += delta
                else:
                    trajectory[i + 1, param_idx] -= delta

            # Scale to actual parameter bounds
            trajectory_scaled = lb + trajectory * (ub - lb)

            # Evaluate model at all trajectory points
            if self.wrapper is not None:
                outputs = self.wrapper.evaluate_batch(trajectory_scaled)
            else:
                raise ValueError("Wrapper required for Morris analysis")

            # Ensure outputs is 1D for single-output case
            if outputs.ndim > 1:
                outputs = outputs[:, 0]

            # Compute elementary effects
            for i, param_idx in enumerate(perm):
                dy = outputs[i + 1] - outputs[i]
                dx = trajectory[i + 1, param_idx] - trajectory[i, param_idx]
                # Scale by parameter range
                param_range = ub[param_idx] - lb[param_idx]
                elementary_effects[traj, param_idx] = dy / (dx * param_range) if dx != 0 else 0

        # Compute statistics
        mu = np.mean(elementary_effects, axis=0)
        mu_star = np.mean(np.abs(elementary_effects), axis=0)
        sigma = np.std(elementary_effects, axis=0)

        return MorrisIndices(
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            parameter_names=self.parameter_space.parameter_names,
            elementary_effects=elementary_effects,
            n_trajectories=n_trajectories,
            n_levels=n_levels,
        )

    def _get_samples_and_outputs(
        self,
        n_samples: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get input samples and outputs.

        Args:
            n_samples: Number of samples to generate (if needed)

        Returns:
            Tuple of (X, Y) arrays
        """
        if self.samples is not None and self.outputs is not None:
            return self.samples, self.outputs

        if isinstance(self.wrapper, PrecomputedWrapper):
            return self.wrapper.get_samples_and_outputs()

        if self.wrapper is None:
            raise ValueError("Either provide samples/outputs or a wrapper for generating them")

        if n_samples is None:
            # Use rule of thumb for PCE: (p + d)! / (p! * d!)
            # where p is polynomial order and d is dimension
            n_samples = 100 * self.parameter_space.n_parameters

        # Generate Latin Hypercube samples
        try:
            from UQpy.distributions import JointIndependent, Uniform
            from UQpy.sampling import LatinHypercubeSampling
        except ImportError:
            # Fallback to simple random sampling
            bounds = self.parameter_space.bounds_array
            X = np.random.uniform(
                bounds[:, 0],
                bounds[:, 1],
                size=(n_samples, self.parameter_space.n_parameters),
            )
        else:
            distributions = []
            for lb, ub in self.parameter_space.parameter_bounds:
                distributions.append(Uniform(loc=lb, scale=ub - lb))

            joint_dist = JointIndependent(marginals=distributions)
            lhs = LatinHypercubeSampling(
                distributions=joint_dist,
                nsamples=n_samples,
            )
            X = lhs.samples

        # Evaluate model
        Y = self.wrapper.evaluate_batch(X)

        self.samples = X
        self.outputs = Y

        return X, Y


def run_sensitivity_analysis(
    sim_data_path: str,
    output_dir: str,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.UNIFORM,
    polynomial_order: int = 3,
    n_samples: Optional[int] = None,
    include_vio: bool = True,
    include_mecillinam: bool = True,
    vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
    vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
    mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    use_uqpy: bool = True,
) -> tuple[SobolIndices, PCESurrogate]:
    """
    Run a complete sensitivity analysis workflow.

    This is a convenience function that sets up the parameter space,
    wrapper, and analyzer, then runs PCE-based sensitivity analysis.

    Args:
        sim_data_path: Path to sim_data pickle file
        output_dir: Directory for outputs and cache
        aggregation_strategy: Strategy for aggregating simulation outputs
        polynomial_order: PCE polynomial order
        n_samples: Number of samples (None for automatic)
        include_vio: Include vio pathway parameters
        include_mecillinam: Include mecillinam parameters
        vio_expression_bounds: Bounds for vio expression
        vio_trl_eff_bounds: Bounds for vio translation efficiency
        mecillinam_conc_bounds: Bounds for mecillinam concentration
        use_uqpy: Use UQPy (True) or PyTUQ (False)

    Returns:
        Tuple of (SobolIndices, PCESurrogate)
    """
    # Set up parameter space
    parameter_space = InputParameterSpace(
        vio_expression_bounds=vio_expression_bounds,
        vio_trl_eff_bounds=vio_trl_eff_bounds,
        mecillinam_conc_bounds=mecillinam_conc_bounds,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )

    # Set up wrapper config
    config = WrapperConfig(
        sim_data_path=sim_data_path,
        output_dir=output_dir,
        cache_dir=f"{output_dir}/cache",
        aggregation_strategy=aggregation_strategy,
    )

    # Create wrapper
    wrapper = SimulationWrapper(config, parameter_space)

    # Create analyzer
    analyzer = SensitivityAnalyzer(parameter_space, wrapper)

    # Run analysis
    return analyzer.analyze_with_pce(
        polynomial_order=polynomial_order,
        n_samples=n_samples,
        use_uqpy=use_uqpy,
    )


@dataclass
class CellCycleRelevanceResult:
    """
    Results from GSA-based cell cycle relevance analysis.

    This captures which observables are most relevant for defining
    the cell cycle variable, as informed by sensitivity analyses
    across aggregation strategies 1-3 per RFC006 Section 3.

    Attributes:
        relevant_observables: List of observable names ranked by CC relevance
        relevance_scores: Dict mapping observable name to relevance score
        variance_by_strategy: Variance explained by each aggregation strategy
        sobol_by_generation: Sobol indices from BY_GENERATION strategy
        sobol_by_seed: Sobol indices from BY_LINEAGE_SEED strategy
        sobol_uniform: Sobol indices from UNIFORM strategy
        residual_variance_fraction: Fraction of variance NOT explained by gen/seed
    """

    relevant_observables: list[str]
    relevance_scores: dict[str, float]
    variance_by_strategy: dict[str, np.ndarray]
    sobol_by_generation: Optional[SobolIndices] = None
    sobol_by_seed: Optional[SobolIndices] = None
    sobol_uniform: Optional[SobolIndices] = None
    residual_variance_fraction: Optional[np.ndarray] = None


def identify_cell_cycle_relevant_observables(
    aggregated_uniform: "AggregatedOutput",
    aggregated_by_gen: "AggregatedOutput",
    aggregated_by_seed: "AggregatedOutput",
    observable_names: list[str],
    min_residual_fraction: float = 0.1,
    top_n: Optional[int] = None,
) -> CellCycleRelevanceResult:
    """
    Identify observables most relevant for cell cycle variable definition.

    Per RFC006 Section 3: "The choice of the 'cell cycle variable' will be
    informed by the sensitivity analyses (1-3)."

    This function analyzes variance decomposition across strategies 1-3 to
    identify observables where variance is NOT explained by generation or
    lineage seed effects - i.e., variance attributable to cell cycle dynamics.

    Args:
        aggregated_uniform: Aggregation results from UNIFORM strategy
        aggregated_by_gen: Aggregation results from BY_GENERATION strategy
        aggregated_by_seed: Aggregation results from BY_LINEAGE_SEED strategy
        observable_names: Names of the observables
        min_residual_fraction: Minimum fraction of residual variance for relevance
        top_n: If provided, return only top N observables

    Returns:
        CellCycleRelevanceResult with ranked observables and relevance scores
    """
    from uq.aggregation import compute_variance_decomposition

    # Compute variance decomposition
    decomposition = compute_variance_decomposition(
        aggregated_by_gen=aggregated_by_gen,
        aggregated_by_seed=aggregated_by_seed,
        aggregated_uniform=aggregated_uniform,
    )

    # Residual variance = variance NOT explained by generation or seed
    # This is the variance attributable to within-cell dynamics (cell cycle!)
    residual_fraction = 1.0 - decomposition["generation_fraction"] - decomposition["seed_fraction"]
    residual_fraction = np.maximum(residual_fraction, 0)  # Clip to non-negative

    # Score observables by residual variance fraction
    # Higher residual = more cell-cycle-related
    relevance_scores = {}
    for i, name in enumerate(observable_names):
        if i < len(residual_fraction):
            score = float(residual_fraction[i]) if residual_fraction.ndim == 1 else float(residual_fraction.mean())
            relevance_scores[name] = score

    # Rank observables by relevance
    ranked = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

    # Filter by minimum residual fraction
    relevant = [(name, score) for name, score in ranked if score >= min_residual_fraction]

    # Optionally limit to top N
    if top_n is not None:
        relevant = relevant[:top_n]

    return CellCycleRelevanceResult(
        relevant_observables=[name for name, _ in relevant],
        relevance_scores=relevance_scores,
        variance_by_strategy={
            "generation_fraction": decomposition["generation_fraction"],
            "seed_fraction": decomposition["seed_fraction"],
            "residual_fraction": residual_fraction,
        },
        residual_variance_fraction=residual_fraction,
    )


def run_gsa_informed_cell_cycle_analysis(
    aggregated_uniform: "AggregatedOutput",
    aggregated_by_gen: "AggregatedOutput",
    aggregated_by_seed: "AggregatedOutput",
    trajectory_data: "pl.DataFrame",
    observable_names: list[str],
    expected_cycle_time: float = 3600.0,
    min_observables: int = 2,
    max_observables: int = 10,
    min_residual_fraction: float = 0.1,
    dt: float = 1.0,
    use_edmd: bool = True,
) -> tuple["CellCycleVariable", CellCycleRelevanceResult]:
    """
    Run the complete GSA-informed cell cycle variable computation.

    This is the main integration point per RFC006 Section 3:
    1. Analyze variance decomposition from strategies 1-3
    2. Identify cell-cycle-relevant observables
    3. Compute Koopman cell cycle variable using those observables

    Args:
        aggregated_uniform: Results from UNIFORM aggregation
        aggregated_by_gen: Results from BY_GENERATION aggregation
        aggregated_by_seed: Results from BY_LINEAGE_SEED aggregation
        trajectory_data: DataFrame with observable time series
        observable_names: All available observable names
        expected_cycle_time: Expected cell cycle duration in seconds
        min_observables: Minimum number of observables to use
        max_observables: Maximum number of observables to use
        min_residual_fraction: Minimum residual variance fraction for relevance
        dt: Timestep of trajectory data
        use_edmd: Whether to use Extended DMD

    Returns:
        Tuple of (CellCycleVariable, CellCycleRelevanceResult)
    """
    from uq.cell_cycle import KoopmanCellCycleVariable

    # Step 1: Identify relevant observables via GSA
    relevance_result = identify_cell_cycle_relevant_observables(
        aggregated_uniform=aggregated_uniform,
        aggregated_by_gen=aggregated_by_gen,
        aggregated_by_seed=aggregated_by_seed,
        observable_names=observable_names,
        min_residual_fraction=min_residual_fraction,
        top_n=max_observables,
    )

    # Ensure minimum number of observables
    selected_observables = relevance_result.relevant_observables
    if len(selected_observables) < min_observables:
        # Fall back to top observables by any variance
        all_ranked = sorted(
            relevance_result.relevance_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        selected_observables = [name for name, _ in all_ranked[:min_observables]]

    # Step 2: Create Koopman cell cycle variable with informed observables
    koopman_cc = KoopmanCellCycleVariable(
        expected_cycle_time=expected_cycle_time,
        dt=dt,
        use_edmd=use_edmd,
        observable_columns=selected_observables,
    )

    # Step 3: Compute cell cycle variable
    cc_variable = koopman_cc.compute(trajectory_data)

    # Add GSA metadata to the result
    cc_variable.metadata["gsa_informed"] = True
    cc_variable.metadata["selected_observables"] = selected_observables
    cc_variable.metadata["relevance_scores"] = {
        obs: relevance_result.relevance_scores.get(obs, 0.0) for obs in selected_observables
    }

    return cc_variable, relevance_result


def analyze_precomputed_results(
    data_dir: str,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.UNIFORM,
    polynomial_order: int = 3,
    include_vio: bool = True,
    include_mecillinam: bool = True,
    use_uqpy: bool = True,
) -> tuple[SobolIndices, PCESurrogate]:
    """
    Run sensitivity analysis on precomputed simulation results.

    Args:
        data_dir: Directory containing simulation outputs
        aggregation_strategy: Strategy for aggregating outputs
        polynomial_order: PCE polynomial order
        include_vio: Include vio pathway parameters
        include_mecillinam: Include mecillinam parameters
        use_uqpy: Use UQPy (True) or PyTUQ (False)

    Returns:
        Tuple of (SobolIndices, PCESurrogate)
    """
    # Set up parameter space
    parameter_space = InputParameterSpace(
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )

    # Create precomputed wrapper
    wrapper = PrecomputedWrapper(
        data_dir=data_dir,
        parameter_space=parameter_space,
        aggregation_strategy=aggregation_strategy,
    )

    # Get samples and outputs
    X, Y = wrapper.get_samples_and_outputs()

    # Create analyzer with precomputed data
    analyzer = SensitivityAnalyzer(
        parameter_space=parameter_space,
        samples=X,
        outputs=Y,
    )

    # Run analysis
    return analyzer.analyze_with_pce(
        polynomial_order=polynomial_order,
        use_uqpy=use_uqpy,
    )
