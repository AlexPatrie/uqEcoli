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
class PCESurrogate:
    """
    Polynomial Chaos Expansion surrogate model.

    Attributes:
        coefficients: PCE coefficients
        multi_indices: Multi-index matrix for polynomial terms
        basis_type: Type of polynomial basis used
        polynomial_order: Maximum polynomial order
        input_dim: Number of input parameters
        output_dim: Number of outputs
        r_squared: Coefficient of determination
    """

    coefficients: np.ndarray
    multi_indices: np.ndarray
    basis_type: str = "legendre"
    polynomial_order: int = 3
    input_dim: int = 0
    output_dim: int = 0
    r_squared: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs using the PCE surrogate.

        Args:
            X: Input array of shape (n_samples, n_params)

        Returns:
            Predicted outputs of shape (n_samples, n_outputs)
        """
        # This is a simplified implementation
        # Full implementation would use proper polynomial evaluation
        raise NotImplementedError("Use UQPy or PyTUQ for PCE prediction")


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

        surrogate = PCESurrogate(
            coefficients=pce.coefficients,
            multi_indices=basis.multi_index_set,
            basis_type="legendre",
            polynomial_order=polynomial_order,
            input_dim=self.parameter_space.n_parameters,
            output_dim=Y.shape[1] if Y.ndim > 1 else 1,
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

        surrogate = PCESurrogate(
            coefficients=pce.coefficients,
            multi_indices=pce.multi_indices,
            basis_type="legendre",
            polynomial_order=polynomial_order,
            input_dim=self.parameter_space.n_parameters,
            output_dim=Y.shape[1] if Y.ndim > 1 else 1,
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
