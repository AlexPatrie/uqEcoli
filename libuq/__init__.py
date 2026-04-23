"""
Uncertainty Quantification (UQ) Framework for vEcoli.

This package implements the UQ framework specified in **RFC006** (``uq/RFC006.md``)
for tracking prediction confidence in vEcoli whole-cell simulations.

See ``uq/RFC006_VERIFICATION.md`` for compliance status and ``uq/CONTEXT.md`` for
the Claude context document.

Milestones
----------
- **MS-08.4.2**: Implement UQ framework to track prediction confidence (Phase 1)
- **MS-10.2.3**: Population-level perturbation analysis (Phase 2)
- **CD2 Evaluation**: Performer-defined population aggregation procedure

Key Components
--------------
inputs
    Input parameter definitions for vio pathway, mecillinam, and gene knockouts
outputs
    Output variable extraction for transcriptome, proteome, fluxes, and properties
aggregation
    Four aggregation strategies: uniform, by generation, by lineage seed, and by cell cycle
wrappers
    Input-to-output wrapper functions compatible with UQPy and PyTUQ
sensitivity
    Global sensitivity analysis using PCE surrogate methods, with Morris
    screening for efficient parameter importance ranking in high dimensions
    (cell_cycle, koopman, viz modules removed from public release —
    see private/spectral-music-apollo branch)

Usage
-----
Basic sensitivity analysis workflow::

    from uq import (
        InputParameterSpace,
        WrapperConfig,
        SimulationWrapper,
        SensitivityAnalyzer,
        AggregationStrategy,
    )

    # Define parameter space (uses generic SimDataParameter specs)
    param_space = InputParameterSpace()

    # Configure wrapper
    config = WrapperConfig(
        sim_data_path="/path/to/sim_data.cPickle",
        output_dir="./uq_outputs",
        aggregation_strategy=AggregationStrategy.UNIFORM,
    )

    # Create wrapper and analyzer
    wrapper = SimulationWrapper(config, param_space)
    analyzer = SensitivityAnalyzer(param_space, wrapper)

    # Run PCE-based sensitivity analysis
    sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
        polynomial_order=3,
        n_samples=100,
    )

    # Get most influential parameters
    top_params = sobol_indices.get_most_influential(n=5)

For precomputed results::

    from uq import analyze_precomputed_results

    sobol_indices, surrogate = analyze_precomputed_results(
        data_dir="./simulation_outputs",
        aggregation_strategy=AggregationStrategy.BY_GENERATION,
    )

Screening workflow (for high-dimensional parameter spaces)::

    from uq import SensitivityAnalyzer, MorrisIndices

    # Stage 1: Morris screening (cheap, O(n) evaluations)
    morris = analyzer.analyze_with_morris(n_trajectories=10)
    print(morris.summary())

    # Identify top candidates for detailed analysis
    important_params = morris.get_screening_candidates(top_n=5)
    print(f"Focus on: {important_params}")

    # Stage 2: Detailed PCE analysis on subset
    # (Create new analyzer with reduced parameter space)
    sobol, pce = reduced_analyzer.analyze_with_pce(polynomial_order=3)

References
----------
- RFC006: ``uq/RFC006.md`` (Authoritative specification)
- RFC006 Verification: ``uq/RFC006_VERIFICATION.md`` (Compliance analysis)
- UQPy: https://uqpyproject.readthedocs.io/
- PyTUQ: https://sandialabs.github.io/pytuq/
"""

# Input parameter definitions
# Aggregation strategies
from libuq.aggregation import (
    AggregatedOutput,
    AggregationStrategy,
    Aggregator,
    compute_variance_decomposition,
)

# Cell cycle stratification (Phase 2)
# The Koopman-based cell cycle variable uses spectral analysis to identify
# Cell cycle and Koopman modules removed from public release.
# Available on the private/spectral-music-apollo branch.
from libuq.inputs import (
    XSpaceVecoli,
    get_available_columns,
    load_dataset,
)

# Output variable extraction
from libuq.outputs import (
    OutputExtractor,
    OutputType,
    OutputVariables,
    get_output_variable_info,
)
from libuq.pce.models import PCEFitResult, PCESurrogateConfig

# Pipeline utilities
from libuq.pce.surrogate import (
    fit_pce_coefficients,
    generate_multi_indices,
    prescreen_parameters,
)
from libuq.pipeline.models import (
    CellCyclePhase,
    CellCycleVariable,
)

# Sensitivity analysis
# CellCycleRelevanceResult and related functions implement RFC006's requirement
# that the cell cycle variable choice be "informed by the sensitivity analyses (1-3)"
# MorrisIndices supports the screening phase for high-dimensional parameter spaces
from libuq.sensitivity import (
    CellCycleRelevanceResult,
    MorrisIndices,
    PCESurrogate,
    SensitivityAnalyzer,
    SensitivityMethod,
    SobolIndices,
    analyze_precomputed_results,
    identify_cell_cycle_relevant_observables,
    run_gsa_informed_cell_cycle_analysis,
    run_sensitivity_analysis,
)

# Wrappers for UQPy/PyTUQ
from libuq.wrappers import (
    PrecomputedWrapper,
    SimulationWrapper,
    WrapperConfig,
    create_pytuq_model,
    create_uqpy_model,
)

__all__ = [
    # Input parameters
    "XSpaceVecoli",
    "get_available_columns",
    "load_dataset",
    # Output extraction
    "OutputExtractor",
    "OutputType",
    "OutputVariables",
    "get_output_variable_info",
    # Aggregation
    "AggregatedOutput",
    "Aggregator",
    "AggregationStrategy",
    "compute_variance_decomposition",
    # Wrappers
    "PrecomputedWrapper",
    "SimulationWrapper",
    "WrapperConfig",
    "create_pytuq_model",
    "create_uqpy_model",
    # Sensitivity analysis
    "CellCycleRelevanceResult",
    "MorrisIndices",
    "PCESurrogate",
    "SensitivityAnalyzer",
    "SensitivityMethod",
    "SobolIndices",
    "analyze_precomputed_results",
    "identify_cell_cycle_relevant_observables",
    "run_gsa_informed_cell_cycle_analysis",
    "run_sensitivity_analysis",
    # Pipeline utilities
    "PCEFitResult",
    "PCESurrogateConfig",
    "fit_pce_coefficients",
    "generate_multi_indices",
    "prescreen_parameters",
]

__version__ = "0.1.0"
