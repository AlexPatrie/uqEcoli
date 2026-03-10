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
    Global sensitivity analysis using PCE surrogate methods
cell_cycle
    Cell cycle stratification for Phase 2 analysis, including Koopman eigenfunction-
    based cell cycle variable (recommended approach per RFC006 Section 1.3)
koopman
    Koopman spectral analysis via DMD for dynamic sensitivity and cell cycle mode
    identification. Provides the foundation for the Koopman cell cycle variable.

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

    # Define parameter space
    param_space = InputParameterSpace(
        include_vio=True,
        include_mecillinam=True,
    )

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

References
----------
- RFC006: ``uq/RFC006.md`` (Authoritative specification)
- RFC006 Verification: ``uq/RFC006_VERIFICATION.md`` (Compliance analysis)
- UQPy: https://uqpyproject.readthedocs.io/
- PyTUQ: https://sandialabs.github.io/pytuq/
"""

# Input parameter definitions
# Aggregation strategies
from uq.aggregation import (
    AggregatedOutput,
    AggregationStrategy,
    Aggregator,
    compute_variance_decomposition,
)

# Cell cycle stratification (Phase 2)
# The Koopman-based cell cycle variable uses spectral analysis to identify
# the cell cycle mode and extract its eigenfunction phase as the cycle coordinate.
# GSAInformedCellCycleVariable implements the RFC006 requirement that the cell
# cycle variable choice be "informed by the sensitivity analyses (1-3)".
from uq.cell_cycle import (
    CellAngleCellCycleVariable,
    CellCycleAggregator,
    CellCyclePhase,
    CellCycleVariable,
    CellCycleVariableComputer,
    CompositeCellCycleVariable,
    DNAReplicationCellCycleVariable,
    GSAInformedCellCycleVariable,
    KoopmanCellCycleVariable,
    MassBasedCellCycleVariable,
    register_cell_cycle_variable,
)
from uq.inputs import (
    GeneKnockoutParams,
    InputParameterSpace,
    MecillinamParams,
    MediaCondition,
    UQInputParameters,
    VioPathwayParams,
    get_available_columns,
    load_dataset,
)

# Koopman spectral analysis
from uq.koopman import (
    CellCycleKoopmanAnalyzer,
    DynamicModeDecomposition,
    ExtendedDMD,
    KoopmanDictionary,
    KoopmanMode,
    KoopmanSensitivityAnalyzer,
    KoopmanSpectrum,
    extract_koopman_features,
)

# Output variable extraction
from uq.outputs import (
    OutputExtractor,
    OutputType,
    OutputVariables,
    get_output_variable_info,
)

# Sensitivity analysis
# CellCycleRelevanceResult and related functions implement RFC006's requirement
# that the cell cycle variable choice be "informed by the sensitivity analyses (1-3)"
from uq.sensitivity import (
    CellCycleRelevanceResult,
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
from uq.wrappers import (
    PrecomputedWrapper,
    SimulationWrapper,
    WrapperConfig,
    create_pytuq_model,
    create_uqpy_model,
)

__all__ = [
    # Input parameters
    "GeneKnockoutParams",
    "InputParameterSpace",
    "MecillinamParams",
    "MediaCondition",
    "UQInputParameters",
    "VioPathwayParams",
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
    "PCESurrogate",
    "SensitivityAnalyzer",
    "SensitivityMethod",
    "SobolIndices",
    "analyze_precomputed_results",
    "identify_cell_cycle_relevant_observables",
    "run_gsa_informed_cell_cycle_analysis",
    "run_sensitivity_analysis",
    # Cell cycle (GSAInformedCellCycleVariable is the RFC006-compliant approach)
    "CellAngleCellCycleVariable",
    "CellCycleAggregator",
    "CellCyclePhase",
    "CellCycleVariable",
    "CellCycleVariableComputer",
    "CompositeCellCycleVariable",
    "DNAReplicationCellCycleVariable",
    "GSAInformedCellCycleVariable",
    "KoopmanCellCycleVariable",
    "MassBasedCellCycleVariable",
    "register_cell_cycle_variable",
    # Koopman spectral analysis
    "CellCycleKoopmanAnalyzer",
    "DynamicModeDecomposition",
    "ExtendedDMD",
    "KoopmanDictionary",
    "KoopmanMode",
    "KoopmanSensitivityAnalyzer",
    "KoopmanSpectrum",
    "extract_koopman_features",
]

__version__ = "0.1.0"
