Changelog
=========

All notable changes to the vEcoli UQ Framework.

[0.1.0] - 2026-03-03
--------------------

Initial release implementing RFC006 requirements for Milestone 08.4.2.

Added
^^^^^

**RFC006 Compliance**

* ``RFC006.md`` - Authoritative specification for UQ framework
* ``RFC006_VERIFICATION.md`` - Compliance analysis and implementation status
* ``CONTEXT.md`` - Claude context document referencing RFC006

**Input Parameters** (``uq/inputs.py``)

* ``VioPathwayParams`` - Violacein pathway configuration
* ``MecillinamParams`` - Mecillinam antibiotic conditions
* ``GeneKnockoutParams`` - Gene knockout configuration
* ``UQInputParameters`` - Combined parameter container
* ``InputParameterSpace`` - Parameter space for sensitivity analysis

**Output Extraction** (``uq/outputs.py``)

* ``OutputExtractor`` - Extract outputs from Parquet data
* ``OutputVariables`` - Container for extracted outputs
* ``OutputType`` - Enum of output types (transcriptome, proteome, fluxes, etc.)

**Aggregation Strategies** (``uq/aggregation.py``)

* ``AggregationStrategy`` - Enum with UNIFORM, BY_GENERATION, BY_LINEAGE_SEED, BY_CELL_CYCLE
* ``Aggregator`` - Main aggregation class
* ``AggregatedOutput`` - Container for aggregated statistics
* ``compute_variance_decomposition()`` - Decompose variance by source

**Wrappers** (``uq/wrappers.py``)

* ``SimulationWrapper`` - Run simulations for sensitivity analysis
* ``PrecomputedWrapper`` - Analyze existing results
* ``WrapperConfig`` - Wrapper configuration
* ``create_uqpy_model()`` - UQPy integration
* ``create_pytuq_model()`` - PyTUQ integration

**Sensitivity Analysis** (``uq/sensitivity.py``)

* ``SensitivityAnalyzer`` - Main analysis class
* ``SobolIndices`` - Container for Sobol indices
* ``PCESurrogate`` - PCE surrogate model
* ``run_sensitivity_analysis()`` - Convenience function
* ``analyze_precomputed_results()`` - Analyze existing data

**Cell Cycle Stratification** (``uq/cell_cycle.py``)

* ``CellCycleAggregator`` - Aggregation by cell cycle stage
* ``CellCycleVariable`` - Cell cycle variable container
* ``MassBasedCellCycleVariable`` - Mass-based implementation
* ``DNAReplicationCellCycleVariable`` - DNA-based implementation
* ``CellAngleCellCycleVariable`` - Cell angle implementation
* ``CompositeCellCycleVariable`` - Custom variable support
* ``register_cell_cycle_variable()`` - Register custom variables

**Koopman Spectral Analysis** (``uq/koopman.py``)

* ``DynamicModeDecomposition`` - Standard DMD for spectral analysis
* ``ExtendedDMD`` - EDMD with dictionary functions
* ``KoopmanMode`` - Single Koopman mode container
* ``KoopmanSpectrum`` - Complete spectral decomposition
* ``KoopmanSensitivityAnalyzer`` - Spectral sensitivity analysis
* ``CellCycleKoopmanAnalyzer`` - Cell cycle harmonic identification
* ``KoopmanDictionary`` - Dictionary types for EDMD (identity, polynomial, Fourier, RBF)
* ``extract_koopman_features()`` - Extract features from simulation data

Dependencies
^^^^^^^^^^^^

* Added ``UQpy>=4.1.0`` as optional dependency (``pip install -e ".[uq]"``)

Documentation
^^^^^^^^^^^^^

* Added comprehensive Sphinx documentation
* Added tutorials for basic sensitivity, variance decomposition, cell cycle analysis, and Koopman analysis
* Added API reference for all modules
