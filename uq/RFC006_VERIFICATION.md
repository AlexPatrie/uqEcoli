---
RFC006 Compliance Analysis: UQ Package Assessment

Executive Summary

The uq/ package substantially satisfies the requirements of RFC006, with the Phase 1 framework being fully implemented and Phase 2 framework being complete pending consensus on a specific cell cycle variable. One notable deviation exists: the implementation uses the ParquetEmitter with DuckDB rather than the XarrayEmitter mentioned in the RFC.

---
Phase 1 Requirements (MS-08.4.2) — Target: Apr 15, 2026
┌─────┬──────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  #  │                             Activity                             │                                                          RFC Requirement                                                          │  Status   │                                                              Evidence                                                               │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1   │ Identify scientifically relevant input/output variables          │ Inputs: vio pathway, mecillinam condition, gene knockouts. Outputs: transcriptome, proteome, metabolic fluxes, exchange fluxes,   │ COMPLETE  │ inputs.py:35-136 — VioPathwayParams, MecillinamParams, GeneKnockoutParams; outputs.py:26-34 — OutputType enum covers all required   │
│     │                                                                  │ higher-order properties                                                                                                           │           │ outputs                                                                                                                             │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2   │ Enable output of relevant variables via XarrayEmitter            │ XarrayEmitter committed to draft PR                                                                                               │ DEVIATION │ Uses ParquetEmitter + DuckDB instead. CONTEXT.md line 198 lists "Integration with XarrayEmitter" as future work                     │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3   │ Implement input→output wrapper functions callable from numerical │ Functions written, tested, committed                                                                                              │ COMPLETE  │ wrappers.py:72-364 — SimulationWrapper, wrappers.py:367-559 — PrecomputedWrapper, wrappers.py:562-635 — create_uqpy_model(),        │
│     │  libraries                                                       │                                                                                                                                   │           │ create_pytuq_model()                                                                                                                │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4   │ Implement PCE-based sensitivity analysis for aggregation         │ Sensitivity analyses implemented                                                                                                  │ COMPLETE  │ sensitivity.py:115-428 — SensitivityAnalyzer with analyze_with_pce() supporting UQPy and PyTUQ; aggregation.py:27-33 — all three    │
│     │ strategies 1-3                                                   │                                                                                                                                   │           │ strategies implemented                                                                                                              │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5   │ Apply sensitivity analysis to representative simulations         │ Report with methods explanation and results                                                                                       │ PENDING   │ Framework ready but CONTEXT.md line 47 states "requires running on representative data"                                             │
└─────┴──────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
---
Phase 2 Requirements (CD2 and Milestone 10) — Not yet scheduled
┌─────┬────────────────────────────────────────────┬───────────────────────────────────────────────┬────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  #  │                  Activity                  │                RFC Requirement                │       Status       │                                                                                       Evidence                                                                                       │
├─────┼────────────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6   │ Develop cell cycle stratification strategy │ Separate RFC defining the cell cycle variable │ FRAMEWORK COMPLETE │ cell_cycle.py:82-118 — CellCycleVariableComputer ABC; lines 121-352 — three implementations: MassBasedCellCycleVariable, DNAReplicationCellCycleVariable, CellAngleCellCycleVariable │
├─────┼────────────────────────────────────────────┼───────────────────────────────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 7   │ Implement cell cycle variable analysis     │ Consensus cell cycle approach implemented     │ FRAMEWORK COMPLETE │ cell_cycle.py:405-654 — CellCycleAggregator with full aggregation support; specific variable choice explicitly deferred per RFC plan (line 69)                                       │
└─────┴────────────────────────────────────────────┴───────────────────────────────────────────────┴────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
---
The Four Aggregation Strategies (RFC Section 1)
┌──────────┬───────────────────────────────────────────────────────────┬──────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
│ Strategy │                        Description                        │  Status  │                                     Implementation                                      │
├──────────┼───────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
│ (1)      │ Uniformly across all simulated cells and times (baseline) │ COMPLETE │ AggregationStrategy.UNIFORM in aggregation.py:30                                        │
├──────────┼───────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
│ (2)      │ Stratified by generation                                  │ COMPLETE │ AggregationStrategy.BY_GENERATION in aggregation.py:31                                  │
├──────────┼───────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
│ (3)      │ Stratified by lineage seed                                │ COMPLETE │ AggregationStrategy.BY_LINEAGE_SEED in aggregation.py:32                                │
├──────────┼───────────────────────────────────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
│ (4)      │ Stratified by cell cycle stage                            │ COMPLETE │ AggregationStrategy.BY_CELL_CYCLE in aggregation.py:33 with CellCycleAggregator support │
└──────────┴───────────────────────────────────────────────────────────┴──────────┴─────────────────────────────────────────────────────────────────────────────────────────┘
---
Computational Workflow (RFC Section 4)

The RFC specifies parametrizing three steps in a distinct, modular architecture:
┌──────┬──────────────────────────────────────────────────────────────┬──────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step │                       RFC Requirement                        │  Status  │                                          Implementation                                          │
├──────┼──────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ A    │ Selection/extraction of subsampled time points and variables │ COMPLETE │ outputs.py — OutputExtractor class with filtering by generation_lower_bound and time_lower_bound │
├──────┼──────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ B    │ Temporal aggregation into output variables Y                 │ COMPLETE │ aggregation.py — Aggregator class with modular strategy selection                                │
├──────┼──────────────────────────────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ C    │ Choice of numerical sensitivity analysis method              │ COMPLETE │ sensitivity.py — SensitivityAnalyzer with PCE and Sobol methods                                  │
└──────┴──────────────────────────────────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘
---
Library Requirements (RFC Section 4, footnotes)
┌─────────┬──────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│ Library │  Status  │                                      Implementation                                       │
├─────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ UQPy    │ COMPLETE │ wrappers.py:562-599 — create_uqpy_model(); sensitivity.py:169-249 — _analyze_pce_uqpy()   │
├─────────┼──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ PyTUQ   │ COMPLETE │ wrappers.py:602-635 — create_pytuq_model(); sensitivity.py:251-308 — _analyze_pce_pytuq() │
└─────────┴──────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
---
Additional Features Beyond RFC Minimum Requirements
┌─────────────────────────────────┬───────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
│             Feature             │                      Implementation                       │                                       Notes                                       │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Variance decomposition          │ aggregation.py:772-820 — compute_variance_decomposition() │ Decomposes variance into generation vs. seed components                           │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ PCE surrogate model             │ sensitivity.py:78-112 — PCESurrogate dataclass            │ Stores coefficients, multi-indices, R²                                            │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Sobol indices container         │ sensitivity.py:33-74 — SobolIndices                       │ First-order, total-order, second-order indices with get_most_influential() helper │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Simulation caching              │ wrappers.py:169-185                                       │ MD5-based cache keys, numpy array storage                                         │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Precomputed wrapper             │ wrappers.py:367-559                                       │ Enables sensitivity analysis on existing simulation data                          │
├─────────────────────────────────┼───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Extensible cell cycle variables │ cell_cycle.py:643-654 — register_cell_cycle_variable()    │ Plugin architecture for custom variables                                          │
└─────────────────────────────────┴───────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘
---
Gaps and Deviations

1. XarrayEmitter vs ParquetEmitter: The RFC specifies using the new XarrayEmitter (Activity 2), but the implementation uses DuckDB with ParquetEmitter. This is acknowledged in CONTEXT.md as awaiting XarrayEmitter availability on main branch. The architecture is compatible with a future switch.
2. Activity 5 incomplete: The framework is built but has not yet been applied to representative simulations to produce the specified report. This is marked as pending in the documentation.
3. Cell cycle variable consensus: Phase 2 explicitly defers the choice of specific cell cycle variable to a separate RFC. The framework provides three candidate implementations, which aligns with the RFC's statement that "consensus on the choice of a specific cell cycle variable is deferred to a separate RFC."

---
Conclusion

The uq package ticks the boxes for RFC006 as follows:

- Phase 1 Activities 1, 3, 4: Fully complete
- Phase 1 Activity 2: Implemented with ParquetEmitter (deviation acknowledged, XarrayEmitter planned)
- Phase 1 Activity 5: Framework complete, application pending
- Phase 2 Activities 6, 7: Framework complete as specified (variable consensus deferred per plan)

The package provides a well-architected, modular UQ framework that:
- Supports all four aggregation strategies
- Integrates with both UQPy and PyTUQ
- Implements PCE-based global sensitivity analysis
- Extracts all required output variables (transcriptome, proteome, fluxes, higher-order properties)
- Parametrizes all required input variables (vio pathway, mecillinam, gene knockouts)
- Provides extensibility for cell cycle variables

The remaining work items align with the RFC timeline: applying the analysis to representative simulations and finalizing the cell cycle variable choice through a separate RFC process.
