# UQ Framework - Design Document

## 1. The Vision

Uncertainty quantification (UQ) is a topic that touches on several ongoing and future model development efforts. Here we attempt to document current thinking and propose how UQ efforts will be framed to satisfy milestone requirements and to support other relevant project goals.

In brief, in the immediate term we first need to satisfy: **08.4.2 – Milestone 8 (extensibility) Implement uncertainty quantification framework to track prediction confidence**. We propose to do that by formalizing and extending the recent work in support of CD1 that aggregated model outputs including multi-omics, exchange fluxes, and higher order properties as a "bulk" population average across time steps per cell and then across cells. We will extend this framework to capture statistics across four types of aggregation:

1. Uniformly across all simulated cells and times (baseline)
2. Stratified by generation (control of convergence towards steady state-growth)
3. Stratified by lineage seed (control of exogenous variance)
4. Stratified by cell cycle stage, according to a physiological variable that is TBD (time course within a cell's lifespan)

This will enable us to deconvolve different types of uncertainty, and to model the relationship between "bulk" and "single-cell" attributes more accurately. This will enable our future milestone 10.2.3 – Milestone 10 Implement population-level perturbation analysis, capturing cell heterogeneity and adaptation, and will enable the development of a performer-defined population aggregation procedure that will likely be supported/expected during the CD2 evaluation.

We see other opportunities for UQ approaches to intersect with relevant areas of model development in the future, including experimental data ingestion, strain design, and ML surrogacy. We explore these ideas in the Parking Lot section below, as a record of current thinking for future consideration. To not confuse the concrete short term with the longer term speculative, we focus the main body of this RFC on work for Milestone 08.4.2 completion.

## 2. Key Themes

**Milestone 08.4.2 completion:**
- Implement UQ framework which characterizes different types of uncertainty: by cell, by lineage, by generation, and across the cell cycle
- Satisfying 08.4.2 – Milestone 8 (extensibility) Implement uncertainty quantification framework to track prediction confidence
- Mapping of single cell to bulk simulations and measurements
- Enable our future milestone 10.2.3 – Milestone 10 Implement population-level perturbation analysis, capturing cell heterogeneity and adaptation, and will enable the development of a performer-defined population aggregation procedure that will likely be supported/expected during the CD2 evaluation.

## 3. Proposed Solution

**Milestone 08.4.2 completion: Implement UQ framework which characterizes different types of uncertainty: by cell, by lineage, by generation, and across the cell cycle**

For satisfaction of CD1 and comparison of simulation and experimental data, we have implemented analysis modules which first average across all timepoints per cell, and then calculate a mean and standard deviation across all cells.

We will extend this by directly applying well established global sensitivity analysis methods for the stochastic function (sim_data -> SIM output), based on the aggregation strategies (1-3). A second type of analysis will need to be generated for the aggregation strategy (4), and will involve the definition of a low-dimensional (possibly scalar) "cell cycle variable" computed from omics variables. This variable or "coordinate" will be used for deterministically binning simulation data into cell stages, in order to then perform a "phenotypic" sensitivity analysis across the physiological time dimension. The choice of the "cell cycle variable" will be informed by the sensitivity analyses (1-3), will be explored through dedicated visualisations, and will be discussed with all subteams. An established example for such a variable is the "cell angle", and in principle, any deterministic function of relevant process variables inside vEcoli may be considered if it has approximately cyclic behaviour. However, in order to also serve the purposes of CD2, a quantitative relationship should be assumed between the input variables to this "cell cycle variable" and omics measurements from IV&V.

## 4. Implementation Status

The implementation plan below focuses on Theme A, addressing MS-08.4.2 in the near term (target completion is Apr 15, 2026).

Alex Patrie owns this implementation, with Boyan Beronov advising on UQ and the xarray emitter, and Chris Long advising on biological data design.

### Phase 1 - Satisfying 08.4.2

| # | Activity | Status | Implementation |
|---|----------|--------|----------------|
| 1 | Identify scientifically relevant input/output variables | ✅ **COMPLETE** | `uq/inputs.py`: `VioPathwayParams`, `MecillinamParams`, `GeneKnockoutParams`; `uq/outputs.py`: `OutputType` enum (transcriptome, proteome, metabolic_fluxes, exchange_fluxes, higher_order_properties) |
| 2 | Implement input→output wrapper functions callable from numerical libraries | ✅ **COMPLETE** | `uq/wrappers.py`: `SimulationWrapper`, `PrecomputedWrapper`, `create_uqpy_model()`, `create_pytuq_model()` |
| 3 | Implement global sensitivity analysis methods (PCE surrogate) for aggregation strategies 1-3 | ✅ **COMPLETE** | `uq/sensitivity.py`: `SensitivityAnalyzer.analyze_with_pce()`, `SobolIndices`, `PCESurrogate`; `uq/aggregation.py`: `AggregationStrategy.UNIFORM`, `BY_GENERATION`, `BY_LINEAGE_SEED` |
| 4 | Apply sensitivity analysis for representative simulations | 🔄 **PENDING** | Framework ready; requires running on representative data |

### Phase 2 - Necessary for CD2 and Milestone 10

| # | Activity | Status | Implementation |
|---|----------|--------|----------------|
| 5 | Develop cell cycle stratification strategy | ✅ **FRAMEWORK COMPLETE** | `uq/cell_cycle.py`: `CellCycleVariableComputer` base class, `MassBasedCellCycleVariable`, `DNAReplicationCellCycleVariable`, `CellAngleCellCycleVariable` |
| 6 | Implement cell cycle variable analysis | ✅ **FRAMEWORK COMPLETE** | `uq/cell_cycle.py`: `CellCycleAggregator`, `AggregationStrategy.BY_CELL_CYCLE`; specific variable choice deferred to separate RFC |

### Libraries Used

- **UQPy** [1]: Primary library for PCE surrogate construction and Sobol sensitivity analysis
- **PyTUQ** [2]: Alternative library with compatible interface

[1] https://uqpyproject.readthedocs.io/en/latest/sensitivity/index.html
[2] https://sandialabs.github.io/pytuq/autoapi/pytuq/gsa/gsa/index.html

## 5. Architecture

The implementation parametrizes each step as specified in the RFC:

### A. Selection/Extraction of Subsampled Data

**Module:** `uq/outputs.py`

The `OutputExtractor` class handles extraction of subsampled time points and variables from emitted simulation trajectories:

```python
from uq import OutputExtractor, OutputType

extractor = OutputExtractor(conn, history_sql, config_sql, sim_data)
outputs = extractor.extract_all(
    output_types=[OutputType.TRANSCRIPTOME, OutputType.EXCHANGE_FLUXES],
    generation_lower_bound=2,  # Skip initial generations
    time_lower_bound=100.0,    # Skip initial transient
)
```

### B. Temporal Aggregation into Output Variables Y

**Module:** `uq/aggregation.py`

The `Aggregator` class implements all four aggregation strategies:

```python
from uq import Aggregator, AggregationStrategy

aggregator = Aggregator(conn, history_sql, config_sql, sim_data)

# Strategy 1: Uniform (baseline)
agg_uniform, ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

# Strategy 2: By generation
agg_by_gen, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

# Strategy 3: By lineage seed
agg_by_seed, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

# Strategy 4: By cell cycle (Phase 2)
from uq import CellCycleAggregator
cc_aggregator = CellCycleAggregator(conn, history_sql, config_sql, variable_type="mass_based")
profile = cc_aggregator.get_cell_cycle_profile("listeners__rna_counts__mRNA_cistron_counts")
```

### C. Numerical Sensitivity Analysis Method

**Module:** `uq/sensitivity.py`

The `SensitivityAnalyzer` applies PCE-based global sensitivity analysis:

```python
from uq import SensitivityAnalyzer, InputParameterSpace

param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
analyzer = SensitivityAnalyzer(param_space, wrapper)

# PCE surrogate method (as specified in RFC)
sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
    polynomial_order=3,
    n_samples=100,
    use_uqpy=True,  # or use_uqpy=False for PyTUQ
)

# Get most influential parameters
top_params = sobol_indices.get_most_influential(n=5)
```

## 6. Package Structure

```
uq/
├── __init__.py          # Public API exports
├── CONTEXT.md           # This design document
├── README.md            # User documentation
├── inputs.py            # Input parameter definitions
├── outputs.py           # Output variable extraction
├── aggregation.py       # Aggregation strategies (1-4)
├── wrappers.py          # UQPy/PyTUQ compatible wrappers
├── sensitivity.py       # PCE-based sensitivity analysis
└── cell_cycle.py        # Cell cycle stratification (Phase 2)
```

## 7. Variance Decomposition

The framework supports decomposing total variance into components:

```python
from uq import compute_variance_decomposition

decomposition = compute_variance_decomposition(
    aggregated_by_gen,
    aggregated_by_seed,
    aggregated_uniform,
)

# Returns:
# - total_variance: Total variance from uniform aggregation
# - between_generation_variance: Variance attributable to generation
# - between_seed_variance: Variance attributable to lineage seed
# - generation_fraction: Fraction of variance from generation effects
# - seed_fraction: Fraction of variance from stochastic seeding
```

## 8. Cell Cycle Variable Framework (Phase 2)

Three cell cycle variable implementations are provided:

1. **Mass-based** (`MassBasedCellCycleVariable`): Uses normalized log-mass ratio
2. **DNA replication-based** (`DNAReplicationCellCycleVariable`): Tracks DNA mass progression
3. **Cell angle** (`CellAngleCellCycleVariable`): 2D projection in (mass, growth_rate) space

Custom variables can be registered:

```python
from uq import register_cell_cycle_variable, CompositeCellCycleVariable

custom_var = CompositeCellCycleVariable(
    name="custom",
    required_columns=["col1", "col2"],
    compute_func=my_compute_function,
)
register_cell_cycle_variable("custom", custom_var)
```

**Note:** Consensus on a specific cell cycle variable is deferred to a separate RFC and will be coordinated with Phase 2 activities.

## 9. Parking Lot (Future Considerations)

- Experimental data ingestion for UQ validation
- Strain design optimization under uncertainty
- ML surrogate models for rapid UQ evaluation
- Integration with XarrayEmitter (when available on main branch)
