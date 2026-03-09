# UQ Framework - Claude Context Document

## Purpose

This package (`uq/`) implements the uncertainty quantification framework specified in **RFC006** (`uq/RFC006.md`). The primary goal is to satisfy **Milestone 08.4.2**: *Implement uncertainty quantification framework to track prediction confidence*.

**Key Documents:**
- `uq/RFC006.md` — Authoritative specification and requirements
- `uq/RFC006_VERIFICATION.md` — Compliance analysis and implementation status

## RFC006 Requirements Summary

The RFC specifies a UQ framework that captures statistics across four aggregation strategies:

1. **Uniformly** across all simulated cells and times (baseline)
2. **Stratified by generation** (control of convergence towards steady-state growth)
3. **Stratified by lineage seed** (control of exogenous variance)
4. **Stratified by cell cycle stage** (time course within a cell's lifespan)

This enables deconvolution of different types of uncertainty and accurate modeling of "bulk" vs "single-cell" relationships, supporting:
- **MS-08.4.2** — Milestone 8 (extensibility): UQ framework for prediction confidence
- **MS-10.2.3** — Milestone 10: Population-level perturbation analysis (future)
- **CD2 evaluation** — Performer-defined population aggregation procedure

## Implementation Status

| Phase | Activity | Status |
|-------|----------|--------|
| **Phase 1** | Identify input/output variables | COMPLETE |
| **Phase 1** | Implement wrapper functions for UQPy/PyTUQ | COMPLETE |
| **Phase 1** | Implement PCE-based sensitivity analysis (strategies 1-3) | COMPLETE |
| **Phase 1** | Apply to representative simulations | PENDING |
| **Phase 2** | Cell cycle stratification framework | FRAMEWORK COMPLETE |
| **Phase 2** | Cell cycle variable consensus | Deferred to separate RFC |

**Deviation:** Uses ParquetEmitter + DuckDB instead of XarrayEmitter (awaiting availability on main branch).

## Package Structure

```
uq/
├── RFC006.md              # Authoritative specification
├── RFC006_VERIFICATION.md # Compliance analysis
├── CONTEXT.md             # This file (Claude context)
├── inputs.py              # Input parameters: VioPathwayParams, MecillinamParams, GeneKnockoutParams
├── outputs.py             # Output extraction: OutputExtractor, OutputType enum
├── aggregation.py         # Aggregation strategies (1-4): Aggregator, AggregationStrategy
├── wrappers.py            # UQPy/PyTUQ wrappers: SimulationWrapper, PrecomputedWrapper
├── sensitivity.py         # PCE-based GSA: SensitivityAnalyzer, SobolIndices, PCESurrogate
├── cell_cycle.py          # Cell cycle stratification: CellCycleAggregator, CellCycleVariableComputer
└── docs/                  # Sphinx documentation
```

## Architecture (per RFC006 Section 4)

The implementation parametrizes three computational steps:

### A. Selection/Extraction (`outputs.py`)

Extract subsampled time points and variables from simulation trajectories:

```python
from uq.outputs import OutputExtractor, OutputType

extractor = OutputExtractor(conn, history_sql, config_sql, sim_data)
outputs = extractor.extract_all(
    output_types=[OutputType.TRANSCRIPTOME, OutputType.EXCHANGE_FLUXES],
    generation_lower_bound=2,
    time_lower_bound=100.0,
)
```

### B. Temporal Aggregation (`aggregation.py`)

Aggregate into output variables Y using one of four strategies:

```python
from uq.aggregation import Aggregator, AggregationStrategy

aggregator = Aggregator(conn, history_sql, config_sql, sim_data)

# Strategy 1: Uniform (baseline)
agg_uniform, ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

# Strategy 2: By generation
agg_by_gen, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

# Strategy 3: By lineage seed
agg_by_seed, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

# Strategy 4: By cell cycle
from uq.cell_cycle import CellCycleAggregator
cc_aggregator = CellCycleAggregator(conn, history_sql, config_sql, variable_type="mass_based")
profile = cc_aggregator.get_cell_cycle_profile("listeners__rna_counts__mRNA_cistron_counts")
```

### C. Sensitivity Analysis (`sensitivity.py`)

Apply PCE-based global sensitivity analysis (using UQPy or PyTUQ):

```python
from uq.sensitivity import SensitivityAnalyzer
from uq.inputs import InputParameterSpace

param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
analyzer = SensitivityAnalyzer(param_space, wrapper)

sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
    polynomial_order=3,
    n_samples=100,
    use_uqpy=True,
)

top_params = sobol_indices.get_most_influential(n=5)
```

## Input Variables (per RFC006)

Defined in `inputs.py`:

- **VioPathwayParams**: Violacein pathway presence, expression, translation efficiency
- **MecillinamParams**: Mecillinam antibiotic concentration timeline
- **GeneKnockoutParams**: Gene deletions and translation knockouts

## Output Variables (per RFC006)

Defined in `outputs.py` via `OutputType` enum:

- `TRANSCRIPTOME`: mRNA cistron counts
- `PROTEOME`: Protein monomer counts
- `METABOLIC_FLUXES`: All reaction fluxes
- `EXCHANGE_FLUXES`: Exchange reaction fluxes
- `HIGHER_ORDER_PROPERTIES`: Mass, volume, growth rate

## Libraries (per RFC006 footnotes)

- **UQPy** [1]: Primary library for PCE surrogate construction and Sobol sensitivity analysis
- **PyTUQ** [2]: Alternative library with compatible interface

[1] https://uqpyproject.readthedocs.io/en/latest/sensitivity/index.html
[2] https://sandialabs.github.io/pytuq/autoapi/pytuq/gsa/gsa/index.html

## Additional Features

### Variance Decomposition

```python
from uq.aggregation import compute_variance_decomposition

decomposition = compute_variance_decomposition(
    aggregated_by_gen,
    aggregated_by_seed,
    aggregated_uniform,
)
# Returns: total_variance, between_generation_variance, between_seed_variance,
#          generation_fraction, seed_fraction
```

### Cell Cycle Variables (Phase 2)

Three implementations provided:

1. **MassBasedCellCycleVariable**: Normalized log-mass ratio
2. **DNAReplicationCellCycleVariable**: DNA mass progression
3. **CellAngleCellCycleVariable**: 2D projection in (mass, growth_rate) space

Custom variables can be registered via `register_cell_cycle_variable()`.

**Note:** Consensus on a specific cell cycle variable is deferred to a separate RFC per the Phase 2 plan.

## Ownership

- **Implementation**: Alex Patrie
- **UQ Advisor**: Boyan Beronov
- **Biological Data Design**: Chris Long

## Future Considerations (Parking Lot)

- Experimental data ingestion for UQ validation
- Strain design optimization under uncertainty
- ML surrogate models for rapid UQ evaluation
- Integration with XarrayEmitter (when available on main branch)
