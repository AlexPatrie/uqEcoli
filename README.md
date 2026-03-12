# vEcoli Uncertainty Quantification (UQ) Framework

A comprehensive framework for tracking prediction confidence in vEcoli whole-cell simulations, implementing **Milestone 08.4.2** (extensibility) and laying groundwork for **Milestone 10.2.3** (population-level perturbation analysis).

## The 7-Step Pipeline

The UQ framework follows a 7-step workflow specified by **RFC006** (`readmes/RFC006.md`):

```
  Parameter Space ──► Load Data ──► Aggregate (4 strategies) ──► Variance Decomposition
                                                                        │
  Sobol Indices ◄── PCE Surrogate ◄── Morris Screening ◄───────────────┘
       │
       ▼
  GSA-informed cell cycle variable ──► Per-stage sensitivity (feedback loop)
```

| Step | Function | Output |
|------|----------|--------|
| 1 | `InputParameterSpaceVecoli()` | Parameter space with bounds |
| 2 | `load_dataset()` | DataFrame of simulation outputs |
| 3a-c | `aggregate_uniformly/by_generation/by_seed()` | AggregatedOutput per strategy |
| 3d | `calculate_cell_cycle()` | CellCycleResult |
| 4 | `compute_variance_decomposition()` | Generation/seed/residual fractions |
| 5 | `prescreen_parameters()` (Morris) | Top K influential parameters |
| 6 | `generate_surrogate()` | PCESurrogate (instant predictions) |
| 7 | Sobol from PCE coefficients | SobolIndices (parameter rankings) |

See [`uq/PIPELINE.md`](uq/PIPELINE.md) for the full workflow diagram, RFC006 traceability, and requirements checklist.

## RFC006 Compliance

This package implements the UQ framework specified in **RFC006** (`readmes/RFC006.md`). The test suite provides explicit verification via `pytest tests/test_milestone_084_2.py -v`.

### Status: 14 of 17 requirements complete

**Phase 1 (MS-08.4.2)** — All software requirements met:
- [x] UQ framework for tracking prediction confidence
- [x] Four aggregation strategies (uniform, by generation, by seed, by cell cycle)
- [x] Variance decomposition across strategies
- [x] Scientific input variables (vio, mecillinam, knockouts)
- [x] Output extraction (transcriptome, proteome, fluxes, properties)
- [x] Input→output wrapper functions (UQPy/PyTUQ compatible)
- [x] PCE surrogate method with Sobol indices
- [x] Morris screening for high-dimensional parameter spaces
- [x] Parametrized extraction, aggregation, and method selection (RFC006 §4 steps A, B, C)
- [x] Bulk-to-single-cell mapping via aggregation + decomposition
- [x] Population-level perturbation analysis foundation (Milestone 10)
- [ ] **Apply pipeline to real use cases and produce report** (Activity 5 — framework ready, awaiting execution)

**Phase 2 (CD2 / Milestone 10)** — Software complete, consensus pending:
- [x] Cell cycle stratification with 5 variable implementations (mass, DNA, cell angle, Koopman, GSA-informed)
- [x] GSA→cell cycle feedback loop (`GSAInformedCellCycleVariable`)
- [ ] **Write consensus RFC on cell cycle variable choice** (Activity 6)
- [ ] **Implement consensus approach with per-stage GSA** (Activity 7 — depends on Activity 6)

The shared technical blocker for the remaining items is `PCESurrogate.compute_sobol_indices()` — analytical Sobol computation from PCE coefficients. See [`uq/PIPELINE.md` § What's Still Missing](uq/PIPELINE.md) for detailed gap analysis.

## Development Environment

**IMPORTANT:** Always use `uv run` instead of `python` or `python3` for this repository:

```bash
# Correct
uv run python script.py
uv run pytest tests/
uv run marimo run tutorials/03b_reactive_sensitivity.py

# Incorrect - do not use
python script.py
python3 script.py
```

## Tutorials

Interactive marimo notebooks are available in the `tutorials/` directory:

```bash
# Run a tutorial
uv run marimo run tutorials/01_introduction.py
```

| Tutorial | Description | Topics |
|----------|-------------|--------|
| **01_introduction.py** | Getting started with UQ | Input parameters, parameter spaces, basic concepts |
| **02_aggregation_strategies.py** | Aggregation and variance | Four strategies, variance decomposition, data visualization |
| **03_sensitivity_analysis.py** | PCE and Sobol indices | Surrogate models, sensitivity ranking, multi-output analysis |
| **03b_reactive_sensitivity.py** | **Reactive parameter exploration** | Real-time parameter → output timeseries visualization |
| **04_cell_cycle_and_koopman.py** | Advanced analysis | **Koopman cell cycle variable**, DMD, spectral mode visualization |
| **05_music_notation.py** | Musical notation for cellular dynamics | Koopman → notes (Layer 1), UQ pipeline → score (Layer 2) |
| **06_calculate_cell_cycle.py** | Cell cycle computation | Koopman, GSA-informed, cell cycle variable from timeseries |
| **07_full_workflow.py** | **Complete sensitivity workflow** | Morris screening → PCE → variance decomposition → reactive exploration |

## Project Structure

```
uqEcoli/
├── uq/                         # Main UQ package
│   ├── __init__.py             # Public API exports
│   ├── inputs.py               # Input parameters (VioPathway, Mecillinam, Knockouts)
│   ├── outputs.py              # Output extraction (Transcriptome, Proteome, Fluxes)
│   ├── aggregation.py          # Four aggregation strategies + variance decomposition
│   ├── sensitivity.py          # Sobol/Morris sensitivity analysis + PCESurrogate
│   ├── pce.py                  # PCE math: fitting, basis generation, surrogate construction
│   ├── cell_cycle.py           # Cell cycle stratification (5 implementations)
│   ├── wrappers.py             # UQPy/PyTUQ wrapper functions
│   ├── koopman.py              # Koopman spectral analysis (DMD)
│   ├── pipeline.py             # Pipeline orchestrator (WIP)
│   ├── cli.py                  # CLI entry point (WIP)
│   ├── io.py                   # Serialization for dataclasses with numpy arrays
│   ├── models.py               # Core dataclasses (Parameter, PCEConfig, etc.)
│   └── PIPELINE.md             # Full workflow diagram + RFC006 requirements checklist
├── apollo/                     # Musical notation for cellular dynamics
│   ├── types.py                # CellularScore, CellularNote, LosslessScore
│   ├── mappings.py             # Bijective frequency↔pitch, amplitude↔dynamics
│   ├── encoding.py             # Spectrum → Score encoding
│   ├── decoding.py             # Score → Spectrum reconstruction
│   ├── m21.py                  # music21 integration (MusicXML export)
│   ├── uq_score.py             # UQ pipeline → musical score (Layer 2)
│   └── README.md               # Apollo documentation (Layer 1 + Layer 2)
├── tutorials/                  # Interactive marimo notebooks
│   ├── 01_introduction.py      # Getting started
│   ├── 02_aggregation_strategies.py  # Aggregation and variance
│   ├── 03_sensitivity_analysis.py    # PCE and Sobol
│   ├── 03b_reactive_sensitivity.py   # Reactive parameter → timeseries
│   ├── 03c_reactive_sensitivity_generalized.py  # Generalized reactive exploration
│   ├── 04_cell_cycle_and_koopman.py  # Advanced analysis
│   ├── 05_music_notation.py    # Apollo musical encoding (Layer 1 + Layer 2)
│   ├── 06_calculate_cell_cycle.py    # Cell cycle computation + GSA-informed
│   └── 07_full_workflow.py     # Morris → PCE → reactive exploration
├── tests/                      # Test suite
│   ├── conftest.py             # Fixtures (synthetic + real data)
│   ├── test_milestone_084_2.py # Explicit RFC006 compliance tests
│   ├── test_inputs.py          # Input parameter tests
│   ├── test_aggregation.py     # Aggregation strategy tests
│   ├── test_sensitivity.py     # Sensitivity analysis tests
│   ├── test_cell_cycle.py      # Cell cycle variable tests
│   ├── test_e2e.py             # End-to-end workflow tests
│   ├── test_real_data.py       # Full pipeline tests with real data
│   └── test_koopman.py         # Koopman analysis tests
├── readmes/                    # Documentation
│   ├── RFC006.md               # Authoritative specification
│   └── CONTEXT.md              # Detailed context document
├── docs/                       # Sphinx documentation
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Overview

This package enables uncertainty quantification by characterizing different types of uncertainty across vEcoli simulations:

- **By cell**: Variance across individual cell trajectories
- **By lineage**: Variance attributable to stochastic seeding (exogenous variance)
- **By generation**: Variance related to convergence towards steady-state growth
- **By cell cycle stage**: Variance across physiological time within a cell's lifespan

The framework maps single-cell simulations to bulk population averages, enabling direct comparison with experimental measurements and supporting the CD2 evaluation requirements.

## Installation

The UQ framework is included with vEcoli. For sensitivity analysis features, install the optional UQ dependencies:

```bash
pip install -e ".[uq]"
```

This installs [UQPy](https://uqpyproject.readthedocs.io/), the primary library used for PCE surrogate construction and Sobol sensitivity analysis.

## Quick Start

### Basic Sensitivity Analysis

```python
from uq import (
    InputParameterSpaceVecoli,
    WrapperConfig,
    SimulationWrapper,
    SensitivityAnalyzer,
    AggregationStrategy,
)

# 1. Define the input parameter space
param_space = InputParameterSpaceVecoli(
    include_vio=True,  # Include violacein pathway parameters
    include_mecillinam=True,  # Include mecillinam antibiotic parameters
    vio_expression_bounds=(0.0, 5.0),  # Expression factor range
    vio_trl_eff_bounds=(0.0, 2.0),  # Translation efficiency range
    mecillinam_conc_bounds=(0.0, 10.0),  # Concentration range (mM)
)

# 2. Configure the simulation wrapper
config = WrapperConfig(
    sim_data_path="/path/to/sim_data.cPickle",
    output_dir="./uq_outputs",
    cache_dir="./uq_cache",
    generations=8,
    aggregation_strategy=AggregationStrategy.UNIFORM,
)

# 3. Create wrapper and analyzer
wrapper = SimulationWrapper(config, param_space)
analyzer = SensitivityAnalyzer(param_space, wrapper)

# 4. Run PCE-based sensitivity analysis
sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
    polynomial_order=3,
    n_samples=100,
)

# 5. Interpret results
print("Most influential parameters:")
for name, value in sobol_indices.get_most_influential(n=5):
    print(f"  {name}: {value:.4f}")
```

### Using Precomputed Results

If you have already run simulations, you can analyze them directly:

```python
from uq import analyze_precomputed_results, AggregationStrategy

sobol_indices, surrogate = analyze_precomputed_results(
    data_dir="./simulation_outputs",
    aggregation_strategy=AggregationStrategy.BY_GENERATION,
    polynomial_order=3,
)
```

## Input Parameters

The framework supports three categories of input parameters that represent scientifically relevant experimental conditions:

### Violacein (vio) Pathway

Controls new gene expression in the model:

```python
from uq import VioPathwayParams

vio_params = VioPathwayParams(
    enabled=True,
    induction_gen=1,              # Generation to induce expression
    expression=2.5,               # Expression factor
    translation_efficiency=1.2,   # Translation efficiency
    condition="basal",            # Media condition
)
```

### Mecillinam Antibiotic

Controls antibiotic stress conditions:

```python
from uq import MecillinamParams

mec_params = MecillinamParams(
    times=[0.0, 3600.0],              # Time points (seconds)
    concentrations=[0.0, 5.0],        # Concentrations (mM)
    knockouts=["murG"],               # Gene knockouts
)
```

### Gene Knockouts

Controls gene deletion experiments:

```python
from uq import GeneKnockoutParams

ko_params = GeneKnockoutParams(
    gene_deletions=["lacZ", "galK"],  # ParCa-level deletions
    translation_knockouts=["murG"],    # Translation-level knockouts
)
```

## Output Variables

The framework extracts and aggregates five categories of simulation outputs:

| Output Type | Description | Column Pattern |
|-------------|-------------|----------------|
| `TRANSCRIPTOME` | mRNA counts per cistron | `listeners__rna_counts__mRNA_cistron_counts` |
| `PROTEOME` | Protein monomer counts | `listeners__monomer_counts` |
| `METABOLIC_FLUXES` | All reaction fluxes | `listeners__fba_results__base_reaction_fluxes` |
| `EXCHANGE_FLUXES` | Exchange reactions only | Filtered from metabolic fluxes |
| `HIGHER_ORDER_PROPERTIES` | Mass, volume, growth rate | `listeners__mass__*` |

### Extracting Outputs Directly

```python
from uq import OutputExtractor, OutputType
from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

conn = create_duckdb_conn()
history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

extractor = OutputExtractor(conn, history_sql, config_sql)

# Extract specific outputs
transcriptome, cistron_ids = extractor.extract_transcriptome(
    generation_lower_bound=2,  # Skip initial generations
    time_lower_bound=100.0,    # Skip transient period
)

# Extract all outputs
outputs = extractor.extract_all(
    output_types=[OutputType.TRANSCRIPTOME, OutputType.EXCHANGE_FLUXES],
)
```

## Aggregation Strategies

The framework implements four aggregation strategies for characterizing different types of uncertainty:

### Strategy 1: Uniform Aggregation (Baseline)

Averages across all cells and all time points—the "bulk" population average:

```python
from uq import Aggregator, AggregationStrategy

aggregator = Aggregator(conn, history_sql, config_sql)
result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

print(f"Mean: {result.mean}")
print(f"Std: {result.std}")
print(f"N samples: {result.n_samples}")
```

### Strategy 2: Stratified by Generation

Controls for convergence towards steady-state growth:

```python
result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

# result.mean has shape (n_generations, n_features)
# result.groups contains generation numbers
for gen, mean in zip(result.groups, result.mean):
    print(f"Generation {gen}: mean = {mean[:5]}...")
```

### Strategy 3: Stratified by Lineage Seed

Controls for exogenous variance (stochastic seeding):

```python
result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

# result.mean has shape (n_seeds, n_features)
# result.groups contains lineage seed values
```

### Strategy 4: Stratified by Cell Cycle Stage

Enables "phenotypic" sensitivity analysis across physiological time:

```python
from uq import CellCycleAggregator

cc_agg = CellCycleAggregator(
    conn, history_sql, config_sql,
    variable_type="mass_based",  # or "dna_replication", "cell_angle"
    n_stages=10,
)

profile = cc_agg.get_cell_cycle_profile(
    "listeners__rna_counts__mRNA_cistron_counts"
)

print(f"Cell cycle variable: {profile['cell_cycle_variable']}")
print(f"Stages: {profile['stage']}")
print(f"Mean per stage: {profile['mean']}")
```

## Variance Decomposition

Decompose total variance into components attributable to different factors:

```python
from uq import Aggregator, AggregationStrategy, compute_variance_decomposition

aggregator = Aggregator(conn, history_sql, config_sql)

# Get aggregated results for each strategy
agg_uniform, _ = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
agg_by_gen, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)
agg_by_seed, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

# Decompose variance
decomposition = compute_variance_decomposition(
    agg_by_gen,
    agg_by_seed,
    agg_uniform,
)

print(f"Total variance: {decomposition['total_variance']}")
print(f"Generation fraction: {decomposition['generation_fraction']}")
print(f"Seed fraction: {decomposition['seed_fraction']}")
```

## Sensitivity Analysis Methods

### PCE Surrogate Method (Recommended)

Polynomial Chaos Expansion builds a surrogate model and computes Sobol indices analytically:

```python
from uq import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(param_space, wrapper)
sobol, pce = analyzer.analyze_with_pce(
    polynomial_order=3,
    n_samples=100,
    use_uqpy=True,  # Use UQPy (default) or PyTUQ
)

# First-order indices: main effects
print("First-order indices:", sobol.first_order)

# Total-order indices: includes interactions
print("Total-order indices:", sobol.total_order)
```

### Direct Sobol Analysis

Monte Carlo-based Sobol analysis (more samples required):

```python
sobol = analyzer.analyze_with_sobol(
    n_samples=1024,
    calc_second_order=True,  # Include interaction effects
)
```

### Morris Screening (for High-Dimensional Problems)

For parameter spaces with many parameters (>10), Morris screening provides an efficient pre-screening step to identify which parameters are influential before running detailed PCE analysis:

```python
from uq import SensitivityAnalyzer, MorrisIndices

# Stage 1: Morris screening (cheap - O(n) evaluations)
morris = analyzer.analyze_with_morris(
    n_trajectories=20,  # More = more stable (typical: 10-50)
    n_levels=4,         # Grid resolution (typical: 4-8)
)

# View results
print(morris.summary())

# Get parameters for detailed analysis
important = morris.get_screening_candidates(top_n=5)
print(f"Focus PCE analysis on: {important}")

# Classification: negligible, linear, or nonlinear/interactions
classification = morris.classify_parameters()
```

**Cost comparison** (20 parameters):
- Morris (20 trajectories): ~420 evaluations
- PCE (order 2): ~500-1000 evaluations
- Sobol (Monte Carlo): ~50,000+ evaluations

**Integration with tutorials**: Export screening results to the reactive tutorial format:

```python
# Convert to PARAMETER_CONFIG for tutorial 03c
PARAMETER_CONFIG = morris.to_parameter_config(
    parameter_bounds=param_space.parameter_bounds,
    top_n=5,
)
```

### PCE Surrogate Prediction

Once trained, PCE surrogates can make **instant predictions** without running simulations:

```python
# After training via analyzer.analyze_with_pce()
params = np.array([2.5, 1.0, 5.0])  # vio_expression, vio_trl_eff, mecillinam
output = pce_surrogate.predict(params)

# Batch prediction
param_batch = np.random.uniform(
    pce_surrogate.input_bounds[:, 0],
    pce_surrogate.input_bounds[:, 1],
    size=(100, 3)
)
outputs = pce_surrogate.predict(param_batch)

# With uncertainty estimate
mean, std = pce_surrogate.predict_with_uncertainty(param_batch)
```

This enables **reactive parameter exploration** in Marimo notebooks - see `tutorials/03b_reactive_sensitivity.py`.

## Cell Cycle Variables

Cell cycle variables implement aggregation strategy #4 from RFC006, enabling "phenotypic" sensitivity analysis across the physiological time dimension within a cell's lifespan.

### GSA-Informed Cell Cycle Variable (RFC006 Compliant)

Per RFC006 Section 3: *"The choice of the 'cell cycle variable' will be informed by the sensitivity analyses (1-3)."*

The **`GSAInformedCellCycleVariable`** implements this requirement by:
1. Running variance decomposition from strategies 1-3 (uniform, by_generation, by_lineage_seed)
2. Identifying observables where variance is NOT explained by generation/seed (i.e., cell-cycle-related variance)
3. Using those observables to compute the Koopman cell cycle variable

```python
from uq import (
    GSAInformedCellCycleVariable,
    Aggregator,
    AggregationStrategy,
)

# Step 1: Run aggregation for strategies 1-3
aggregator = Aggregator(conn, history_sql, config_sql)
agg_uniform, _ = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
agg_by_gen, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)
agg_by_seed, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

# Step 2: Create GSA-informed cell cycle variable
gsa_cc = GSAInformedCellCycleVariable(
    aggregated_uniform=agg_uniform,
    aggregated_by_gen=agg_by_gen,
    aggregated_by_seed=agg_by_seed,
    observable_names=observable_names,
    expected_cycle_time=3600.0,  # Expected cell cycle in seconds
)

# Step 3: Compute cell cycle variable
cc_var = gsa_cc.compute(trajectory_data)

# Step 4: Inspect which observables were selected by GSA
print(f"Selected observables: {gsa_cc.selected_observables}")
print(f"Relevance summary: {gsa_cc.get_relevance_summary()}")
```

**The workflow:**

```
Step 1: Run GSA (strategies 1-3)
        ↓
        Variance decomposition:
        - What fraction explained by generation?
        - What fraction explained by seed?
        - Residual fraction = cell-cycle-related!
        ↓
Step 2: Identify cell-cycle-relevant observables
        ↓
        Select observables with high residual variance
        (not explained by gen/seed)
        ↓
Step 3: Compute Koopman cell cycle variable
        ↓
        Uses ONLY the GSA-selected observables
        ↓
Step 4: Result includes GSA metadata
        ↓
        cc_var.metadata["gsa_informed"] = True
        cc_var.metadata["selected_observables"] = [...]
        cc_var.metadata["relevance_scores"] = {...}
```

This closes the loop required by RFC006 between sensitivity analyses (1-3) and the cell cycle variable (strategy 4).

### Convenience Function

For a streamlined workflow:

```python
from uq import run_gsa_informed_cell_cycle_analysis

# All-in-one function
cc_var, relevance_result = run_gsa_informed_cell_cycle_analysis(
    aggregated_uniform=agg_uniform,
    aggregated_by_gen=agg_by_gen,
    aggregated_by_seed=agg_by_seed,
    trajectory_data=data,
    observable_names=observable_names,
    expected_cycle_time=3600.0,
)

# Access results
print(f"Selected observables: {relevance_result.relevant_observables}")
print(f"Variance by strategy: {relevance_result.variance_by_strategy}")
```

### Koopman Eigenfunction Phase (Standalone)

If you want to use the Koopman approach **without** GSA-informed observable selection (e.g., for quick prototyping), you can use `KoopmanCellCycleVariable` directly:

```python
from uq import KoopmanCellCycleVariable, CellCycleAggregator

# Use Koopman-based cell cycle variable
cc_agg = CellCycleAggregator(
    conn, history_sql, config_sql,
    variable_type="koopman",  # Uses KoopmanCellCycleVariable
    n_stages=10,
)

profile = cc_agg.get_cell_cycle_profile("listeners__rna_counts__mRNA_cistron_counts")

# Or use directly for more control
koopman_cc = KoopmanCellCycleVariable(
    expected_cycle_time=3600.0,  # Expected cell cycle in seconds
    frequency_tolerance=0.3,      # Tolerance for matching cell cycle frequency
    use_edmd=True,                # Use Extended DMD for nonlinear dynamics
)

result = koopman_cc.compute(data)
print(f"Cell cycle mode frequency: {koopman_cc.cell_cycle_mode.frequency}")
print(f"Estimated period: {koopman_cc.cell_cycle_mode.period}")
```

### Why Koopman is Ideal for Cell Cycle Variables

RFC006 Section 3 specifies that aggregation strategy #4 requires:

> *"the definition of a low-dimensional (possibly scalar) 'cell cycle variable' computed from omics variables... used for deterministically binning simulation data into cell stages"*

The RFC mentions **"cell angle"** as an established example and notes that *"any deterministic function of relevant process variables inside vEcoli may be considered if it has approximately cyclic behaviour."*

The Koopman eigenfunction phase is the **ideal approach** for satisfying these RFC006 requirements—superior to both the mentioned "cell angle" approach and ad-hoc heuristic methods. Here's why:

#### 1. Theoretical Foundation

The cell cycle is fundamentally a **periodic dynamical process**. Koopman operator theory provides the natural mathematical framework for analyzing such systems:

- **Koopman eigenfunctions** are the intrinsic coordinates of dynamical systems
- For periodic dynamics, the eigenfunction associated with the fundamental frequency has a phase that advances uniformly through the cycle
- This phase is **invariant to coordinate choice**—it captures the true "progress" through the cycle regardless of which observables we measure

In contrast, heuristic approaches (mass-based, DNA-based) are coordinate-dependent and may not capture the true cyclic structure.

#### 2. Data-Driven Discovery

The Koopman approach **discovers** the cell cycle from data rather than assuming it:

| Aspect | Heuristic Methods | Koopman Approach |
|--------|-------------------|------------------|
| Mechanism | Assumes specific growth law (exponential mass, DNA replication timing) | No mechanistic assumptions |
| Periodicity | Assumes cycle exists | Verifies and identifies periodic modes |
| Frequency | Must be specified or inferred | Automatically extracted from spectrum |
| Validation | Requires external validation | Self-validating (mode must be oscillatory) |

This is critical for vEcoli simulations where the emergent cell cycle may deviate from idealized models.

#### 3. Superior to "Cell Angle" (RFC006's Example)

RFC006 mentions "cell angle" as an established example. While cell angle has been used in the literature, **Koopman is superior** for several reasons:

| Aspect | Cell Angle | Koopman Eigenfunction Phase |
|--------|------------|----------------------------|
| **Definition** | 2D PCA projection of (mass, growth_rate) | Eigenfunction of dominant oscillatory mode |
| **Dimensionality** | Requires choosing 2 specific observables | Uses all available observables jointly |
| **Periodicity** | Geometric (may not align with true cycle) | Spectral (captures actual periodic dynamics) |
| **Coordinate dependence** | Sensitive to which variables are chosen | Invariant to observable choice |
| **Validation** | No built-in check | Mode frequency must match cell cycle |
| **Theoretical basis** | Empirical/heuristic | Dynamical systems theory |

The cell angle approach assumes that plotting mass vs. growth rate creates a circular trajectory, and the angle in this 2D space represents cycle progress. This is a **geometric approximation** that may not capture the true dynamical structure. The Koopman eigenfunction, by contrast, is the **mathematically correct** coordinate for periodic dynamics.

#### 4. Natural Phase Wrapping

RFC006 requires the cell cycle variable to map each cell state to a value in [0, 1] that wraps once per cycle. The Koopman eigenfunction phase **naturally satisfies this requirement**:

```
φ(x) = arg(ψ(x)) / 2π  ∈ [0, 1]
```

where `ψ(x)` is the Koopman eigenfunction. This phase:
- Advances monotonically through the cycle
- Wraps from 1 back to 0 at cell division
- Is **deterministic** (same state → same phase)
- Is **continuous** (nearby states → nearby phases)

Heuristic methods can fail these properties at cycle boundaries or during non-exponential growth phases.

#### 5. Robustness to Noise and Perturbations

DMD extracts the **dominant coherent structures** from noisy data:

- High-energy modes (including the cell cycle) are reliably identified
- Noise distributes across many low-energy modes that are filtered out
- The cell cycle mode is identified by its characteristic frequency, providing a consistency check

This is especially important for stochastic whole-cell simulations where individual trajectories are noisy.

#### 6. Multi-Observable Integration

The Koopman approach naturally integrates information from **multiple observables**:

```python
koopman_cc = KoopmanCellCycleVariable(
    observable_columns=[
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__dna_mass",
        "listeners__fba_results__growth",
    ],
)
```

The DMD finds the cell cycle mode that **best explains the joint dynamics** of all observables, rather than relying on a single proxy variable.

#### 7. Interpretable Diagnostics

The Koopman approach provides rich diagnostic information:

```python
result = koopman_cc.compute(data)
mode = koopman_cc.cell_cycle_mode

print(f"Detected frequency: {mode.frequency:.4f} Hz")
print(f"Estimated cycle time: {mode.period:.1f} seconds")
print(f"Mode stability: {mode.growth_rate:.4f}")  # Should be ~0 for limit cycle
print(f"Is oscillatory: {mode.is_oscillatory}")   # Should be True
```

If the cell cycle mode has unexpected properties (wrong frequency, decaying, not oscillatory), this indicates a problem with the simulation or data—providing built-in validation.

#### 8. Satisfies RFC006's IV&V Requirement

RFC006 specifies that the cell cycle variable should have *"a quantitative relationship... between the input variables to this 'cell cycle variable' and omics measurements from IV&V."*

The Koopman approach **naturally satisfies this requirement**:

- **Mode shape analysis**: The Koopman mode shape reveals exactly which omics variables (transcriptome, proteome, fluxes) participate in the cell cycle dynamics and with what amplitude
- **Quantitative relationship**: The projection onto the mode eigenvector provides an explicit linear combination of omics measurements
- **Interpretable coefficients**: The mode coefficients can be directly related to experimental measurements

```python
mode = koopman_cc.cell_cycle_mode
# mode.mode contains coefficients for each observable
# These define the quantitative relationship with omics measurements
for i, (obs, coef) in enumerate(zip(observable_names, mode.mode)):
    print(f"{obs}: {np.abs(coef):.3f}")
```

Heuristic methods (mass-based, cell angle) use pre-defined formulas that may not reflect the actual quantitative relationships in the simulation data.

#### 9. Integration with Sensitivity Analysis (RFC006 Phase 2)

RFC006 specifies that the cell cycle variable choice should be *"informed by the sensitivity analyses (1-3)"*. The Koopman approach **directly supports this**:

- **Spectral sensitivity**: We can measure how input parameters affect the cell cycle mode's frequency, amplitude, and shape
- **Mode stability**: Parameters that destabilize the cell cycle mode (shift eigenvalue off unit circle) are identified
- **Harmonic analysis**: Changes in the harmonic content reveal how parameters affect cell cycle regularity

```python
from uq import KoopmanSensitivityAnalyzer

# Compare cell cycle mode across parameter variations
analyzer = KoopmanSensitivityAnalyzer()
sensitivity = analyzer.spectral_sensitivity(
    X_baseline=baseline_trajectory,
    X_perturbed=[perturbed_trajectory],
    parameter_names=["vio_expression"],
)
# Reveals which parameters most affect cell cycle dynamics
```

This closes the loop between aggregation strategies (1-3) and strategy (4), as RFC006 envisions.

#### 10. Consistency with UQ Framework Goals

RFC006 aims to characterize uncertainty across different aggregation strategies. The Koopman cell cycle variable aligns with this goal:

- **Variance decomposition**: The mode energy tells us how much variance is explained by cell cycle dynamics vs. other modes
- **Sensitivity analysis**: We can measure how input parameters affect the cell cycle frequency and mode shape
- **Phenotypic analysis**: The phase provides a principled stratification for "phenotypic" sensitivity analysis

#### Summary: When to Use Each Approach

| Use Case | Recommended Method |
|----------|-------------------|
| **General analysis** | `KoopmanCellCycleVariable` |
| **Quick prototyping** | `MassBasedCellCycleVariable` |
| **DNA replication focus** | `DNAReplicationCellCycleVariable` |
| **Literature comparison** | `CellAngleCellCycleVariable` |
| **Custom requirements** | `CompositeCellCycleVariable` |

For production UQ analysis per RFC006, **always use the Koopman approach** unless you have a specific reason to use a heuristic method.

### Alternative Implementations

For comparison or specialized use cases, three heuristic implementations are also available:

#### Mass-Based

Tracks progression using normalized log-mass ratio:

```python
from uq import MassBasedCellCycleVariable

cc_var = MassBasedCellCycleVariable()
# Computes: (log(M) - log(M_birth)) / (log(M_div) - log(M_birth))
```

#### DNA Replication-Based

Tracks DNA mass as a proxy for replication progress:

```python
from uq import DNAReplicationCellCycleVariable

cc_var = DNAReplicationCellCycleVariable()
# Labels phases: B_period, C_period, D_period
```

#### Cell Angle

2D projection in (mass, growth_rate) space:

```python
from uq import CellAngleCellCycleVariable

cc_var = CellAngleCellCycleVariable()
# Approximates established "cell angle" literature approach
```

### Custom Cell Cycle Variables

Register custom variables for domain-specific analysis:

```python
from uq import CompositeCellCycleVariable, register_cell_cycle_variable
import numpy as np

def my_compute(data):
    # Custom computation from simulation data
    mass = data["listeners__mass__dry_mass"].to_numpy()
    return (mass - mass.min()) / (mass.max() - mass.min())

custom_var = CompositeCellCycleVariable(
    name="my_variable",
    required_columns=["listeners__mass__dry_mass"],
    compute_func=my_compute,
)

register_cell_cycle_variable("my_variable", custom_var)
```

## Koopman Spectral Analysis

The UQ framework includes Koopman operator methods as a complementary approach to PCE-based sensitivity analysis. Where PCE builds polynomial surrogates, Koopman analysis extracts the fundamental dynamical modes ("harmonics") of the system—providing insight into oscillatory behaviors, growth dynamics, and cell cycle periodicity.

### Dynamic Mode Decomposition (DMD)

Extract dominant modes from simulation trajectories:

```python
from uq import DynamicModeDecomposition, KoopmanSpectrum
import numpy as np

# X is a (n_timesteps, n_observables) trajectory
dmd = DynamicModeDecomposition(rank=10)
dmd.fit(X)

spectrum = dmd.get_spectrum(observable_names=["mass", "growth_rate", "lacZ_mRNA"])

# Examine dominant modes
for mode in spectrum.modes[:5]:
    print(f"Mode: freq={mode.frequency:.4f} Hz, decay={mode.decay_rate:.4f}")
    print(f"  Dominant observables: {mode.dominant_observables[:3]}")
```

### Extended DMD (EDMD) with Dictionary Functions

Use nonlinear dictionaries for richer spectral analysis:

```python
from uq import ExtendedDMD, KoopmanDictionary

# Create dictionary with polynomial and Fourier features
dictionary = KoopmanDictionary(
    polynomial_degree=3,
    include_fourier=True,
    fourier_terms=5,
)

edmd = ExtendedDMD(dictionary=dictionary, rank=20)
edmd.fit(X)

spectrum = edmd.get_spectrum()
print(f"Found {len(spectrum.modes)} Koopman modes")
```

### Spectral Sensitivity Analysis

Measure how perturbations affect the Koopman spectrum:

```python
from uq import KoopmanSensitivityAnalyzer

analyzer = KoopmanSensitivityAnalyzer(rank=15)

# Compare baseline vs perturbed trajectories
sensitivity = analyzer.spectral_sensitivity(
    X_baseline=baseline_trajectory,
    X_perturbed=perturbed_trajectory,
    parameter_names=["vio_expression"],
)

print(f"Eigenvalue shift: {sensitivity['eigenvalue_shift']}")
print(f"Mode shape change: {sensitivity['mode_shape_change']}")
print(f"Frequency perturbation: {sensitivity['frequency_perturbation']}")
```

### Cell Cycle Harmonics

Identify cell cycle-related periodic modes in the spectrum:

```python
from uq import CellCycleKoopmanAnalyzer

cc_analyzer = CellCycleKoopmanAnalyzer(
    expected_cycle_time=3600.0,  # Expected cell cycle duration in seconds
    dt=1.0,                       # Time step between samples
    harmonic_tolerance=0.1,       # Frequency matching tolerance
)

# Identify modes that correspond to cell cycle harmonics
cc_modes = cc_analyzer.identify_cell_cycle_modes(spectrum)

print(f"Found {len(cc_modes)} cell cycle-related modes:")
for mode in cc_modes:
    harmonic_order = mode.frequency * 3600.0  # Cycles per cell cycle
    print(f"  {harmonic_order:.1f}x harmonic, decay={mode.decay_rate:.4f}")
```

### Extracting Koopman Features

Convenient function to extract features for downstream analysis:

```python
from uq import extract_koopman_features

features = extract_koopman_features(
    X=trajectory_data,
    n_modes=10,
    include_frequencies=True,
    include_amplitudes=True,
    include_growth_rates=True,
)

# features is a dict with 'frequencies', 'amplitudes', 'growth_rates', 'mode_energies'
print(f"Dominant frequency: {features['frequencies'][0]:.4f} Hz")
print(f"Mode energies: {features['mode_energies']}")
```

### When to Use Koopman vs PCE

| Aspect | PCE/Sobol | Koopman |
|--------|-----------|---------|
| **Best for** | Parameter importance ranking | Understanding system dynamics |
| **Output** | Sensitivity indices | Frequencies, modes, growth rates |
| **Insight** | "Which inputs matter most?" | "What are the dominant dynamics?" |
| **Cell cycle** | Stratify by stage | Identify harmonic modes |
| **Interpretation** | Statistical | Dynamical systems / spectral |

Both approaches complement each other: PCE tells you *which parameters matter*, while Koopman tells you *how the system behaves dynamically*.

## Integration with UQPy and PyTUQ

### UQPy Integration

```python
from uq import create_uqpy_model

model = create_uqpy_model(config, param_space)

# Use with UQPy sampling and sensitivity tools
from UQpy.sampling import LatinHypercubeSampling
from UQpy.distributions import Uniform, JointIndependent

distributions = param_space.get_uqpy_distributions()
joint = JointIndependent(marginals=distributions)
lhs = LatinHypercubeSampling(distributions=joint, nsamples=50)
```

### PyTUQ Integration

```python
from uq import create_pytuq_model

model_func = create_pytuq_model(config, param_space)
lb, ub = param_space.get_pytuq_bounds()

# Use with PyTUQ
# from pytuq.surrogates import PCE
# pce = PCE(order=3, bounds=(lb, ub))
# pce.fit(X, model_func(X))
```

## Example Workflows

### Complete Sensitivity Analysis Pipeline

```python
from uq import (
    InputParameterSpaceVecoli,
    WrapperConfig,
    SimulationWrapper,
    SensitivityAnalyzer,
    AggregationStrategy,
    OutputType,
)

# Configure
param_space = InputParameterSpaceVecoli(include_vio=True, include_mecillinam=True)
config = WrapperConfig(
    sim_data_path="./sim_data.cPickle",
    output_dir="./uq_analysis",
    aggregation_strategy=AggregationStrategy.BY_GENERATION,
    output_types=[OutputType.EXCHANGE_FLUXES, OutputType.HIGHER_ORDER_PROPERTIES],
    generation_lower_bound=2,
)

# Analyze
wrapper = SimulationWrapper(config, param_space)
analyzer = SensitivityAnalyzer(param_space, wrapper)
sobol, pce = analyzer.analyze_with_pce(polynomial_order=3)

# Report
print("=== Sensitivity Analysis Results ===")
print(f"Parameters: {param_space.parameter_names}")
print(f"First-order indices: {sobol.first_order}")
print(f"Total-order indices: {sobol.total_order}")
print(f"\nMost influential:")
for name, val in sobol.get_most_influential(3):
    print(f"  {name}: {val:.4f}")
```

### Multi-Strategy Comparison

```python
from uq import Aggregator, AggregationStrategy, compute_variance_decomposition

strategies = [
    AggregationStrategy.UNIFORM,
    AggregationStrategy.BY_GENERATION,
    AggregationStrategy.BY_LINEAGE_SEED,
]

results = {}
for strategy in strategies:
    agg, ids = aggregator.aggregate_fluxes(strategy, exchange_only=True)
    results[strategy.value] = agg

# Compare variance contributions
decomp = compute_variance_decomposition(
    results["by_generation"],
    results["by_lineage_seed"],
    results["uniform"],
)

print(f"Generation explains {100*decomp['generation_fraction'].mean():.1f}% of variance")
print(f"Seed explains {100*decomp['seed_fraction'].mean():.1f}% of variance")
```

## Full RFC006-proposed workflow:

### Given just experiment_id, outdir_root, and parameter space definition:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                        RFC006 FULL UQ WORKFLOW                              │
  │                                                                             │
  │  Inputs: experiment_id, outdir_root, parameter_config                       │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 1: Define Parameter Space                                             │
  │  ─────────────────────────────────                                          │
  │  • Create InputParameterSpaceVecoli(vio, mecillinam, knockouts)             │
  │  • Define bounds for each parameter                                         │
  │  • Output: parameter_space with n parameters                                │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 2: Load Simulation Data                                               │
  │  ────────────────────────────                                               │
  │  • load_dataset(experiment_id, outdir_root)                                 │
  │  • Extract output variables (dry_mass, growth, fluxes, etc.)                │
  │  • Output: DataFrame with N data points                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 3: Apply Aggregation Strategies 1-3                                   │
  │  ────────────────────────────────────────                                   │
  │                                                                             │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
  │  │ Strategy 1      │  │ Strategy 2      │  │ Strategy 3      │              │
  │  │ UNIFORM         │  │ BY GENERATION   │  │ BY LINEAGE SEED │              │
  │  │                 │  │                 │  │                 │              │
  │  │ aggregate_      │  │ aggregate_      │  │ aggregate_      │              │
  │  │ uniformly()     │  │ by_generation() │  │ by_seed()       │              │
  │  │                 │  │                 │  │                 │              │
  │  │ → mean, std     │  │ → per-gen stats │  │ → per-seed stats│              │
  │  │   across ALL    │  │   (convergence) │  │   (exogenous    │              │
  │  │   cells/times   │  │                 │  │    variance)    │              │
  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
  │           │                    │                    │                       │
  │           └────────────────────┼────────────────────┘                       │
  │                                ▼                                            │
  │                    AggregatedOutput × 3                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 3b: Cell Cycle Stratification (Strategy 4)                            │
  │  ───────────────────────────────────────────────                            │
  │                                                                             │
  │  • calculate_cell_cycle(experiment_id, outdir_root)                         │
  │      │                                                                      │
  │      ├─► Compute cell cycle variable (mass-based, DNA, Koopman, etc.)       │
  │      ├─► Normalize to [0, 1]                                                │
  │      ├─► Bin into N stages                                                  │
  │      └─► Compute per-stage statistics                                       │
  │                                                                             │
  │  • Output: CellCycleResult with stage_stats, phenotypic_variation_cv        │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 4: Variance Decomposition                                             │
  │  ──────────────────────────────                                             │
  │                                                                             │
  │  • compute_variance_decomposition(agg_uniform, agg_by_gen, agg_by_seed)     │
  │                                                                             │
  │  • Deconvolve uncertainty types:                                            │
  │      ├─► generation_fraction (convergence to steady-state)                  │
  │      ├─► seed_fraction (exogenous/stochastic variance)                      │
  │      └─► residual_fraction (cell-cycle-related variance)                    │
  │                                                                             │
  │  • Output: variance fractions per observable                                │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 5: Morris Screening (O(n) - cheap)                                    │
  │  ───────────────────────────────────────                                    │
  │                                                                             │
  │  • prescreen_parameters(parameter_space, f, n_trajectories=20)              │
  │                                                                             │
  │  • Compute elementary effects for each parameter                            │
  │  • Rank by μ* (mean absolute effect)                                        │
  │  • Select top K influential parameters                                      │
  │                                                                             │
  │  • Output: MorrisIndices, selected_parameters (reduced from n → K)          │
  │                                                                             │
  │  ┌─────────────────┐         ┌─────────────────┐                            │
  │  │  n parameters   │  ────►  │  K parameters   │  (K << n)                  │
  │  │  (10-100+)      │         │  (3-10)         │                            │
  │  └─────────────────┘         └─────────────────┘                            │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 6: PCE Surrogate Fitting                                              │
  │  ─────────────────────────────                                              │
  │                                                                             │
  │  6a. Generate LHS Samples                                                   │
  │      • create_samples(N=sample_size, selected=K_parameters)                 │
  │      • X shape: (N, K)                                                      │
  │                                                                             │
  │  6b. Evaluate Model at Sample Points                                        │
  │      • process_samples(X, f, min_reps, max_reps)                            │
  │      • Handle stochastic outputs with adaptive replicates                   │
  │      • Y shape: (N,) or (N, n_outputs)                                      │
  │                                                                             │
  │  6c. Fit PCE Coefficients                                                   │
  │      • fit_pce_coefficients(X, Y, polynomial_order, method)                 │
  │      • Methods: least_squares, lasso, omp                                   │
  │      • Output: PCEFitResult with coefficients, R², sparsity                 │
  │                                                                             │
  │  6d. Create Surrogate                                                       │
  │      • pce_result.to_surrogate()                                            │
  │      • Output: PCESurrogate (instant predictions)                           │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STEP 7: Sobol Sensitivity Analysis                                         │
  │  ──────────────────────────────────                                         │
  │                                                                             │
  │  • Compute from PCE coefficients:                                           │
  │      ├─► First-order indices S_i (main effects)                             │
  │      ├─► Total-order indices ST_i (includes interactions)                   │
  │      └─► Second-order indices S_ij (pairwise interactions)                  │
  │                                                                             │
  │  • Output: SobolIndices with parameter rankings                             │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  OUTPUTS                                                                    │
  │  ───────                                                                    │
  │                                                                             │
  │  • PCESurrogate: Instant predictions for any parameter combination          │
  │  • SobolIndices: Which parameters matter most                               │
  │  • VarianceDecomposition: Sources of uncertainty                            │
  │  • CellCycleResult: Phenotypic variation across cell cycle                  │
  │  • MorrisIndices: Parameter screening results                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  Summary of Steps:
  ┌──────┬────────────────────────────────────┬────────────────────────────┬─────────────────────────────┐
  │ Step │              Function              │           Input            │           Output            │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 1    │ InputParameterSpaceVecoli()        │ bounds, flags              │ parameter_space             │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 2    │ load_dataset()                     │ experiment_id, outdir_root │ DataFrame                   │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 3a   │ aggregate_uniformly()              │ DataFrame                  │ AggregatedOutput            │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 3b   │ aggregate_by_generation()          │ DataFrame                  │ AggregatedOutput            │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 3c   │ aggregate_by_seed()                │ DataFrame                  │ AggregatedOutput            │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 3d   │ calculate_cell_cycle()             │ experiment_id, outdir_root │ CellCycleResult             │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 4    │ compute_variance_decomposition()   │ 3 AggregatedOutputs        │ variance fractions          │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 5    │ prescreen_parameters()             │ parameter_space, f         │ MorrisIndices, top K params │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 6    │ generate_surrogate() or manual PCE │ K params, f                │ PCESurrogate                │
  ├──────┼────────────────────────────────────┼────────────────────────────┼─────────────────────────────┤
  │ 7    │ (from PCE coefficients)            │ PCEFitResult               │ SobolIndices                │
  └──────┴────────────────────────────────────┴────────────────────────────┴─────────────────────────────┘

## Apollo: Musical Notation for Cellular Dynamics

The `apollo/` package encodes computational biology results as Western musical scores. It operates at two levels:

**Layer 1 — Koopman Mode → Note**: Maps individual DMD eigenvalues to individual musical notes. Frequency → pitch, amplitude → dynamic, growth rate → duration. Bijective (lossless round-trip possible).

**Layer 2 — UQ Pipeline → Score** (`apollo/uq_score.py`): Maps the *entire* sensitivity analysis output into a musical score. Sobol indices → dynamics (pp–ff), variance decomposition → registral balance (bass/tenor/treble), cell cycle stages → beats, PCE surrogate → the score itself.

This is not metaphor — PCE coefficients are a spectral decomposition in an orthogonal polynomial basis, and Sobol indices partition variance exactly as a power spectrum partitions energy. See [`apollo/README.md`](apollo/README.md) for the full structural correspondence table.

```python
from apollo.uq_score import encode_sensitivity

score = encode_sensitivity(
    sobol_first_order=sobol.first_order,
    sobol_total_order=sobol.total_order,
    parameter_names=param_space.parameter_names,
    variance_decomposition=decomp,
    cell_cycle_time=3600.0,
    output_name="listeners__mass__dry_mass",
)

print(score.to_ascii())    # ASCII-rendered score
print(score.read_aloud())  # Natural language "program notes"
```

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `InputParameterSpace` | `uq.inputs` | Defines parameter ranges for sensitivity analysis |
| `UQInputParameters` | `uq.inputs` | Container for complete input parameter set |
| `OutputExtractor` | `uq.outputs` | Extracts outputs from Parquet data |
| `Aggregator` | `uq.aggregation` | Implements aggregation strategies 1-3 |
| `CellCycleAggregator` | `uq.cell_cycle` | Implements aggregation strategy 4 |
| `GSAInformedCellCycleVariable` | `uq.cell_cycle` | **RFC006-compliant** GSA-informed cell cycle variable |
| `KoopmanCellCycleVariable` | `uq.cell_cycle` | Koopman eigenfunction-based cell cycle variable |
| `CellCycleRelevanceResult` | `uq.sensitivity` | Results from GSA cell cycle relevance analysis |
| `SimulationWrapper` | `uq.wrappers` | Runs simulations for sensitivity analysis |
| `PrecomputedWrapper` | `uq.wrappers` | Uses existing simulation results |
| `SensitivityAnalyzer` | `uq.sensitivity` | PCE and Sobol sensitivity analysis |
| `PCESurrogate` | `uq.sensitivity` | Trained surrogate for instant predictions |
| `MorrisIndices` | `uq.sensitivity` | Morris screening results |
| `SobolIndices` | `uq.sensitivity` | Sobol sensitivity index results |
| `SensitivityScore` | `apollo.uq_score` | UQ analysis encoded as musical score |
| `DynamicModeDecomposition` | `uq.koopman` | Standard DMD for Koopman spectral analysis |
| `ExtendedDMD` | `uq.koopman` | EDMD with dictionary functions |
| `KoopmanSensitivityAnalyzer` | `uq.koopman` | Spectral sensitivity analysis |
| `CellCycleKoopmanAnalyzer` | `uq.koopman` | Cell cycle harmonic identification |

### Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `run_sensitivity_analysis` | `uq.sensitivity` | Complete analysis workflow |
| `analyze_precomputed_results` | `uq.sensitivity` | Analyze existing results |
| `identify_cell_cycle_relevant_observables` | `uq.sensitivity` | **RFC006** Identify CC-relevant observables via GSA |
| `run_gsa_informed_cell_cycle_analysis` | `uq.sensitivity` | **RFC006** Complete GSA→CC variable workflow |
| `compute_variance_decomposition` | `uq.aggregation` | Decompose variance by source |
| `create_uqpy_model` | `uq.wrappers` | Create UQPy-compatible model |
| `create_pytuq_model` | `uq.wrappers` | Create PyTUQ-compatible model |
| `register_cell_cycle_variable` | `uq.cell_cycle` | Register custom cell cycle variable |
| `generate_surrogate` | `uq.pce` | End-to-end Morris → PCE → surrogate pipeline |
| `fit_pce_coefficients` | `uq.pce` | Fit PCE coefficients with LS/LASSO/OMP |
| `calculate_cell_cycle` | `uq.cell_cycle` | Convenience function: compute CC variable + bin + stats |
| `extract_koopman_features` | `uq.koopman` | Extract spectral features from trajectories |
| `encode_sensitivity` | `apollo.uq_score` | Encode UQ pipeline output as musical score |

## Testing

### Running the Full Test Suite

```bash
# Install test dependencies
pip install pytest polars numpy

# Run all tests
pytest tests/ -v

# Run with markers
pytest tests/ -v -m unit          # Unit tests only
pytest tests/ -v -m e2e           # End-to-end tests only
pytest tests/ -v -m milestone     # RFC006 compliance tests
```

### Verifying RFC006 Compliance

```bash
# Run the explicit milestone verification tests
pytest tests/test_milestone_084_2.py -v

# Expected output includes:
# MILESTONE 08.4.2 VERIFICATION COMPLETE
# ✓ Uncertainty characterized by cell (UNIFORM)
# ✓ Uncertainty characterized by lineage (BY_LINEAGE_SEED)
# ✓ Uncertainty characterized by generation (BY_GENERATION)
# ✓ Uncertainty characterized by cell cycle (BY_CELL_CYCLE)
# ...
```

### Test Categories

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit tests for individual components |
| `@pytest.mark.e2e` | End-to-end workflow tests |
| `@pytest.mark.milestone` | Explicit RFC006 requirement verification |
| `@pytest.mark.real_data` | Tests using actual simulation data |
| `@pytest.mark.slow` | Long-running tests |

### Test Data

Tests use both synthetic and real data:

- **Synthetic data**: Generated in `conftest.py` fixtures with realistic structure
- **Real data**: Loaded from `api_integration/sims/api_simulation_default/` when available

## Development

### Dependencies

Core dependencies:
- `numpy` - Numerical operations
- `polars` - DataFrame operations
- `duckdb` - SQL queries on Parquet data

Optional dependencies for sensitivity analysis:
- `UQPy` - Primary PCE/Sobol library
- `PyTUQ` - Alternative PCE library

### Adding Custom Cell Cycle Variables

```python
from uq import CellCycleVariableComputer, CellCycleVariable, register_cell_cycle_variable
import numpy as np
import polars as pl

class MyCustomVariable(CellCycleVariableComputer):
    @property
    def name(self) -> str:
        return "my_custom"

    @property
    def required_columns(self) -> list[str]:
        return ["listeners__mass__dry_mass", "time"]

    def compute(self, data: pl.DataFrame, sim_data=None) -> CellCycleVariable:
        mass = data["listeners__mass__dry_mass"].to_numpy()
        normalized = (mass - mass.min()) / (mass.max() - mass.min())
        return CellCycleVariable(values=normalized, variable_name=self.name)

register_cell_cycle_variable("my_custom", MyCustomVariable())
```

### Contributing

1. Ensure all tests pass: `pytest tests/ -v`
2. Run milestone compliance: `pytest tests/test_milestone_084_2.py -v`
3. Add tests for new functionality
4. Update documentation as needed

## References

1. UQPy Documentation: https://uqpyproject.readthedocs.io/
2. PyTUQ Documentation: https://sandialabs.github.io/pytuq/
3. Sobol Sensitivity Analysis: Sobol, I.M. (2001). "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates"
4. Polynomial Chaos Expansion: Xiu, D. & Karniadakis, G.E. (2002). "The Wiener-Askey polynomial chaos for stochastic differential equations"

## License

This project is part of the vEcoli whole-cell modeling framework.
