# vEcoli Uncertainty Quantification (UQ) Framework

A comprehensive framework for tracking prediction confidence in vEcoli whole-cell simulations, implementing **Milestone 08.4.2** (extensibility) and laying groundwork for **Milestone 10.2.3** (population-level perturbation analysis).

## RFC006 Compliance

This package implements the UQ framework specified in **RFC006** (`readmes/RFC006.md`). The test suite provides explicit verification that all milestone requirements are satisfied.

### Verified Requirements

| Requirement | Description | Test Coverage |
|-------------|-------------|---------------|
| **R1** | Uniform aggregation (baseline) | `test_milestone_084_2.py::TestRequirement1_UniformAggregation` |
| **R2** | Stratified by lineage seed (exogenous variance) | `test_milestone_084_2.py::TestRequirement2_LineageSeedStratification` |
| **R3** | Stratified by generation (convergence) | `test_milestone_084_2.py::TestRequirement3_GenerationStratification` |
| **R4** | Stratified by cell cycle (phenotypic) | `test_milestone_084_2.py::TestRequirement4_CellCycleStratification` |
| **R5** | Single-cell to bulk mapping | `test_milestone_084_2.py::TestRequirement5_SingleCellToBulkMapping` |
| **R6** | Population-level analysis foundation | `test_milestone_084_2.py::TestRequirement6_PopulationLevelAnalysis` |
| **R7** | PCE surrogate method | `test_milestone_084_2.py::TestRequirement7_PCESurrogateMethod` |
| **R8** | Sobol sensitivity indices | `test_milestone_084_2.py::TestRequirement8_SobolIndices` |
| **R9** | UQPy/PyTUQ library support | `test_milestone_084_2.py::TestRequirement9_LibrarySupport` |
| **R10** | Scientific input parameters | `test_milestone_084_2.py::TestRequirement10_ScientificInputs` |

Run `pytest tests/test_milestone_084_2.py -v` to verify compliance.

## Project Structure

```
uqEcoli/
├── uq/                         # Main package
│   ├── __init__.py             # Public API exports
│   ├── inputs.py               # Input parameters (VioPathway, Mecillinam, Knockouts)
│   ├── outputs.py              # Output extraction (Transcriptome, Proteome, Fluxes)
│   ├── aggregation.py          # Four aggregation strategies
│   ├── sensitivity.py          # PCE-based global sensitivity analysis
│   ├── cell_cycle.py           # Cell cycle stratification (Phase 2)
│   ├── wrappers.py             # UQPy/PyTUQ wrapper functions
│   └── koopman.py              # Koopman spectral analysis
├── tests/                      # Test suite
│   ├── conftest.py             # Fixtures (synthetic + real data)
│   ├── test_milestone_084_2.py # Explicit RFC006 compliance tests
│   ├── test_inputs.py          # Input parameter tests
│   ├── test_aggregation.py     # Aggregation strategy tests
│   ├── test_sensitivity.py     # Sensitivity analysis tests
│   ├── test_cell_cycle.py      # Cell cycle variable tests
│   ├── test_e2e.py             # End-to-end workflow tests
│   └── test_koopman.py         # Koopman analysis tests
├── readmes/                    # Documentation
│   ├── RFC006.md               # Authoritative specification
│   └── CONTEXT.md              # Claude context document
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
    InputParameterSpace,
    WrapperConfig,
    SimulationWrapper,
    SensitivityAnalyzer,
    AggregationStrategy,
)

# 1. Define the input parameter space
param_space = InputParameterSpace(
    include_vio=True,                          # Include violacein pathway parameters
    include_mecillinam=True,                   # Include mecillinam antibiotic parameters
    vio_expression_bounds=(0.0, 5.0),          # Expression factor range
    vio_trl_eff_bounds=(0.0, 2.0),             # Translation efficiency range
    mecillinam_conc_bounds=(0.0, 10.0),        # Concentration range (mM)
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

## Cell Cycle Variables

Three built-in cell cycle variable implementations:

### Mass-Based (Default)

Tracks progression using normalized log-mass ratio:

```python
from uq import MassBasedCellCycleVariable

cc_var = MassBasedCellCycleVariable()
# Computes: (log(M) - log(M_birth)) / (log(M_div) - log(M_birth))
```

### DNA Replication-Based

Tracks DNA mass as a proxy for replication progress:

```python
from uq import DNAReplicationCellCycleVariable

cc_var = DNAReplicationCellCycleVariable()
# Labels phases: B_period, C_period, D_period
```

### Cell Angle

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
    InputParameterSpace,
    WrapperConfig,
    SimulationWrapper,
    SensitivityAnalyzer,
    AggregationStrategy,
    OutputType,
)

# Configure
param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
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

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `InputParameterSpace` | `uq.inputs` | Defines parameter ranges for sensitivity analysis |
| `UQInputParameters` | `uq.inputs` | Container for complete input parameter set |
| `OutputExtractor` | `uq.outputs` | Extracts outputs from Parquet data |
| `Aggregator` | `uq.aggregation` | Implements aggregation strategies 1-3 |
| `CellCycleAggregator` | `uq.cell_cycle` | Implements aggregation strategy 4 |
| `SimulationWrapper` | `uq.wrappers` | Runs simulations for sensitivity analysis |
| `PrecomputedWrapper` | `uq.wrappers` | Uses existing simulation results |
| `SensitivityAnalyzer` | `uq.sensitivity` | PCE and Sobol sensitivity analysis |
| `DynamicModeDecomposition` | `uq.koopman` | Standard DMD for Koopman spectral analysis |
| `ExtendedDMD` | `uq.koopman` | EDMD with dictionary functions |
| `KoopmanSensitivityAnalyzer` | `uq.koopman` | Spectral sensitivity analysis |
| `CellCycleKoopmanAnalyzer` | `uq.koopman` | Cell cycle harmonic identification |

### Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `run_sensitivity_analysis` | `uq.sensitivity` | Complete analysis workflow |
| `analyze_precomputed_results` | `uq.sensitivity` | Analyze existing results |
| `compute_variance_decomposition` | `uq.aggregation` | Decompose variance by source |
| `create_uqpy_model` | `uq.wrappers` | Create UQPy-compatible model |
| `create_pytuq_model` | `uq.wrappers` | Create PyTUQ-compatible model |
| `register_cell_cycle_variable` | `uq.cell_cycle` | Register custom cell cycle variable |
| `extract_koopman_features` | `uq.koopman` | Extract spectral features from trajectories |

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
