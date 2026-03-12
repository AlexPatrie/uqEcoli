# RFC006 Full UQ Workflow

```
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
```

## Summary of Steps

| Step | Function | Input | Output | RFC006 Call to Action |
|------|----------|-------|--------|-----------------------|
| 1 | `InputParameterSpaceVecoli()` | bounds, flags | `parameter_space` | RFC006 §4 Activity 1: *"Identify the scientifically most relevant input and output variables. Expected: Inputs: vio pathway presence, mecillinam condition, gene knockouts"* |
| 2 | `load_dataset()` | experiment_id, outdir_root | `DataFrame` | RFC006 §4 Activity 2: *"Enable output of relevant variables"* — *"Outputs: transcriptome, proteome, metabolic fluxes (particularly exchange fluxes), and higher order properties"* |
| 3a | `aggregate_uniformly()` | DataFrame | `AggregatedOutput` | RFC006 §1: *"capture statistics across four types of aggregation: 1. Uniformly across all simulated cells and times (baseline)"* |
| 3b | `aggregate_by_generation()` | DataFrame | `AggregatedOutput` | RFC006 §1: *"2. Stratified by generation (control of convergence towards steady state-growth)"* |
| 3c | `aggregate_by_seed()` | DataFrame | `AggregatedOutput` | RFC006 §1: *"3. Stratified by lineage seed (control of exogenous variance)"* |
| 3d | `calculate_cell_cycle()` | experiment_id, outdir_root | `CellCycleResult` | RFC006 §1: *"4. Stratified by cell cycle stage, according to a physiological variable that is TBD (time course within a cell's lifespan)"* |
| 4 | `compute_variance_decomposition()` | 3 AggregatedOutputs | variance fractions | RFC006 §1: *"This will enable us to deconvolve different types of uncertainty, and to model the relationship between 'bulk' and 'single-cell' attributes more accurately"* |
| 5 | `prescreen_parameters()` | parameter_space, f | MorrisIndices, top K params | RFC006 §4 Activity 4: *"Implement well established global sensitivity analysis methods"* — Morris screening is the prescreening step before PCE. |
| 6 | `generate_surrogate()` or manual PCE | K params, f | `PCESurrogate` | RFC006 §4 Activity 4: *"Expected to use PCE surrogate method for the stochastic function `(sim_data -> SIM output)`"* |
| 7 | (from PCE coefficients) | `PCEFitResult` | `SobolIndices` | RFC006 §3: *"directly applying well established global sensitivity analysis methods for the stochastic function `(sim_data -> SIM output)`, based on the aggregation strategies (1-3)"* |

---

## RFC006 Requirements Checklist

Each item below is taken verbatim or near-verbatim from RFC006. An item is checked only if `uq/` exposes the described functionality with a verifiable code path.

---

### Phase 1 — Satisfying MS-08.4.2

- [x] **"Implement uncertainty quantification framework to track prediction confidence"** (RFC006 §1, MS-08.4.2)

  The `uq/` package as a whole implements this framework. The 7-step pipeline (parameter space -> aggregation -> variance decomposition -> Morris screening -> PCE surrogate -> Sobol indices) constitutes the UQ framework for tracking prediction confidence.

  ```python
  from uq import (
      InputParameterSpaceVecoli,
      SensitivityAnalyzer,
      AggregationStrategy,
  )

  param_space = InputParameterSpaceVecoli(include_vio=True, include_mecillinam=True)
  analyzer = SensitivityAnalyzer(param_space, wrapper=simulation_wrapper)
  sobol_indices, pce_surrogate = analyzer.analyze_with_pce(polynomial_order=3, n_samples=100)

  # Track prediction confidence: surrogate predicts with R² quality metric
  print(f"Surrogate R²: {pce_surrogate.r_squared}")
  top = sobol_indices.get_most_influential(n=5)
  print(f"Most influential parameters: {top}")
  ```

- [x] **"Capture statistics across four types of aggregation: 1. Uniformly across all simulated cells and times (baseline)"** (RFC006 §1)

  Implemented via `Aggregator._aggregate_uniform()` and the `AggregationStrategy.UNIFORM` enum.

  ```python
  from uq.aggregation import Aggregator, AggregationStrategy

  aggregator = Aggregator(conn, history_sql, config_sql, sim_data)
  agg_uniform, cistron_ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

  print(f"Mean shape: {agg_uniform.mean.shape}")  # (n_cistrons,)
  print(f"Std shape: {agg_uniform.std.shape}")     # (n_cistrons,)
  print(f"N samples: {agg_uniform.n_samples}")
  ```

- [x] **"2. Stratified by generation (control of convergence towards steady state-growth)"** (RFC006 §1)

  Implemented via `Aggregator._aggregate_stratified()` with `AggregationStrategy.BY_GENERATION`.

  ```python
  agg_by_gen, cistron_ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

  print(f"Groups (generations): {agg_by_gen.groups}")    # e.g. [0, 1, 2, ...]
  print(f"Per-group means shape: {agg_by_gen.mean.shape}")  # (n_generations, n_cistrons)
  print(f"Per-group stds shape: {agg_by_gen.std.shape}")
  ```

- [x] **"3. Stratified by lineage seed (control of exogenous variance)"** (RFC006 §1)

  Implemented via `Aggregator._aggregate_stratified()` with `AggregationStrategy.BY_LINEAGE_SEED`.

  ```python
  agg_by_seed, cistron_ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

  print(f"Groups (seeds): {agg_by_seed.groups}")
  print(f"Per-seed means shape: {agg_by_seed.mean.shape}")  # (n_seeds, n_cistrons)
  ```

- [x] **"4. Stratified by cell cycle stage, according to a physiological variable [...] (time course within a cell's lifespan)"** (RFC006 §1)

  Implemented via `calculate_cell_cycle()` in `uq/cell_cycle.py` and `AggregationStrategy.BY_CELL_CYCLE`.

  ```python
  from uq import calculate_cell_cycle

  result = calculate_cell_cycle(
      experiment_id="api_simulation_default",
      outdir_root="/path/to/sims",
      variable_type="mass_based",
      n_bins=10,
      output_column="listeners__mass__dry_mass",
  )

  result.print_summary()
  print(f"Phenotypic CV: {result.phenotypic_variation_cv}")
  print(f"N stages with data: {result.n_stages_with_data}")
  print(f"Stage stats: {result.stage_stats}")
  ```

- [x] **"Enable us to deconvolve different types of uncertainty"** (RFC006 §1)

  Implemented via `compute_variance_decomposition()` in `uq/aggregation.py`.

  ```python
  from uq.aggregation import compute_variance_decomposition

  decomp = compute_variance_decomposition(agg_by_gen, agg_by_seed, agg_uniform)

  print(f"Total variance: {decomp['total_variance']}")
  print(f"Generation fraction: {decomp['generation_fraction']}")  # convergence effect
  print(f"Seed fraction: {decomp['seed_fraction']}")              # exogenous stochasticity
  # Residual = 1 - generation_fraction - seed_fraction             # cell-cycle-related
  ```

- [x] **"Identify the scientifically most relevant input and output variables. Expected: Inputs: vio pathway presence, mecillinam condition, gene knockouts. Outputs: transcriptome, proteome, metabolic fluxes (particularly exchange fluxes), and higher order properties"** (RFC006 §4, Activity 1)

  Inputs defined in `uq/models.py`: `VioPathwayParams`, `MecillinamParams`, `GeneKnockoutParams`. Parameterized in `InputParameterSpaceVecoli` (`uq/inputs.py`). Outputs defined in `uq/outputs.py`: `OutputType` enum with `TRANSCRIPTOME`, `PROTEOME`, `METABOLIC_FLUXES`, `EXCHANGE_FLUXES`, `HIGHER_ORDER_PROPERTIES`. Extraction via `OutputExtractor`.

  ```python
  from uq.models import VioPathwayParams, MecillinamParams, GeneKnockoutParams
  from uq.inputs import InputParameterSpaceVecoli
  from uq.outputs import OutputExtractor, OutputType

  # Inputs
  param_space = InputParameterSpaceVecoli(
      include_vio=True,          # vio pathway presence
      include_mecillinam=True,   # mecillinam condition
      knockout_genes=["geneA"],  # gene knockouts
  )
  print(f"Parameters: {param_space.parameter_names}")
  # ['vio_expression', 'vio_trl_eff', 'mecillinam_concentration']

  # Outputs
  extractor = OutputExtractor(conn, history_sql, config_sql, sim_data)
  outputs = extractor.extract_all(output_types=[
      OutputType.TRANSCRIPTOME,
      OutputType.PROTEOME,
      OutputType.EXCHANGE_FLUXES,
      OutputType.HIGHER_ORDER_PROPERTIES,
  ])
  ```

- [x] **"Implement input->output wrapper functions that can be called from numerical libraries"** (RFC006 §4, Activity 3)

  Implemented in `uq/wrappers.py`: `SimulationWrapper`, `PrecomputedWrapper`, `create_uqpy_model()`, `create_pytuq_model()`.

  ```python
  from uq.wrappers import SimulationWrapper, WrapperConfig, create_uqpy_model, create_pytuq_model

  config = WrapperConfig(
      sim_data_path="/path/to/sim_data.cPickle",
      output_dir="./uq_outputs",
      aggregation_strategy=AggregationStrategy.UNIFORM,
  )

  # Direct wrapper (callable from any numerical library)
  wrapper = SimulationWrapper(config, param_space)
  output = wrapper(np.array([2.5, 1.0, 5.0]))  # numpy in, numpy out

  # UQPy-compatible model
  uqpy_model = create_uqpy_model(config, param_space)

  # PyTUQ-compatible model
  pytuq_func = create_pytuq_model(config, param_space)
  ```

- [x] **"Implement well established global sensitivity analysis methods [...] Expected to use PCE surrogate method for the stochastic function `(sim_data -> SIM output)`"** (RFC006 §4, Activity 4)

  Implemented in `uq/sensitivity.py`: `SensitivityAnalyzer.analyze_with_pce()` with both UQPy and PyTUQ backends. PCE fitting in `uq/pce.py`: `fit_pce_coefficients()` with Legendre/Hermite bases and LS/LASSO/OMP solvers.

  ```python
  from uq import SensitivityAnalyzer, InputParameterSpaceVecoli

  param_space = InputParameterSpaceVecoli(include_vio=True, include_mecillinam=True)
  analyzer = SensitivityAnalyzer(param_space, wrapper=wrapper)

  # PCE-based GSA (using UQPy)
  sobol, surrogate = analyzer.analyze_with_pce(polynomial_order=3, n_samples=100, use_uqpy=True)

  # PCE-based GSA (using PyTUQ)
  sobol, surrogate = analyzer.analyze_with_pce(polynomial_order=3, n_samples=100, use_uqpy=False)

  print(f"First-order Sobol indices: {sobol.first_order}")
  print(f"Total-order Sobol indices: {sobol.total_order}")
  ```

- [x] **"*using UQPy or PyTUQ libraries"** (RFC006 §4, footnote)

  Both libraries are supported. `SensitivityAnalyzer.analyze_with_pce(use_uqpy=True|False)` dispatches to `_analyze_pce_uqpy()` or `_analyze_pce_pytuq()`. Factory functions `create_uqpy_model()` and `create_pytuq_model()` in `uq/wrappers.py`.

  ```python
  # UQPy path
  sobol, surrogate = analyzer.analyze_with_pce(use_uqpy=True)

  # PyTUQ path
  sobol, surrogate = analyzer.analyze_with_pce(use_uqpy=False)

  # Or create library-specific model objects directly
  from uq.wrappers import create_uqpy_model, create_pytuq_model
  uqpy_model = create_uqpy_model(config, param_space)
  pytuq_func = create_pytuq_model(config, param_space)
  ```

- [x] **"Explicitly parametrise each of the following steps: A. The selection/extraction of subsampled time points and variables from the emitted simulation trajectories"** (RFC006 §4)

  Implemented via `OutputExtractor.extract_all()` in `uq/outputs.py` with `generation_lower_bound` and `time_lower_bound` filtering parameters.

  ```python
  from uq.outputs import OutputExtractor, OutputType

  extractor = OutputExtractor(conn, history_sql, config_sql, sim_data)
  outputs = extractor.extract_all(
      output_types=[OutputType.TRANSCRIPTOME, OutputType.EXCHANGE_FLUXES],
      generation_lower_bound=2,    # subsampled: skip early generations
      time_lower_bound=100.0,      # subsampled: skip early timepoints
  )
  ```

- [x] **"B. The temporal aggregation of these samples into 'output variables Y' from the perspective of the sensitivity analysis"** (RFC006 §4)

  Implemented via `Aggregator` in `uq/aggregation.py` with the `AggregationStrategy` enum parametrizing the aggregation method.

  ```python
  from uq.aggregation import Aggregator, AggregationStrategy

  aggregator = Aggregator(conn, history_sql, config_sql, sim_data)

  # Parametrized aggregation: change strategy to change how Y is computed
  for strategy in [AggregationStrategy.UNIFORM, AggregationStrategy.BY_GENERATION,
                   AggregationStrategy.BY_LINEAGE_SEED]:
      agg, ids = aggregator.aggregate_transcriptome(
          strategy=strategy,
          generation_lower_bound=2,
      )
      print(f"{strategy.value}: mean shape = {agg.mean.shape}")
  ```

- [x] **"C. The choice of a numerical sensitivity analysis method that is applied to these aggregated output variables"** (RFC006 §4)

  Parametrized via `SensitivityMethod` enum and `SensitivityAnalyzer` methods: `analyze_with_pce()`, `analyze_with_morris()`. PCE solver choice further parametrized in `PCESolverConfig` (least_squares, lasso, omp) and basis type (legendre, hermite).

  ```python
  from uq.sensitivity import SensitivityAnalyzer, SensitivityMethod

  analyzer = SensitivityAnalyzer(param_space, wrapper=wrapper)

  # Choice 1: PCE-based Sobol analysis
  sobol, surrogate = analyzer.analyze_with_pce(polynomial_order=3)

  # Choice 2: Morris screening
  morris = analyzer.analyze_with_morris(n_trajectories=20)
  ```

- [ ] **"Apply the sensitivity analysis for aggregation strategies (1-3) for representative simulations from each use case"** (RFC006 §4, Activity 5)

  *Not yet complete.* The software infrastructure is in place, but the actual application to representative simulations from each use case and the resulting report (slides) have not been produced yet. This is an application task, not a software task.

---

### Phase 2 — Necessary for CD2 and Milestone 10

- [x] **"Develop a cell cycle stratification strategy [...] the definition of a low-dimensional (possibly scalar) 'cell cycle variable' computed from omics variables"** (RFC006 §3, §4 Activity 6)

  Implemented in `uq/cell_cycle.py` via the `CellCycleVariableComputer` ABC and multiple implementations: `MassBasedCellCycleVariable`, `DNAReplicationCellCycleVariable`, `CellAngleCellCycleVariable`, `KoopmanCellCycleVariable`, `GSAInformedCellCycleVariable`. All produce a scalar variable in [0, 1].

  ```python
  from uq.cell_cycle import MassBasedCellCycleVariable, KoopmanCellCycleVariable

  # Mass-based (heuristic)
  mass_cc = MassBasedCellCycleVariable()
  cc_var = mass_cc.compute(data)
  print(f"Cell cycle values: shape={cc_var.values.shape}, range=[0, 1]")

  # Koopman eigenfunction phase (recommended, data-driven)
  koopman_cc = KoopmanCellCycleVariable(expected_cycle_time=3600.0)
  cc_var = koopman_cc.compute(data)
  ```

- [x] **"This variable or 'coordinate' will be used for deterministically binning simulation data into cell stages, in order to then perform a 'phenotypic' sensitivity analysis across the physiological time dimension"** (RFC006 §3)

  Implemented via `CellCycleVariable.to_stage_bins()` and the `CellCycleAggregator` class. The `calculate_cell_cycle()` convenience function performs the full workflow: compute variable, bin into stages, and report per-stage statistics.

  ```python
  from uq import calculate_cell_cycle

  result = calculate_cell_cycle(
      experiment_id="api_simulation_default",
      outdir_root="/path/to/sims",
      variable_type="koopman",
      n_bins=10,
      output_column="listeners__mass__dry_mass",
  )

  # Deterministic binning into cell stages
  print(f"Stage bins: {result.stages}")            # array of bin indices 0..9
  print(f"Per-stage stats: {result.stage_stats}")   # mean, std, n per bin
  print(f"Phenotypic CV: {result.phenotypic_variation_cv}")
  ```

- [x] **"The choice of the 'cell cycle variable' will be informed by the sensitivity analyses (1-3)"** (RFC006 §3)

  Implemented via `GSAInformedCellCycleVariable` in `uq/cell_cycle.py` and `identify_cell_cycle_relevant_observables()` in `uq/sensitivity.py`. The GSA from strategies 1-3 identifies observables with high residual variance (not explained by generation or seed), which are then fed into Koopman DMD for cell cycle mode extraction.

  ```python
  from uq.cell_cycle import GSAInformedCellCycleVariable
  from uq.sensitivity import identify_cell_cycle_relevant_observables

  # Step 1: Identify CC-relevant observables from strategies 1-3
  relevance = identify_cell_cycle_relevant_observables(
      aggregated_uniform=agg_uniform,
      aggregated_by_gen=agg_by_gen,
      aggregated_by_seed=agg_by_seed,
      observable_names=observable_names,
  )

  # Step 2: Create GSA-informed cell cycle variable
  gsa_cc = GSAInformedCellCycleVariable(
      aggregated_uniform=agg_uniform,
      aggregated_by_gen=agg_by_gen,
      aggregated_by_seed=agg_by_seed,
      observable_names=observable_names,
      expected_cycle_time=3600.0,
  )

  # Step 3: Compute — automatically uses only GSA-selected observables
  cc_var = gsa_cc.compute(trajectory_data)
  print(f"Selected observables: {gsa_cc.selected_observables}")
  ```

- [x] **"An established example for such a variable is the 'cell angle', and in principle, any deterministic function of relevant process variables inside vEcoli may be considered if it has approximately cyclic behaviour"** (RFC006 §3)

  `CellAngleCellCycleVariable` implements the cell angle approach. The `register_cell_cycle_variable()` function allows registration of arbitrary custom implementations.

  ```python
  from uq.cell_cycle import CellAngleCellCycleVariable, register_cell_cycle_variable

  # Built-in cell angle implementation
  cell_angle_cc = CellAngleCellCycleVariable()
  cc_var = cell_angle_cc.compute(data)

  # Register a custom cell cycle variable
  class MyCustomCCVariable(CellCycleVariableComputer):
      @property
      def name(self): return "custom_cyclic"
      @property
      def required_columns(self): return ["my_column"]
      def compute(self, data, sim_data=None):
          values = np.sin(2 * np.pi * data["my_column"].to_numpy())
          return CellCycleVariable(values=(values + 1) / 2, normalized=True)

  register_cell_cycle_variable("custom_cyclic", MyCustomCCVariable)
  ```

- [ ] **"A dedicated, separate RFC is written on the topic of defining the cell cycle variable"** (RFC006 §4, Activity 6 measure of completion)

  *Deferred.* The software framework supports multiple cell cycle variable implementations and extensibility, but the separate consensus RFC has not been written.

- [ ] **"Consensus cell cycle approach (from RFC above) is implemented, committed, and reviewed"** (RFC006 §4, Activity 7 measure of completion)

  *Deferred.* Depends on the consensus RFC above. The implementation infrastructure is ready.

---

### Cross-Cutting Requirements

- [x] **"Model the relationship between 'bulk' and 'single-cell' attributes more accurately"** (RFC006 §1)

  The four aggregation strategies (uniform = bulk; by_generation, by_seed, by_cell_cycle = single-cell decomposition) together with `compute_variance_decomposition()` enable quantitative decomposition of how bulk statistics arise from single-cell heterogeneity.

  ```python
  from uq.aggregation import compute_variance_decomposition

  decomp = compute_variance_decomposition(agg_by_gen, agg_by_seed, agg_uniform)

  # How much of "bulk" variance comes from each source:
  print(f"From generation convergence: {decomp['generation_fraction'].mean():.1%}")
  print(f"From lineage stochasticity: {decomp['seed_fraction'].mean():.1%}")
  residual = 1 - decomp['generation_fraction'] - decomp['seed_fraction']
  print(f"From cell cycle / intrinsic: {residual.mean():.1%}")
  ```

- [x] **"Enable our future milestone 10.2.3 — Milestone 10 Implement population-level perturbation analysis, capturing cell heterogeneity and adaptation"** (RFC006 §1, §2)

  The framework provides the foundation: per-generation and per-seed aggregation captures population heterogeneity, variance decomposition quantifies adaptation effects, and the cell cycle stratification captures within-cell-cycle variation. PCE surrogates enable rapid perturbation analysis.

  ```python
  # Population heterogeneity: compare per-seed statistics
  for i, seed in enumerate(agg_by_seed.groups):
      print(f"Seed {seed}: mean={agg_by_seed.mean[i].mean():.2f}, "
            f"std={agg_by_seed.std[i].mean():.2f}")

  # Perturbation analysis via surrogate
  baseline = surrogate.predict(np.array([1.0, 1.0, 0.0]))
  perturbed = surrogate.predict(np.array([3.0, 1.0, 5.0]))
  print(f"Perturbation effect: {perturbed - baseline}")
  ```

- [x] **"Formalizing and extending the recent work in support of CD1 that aggregated model outputs including multi-omics, exchange fluxes, and higher order properties as a 'bulk' population average across time steps per cell and then across cells"** (RFC006 §1)

  The `Aggregator` class provides methods for each output type: `aggregate_transcriptome()`, `aggregate_proteome()`, `aggregate_fluxes(exchange_only=True|False)`, `aggregate_higher_order_properties()`. Uniform aggregation reproduces the CD1 bulk averaging; stratified aggregation extends it.

  ```python
  aggregator = Aggregator(conn, history_sql, config_sql, sim_data)

  # Multi-omics aggregation (extending CD1 bulk approach)
  transcriptome_agg, mrna_ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
  proteome_agg, monomer_ids = aggregator.aggregate_proteome(AggregationStrategy.UNIFORM)
  flux_agg, rxn_ids = aggregator.aggregate_fluxes(AggregationStrategy.UNIFORM, exchange_only=True)
  props = aggregator.aggregate_higher_order_properties(AggregationStrategy.UNIFORM)

  print(f"Transcriptome: {len(mrna_ids)} cistrons")
  print(f"Proteome: {len(monomer_ids)} monomers")
  print(f"Exchange fluxes: {len(rxn_ids)} reactions")
  print(f"Properties: {list(props.keys())}")
  ```

---

## What's Still Missing: Remaining Work to Check Every Box

Three requirements remain unchecked. Below is a deep breakdown of each: what the RFC is actually asking, why it matters, what the real user experience should look like, and working sketches for how to get there.

---

### 1. Run the Pipeline on Real Use Cases and Produce a Report

**The RFC requirement (§4, Activity 5):**
> *"Apply the sensitivity analysis for aggregation strategies (1-3) for representative simulations from each use case."*
>
> **Measure of completion:** *"Report (slides) containing explanation of methods and results on representative simulations"*
>
> **Target date:** Apr 15, 2026

**What the RFC is really asking (big picture):**

The software is built. Activity 5 is the moment where you prove it works on real biology, not synthetic test functions. The RFC is asking: take the three use cases that vEcoli supports (vio pathway induction, mecillinam antibiotic response, gene knockouts), run the full Steps 1-7 pipeline against real simulation data from each, and distill the results into something a biologist or program manager can look at and say "yes, we understand which parameters matter and how confident we are in these predictions."

This is the difference between "we have a framework" and "we have results." It is the deliverable that satisfies MS-08.4.2.

**Why it matters for the overall UQ goal:**

Without this, the framework is an engine without fuel. The RFC's vision (§1) is to *"track prediction confidence"* — but confidence in what? In the specific biological predictions that the vEcoli model makes about gene expression, metabolic fluxes, and growth under perturbation. Activity 5 turns abstract Sobol indices into concrete statements like: "vio expression dominates transcriptome variance (S_T = 0.72), mecillinam concentration dominates exchange flux variance (S_T = 0.65), and growth rate predictions are robust across lineage seeds (seed_fraction < 5%)."

**What the real user experience looks like:**

A researcher (or program reviewer) wants to ask:

> "I ran 32 cells over 8 generations with the vio pathway induced. How confident am I that the predicted exchange fluxes are stable? What's driving the variability I see?"

They should be able to:
1. Point `pipeline.py` at their simulation output directory
2. Get back a structured result object with Sobol indices, variance decomposition fractions, and a surrogate they can query
3. See a report (HTML, PDF, or slides) with figures showing: parameter importance rankings, variance decomposition pie charts, surrogate prediction vs. actual scatter plots, and confidence bands on key outputs

**Working sketch — what needs to happen:**

```
Deliverable: A script or CLI command that runs the full pipeline on real data
and produces a structured report.

Where it lives: uq/pipeline.py (the run_full_pipeline function) + uq/cli.py
(the CLI entry point) + a report generation module or Marimo notebook.
```

```python
# uq/pipeline.py — the missing orchestrator

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from uq.inputs import InputParameterSpaceVecoli, load_dataset
from uq.aggregation import Aggregator, AggregationStrategy, compute_variance_decomposition
from uq.cell_cycle import calculate_cell_cycle
from uq.pce import prescreen_parameters, create_samples, process_samples, fit_pce_coefficients
from uq.sensitivity import SobolIndices
from uq.models import PCEParameterSelectionConfig


@dataclass
class PipelineResult:
    """Everything Activity 5 needs to produce a report."""
    # Step 3: Aggregation results
    agg_uniform: AggregatedOutput
    agg_by_generation: AggregatedOutput
    agg_by_seed: AggregatedOutput
    cell_cycle_result: CellCycleResult

    # Step 4: Variance decomposition
    variance_decomposition: dict[str, np.ndarray]

    # Step 5: Morris screening
    morris_indices: MorrisIndices
    selected_parameters: list[Parameter]

    # Step 6: PCE surrogate
    surrogate: PCESurrogate
    pce_fit: PCEFitResult

    # Step 7: Sobol indices
    sobol_indices: SobolIndices


def run_full_pipeline(
    experiment_id: str,
    outdir_root: str,
    f: Callable,                     # the simulation function or precomputed wrapper
    param_space: InputParameterSpaceVecoli | None = None,
    n_morris_trajectories: int = 20,
    n_top_params: int = 5,
    n_pce_samples: int = 100,
    polynomial_order: int = 2,
    cell_cycle_variable_type: str = "mass_based",
    generation_lower_bound: int | None = None,
) -> PipelineResult:
    """
    Execute the complete RFC006 7-step UQ workflow.

    This is Activity 5: apply the analysis to real data and return
    everything needed for the report.
    """
    # Step 1: Parameter space
    if param_space is None:
        param_space = InputParameterSpaceVecoli(
            include_vio=True, include_mecillinam=True
        )

    # Step 2: Load data
    df = load_dataset(experiment_id, Path(outdir_root))

    # Step 3: Aggregate (strategies 1-3)
    # ... (need DuckDB connection from real data path)
    # This is the part that needs real Aggregator wiring

    # Step 3d: Cell cycle
    cc_result = calculate_cell_cycle(
        experiment_id=experiment_id,
        outdir_root=outdir_root,
        variable_type=cell_cycle_variable_type,
    )

    # Step 4: Variance decomposition
    decomp = compute_variance_decomposition(agg_by_gen, agg_by_seed, agg_uniform)

    # Step 5: Morris screening
    selected = prescreen_parameters(
        full_space=param_space,
        f=f,
        config=PCEParameterSelectionConfig(
            n_trajectories=n_morris_trajectories,
            n_top=n_top_params,
        ),
    )

    # Step 6: PCE surrogate
    X = create_samples(N=n_pce_samples, selected=selected)
    Y = process_samples(X, f)
    bounds = np.array([p.bounds for p in selected])
    pce_fit = fit_pce_coefficients(X, Y, polynomial_order, bounds=bounds)
    surrogate = pce_fit.to_surrogate()

    # Step 7: Sobol indices from PCE coefficients
    sobol = surrogate.compute_sobol_indices(
        parameter_names=[p.name for p in selected]
    )

    return PipelineResult(
        agg_uniform=agg_uniform,
        agg_by_generation=agg_by_gen,
        agg_by_seed=agg_by_seed,
        cell_cycle_result=cc_result,
        variance_decomposition=decomp,
        morris_indices=morris,
        selected_parameters=selected,
        surrogate=surrogate,
        pce_fit=pce_fit,
        sobol_indices=sobol,
    )
```

**The concrete gaps that need filling:**

1. **`run_full_pipeline()` in `pipeline.py`**: The orchestrator function that wires Steps 1-7 together. Currently `pipeline.py` has a stub `pipeline()` that only calls `generate_surrogate()` — it's missing Steps 2-4, 7, and the aggregation-to-DuckDB connection.

2. **`PCESurrogate.compute_sobol_indices()`**: Step 7 says Sobol indices come "from PCE coefficients." The test at line 880 of `test_real_data.py` currently creates *mock* Sobol indices — meaning the actual analytical Sobol computation from PCE coefficients is not wired up. The math is: for each parameter `i`, the first-order Sobol index `S_i` is the sum of squared PCE coefficients for all multi-indices where *only* parameter `i` appears, divided by total output variance. This is a well-known closed-form formula for PCE-based Sobol and should be a method on `PCESurrogate`.

3. **Real `f` (simulation function)**: The test uses a `synthetic_model` function. For Activity 5, `f` needs to be a `SimulationWrapper` or `PrecomputedWrapper` pointed at actual vEcoli simulation data for each use case.

4. **Report generation**: A script, Marimo notebook, or function that takes `PipelineResult` and generates figures + summary. Could be as simple as a `PipelineResult.generate_report(output_dir)` method that writes matplotlib figures and a summary JSON/HTML.

5. **CLI integration**: `uq/cli.py` currently has an empty `uq()` command. Wire it to `run_full_pipeline()` so a user can run:
   ```bash
   uv run python -m uq.cli api_simulation_default /path/to/sims
   ```

---

### 2. Write the Cell Cycle Variable Consensus RFC

**The RFC requirement (§4, Activity 6):**
> *"Develop a cell cycle stratification strategy. Expected to involve the definition of a low-dimensional (possibly scalar) 'cell cycle variable'. Gather input and feedback from all subteams."*
>
> **Measure of completion:** *"A dedicated, separate RFC is written on the topic of defining the cell cycle variable"*

**What the RFC is really asking (big picture):**

RFC006 deliberately punts the *choice* of cell cycle variable to a future RFC. This is a scientific/consensus decision, not just an engineering one. The question is: what single scalar number best represents "where a cell is in its life cycle" such that binning simulation data by that number reveals meaningful phenotypic variation?

The `uq/` package already provides five candidate implementations (mass-based, DNA replication, cell angle, Koopman eigenfunction, GSA-informed composite). What's missing is not software — it's the document that evaluates these candidates against real data, proposes a recommended default, and gets sign-off from the biology, modeling, and data subteams.

**Why it matters for the overall UQ goal:**

Strategy (4) — cell cycle stratification — is the most scientifically novel part of the RFC. Strategies 1-3 are straightforward grouping operations. Strategy 4 is where the framework produces its most distinctive insight: **how do observables like gene expression, protein counts, and metabolic fluxes change *within a single cell's lifetime*?** Without a consensus variable, results from Strategy 4 are exploratory rather than authoritative.

Moreover, RFC006 §3 explicitly states: *"in order to also serve the purposes of CD2, a quantitative relationship should be assumed between the input variables to this 'cell cycle variable' and omics measurements from IV&V."* This means the cell cycle variable must be something that can be *measured experimentally* (or at least correlated with experimental measurements), not just computed from simulation internals.

**What the real user experience looks like:**

A researcher running cell cycle analysis should not have to choose between five variable types and wonder which is "right." They should have a well-motivated default:

> "I want to stratify my simulation by cell cycle stage."
> "Use `variable_type='koopman'` — it's the consensus recommendation because [reasons from the RFC]."

Currently a user sees:
```python
calculate_cell_cycle(variable_type="mass_based")   # ???
calculate_cell_cycle(variable_type="koopman")       # ???
calculate_cell_cycle(variable_type="cell_angle")    # ???
```
...and has no basis for choosing. The consensus RFC provides that basis.

**Working sketch — what needs to happen:**

This is a document, not code. But the document needs to be informed by computational results. Here's the workflow:

```
1. Run all five cell cycle variable implementations on the same
   representative simulation dataset (the one from Activity 5).

2. For each, compute:
   - Stage binning quality: How evenly distributed are data points across bins?
   - Phenotypic variation CV: How much does the observable vary across the cycle?
   - Correlation with known biology: Do the B/C/D period labels align with
     expected DNA replication timing?
   - Experimental measurability: Can this variable be estimated from bulk
     RNA-seq or proteomics data? (CD2 requirement)

3. Present comparison to subteams. The Koopman approach is currently recommended
   because it's data-driven and automatically identifies periodicity, but the
   consensus RFC should evaluate whether simpler approaches (mass-based) are
   sufficient or whether the GSA-informed variant adds meaningful signal.

4. Write the RFC with:
   - Comparison table of all candidates
   - Recommended default with justification
   - Definition of how the variable relates to IV&V measurements
   - Acceptance criteria for future alternative implementations
```

The software hook is already in place — `register_cell_cycle_variable()` in `uq/cell_cycle.py` allows new implementations to be plugged in once consensus is reached. What the RFC determines becomes the default `variable_type` in `calculate_cell_cycle()`.

**What the `uq/` package specifically needs (software support for this RFC):**

A comparison script or notebook that runs all five implementations on the same data and produces a side-by-side evaluation. Something like:

```python
# scripts/compare_cell_cycle_variables.py or a tutorial notebook

from uq.cell_cycle import (
    MassBasedCellCycleVariable,
    DNAReplicationCellCycleVariable,
    CellAngleCellCycleVariable,
    KoopmanCellCycleVariable,
    GSAInformedCellCycleVariable,
    calculate_cell_cycle,
)

variable_types = ["mass_based", "dna_replication", "cell_angle", "koopman"]
results = {}
for vtype in variable_types:
    results[vtype] = calculate_cell_cycle(
        experiment_id="api_simulation_default",
        outdir_root="/path/to/sims",
        variable_type=vtype,
        n_bins=10,
        output_column="listeners__mass__dry_mass",
    )

# Compare
for vtype, result in results.items():
    print(f"{vtype:20s}  CV={result.phenotypic_variation_cv:.4f}  "
          f"stages_with_data={result.n_stages_with_data}/10  "
          f"n_points={result.n_data_points}")
```

---

### 3. Implement the Consensus Cell Cycle Approach with GSA-Informed Sensitivity Analysis

**The RFC requirement (§4, Activity 7):**
> *"Implement a new analysis to capture this cell cycle variable and apply the global sensitivity analysis methods."*
>
> **Measure of completion:** *"Consensus cell cycle approach (from RFC above) is implemented, committed, and reviewed"*

**What the RFC is really asking (big picture):**

This is the culmination of both phases. Once Activity 6 determines *which* cell cycle variable to use, Activity 7 asks: run the full sensitivity analysis (Sobol via PCE) using Strategy 4 aggregation, and produce the same quality of results as Strategies 1-3.

But there's a deeper requirement embedded in RFC006 §3 that the current workflow diagram doesn't show. The RFC says:

> *"The choice of the 'cell cycle variable' will be informed by the sensitivity analyses (1-3)"*

This creates a **feedback loop** that's currently invisible in the Step 1-7 pipeline:

```
Steps 1-4 (Strategies 1-3)
        │
        ▼
Variance Decomposition → identifies observables with high residual variance
        │
        ▼
GSA-informed observable selection → feeds into Koopman DMD
        │
        ▼
Cell cycle variable θ(t) ∈ [0, 1]
        │
        ▼
Strategy 4 aggregation → bin by θ, compute per-stage statistics
        │
        ▼
Run PCE/Sobol on Strategy 4 outputs → "phenotypic sensitivity analysis"
```

**This is the flow that's missing from the main workflow diagram.** Steps 3d and 4 currently appear as independent boxes, but in the RFC's vision, Step 4's variance decomposition *feeds into* Step 3d's cell cycle variable definition, which then feeds into a *second round* of sensitivity analysis specific to Strategy 4.

**Why it matters for the overall UQ goal:**

Strategy 4 answers a question the other three cannot: **how does the cell's phenotype change as it progresses through its life cycle, and which input parameters control that progression?** Strategies 1-3 tell you about inter-cell and inter-generation variance. Strategy 4 tells you about *intra-cell temporal dynamics*. This is the "phenotypic sensitivity analysis across the physiological time dimension" that the RFC envisions.

For CD2 and Milestone 10, this is essential because population-level perturbation analysis needs to account for the fact that a population of cells is a *mixture* of cells at different cycle stages, not a homogeneous pool. The cell cycle stratification lets you predict: "if I perturb vio expression, how does that change the growth rate *during the C period* vs. *during the D period*?"

**What the real user experience looks like:**

A researcher should be able to run the complete pipeline, including the GSA-informed cell cycle analysis, in one call:

```python
result = run_full_pipeline(
    experiment_id="api_simulation_default",
    outdir_root="/path/to/sims",
    f=simulation_wrapper,
    cell_cycle_variable_type="gsa_informed",  # <-- the key addition
)

# Result now includes:
# - Sobol indices for Strategies 1-3 (parameter importance for bulk outputs)
# - Sobol indices for Strategy 4 (parameter importance for cell-cycle-stage outputs)
# - The cell cycle variable that was computed, and which observables were selected
# - Per-stage sensitivity: "vio_expression has S_T=0.8 during C period but S_T=0.2 during B period"

for stage_idx, stage_sobol in enumerate(result.per_stage_sobol_indices):
    print(f"Stage {stage_idx}: top param = {stage_sobol.get_most_influential(n=1)}")
```

**Working sketch — what needs to happen:**

The individual pieces exist (`GSAInformedCellCycleVariable`, `CellCycleAggregator`, `PCESurrogate`). What's missing is the *wiring* that connects them into the feedback loop and produces per-stage sensitivity results.

```python
# The missing orchestration in pipeline.py

def run_cell_cycle_sensitivity(
    agg_uniform: AggregatedOutput,
    agg_by_gen: AggregatedOutput,
    agg_by_seed: AggregatedOutput,
    observable_names: list[str],
    trajectory_data: pl.DataFrame,
    f: Callable,
    param_space: InputParameterSpace,
    expected_cycle_time: float = 3600.0,
    n_bins: int = 10,
) -> dict:
    """
    The RFC006 §3 feedback loop:
    GSA(strategies 1-3) → select observables → Koopman → θ(t) → bin → per-stage GSA
    """
    from uq.cell_cycle import GSAInformedCellCycleVariable, CellCycleAggregator
    from uq.sensitivity import identify_cell_cycle_relevant_observables

    # Phase 1 output: variance decomposition identifies CC-relevant observables
    relevance = identify_cell_cycle_relevant_observables(
        aggregated_uniform=agg_uniform,
        aggregated_by_gen=agg_by_gen,
        aggregated_by_seed=agg_by_seed,
        observable_names=observable_names,
    )

    # Feedback: GSA-selected observables → Koopman → cell cycle variable
    gsa_cc = GSAInformedCellCycleVariable(
        aggregated_uniform=agg_uniform,
        aggregated_by_gen=agg_by_gen,
        aggregated_by_seed=agg_by_seed,
        observable_names=observable_names,
        expected_cycle_time=expected_cycle_time,
    )
    cc_var = gsa_cc.compute(trajectory_data)

    # Bin data into cell cycle stages
    stage_bins = cc_var.to_stage_bins(n_bins=n_bins)

    # Per-stage sensitivity analysis:
    # For each stage, aggregate outputs within that stage,
    # then run PCE/Sobol on the stage-specific outputs
    per_stage_sobol = []
    for stage in range(n_bins):
        stage_mask = stage_bins == stage
        if stage_mask.sum() < 10:
            continue  # not enough data in this stage

        # Define a stage-specific objective function:
        # f_stage(x) = mean output when cell is in this stage
        def f_stage(x, _mask=stage_mask):
            y_full = f(x)  # full trajectory
            return np.mean(y_full[_mask])  # mean within this stage

        # Run PCE on this stage's objective
        selected = prescreen_parameters(param_space, f=f_stage)
        X = create_samples(N=50, selected=selected)
        Y = process_samples(X, f_stage)
        bounds = np.array([p.bounds for p in selected])
        pce_fit = fit_pce_coefficients(X, Y, polynomial_order=2, bounds=bounds)
        surrogate = pce_fit.to_surrogate()
        sobol = surrogate.compute_sobol_indices(
            parameter_names=[p.name for p in selected]
        )
        per_stage_sobol.append({"stage": stage, "sobol": sobol})

    return {
        "cell_cycle_variable": cc_var,
        "selected_observables": gsa_cc.selected_observables,
        "relevance_result": relevance,
        "per_stage_sobol": per_stage_sobol,
    }
```

**The concrete gaps that need filling:**

1. **`PCESurrogate.compute_sobol_indices()`** (same as Item 1 above): The analytical Sobol-from-PCE computation. This is the shared blocker across Items 1 and 3.

2. **Per-stage sensitivity loop**: The code above sketches it, but it needs careful design for how `f_stage` interacts with the wrapper. For precomputed data (the realistic case), the stage-specific objective is just a filtering + aggregation operation on already-available data, not a new simulation run.

3. **Integration into `run_full_pipeline()`**: The cell cycle sensitivity should be an optional but default-on step in the pipeline, producing `per_stage_sobol_indices` alongside the bulk Sobol indices.

4. **Depends on Item 2**: The consensus RFC determines which cell cycle variable to use. Until then, the code can run with any of the five implementations, but the "consensus" checkbox requires the RFC to exist.

---

### Summary: The Three Gaps at a Glance

| # | Requirement | Type | Blocker | What ships |
|---|-------------|------|---------|------------|
| 1 | Apply pipeline to real use cases, produce report | **Execution + Deliverable** | `run_full_pipeline()` orchestrator, `PCESurrogate.compute_sobol_indices()`, real `f` wiring, report generation | A `PipelineResult` object and a report (slides/HTML) showing Sobol indices, variance decomposition, and surrogate quality for each vEcoli use case |
| 2 | Write consensus RFC on cell cycle variable | **Document + Science** | Running all 5 CC variable implementations on real data, comparing results, getting subteam sign-off | An RFC document with comparison table, recommended default, and IV&V measurability analysis |
| 3 | Implement consensus CC approach with per-stage GSA | **Software + Execution** | Items 1 and 2, plus `run_cell_cycle_sensitivity()` wiring and per-stage PCE loop | Per-stage Sobol indices showing which parameters matter at each point in the cell cycle |

**The shared technical blocker across Items 1 and 3 is `PCESurrogate.compute_sobol_indices()`** — the analytical computation of Sobol indices from PCE coefficients. This is well-known math (partition coefficient variance by multi-index) and is the single highest-leverage piece of code to write next.
