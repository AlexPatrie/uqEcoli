"""Tutorial 07: Full RFC006 UQ Pipeline Workflow

This tutorial demonstrates the **complete UQ framework workflow** as specified by
RFC006, verifying each requirement against its implementation in the `uq` package.

RFC006 Activities:
  Phase 1 (MS-08.4.2):
    1. Identify input/output variables         → uq.inputs, uq.outputs
    2. Enable output via emitter               → ParquetEmitter + hive partitioning
    3. Implement wrapper functions              → uq.wrappers
    4. Implement PCE-based sensitivity (1-3)    → uq.sensitivity, uq.pce
    5. Apply to representative simulations      → this tutorial

  Phase 2 (CD2 / Milestone 10):
    6. Cell cycle stratification strategy       → uq.cell_cycle
    7. Cell cycle variable + per-stage GSA      → uq.pipeline.workflow

  RFC006 §4 parametrized steps:
    A. Selection/extraction of variables        → uq.outputs.OutputExtractor
    B. Temporal aggregation into Y              → uq.aggregation.Aggregator
    C. Sensitivity analysis method              → uq.sensitivity.SensitivityAnalyzer

Run with: uv run marimo run tutorials/07_full_workflow.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Full RFC006 UQ Pipeline

    This tutorial walks through the **complete 7-step workflow** specified by RFC006,
    verifying that each requirement is satisfied by the `uq` package.

    ```
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │  RFC006 Pipeline                                                                     │
    │                                                                                      │
    │  Steps 1-4 (shared):                                                                 │
    │  Parameter Space → Load Data → Aggregate (strategies 1-3) → Variance Decomposition   │
    │                                                                       │               │
    │                                                    ┌──────────────────┼──────────┐    │
    │                                                    │                  │          │    │
    │  Phase 1 (bulk):                                   │  Phase 2 (cell cycle):      │    │
    │  Morris → PCE → Sobol                              │  GSA obs → Koopman θ →      │    │
    │  "Which params drive bulk variance?"               │  Strategy4 → per-stage Sobol │    │
    │                                                    │  "Which params drive         │    │
    │                                                    │   within-stage variance?"    │    │
    │                                                    └─────────────────────────────┘    │
    │                                                                                      │
    │  Output: PipelineResult(population=UqProfile, cell_cycle=UqProfile)                   │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl

    return np, pl


# =============================================================================
# RFC006 ACTIVITY 1: Identify Input/Output Variables
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Activity 1: Identify Input/Output Variables

    **RFC006 requirement:** *"Identify the scientifically most relevant input and output
    variables. Expected: Inputs: vio pathway presence, mecillinam condition, gene knockouts.
    Outputs: transcriptome, proteome, metabolic fluxes, and higher order properties."*

    **Satisfied by:** `uq.inputs.XSpaceVecoli` (input parameter space) and
    `uq.outputs.OutputExtractor` (output variable extraction).
    """)
    return


@app.cell
def _():
    from uq import XSpaceVecoli, OutputType, get_output_variable_info
    from uq.pipeline.models import VioPathwayParams, MecillinamParams, GeneKnockoutParams

    # Define the input parameter space (Ξ)
    param_space = XSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.0, 5.0),
        vio_trl_eff_bounds=(0.0, 2.0),
        mecillinam_conc_bounds=(0.0, 10.0),
    )

    print("=== Activity 1: Input/Output Variables ===")
    print(f"Input parameters (Ξ): {param_space.parameter_names}")
    print(f"Parameter bounds: {param_space.parameter_bounds}")
    print(f"Dimension: {param_space.n_parameters}")
    print()

    # Output variable types per RFC006
    print("Output variable types:")
    for ot in OutputType:
        print(f"  {ot.name}: {ot.value}")

    print()
    print("Vio pathway params:", VioPathwayParams())
    print("Mecillinam params:", MecillinamParams())
    print("Gene knockout params:", GeneKnockoutParams())

    return (
        GeneKnockoutParams,
        MecillinamParams,
        OutputType,
        VioPathwayParams,
        get_output_variable_info,
        param_space,
    )


# =============================================================================
# RFC006 ACTIVITY 2-3: Emitter + Wrapper Functions
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Activities 2-3: Output Emitter + Wrapper Functions

    **Activity 2:** *"Enable output of relevant variables via emitter"*
    — Uses `ParquetEmitter` + Polars hive partitioning (deviation from `XarrayEmitter`).

    **Activity 3:** *"Implement input→output wrapper functions that can be called from
    numerical libraries"*
    — `uq.wrappers.SimulationWrapper` and `uq.wrappers.PrecomputedWrapper`.

    For this tutorial, we use **synthetic data** and a **synthetic wrapper**.
    """)
    return


@app.cell
def _(np, param_space):
    from uq.synthetic import generate_signal, generate_synthetic_simulation_data

    # Load synthetic simulation data (stands in for real ParquetEmitter output)
    sim_data = generate_synthetic_simulation_data()
    observable_columns = ["listeners__mass__dry_mass", "listeners__fba_results__growth"]

    print("=== Activities 2-3: Data Loading + Wrapper ===")
    print(f"Loaded {len(sim_data)} rows of synthetic simulation data")
    print(f"Columns: {sim_data.columns}")
    print(f"Experiments: {sim_data['experiment_id'].unique().to_list()}")
    print(f"Seeds: {sim_data['lineage_seed'].unique().to_list()}")
    print(f"Generations: {sorted(sim_data['generation'].unique().to_list())}")

    # Synthetic bulk wrapper: params → scalar output (stands in for SimulationWrapper)
    class SyntheticBulkWrapper:
        def __init__(self, param_names):
            self.param_names = param_names

        def __call__(self, x):
            signal = generate_signal(x, self.param_names, baseline_value=1.5, n_timesteps=400)
            return np.array([signal.mean()])

        def evaluate_batch(self, X):
            return np.vstack([self(x) for x in X])

    bulk_wrapper = SyntheticBulkWrapper(param_space.parameter_names)
    print(f"\nBulk wrapper test: f([2.5, 1.0, 5.0]) = {bulk_wrapper(np.array([2.5, 1.0, 5.0]))}")

    return (
        SyntheticBulkWrapper,
        bulk_wrapper,
        generate_signal,
        generate_synthetic_simulation_data,
        observable_columns,
        sim_data,
    )


# =============================================================================
# RFC006 §4 STEP A+B: Aggregation Strategies 1-3
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## RFC006 §4 Steps A+B: Aggregation Strategies 1-3

    **RFC006 §1:** *"Capture statistics across four types of aggregation:
    1) Uniformly across all simulated cells, 2) Stratified by generation,
    3) Stratified by lineage seed, 4) Stratified by cell cycle stage."*

    **RFC006 §4:** *"Parametrise: A) Selection/extraction, B) Temporal aggregation,
    C) Sensitivity analysis method."*

    **Satisfied by:** `uq.pipeline.workflow.aggregate_timeseries()` — runs strategies
    1-3 on a Polars DataFrame, returning an `AggregationResult`.
    """)
    return


@app.cell
def _(observable_columns, sim_data):
    from uq.pipeline.workflow import aggregate_timeseries

    # Step 3: Aggregate via strategies 1-3
    agg_result = aggregate_timeseries(sim_data, observable_columns)

    print("=== Step 3: Aggregation Strategies 1-3 ===")
    print(f"Strategy 1 (UNIFORM): mean={agg_result.uniform.mean}, n={agg_result.uniform.n_samples}")
    print(f"Strategy 2 (BY_GENERATION): {len(agg_result.generation.groups)} groups")
    print(f"Strategy 3 (BY_LINEAGE_SEED): {len(agg_result.seed.groups)} groups")

    return agg_result, aggregate_timeseries


# =============================================================================
# RFC006: VARIANCE DECOMPOSITION (Step 4)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Variance Decomposition

    **RFC006 §1:** *"Enable us to deconvolve different types of uncertainty."*

    Variance decomposition splits total variance into:
    - **Generation effects** — convergence toward steady-state growth
    - **Seed effects** — exogenous stochastic variance
    - **Residual** — cell-cycle-related dynamics (feeds into Phase 2)
    """)
    return


@app.cell
def _(agg_result, np):
    from uq.pipeline.workflow import get_variance_decomposition

    decomp = get_variance_decomposition(agg_result)

    gen_pct = np.mean(decomp["generation_fraction"]) * 100
    seed_pct = np.mean(decomp["seed_fraction"]) * 100
    residual_pct = max(0, 100 - gen_pct - seed_pct)

    print("=== Step 4: Variance Decomposition ===")
    print(f"Generation effects:    {gen_pct:.1f}% of variance")
    print(f"Stochastic seeding:    {seed_pct:.1f}% of variance")
    print(f"Residual (cell cycle): {residual_pct:.1f}% of variance")
    print()
    print("The residual fraction feeds into Phase 2 observable selection.")

    return decomp, gen_pct, get_variance_decomposition, residual_pct, seed_pct


# =============================================================================
# RFC006 ACTIVITY 4 / PHASE 1: PCE-Based Sensitivity (Steps 5a-7a)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Phase 1: Population-Level GSA (Activity 4)

    **RFC006 Activity 4:** *"Implement well established global sensitivity analysis
    methods based on the aggregation strategies (1-3). Expected to use PCE surrogate
    method for the stochastic function `(sim_data → SIM output)`."*

    **Satisfied by:** `uq.pipeline.workflow.run_phase1()` which calls
    `SensitivityAnalyzer.analyze_with_pce()` → PyTUQ `PCSobol` →
    Sobol indices from PCE coefficients.

    Steps:
    - **5a.** Morris prescreening → identify top-K parameters
    - **6a.** PCE surrogate construction (Legendre basis, LHS sampling)
    - **7a.** Sobol indices from PCE coefficients
    """)
    return


@app.cell
def _(bulk_wrapper, param_space):
    from uq.pipeline.workflow import run_phase1

    sobol_bulk, surrogate_bulk, _morris = run_phase1(
        param_space=param_space,
        simulation_func=bulk_wrapper,
        polynomial_order=2,
        n_samples=50,
    )

    print("=== Phase 1: Population-Level Sobol Indices ===")
    print(f"{'Parameter':<30s} {'S_i (first)':>12s} {'S_Ti (total)':>12s}")
    print("-" * 56)
    _fo = sobol_bulk.first_order.flatten()
    _to = sobol_bulk.total_order.flatten()
    for _i, _name in enumerate(sobol_bulk.parameter_names):
        _si = _fo[_i] if _i < len(_fo) else 0.0
        _sti = _to[_i] if _i < len(_to) else 0.0
        _bar = "█" * int(abs(_sti) * 40)
        print(f"{_name:<30s} {_si:>12.4f} {_sti:>12.4f}  {_bar}")

    print(f"\nPCE surrogate: R² = {surrogate_bulk.r_squared:.4f}")
    print(f"Dimensions: {surrogate_bulk.input_dim} → {surrogate_bulk.output_dim}")
    print("\nPhase 1 answers: 'Which parameters drive BULK output variance?'")

    return run_phase1, sobol_bulk, surrogate_bulk


# =============================================================================
# RFC006 ACTIVITY 5b-7b / PHASE 2: Cell Cycle Stratified GSA
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Phase 2: Cell-Cycle-Stratified GSA (Activities 6-7)

    **RFC006 §3:** *"A second type of analysis will need to be generated for the
    aggregation strategy (4), and will involve the definition of a low-dimensional
    'cell cycle variable' computed from omics variables. This variable will be used
    for deterministically binning simulation data into cell stages, in order to then
    perform a 'phenotypic' sensitivity analysis across the physiological time dimension.
    The choice of the 'cell cycle variable' will be informed by the sensitivity
    analyses (1-3)."*

    **Satisfied by:** `uq.pipeline.workflow.run_phase2()` which chains:
    - **5b.** `identify_cell_cycle_relevant_observables()` — GSA-informed obs selection
    - **6b.** `KoopmanCellCycleVariable` — DMD → θ(x) = arg(φ)/2π ∈ [0,1]
    - **6d.** `Strategy4Wrapper` — params → sim → θ-binning → per-stage means
    - **7b.** `compute_strategy4_sobol()` — per-stage PCE + Sobol indices
    """)
    return


@app.cell
def _(agg_result, np, observable_columns, param_space):
    from uq import identify_cell_cycle_relevant_observables
    from uq.pipeline.workflow import compute_strategy4_sobol
    from uq.synthetic import generate_signal as _gen_signal

    # Step 5b: GSA-informed observable selection
    relevance = identify_cell_cycle_relevant_observables(
        aggregated_uniform=agg_result.uniform,
        aggregated_by_gen=agg_result.generation,
        aggregated_by_seed=agg_result.seed,
        observable_names=observable_columns,
    )
    print("=== Step 5b: GSA-Informed Observable Selection ===")
    print(f"Relevant observables: {relevance.relevant_observables}")
    for _obs, _score in relevance.relevance_scores.items():
        print(f"  {_obs}: residual_fraction = {_score:.4f}")

    # Steps 6d + 7b: Strategy 4 wrapper → per-stage PCE + Sobol
    # Using synthetic wrapper (in production: uq.pipeline.workflow.Strategy4Wrapper)
    N_BINS = 10

    class _SyntheticStrategy4:
        def __init__(self, param_names, n_bins):
            self.param_names = param_names
            self.n_bins = n_bins

        def __call__(self, params):
            signal = _gen_signal(
                params,
                self.param_names,
                baseline_value=1.5,
                n_timesteps=self.n_bins * 40,
                random_seed=int(abs(params.sum() * 1000)) % (2**31),
            )
            _n_per_bin = len(signal) // self.n_bins
            return np.array([signal[s * _n_per_bin : (s + 1) * _n_per_bin].mean() for s in range(self.n_bins)])

        def evaluate_batch(self, X):
            return np.vstack([self(x) for x in X])

    _f_stage4 = _SyntheticStrategy4(param_space.parameter_names, N_BINS)

    print(f"\n=== Step 6d: Strategy 4 Wrapper (params → {N_BINS} stage means) ===")
    print(f"Test: f_stage4([2.5, 1.0, 5.0]) shape = {_f_stage4(np.array([2.5, 1.0, 5.0])).shape}")

    per_stage_sobol, surrogate_cc = compute_strategy4_sobol(
        param_space=param_space,
        f_stage4=_f_stage4,
        polynomial_order=2,
        n_samples=50,
    )

    print(f"\n=== Step 7b: Per-Stage Sobol Indices ({len(per_stage_sobol)} stages) ===")
    print(f"{'Stage':<8s} ", end="")
    for _name in param_space.parameter_names:
        print(f"{_name:>20s} ", end="")
    print()
    print("-" * (8 + 21 * len(param_space.parameter_names)))
    for _i, _s in enumerate(per_stage_sobol):
        _vals = _s.total_order.flatten()
        print(f"{'θ=' + f'{_i / len(per_stage_sobol):.1f}':<8s} ", end="")
        for _j in range(len(param_space.parameter_names)):
            _v = _vals[_j] if _j < len(_vals) else 0.0
            print(f"{_v:>20.4f} ", end="")
        print()

    print(f"\nPCE surrogate (phenotypic): R² = {surrogate_cc.r_squared:.4f}")
    print("\nPhase 2 answers: 'Which parameters drive variance WITHIN each cell cycle stage?'")

    return (
        N_BINS,
        compute_strategy4_sobol,
        identify_cell_cycle_relevant_observables,
        per_stage_sobol,
        relevance,
        surrogate_cc,
    )


# =============================================================================
# ASSEMBLE PipelineResult
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Pipeline Result Assembly

    Both phases produce a `UqProfile`, assembled into a `PipelineResult`.

    - **Population profile:** 1 × SobolIndices + 1 × PCESurrogate (bulk)
    - **Cell cycle profile:** n_bins × SobolIndices + 1 × PCESurrogate (phenotypic)

    `PipelineResult` supports serialization via `export()` / `from_export()`.
    """)
    return


@app.cell
def _(per_stage_sobol, sobol_bulk, surrogate_bulk, surrogate_cc):
    from uq.pipeline.models import PipelineResult, StratificationLens, UqProfile

    pipeline_result = PipelineResult(
        population=UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[sobol_bulk],
            surrogate=surrogate_bulk,
        ),
        cell_cycle=UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=per_stage_sobol,
            surrogate=surrogate_cc,
        ),
    )

    print("=== PipelineResult ===")
    print(f"Population: {pipeline_result.population.stratification}")
    print(f"  SobolIndices: {len(pipeline_result.population.sobol_indices)} set")
    print(f"  Surrogate R²: {surrogate_bulk.r_squared:.4f}")
    print(f"Cell Cycle: {pipeline_result.cell_cycle.stratification}")
    print(f"  SobolIndices: {len(pipeline_result.cell_cycle.sobol_indices)} sets (per θ-bin)")
    print(f"  Surrogate R²: {surrogate_cc.r_squared:.4f}")

    return PipelineResult, StratificationLens, UqProfile, pipeline_result


# =============================================================================
# SERIALIZATION
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Serialization: Export / Import

    `PipelineResult.export(path)` persists surrogates (via `PCESurrogate.export()`),
    Sobol indices (via `DataclassIO.save()`), and metadata to disk.

    `PipelineResult.from_export(path)` reconstructs the full result.
    """)
    return


@app.cell
def _(PipelineResult, per_stage_sobol, pipeline_result):
    import tempfile
    from pathlib import Path

    _export_dir = Path(tempfile.mkdtemp()) / "uq_pipeline_result"

    print("=== Serialization Round-Trip ===")
    pipeline_result.export(_export_dir)
    print(f"Exported to: {_export_dir}")
    for _p in sorted(_export_dir.rglob("*")):
        if _p.is_file():
            print(f"  {_p.relative_to(_export_dir)}")

    # Round-trip verification
    loaded = PipelineResult.from_export(_export_dir)
    assert len(loaded.population.sobol_indices) == 1
    assert len(loaded.cell_cycle.sobol_indices) == len(per_stage_sobol)
    print("\nRound-trip verification: PASSED")

    return Path, loaded, tempfile


# =============================================================================
# KOOPMAN SPECTRAL ANALYSIS (Bonus)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Bonus: Koopman Spectral Analysis

    The `uq.koopman` module provides DMD-based spectral decomposition for
    identifying cell cycle harmonics and dynamical modes.
    """)
    return


@app.cell
def _(observable_columns, pl, sim_data):
    from uq import CellCycleKoopmanAnalyzer, DynamicModeDecomposition

    _trajectory = (
        sim_data.filter((pl.col("experiment_id") == 0) & (pl.col("lineage_seed") == 0))
        .sort("time")
        .select(observable_columns)
        .to_numpy()
    )

    _dmd = DynamicModeDecomposition(rank=5)
    _dmd.fit(_trajectory)
    _spectrum = _dmd.get_spectrum(observable_names=["mass", "growth_rate"])

    print("=== Koopman Spectral Analysis ===")
    print(f"Trajectory shape: {_trajectory.shape}")
    print(f"Extracted {len(_spectrum.modes)} modes:")
    for _i, _mode in enumerate(_spectrum.get_dominant_modes(3)):
        print(
            f"  Mode {_i + 1}: freq={_mode.frequency:.6f} Hz, "
            f"|amp|={abs(_mode.amplitude):.4f}, "
            f"{'oscillatory' if _mode.is_oscillatory else 'non-oscillatory'}"
        )

    return CellCycleKoopmanAnalyzer, DynamicModeDecomposition


# =============================================================================
# REACTIVE PARAMETER EXPLORATION
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Reactive Parameter Exploration

    Once we have a PCE surrogate, we can explore the parameter space interactively.
    The surrogate evaluates in microseconds — fast enough for reactive UI.

    **Drag the sliders** to see how each parameter affects the bulk output.
    """)
    return


@app.cell
def _(mo, param_space, surrogate_bulk):
    _slider_list = []
    for _i, _name in enumerate(param_space.parameter_names):
        _lb, _ub = param_space.parameter_bounds[_i]
        _mid = (_lb + _ub) / 2
        _slider = mo.ui.slider(
            start=_lb,
            stop=_ub,
            step=(_ub - _lb) / 100,
            value=_mid,
            label=_name,
            show_value=True,
        )
        _slider_list.append(_slider)

    reactive_sliders = mo.ui.array(_slider_list)

    return (reactive_sliders,)


@app.cell
def _(mo, np, param_space, reactive_sliders, surrogate_bulk):
    _params = np.array(reactive_sliders.value)
    _prediction = surrogate_bulk.predict(_params.reshape(1, -1))
    _baseline = surrogate_bulk.predict(
        np.array([(lb + ub) / 2 for lb, ub in param_space.parameter_bounds]).reshape(1, -1)
    )

    _info = mo.vstack([
        mo.md("### Parameter Controls"),
        reactive_sliders,
        mo.md("---"),
        mo.md(f"""
**Surrogate prediction:** `{_prediction.flatten()[0]:.4f}`

**Baseline (midpoint):** `{_baseline.flatten()[0]:.4f}`

**Change:** `{(_prediction - _baseline).flatten()[0]:+.4f}`

*Instant evaluation via PCE surrogate (R² = {surrogate_bulk.r_squared:.4f})*
        """),
    ])

    mo.hstack([_info], widths=[1])
    return


# =============================================================================
# RFC006 VERIFICATION SUMMARY
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## RFC006 Verification Summary

    | # | Activity | RFC006 Ref | Status | Implementation |
    |---|----------|-----------|--------|----------------|
    | 1 | Identify input/output variables | §4 Act. 1 | COMPLETE | `XSpaceVecoli`, `OutputType` |
    | 2 | Enable output via emitter | §4 Act. 2 | COMPLETE | `ParquetEmitter` + hive partitioning |
    | 3 | Implement wrapper functions | §4 Act. 3 | COMPLETE | `SimulationWrapper`, `PrecomputedWrapper` |
    | 4 | PCE-based sensitivity (strategies 1-3) | §4 Act. 4 | COMPLETE | `SensitivityAnalyzer.analyze_with_pce()` |
    | 5 | Apply to representative simulations | §4 Act. 5 | DEMONSTRATED | This tutorial + `examples/uq_pipeline.py` |
    | 6 | Cell cycle stratification | §4 Act. 6 | COMPLETE | `KoopmanCellCycleVariable`, `GSAInformedCellCycleVariable` |
    | 7 | Cell cycle variable + per-stage GSA | §4 Act. 7 | COMPLETE | `Strategy4Wrapper`, `compute_strategy4_sobol()` |

    **RFC006 §4 parametrized steps:**

    | Step | Description | Implementation |
    |------|-------------|----------------|
    | A | Selection/extraction | `uq.outputs.OutputExtractor` |
    | B | Temporal aggregation | `uq.pipeline.workflow.aggregate_timeseries()` |
    | C | Sensitivity analysis | `uq.sensitivity.SensitivityAnalyzer` |

    **Pipeline module (`uq.pipeline`):**

    | Function | Description |
    |----------|-------------|
    | `execute_pipeline()` | Full orchestrator: steps 3-7 → `PipelineResult` |
    | `run_phase1()` | Phase 1: bulk PCE + Sobol |
    | `run_phase2()` | Phase 2: GSA obs → Koopman → Strategy4 → per-stage Sobol |
    | `Strategy4Wrapper` | Step 6d: params → Koopman θ → per-stage means |
    | `compute_strategy4_sobol()` | Step 7b: per-stage PCE + Sobol |
    | `aggregate_timeseries()` | Step 3: strategies 1-3 on Polars DataFrame |
    | `PipelineResult.export()` | Serialize surrogates + Sobol + metadata |
    | `PipelineResult.from_export()` | Deserialize from disk |
    """)
    return


if __name__ == "__main__":
    app.run()
