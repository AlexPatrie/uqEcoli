"""
Uncertainty Quantification framework execution pipeline (as proposed by RFC006)

Workflow:
      Inputs: experiment_id: str, hpc_sim_base_path: Path, param_space: InputParameterSpaceVecoli, f: Callable[[np.ndarray], np.ndarray]

      1. Define Parameter Space — param_space = InputParameterSpaceVecoli(include_vio=True, include_mecillinam=True) → n parameters with
      bounds
      2. Load Simulation Data — df = load_dataset(experiment_id, hpc_sim_base_path) → Polars DataFrame from hive-partitioned Parquet
      3. Aggregation Strategies 1-3 (parallel) — aggregator = Aggregator(conn, history_sql, config_sql) → 3 × AggregatedOutput
        - 3a. Strategy 1: UNIFORM — mean, std across all cells/times
        - 3b. Strategy 2: BY_GENERATION — per-gen stats (convergence)
        - 3c. Strategy 3: BY_LINEAGE_SEED — per-seed stats (exogenous variance)
      4. Variance Decomposition — compute_variance_decomposition(agg_uniform, agg_by_gen, agg_by_seed) → per observable:
      generation_fraction, seed_fraction, residual_fraction (cell-cycle-related, key output)

      Branches into Phase 1 and Phase 2 (parallel):

      Phase 1 (Strategies 1-3 GSA):

      5a. Morris Prescreening — prescreen_parameters(param_space, f) → n params → K params (K << n) → selected: list[Parameter],
      MorrisIndices

      6a. PCE Surrogate (Strategies 1-3) — generate_surrogate(param_space, f, sample_size) — internally: X = create_samples(), Y =
      process_samples(X, f), fit_pce_coefficients(X, Y, order) → PCESurrogate, PCEFitResult

      7a. Sobol Indices (from PCE) — S_i, S_Ti from PCE coefficients — "Which parameters drive bulk output variance?" → SobolIndices

      Phase 2 (Strategy 4 — Cell Cycle):

      5b. GSA-Informed Observable Selection — identify_cell_cycle_relevant_observables(decomp) — filter by high residual_fraction →
      relevant_obs: list[str]

      6b. Koopman DMD — KoopmanCellCycleVariable(observable_columns=relevant_obs) — DMD/EDMD → find eigenvalue at ω_cc → extract φ_cc(x) →
      θ(x) = arg(φ)/2π → θ ∈ [0, 1]

      6c. Strategy 4 Aggregation — CellCycleAggregator — bin by θ → per-stage mean, std → stage_stats

      6d. Stage4 Wrapper (MISSING) — f_stage4(params): raw = f(params), θ = koopman(raw), bin by θ, return stage_means

      7b. PCE + Sobol on Strategy 4 (MISSING) — generate_surrogate(param_space, f_stage4, sample_size) — "Which parameters drive variation
      WITHIN each cell cycle stage?" → PCESurrogate, SobolIndices

      Phases converge:

      Pipeline Outputs:
      - From Phase 1: AggregatedOutput × 3, variance decomposition (fractions), MorrisIndices, PCESurrogate (bulk), SobolIndices (bulk)
      - From Phase 2: CellCycleResult (θ, stages), per-stage statistics, PCESurrogate (phenotypic) [MISSING], SobolIndices (phenotypic)
      [MISSING]
      - Feedback loop: Step 4 residual_fraction → Step 5b observable selection → Step 6b Koopman ("Strategies 1-3 tell you WHICH
      observables to give to Koopman")

      In other words, final user-facing outputs:
        1. Phase 1 Sobol: "vio_expression drives 60% of bulk mass variance, mecillinam_conc drives 25%, ..."
        2. Phase 2 Sobol: "During C-period (DNA replication), mecillinam_conc drives 80% of variance; during D-period, vio_expression
      dominates"

    1. dataset = db.get_dataset(config.dataset_id)
    2. x = dataset.load_simdata(); y = dataset.load_timeseries()
    3. agg = aggregate(y)
    4. decomp = variance_decomp(agg)
    5. start_stategies(decomp, x) --> concurrently runs phase1() -> phase1_outputs -> phase2()

┌─────────────────────────────────────────────────────────────────────────────┐
  │  PIPELINE OUTPUTS                                                           │
  │  ────────────────                                                           │
  │                                                                             │
  │  • PCESurrogate: Instant predictions for any parameter combination          │
  │  • SobolIndices: Which parameters matter most                               │
  │  • VarianceDecomposition: Sources of uncertainty                            │
  │  • CellCycleResult: Phenotypic variation across cell cycle                  │
  │  • MorrisIndices: Parameter screening results                               │
  └─────────────────────────────────────────────────────────────────────────────┘

Key points:

  - Steps 1-4 are sequential and shared by both phases
  - Phase 1 (left) and Phase 2 (right) run in parallel after Step 4 — they're independent analyses on the same data
  - The feedback loop is Step 4 → Step 5b: variance decomposition residuals select observables for Koopman
  - Steps 6d and 7b are the two missing pieces — the wrapper and the Strategy 4 PCE/Sobol
  - Phase 1 answers "which parameters drive bulk output variance?"
  - Phase 2 answers "which parameters drive variation within each cell-cycle stage?" — this is the "phenotypic sensitivity analysis"
  RFC006 §3 describes
Phase 1 asks: "Across all cells, all times, all generations — which parameters drive the most variance in bulk output?" This
collapses time. It's not a "static version" of Phase 2 — it's measuring different variance. Specifically, Phase 1's three strategies
decompose variance into generation effects (convergence), seed effects (exogenous stochasticity), and residual (everything else,
including cell cycle).

Phase 2 asks: "Within a single cell cycle stage — say, during DNA replication specifically — which parameters drive variance?" This
doesn't add temporal resolution to Phase 1's answer. It's asking about a different slice of the data that Phase 1 couldn't access at
all, because Phase 1 had no notion of "where in the cell cycle are we."

A concrete example of why they're not static-vs-temporal versions of each other:

  - Phase 1 might say: "vio_expression explains 40% of total mass variance"
  - Phase 2 might say: "vio_expression explains 5% of mass variance during B-period, but 85% during D-period"

  Phase 1's "40%" is not the time-average of Phase 2's stage-specific numbers. It's computed from a differently aggregated dataset (all
   cells pooled uniformly vs. binned by θ). The populations being analyzed are literally different subsets organized differently.

The mental model: Phase 1 gives you the population-level view (bulk). Phase 2 gives you the within-cell-lifecycle view
(phenotypic). They decompose the same total variance into different components — like how you can decompose the total variance of
human height into "between countries" vs "within countries." Those aren't static vs temporal versions of each other; they're
orthogonal decompositions.

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  PIPELINE OUTPUTS                                                                   │
  │                                                                                     │
  │  From Phase 1 (Strategies 1-3):              From Phase 2 (Strategy 4):             │
  │  ├─ AggregatedOutput × 3                     ├─ CellCycleResult (θ, stages)         │
  │  ├─ variance decomposition (fractions)       ├─ per-stage statistics                │
  │  ├─ MorrisIndices (parameter screening)      ├─ PCESurrogate (phenotypic) ◄ MISSING │
  │  ├─ PCESurrogate (bulk)                      └─ SobolIndices (phenotypic) ◄ MISSING │
  │  └─ SobolIndices (bulk)                                                             │
  │                                                                                     │
  │  ┌───────────────────────────────────────────────────────────────────────────┐       │
  │  │ FEEDBACK LOOP (variance decomp → Phase 2):                               │       │
  │  │                                                                           │       │
  │  │  Step 4 residual_fraction ──► Step 5b observable selection ──► Step 6b   │       │
  │  │                                                                           │       │
  │  │  "Strategies 1-3 tell you WHICH observables to give to Koopman"          │       │
  │  └───────────────────────────────────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────────────────────────────────────┘
"""

import abc
import os

# TODO: Implement the above!!! THEN update tutorials/docs!!!
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import polars
from ecoli.library.sim_data import LoadSimData
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from uq import (
    AggregatedOutput,
    PCESurrogate,
    SensitivityAnalyzer,
    SimulationWrapper,
    SobolIndices,
    WrapperConfig,
    XSpaceVecoli,
    cell_cycle,
    compute_variance_decomposition,
    inputs,
)
from uq import (
    calculate_cell_cycle as _cell_cycle,
)
from uq.inputs import XSpace
from uq.pce.models import PCEParameterSelectionConfig
from uq.pce.surrogate import generate_surrogate, prescreen_parameters
from uq.pipeline.models import Pipeline, PipelineConfig

# TODO: make process-bigraph steps out of this and use Nextflow py to run


# 1.)
def define_parameter_space(
    sim_data_path: Path | None = None,
    include_vio=True,
    include_mecillinam=True,
    vio_expression_bounds=(0.0, 5.0),
    vio_trl_eff_bounds=(0.0, 2.0),
    mecillinam_conc_bounds=(0.0, 10.0),
) -> XSpace | XSpaceVecoli:
    """
    Creates an instance of InputParameterSpace for pipeline from SimData.
    """
    # TODO: first load SimulationDataEcoli, then extract list[Parameter],
    #  where bounds are inferred from initial condition/parca?
    return XSpaceVecoli(
        include_vio,
        include_mecillinam,
        vio_expression_bounds,
        vio_expression_bounds,
        vio_trl_eff_bounds,
        mecillinam_conc_bounds,
    )


# 2.)
def load_timeseries(
    experiment_id: str, outdir_root: Path, observables: list[str] | None = None, include_metadata: bool = True
) -> polars.DataFrame:
    return inputs.load_dataset(experiment_id, outdir_root, observables, include_metadata)


@dataclass
class AggregationResult:
    uniform: AggregatedOutput
    generation: AggregatedOutput
    seed: AggregatedOutput


# 3.)
async def aggregate_timeseries(timeseries) -> AggregationResult:
    """
    This function should call Aggregator(conn, history_sql, config_sql)
    with queries tailored to each level of stratification in (cell, generation, seed).
    """
    # aggregator = Aggregator(conn, history_sql, config_sql)
    pass


VarianceDecomposition = dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]]


# 4.)
def get_variance_decomposition(agg: AggregationResult) -> VarianceDecomposition:
    """
    Returns Total variance from uniform aggregation
    - 'between_generation_variance': Variance between generations
    - 'between_seed_variance': Variance between lineage seeds
    - 'within_group_variance': Residual variance
    - 'generation_fraction': Fraction of variance from generation
    - 'seed_fraction': Fraction of variance from lineage seed
    """
    return compute_variance_decomposition(agg.generation, agg.seed, agg.uniform)


# 5. )
async def phase1(
    input_parameter_space: XSpaceVecoli,
    simulation_func: Callable,
    sample_size: int,
    export_surrogate: bool = True,
    prescreen_config: PCEParameterSelectionConfig | None = None,
    **kwargs,
):
    async def generate_surrogate():
        return create_surrogate(
            full_space=input_parameter_space,
            f=simulation_func,
            sample_size=sample_size,
            prescreen_config=prescreen_config,
            export=export_surrogate,
            **kwargs,
        )

    async def get_sobol_indices():
        wrapper = SimulationWrapper(config=WrapperConfig(**kwargs["simulation"]), parameter_space=input_parameter_space)
        analyzer = SensitivityAnalyzer(
            parameter_space=input_parameter_space,
            wrapper=wrapper,
            # TODO: pick samples?,
            # TODO: pick outputs?
        )
        pass


def create_surrogate(
    full_space: XSpace,
    f: Callable,
    sample_size: int,
    config: PCEParameterSelectionConfig | None = None,
    export: bool = True,
    **kwargs,
) -> PCESurrogate:
    surrogate = generate_surrogate(space=full_space, generator=f, sample_size=sample_size, config=config)
    if export:
        path = kwargs.get("path", Path(os.getcwd()).absolute())
        surrogate.export(path)
    return surrogate


def calculate_cell_cycle(
    experiment_id: str,
    outdir_root: str,
    variable_type: Literal["mass_based", "dna_replication", "cell_angle", "koopman"],
    n_bins: int,
    output_column: str,
    verbose: bool = True,
):
    result = _cell_cycle(
        experiment_id="api_simulation_default",
        outdir_root="/path/to/sims",
        variable_type="mass_based",  # or "dna_replication", "cell_angle", "koopman"
        n_bins=10,
        output_column="listeners__mass__dry_mass",
        verbose=True,
    )
    # Access results
    result.print_summary()
    print(f"Phenotypic CV: {result.phenotypic_variation_cv}")
    print(f"Cell cycle values: {result.cell_cycle_variable.values}")
    return result


def execute_pipeline(config: PipelineConfig) -> Pipeline:
    """
    The missing steps pick up after step 4 (variance decomposition) and run a second, parallel analysis:

      4. Variance decomposition (already computed)
             │
             ▼
      4a. GSA-informed observable selection (DONE — identify_cell_cycle_relevant_observables())
             │
             ▼
      4b. Koopman DMD on selected observables → θ(x) ∈ [0,1] (DONE — GSAInformedCellCycleVariable)
             │
             ▼
      4c. Bin by θ, compute per-stage statistics (DONE — CellCycleAggregator)
             │
             ▼
      4d. ← THIS IS MISSING: wrap "params → per-stage output" as a callable
             │
             ▼
      4e. ← THIS IS MISSING: feed that callable into generate_surrogate() (or manual PCE flow)
             │
             ▼
      4f. ← THIS IS MISSING: Sobol indices from Strategy 4 PCE → "phenotypic sensitivity"

      So the three missing things (4d, 4e, 4f) are exactly what I listed before. But they come after the existing Phase 1 pipeline, not
      before it. The prescreening code you highlighted would still run first — it's just that its results (the selected parameters) would
      also be used as inputs to the Strategy 4 PCE, not only the Strategy 1-3 PCE.

      In concrete terms, the missing wrapper (4d) would look roughly like:

      def strategy4_generator(params: np.ndarray) -> np.ndarray:
          # params → run sim → compute θ via Koopman → bin → return per-stage means
          raw_output = simulation_func(params)          # existing wrapper
          theta = koopman_cc.compute(raw_output).values  # existing: θ(x)
          bins = np.digitize(theta, stage_edges)          # existing: to_stage_bins()
          return np.array([raw_output[bins == s].mean() for s in range(n_stages)])

      Then you'd call generate_surrogate(space, generator=strategy4_generator, ...) — the exact same PCE machinery, just with a different
      f.
    """
    # Whats missing:
    #     1. A wrapper that takes a parameter vector, runs (or looks up) the simulation, computes the Koopman/GSA-informed cell cycle variable,
    #     bins by θ, and returns per-stage statistics as the output vector Y
    #     2. Feed that wrapper into generate_surrogate() or the manual create_samples → process_samples → fit_pce_coefficients flow
    #     3. Compute Sobol indices from the resulting PCE — this gives you "phenotypic sensitivity analysis" (which parameters drive variation
    #     within each cell-cycle stage)
    pass
