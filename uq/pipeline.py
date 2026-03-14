"""
Uncertainty Quantification framework execution pipeline (as proposed by RFC006)

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
"""

# TODO: Implement the above!!! THEN update tutorials/docs!!!
import math
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import polars
from ecoli.library.sim_data import LoadSimData
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from scipy.linalg import lstsq
from scipy.stats import qmc

from uq import (
    InputParameterSpaceVecoli,
    PCESurrogate,
    SensitivityAnalyzer,
    SobolIndices,
    calculate_cell_cycle,
    load_dataset,
)
from uq.inputs import InputParameterSpace
from uq.models import (
    BaseClass,
    Parameter,
    PCEConfig,
    PCEFitResult,
    PCEParameterSelectionConfig,
    PCEPreprocessingConfig,
    PCESolverConfig,
    PCESurrogateConfig,
)
from uq.pce import generate_surrogate


def cell_cycle():
    result = calculate_cell_cycle(
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


def create_surrogate(
    full_space: InputParameterSpace, f: Callable, sample_size: int, config: PCEParameterSelectionConfig | None = None
):
    surrogate = generate_surrogate(space=full_space, generator=f, sample_size=sample_size, config=config)


"""
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
  └──────┴────────────────────────────────────┴────────────────────────────┴──────────────────────────

  The missing piece is Phase 2 — Strategy 4. The full workflow in order is:

  What's done (Phase 1):

  1. Define parameter space
  2. Load data
  3. Aggregate via Strategies 1-3 → three AggregatedOutputs
  4. Variance decomposition → residual_fraction per observable
  5. Morris prescreening → select top-K parameters
  6. PCE surrogate on Strategy 1-3 outputs → PCESurrogate
  7. Sobol indices from PCE → "which parameters matter"

  All of this is implemented. The generate_surrogate() code you highlighted does steps 5-6 in one shot.

  What's missing (Phase 2 — the Strategy 4 loop):

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

⏺ ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │                          RFC006 FULL UQ WORKFLOW                                    │
  │                                                                                     │
  │  Inputs:  experiment_id: str                                                        │
  │           hpc_sim_base_path: Path                                                   │
  │           param_space: InputParameterSpaceVecoli                                    │
  │           f: Callable[[np.ndarray], np.ndarray]   (simulation or precomputed)       │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 1: Define Parameter Space                                                     │
  │                                                                                     │
  │  param_space = InputParameterSpaceVecoli(                                           │
  │      include_vio=True, include_mecillinam=True                                      │
  │  )                                                                                  │
  │  → n parameters with bounds                                                         │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 2: Load Simulation Data                                                       │
  │                                                                                     │
  │  df = load_dataset(experiment_id, hpc_sim_base_path)                                │
  │  → Polars DataFrame from hive-partitioned Parquet                                   │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 3: Aggregation Strategies 1-3  (PARALLEL)                                     │
  │                                                                                     │
  │  aggregator = Aggregator(conn, history_sql, config_sql)                             │
  │                                                                                     │
  │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐                │
  │  │ Strategy 1        │  │ Strategy 2        │  │ Strategy 3        │                │
  │  │ UNIFORM           │  │ BY_GENERATION     │  │ BY_LINEAGE_SEED   │                │
  │  │                   │  │                   │  │                   │                │
  │  │ mean, std across  │  │ per-gen stats     │  │ per-seed stats    │                │
  │  │ ALL cells/times   │  │ (convergence)     │  │ (exogenous var.)  │                │
  │  └────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘                │
  │           │                      │                      │                           │
  │           └──────────────────────┼──────────────────────┘                           │
  │                                  ▼                                                  │
  │                     3 × AggregatedOutput                                            │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  STEP 4: Variance Decomposition                                                     │
  │                                                                                     │
  │  decomp = compute_variance_decomposition(agg_uniform, agg_by_gen, agg_by_seed)      │
  │                                                                                     │
  │  → Per observable:                                                                  │
  │      generation_fraction   (convergence-related variance)                           │
  │      seed_fraction         (stochastic/exogenous variance)                          │
  │      residual_fraction     (cell-cycle-related variance)  ◄── KEY OUTPUT            │
  └─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
      ═════════════╪═════════════════════════════════╪═══════════════════════════════
      ║  PHASE 1   ▼  (Strategies 1-3 GSA)          ║  PHASE 2   ▼  (Strategy 4)  ║
      ║            │                                 ║            │                ║
      ║            │                                 ║            │                ║
      ║            ▼                                 ║            ▼                ║
      ║  ┌──────────────────────┐                    ║  ┌──────────────────────┐   ║
      ║  │ STEP 5a:             │                    ║  │ STEP 5b:             │   ║
      ║  │ Morris Prescreening  │                    ║  │ GSA-Informed         │   ║
      ║  │                      │                    ║  │ Observable Selection │   ║
      ║  │ prescreen_parameters(│                    ║  │                      │   ║
      ║  │   param_space, f)    │                    ║  │ identify_cell_cycle_ │   ║
      ║  │                      │                    ║  │ relevant_observables(│   ║
      ║  │ n params → K params  │                    ║  │   decomp)            │   ║
      ║  │ (K << n)             │                    ║  │                      │   ║
      ║  │                      │                    ║  │ Filter by high       │   ║
      ║  │ → selected: list[    │                    ║  │ residual_fraction    │   ║
      ║  │     Parameter]       │                    ║  │                      │   ║
      ║  │ → MorrisIndices      │                    ║  │ → relevant_obs:      │   ║
      ║  │                      │                    ║  │   list[str]          │   ║
      ║  └──────────┬───────────┘                    ║  └──────────┬───────────┘   ║
      ║             │                                ║             │               ║
      ║             ▼                                ║             ▼               ║
      ║  ┌──────────────────────┐                    ║  ┌──────────────────────┐   ║
      ║  │ STEP 6a:             │                    ║  │ STEP 6b:             │   ║
      ║  │ PCE Surrogate        │                    ║  │ Koopman DMD          │   ║
      ║  │ (Strategies 1-3)     │                    ║  │                      │   ║
      ║  │                      │                    ║  │ KoopmanCellCycle     │   ║
      ║  │ generate_surrogate(  │                    ║  │ Variable(            │   ║
      ║  │   param_space,       │                    ║  │   observable_columns │   ║
      ║  │   f,                 │                    ║  │   = relevant_obs     │   ║
      ║  │   sample_size)       │                    ║  │ )                    │   ║
      ║  │                      │                    ║  │                      │   ║
      ║  │ Internally:          │                    ║  │ DMD/EDMD → find      │   ║
      ║  │  X = create_samples()│                    ║  │ eigenvalue at ω_cc   │   ║
      ║  │  Y = process_samples │                    ║  │ → extract φ_cc(x)    │   ║
      ║  │       (X, f)         │                    ║  │ → θ(x) = arg(φ)/2π  │   ║
      ║  │  fit_pce_coefficients│                    ║  │                      │   ║
      ║  │       (X, Y, order)  │                    ║  │ → θ ∈ [0, 1]        │   ║
      ║  │                      │                    ║  └──────────┬───────────┘   ║
      ║  │ → PCESurrogate       │                    ║             │               ║
      ║  │ → PCEFitResult       │                    ║             ▼               ║
      ║  └──────────┬───────────┘                    ║  ┌──────────────────────┐   ║
      ║             │                                ║  │ STEP 6c:             │   ║
      ║             ▼                                ║  │ Strategy 4           │   ║
      ║  ┌──────────────────────┐                    ║  │ Aggregation          │   ║
      ║  │ STEP 7a:             │                    ║  │                      │   ║
      ║  │ Sobol Indices        │                    ║  │ CellCycleAggregator  │   ║
      ║  │ (from PCE)           │                    ║  │ Bin by θ → per-stage │   ║
      ║  │                      │                    ║  │ mean, std            │   ║
      ║  │ S_i, S_Ti from       │                    ║  │                      │   ║
      ║  │ PCE coefficients     │                    ║  │ → stage_stats        │   ║
      ║  │                      │                    ║  └──────────┬───────────┘   ║
      ║  │ "Which parameters    │                    ║             │               ║
      ║  │  drive bulk output   │                    ║             ▼               ║
      ║  │  variance?"          │                    ║  ┌──────────────────────┐   ║
      ║  │                      │                    ║  │ STEP 6d: ◄── MISSING │   ║
      ║  │ → SobolIndices       │                    ║  │ Stage4 Wrapper       │   ║
      ║  └──────────┬───────────┘                    ║  │                      │   ║
      ║             │                                ║  │ f_stage4(params):    │   ║
      ║             │                                ║  │   raw = f(params)    │   ║
      ║             │                                ║  │   θ = koopman(raw)   │   ║
      ║             │                                ║  │   bin by θ           │   ║
      ║             │                                ║  │   return stage_means │   ║
      ║             │                                ║  └──────────┬───────────┘   ║
      ║             │                                ║             │               ║
      ║             │                                ║             ▼               ║
      ║             │                                ║  ┌──────────────────────┐   ║
      ║             │                                ║  │ STEP 7b: ◄── MISSING │   ║
      ║             │                                ║  │ PCE + Sobol on       │   ║
      ║             │                                ║  │ Strategy 4 outputs   │   ║
      ║             │                                ║  │                      │   ║
      ║             │                                ║  │ generate_surrogate(  │   ║
      ║             │                                ║  │   param_space,       │   ║
      ║             │                                ║  │   f_stage4,          │   ║
      ║             │                                ║  │   sample_size)       │   ║
      ║             │                                ║  │                      │   ║
      ║             │                                ║  │ "Which parameters    │   ║
      ║             │                                ║  │  drive variation     │   ║
      ║             │                                ║  │  WITHIN each cell    │   ║
      ║             │                                ║  │  cycle stage?"       │   ║
      ║             │                                ║  │                      │   ║
      ║             │                                ║  │ → PCESurrogate       │   ║
      ║             │                                ║  │ → SobolIndices       │   ║
      ║             │                                ║  └──────────┬───────────┘   ║
      ║             │                                ║             │               ║
      ═════════════╪═════════════════════════════════╪═════════════╪═══════════════
                   │                                              │
                   └──────────────────┬───────────────────────────┘
                                      │
                                      ▼
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
"""


class NextflowProfile(StrEnum):
    STANDARD = "standard"
    AWS = "aws"
    CCAM = "ccam"


@dataclass
class OutputEmitterConfig(BaseClass):
    type: Literal["parquet", "timeseries", "xarray"]
    out_dir: str | None = None
    out_uri: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig(BaseClass):
    experiment_id: str
    sim_data_path: str
    emitter_arg: OutputEmitterConfig

    def model_dump(self) -> dict[str, Any]:
        outdir_key = "out_dir"
        path = self.emitter_arg.out_dir or self.emitter_arg.out_uri
        if path == self.emitter_arg.out_uri:
            outdir_key = "out_uri"
        return {"emitter": self.emitter_arg.type, "emitter_arg": {outdir_key: path}}


@dataclass
class Simulation(BaseClass):
    database_id: int
    config: SimulationConfig


@dataclass
class TimeseriesDataset(BaseClass):
    database_id: int  # TODO: used to perform loookup dataset from db/s3 (cross-reference with simulation attr)
    simulation: Simulation
    metadata_included: bool = True
    selections: list[str] | None = None

    @property
    def outdir_root(self):
        emitter = self.simulation.config.emitter_arg
        return emitter.out_dir or emitter.out_uri

    @property
    def x(self):
        return self.load_simdata()

    @property
    def y(self):
        return self.load_timeseries()

    def load_timeseries(self) -> polars.DataFrame:
        return load_dataset(
            self.simulation.config.experiment_id,
            Path(self.outdir_root),
            observables=self.selections,
            include_metadata=self.metadata_included,
        )

    def load_simdata(self) -> SimulationDataEcoli:
        # TODO: be able to turn into pl.DataFrame!
        return LoadSimData(sim_data_path=self.simulation.config.sim_data_path).sim_data


@dataclass
class PipelineConfig:
    name: str
    pce_config: PCESurrogateConfig


@dataclass
class PCESurrogateProfile:
    """
    The mental model: Phase 1 gives you the population-level view (bulk). Phase 2 gives you the within-cell-lifecycle view
    (phenotypic). They decompose the same total variance into different components — like how you can decompose the total variance of
    human height into "between countries" vs "within countries." Those aren't static vs temporal versions of each other; they're
    orthogonal decompositions.
    """

    population: PCESurrogate
    cell: PCESurrogate


@dataclass
class SobolIndexProfile(BaseClass):
    """
    each attribute of size 2 => one for fist order, one for total order
    """

    population: SobolIndices
    cell: list[SobolIndices]  # (of clen(n_cell_cycle_bins))


@dataclass
class UqProfile:
    lens: Literal["population", "cell"]
    sobol_indices: SobolIndices | list[SobolIndices]
    surrogate: PCESurrogate


@dataclass
class PipelineResult:
    population: UqProfile
    cell: UqProfile


@dataclass
class Pipeline:
    """
    Attributes:
        database_id: int
        config: UqPipelineConfig consisting of pipleine name, and pce config.
        dataset: Timeseries dataset containing the following attributes: timeseries data(y), parameter dataset (x), database_id, and simulation. Simulation itself
            has a database_id, and a config (vecoli config/api request config?)
        pce_surrogate: Object containing 2 pce surrogates generated from pipeline (bulk/phase1, phenotypic/phase2)
    """

    database_id: int
    config: PipelineConfig
    dataset_id: int  # used to look up TimeseriesDataset in DB for SMS API :)
    result: PipelineResult


def pipeline():
    """
        a. sobol_bulk:       (2, n_params)              — first/total order, population level
        b. sobol_stages:     (2, n_params, n_bins)       — first/total order, per cell cycle stage
        c. surrogates:       2 × PCESurrogate            — instant prediction without running WCM

      And (c) is what makes the whole thing practically useful — the Sobol indices tell you where to look, the surrogates let you explore
      cheaply.

    ❯ so again, then say for bulk, if wanting to know which params, they can be found by doing parameter_dataset[i] for i in sobol_bulk?

    ⏺ Right idea, but you'd sort/threshold first since the Sobol array has an entry for every parameter, not just the important ones.
      Something like:

      # Top 3 most influential parameters (by total-order)
      top_idx = np.argsort(sobol_bulk[1])[::-1][:3]  # [1] = total order
      for i in top_idx:
          print(f"{param_names[i]}: S_T = {sobol_bulk[1, i]:.1%}")

      Might give you:

      vio_expression: S_T = 55.2%
      mecillinam_concentration: S_T = 28.7%
      vio_trl_eff: S_T = 12.1%

      This is what SobolIndices.get_most_influential() already does in your code. The indices are the ranking; param_names[i] is the
      lookup.
    """


"""
Whats missing:

    1. A wrapper that takes a parameter vector, runs (or looks up) the simulation, computes the Koopman/GSA-informed cell cycle variable,
    bins by θ, and returns per-stage statistics as the output vector Y
    2. Feed that wrapper into generate_surrogate() or the manual create_samples → process_samples → fit_pce_coefficients flow
    3. Compute Sobol indices from the resulting PCE — this gives you "phenotypic sensitivity analysis" (which parameters drive variation
    within each cell-cycle stage)

    Music:
    wait so then is a better mental model maybe something like, phase 2 in the ./uq (rfc) (full workflow) metaphorically converts the
raw dataset/WCM function/system into a temporal signal, like audio?

⏺ Yes — and it's not even a metaphor. It's mathematically literal.

  Koopman/DMD spectral decomposition is the same operation as decomposing audio into frequencies. The math is the same class of
  operation as a Fourier transform:

  ┌────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │               Audio                │                        Phase 2 (Koopman/DMD)                         │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Raw waveform (pressure vs time)    │ Raw trajectory (observables vs time)                                 │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Fourier transform                  │ DMD eigendecomposition                                               │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Frequencies (Hz)                   │ Koopman eigenvalues (ω)                                              │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Amplitude per frequency            │ Mode amplitudes (bⱼ)                                                 │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Which instruments play each note   │ Koopman modes (φⱼ) — which observables participate in each frequency │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Fundamental pitch                  │ Cell cycle frequency (ω_cc)                                          │
  ├────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Harmonics (2×, 3×, 4× fundamental) │ Cell cycle harmonics (your code checks up to 4th harmonic)           │
  └────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘

  The cell cycle variable θ is literally the phase of the fundamental frequency — like asking "where are we in the current cycle of the
   dominant oscillation?" Phase 0 = start of cycle, phase 0.5 = halfway, phase 1.0 = back to start.

  Your CellCycleKoopmanAnalyzer.identify_cell_cycle_modes() is doing exactly what a pitch detector does: scanning the spectrum for the
  expected fundamental frequency and its harmonics:

  for harmonic in range(1, 5):  # up to 4th harmonic
      expected_freq = harmonic * self.expected_frequency
      if |mode.frequency - expected_freq| < tolerance * expected_freq:
          cell_cycle_modes.append(mode)

  And then θ = arg(φ_cc)/2π extracts the phase angle of that mode — exactly like reading the instantaneous phase of a sine wave.

  This is also why you have apollo/ — it's not a gimmick. The mapping from Koopman spectrum to musical score is bijective precisely
  because the mathematical objects are the same: frequencies, amplitudes, and phases. apollo/ is just choosing to render the spectrum
  as Western musical notation instead of as a power spectrum plot.

  So the mental model is: Phase 1 asks "which knobs matter?" (input sensitivity). Phase 2 asks "what song is the cell singing?"
  (temporal spectral structure). And θ is "where in the song are we right now?"


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


"""
