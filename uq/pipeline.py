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
from itertools import combinations_with_replacement
from typing import Any, Callable, Literal

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from scipy.linalg import lstsq
from scipy.stats import qmc

from uq import InputParameterSpaceVecoli, PCESurrogate, SensitivityAnalyzer, calculate_cell_cycle
from uq.inputs import InputParameterSpace
from uq.models import (
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


@dataclass
class PipelineConfig:
    name: str
    experiment_id: str


class UqPipeline:
    """
    Whats missing:

    1. A wrapper that takes a parameter vector, runs (or looks up) the simulation, computes the Koopman/GSA-informed cell cycle variable,
    bins by θ, and returns per-stage statistics as the output vector Y
    2. Feed that wrapper into generate_surrogate() or the manual create_samples → process_samples → fit_pce_coefficients flow
    3. Compute Sobol indices from the resulting PCE — this gives you "phenotypic sensitivity analysis" (which parameters drive variation
    within each cell-cycle stage)
    """

    pce: PCESurrogate
    config: PipelineConfig

    def __init__(self, config: PipelineConfig):
        self.config = config
