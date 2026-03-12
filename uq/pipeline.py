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

"""


class Pipeline:
    pce: PCESurrogate
