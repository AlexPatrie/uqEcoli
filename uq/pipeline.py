"""
Uncertainty Quantification framework execution pipeline (as proposed by RFC006)
"""

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


def pipeline(
    full_space: InputParameterSpace, f: Callable, sample_size: int, config: PCEParameterSelectionConfig | None = None
):
    pce = generate_surrogate(space=full_space, f=f, sample_size=sample_size, config=config)
