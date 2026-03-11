import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable, Optional
from itertools import combinations_with_replacement

"""
=========================================================================
PARAMETER CONFIGURATION - CUSTOMIZE THIS FOR YOUR USE CASE
=========================================================================

Define your parameters as a list of dictionaries with:
    - name: str           - Parameter name (used in labels)
    - bounds: [min, max]  - Parameter range
    - default: float      - Default/baseline value (optional, defaults to midpoint)
    - step: float         - Slider step size (optional, auto-calculated if omitted)
    - description: str    - What this parameter does (optional, for documentation)

Example configurations are provided below. Uncomment/modify as needed.
"""


@dataclass
class ParameterConfig:
    name: str
    bounds: tuple[float, float]
    default: float | int | complex
    step: float
    description: str

    def model_dump(self):
        d = asdict(self)
        bounds = tuple(self.bounds)
        d['bounds'] = bounds
        return d

# from uq import SensitivityAnalyzer, InputParameterSpace
# # 1. Full parameter space (many params)
# full_space = InputParameterSpace(include_vio=True, include_mecillinam=True, ...)
# # 2. Morris screening (cheap)
# analyzer = SensitivityAnalyzer(full_space, wrapper)
# morris = analyzer.analyze_with_morris(n_trajectories=20)
# print(morris.summary())
# # 3. Get PARAMETER_CONFIG for reactive tutorial
# PARAMETER_CONFIG = morris.to_parameter_config(
#     parameter_bounds=full_space.parameter_bounds,
#     top_n=5,
# )

# -------------------------------------------------------------------------
# EXAMPLE 1: Default vEcoli-like parameters (3 params)
# -------------------------------------------------------------------------
PARAMETER_CONFIG = [
    {
        "name": "expression_factor",
        "bounds": [0.5, 5.0],
        "default": 2.75,
        "step": 0.1,
        "description": "Gene expression multiplier (1.0 = baseline)",
    },
    {
        "name": "translation_efficiency",
        "bounds": [0.5, 2.0],
        "default": 1.25,
        "step": 0.05,
        "description": "Translation efficiency factor",
    },
    {
        "name": "inhibitor_conc",
        "bounds": [0.0, 10.0],
        "default": 5.0,
        "step": 0.5,
        "description": "Inhibitor concentration (reduces output)",
    },
]
# -------------------------------------------------------------------------
# PCE Configuration
# -------------------------------------------------------------------------
PCE_ORDER = 2  # Polynomial order for the surrogate (1, 2, or 3 recommended)

# -------------------------------------------------------------------------
# Timeseries Configuration
# -------------------------------------------------------------------------
TIMESERIES_LENGTH = 1000  # Number of timesteps
RANDOM_SEED = 42  # For reproducible noise