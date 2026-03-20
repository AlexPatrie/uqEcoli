"""
uq_simple — Scientifically transparent UQ for vEcoli.

A focused subset of the full ``uq`` package that uses only
well-established, biologically grounded methods:

- **LHS sampling** (scipy) for space-filling parameter exploration
- **PCE surrogates** (PyTUQ / Legendre polynomials) for efficient sensitivity analysis
- **Sobol indices** (analytical from PCE coefficients) for variance attribution
- **ANOVA variance decomposition** for generation / seed / within-group effects
- **Growth-stratified sensitivity** (normalized log dry mass) to reveal
  how parameter importance changes as cells grow — no claim about
  cell cycle phases, no spectral decomposition, no assumed cycle time

Every computation has a direct biological or statistical interpretation
that a domain scientist can audit.
"""

from uq_simple.growth import compute_growth_fraction
from uq_simple.pipeline import run_pipeline

__all__ = [
    "compute_growth_fraction",
    "run_pipeline",
]
