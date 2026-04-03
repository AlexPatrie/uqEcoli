"""
uq_simple — Scientifically transparent UQ for vEcoli.

Follows the `UQPC workflow <https://sandialabs.github.io/pytuq/apps/uqpc.html>`_
from PyTUQ (Sandia National Labs) with all four RFC006 aggregation strategies:

- **LHS sampling** (scipy) for space-filling parameter exploration
- **PCE surrogates** via ``pytuq.surrogates.pce.PCE`` (Legendre basis, lsq/bcs/anl)
- **Sobol indices** from PCRV coefficients (Sudret, 2008)
- **Four aggregation strategies** per RFC006:
  1. Uniform across all cells/times (baseline bulk)
  2. Stratified by generation (convergence control)
  3. Stratified by lineage seed (stochastic variance control)
  4. Growth-stratified (normalized log dry mass) — how parameter
     importance changes as cells grow, no spectral decomposition

Every computation has a direct biological or statistical interpretation
that a domain scientist can audit.
"""

from uq_simple.growth import compute_growth_fraction
from uq_simple.pipeline import run_by_generation, run_by_seed, run_pipeline

__all__ = [
    "compute_growth_fraction",
    "run_by_generation",
    "run_by_seed",
    "run_pipeline",
]
