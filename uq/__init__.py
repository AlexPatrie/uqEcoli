"""
uq_simple — Scientifically transparent UQ for vEcoli.

Follows the `UQPC workflow <https://sandialabs.github.io/pytuq/apps/uqpc.html>`_
from PyTUQ (Sandia National Labs) with all four RFC006 aggregation strategies:

- **PCRV.sampleGerm()** (PyTUQ) for random sampling from the germ measure
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

from uq.growth import bin_by_growth_stage, compute_growth_fraction
from uq.workflow import (
    QuantifyResult,
    UQPCResult,
    quantify,
    run_strategy1_uniform,
    run_strategy2_by_generation,
    run_strategy3_by_seed,
    run_strategy4_growth_stratified,
    sample,
)

__all__ = [
    "bin_by_growth_stage",
    "compute_growth_fraction",
    "quantify",
    "QuantifyResult",
    "run_strategy1_uniform",
    "run_strategy2_by_generation",
    "run_strategy3_by_seed",
    "run_strategy4_growth_stratified",
    "sample",
    "UQPCResult",
]
