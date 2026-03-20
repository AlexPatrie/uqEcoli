"""
Growth progress variable for mass-stratified sensitivity analysis.

Biological rationale
--------------------
E. coli cells grow approximately exponentially between birth and
division (Schaechter et al., J Gen Microbiol 1958; Taheri-Araghi et al.,
Current Biology 2015).  In the vEcoli whole-cell model (Macklin et al.,
Science 2020; Ahn-Horst et al., npj Syst Biol Appl 2022), dry mass is
a primary integrated output reflecting the net balance of all
biosynthetic processes — transcription, translation, and metabolism —
and is directly comparable to experimental measurements via optical
density or buoyant mass (Godin et al., Nat Methods 2010).

Normalizing log(dry_mass) to [0, 1] over a cell's observation window
yields a monotonic measure of growth progress:

    θ(t) = [log M(t) - log M_min] / [log M_max - log M_min]

This is the fraction of the observed mass range that has been traversed,
equivalent to fractional doublings completed for exponential growth.

Why dry mass?
  - It is the most direct integrator of all cellular processes:
    protein mass (~55% of dry mass) + RNA mass (~20%) + DNA + lipids
  - It is the variable that defines the growth rate itself:
    μ = d(ln M)/dt (Bremer & Dennis, 1996)
  - It monotonically increases within a cell's lifespan
  - It is experimentally measurable (OD, buoyant mass, Coulter counter)
  - The vEcoli model validates growth rate distributions and RNA:protein
    ratios against experimental data (Ahn-Horst et al., Fig. 2e)

This is NOT a cell cycle variable.  It does not identify B/C/D-period
phases or make any claim about replication timing.  It simply bins the
growth trajectory by mass accumulation so that sensitivity analysis can
reveal whether parameter importance changes as the cell gets larger.
"""

from __future__ import annotations

import numpy as np


def compute_growth_fraction(
    timeseries: np.ndarray,
    mass_col_index: int = 0,
) -> np.ndarray:
    """Compute growth progress from normalized log(dry_mass).

    Args:
        timeseries: Array of shape ``(n_timesteps, n_observables)``.
            Column *mass_col_index* must contain dry mass values (fg).
        mass_col_index: Index of the dry-mass column (default 0,
            matching the default observable order
            ``listeners__mass__dry_mass``).

    Returns:
        θ array of shape ``(n_timesteps,)`` in [0, 1], where 0 is
        the start of the observation window and 1 is the end.
    """
    mass = timeseries[:, mass_col_index]
    if mass.max() <= 0:
        return np.linspace(0.0, 1.0, len(mass))

    log_mass = np.log(np.maximum(mass, 1e-30))
    lo, hi = log_mass.min(), log_mass.max()
    span = hi - lo
    if span < 1e-15:
        return np.linspace(0.0, 1.0, len(mass))

    theta = (log_mass - lo) / span
    return np.clip(theta, 0.0, 1.0)


def bin_by_growth_stage(
    theta: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Assign each timepoint to a growth-progress bin.

    Uniformly divides [0, 1] into *n_bins* equal intervals.
    Early bins = small cells (recently divided); late bins = large cells
    (approaching division).

    Args:
        theta: Growth progress array in [0, 1].
        n_bins: Number of stage bins.

    Returns:
        Integer array with values in ``[0, n_bins - 1]``.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(theta, edges) - 1
    return np.clip(bins, 0, n_bins - 1)
