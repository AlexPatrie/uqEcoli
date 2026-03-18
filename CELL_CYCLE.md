# E. coli Cell Cycle vs. θ Bins: What's Actually Going On

## The Real E. coli Cell Cycle (B, C, D)

You're right that E. coli has three canonical cell cycle periods:
- B period: Birth to initiation of DNA replication. The cell grows but hasn't started replicating its chromosome. Duration varies with growth rate — in fast growth (doubling time ~20 min), B period can be essentially zero.
- C period: Active DNA replication (~40 min, relatively constant regardless of growth rate). The chromosome is being copied from oriC to terC.
- D period: Completion of replication to cell division (~20 min). The cell finishes septation and divides.

These are well-defined biological phases. But the problem the RFC faces is: how do you assign a continuous coordinate to "where in the cell cycle is this cell?" in a way that's useful for variance decomposition.

### Why B/C/D Alone Isn't Enough

Three discrete bins (B, C, D) are too coarse for sensitivity analysis. Consider:

1. Unequal durations: In fast-growing E. coli, B can be ~0%, C ~67%, D ~33% of the cycle. Two of your three bins would cover C period. In slow-growing cells, B dominates. The bins aren't comparable across conditions.
2. Within-period dynamics: The cell at the start of C period (1 replication fork just initiated) is biologically very different from the cell at the end of C period (chromosome almost fully replicated, oriC may have
re-fired for overlapping rounds). A single "C period" bin collapses this.
3. PCE needs more resolution: To fit a polynomial surrogate within each stage and compute per-stage Sobol indices, you need enough bins to resolve how parameter sensitivity varies across the cycle. With 3 bins you'd get 3

_Sobol decompositions — not enough to see trends._

## θ: Cell cycle variable

### What θ Actually Is

The cell cycle variable θ is a continuous scalar in [0, 1] that maps every simulation timepoint to a "position" in the cell cycle. θ = 0 means "just born," θ = 1 means "about to divide." It's a dimensionless progress
coordinate.

The 10 bins (stages 0-9) in your output are uniform partitions of θ-space, not biological phase labels:

┌───────┬───────────┬────────────────────────────────────┐
│ Stage │  θ range  │  Rough biological correspondence   │
├───────┼───────────┼────────────────────────────────────┤
│ 0     │ 0.0 - 0.1 │ Early B (just born)                │
├───────┼───────────┼────────────────────────────────────┤
│ 1     │ 0.1 - 0.2 │ Late B / replication initiation    │
├───────┼───────────┼────────────────────────────────────┤
│ 2     │ 0.2 - 0.3 │ Early C (replication starting)     │
├───────┼───────────┼────────────────────────────────────┤
│ 3-6   │ 0.3 - 0.7 │ Mid-to-late C (active replication) │
├───────┼───────────┼────────────────────────────────────┤
│ 7-8   │ 0.7 - 0.9 │ D period (post-replication)        │
├───────┼───────────┼────────────────────────────────────┤
│ 9     │ 0.9 - 1.0 │ Late D (septation, pre-division)   │
└───────┴───────────┴────────────────────────────────────┘

The mapping isn't hardcoded — it's data-driven. The DNAReplicationCellCycleVariable does approximate B < 0.2, C = 0.2-0.8, D > 0.8, but the Koopman method doesn't assume any specific boundary.

### The Four Methods for Computing θ

`uq` has four implementations, from simplest to most sophisticated:

#### 1. Mass-based (`MassBasedCellCycleVariable`):

`θ = (log(M) - log(M_birth)) / (log(M_div) - log(M_birth))`

Assumes exponential growth. Simple, robust, but doesn't capture the biological cell cycle — just size progression. A cell that's 50% of the way to its division mass gets θ = 0.5, regardless of whether DNA replication has
started.

#### 2. DNA replication-based (`DNAReplicationCellCycleVariable`):
Normalizes DNA mass within each cell's lifetime. More biologically meaningful — DNA content jumps during C period. But requires DNA mass data.

#### 3. Cell angle (`CellAngleCellCycleVariable`):

Projects each timepoint into (log_mass, growth_rate) 2D space and computes the polar angle. The idea is that cells trace an approximately circular trajectory in this space: they grow in mass, growth rate rises, then as
division approaches growth rate dips. The angle in this loop approximates cycle position. This is an established technique from the literature.

#### 4. Koopman eigenfunction phase (`KoopmanCellCycleVariable`, **recommended**):

This is the most sophisticated. It runs Dynamic Mode Decomposition (DMD) on the multivariate timeseries, identifies the oscillatory mode whose frequency matches the expected cell cycle period (~1/3600 Hz), and extracts the
phase angle of the corresponding Koopman eigenfunction.
The math: for a Koopman mode with eigenvalue λ = |λ|e^(iω), the eigenfunction φ(x) maps each state to a complex number. The phase θ = arg(φ(x)) / 2π gives a [0,1]-valued coordinate that advances uniformly once per cycle.
This is data-driven — no assumptions about which observables track the cycle.
The GSA-Informed Selection (How Phase 1 Feeds Phase 2)
The GSAInformedCellCycleVariable is the full RFC006-compliant pipeline. The key insight is: not all observables carry cell cycle information. Some observables vary because of generation effects, some because of seed
effects, and the residual — what's left after removing those — is the cell-cycle-related variance.

### The flow:

#### 1. Phase 1 computes variance decomposition across strategies 1-3:
- generation_fraction: how much variance is explained by "which generation"
- seed_fraction: how much is explained by "which lineage seed"
- residual = 1 - generation_fraction - seed_fraction

#### 2. Step 5b (identify_cell_cycle_relevant_observables): ranks observables by residual variance fraction. Observables with high residual are the ones whose variance is NOT explained by generation or seed — so it's probably
cell-cycle-driven.

#### 3. Those selected observables are fed to the Koopman DMD as the input channels. The DMD finds periodic structure in exactly those observables that have unexplained-by-generation-or-seed variance.

#### 4. The extracted θ is used to bin all simulation timepoints into stages 0-9.

### How θ Is Used for Population-Level Analysis

This is the critical point you're asking about. θ doesn't replace the population-level analysis — it enables a second, orthogonal decomposition.
For each of the 10 θ-bins, Phase 2 asks: "Among all timepoints that fall in this bin (i.e., all cells at roughly the same cell cycle stage), which input parameters drive variance in the outputs?"

_Using your output as an example_:

- Stage 0 (θ = 0.0-0.1, just-born cells): mass_mean = 1.027, growth_mean = 0.01086
- Stage 5 (θ = 0.5-0.6, mid-cycle): mass_mean = 1.443, growth_mean = 0.00971
- Stage 9 (θ = 0.9-1.0, pre-division): mass_mean = 1.908, growth_mean = 0.00894

You can see mass doubles across the cycle (1.03 → 1.91) as expected. Growth rate peaks early (stage 1-2: 0.0114) then declines through D period (stage 8: 0.0084) before a slight recovery at stage 9 — this matches the known
E. coli behavior where growth slows during septation.

_The per-stage Sobol indices (not shown in the simplified JSON but present in the full PipelineResult) would tell you things like:_

- "During early growth (stage 0-1), mecillinam_concentration drives 80% of mass variance" (because mecillinam targets PBP2 which affects cell wall synthesis early)
- "During DNA replication (stages 3-6), vio_expression dominates" (because the violacein pathway competes for resources during replication)
- "Near division (stage 9), both parameters interact strongly" (total_order >> first_order)

Phase 1's bulk answer ("vio and mecillinam each explain ~50%") is the average picture. Phase 2's per-stage answers reveal when in the lifecycle each parameter matters. They're orthogonal decompositions of the same total
variance.

### Why 10 Bins and Not 3

The n_bins=10 default is a practical choice:

- 3 bins (B/C/D) would match biology but is too coarse for polynomial fitting — you can't fit a meaningful PCE with 3 data partitions
- 10 bins gives enough resolution to see trends (e.g., "parameter X's importance increases linearly across the cycle") while keeping each bin well-populated enough for stable statistics
- 100 bins would be too fine — not enough data per bin for reliable variance estimates
- The user can set n_bins to any value via --n-bins on the CLI

The bins are just uniform partitions of θ-space. They don't correspond 1:1 to B/C/D, but stages 0-1 approximately cover B, stages 2-7 approximately cover C, and stages 8-9 approximately cover D — weighted by the relative
durations of those periods under the specific growth conditions being simulated.
