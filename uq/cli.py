"""
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

  The better mental model: Phase 1 gives you the population-level view (bulk). Phase 2 gives you the within-cell-lifecycle view
  (phenotypic). They decompose the same total variance into different components — like how you can decompose the total variance of
  human height into "between countries" vs "within countries." Those aren't static vs temporal versions of each other; they're
  orthogonal decompositions.
"""

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq import XSpaceVecoli
from uq.pipeline.workflow import execute_pipeline

app = typer.Typer()
console = Console()


def show(self):
    # Neon 90s header
    title = Text()
    title.append("⚡ ", style="bold yellow")
    title.append("INPUT PARAMETER SPACE", style="bold magenta")
    title.append("  //  ", style="dim cyan")
    title.append(self.experiment_id, style="bold cyan")

    # Parameter table
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        pad_edge=False,
    )
    table.add_column("PARAM", style="bold yellow", no_wrap=True)
    table.add_column("VALUE", style="bright_white")
    table.add_column("TYPE", style="dim cyan")

    for name, val in self.parameters.items():
        table.add_row(name, str(val), type(val).__name__)

    panel = Panel(
        table,
        title=title,
        subtitle=Text(f"n = {self.n_parameters} parameters", style="bold green"),
        border_style="magenta",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    )

    console.print(panel)


@app.command()
def uq(experiment_id: str, outdir_root: str) -> None:
    out_parent = Path(outdir_root)
    if not out_parent.exists():
        raise FileNotFoundError(f"The directory {outdir_root} does not exist")

    exp_dir = out_parent / experiment_id
    pq_root_dir = exp_dir / "history" / f"experiment_id={experiment_id}"

    # 1. Define input parameter space
    param_space = XSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.0, 5.0),
        vio_trl_eff_bounds=(0.0, 2.0),
        mecillinam_conc_bounds=(0.0, 10.0),
    )

    # 2. Run the full pipeline — data loading, aggregation, variance
    #    decomposition, PCE surrogate, and Sobol indices are all handled
    #    internally. Just point it at your simulation output directory.
    result = execute_pipeline(
        param_space=param_space,
        simulation_func=your_simulation_wrapper,  # callable with evaluate_batch(X) → Y
        experiment_id="mecillinam",
        sim_base_path="/path/to/vEcoli/api_integration/sims",
        output_types=["higher_order_properties", "exchange_fluxes"],
        generation_lower_bound=2,  # skip initial transient generations
        time_lower_bound=100.0,  # skip early transient timesteps
        polynomial_order=3,
        n_samples=200,
        export_path="./uq_results",
    )

    # 3. Inspect population-level results (Phase 1)
    sobol = result.population.sobol_indices[0]
    for name, value in sobol.select(n=5):
        print(f"{name}: {value:.4f}")

    # 4. Variance decomposition (Step 4)
    print(f"Generation: {result.variance_decomposition['generation_fraction']}")
    print(f"Seed:       {result.variance_decomposition['seed_fraction']}")

    # 5. Inspect per-cell-cycle-stage results (Phase 2)
    for i, stage_sobol in enumerate(result.cell_cycle.sobol_indices):
        print(f"Stage {i}: {stage_sobol.select(n=3)}")

    # 6. Reload results later
    from uq.pipeline.models import PipelineResult

    loaded = PipelineResult.from_export("./uq_results")


@app.command()
def readme(rfc_id: str = "RFC006") -> None:
    txt = (
        None
        if not rfc_id == "RFC006"
        else """
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
    )
    print(txt)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
