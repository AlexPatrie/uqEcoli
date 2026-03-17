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

from uq.pce.models import PCEParameterSelectionConfig
from uq.pipe import pipeline
from uq.pipeline import PipelineResult

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
def uq(
    experiment_ids: str,
    outdir_root: str,
    lb_generation: int | None = 2,
    lb_time: float | None = 100.0,
    n_bins: int = 10,
    pce_polynomial_order: int = 3,
    n_samples: int = 20,
    expected_cycle_time: float = 3600.0,
    pce_n_trajectories: int = 10,
    pce_n_selected_params: int = 5,
    export_path: str | None = None,
) -> None:
    param_prescreen_config = PCEParameterSelectionConfig(n_trajectories=pce_n_trajectories, n_top=pce_n_selected_params)
    result: PipelineResult = pipeline(
        experiment_ids=experiment_ids,
        sim_base_path=outdir_root,
        prescreen_config=param_prescreen_config,
        lb_time=lb_time,
        lb_generation=lb_generation,
        n_bins=n_bins,
        polynomial_order=pce_polynomial_order,
        n_samples=n_samples,
        expected_cycle_time=expected_cycle_time,
        export_path=Path(export_path) if export_path else None,
    )
    if export_path:
        console.print(f"[bold green]Pipeline complete.[/bold green] Results exported to {export_path}")
    else:
        console.print("[bold green]Pipeline complete.[/bold green]")


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
"""
    )
    print(txt)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
