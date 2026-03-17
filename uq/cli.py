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

import tempfile
from pathlib import Path
from pprint import pp

import numpy as np
import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq import handlers
from uq.handlers import generate_samples
from uq.pce.models import PCEParameterSelectionConfig
from uq.pipe import Pipeline, execute_pipeline
from uq.pipeline import PipelineResult

app = typer.Typer()
console = Console()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _sobol_table(
    title: str,
    sobol,
    *,
    border_style: str = "cyan",
    n_top: int = 10,
) -> Table:
    """Build a Rich table for a single SobolIndices object."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style=border_style,
        title=title,
        title_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    table.add_column("S_i (first)", style="bright_white", justify="right")
    table.add_column("S_Ti (total)", style="bright_green", justify="right")

    first = sobol.first_order
    total = sobol.total_order
    if first.ndim > 1:
        first = np.mean(first, axis=0)
    if total.ndim > 1:
        total = np.mean(total, axis=0)

    ranked = np.argsort(total)[::-1][:n_top]
    for i in ranked:
        name = sobol.parameter_names[i] if i < len(sobol.parameter_names) else f"x{i}"
        table.add_row(name, _pct(first[i]), _pct(total[i]))
    return table


def print_report(result: PipelineResult, export_path: str | None = None) -> None:
    """Render PipelineResult as a rich terminal report."""
    console.print()

    # -- Header --
    header = Text()
    header.append("  UQ REPORT  ", style="bold white on magenta")
    header.append("  RFC006 Pipeline Results", style="bold cyan")
    console.print(Panel(header, box=box.DOUBLE_EDGE, border_style="magenta", padding=(0, 1)))

    # -- Variance Decomposition --
    decomp = result.variance_decomposition
    if decomp:
        gen_frac = decomp.get("generation_fraction")
        seed_frac = decomp.get("seed_fraction")
        residual_frac = decomp.get("residual_fraction")

        vd_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="yellow",
            title="VARIANCE DECOMPOSITION",
            title_style="bold yellow",
            pad_edge=False,
        )
        vd_table.add_column("SOURCE", style="bold yellow", no_wrap=True)
        vd_table.add_column("MEAN FRACTION", style="bright_white", justify="right")
        vd_table.add_column("RANGE", style="dim cyan", justify="right")

        for label, arr in [
            ("Generation (convergence)", gen_frac),
            ("Seed (exogenous)", seed_frac),
            ("Residual (cell cycle)", residual_frac),
        ]:
            if arr is not None:
                arr = np.asarray(arr)
                vd_table.add_row(label, _pct(arr.mean()), f"[{_pct(arr.min())} - {_pct(arr.max())}]")

        console.print(Panel(vd_table, border_style="yellow", box=box.ROUNDED, padding=(0, 1)))

    # -- Morris Screening --
    if result.morris_indices is not None:
        morris = result.morris_indices
        m_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="red",
            title="MORRIS PRESCREENING",
            title_style="bold red",
            pad_edge=False,
        )
        m_table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
        m_table.add_column("mu*", style="bright_white", justify="right")
        m_table.add_column("sigma", style="dim cyan", justify="right")
        m_table.add_column("CLASS", style="bright_green")

        mu_star = np.asarray(morris.mu_star)
        sigma = np.asarray(morris.sigma)
        ranked = np.argsort(mu_star)[::-1]
        for i in ranked:
            name = morris.parameter_names[i] if i < len(morris.parameter_names) else f"x{i}"
            ms = mu_star[i]
            sg = sigma[i]
            cls = "linear" if sg < 0.5 * ms else "nonlinear/interactive"
            m_table.add_row(name, f"{ms:.4f}", f"{sg:.4f}", cls)

        console.print(Panel(m_table, border_style="red", box=box.ROUNDED, padding=(0, 1)))

    # -- Phase 1: Population Sobol --
    pop_sobol = result.population.sobol_indices[0]
    pop_table = _sobol_table("GSA PHASE 1 // POPULATION (bulk)", pop_sobol, border_style="cyan")
    console.print(Panel(pop_table, border_style="cyan", box=box.ROUNDED, padding=(0, 1)))

    # -- Phase 2: Cell Cycle Sobol (per stage) --
    cc_sobols = result.cell_cycle.sobol_indices
    if cc_sobols:
        n_stages = len(cc_sobols)
        stage_tables = []
        for stage_idx, sobol in enumerate(cc_sobols):
            theta_lo = stage_idx / n_stages
            theta_hi = (stage_idx + 1) / n_stages
            stage_tables.append(
                _sobol_table(
                    f"Stage {stage_idx} (theta {theta_lo:.2f}-{theta_hi:.2f})",
                    sobol,
                    border_style="green",
                    n_top=5,
                )
            )

        cc_panel_title = Text()
        cc_panel_title.append("GSA PHASE 2 // CELL CYCLE ", style="bold green")
        cc_panel_title.append(f"({n_stages} stages)", style="dim green")
        console.print(
            Panel(
                Columns(stage_tables, equal=True, expand=True),
                title=cc_panel_title,
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # -- Cell Cycle Relevance --
    if result.cell_cycle_relevance is not None:
        ccr = result.cell_cycle_relevance
        rel_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title="CELL CYCLE RELEVANT OBSERVABLES",
            title_style="bold blue",
            pad_edge=False,
        )
        rel_table.add_column("OBSERVABLE", style="bold yellow", no_wrap=True)
        rel_table.add_column("RELEVANCE", style="bright_green", justify="right")

        for obs_name in ccr.relevant_observables[:10]:
            score = ccr.relevance_scores.get(obs_name, 0.0)
            rel_table.add_row(obs_name, f"{score:.4f}")

        console.print(Panel(rel_table, border_style="blue", box=box.ROUNDED, padding=(0, 1)))

    # -- Footer --
    if export_path:
        console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")
    console.print()


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
def quantify(
    experiment_ids: list[str],
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
    precomputed_path: str | None = None,
) -> None:
    """Run the full RFC006 UQ pipeline."""
    pipeline: Pipeline = handlers.pipeline(
        experiment_ids=experiment_ids,
        sim_base_path=outdir_root,
        lb_generation=lb_generation,
        lb_time=lb_time,
        n_bins=n_bins,
        polynomial_order=pce_polynomial_order,
        n_samples=n_samples,
        expected_cycle_time=expected_cycle_time,
        prescreen_config=PCEParameterSelectionConfig(n_trajectories=pce_n_trajectories, n_top=pce_n_selected_params),
        export_path=export_path,
        precomputed_path=precomputed_path,
        execute=True,
    )
    print_report(pipeline.result, export_path=export_path)


# @app.command()
# def demo(
#     export_path: str | None = None,
#     precomputed_path: str | None = None,
# ) -> None:
#     """Run the pipeline on the 3 default experiments with real vEcoli data."""
#     experiment_ids = [
#         "api_simulation_default",
#         "mecillinam",
#         "test_violacein_with_metabolism",
#     ]
#     base_path = Path("/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims")
#     observable_columns = [
#         "listeners__mass__dry_mass",
#         "listeners__mass__cell_mass",
#         "listeners__mass__volume",
#         "listeners__mass__growth",
#     ]
#     param_prescreen_config = PCEParameterSelectionConfig(n_trajectories=10, n_top=5)
#     result: PipelineResult = execute_pipeline(
#         experiment_ids=experiment_ids,
#         sim_base_path=str(base_path),
#         observable_columns=observable_columns,
#         prescreen_config=param_prescreen_config,
#         n_bins=5,
#         n_samples=20,
#         polynomial_order=2,
#         export_path=Path(export_path) if export_path else None,
#         precomputed_path=Path(precomputed_path) if precomputed_path else None,
#     )
#     if export_path:
#         console.print(f"[bold green]Pipeline complete.[/bold green] Results exported to {export_path}")
#     else:
#         console.print("[bold green]Pipeline complete.[/bold green]")


@app.command()
def demo(
    demo_type: str = "sampling",
    export_path: str | None = None,
    precomputed_path: str | None = None,
) -> None:
    experiment_ids = [
        "api_simulation_default",
        "mecillinam",
        "test_violacein_with_metabolism",
    ]
    base_path = Path("/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims")
    observable_columns = [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__growth",
    ]
    cache_dir = tempfile.TemporaryDirectory()

    def demo_full():
        """Run the pipeline on the 3 default experiments with real vEcoli data."""
        param_prescreen_config = PCEParameterSelectionConfig(n_trajectories=10, n_top=5)
        result: PipelineResult = execute_pipeline(
            experiment_ids=experiment_ids,
            sim_base_path=str(base_path),
            observable_columns=observable_columns,
            prescreen_config=param_prescreen_config,
            n_bins=5,
            n_samples=20,
            polynomial_order=2,
            export_path=Path(export_path) if export_path else None,
            precomputed_path=Path(precomputed_path) if precomputed_path else None,
        )
        if export_path:
            console.print(f"[bold green]Pipeline complete.[/bold green] Results exported to {export_path}")
        else:
            console.print("[bold green]Pipeline complete.[/bold green]")

    def demo_sample_generation():
        samples = handlers.generate_samples(
            experiment_ids=experiment_ids,
            sim_base_path=base_path,
            cache_dir=cache_dir.name,
            n_samples=10,
            seed=1111,
            observable_columns=observable_columns,
        )
        print("Created samples:")
        pp(samples)

    demo_sample_generation() if demo_type == "sampling" else demo_full()
    cache_dir.cleanup()


@app.command(name="generate-samples")
def create_samples(
    experiment_ids: list[str],
    sim_base_path: str,
    cache_dir: str,
    n_samples: int = 200,
    seed: int = 42,
    observable_columns: list[str] | None = None,
) -> None:
    samples = handlers.generate_samples(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        cache_dir=cache_dir,
        n_samples=n_samples,
        seed=seed,
        observable_columns=observable_columns,
    )
    print(samples)


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
