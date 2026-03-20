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


# TODO:
  Path A: Precomputed Cache (No live sims required)

  This is the realistic path. You already have api_simulation_default Parquet data on disk.

  What's needed:

  1. Generate (X, Y) pairs from existing sim data. You don't need to re-run sims. You already have outputs for one
  parameter configuration. The gap is: the pipeline needs outputs at multiple parameter configurations (LHS samples
  across the vio/mecillinam space) to fit a surrogate. A single experiment at one parameter point gives you
  aggregation and variance decomposition, but not sensitivity analysis.
  2. Run generate-samples against real vEcoli. This requires:
    - ecoli package importable (hard imports in uq/pipe.py, uq/generators/vecoli.py)
    - simData.cPickle at the expected path (you have this)
    - Each LHS sample triggers EcoliSim to run a full whole-cell simulation — this is the expensive step (~minutes to
  hours per sample depending on max_duration and generations)
    - With n_samples=200 and even a 3-minute sim, that's ~10 hours serial, or ~1 hour with 10 workers
  3. Once cached, quantify --precomputed-path ./cache runs Phase 1 + Phase 2 in seconds with no vEcoli dependency.

  Concrete steps:
  # Stage 1: Generate + cache (EXPENSIVE — run on HPC or with --max-workers)
  uv run uq generate-samples api_simulation_default \
      /path/to/sims ./cache \
      --n-samples 50 --max-workers 4

  # Stage 2: Analyze (FAST — seconds)
  uv run uq quantify api_simulation_default \
      /path/to/sims \
      --precomputed-path ./cache \
      --export-path ./results

  Blockers for this path:
  - ecoli package must be importable for Stage 1 (is it installed in this env?)
  - Stage 1 wall-clock time depends on sim duration
  - The VecoliSimulationFunc must correctly apply parameter variants (vio expression, mecillinam concentration) — this
   was the knockout gap that MISSING.md documented as fixed
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from pprint import pp
from typing import Literal

import dotenv
import numpy as np
import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq import handlers
from uq.common import get_repo_root, PARAM_CONFIG_DEMO
from uq.handlers import generate_samples
from uq.models import PipelineConfig, SamplingConfig
from uq.pce.models import PCEParameterSelectionConfig
from uq.pipe import Pipeline, execute_pipeline
from uq.pipeline import PipelineResult


console = Console()
dotenv.load_dotenv()


demo_app = typer.Typer(help="Perform Uncertainty Quantification.")
app = typer.Typer(help="Perform Uncertainty Quantification.")
app.add_typer(demo_app, name="demo")


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


@app.command()
def dashboard(
        run_mode: Literal["tk", "mo"] = "tk",
        results_path: str | None = None,
) -> None:
    """Launch the UQ results dashboard.

    Args:
        run_mode: 'tk' for Tkinter DAW (default) or 'mo' for Marimo notebook.
        results_path: Path to uq_results.json (tk mode only). If omitted,
            opens a file picker dialog.
    """
    if run_mode == "mo":
        _ = subprocess.run(["uv", "run", "marimo", "edit", "--no-token", "app/dashboard.py"], check=True)
    else:
        from app.uq_daw import run_tk_dashboard
        run_tk_dashboard(data_path=results_path)


def _verify_out_dirs(sim_base_path: str, experiment_ids: list[str]) -> bool:
    if not all([(Path(sim_base_path) / p).exists() for p in experiment_ids]):
        raise ValueError(
            f"One or more of the following experiment outdirs do not exist in the sim base path: {sim_base_path!s}:\n{experiment_ids}")


@app.command(
    name="sample",
    help=(""" \
        Generate Latin Hypercube Samples for UQ sensitivity analysis.

        By default, varies 5 physiologically relevant sim_data parameters.
        Use --params-file to specify custom parameters via a JSON file.
        Use --include-vio / --include-mecillinam for legacy vio/mecillinam mode.
    """)
)
def sample(
        experiment_ids: list[str],
        sim_base_path: str | None = None,
        cache_dir: str | None = None,
        n_samples: int = 200,
        seed: int = 42,
        observable_columns: list[str] | None = None,
        max_workers: int | None = None,
        max_duration: float = 10800.0,
        generations: int = 1,
        live: bool = True,
        include_vio: bool | None = None,
        include_mecillinam: bool | None = None,
        params_file: str | None = None,
        batch_dir: str | None = None
) -> None:
    """Generate LHS samples, evaluate simulation function, cache (X, Y).

    By default varies 5 physiologically relevant scalar sim_data
    parameters.  Pass --params-file to specify custom parameters.
    Pass --include-vio / --include-mecillinam for legacy mode.
    Pass --live to run real vEcoli simulations as subprocesses.
    """
    _verify_out_dirs(sim_base_path, experiment_ids)

    if batch_dir is not None:
        batch_dir = Path(batch_dir)

    samples = handlers.generate_samples(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        cache_dir=cache_dir,
        n_samples=n_samples,
        seed=seed,
        observable_columns=observable_columns,
        max_workers=max_workers,
        max_duration=max_duration,
        generations=generations,
        live=live,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
        params_file=params_file,
        batch_dir=batch_dir
    )
    print(samples)


@demo_app.command(name="sampling")
def demo_sampling() -> None:
    observable_columns = [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__growth",
    ]
    sim_base_path = Path("/Users/alexanderpatrie/sms/vecoli_data/outputs")
    experiment_ids = [p.name for p in sim_base_path.iterdir()]
    cache_dir = "examples/uq_artifacts/demos"
    batch_dir = Path("examples/uq_artifacts/batch")
    n_samples = 3
    seed = 1111
    samples = handlers.generate_samples(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        cache_dir=cache_dir,
        n_samples=n_samples,
        seed=seed,
        observable_columns=observable_columns,
        max_workers=4,
        max_duration=22.0,
        generations=1,
        live=True,
        include_vio=False,
        include_mecillinam=False,
        params_file=PARAM_CONFIG_DEMO.__str__(),
        batch_dir=batch_dir,
    )
    print(samples)


@app.command(name="export-configs")
def export_configs(
        sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
        batch_dir: str = typer.Argument(..., help="Output directory for batch configs"),
        n_samples: int = 200,
        seed: int = 42,
        include_vio: bool = True,
        include_mecillinam: bool = True,
        base_config_path: str | None = None,
        generations: int = 1,
        emitter: str = "parquet",
) -> None:
    """Export per-sample vEcoli configs for Nextflow/HPC batch execution.

    Generates LHS samples, applies variants to sim_data, writes per-sample
    JSON configs and pickled sim_data files.  Submit the resulting directory
    to Nextflow for parallel execution on HPC.
    """
    handlers.export_configs(
        sim_data_path=sim_data_path,
        batch_dir=batch_dir,
        n_samples=n_samples,
        seed=seed,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
        base_config_path=base_config_path,
        generations=generations,
        emitter=emitter,
    )


@app.command(name="collect-results")
def collect_results(
        batch_dir: str = typer.Argument(..., help="Directory from export-configs"),
        output_dir: str = typer.Argument(..., help="Root dir with per-sample Parquet outputs"),
        observable_columns: list[str] | None = None,
        cache_dir: str | None = None,
) -> None:
    """Collect completed Nextflow/HPC batch outputs into a PrecomputedCache.

    After Nextflow completes, run this to assemble (X, Y) from per-sample
    Parquet outputs.  The resulting cache can be passed to
    ``quantify --precomputed-path``.
    """
    handlers.collect_results(
        batch_dir=batch_dir,
        output_dir=output_dir,
        observable_columns=observable_columns,
        cache_dir=cache_dir,
    )


@app.command(name="configure-pipeline")
def configure_pipeline(name: str, dest: str | None = None):
    d = dest or os.path.join(os.getcwd(), f"{name}.json")
    # from uq.pipe import PipelineConfig
    config = PipelineConfig(
        experiment_ids=["api_simulation_default", "mecillinam", "test_violacein_with_metabolism"],
        sim_base_path=os.getenv("SIM_BASE_PATH"),
        export_path="uq_results",
        samples=SamplingConfig(
            cache_dir="uq_cache", n_samples=22, max_workers=4, include_vio=False, include_mecillinam=True
        ),
    )
    with open(d, "w") as fp:
        json.dump(config.model_dump(), fp, indent=3)


@app.command()
def flow_chart(rfc_id: str = "RFC006") -> None:
    if rfc_id != "RFC006":
        console.print("[red]Only RFC006 is supported.[/red]")
        return

    from rich.text import Text as RichText

    def _header(title: str) -> Panel:
        return Panel(
            RichText(title, style="bold white"),
            border_style="bright_magenta",
            box=box.DOUBLE_EDGE,
            padding=(0, 2),
        )

    def _step(title: str, body: str, border: str = "cyan") -> Panel:
        markup = f"[bold bright_yellow]{title}[/bold bright_yellow]\n{body}"
        return Panel(markup, border_style=border, box=box.ROUNDED, padding=(0, 1))

    def _phase(title: str, body: str, border: str = "green") -> Panel:
        markup = f"[bold bright_green]{title}[/bold bright_green]\n{body}"
        return Panel(markup, border_style=border, box=box.HEAVY, padding=(0, 1))

    def _arrow() -> str:
        return "[dim cyan]                                         |[/dim cyan]"

    def _arrow_v() -> str:
        return "[dim cyan]                                         v[/dim cyan]"

    # ── Header ──
    console.print()
    console.print(_header("RFC006 FULL UQ WORKFLOW"))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── Step 1 ──
    console.print(_step(
        "STEP 1: Define Parameter Space",
        "[bold magenta]Generic mode (default):[/bold magenta]\n"
        "  params = [SimDataParameter(name, attr_path, bounds), ...]\n"
        "  e.g. [cyan]\"process.transcription.fraction_active_rnap_free\"[/cyan] bounds=(0.25, 0.47)\n"
        "  Any scalar [bold]SimulationDataEcoli[/bold] attribute by dot-path.\n"
        "  Default: 3 params (see [cyan]DEFAULT_SIM_DATA_PARAMETERS[/cyan])\n"
        "  Custom: [green]--params-file params.json[/green]\n\n"
        "[bold magenta]Legacy mode:[/bold magenta] [green]--include-vio[/green] / [green]--include-mecillinam[/green]\n\n"
        "[dim]-> XSpaceVecoli with n parameters and bounds [a_i, b_i][/dim]",
    ))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── Step 2 ──
    console.print(_step(
        "STEP 2: Load Simulation Data",
        "[cyan]initialize_data(experiment_ids, sim_base_path, observable_columns)[/cyan]\n"
        "-> DatasetMultiExperiment:\n"
        "   .x = list[ParameterDataset]  (baseline simData.cPickle per experiment)\n"
        "   .y = Polars DataFrame from hive-partitioned Parquet\n"
        "   .parameter_space = XSpaceVecoli\n"
        "Outputs: transcriptome, proteome, metabolic fluxes, mass/volume/growth",
    ))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── Step 3 ──
    s1 = Panel("[bold yellow]S1: UNIFORM[/bold yellow]\nY = (1/N) sum Y_i", border_style="yellow", box=box.ROUNDED)
    s2 = Panel("[bold yellow]S2: BY GEN[/bold yellow]\nY_g (convergence)", border_style="yellow", box=box.ROUNDED)
    s3 = Panel("[bold yellow]S3: BY SEED[/bold yellow]\nY_s (exogenous)", border_style="yellow", box=box.ROUNDED)
    console.print(_step("STEP 3: Aggregation Strategies 1-3", ""))
    console.print(Columns([s1, s2, s3], equal=True, expand=True))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── Step 4 ──
    console.print(_step(
        "STEP 4: Variance Decomposition",
        "Var(Y) = Var_gen + Var_seed + Var_resid\n"
        "ANOVA-style decomposition per observable:\n"
        "  gen_frac = Var_between_gen / Var_total\n"
        "  residual = cell cycle + param sensitivity",
        border="yellow",
    ))

    console.print()
    console.print("[dim cyan]              Bulk population                          Cell cycle conditioned[/dim cyan]")
    console.print("[dim cyan]                       |                                        |[/dim cyan]")
    console.print("[dim cyan]                       v                                        v[/dim cyan]")

    # ── Phase 1 ──
    phase1 = _phase(
        "GSA PHASE 1: Population (Bulk)",
        "[bold bright_yellow]STEP 5a:[/bold bright_yellow] Morris Prescreening\n"
        "  EE_i = [f(x+De_i) - f(x)] / D\n"
        "  mu* = mean(|EE_i|),  sigma = std(EE_i)\n"
        "  n params -> K params (K << n)\n"
        "  -> [cyan]MorrisIndices[/cyan]\n\n"
        "[bold bright_yellow]STEP 6a:[/bold bright_yellow] PCE Surrogate\n"
        "  f(x) ~ sum c_a Psi_a(x)\n"
        "  Legendre basis, least-squares on LHS samples\n"
        "  -> [cyan]PCESurrogate[/cyan]\n\n"
        "[bold bright_yellow]STEP 7a:[/bold bright_yellow] Sobol Indices from PCE\n"
        "  S_i  = first-order (main effect)\n"
        "  S_Ti = total-order (w/ interactions)\n"
        "  -> [cyan]SobolIndices[/cyan]\n\n"
        "[bold]Output:[/bold] [bright_green]UqProfile(strat=POPULATION)[/bright_green]\n"
        "  SobolIndices x 1, PCESurrogate, MorrisIndices\n"
        "  AggregatedOutput x 3, variance_decomposition",
        border="bright_blue",
    )

    # ── Phase 2 ──
    phase2 = _phase(
        "GSA PHASE 2: Cell Cycle (Phenotypic)",
        "[bold bright_yellow]STEP 5b:[/bold bright_yellow] GSA-Informed Observable Selection\n"
        "  resid_i = 1 - gen_frac_i - seed_frac_i\n"
        "  Rank by residual -> top-K cell-cycle obs\n"
        "  -> [cyan]CellCycleRelevanceResult[/cyan]\n\n"
        "[bold bright_yellow]STEP 6b:[/bold bright_yellow] Koopman DMD -> Cell Cycle theta\n"
        "  lambda = |lambda| e^(iw),  theta = arg(phi)/2pi\n"
        "  Data-driven cell cycle coordinate in [0,1]\n\n"
        "[bold bright_yellow]STEP 6c/6d:[/bold bright_yellow] Strategy 4 Stratification\n"
        "  Bin timepoints by theta into n_bins stages\n"
        "  B-period -> C-period -> D-period\n\n"
        "[bold bright_yellow]STEP 7b:[/bold bright_yellow] Per-Stage PCE + Sobol\n"
        "  S_i^(k) per stage, orthogonal to Phase 1\n"
        "  -> [cyan]list[SobolIndices] (n_bins sets)[/cyan]\n\n"
        "[bold]Output:[/bold] [bright_green]UqProfile(strat=CELL_CYCLE)[/bright_green]\n"
        "  list[SobolIndices] x n_bins, PCESurrogate\n"
        "  CellCycleRelevanceResult, Koopman spectrum",
        border="bright_green",
    )

    console.print(Columns([phase1, phase2], equal=True, expand=True))

    console.print()
    console.print("[dim cyan]                       |                                        |[/dim cyan]")
    console.print("[dim cyan]                       +--------------------+-------------------+[/dim cyan]")
    console.print("[dim cyan]                                            |[/dim cyan]")
    console.print("[dim cyan]                                            v[/dim cyan]")

    # ── Feedback Loop ──
    console.print(Panel(
        "[bold bright_yellow]FEEDBACK LOOP[/bold bright_yellow] [dim](Phase 1 -> Phase 2, RFC006 S3)[/dim]\n\n"
        "Step 4: residual_frac per obs -> Step 5b: rank by residual, select top-K\n"
        "-> Step 6b: Koopman DMD on selected obs -> Step 6c/6d: theta-binned aggregation\n"
        "-> Step 7b: per-stage Sobol",
        border_style="bright_red",
        box=box.ROUNDED,
        padding=(0, 1),
    ))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── PipelineResult ──
    result_p1 = Panel(
        "[bold]Phase 1 (Population / Bulk)[/bold]\n"
        "[cyan]SobolIndices[/cyan] (1 set) -- S_i, S_Ti\n"
        "[cyan]PCESurrogate[/cyan] -- polynomial f(x)\n"
        "[cyan]MorrisIndices[/cyan] -- mu*, sigma\n"
        "[cyan]variance_decomposition[/cyan]\n"
        "[cyan]AggregatedOutput[/cyan] x 3",
        border_style="bright_blue",
        box=box.ROUNDED,
    )
    result_p2 = Panel(
        "[bold]Phase 2 (Cell Cycle / Phenotypic)[/bold]\n"
        "[cyan]list[SobolIndices][/cyan] x n_bins\n"
        "[cyan]PCESurrogate[/cyan] (stage-conditioned)\n"
        "[cyan]CellCycleRelevanceResult[/cyan]\n"
        "[cyan]cell_cycle_profile[/cyan]\n"
        "[cyan]Koopman spectrum[/cyan]",
        border_style="bright_green",
        box=box.ROUNDED,
    )
    console.print(Panel(
        Columns([result_p1, result_p2], equal=True, expand=True),
        title="[bold white on magenta] PipelineResult -- Complete RFC006 Output [/bold white on magenta]",
        subtitle="[dim]Phase 1 & Phase 2 are orthogonal decompositions of the same total variance[/dim]",
        border_style="magenta",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    ))
    console.print(_arrow())
    console.print(_arrow_v())

    # ── CLI Workflow ──
    console.print(Panel(
        "[bold bright_yellow]TWO-STAGE WORKFLOW (CLI)[/bold bright_yellow]\n\n"
        "[bold]Stage 1[/bold] [dim](compute-intensive, via vEcoli workflow.py + Nextflow):[/dim]\n"
        "[green]uv run uq sample <experiment_ids>[/green] \\\n"
        "    [green]--sim-base-path[/green] /path/to/sims \\\n"
        "    [green]--cache-dir[/green] ./uq_cache [green]--n-samples[/green] 200 [green]--live[/green] \\\n"
        "    [green]--params-file[/green] params.json [green]--batch-dir[/green] ./batch\n\n"
        "[dim]Execution: LHS -> sim_data_setattr variant (op: \"zip\") ->[/dim]\n"
        "[dim]           workflow.py -> Nextflow -> Parquet -> PrecomputedCache[/dim]\n"
        "[dim]           No EcoliSim in UQ process memory.[/dim]\n\n"
        "                         [bright_cyan]|[/bright_cyan]\n"
        "                         [bright_cyan]v[/bright_cyan]  PrecomputedCache (X.npy, Y.npy, timeseries/)\n\n"
        "[bold]Stage 2[/bold] [dim](fast, repeatable, no simulation):[/dim]\n"
        "[green]uv run uq quantify <experiment_ids>[/green] \\\n"
        "    [green]--sim-base-path[/green] /path/to/sims \\\n"
        "    [green]--precomputed-path[/green] ./uq_cache [green]--export-path[/green] ./uq_results",
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
        title="[bold white on cyan] CLI [/bold white on cyan]",
    ))
    console.print()

    pass  # all rendering done above via Rich


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
