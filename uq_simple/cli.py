"""
uq_simple CLI — scientifically transparent UQ for vEcoli.

Three commands:

    uv run uq-simple sample ...     # Stage 1: generate + cache LHS samples
    uv run uq-simple quantify ...   # Stage 2: all 4 RFC006 strategies via PyTUQ PCE
    uv run uq-simple dashboard ...  # Interactive visualization (tk or marimo)

The ``sample`` command is identical to ``uq sample`` — uses LHS sampling,
subprocess-based vEcoli execution, and PrecomputedCache.  Also caches
per-row generation/lineage_seed metadata for strategies 2-3.

The ``quantify`` command fits PCE surrogates via ``pytuq.surrogates.pce.PCE``
(following the UQPC workflow) and computes Sobol indices across all 4
RFC006 aggregation strategies:
    1. Uniform (bulk) — population-averaged sensitivity
    2. By generation — controls for convergence toward steady-state
    3. By lineage seed — controls for stochastic variance
    4. Growth-stratified — θ = normalized log(dry_mass), no spectral decomposition

Regression backends (--regression): lsq (default), bcs (sparse), anl (analytical).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
app = typer.Typer(help="Scientifically transparent UQ for vEcoli.")


# ── sample: reuse uq sample directly ────────────────────────────────


@app.command()
def sample(
    experiment_ids: list[str] = [
        "first_sms_perturb_growth",
        "sms_variants_multigeneration",
        "sms_multiseed",
        "sms",
        "sms_multiseed_multigen",
        "single",
        "sms_multigen",
        "sms_perturb_growth_10800",
        "sms_perturb",
        "test_installation",
        "sms_perturb_growth",
        "sms_variants",
        "sms_single",
        "multigeneration",
    ],
    sim_base_path: str | None = None,
    cache_dir: str | None = None,
    n_samples: int = 200,
    seed: int = 42,
    observable_columns: list[str] | None = None,
    max_workers: int | None = None,
    max_duration: float = 10800.0,
    generations: int = 1,
    live: bool = True,
    params_file: str | None = None,
    batch_dir: str | None = None,
) -> None:
    """Generate LHS samples, run vEcoli, cache (X, Y).

    Identical to ``uq sample`` — uses Latin Hypercube Sampling,
    subprocess-based vEcoli execution, and PrecomputedCache.

    Use --generations >= 2 to enable Strategy 2 (by-generation GSA)
    in the quantify step.  Generation and lineage seed metadata are
    automatically extracted from hive-partitioned Parquet outputs.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from uq.handlers import generate_samples, verify_out_dirs

    verify_out_dirs(sim_base_path, experiment_ids)

    _batch_dir = Path(batch_dir) if batch_dir else None

    with Progress(
        SpinnerColumn("dots", style="bold magenta"),
        TextColumn("[bold cyan]{task.description:<35}"),
        BarColumn(bar_width=40, complete_style="magenta", finished_style="green"),
        TextColumn("[bold green]{task.percentage:>5.1f}%"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Initializing", total=100)

        def _on_progress(description: str, advance: int) -> None:
            progress.update(task, advance=advance, description=description)

        result = generate_samples(
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
            params_file=params_file,
            batch_dir=_batch_dir,
            on_progress=_on_progress,
        )

    console.print(
        f"[bold green]Cached {result.X.shape[0]} samples[/bold green] "
        f"({result.X.shape[1]} params, {result.Y.shape[1]} outputs) "
        f"to [cyan]{cache_dir}[/cyan]"
    )
    if result.Y_timeseries is not None:
        console.print(f"  [dim]Timeseries: {len(result.Y_timeseries)} samples[/dim]")
    if result.Y_timeseries_meta is not None:
        console.print("  [dim]Metadata: generation/seed labels cached (strategies 2-3 enabled)[/dim]")
    elif generations < 2:
        console.print(
            "  [dim yellow]Hint: use --generations >= 2 to enable strategy 2 (by-generation GSA)[/dim yellow]"
        )


# ── quantify: simplified pipeline ────────────────────────────────────


@app.command()
def quantify(
    experiment_ids: list[str],
    sim_base_path: str,
    precomputed_path: str,
    export_path: str,
    n_bins: int = 10,
    polynomial_order: int = 1,
    observable_columns: list[str] | None = None,
    regression: str = "lsq",
) -> None:
    """Run the simplified UQ pipeline (all 4 RFC006 strategies).

    Loads cached samples, fits PCE surrogates, computes Sobol indices
    across all four aggregation strategies:

    \b
    Strategy 1: Uniform (bulk) — population-averaged sensitivity
    Strategy 2: By generation — convergence control (needs generations >= 2)
    Strategy 3: By lineage seed — stochastic variance control
    Strategy 4: Growth-stratified — θ = normalized log(dry_mass)

    \b
    Regression methods (--regression):
      lsq  Least squares (default) — standard overdetermined solve
      bcs  Bayesian Compressed Sensing — sparse PCE (fewer terms)
      anl  Analytical — posterior predictive with uncertainty

    Strategies 2-3 require generation/seed metadata in the cache
    (automatically stored when sampling with live vEcoli).
    """
    from uq.pipe import initialize_datasets
    from uq.sampling import PrecomputedCache
    from uq_simple.pipeline import run_pipeline

    obs = observable_columns or [
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__mass__growth",
    ]

    console.print("[bold cyan]Loading cached samples...[/bold cyan]")
    cache = PrecomputedCache.load(precomputed_path)

    console.print("[bold cyan]Loading parameter space...[/bold cyan]")
    ds = initialize_datasets(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        observable_columns=obs,
    )

    strategies_available = "1"
    if cache.Y_timeseries_meta is not None:
        strategies_available = "1,2,3,4"
    else:
        strategies_available = "1,4"
    console.print(
        f"[bold cyan]Running pipeline[/bold cyan] "
        f"(order={polynomial_order}, bins={n_bins}, "
        f"regression={regression}, strategies={strategies_available})"
    )
    result = run_pipeline(
        cache=cache,
        param_space=ds.parameter_space,
        observable_names=obs,
        polynomial_order=polynomial_order,
        n_bins=n_bins,
        export_path=export_path,
        regression=regression,
    )

    _print_report(result)
    console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")


# ── Report rendering ─────────────────────────────────────────────────


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _print_report(result) -> None:
    """Render results as a rich terminal report."""
    from rich.columns import Columns

    from uq_simple.pipeline import _stage_description

    console.print()

    header = Text()
    header.append("  UQ SENSITIVITY REPORT  ", style="bold white on magenta")
    header.append("  vEcoli Whole-Cell Model", style="bold cyan")
    console.print(Panel(header, box=box.DOUBLE_EDGE, border_style="magenta", padding=(0, 1)))
    console.print("[dim]  Methods: PCE surrogate (Legendre, PyTUQ) + Sobol indices (Sudret 2008)[/dim]")
    console.print("[dim]  Strategies: 1=uniform, 2=by generation, 3=by seed, 4=growth-stratified (RFC006)[/dim]")
    console.print("[dim]  Refs: Macklin et al. Science 2020; Ahn-Horst et al. npj Syst Biol Appl 2022[/dim]")
    console.print()

    # Phase 1 / Strategy 1
    _print_sobol_table(
        "STRATEGY 1 // POPULATION-AVERAGED (all cells, all times)",
        result.population_sobol,
        "cyan",
    )

    # Strategy 2: by generation
    if result.per_generation_sobol:
        gen_tables = []
        for gen, sobol in sorted(result.per_generation_sobol.items()):
            gen_tables.append(
                _sobol_table(
                    f"Generation {gen}",
                    sobol,
                    "blue",
                    n_top=3,
                )
            )
        console.print(
            Panel(
                Columns(gen_tables, equal=True, expand=True),
                title="[bold blue]STRATEGY 2 // BY GENERATION (convergence control)[/bold blue]",
                subtitle="[dim]Controls for transient dynamics in early generations[/dim]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # Strategy 3: by lineage seed
    if result.per_seed_sobol:
        seed_tables = []
        for seed, sobol in sorted(result.per_seed_sobol.items()):
            seed_tables.append(
                _sobol_table(
                    f"Seed {seed}",
                    sobol,
                    "yellow",
                    n_top=3,
                )
            )
        console.print(
            Panel(
                Columns(seed_tables, equal=True, expand=True),
                title="[bold yellow]STRATEGY 3 // BY LINEAGE SEED (stochastic variance control)[/bold yellow]",
                subtitle="[dim]Controls for gene expression noise and stochastic partitioning[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    # Phase 2 / Strategy 4: growth-stratified
    if result.per_stage_sobol:
        n = len(result.per_stage_sobol)
        tables = []
        for i, sobol in enumerate(result.per_stage_sobol):
            lo, hi = i / n, (i + 1) / n
            desc = _stage_description(i, n)
            tables.append(
                _sobol_table(
                    f"θ {lo:.0%}–{hi:.0%} ({desc.split('(')[0].strip()})",
                    sobol,
                    "green",
                    n_top=3,
                )
            )

        console.print(
            Panel(
                Columns(tables, equal=True, expand=True),
                title=f"[bold green]STRATEGY 4 // GROWTH-STRATIFIED SENSITIVITY ({n} stages)[/bold green]",
                subtitle=(
                    "[dim]θ = normalized log(dry_mass), 0 = birth, 1 = division. "
                    "How does parameter importance change as the cell grows?[/dim]"
                ),
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )


def _sobol_table(title: str, sobol, border: str, n_top: int = 10) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style=border,
        title=title,
        title_style=f"bold {border}",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    table.add_column("S_Ti", style="bright_green", justify="right")

    total = sobol.total_order
    if total.ndim > 1:
        total = np.mean(total, axis=0)
    ranked = np.argsort(total)[::-1][:n_top]
    for i in ranked:
        name = sobol.parameter_names[i] if i < len(sobol.parameter_names) else f"x{i}"
        table.add_row(name, _pct(total[i]))
    return table


def _print_sobol_table(title: str, sobol, border: str) -> None:
    console.print(
        Panel(
            _sobol_table(title, sobol, border),
            border_style=border,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


@app.command()
def dashboard(
    results_path: str | None = None,
    run_mode: str = "tk",
) -> None:
    """Launch the uq_simple interactive dashboard.

    \b
    --run-mode tk   Tkinter DAW (default) — draggable parameter markers
    --run-mode mo   Marimo notebook — slider-reactive
    """
    import subprocess as _sp

    if run_mode == "mo":
        _sp.run(["uv", "run", "marimo", "edit", "--no-token", "app/dashboard_simple.py"], check=True)
    else:
        from app.uq_daw_simple import run_tk_dashboard_simple

        run_tk_dashboard_simple(data_path=results_path)


@app.command()
def tui() -> None:
    """Launch the UQPC interactive terminal UI (Textual).

    \b
    Four tabs: Sample, Quantify, Results, Log.
    Keyboard: [s] Sample  [u] Quantify  [r] Results  [l] Log  [q] Quit
    """
    from uq.common.tui import UQPCApp

    UQPCApp().run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
