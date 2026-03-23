"""
uq_simple CLI — scientifically transparent UQ for vEcoli.

Two commands mirror the full ``uq`` CLI:

    uv run uq-simple sample ...     # Stage 1: generate + cache LHS samples
    uv run uq-simple quantify ...   # Stage 2: PCE / Sobol / growth-stratified

The ``sample`` command is identical to ``uq sample`` — it uses the same
LHS sampling, subprocess-based vEcoli execution, and PrecomputedCache.

The ``quantify`` command replaces Koopman spectral decomposition with
growth-stratified sensitivity analysis (normalized log-mass ratio) —
no claim about cell cycle phases, just how sensitivity changes as cells grow.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import numpy as np

console = Console()
app = typer.Typer(help="Scientifically transparent UQ for vEcoli.")


# ── sample: reuse uq sample directly ────────────────────────────────


@app.command()
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
    params_file: str | None = None,
    batch_dir: str | None = None,
) -> None:
    """Generate LHS samples, run vEcoli, cache (X, Y).

    Identical to ``uq sample`` — uses Latin Hypercube Sampling,
    subprocess-based vEcoli execution, and PrecomputedCache.
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
) -> None:
    """Run the simplified UQ pipeline (growth-stratified sensitivity).

    Loads cached samples, fits PCE surrogates, computes Sobol indices
    for both bulk (Phase 1) and growth-stratified (Phase 2) outputs.
    Phase 2 bins by normalized log(dry_mass) to reveal how parameter
    importance changes as cells grow.
    """
    from uq.pipe import initialize_data
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
    ds = initialize_data(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        observable_columns=obs,
    )

    console.print(
        f"[bold cyan]Running pipeline[/bold cyan] "
        f"(order={polynomial_order}, bins={n_bins}, "
        f"stratification=growth_progress)"
    )
    result = run_pipeline(
        cache=cache,
        param_space=ds.parameter_space,
        observable_names=obs,
        polynomial_order=polynomial_order,
        n_bins=n_bins,
        export_path=export_path,
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
    console.print(
        "[dim]  Methods: PCE surrogate (Legendre, PyTUQ) + Sobol indices (Sudret 2008)[/dim]"
    )
    console.print(
        "[dim]  Phase 2 stratification: θ = normalized log(dry_mass) — growth progress, not cell cycle[/dim]"
    )
    console.print(
        "[dim]  Refs: Macklin et al. Science 2020; Ahn-Horst et al. npj Syst Biol Appl 2022[/dim]"
    )
    console.print()

    # Phase 1
    _print_sobol_table(
        "PHASE 1 // POPULATION-AVERAGED (all cells, all times)",
        result.population_sobol, "cyan",
    )

    # Phase 2
    if result.per_stage_sobol:
        n = len(result.per_stage_sobol)
        tables = []
        for i, sobol in enumerate(result.per_stage_sobol):
            lo, hi = i / n, (i + 1) / n
            desc = _stage_description(i, n)
            tables.append(_sobol_table(
                f"θ {lo:.0%}–{hi:.0%} ({desc.split('(')[0].strip()})",
                sobol, "green", n_top=3,
            ))

        console.print(Panel(
            Columns(tables, equal=True, expand=True),
            title=f"[bold green]PHASE 2 // GROWTH-STRATIFIED SENSITIVITY ({n} stages)[/bold green]",
            subtitle=(
                "[dim]θ = normalized log(dry_mass), 0 = birth, 1 = division. "
                "How does parameter importance change as the cell grows?[/dim]"
            ),
            border_style="green", box=box.ROUNDED, padding=(0, 1),
        ))


def _sobol_table(title: str, sobol, border: str, n_top: int = 10) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY, show_header=True,
        header_style="bold magenta", border_style=border,
        title=title, title_style=f"bold {border}",
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
    console.print(Panel(
        _sobol_table(title, sobol, border),
        border_style=border, box=box.ROUNDED, padding=(0, 1),
    ))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
