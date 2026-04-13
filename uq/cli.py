"""
uq CLI — scientifically transparent UQ for vEcoli.

Three commands:

    uv run uq sample ...     # Stage 1: generate + cache samples (PCRV germ sampling)
    uv run uq quantify ...   # Stage 2: all 4 RFC006 strategies via PyTUQ PCE
    uv run uq dashboard ...  # Interactive visualization (tk or marimo)

The ``sample`` command is identical to ``uq sample`` — uses PCRV.sampleGerm()
(PyTUQ-native random sampling), subprocess-based vEcoli execution, and
PrecomputedCache.  Also caches
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

from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq.models import CliType

console = Console()
app = typer.Typer(
    help=(
        "Scientifically transparent UQ for vEcoli.\n\n"
        "Tip: every subcommand accepts a trailing ``help`` word as an alias "
        "for ``--help``, e.g. ``uq sample help``, ``uq quantify help``."
    ),
)


# ── sample: reuse uq sample directly ────────────────────────────────


@app.command()
def sample(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    cache_dir: str = typer.Option("./uq_cache", help="Cache output directory"),
    n_samples: int = 20,
    n_test: int = typer.Option(
        0,
        help="Extra held-out validation samples (UQPC --ntst). Evaluated via vEcoli "
        "alongside training samples; used by `quantify` to compute test relative errors.",
    ),
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    max_duration: float = 10800.0,
    params_file: str | None = None,
    observables: list[str] = typer.Option(
        ["mass"],
        help="Observable presets to extract (cd1 analysis modules). "
        "Options: mass, higher_order, exchange_fluxes, transcriptome, "
        "proteome, fluxome. Pass multiple: --observables higher_order "
        "--observables exchange_fluxes",
    ),
    generation_lower_bound: int = typer.Option(
        0,
        help="Skip generations below this value when aggregating Y "
        "(cd1 generation_lower_bound). 0 = keep all.",
    ),
) -> None:
    """UQPC Steps 1-3: sample via PCRV.sampleGerm(), run vEcoli workflow.py.

    Generates samples using PyTUQ's native PCRV sampling, evaluates
    vEcoli as a subprocess via runscripts/workflow.py, and caches
    (X, Y, timeseries) to disk as a PrecomputedCache.

    \b
    Use --generations >= 2 to enable Strategy 2 (by-generation GSA).
    Use --n-test > 0 for held-out validation (UQPC --ntst).
    Use --observables to select cd1-style observable categories:
      mass             Raw mass/growth scalars (default)
      higher_order     Derived metrics: doubling time, biomass composition
      exchange_fluxes  External metabolite fluxes (~87)
      transcriptome    mRNA cistron counts (~4300 genes)
      proteome         Protein monomer counts (~4300 monomers)
      fluxome          Base reaction fluxes (~2800, dry-mass normalized)
    Use --generation-lower-bound N to skip early generations.
    """
    import json as _json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time as _time

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import DEFAULT_SIM_DATA_PARAMETERS, ParameterDataset
    from libuq.sampling import PrecomputedCache
    from uq.tui import (
        _build_config,
        _build_variants_from_samples,
        _collect_variant_timeseries,
        _count_completed_variants,
        _get_vecoli_root,
    )
    from uq.workflow import _setup_input_pc

    # ── Step 1: Setup inputs ──
    sim_data_path = str(Path(sim_data_path).resolve())
    cache_path = Path(cache_dir).resolve()

    console.print(f"[bold cyan]Sampling:[/bold cyan] {n_samples} variants, {n_init_sims} seeds, {generations} gens")
    console.print(f"  [dim]simData: {sim_data_path}[/dim]")
    console.print(f"  [dim]cache:   {cache_path}[/dim]")

    console.print("[dim]Step 1: loading simData...[/dim]")
    ds = ParameterDataset(sim_data_path=sim_data_path)
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        parameters = [SimDataParameter.from_dict(p) for p in raw]
    else:
        parameters = None
    param_space = ds.to_parameter_space(parameters=parameters)
    bounds = np.array(param_space.parameter_bounds)
    console.print(f"  [dim]{param_space.n_parameters} params: {param_space.parameter_names}[/dim]")

    # ── Step 2: Generate samples via PCRV ──
    console.print("[dim]Step 2: PCRV.sampleGerm()...[/dim]")
    input_pc, _, _ = _setup_input_pc(bounds)
    np.random.seed(seed)
    germ_train = input_pc.sampleGerm(n_samples)
    X_train = input_pc.evalPC(germ_train)
    console.print(f"  [dim]X shape: {X_train.shape}[/dim]")

    # UQPC ``--ntst``: draw additional held-out validation samples
    germ_test: np.ndarray | None = None
    X_test: np.ndarray | None = None
    if n_test > 0:
        np.random.seed(seed + 1)
        germ_test = input_pc.sampleGerm(n_test)
        X_test = input_pc.evalPC(germ_test)
        console.print(f"  [dim]X_test shape: {X_test.shape} (held-out validation)[/dim]")

    # Concatenate train + test; vEcoli evaluates all variants in one workflow.
    X_all = np.vstack([X_train, X_test]) if X_test is not None else X_train

    # ── Step 3: Build config + run workflow.py ──
    batch_dir = cache_path / "_batch"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    output_dir = batch_dir / "output"
    output_dir.mkdir(exist_ok=True)
    experiment_id = "uqpc_batch"

    variants = _build_variants_from_samples(X_all, param_space._sim_data_parameters)
    config = _build_config(
        sim_data_path=sim_data_path,
        output_dir=str(output_dir),
        variants_section=variants,
        experiment_id=experiment_id,
        n_init_sims=n_init_sims,
        generations=generations,
        max_duration=max_duration,
    )
    config_path = batch_dir / "workflow_config.json"
    config_path.write_text(_json.dumps(config, indent=2))
    console.print(f"  [dim]Config JSON: {config_path}[/dim]")

    vecoli_root = _get_vecoli_root()
    nf_temp = Path(vecoli_root) / "nextflow_temp" / experiment_id
    if nf_temp.exists():
        shutil.rmtree(nf_temp)

    n_variants = X_all.shape[0]
    total_sims = (n_variants + 1) * n_init_sims * generations
    history_base = output_dir / experiment_id / "history"
    ansi_re = re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\[\d*[A-GJK]|\x07")

    console.print(f"[dim]Step 3: running vEcoli workflow.py ({total_sims} sims)...[/dim]")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = vecoli_root + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"

    workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
    cmd = [sys.executable, workflow_script, "--config", str(config_path)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=vecoli_root,
        env=env,
    )

    with Progress(
        SpinnerColumn("dots", style="bold magenta"),
        TextColumn("[bold cyan]{task.description:<50}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="green"),
        TextColumn("[bold green]{task.percentage:>5.1f}%"),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Launching Nextflow", total=total_sims)
        start = _time.monotonic()

        while proc.poll() is None:
            # Read available stdout
            if proc.stdout is not None:
                fd = proc.stdout.fileno()
                try:
                    chunk = os.read(fd, 8192)
                except OSError:
                    chunk = b""
                if chunk:
                    for raw_line in chunk.decode("utf-8", errors="replace").splitlines():
                        clean = ansi_re.sub("", raw_line).strip()
                        if not clean:
                            continue
                        low = clean.lower()
                        if any(kw in low for kw in ["error", "fail", "exception"]):
                            console.print(f"  [red]{clean}[/red]")
                        elif any(kw in low for kw in ["completed at", "duration", "succeeded"]):
                            console.print(f"  [green]{clean}[/green]")
                        elif any(kw in low for kw in ["warn", "note"]):
                            console.print(f"  [yellow]{clean}[/yellow]")
                        else:
                            console.print(f"  [dim]{clean}[/dim]")

            # Poll completed variants
            n_done = _count_completed_variants(history_base)
            elapsed = int(_time.monotonic() - start)
            progress.update(
                task,
                completed=n_done,
                description=f"Simulating  {n_done}/{total_sims}  [{elapsed}s]",
            )
            _time.sleep(0.5)

    # Drain remaining stdout
    if proc.stdout is not None:
        remaining = proc.stdout.read()
        if remaining:
            for raw_line in remaining.decode("utf-8", errors="replace").splitlines():
                clean = ansi_re.sub("", raw_line).strip()
                if clean:
                    console.print(f"  [dim]{clean}[/dim]")

    exit_code = proc.returncode
    if exit_code != 0:
        console.print(f"[yellow]workflow.py exited with code {exit_code}[/yellow]")

    # ── Step 4: Collect + cache ──
    console.print("[dim]Collecting Parquet outputs...[/dim]")
    if not history_base.exists():
        candidates = list(output_dir.glob("*/history"))
        if candidates:
            history_base = candidates[0]

    from uq.observables import collect_observables

    console.print(f"[dim]Observables: {observables}, gen_lower_bound={generation_lower_bound}[/dim]")
    Y_all, obs, Y_ts_all, Y_meta_all = collect_observables(
        history_base,
        n_variants,
        presets=observables,
        generation_lower_bound=generation_lower_bound,
    )

    # Split train and held-out test portions
    Y_agg = Y_all[:n_samples]
    Y_ts = Y_ts_all[:n_samples] if Y_ts_all else Y_ts_all
    Y_meta = Y_meta_all[:n_samples] if Y_meta_all else Y_meta_all

    Y_test_arr: np.ndarray | None = None
    Y_test_ts: list | None = None
    Y_test_meta: list | None = None
    if n_test > 0:
        Y_test_arr = Y_all[n_samples:]
        Y_test_ts = Y_ts_all[n_samples:] if Y_ts_all else None
        Y_test_meta = Y_meta_all[n_samples:] if Y_meta_all else None

    cache_path.mkdir(parents=True, exist_ok=True)
    cache = PrecomputedCache(
        cache_dir=cache_path,
        X=X_train,
        Y=Y_agg,
        parameter_names=param_space.parameter_names,
        metadata={"bounds": bounds.tolist(), "seed": seed, "observable_columns": obs},
        Y_timeseries=Y_ts,
        Y_timeseries_meta=Y_meta,
        X_test=X_test,
        Y_test=Y_test_arr,
        Y_test_timeseries=Y_test_ts,
        Y_test_timeseries_meta=Y_test_meta,
    )
    cache.save()
    np.save(cache_path / "germ_train.npy", germ_train)
    if germ_test is not None:
        np.save(cache_path / "germ_test.npy", germ_test)

    console.print(
        f"[bold green]Cached {Y_agg.shape[0]} training samples[/bold green] "
        f"({Y_agg.shape[1]} outputs) to [cyan]{cache_path}[/cyan]"
    )
    if Y_test_arr is not None:
        console.print(f"  [dim]Held-out validation: {Y_test_arr.shape[0]} samples[/dim]")
    if Y_ts:
        console.print(f"  [dim]Timeseries: {len(Y_ts)} samples[/dim]")
    if Y_meta:
        console.print("  [dim]Metadata: generation/seed labels (strategies 2-3 enabled)[/dim]")


# ── quantify ─────────────────────────────────────────────────────────


@app.command()
def quantify(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    cache_dir: str = typer.Option("./uq_cache", help="Cache from sample step"),
    export_path: str = typer.Option("./uq_results", help="Export directory"),
    n_bins: int = 10,
    polynomial_order: int = 2,
    regression: str = "lsq",
    tol: float = typer.Option(1e-3, help="BCS sparsity tolerance (UQPC --tol, only used when --regression=bcs)"),
) -> None:
    """UQPC Steps 4-5: fit PCE surrogates, compute Sobol (all 4 strategies).

    \b
    Strategy 1: Uniform (bulk) — population-averaged sensitivity
    Strategy 2: By generation — convergence control (needs generations >= 2)
    Strategy 3: By lineage seed — stochastic variance control
    Strategy 4: Growth-stratified — θ = normalized log(dry_mass)

    \b
    Regression methods (--regression):
      lsq  Least squares (default)
      bcs  Bayesian Compressed Sensing (sparse)
      anl  Analytical Bayesian
    """
    from uq.workflow import quantify as wf_quantify

    console.print(
        f"[bold cyan]Quantify:[/bold cyan] order={polynomial_order}, bins={n_bins}, regression={regression}, tol={tol}"
    )

    result = wf_quantify(
        cache_dir=cache_dir,
        sim_data_path=sim_data_path,
        polynomial_order=polynomial_order,
        n_bins=n_bins,
        regression=regression,
        tolerance=tol,
        export_path=export_path,
    )

    _print_report(result)
    console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")


# ── Report rendering ─────────────────────────────────────────────────


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _print_report(result: Any) -> None:
    """Render QuantifyResult as a rich terminal report."""
    from rich.columns import Columns

    console.print()

    header = Text()
    header.append("  UQ SENSITIVITY REPORT  ", style="bold white on magenta")
    header.append("  vEcoli Whole-Cell Model", style="bold cyan")
    console.print(Panel(header, box=box.DOUBLE_EDGE, border_style="magenta", padding=(0, 1)))
    console.print("[dim]  Methods: PCE surrogate (Legendre, PyTUQ) + Sobol indices (Sudret 2008)[/dim]")
    console.print("[dim]  Strategies: 1=uniform, 2=by generation, 3=by seed, 4=growth-stratified (RFC006)[/dim]")
    console.print("[dim]  Refs: Macklin et al. Science 2020; Ahn-Horst et al. npj Syst Biol Appl 2022[/dim]")
    console.print()

    # Surrogate quality diagnostics (UQPC step 5)
    _print_relerr_panel(result.strategy1)

    # Strategy 1: population
    _print_sobol_table(
        "STRATEGY 1 // POPULATION-AVERAGED (all cells, all times)",
        result.strategy1.sobol,
        "cyan",
    )

    # Strategy 2: by generation
    if result.strategy2:
        gen_tables = []
        for gen, r in sorted(result.strategy2.items()):
            gen_tables.append(_sobol_table(f"Generation {gen}", r.sobol, "blue", n_top=3))
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
    if result.strategy3:
        seed_tables = []
        for seed, r in sorted(result.strategy3.items()):
            seed_tables.append(_sobol_table(f"Seed {seed}", r.sobol, "yellow", n_top=3))
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

    # Strategy 4: growth-stratified
    if result.strategy4_per_stage:
        n = len(result.strategy4_per_stage)
        tables = []
        for i, r in enumerate(result.strategy4_per_stage):
            lo, hi = i / n, (i + 1) / n
            tables.append(_sobol_table(f"θ {lo:.0%}–{hi:.0%}", r.sobol, "green", n_top=3))
        console.print(
            Panel(
                Columns(tables, equal=True, expand=True),
                title=f"[bold green]STRATEGY 4 // GROWTH-STRATIFIED ({n} stages)[/bold green]",
                subtitle="[dim]θ = normalized log(dry_mass), 0=birth, 1=division[/dim]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )


def _sobol_table(title: str, sobol: Any, border: str, n_top: int = 10) -> Table:
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


def _print_relerr_panel(s1: Any) -> None:
    """Show PCE surrogate relative errors (UQPC step 5 diagnostic)."""
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="SURROGATE QUALITY // RELATIVE ERRORS",
        title_style="bold magenta",
    )
    table.add_column("OBSERVABLE", style="bold yellow", justify="left")
    table.add_column("TRAIN", style="bright_green", justify="right")
    if s1.relerr_test is not None:
        table.add_column("TEST", style="bright_cyan", justify="right")

    train = s1.relerr_train
    test = s1.relerr_test
    n_out = len(train)
    for i in range(n_out):
        row = [f"output[{i}]", f"{train[i]:.3e}"]
        if test is not None:
            row.append(f"{test[i]:.3e}")
        table.add_row(*row)

    console.print(
        Panel(
            table,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
            subtitle="[dim]||Y − Ŷ||₂ / ||Y||₂ per output column[/dim]",
        )
    )


def _print_sobol_table(title: str, sobol: Any, border: str) -> None:
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
    """Launch the uq interactive dashboard.

    \b
    --run-mode tk   Tkinter DAW (default) — draggable parameter markers
    --run-mode mo   Marimo notebook — slider-reactive

    If --results-path is not given, looks for ./uq_results/uq_results.json.
    """
    import subprocess as _sp

    # Auto-detect results path
    if results_path is None:
        default = Path("./uq_results/uq_results.json")
        if default.exists():
            results_path = str(default)

    if run_mode == "mo":
        _sp.run(["uv", "run", "marimo", "edit", "--no-token", "app/dashboard_simple.py"], check=True)  # noqa: S607
    else:
        from app.uq_daw_simple import run_tk_dashboard_simple

        run_tk_dashboard_simple(data_path=results_path)  # type: ignore[no-untyped-call]


@app.command()
def tui() -> None:
    """Launch the UQPC interactive terminal UI (Textual).

    \b
    Sidebar: config + sample + quantify + results.
    Keyboard: q=quit  d=theme
    """
    from uq.tui import UQPCApp

    UQPCApp().run()


@app.command()
def gui() -> None:
    """Launch the UQPC interactive GUI (marimo).

    Full workflow in the browser: configure, sample, quantify, explore.
    """
    import subprocess as _sp

    _sp.run(
        ["uv", "run", "marimo", "run", "--no-token", "app/gui.py"],  # noqa: S607
        check=True,
    )


@app.command(name="show-config")
def show_config(
    sim_data_path: str = typer.Argument(..., help="Path to simData.cPickle"),
    n_samples: int = 5,
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    max_duration: float = 10800.0,
    params_file: str | None = None,
    output_file: str | None = typer.Option(None, help="Write JSON to this file instead of stdout"),
) -> None:
    """Show the full vEcoli workflow config JSON that `sample` would pass.

    \b
    Generates the config without running vEcoli — useful for review,
    debugging, or manual execution via workflow.py --config.
    """
    import json as _json

    from libuq.pipeline.models import SimDataParameter
    from libuq.pipeline.param_loader import ParameterDataset
    from uq.tui import _build_config, _build_variants_from_samples
    from uq.workflow import _setup_input_pc

    sim_data_path = str(Path(sim_data_path).resolve())
    ds = ParameterDataset(sim_data_path=sim_data_path)
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        parameters = [SimDataParameter.from_dict(p) for p in raw]
    else:
        parameters = None
    param_space = ds.to_parameter_space(parameters=parameters)
    bounds = np.array(param_space.parameter_bounds)

    input_pc, _, _ = _setup_input_pc(bounds)
    np.random.seed(seed)
    germ = input_pc.sampleGerm(n_samples)
    X = input_pc.evalPC(germ)

    variants = _build_variants_from_samples(X, param_space._sim_data_parameters)
    config = _build_config(
        sim_data_path=sim_data_path,
        output_dir="<OUTPUT_DIR>",
        variants_section=variants,
        n_init_sims=n_init_sims,
        generations=generations,
        max_duration=max_duration,
    )

    text = _json.dumps(config, indent=2)
    if output_file:
        Path(output_file).write_text(text)
        console.print(f"[green]Config written to {output_file}[/green]")
    else:
        from rich.syntax import Syntax

        console.print(Syntax(text, "json", theme="monokai", line_numbers=True))

    console.print(f"\n[dim]{n_samples} variants × {n_init_sims} seeds × {generations} gens "
                  f"= {(n_samples + 1) * n_init_sims * generations} total sims[/dim]")


_HELP_SUBCOMMAND_ALIASES = {"help", "--help", "-h"}


def main() -> None:
    """Entry point.

    Rewrites ``uq <cmd> ... help`` → ``uq <cmd> ... --help`` so users can
    discover flags with a trailing ``help`` word on any subcommand
    (``uq sample help``, ``uq quantify help``, ``uq tui help``, …)
    in addition to the standard ``uq <cmd> --help`` form.
    """
    import sys

    argv = sys.argv
    if len(argv) >= 3 and argv[-1] in _HELP_SUBCOMMAND_ALIASES:
        argv[-1] = "--help"
    app()


if __name__ == "__main__":
    main()
