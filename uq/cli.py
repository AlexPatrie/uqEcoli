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
    base_config: str | None = typer.Option(
        None,
        help="Base vEcoli config JSON to merge with. Preserves parca_variants, "
        "analysis_options, and other multi-parca keys.",
    ),
    conditions: list[str] = typer.Option(
        [],
        help="RNA-seq dataset IDs for multi-condition UQ (one per --conditions). "
        "Populates parca_variants for cross-condition sensitivity analysis. "
        "Requires vEcoli multi-parca-aws branch.",
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
        base_config_path=base_config,
        conditions=conditions if conditions else None,
    )
    config_path = batch_dir / "workflow_config.json"
    config_path.write_text(_json.dumps(config, indent=2))
    console.print(f"  [dim]Config JSON: {config_path}[/dim]")
    if conditions:
        console.print(f"  [dim]Multi-condition: {len(conditions)} parca_variants[/dim]")

    # Save conditions metadata for quantify to detect multi-condition cache
    if conditions:
        cond_meta = {"conditions": conditions, "n_samples": n_samples, "n_test": n_test}
        (cache_path / "conditions.json").write_text(_json.dumps(cond_meta, indent=2))

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
    from uq.multi_condition import is_multi_condition_cache, quantify_multi_condition
    from uq.workflow import quantify as wf_quantify

    console.print(
        f"[bold cyan]Quantify:[/bold cyan] order={polynomial_order}, bins={n_bins}, regression={regression}, tol={tol}"
    )

    # Auto-detect multi-condition cache
    if is_multi_condition_cache(cache_dir):
        console.print("[bold magenta]Multi-condition cache detected — running cross-condition GSA (extension beyond RFC006)[/bold magenta]")
        mc_result = quantify_multi_condition(
            cache_dir=cache_dir,
            sim_data_path=sim_data_path,
            polynomial_order=polynomial_order,
            n_bins=n_bins,
            regression=regression,
            tolerance=tol,
            export_path=export_path,
        )
        for cond_id, cond_result in mc_result.per_condition.items():
            console.print(f"\n[bold cyan]── Condition: {cond_id} ──[/bold cyan]")
            _print_report(cond_result)
        _print_multi_condition_report(mc_result)
        console.print(f"\n  [bold green]Artifacts exported to:[/bold green] {export_path}")
        return

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
    _print_narrative(result)
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


def _print_narrative(result: Any) -> None:
    """Print a plain-English 'Key Findings' summary after the Sobol tables."""
    names = result.parameter_names
    s1_total = result.strategy1.sobol.total_order
    if s1_total.ndim > 1:
        s1_total = np.mean(s1_total, axis=0)

    # Top parameter (bulk)
    top_idx = int(np.argmax(s1_total))
    top_name = names[top_idx]
    top_pct = s1_total[top_idx] * 100

    # Negligible parameter (lowest S_Ti across all strategies)
    min_idx = int(np.argmin(s1_total))
    min_name = names[min_idx]
    min_pct = s1_total[min_idx] * 100

    # Most sensitive growth stage (strategy 4)
    stage_line = ""
    if result.strategy4_per_stage:
        stage_sums = []
        for sr in result.strategy4_per_stage:
            st = sr.sobol.total_order
            if st.ndim > 1:
                st = np.mean(st, axis=0)
            stage_sums.append(float(np.sum(st)))
        peak_stage = int(np.argmax(stage_sums))
        n = len(result.strategy4_per_stage)
        lo_pct = int(100 * peak_stage / n)
        hi_pct = int(100 * (peak_stage + 1) / n)
        stage_line = f"  • Growth stage θ {lo_pct}–{hi_pct}% shows the strongest parameter sensitivity."

    # Recommendation
    if top_pct > 50:
        rec = f"[bold]{top_name}[/bold] dominates output variance — prioritize its measurement accuracy."
    elif top_pct > 20:
        rec = f"[bold]{top_name}[/bold] is the leading driver — consider tighter bounds or dedicated experiments."
    else:
        rec = "No single parameter dominates — output variance is distributed across multiple inputs."

    lines = [
        f"  • [bold]{top_name}[/bold] explains [bold]{top_pct:.1f}%[/bold] of population-level output variance (S_Ti).",
        f"  • [bold]{min_name}[/bold] is least influential at [bold]{min_pct:.1f}%[/bold] — candidate for fixing at nominal.",
    ]
    if stage_line:
        lines.append(stage_line)
    lines.append(f"  • Recommendation: {rec}")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold white on blue] KEY FINDINGS [/bold white on blue]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _print_multi_condition_report(mc_result: Any) -> None:
    """Print cross-condition comparison table and narrative."""
    from uq.multi_condition import MultiConditionResult

    names = mc_result.parameter_names
    conditions = mc_result.conditions

    # ── Cross-condition S_Ti comparison table ──
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="CROSS-CONDITION SENSITIVITY COMPARISON",
        title_style="bold magenta",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    for cond_id in conditions:
        table.add_column(cond_id, justify="right")
    table.add_column("STABILITY", justify="right")

    stability = mc_result.rank_stability
    for i, name in enumerate(names):
        row = [name]
        for cond_id in conditions:
            s_ti = mc_result.per_condition[cond_id].strategy1.sobol.total_order
            if s_ti.ndim > 1:
                s_ti = np.mean(s_ti, axis=0)
            row.append(_pct(s_ti[i]))
        # Stability indicator
        n_dots = max(1, int(stability[i] * 5))
        dots = "●" * n_dots + "○" * (5 - n_dots)
        row.append(f"{dots} ({stability[i]:.2f})")
        table.add_row(*row)

    console.print()
    console.print(
        Panel(
            table,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    # ── Cross-condition narrative ──
    lines = []
    if mc_result.universal_drivers:
        drivers = ", ".join(f"[bold]{d}[/bold]" for d in mc_result.universal_drivers)
        lines.append(f"  • Universal drivers (S_Ti > 10% in all conditions): {drivers}")
        lines.append("    These should be measured precisely regardless of growth condition.")

    for cond_id, params in mc_result.condition_specific.items():
        param_str = ", ".join(f"[bold]{p}[/bold]" for p in params)
        lines.append(f"  • Condition-specific to [cyan]{cond_id}[/cyan]: {param_str}")

    if not lines:
        lines.append("  • No clear universal or condition-specific patterns detected.")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold white on magenta] CROSS-CONDITION FINDINGS [/bold white on magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
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
    --run-mode web  Standalone web app (Dash) — shareable URL

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
    elif run_mode == "web":
        from app.web_dashboard import run_web_dashboard

        # Pass the directory, not the JSON file
        rp = results_path
        if rp and rp.endswith(".json"):
            rp = str(Path(rp).parent)
        run_web_dashboard(results_path=rp)
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


@app.command()
def compare(
    dirs: list[str] = typer.Argument(..., help="Two or more uq export directories to compare"),
) -> None:
    """Compare Sobol indices across multiple UQ experiments.

    \b
    Loads uq_results.json from each directory and prints side-by-side
    Sobol tables with differential highlighting.

    Example:
      uq compare ./results_baseline ./results_perturbed
    """
    import json as _json

    if len(dirs) < 2:
        console.print("[red]Need at least 2 directories to compare.[/red]")
        raise typer.Exit(1)

    results = []
    for d in dirs:
        p = Path(d) / "uq_results.json"
        if not p.exists():
            console.print(f"[red]Not found: {p}[/red]")
            raise typer.Exit(1)
        results.append((_json.loads(p.read_text()), Path(d).name))

    # Gather all param names (union)
    all_params = list(results[0][0]["parameters"].keys())

    # Side-by-side S_Ti table
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title="S_Ti COMPARISON (Population)",
        title_style="bold cyan",
    )
    table.add_column("PARAMETER", style="bold yellow", no_wrap=True)
    for _, name in results:
        table.add_column(name, justify="right")
    if len(results) == 2:
        table.add_column("Δ", justify="right", style="bold")

    for p in all_params:
        row = [p]
        vals = []
        for data, _ in results:
            v = data["phase1_population"]["sobol_total_order"].get(p, 0)
            vals.append(v)
            row.append(f"{v:.3f}")
        if len(results) == 2:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            color = "green" if abs(delta) < 0.05 else ("red" if delta > 0 else "blue")
            row.append(f"[{color}]{sign}{delta:.3f}[/{color}]")
        table.add_row(*row)

    console.print(Panel(table, border_style="cyan", box=box.ROUNDED, padding=(0, 1)))

    # Per-strategy summary
    for i, (data, name) in enumerate(results):
        s2 = data.get("strategy2_by_generation", {})
        s3 = data.get("strategy3_by_seed", {})
        s4 = data.get("phase2_growth_stratified", {})
        console.print(
            f"  [dim]{name}: S2={s2.get('n_generations', 0)} gens, "
            f"S3={s3.get('n_seeds', 0)} seeds, "
            f"S4={s4.get('n_stages', 0)} stages[/dim]"
        )


@app.command(name="export-figures")
def export_figures(
    results_path: str = typer.Option("./uq_results", help="Path to uq export directory"),
    output_dir: str | None = typer.Option(None, help="Output directory (default: <results>/figures/)"),
) -> None:
    """Generate publication-ready PDFs and LaTeX from UQ results.

    \b
    Outputs:
      sobol_bar_chart.pdf    Grouped bars (S_Ti per strategy, all params)
      spectrogram.pdf        Print-quality sensitivity heatmap
      response_curves.pdf    PCE response curves at midpoint
      sobol_table.tex        LaTeX table of top-K params per strategy

    Requires kaleido: uv pip install kaleido
    """
    from uq.viz_export import export_all_figures

    console.print(f"[bold cyan]Exporting figures from:[/bold cyan] {results_path}")
    generated = export_all_figures(results_path, output_dir)
    for p in generated:
        console.print(f"  [green]✓[/green] {p}")
    console.print(f"\n[bold green]{len(generated)} figures exported.[/bold green]")


@app.command(name="suggest-experiment")
def suggest_experiment(
    results_path: str = typer.Option("./uq_results", help="Path to uq export directory"),
    n_grid: int = typer.Option(1000, help="Grid points for variance scanning"),
) -> None:
    """Identify where to measure next to reduce uncertainty most.

    \b
    Loads the fitted PCE surrogate, evaluates prediction variance
    across the parameter space, and reports the region of maximum
    uncertainty in physical units.
    """
    import json as _json

    results_dir = Path(results_path)
    pop_dir = results_dir / "population_surrogate"
    if not (pop_dir / "coefficients.npy").exists():
        console.print("[red]Surrogate files not found. Run `uq quantify` first.[/red]")
        raise typer.Exit(1)

    coeffs = np.load(pop_dir / "coefficients.npy")
    mi = np.load(pop_dir / "multi_indices.npy")
    bounds = np.load(pop_dir / "input_bounds.npy")

    data = _json.loads((results_dir / "uq_results.json").read_text())
    params = list(data["parameters"].keys())
    n_params = len(params)

    # Grid-sample parameter space, estimate prediction variance
    # via PCE coefficient variance approximation
    rng = np.random.default_rng(42)
    X_grid = bounds[:, 0] + rng.random((n_grid, n_params)) * (bounds[:, 1] - bounds[:, 0])

    # Evaluate PCE at each grid point — variance proxy: |Ŷ - Ŷ_midpoint|
    mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
    mid_norm = 2.0 * (mid - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0

    def _legendre_eval_vec(x_phys):
        x_norm = 2.0 * (x_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-12) - 1.0
        max_ord = int(mi.max()) if mi.size else 0
        P = np.zeros((max_ord + 1, n_params))
        P[0, :] = 1.0
        if max_ord >= 1:
            P[1, :] = x_norm
        for n in range(2, max_ord + 1):
            P[n, :] = ((2 * n - 1) * x_norm * P[n - 1, :] - (n - 1) * P[n - 2, :]) / n
        result = 0.0
        for t in range(len(coeffs)):
            term = float(coeffs[t])
            for p in range(n_params):
                term *= P[mi[t, p], p]
            result += term
        return result

    y_mid = _legendre_eval_vec(mid)
    variances = np.zeros(n_grid)
    for i in range(n_grid):
        y_i = _legendre_eval_vec(X_grid[i])
        variances[i] = (y_i - y_mid) ** 2

    # Find max-variance point
    best_idx = int(np.argmax(variances))
    best_x = X_grid[best_idx]
    best_var = variances[best_idx]

    console.print("\n[bold cyan]SUGGESTED NEXT EXPERIMENT[/bold cyan]")
    console.print(f"  [dim]Scanned {n_grid} parameter combinations[/dim]\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("PARAMETER", style="bold yellow")
    table.add_column("VALUE", justify="right")
    table.add_column("RANGE", justify="right", style="dim")

    for i, p in enumerate(params):
        table.add_row(
            p,
            f"{best_x[i]:.4f}",
            f"[{bounds[i, 0]:.4f}, {bounds[i, 1]:.4f}]",
        )

    console.print(Panel(
        table,
        title="[bold]Max-uncertainty parameter setting[/bold]",
        subtitle=f"[dim]Prediction variance: {best_var:.3e}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # Which param contributes most to variance at this point?
    sensitivities = []
    for pi in range(n_params):
        delta = (bounds[pi, 1] - bounds[pi, 0]) * 0.01
        x_plus = best_x.copy()
        x_plus[pi] = min(best_x[pi] + delta, bounds[pi, 1])
        x_minus = best_x.copy()
        x_minus[pi] = max(best_x[pi] - delta, bounds[pi, 0])
        sens = abs(_legendre_eval_vec(x_plus) - _legendre_eval_vec(x_minus)) / (2 * delta + 1e-12)
        sensitivities.append((params[pi], sens))
    sensitivities.sort(key=lambda x: -x[1])
    top_param = sensitivities[0][0]
    console.print(
        f"\n  [bold]Recommendation:[/bold] Measure [bold]{top_param}[/bold] more precisely "
        f"in the range [{bounds[params.index(top_param), 0]:.4f}, "
        f"{bounds[params.index(top_param), 1]:.4f}] to reduce uncertainty most."
    )


@app.command(name="init")
def init_project() -> None:
    """Guided setup wizard — configure a UQ project interactively.

    \b
    Detects vEcoli, validates simData, chooses observable presets,
    and writes uq_config.json for use with `uq sample`.
    """
    import json as _json

    console.print("[bold cyan]UQ Project Setup Wizard[/bold cyan]\n")

    # Step 1: Detect vEcoli simData
    sim_data_path = None
    search_paths = [
        Path("../vEcoli/reconstruction/sim_data/kb/simData.cPickle"),
        Path("sim_data/baseline/kb/simData.cPickle"),
        Path.home() / ".local/share/vEcoli/simData.cPickle",
    ]
    for sp in search_paths:
        if sp.resolve().exists():
            sim_data_path = str(sp.resolve())
            console.print(f"  [green]Auto-detected simData:[/green] {sim_data_path}")
            break

    if sim_data_path is None:
        sim_data_path = typer.prompt("Path to simData.cPickle")
    else:
        override = typer.prompt(f"simData path [{sim_data_path}]", default=sim_data_path)
        sim_data_path = override

    if not Path(sim_data_path).exists():
        console.print(f"[red]Not found: {sim_data_path}[/red]")
        raise typer.Exit(1)

    # Step 2: Validate vEcoli importable
    try:
        import ecoli  # type: ignore[import-not-found]  # noqa: F401
        console.print("  [green]vEcoli: importable[/green]")
    except ImportError:
        console.print("  [yellow]vEcoli not importable — sampling will fail.[/yellow]")

    # Step 3: Observable preset
    console.print("\n[bold]Observable presets:[/bold]")
    presets = {
        "mass": "Raw mass/growth scalars (5 features, fastest)",
        "higher_order": "Derived cd1 metrics: doubling time, growth rate, compositions (6 features)",
        "exchange_fluxes": "External metabolite fluxes (~87 features)",
        "transcriptome": "mRNA cistron counts (~4300 genes)",
        "proteome": "Protein monomer counts (~4300 monomers)",
        "fluxome": "Base reaction fluxes (~2800, dry-mass normalized)",
    }
    for k, v in presets.items():
        console.print(f"  [cyan]{k:<20s}[/cyan] {v}")
    obs_input = typer.prompt("\nObservable presets (comma-separated)", default="mass")
    obs_list = [o.strip() for o in obs_input.split(",") if o.strip()]

    # Step 4: Sample count
    console.print("\n[bold]Sample count:[/bold]")
    console.print("  20  — quick exploration (~2h)")
    console.print("  50  — production quality (~5h)")
    console.print("  200 — high-fidelity (~20h)")
    n_samples = int(typer.prompt("Number of samples", default="20"))

    # Step 5: Regression method
    reg = typer.prompt("Regression method (lsq/bcs/anl)", default="lsq")

    # Step 6: Write config
    config = {
        "sim_data_path": sim_data_path,
        "cache_dir": "./uq_cache",
        "export_path": "./uq_results",
        "n_samples": n_samples,
        "observables": obs_list,
        "polynomial_order": 2,
        "regression": reg,
        "n_bins": 10,
        "generations": 1,
        "n_init_sims": 1,
        "seed": 42,
    }

    config_path = Path("uq_config.json")
    config_path.write_text(_json.dumps(config, indent=2))
    console.print(f"\n[bold green]Config written to:[/bold green] {config_path}")
    console.print(
        f"\n[bold]Ready![/bold] Run:\n"
        f"  [cyan]uv run uq sample {sim_data_path} "
        f"--cache-dir ./uq_cache --n-samples {n_samples} "
        f"--observables {' --observables '.join(obs_list)}[/cyan]\n"
        f"  [cyan]uv run uq quantify {sim_data_path} "
        f"--cache-dir ./uq_cache --export-path ./uq_results "
        f"--regression {reg}[/cyan]"
    )


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
