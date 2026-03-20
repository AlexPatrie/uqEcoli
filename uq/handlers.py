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

import os
from pathlib import Path

import dotenv
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq.common import get_repo_root
from uq.models import PipelineConfig
from uq.pce.models import PCEParameterSelectionConfig
from uq.pipe import Pipeline, execute_pipeline
from uq.pipeline import PipelineResult
from uq.sampling import PrecomputedCache

app = typer.Typer()
console = Console()

dotenv.load_dotenv(get_repo_root() / ".env")


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


def throw_env():
    raise OSError(
        "You must provide a value for sim_base_path either explicitly or as a env variable val (SIM_BASE_PATH)"
    )


def pipeline(config: PipelineConfig | None = None, execute: bool = True, **kwargs) -> Pipeline:
    """
    :param config:  (PipelineConfig | None)
    :param execute: (bool) = True
    :param kwargs:  (keyword-args) as follows:
        experiment_ids: str | list[str],
        sim_base_path: str | Path | None = None,
        observable_columns: list[str] | None = None,
        lb_generation: int | None = 2,
        lb_time: float | None = 100.0,
        n_bins: int = 10,
        polynomial_order: int = 3,
        n_samples: int = 200,
        expected_cycle_time: float = 3600.0,
        max_duration: float = 10800.0,
        prescreen_config: PCEParameterSelectionConfig | None = None,
        export_path: Path | None = None,
        precomputed_path: Path | str | None = None,
        sim_config_path: str | None = None,
        init: bool = True
    """
    if len(kwargs) and config is None:
        config = PipelineConfig(**kwargs)
    if config.sim_base_path is None:
        config.sim_base_path = os.getenv("SIM_BASE_PATH", throw_env())
    _pipe = Pipeline(
        experiment_ids=config.experiment_ids,
        sim_base_path=config.sim_base_path,
        observable_columns=config.observable_columns,
        lb_generation=config.lb_generation,
        lb_time=config.lb_time,
        n_bins=config.n_bins,
        polynomial_order=config.polynomial_order,
        n_samples=config.n_samples,
        expected_cycle_time=config.expected_cycle_time,
        max_duration=config.max_duration,
        prescreen_config=PCEParameterSelectionConfig(**config.prescreen_config.model_dump()),
        export_path=config.export_path,
        precomputed_path=config.precomputed_path,
        sim_config_path=config.sim_config_path,
        init=config.init,
    )
    if execute:
        _pipe.run()

    return _pipe


# def pipe(
#     experiment_ids: list[str],
#     outdir_root: str,
#     lb_generation: int | None = 2,
#     lb_time: float | None = 100.0,
#     n_bins: int = 10,
#     pce_polynomial_order: int = 3,
#     n_samples: int = 20,
#     expected_cycle_time: float = 3600.0,
#     pce_n_trajectories: int = 10,
#     pce_n_selected_params: int = 5,
#     export_path: str | None = None,
#     precomputed_path: str | None = None,
# ) -> PipelineResult:
#     """Run the full RFC006 UQ pipeline."""
#     param_prescreen_config = PCEParameterSelectionConfig(n_trajectories=pce_n_trajectories, n_top=pce_n_selected_params)
#     result: PipelineResult = execute_pipeline(
#         experiment_ids=experiment_ids,
#         sim_base_path=outdir_root,
#         prescreen_config=param_prescreen_config,
#         lb_time=lb_time,
#         lb_generation=lb_generation,
#         n_bins=n_bins,
#         polynomial_order=pce_polynomial_order,
#         n_samples=n_samples,
#         expected_cycle_time=expected_cycle_time,
#         export_path=Path(export_path) if export_path else None,
#         precomputed_path=Path(precomputed_path) if precomputed_path else None,
#     )
#     if export_path:
#         console.print(f"[bold green]Pipeline complete.[/bold green] Results exported to {export_path}")
#     else:
#         console.print("[bold green]Pipeline complete.[/bold green]")
#     return result


def demo(
    demo_type: str = "full",
    export_path: str | None = None,
    precomputed_path: str | None = None,
) -> PrecomputedCache | PipelineResult:
    # start with exp ids, outdir_base, observable cols if needed
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
        return result

    def demo_sample_generation():
        return generate_samples(experiment_ids, sim_base_path=base_path, cache_dir="uq_outputs", n_samples=20)

    return demo_sample_generation() if demo_type == "sample" else demo_full()


def generate_samples(
    experiment_ids: list[str],
    sim_base_path: str,
    cache_dir: str,
    n_samples: int = 200,
    seed: int = 42,
    observable_columns: list[str] | None = None,
    max_workers: int | None = None,
    max_duration: float = 10800.0,
    generations: int = 1,
    live: bool = False,
    include_vio: bool | None = None,
    include_mecillinam: bool | None = None,
    params_file: str | None = None,
) -> PrecomputedCache:
    """Stage 1: Generate LHS samples, evaluate simulation, cache (X, Y).

    Run this once to pre-compute simulation evaluations.  Then pass
    --precomputed-path to ``pipe`` or ``demo`` for Stage 2 analysis.

    By default uses ``DataDrivenWrapper`` (synthetic response surface
    built from the existing data's statistics).  Pass ``live=True``
    to run real vEcoli simulations as subprocesses.

    **Parameter selection:** By default, uses 5 physiologically relevant
    scalar sim_data parameters (see ``DEFAULT_SIM_DATA_PARAMETERS``).
    To specify custom parameters, pass ``--params-file`` pointing to a
    JSON file with a list of ``SimDataParameter`` specs.  To use the
    legacy vio/mecillinam parameters, pass ``--include-vio`` or
    ``--include-mecillinam``.

    Args:
        experiment_ids: Experiment IDs whose sim_data to load.
        sim_base_path: Root directory containing simulation outputs.
        cache_dir: Where to write the PrecomputedCache.
        n_samples: Number of LHS samples.
        seed: Random seed for LHS generation.
        observable_columns: Which output columns to extract.
        max_workers: Max parallel subprocesses. None = sequential.
        max_duration: Simulation wall-clock limit in seconds (live mode).
        generations: Number of generations per sim (live mode).
        live: If True, run real vEcoli simulations as subprocesses.
        include_vio: Include vio pathway parameters (legacy mode).
        include_mecillinam: Include mecillinam concentration (legacy mode).
        params_file: Path to a JSON file with a list of
            ``SimDataParameter`` specs for custom parameter selection.
    """
    import json as _json

    from uq.pipe import initialize_data
    from uq.pipeline.models import SimDataParameter
    from uq.pipeline.workflow import aggregate_timeseries
    from uq.sampling import run_and_cache
    from uq.wrappers import DataDrivenWrapper

    # Default observables to avoid DuckDB OOM on wide tables
    if observable_columns is None:
        observable_columns = [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__volume",
            "listeners__mass__growth",
        ]

    ds = initialize_data(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        observable_columns=observable_columns,
    )

    if not ds.x:
        raise RuntimeError(
            f"No ParameterDataset loaded. Ensure simData.cPickle exists under {sim_base_path}/*/parca/kb/"
        )

    # Load custom parameter specs from JSON file if provided
    sim_data_parameters = None
    if params_file is not None:
        raw = _json.loads(Path(params_file).read_text())
        sim_data_parameters = [SimDataParameter.from_dict(p) for p in raw]

    # Build parameter space
    param_space = ds.x[0].to_parameter_space(
        parameters=sim_data_parameters,
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )
    if param_space.n_parameters == 0:
        raise RuntimeError(
            "Parameter space is empty. Provide --params-file, "
            "--include-vio, or --include-mecillinam."
        )

    if live:
        # Real vEcoli simulation via subprocesses (no in-process EcoliSim)
        from uq.generators.vecoli import TimeseriesGeneratorVecoli
        from uq.sampling import run_batch_and_cache

        sim_func = TimeseriesGeneratorVecoli(
            baseline_sim_data=ds.x[0].sim_data,
            param_space=param_space,
            max_duration=max_duration,
            output_keys=[c.split("__")[-1] for c in observable_columns],
        )
        console.print(
            f"[bold cyan]Live mode:[/bold cyan] subprocess execution "
            f"(max_duration={max_duration:.0f}s, generations={generations}, "
            f"params={param_space.parameter_names})"
        )

        cache = run_batch_and_cache(
            parameter_space=param_space,
            simulation_func=sim_func,
            n_samples=n_samples,
            cache_dir=Path(cache_dir),
            seed=seed,
            max_workers=max_workers,
        )
    else:
        # Synthetic response surface built from existing data statistics
        agg = aggregate_timeseries(ds.y, ds.observables)
        sim_func = DataDrivenWrapper(
            parameter_space=param_space,
            observable_means=agg.uniform.mean,
            observable_stds=agg.uniform.std,
        )
        console.print("[bold yellow]Synthetic mode:[/bold yellow] DataDrivenWrapper")

        cache = run_and_cache(
            parameter_space=param_space,
            simulation_func=sim_func,
            n_samples=n_samples,
            cache_dir=Path(cache_dir),
            seed=seed,
            max_workers=max_workers,
        )

    console.print(
        f"[bold green]Cached {cache.X.shape[0]} samples[/bold green] "
        f"({cache.X.shape[1]} params, {cache.Y.shape[1]} outputs) "
        f"to {cache_dir}"
    )
    if cache.Y_timeseries is not None:
        console.print(f"  Timeseries cached: {len(cache.Y_timeseries)} samples")
    return cache


def export_configs(
    sim_data_path: str,
    batch_dir: str,
    n_samples: int = 200,
    seed: int = 42,
    include_vio: bool = True,
    include_mecillinam: bool = True,
    base_config_path: str | None = None,
    generations: int = 1,
    emitter: str = "parquet",
) -> Path:
    """Export per-sample vEcoli configs for Nextflow/HPC batch execution.

    This is the HPC-scale alternative to ``generate_samples()``:
    instead of running simulations locally, it writes variant-applied
    sim_data pickles + JSON configs that Nextflow can consume.

    Returns:
        Path to the batch directory.
    """
    from uq.generators.vecoli import TimeseriesGeneratorVecoli, export_batch_configs
    from uq.pipeline.param_loader import ParameterDataset
    from uq.sampling import generate_lhs_samples

    ds = ParameterDataset(sim_data_path=sim_data_path)
    param_space = ds.to_parameter_space(
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )

    if param_space.n_parameters == 0:
        raise ValueError("Parameter space is empty. Set include_vio=True and/or include_mecillinam=True.")

    sim_func = TimeseriesGeneratorVecoli(
        baseline_sim_data=ds.sim_data,
        param_space=param_space,
    )

    X = generate_lhs_samples(param_space, n_samples, seed=seed)

    result_dir = export_batch_configs(
        sim_func=sim_func,
        X=X,
        batch_dir=batch_dir,
        base_config_path=base_config_path,
        generations=generations,
        emitter=emitter,
    )

    console.print(
        f"[bold green]Exported {n_samples} batch configs[/bold green] "
        f"({param_space.n_parameters} params) to {result_dir}"
    )
    console.print("  configs/  → per-sample JSON configs for Nextflow")
    console.print("  sim_data/ → variant-applied pickled sim_data")
    console.print("  metadata.json → sample-to-parameter mapping")
    return result_dir


def collect_results(
    batch_dir: str,
    output_dir: str,
    observable_columns: list[str] | None = None,
    cache_dir: str | None = None,
) -> PrecomputedCache:
    """Collect Parquet outputs from completed Nextflow/HPC batch into cache.

    After Nextflow completes, call this to assemble (X, Y) from the
    per-sample outputs.  The resulting PrecomputedCache can be passed
    to ``quantify --precomputed-path``.

    Args:
        batch_dir: Directory produced by ``export_configs()``.
        output_dir: Root directory containing per-sample Parquet outputs.
        observable_columns: Which columns to extract from Parquet.
        cache_dir: Where to save the cache (default: ``{batch_dir}/cache``).

    Returns:
        PrecomputedCache ready for Stage 2 analysis.
    """
    from uq.generators.vecoli import collect_batch_results

    cache = collect_batch_results(
        batch_dir=batch_dir,
        output_dir=output_dir,
        observable_columns=observable_columns,
        cache_dir=cache_dir,
    )

    console.print(
        f"[bold green]Collected {cache.X.shape[0]} samples[/bold green] "
        f"({cache.X.shape[1]} params, {cache.Y.shape[1]} outputs) "
        f"from {output_dir}"
    )
    console.print(f"  Cache saved to: {cache.cache_dir}")
    return cache


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
