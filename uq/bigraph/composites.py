"""
Pre-wired Composite builders for common UQ workflows.

Three composites corresponding to the three main usage patterns:

1. **Full pipeline** (``build_full_composite``) — sample → quantify →
   export.  Mirrors ``uv run uq sample ... && uv run uq quantify ...``
   but with real-time progress emission from the simulation Process.

2. **Quantify-only** (``build_quantify_composite``) — loads an
   existing PrecomputedCache and runs all 4 strategies.  Mirrors
   ``uv run uq quantify ...``.

3. **Explorer** (``build_explorer_composite``) — loads exported
   surrogate artifacts and provides interactive PCE evaluation.
   Replaces the duplicated ``legendre_eval()`` in each dashboard.
   Mirrors the reactive loop inside ``uv run uq dashboard``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bigraph_schema import allocate_core
from process_bigraph import Composite
from process_bigraph.emitter import RAMEmitter

from uq.bigraph.processes import (
    CollectCache,
    Export,
    PCEEvaluate,
    Quantify,
    RunSimulations,
    SetupInputs,
    StrategyFit,
)


def get_core(**extra_types: type) -> Any:
    """Allocate a Core with all UQ bigraph types registered."""
    top = {
        "SetupInputs": SetupInputs,
        "RunSimulations": RunSimulations,
        "CollectCache": CollectCache,
        "Quantify": Quantify,
        "Export": Export,
        "PCEEvaluate": PCEEvaluate,
        "StrategyFit": StrategyFit,
        "RAMEmitter": RAMEmitter,
        **extra_types,
    }
    return allocate_core(top=top)


# ═══════════════════════════════════════════════════════════════════
#  Full pipeline: sample → quantify → export
# ═══════════════════════════════════════════════════════════════════


def build_full_composite(
    sim_data_path: str,
    cache_dir: str = "./uq_cache",
    export_path: str = "./uq_results",
    n_samples: int = 20,
    seed: int = 42,
    generations: int = 1,
    n_init_sims: int = 1,
    polynomial_order: int = 2,
    n_bins: int = 10,
    regression: str = "lsq",
    max_duration: float = 10800.0,
    params_file: str = "",
    sim_interval: float = 2.0,
    with_emitter: bool = True,
    core: Any | None = None,
) -> Composite:
    """Build a Composite for the full sample → quantify → export pipeline.

    The simulation step is a **Process** (time-driven) that emits
    ``n_completed`` and ``total_sims`` at each interval tick, enabling
    real-time progress tracking in any UI.

    Usage::

        from uq.bigraph import build_full_composite

        composite = build_full_composite(
            sim_data_path="/path/to/simData.cPickle",
            n_samples=20,
        )

        # Run — the simulation Process ticks at sim_interval
        composite.run(sim_interval)

        # Progress is available via composite.state
        print(composite.state["n_completed"], "/", composite.state["total_sims"])
        print(composite.state["pipeline_complete"])

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Cache directory.
        export_path: Export directory for artifacts.
        n_samples: Number of PCRV samples.
        seed: Random seed.
        generations: Cell generations per simulation.
        n_init_sims: Initial seeds per variant.
        polynomial_order: PCE polynomial order.
        n_bins: Growth-progress bins (Strategy 4).
        regression: Regression method ('lsq', 'bcs', 'anl').
        max_duration: vEcoli wall-clock limit (seconds).
        params_file: Optional JSON with SimDataParameter specs.
        sim_interval: Composite interval for simulation polling (seconds).
        with_emitter: Attach RAMEmitter for progress observation.
        core: Optional pre-allocated Core.

    Returns:
        Ready-to-run Composite.
    """
    if core is None:
        core = get_core()

    process_config = {"params_file": params_file} if params_file else {}
    sim_config = {**process_config, "max_duration": max_duration}

    state: dict[str, Any] = {
        # Pipeline parameters
        "sim_data_path": str(Path(sim_data_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "export_path": str(Path(export_path).resolve()),
        "n_samples": n_samples,
        "seed": seed,
        "generations": generations,
        "n_init_sims": n_init_sims,
        "polynomial_order": polynomial_order,
        "n_bins": n_bins,
        "regression": regression,
        # Intermediate stores
        "X_train": "",
        "germ_train": "",
        "parameter_names": "",
        "bounds": "",
        "n_parameters": 0,
        "history_base": "",
        "return_code": 0,
        "n_completed": 0,
        "total_sims": 0,
        "sim_stdout": "",
        "cache_ready": False,
        "n_cached_samples": 0,
        "quantify_complete": False,
        "results_json": "",
        "pipeline_complete": False,
        "summary": "",
        # Step 1: SetupInputs (fires immediately)
        "setup_inputs": {
            "_type": "step",
            "address": "local:SetupInputs",
            "config": process_config,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "n_samples": ["n_samples"],
                "seed": ["seed"],
            },
            "outputs": {
                "X_train": ["X_train"],
                "germ_train": ["germ_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_parameters": ["n_parameters"],
            },
        },
        # Step 2: RunSimulations (Process — polls at each tick)
        "run_simulations": {
            "_type": "process",
            "address": "local:RunSimulations",
            "config": sim_config,
            "interval": sim_interval,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "cache_dir": ["cache_dir"],
                "X_train": ["X_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_samples": ["n_samples"],
                "generations": ["generations"],
                "n_init_sims": ["n_init_sims"],
            },
            "outputs": {
                "history_base": ["history_base"],
                "return_code": ["return_code"],
                "n_completed": ["n_completed"],
                "total_sims": ["total_sims"],
                "sim_stdout": ["sim_stdout"],
            },
        },
        # Step 3: CollectCache
        "collect_cache": {
            "_type": "step",
            "address": "local:CollectCache",
            "config": {},
            "inputs": {
                "history_base": ["history_base"],
                "cache_dir": ["cache_dir"],
                "X_train": ["X_train"],
                "germ_train": ["germ_train"],
                "parameter_names": ["parameter_names"],
                "bounds": ["bounds"],
                "n_samples": ["n_samples"],
                "seed": ["seed"],
            },
            "outputs": {
                "cache_ready": ["cache_ready"],
                "n_cached_samples": ["n_cached_samples"],
            },
        },
        # Step 4: Quantify
        "quantify": {
            "_type": "step",
            "address": "local:Quantify",
            "config": process_config,
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "cache_dir": ["cache_dir"],
                "export_path": ["export_path"],
                "cache_ready": ["cache_ready"],
                "polynomial_order": ["polynomial_order"],
                "n_bins": ["n_bins"],
                "regression": ["regression"],
            },
            "outputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
            },
        },
        # Step 5: Export
        "export": {
            "_type": "step",
            "address": "local:Export",
            "config": {"log_report": True},
            "inputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
                "export_path": ["export_path"],
            },
            "outputs": {
                "pipeline_complete": ["pipeline_complete"],
                "summary": ["summary"],
            },
        },
    }

    if with_emitter:
        state["emitter"] = {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "n_completed": "integer",
                    "total_sims": "integer",
                    "pipeline_complete": "boolean",
                    "summary": "string",
                },
            },
            "inputs": {
                "n_completed": ["n_completed"],
                "total_sims": ["total_sims"],
                "pipeline_complete": ["pipeline_complete"],
                "summary": ["summary"],
            },
            "outputs": {},
        }

    return Composite({"state": state}, core=core)


# ═══════════════════════════════════════════════════════════════════
#  Quantify-only: existing cache → all 4 strategies → export
# ═══════════════════════════════════════════════════════════════════


def build_quantify_composite(
    sim_data_path: str,
    cache_dir: str = "./uq_cache",
    export_path: str = "./uq_results",
    polynomial_order: int = 2,
    n_bins: int = 10,
    regression: str = "lsq",
    with_emitter: bool = True,
    core: Any | None = None,
) -> Composite:
    """Build a Composite that runs quantify + export from existing cache.

    Equivalent to ``uv run uq quantify ...``.

    Usage::

        from uq.bigraph import build_quantify_composite

        composite = build_quantify_composite(
            sim_data_path="/path/to/simData.cPickle",
            cache_dir="./uq_cache",
        )
        composite.run(1.0)
        print(composite.state["pipeline_complete"])

    Args:
        sim_data_path: Path to simData.cPickle.
        cache_dir: Existing cache directory.
        export_path: Export directory for artifacts.
        polynomial_order: PCE polynomial order.
        n_bins: Growth-progress bins.
        regression: Regression method.
        with_emitter: Attach RAMEmitter.
        core: Optional pre-allocated Core.
    """
    if core is None:
        core = get_core()

    state: dict[str, Any] = {
        "sim_data_path": str(Path(sim_data_path).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "export_path": str(Path(export_path).resolve()),
        "cache_ready": True,
        "polynomial_order": polynomial_order,
        "n_bins": n_bins,
        "regression": regression,
        "quantify_complete": False,
        "results_json": "",
        "pipeline_complete": False,
        "summary": "",
        "quantify": {
            "_type": "step",
            "address": "local:Quantify",
            "config": {},
            "inputs": {
                "sim_data_path": ["sim_data_path"],
                "cache_dir": ["cache_dir"],
                "export_path": ["export_path"],
                "cache_ready": ["cache_ready"],
                "polynomial_order": ["polynomial_order"],
                "n_bins": ["n_bins"],
                "regression": ["regression"],
            },
            "outputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
            },
        },
        "export": {
            "_type": "step",
            "address": "local:Export",
            "config": {"log_report": True},
            "inputs": {
                "quantify_complete": ["quantify_complete"],
                "results_json": ["results_json"],
                "export_path": ["export_path"],
            },
            "outputs": {
                "pipeline_complete": ["pipeline_complete"],
                "summary": ["summary"],
            },
        },
    }

    if with_emitter:
        state["emitter"] = {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {"pipeline_complete": "boolean", "summary": "string"}},
            "inputs": {"pipeline_complete": ["pipeline_complete"], "summary": ["summary"]},
            "outputs": {},
        }

    return Composite({"state": state}, core=core)


# ═══════════════════════════════════════════════════════════════════
#  Explorer: load surrogate artifacts → interactive PCE evaluation
# ═══════════════════════════════════════════════════════════════════


def build_explorer_composite(
    export_path: str = "./uq_results",
    x_physical: list[float] | None = None,
    core: Any | None = None,
) -> Composite:
    """Build a Composite for interactive PCE surrogate exploration.

    Loads exported artifacts and evaluates the PCE at the given
    parameter vector.  Produces sweep curves and local sensitivity
    for dashboard visualization.

    This replaces the duplicated ``legendre_eval()`` code across
    ``app/dashboard_simple.py``, ``app/uq_daw_simple.py``.

    Usage::

        import json, numpy as np
        from uq.bigraph import build_explorer_composite

        composite = build_explorer_composite(
            export_path="./uq_results",
            x_physical=[0.48, 0.15, 2.5e-7, 0.002, 0.30],
        )
        composite.run(1.0)

        y_hat = composite.state["y_hat"]
        curves = json.loads(composite.state["sweep_curves"])
        sens = json.loads(composite.state["local_sensitivity"])

    Args:
        export_path: Directory with uq_results.json + population_surrogate/.
        x_physical: Parameter values in physical space.  If None,
            uses midpoints of bounds.
        core: Optional pre-allocated Core.
    """
    if core is None:
        core = get_core()

    import json

    export_dir = Path(export_path).resolve()

    # Auto-detect midpoints if no x given
    if x_physical is None:
        import numpy as np

        bounds_path = export_dir / "population_surrogate" / "input_bounds.npy"
        if bounds_path.exists():
            bounds = np.load(bounds_path)
            x_physical = ((bounds[:, 0] + bounds[:, 1]) / 2).tolist()
        else:
            x_physical = []

    state: dict[str, Any] = {
        "export_path": str(export_dir),
        "x_physical": json.dumps(x_physical),
        "y_hat": 0.0,
        "sweep_curves": "{}",
        "local_sensitivity": "{}",
        "pce_eval": {
            "_type": "step",
            "address": "local:PCEEvaluate",
            "config": {"n_sweep": 80},
            "inputs": {
                "export_path": ["export_path"],
                "x_physical": ["x_physical"],
            },
            "outputs": {
                "y_hat": ["y_hat"],
                "sweep_curves": ["sweep_curves"],
                "local_sensitivity": ["local_sensitivity"],
            },
        },
    }

    return Composite({"state": state}, core=core)
