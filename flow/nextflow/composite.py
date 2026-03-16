"""
Composite document construction for the Nextflow-backed UQ pipeline.

Provides:
    build_composite  — constructs a process_bigraph.Composite wired so that
        shared pipeline state (experiment_id, sim_base_path, parameter space
        bounds, etc.) is read by a NextflowUQProcess, which writes the
        pipeline_result back into the composite state.

    COMPOSITE_SCHEMA / COMPOSITE_STATE — the raw schema and default state
        dicts, exposed for programmatic manipulation before handing to
        Composite().

Usage:
    >>> from flow.nextflow.composite import build_composite
    >>> sim = build_composite(
    ...     experiment_id="mecillinam",
    ...     sim_base_path="/data/sims",
    ...     output_dir="./results",
    ... )
    >>> sim.run(1.0)          # single interval → launches Nextflow
    >>> print(sim.state)      # pipeline_result populated
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bigraph_schema import allocate_core
from process_bigraph import Composite
from process_bigraph.emitter import RAMEmitter

from flow.nextflow.processes import NextflowProcess, NextflowRunStep, NextflowUQProcess


def get_core(**extra_types: type):
    """Allocate a Core with the Nextflow types registered."""
    top = {
        "NextflowProcess": NextflowProcess,
        "NextflowUQProcess": NextflowUQProcess,
        "NextflowRunStep": NextflowRunStep,
        "RAMEmitter": RAMEmitter,
        **extra_types,
    }
    return allocate_core(top=top)


# ─── Schema & State Templates ──────────────────────────────────────────────────

COMPOSITE_SCHEMA: dict[str, Any] = {
    # Pipeline input parameters (shared state)
    "experiment_id": "string",
    "sim_base_path": "string",
    "output_dir": "string",
    "include_vio": "boolean",
    "include_mecillinam": "boolean",
    "vio_expression_bounds": "tuple[float,float]",
    "mecillinam_conc_bounds": "tuple[float,float]",
    "polynomial_order": "integer",
    "n_samples": "integer",
    "n_bins": "integer",
    "expected_cycle_time": "float",
    "prescreen": "boolean",
    # Pipeline outputs (written by the process)
    "pipeline_result": "tree[string]",
    "return_code": "integer",
}


def build_state(
    experiment_id: str,
    sim_base_path: str,
    output_dir: str = "./uq_results",
    include_vio: bool = True,
    include_mecillinam: bool = True,
    vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
    mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
    polynomial_order: int = 3,
    n_samples: int = 200,
    n_bins: int = 10,
    expected_cycle_time: float = 3600.0,
    prescreen: bool = False,
    process_config: dict[str, Any] | None = None,
    use_step: bool = False,
    with_emitter: bool = True,
) -> dict[str, Any]:
    """
    Build the composite state document.

    Args:
        experiment_id: Experiment identifier passed to Nextflow.
        sim_base_path: Root path to simulation outputs.
        output_dir: Where Nextflow writes results.
        include_vio: Include vio pathway parameters.
        include_mecillinam: Include mecillinam parameters.
        vio_expression_bounds: (lo, hi) for vio expression.
        mecillinam_conc_bounds: (lo, hi) for mecillinam concentration.
        polynomial_order: PCE polynomial order.
        n_samples: Number of LHS samples.
        n_bins: Number of cell cycle bins.
        expected_cycle_time: Expected cell cycle period (seconds).
        prescreen: Whether to run Morris prescreening.
        process_config: Extra config dict for the Nextflow process.
        use_step: If True, use NextflowRunStep instead of NextflowUQProcess.
        with_emitter: If True, attach a RAMEmitter to observe results.

    Returns:
        State dict consumable by ``Composite({"state": ...})``.
    """
    config = process_config or {}

    # Shared pipeline parameters
    state: dict[str, Any] = {
        "experiment_id": experiment_id,
        "sim_base_path": str(sim_base_path),
        "output_dir": str(output_dir),
        "include_vio": include_vio,
        "include_mecillinam": include_mecillinam,
        "vio_expression_bounds": list(vio_expression_bounds),
        "mecillinam_conc_bounds": list(mecillinam_conc_bounds),
        "polynomial_order": polynomial_order,
        "n_samples": n_samples,
        "n_bins": n_bins,
        "expected_cycle_time": expected_cycle_time,
        "prescreen": prescreen,
        # Output placeholders
        "pipeline_result": {},
        "return_code": 0,
    }

    # Wire the Nextflow process/step
    port_keys = [
        "experiment_id",
        "sim_base_path",
        "output_dir",
        "include_vio",
        "include_mecillinam",
        "vio_expression_bounds",
        "mecillinam_conc_bounds",
        "polynomial_order",
        "n_samples",
        "n_bins",
        "expected_cycle_time",
        "prescreen",
    ]

    input_wires = {k: [k] for k in port_keys}
    output_wires = {
        "pipeline_result": ["pipeline_result"],
        "return_code": ["return_code"],
        "output_dir": ["output_dir"],
    }

    if use_step:
        # Step variant — fires once when inputs are satisfied
        step_input_wires = {k: [k] for k in [
            "experiment_id", "sim_base_path", "output_dir",
            "include_vio", "include_mecillinam",
            "polynomial_order", "n_samples", "n_bins",
        ]}
        step_output_wires = {
            "pipeline_result": ["pipeline_result"],
            "return_code": ["return_code"],
        }
        state["nf_pipeline"] = {
            "_type": "step",
            "address": "local:NextflowRunStep",
            "config": config,
            "inputs": step_input_wires,
            "outputs": step_output_wires,
        }
    else:
        # Process variant — time-driven
        state["nf_pipeline"] = {
            "_type": "process",
            "address": "local:NextflowUQProcess",
            "config": config,
            "interval": 1.0,
            "inputs": input_wires,
            "outputs": output_wires,
        }

    # Optional emitter for observing pipeline_result
    if with_emitter:
        state["emitter"] = {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": {"pipeline_result": "tree[string]", "return_code": "integer"}},
            "inputs": {
                "pipeline_result": ["pipeline_result"],
                "return_code": ["return_code"],
            },
            "outputs": {},
        }

    return state


def build_composite(
    experiment_id: str,
    sim_base_path: str,
    output_dir: str = "./uq_results",
    core=None,
    **kwargs: Any,
) -> Composite:
    """
    Build and return a ready-to-run Composite for the Nextflow UQ pipeline.

    Args:
        experiment_id: Experiment identifier.
        sim_base_path: Path to simulation data.
        output_dir: Output directory for results.
        core: Optional pre-allocated Core. If None, one is created.
        **kwargs: Forwarded to ``build_state()``.

    Returns:
        A ``process_bigraph.Composite`` instance. Call ``.run(1.0)`` to
        execute the pipeline.
    """
    if core is None:
        core = get_core()

    state = build_state(
        experiment_id=experiment_id,
        sim_base_path=sim_base_path,
        output_dir=output_dir,
        **kwargs,
    )

    return Composite({"state": state}, core=core)
