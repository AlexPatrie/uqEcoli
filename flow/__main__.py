"""
flow.__main__ — CLI entry point for the Nextflow-backed UQ pipeline.

Parameterizes and executes the full RFC006 UQ pipeline via process-bigraph
Composite orchestration. The Composite wires shared pipeline state
(experiment_id, parameter bounds, PCE settings) into a NextflowUQProcess,
which launches ``nextflow run uq_pipeline.nf`` and writes the assembled
PipelineResult back into the composite state.

Usage:
    uv run python -m flow \\
        --experiment-id mecillinam \\
        --sim-base-path /path/to/vEcoli/api_integration/sims \\
        --output-dir ./uq_results \\
        --n-samples 200 \\
        --polynomial-order 3

    # Or use the Step variant (single-shot, no time loop):
    uv run python -m flow --use-step ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flow.nextflow.composite import build_composite, build_state, get_core


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m flow",
        description="Run the RFC006 UQ pipeline via Nextflow + process-bigraph.",
    )

    # Required
    p.add_argument("--experiment-id", required=True, help="Experiment identifier (e.g. 'mecillinam')")
    p.add_argument("--sim-base-path", required=True, help="Root path to simulation outputs")

    # Output
    p.add_argument("--output-dir", default="./uq_results", help="Where Nextflow writes results")

    # Parameter space
    p.add_argument("--include-vio", type=_bool, default=True, help="Include vio pathway parameters")
    p.add_argument("--include-mecillinam", type=_bool, default=True, help="Include mecillinam parameters")
    p.add_argument("--vio-expression-lo", type=float, default=0.0)
    p.add_argument("--vio-expression-hi", type=float, default=5.0)
    p.add_argument("--mecillinam-conc-lo", type=float, default=0.0)
    p.add_argument("--mecillinam-conc-hi", type=float, default=10.0)

    # Pipeline tuning
    p.add_argument("--polynomial-order", type=int, default=3, help="PCE polynomial order")
    p.add_argument("--n-samples", type=int, default=200, help="Number of LHS samples for PCE")
    p.add_argument("--n-bins", type=int, default=10, help="Number of cell cycle bins")
    p.add_argument("--expected-cycle-time", type=float, default=3600.0, help="Expected cell cycle period (s)")
    p.add_argument("--prescreen", action="store_true", help="Run Morris prescreening")

    # Process-bigraph options
    p.add_argument("--use-step", action="store_true", help="Use Step (one-shot) instead of Process (time-driven)")
    p.add_argument("--no-emitter", action="store_true", help="Skip RAMEmitter attachment")
    p.add_argument("--interval", type=float, default=1.0, help="Composite run interval (Process mode)")

    # Nextflow options
    p.add_argument("--nf-profile", default=None, help="Nextflow profile (e.g. 'docker', 'slurm')")
    p.add_argument("--nf-work-dir", default=None, help="Nextflow work directory")
    p.add_argument("--python", default="python", help="Python executable for Nextflow script blocks")

    # Output control
    p.add_argument("--print-state", action="store_true", help="Print final composite state as JSON")
    p.add_argument("--print-document", action="store_true", help="Print the composite document and exit (dry run)")

    return p.parse_args(argv)


def _bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Build process config from CLI flags
    process_config: dict = {}
    if args.python != "python":
        process_config["python"] = args.python
    if args.nf_profile:
        process_config["profile"] = args.nf_profile
    if args.nf_work_dir:
        process_config["work_dir"] = args.nf_work_dir

    # Build the composite state document
    state = build_state(
        experiment_id=args.experiment_id,
        sim_base_path=args.sim_base_path,
        output_dir=args.output_dir,
        include_vio=args.include_vio,
        include_mecillinam=args.include_mecillinam,
        vio_expression_bounds=(args.vio_expression_lo, args.vio_expression_hi),
        mecillinam_conc_bounds=(args.mecillinam_conc_lo, args.mecillinam_conc_hi),
        polynomial_order=args.polynomial_order,
        n_samples=args.n_samples,
        n_bins=args.n_bins,
        expected_cycle_time=args.expected_cycle_time,
        prescreen=args.prescreen,
        process_config=process_config,
        use_step=args.use_step,
        with_emitter=not args.no_emitter,
    )

    # Dry run: just print the document
    if args.print_document:
        print(json.dumps({"state": state}, indent=2, default=str))
        return 0

    # Allocate core and build composite
    core = get_core()
    composite = build_composite(
        experiment_id=args.experiment_id,
        sim_base_path=args.sim_base_path,
        output_dir=args.output_dir,
        core=core,
        include_vio=args.include_vio,
        include_mecillinam=args.include_mecillinam,
        vio_expression_bounds=(args.vio_expression_lo, args.vio_expression_hi),
        mecillinam_conc_bounds=(args.mecillinam_conc_lo, args.mecillinam_conc_hi),
        polynomial_order=args.polynomial_order,
        n_samples=args.n_samples,
        n_bins=args.n_bins,
        expected_cycle_time=args.expected_cycle_time,
        prescreen=args.prescreen,
        process_config=process_config,
        use_step=args.use_step,
        with_emitter=not args.no_emitter,
    )

    # Run the composite
    print(f"Running UQ pipeline via process-bigraph Composite...")
    print(f"  Experiment: {args.experiment_id}")
    print(f"  Sim base:   {args.sim_base_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Mode:       {'Step (one-shot)' if args.use_step else f'Process (interval={args.interval})'}")
    print()

    composite.run(args.interval)

    # Report results
    result = composite.state.get("pipeline_result", {})
    return_code = composite.state.get("return_code", -1)

    if return_code == 0:
        print("Pipeline completed successfully.")
    else:
        print(f"Pipeline finished with return code: {return_code}")

    if args.print_state:
        print("\n=== Final Composite State ===")
        # Filter to interesting keys
        display = {
            "experiment_id": composite.state.get("experiment_id"),
            "return_code": return_code,
            "pipeline_result": result,
            "output_dir": composite.state.get("output_dir"),
        }
        print(json.dumps(display, indent=2, default=str))

    if result and "error" not in result:
        # Summarize key outputs
        pop = result.get("population", {})
        cc = result.get("cell_cycle", {})
        vd = result.get("variance_decomposition", {})

        print(f"\nResults written to: {args.output_dir}")

        if pop.get("sobol_indices"):
            sobol = pop["sobol_indices"]
            if isinstance(sobol, dict):
                names = sobol.get("parameter_names", [])
                first = sobol.get("first_order", [])
                print(f"\nPopulation Sobol (first-order):")
                for name, val in zip(names, first):
                    print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

        if vd:
            print(f"\nVariance decomposition keys: {list(vd.keys())}")

    return 0 if return_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
