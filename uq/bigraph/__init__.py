"""
uq.bigraph — process-bigraph integration for the RFC006 UQ pipeline.

Identifies three categories of computation that benefit from
process-bigraph's compositional runtime:

1. **Long-running subprocesses** (vEcoli workflow.py) — managed as
   Processes with real-time progress emission instead of manual
   polling loops.

2. **PCE surrogate evaluation** — a stateless Step shared across
   all UI frontends (tkinter, marimo, TUI) instead of duplicated
   ``legendre_eval()`` in each dashboard.

3. **Per-strategy sensitivity analysis** — independent UQPC fits
   (strategies 1-4) composed as parallel Steps instead of
   sequential function calls.

Modules:

    processes   Atomic Process/Step implementations
    composites  Pre-wired Composite builders for common workflows
"""

from uq.bigraph.composites import (
    build_explorer_composite,
    build_full_composite,
    build_quantify_composite,
    get_core,
)
from uq.bigraph.processes import (
    CollectCache,
    Export,
    PCEEvaluate,
    Quantify,
    RunSimulations,
    SetupInputs,
    StrategyFit,
)

__all__ = [
    # Steps
    "SetupInputs",
    "RunSimulations",
    "CollectCache",
    "Quantify",
    "Export",
    "PCEEvaluate",
    "StrategyFit",
    # Composites
    "build_full_composite",
    "build_quantify_composite",
    "build_explorer_composite",
    "get_core",
]
