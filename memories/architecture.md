# Architecture — Package Layout & Client/Workflow Relationship

*Last updated: 2026-04-03*

## Package Layout

```
uqEcoli/
├── uq/                          # Main UQ package (entrypoint: uv run uq)
│   ├── workflow.py              # PyTUQ UQPC engine: sample(), quantify()
│   ├── cli.py                   # Typer CLI: sample, quantify, tui, dashboard
│   ├── tui.py                   # Textual TUI: interactive terminal interface
│   ├── pipeline.py              # Legacy pipeline (still exists, not primary)
│   ├── growth.py                # Growth fraction θ computation
│   └── __init__.py
├── libuq/                       # Full RFC006 pipeline (entrypoint: uv run uq-ultra)
│   ├── cli.py                   # Phase 1 + Phase 2 (Koopman, Morris, etc.)
│   ├── pipe.py                  # Pipeline orchestration
│   ├── sensitivity.py           # SensitivityAnalyzer, SobolIndices, PCESurrogate
│   ├── generators/vecoli.py     # TimeseriesGeneratorVecoli
│   ├── pipeline/                # param_loader, output_loader, workflow, models
│   ├── sampling.py              # PrecomputedCache, generate_lhs_samples
│   ├── pce/                     # PCE surrogate math
│   └── ...
├── app/
│   └── dashboard_simple.py      # Marimo dashboard (reads exported artifacts)
├── tests/
│   └── test_uqpc_handlers.py    # 46 tests for uq.workflow
├── TUTORIAL.md                  # End-user workflow guide
└── memories/                    # This directory
```

## Entrypoints

```toml
[project.scripts]
uq = "uq.cli:app"           # Main UQ CLI
uq-ultra = "libuq.cli:app"  # Full pipeline CLI
```

## Three Clients, One Workflow

`uq.workflow` is the single source of truth. Three user-facing clients
expose it in different formats:

```
                    ┌──────────────────────┐
                    │   uq.workflow        │
                    │   sample()           │
                    │   quantify()         │
                    └──────┬───────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌───────────┐ ┌─────────────────┐
    │ uq.cli       │ │ uq.tui    │ │ dashboard_simple │
    │ (Typer)      │ │ (Textual) │ │ (Marimo)         │
    │              │ │           │ │                   │
    │ uv run uq    │ │ uv run uq │ │ uv run marimo    │
    │  sample      │ │  tui      │ │  run app/...     │
    │  quantify    │ │           │ │                   │
    └──────────────┘ └───────────┘ └─────────────────┘
```

The CLI and TUI call `uq.workflow.sample()` / `uq.workflow.quantify()`
directly. The dashboard reads the exported artifacts (`uq_results.json`
+ `.npy` files) produced by `QuantifyResult.export()`.
