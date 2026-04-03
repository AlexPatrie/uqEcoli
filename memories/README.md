# Development Memories

Engineering decisions, architectural context, and troubleshooting notes
from recent development on the UQ framework. These are written to help
future contributors (and future-us) understand *why* things are the way
they are — not just *what* the code does.

## Index

| File | Topic |
|------|-------|
| [architecture.md](architecture.md) | Package layout, entrypoints, client/workflow relationship |
| [pytuq_integration.md](pytuq_integration.md) | How we use PyTUQ, what's from PyTUQ vs handrolled, critical gotchas |
| [sampling_pipeline.md](sampling_pipeline.md) | How sampling works end-to-end (PCRV → workflow.py → Parquet → cache) |
| [design_decisions.md](design_decisions.md) | Key design choices and the reasoning behind them |
| [troubleshooting.md](troubleshooting.md) | Known failure modes and how to fix them |
