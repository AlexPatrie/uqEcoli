# Design Decisions

*Last updated: 2026-04-03*

## PyTUQ-native sampling, not scipy LHS

We use `PCRV.sampleGerm()` + `PCRV.evalPC()` for sample generation
instead of `scipy.stats.qmc.LatinHypercube`. This follows the PyTUQ
UQPC workflow directly and ensures consistency between the sampling
distribution and the Legendre basis used in PCE fitting.

## No mock data in production workflow

`uq.workflow.sample()` has no `live=False` synthetic mode. It always
runs real vEcoli via `runscripts/workflow.py`. `DataDrivenWrapper` is
not imported. Test fixtures use hand-crafted numpy arrays with known
analytical properties (e.g., `Y = 2*x0 + 0.5*x1^2`).

## Self-contained workflow module

`uq.workflow` imports only atomic pieces — `ParameterDataset`,
`TimeseriesGeneratorVecoli`, `PrecomputedCache`, PyTUQ classes. It does
NOT delegate to `libuq.handlers.generate_samples()` or
`libuq.pipe.initialize_datasets()`. This keeps the module simple,
testable, and presentable to stakeholders without requiring them to
understand the full `libuq` architecture.

## ANOVA excluded by stakeholder agreement

A stakeholder confirmed ANOVA is optional for RFC006 compliance. The
`uq.workflow` pipeline does NOT include ANOVA or variance decomposition.
It focuses purely on PCE surrogate → Sobol indices.

## Four strategies, one algorithm

All four RFC006 aggregation strategies call the same `run_uqpc()` core.
They differ only in how they aggregate the raw timeseries into the `Y`
matrix before feeding it to PCE:

| Strategy | Aggregation |
|----------|-------------|
| 1 | Mean across all timesteps, seeds, generations |
| 2 | Mean within each generation (one PCE per generation) |
| 3 | Mean within each seed (one PCE per seed) |
| 4 | Mean within each growth-stage bin (one PCE per θ bin) |

The `n_variants` axis is always the row dimension — the axis PCE
regresses over. The strategies just change what's in the columns.

## Absolute paths for workflow.py

All paths passed to `workflow.py` must be absolute. The subprocess runs
with `cwd=vEcoli_root`, so relative paths resolve against the vEcoli
directory, not the uqEcoli directory. The TUI's `_do_sample()` calls
`.resolve()` on both `sim_data_path` and `cache_dir` before writing
the config JSON.

## Clean stale outputs before each run

`workflow.py` refuses to write into an existing experiment directory.
The TUI's `_do_sample()` removes `_batch/` and
`vEcoli/nextflow_temp/uqpc_batch/` before launching, ensuring a clean
slate every time.
