# Troubleshooting

*Last updated: 2026-04-03*

## Sampling crashes with `createVariants failed: 1`

**Cause:** The `sim_data_setattr` variant module doesn't exist in vEcoli.

**Fix:** Ensure `ecoli/variants/sim_data_setattr.py` exists in the vEcoli
repo. It provides `apply_variant(sim_data, params)` which mutates sim_data
attributes by dot-path. Without it, `create_variants.py` fails with
`ModuleNotFoundError`.

```bash
ls /path/to/vEcoli/ecoli/variants/sim_data_setattr.py
```

## Sampling crashes with `FileNotFoundError: workflow_config.json`

**Cause:** Relative paths. `workflow.py` runs with `cwd=vEcoli_root`, so
`./uq_cache/_batch/workflow_config.json` resolves against the vEcoli
directory.

**Fix:** All paths in the config JSON must be absolute. The TUI calls
`.resolve()` on `sim_data_path` and `cache_dir`. If using the Python API
directly, pass absolute paths:

```python
sample(sim_data_path=str(Path(sd).resolve()), cache_dir=str(Path(cd).resolve()))
```

## Sampling crashes with `Output directory already exists`

**Cause:** A previous run left stale output in the batch directory.
`workflow.py` refuses to overwrite.

**Fix:** Delete the stale directories:

```bash
rm -rf ./uq_cache/_batch
rm -rf /path/to/vEcoli/nextflow_temp/uqpc_batch
```

The TUI does this automatically before each run.

## TUI hangs during sampling (no progress, no output)

**Cause (fixed):** The old implementation used `readline()` which blocks
until `\n`. Nextflow uses `\r` for its live progress updates, so
`readline()` hung indefinitely.

**Fix (already applied):** The TUI reads stdout in chunks via
`proc.stdout.read(4096)` in a dedicated thread, splitting on both `\r`
and `\n`. The main thread polls `proc.poll()` every second.

## Sobol indices are all zero

**Cause:** Coefficients weren't synced to PCRV after fitting. This happens
when using `PCE.build(regression='lsq')` without calling
`pce.pcrv.setCfs([pce.lreg.cf])` afterward.

**Fix:** Our `_fit_surrogate()` uses `pcrv.setMiCfs(mi_list, cfs_list)`
which handles coefficient syncing correctly. If you're calling PyTUQ
directly, always call `setCfs` after `build()` for `lsq` and `anl`
methods. Only `bcs` syncs automatically.

## `sampleGerm` TypeError about `seed` kwarg

**Cause:** `PCRV.sampleGerm(nsam)` does not accept a `seed` parameter
(despite some documentation suggesting it does).

**Fix:** Set `np.random.seed(seed)` before calling `sampleGerm()`:

```python
np.random.seed(42)
germ = pc.sampleGerm(n_samples)
```

## Dashboard shows "Surrogate .npy files not found"

**Cause:** The dashboard looks for `population_surrogate/coefficients.npy`
adjacent to the loaded `uq_results.json`. If the export directory isn't
where the dashboard expects it, the surrogate files won't be found.

**Fix:** Place the export directory where the dashboard searches
(the Marimo cell checks `examples/expected_output_simple/results`,
`examples/expected_output/results`, and `.`), or upload the
`uq_results.json` from the export directory that contains
`population_surrogate/`.
