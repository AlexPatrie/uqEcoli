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

This went through three iterations:

**v1 bug:** Used `readline()` which blocks until `\n`. Nextflow uses `\r`
for progress. Fix: split on both `\r` and `\n`.

**v2 bug:** Used `proc.stdout.read(4096)` which blocks until 4096 bytes
accumulate or EOF. Nextflow outputs a few lines every 10-30 seconds, so
the buffer never fills — all output appears at once when the process ends.
Fix: use `os.read(proc.stdout.fileno(), 4096)` which returns as soon as
ANY bytes are available (raw POSIX non-greedy read).

**v3 bug:** Daemon threads called `call_from_thread()` directly, but
Textual only processes `call_from_thread` from the `@work` thread. Fix:
daemon threads write to a shared `log_queue` (with lock), and the main
`@work` loop drains it every second.

**Current (working):** `os.read(fd, 4096)` in a daemon thread → shared
`log_queue` → `@work` main loop drains + `call_from_thread(write_log)`.

## TUI shows garbage characters / Mac beeps during sampling

**Cause:** Nextflow uses ANSI escape codes beyond simple colors — cursor
movement (`\x1b[5A`), erase-line (`\x1b[K`), 256-color (`\x1b[38;5;232m`),
and bell (`\x07`). The original regex `\x1b\[[0-9;]*[a-zA-Z]` missed these.

**Fix:** Comprehensive regex:
```python
re.compile(r"\x1b\[[\d;]*[A-Za-z]|\x1b\[\d*[A-GJK]|\x07")
```

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
