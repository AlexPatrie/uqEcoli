# Sampling Pipeline — End-to-End Flow

*Last updated: 2026-04-03*

## Overview

Sampling is a single call to `uq.workflow.sample()` which executes
UQPC Steps 1-3 and produces a `PrecomputedCache` on disk.

```
simData.cPickle
      │
      ▼
  ParameterDataset          ← Step 1: load sim_data, build param space
      │
      ▼
  _setup_input_pc()         ← bounds → PCRV (Legendre, order 1)
      │
      ▼
  PCRV.sampleGerm(n)        ← Step 2: germ-space samples
  PCRV.evalPC(germ)         ← map to physical space
      │
      ▼
  Build config JSON          ← sim_data_setattr variants + workflow params
      │
      ▼
  workflow.py subprocess     ← Step 3: single Nextflow invocation
      │
      ▼
  Hive-partitioned Parquet   ← variant=N/lineage_seed=S/generation=G/
      │
      ▼
  _collect_variant_timeseries() ← read Parquet, extract per-variant arrays
      │
      ▼
  PrecomputedCache.save()    ← X.npy, Y.npy, germ_train.npy, timeseries/
```

## The Config JSON

A single vEcoli workflow config encodes all LHS samples as variants:

```json
{
    "sim_data_path": "/absolute/path/to/simData.cPickle",
    "experiment_id": "uqpc_batch",
    "emitter": "parquet",
    "emitter_arg": {"out_dir": "/absolute/path/to/output", "batch_size": 10800},
    "max_duration": 10800.0,
    "n_init_sims": 1,
    "generations": 1,
    "single_daughters": true,
    "variants": {
        "sim_data_setattr": {
            "mutations": {
                "value": [
                    {"process.transcription.fraction_active_rnap_free": 0.36, ...},
                    {"process.transcription.fraction_active_rnap_free": 0.40, ...}
                ]
            }
        }
    }
}
```

Each entry in `value` becomes one variant. Nextflow runs them concurrently.

## Output Matrix Shape

```
Total sims = (n_variants + 1) × n_init_sims × generations
```

The `+1` is the baseline (variant=0, unmodified sim_data).

## The sim_data_setattr Variant

`ecoli/variants/sim_data_setattr.py` in vEcoli provides `apply_variant()`
which mutates sim_data attributes by dot-path. This module must exist in
the vEcoli repo for `createVariants` to succeed.

## Cache Structure

```
uq_cache/
├── X.npy                    # (n_samples, n_params) — physical space
├── Y.npy                    # (n_samples, n_outputs) — time-averaged
├── germ_train.npy           # (n_samples, n_params) — germ space [-1,1]
├── metadata.json            # parameter names, bounds, observable columns
└── timeseries/
    ├── sample_0000.npy      # (n_timesteps, n_obs) — raw timeseries
    ├── sample_0000_meta.npz # generation + lineage_seed labels per row
    └── ...
```

`germ_train.npy` is saved so `quantify()` can load germ samples directly
without inverse-transforming from physical space.

## TUI Sampling

The TUI (`uq.tui`) runs sampling differently from the CLI — it launches
`workflow.py` via `subprocess.Popen` (not `.run`) with:

- **Live stdout streaming** in a background thread (reads chunks, splits
  on `\r` and `\n` to handle Nextflow's ANSI progress)
- **Background variant polling** — daemon thread rglobs for `.pq` files
  every 2 seconds, updates the progress bar
- **Phase detection** — parses Nextflow output to detect createVariants →
  simulating transitions
- **Cancel button** — sets a `threading.Event`, terminates subprocess
