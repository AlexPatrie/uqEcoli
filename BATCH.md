# Batch Execution for UQ Sample Generation

## Overview

The UQ pipeline supports three modes for generating samples (Stage 1):

| Mode | Command | When to use |
|------|---------|-------------|
| **Local sequential** | `generate-samples` | Small sample counts, testing |
| **Local parallel** | `generate-samples --max-workers N` | Moderate samples, multi-core machine |
| **HPC batch** | `export-configs` + Nextflow + `collect-results` | Large samples (100+), cluster access |

All three produce the same `PrecomputedCache` that feeds into `quantify` (Stage 2).

---

## Mode A: Local Sequential

The default. Each sample is evaluated one at a time.

```bash
uv run uq generate-samples \
    mecillinam api_simulation_default \
    /path/to/sims \
    ./uq_cache \
    --n-samples 50 \
    --seed 42
```

## Mode B: Local Parallel

Uses `ProcessPoolExecutor` to evaluate multiple samples concurrently. Set `--max-workers` to the number of CPU cores you can dedicate (each EcoliSim is memory-heavy).

```bash
uv run uq generate-samples \
    mecillinam api_simulation_default \
    /path/to/sims \
    ./uq_cache \
    --n-samples 50 \
    --seed 42 \
    --max-workers 4
```

## Mode C: HPC Batch (Nextflow)

For large-scale runs, split sample generation into three steps:

### Step 1: Export configs

Generate per-sample vEcoli configs and variant-applied sim_data pickles:

```bash
uv run uq export-configs \
    /path/to/sims/mecillinam/parca/kb/simData.cPickle \
    ./batch \
    --n-samples 200 \
    --seed 42 \
    --include-vio \
    --include-mecillinam \
    --generations 1
```

This creates:

```
batch/
  configs/
    0.json            # Per-sample EcoliSim JSON config
    1.json
    ...
  sim_data/
    0.cPickle         # Variant-applied sim_data (ready to simulate)
    1.cPickle
    ...
  metadata.json       # Maps sample index -> parameter vector x
```

Each `configs/{i}.json` points to the corresponding `sim_data/{i}.cPickle` and has `"variants": {}` because the variants are already baked into the pickled sim_data. This means Nextflow (or any batch runner) just needs to run `EcoliSim.from_file(configs/{i}.json)` for each sample.

### Step 2: Run on Nextflow/HPC

Submit the configs to your Nextflow pipeline. The simplest approach:

```bash
# Example: run each sample config through vEcoli's Nextflow sim process
for config in batch/configs/*.json; do
    nextflow run sim.nf --config "$config"
done

# Or submit as a batch array job (SLURM example):
# sbatch --array=0-199 run_sample.sh
```

Each sample produces Parquet output under a directory named by `experiment_id` (e.g., `uq_sample_0000/history/`).

### Step 3: Collect results

After all Nextflow jobs complete, assemble outputs into a `PrecomputedCache`:

```bash
uv run uq collect-results \
    ./batch \
    /path/to/nextflow/outputs \
    --cache-dir ./uq_cache
```

This reads the hive-partitioned Parquet files from each `uq_sample_NNNN/history/` directory, computes time-mean aggregations, and writes the cache.

### Step 4: Analyze (same as always)

```bash
uv run uq quantify \
    mecillinam api_simulation_default \
    /path/to/sims \
    --precomputed-path ./uq_cache \
    --export-path ./uq_results
```

---

## What `export-configs` Does Internally

For each LHS sample `x[i]`:

1. **`param_space.sample_to_params(x)`** converts the numpy vector to `UQInputParametersVecoli`
2. **`.to_simulation_config()`** produces variant param dicts (e.g., `new_gene_internal_shift_variable_strength`, `mecillinam_timeline`)
3. **`_apply_variants(sim_data, variants)`** deep-copies the baseline `sim_data` and calls the real `ecoli.variants.<name>.apply_variant()` functions
4. The mutated sim_data is pickled to `sim_data/{i}.cPickle`
5. A minimal JSON config is written to `configs/{i}.json` pointing to the pickle

This means the variant application (the complex part with `internal_shift_dict`, `field_timeline`, expression/translation adjustments) happens once up front, not inside each simulation run.

## What `collect-results` Does Internally

1. Reads `metadata.json` to recover the sample-to-parameter mapping
2. For each sample `i`, reads Parquet from `{output_dir}/uq_sample_{i:04d}/history/`
3. Extracts observable columns (default: `dry_mass`, `cell_mass`, `volume`, `growth`)
4. Computes time-mean for each observable -> one `y` vector per sample
5. Stacks all `(x, y)` pairs into `X.npy` and `Y.npy`
6. Saves as a standard `PrecomputedCache` (same format as `generate-samples` output)

---

## Quick Reference

```bash
# Full HPC workflow:
uv run uq export-configs /path/to/simData.cPickle ./batch --n-samples 200
# ... run on Nextflow/SLURM ...
uv run uq collect-results ./batch /path/to/outputs --cache-dir ./cache
uv run uq quantify exp1 /sims --precomputed-path ./cache --export-path ./results

# Local parallel (no HPC needed):
uv run uq generate-samples exp1 /sims ./cache --n-samples 50 --max-workers 4
uv run uq quantify exp1 /sims --precomputed-path ./cache

# Check what's in a batch:
cat batch/metadata.json | python -m json.tool | head -20
```
