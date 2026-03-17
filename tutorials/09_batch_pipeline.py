"""Tutorial 09: Batch Execution for UQ Sample Generation

This tutorial demonstrates the three modes of UQ sample generation:

  Mode A: Local sequential — one sample at a time (default)
  Mode B: Local parallel — ProcessPoolExecutor for multi-core machines
  Mode C: HPC batch — export configs for Nextflow, collect results after

All three produce the same PrecomputedCache format that feeds into
the ``quantify`` command (Stage 2 analysis).

The tutorial walks through Mode C step by step, since it's the most
involved and the primary path for production-scale UQ:

  1. Build a parameter space from sim_data
  2. Generate LHS samples
  3. Export per-sample vEcoli configs + variant-applied sim_data pickles
  4. Inspect the batch directory structure
  5. (Simulate) Collect results into PrecomputedCache
  6. Feed the cache into the full pipeline

Run with: uv run marimo run tutorials/09_batch_pipeline.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Tutorial 09: Batch Execution for UQ Sample Generation

    The UQ pipeline's most expensive step is **Stage 1: sample generation** —
    running the simulation function `f(x) -> y` at each LHS sample point.
    For whole-cell vEcoli simulations, each evaluation can take minutes to hours.

    This tutorial covers three execution strategies:

    | Mode | Strategy | Use case |
    |------|----------|----------|
    | **A** | Sequential | Testing, small runs |
    | **B** | Local parallel | Multi-core workstation |
    | **C** | HPC batch | Cluster with Nextflow/SLURM |

    All three produce the same `PrecomputedCache` → same `quantify` analysis.

    **This tutorial focuses on Mode C** (the HPC batch workflow), walking
    through each step programmatically.
    """)
    return


# =============================================================================
# OVERVIEW: THE THREE-MODE WORKFLOW
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## The Three Execution Modes

    ```
    Mode A (sequential):
      generate-samples ──► PrecomputedCache ──► quantify

    Mode B (parallel):
      generate-samples --max-workers 4 ──► PrecomputedCache ──► quantify

    Mode C (HPC batch):
      export-configs ──► Nextflow/SLURM ──► collect-results ──► PrecomputedCache ──► quantify
    ```

    Mode C splits the work into three CLI commands so that the expensive
    simulation runs happen on HPC infrastructure (Nextflow, SLURM, etc.)
    while the analysis runs locally or on a login node.
    """)
    return


# =============================================================================
# STEP 1: BUILD PARAMETER SPACE
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Build the Parameter Space

    The parameter space defines which vEcoli parameters are varied and
    their bounds.  It's constructed from `sim_data` — the same baseline
    simulation data that vEcoli uses.

    For this tutorial, we use `XSpaceVecoli` directly with synthetic
    bounds (no real sim_data needed).
    """)
    return


@app.cell
def _():
    import numpy as np

    from uq import XSpaceVecoli

    # Build parameter space (same as what export-configs does internally)
    param_space = XSpaceVecoli(
        include_vio=True,
        include_mecillinam=True,
        vio_expression_bounds=(0.0, 5.0),
        vio_trl_eff_bounds=(0.0, 2.0),
        mecillinam_conc_bounds=(0.0, 10.0),
    )

    print(f"Parameter space: {param_space.n_parameters} parameters")
    print(f"  Names:  {param_space.parameter_names}")
    print(f"  Bounds: {param_space.parameter_bounds}")

    return np, param_space, XSpaceVecoli


# =============================================================================
# STEP 2: GENERATE LHS SAMPLES
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Generate LHS Samples

    Latin Hypercube Sampling ensures good coverage of the parameter space.
    The samples are scaled from the unit hypercube to the physical parameter bounds.

    This step is fast (milliseconds) — no simulations are run yet.
    """)
    return


@app.cell
def _(np, param_space):
    from uq.sampling import generate_lhs_samples

    N_SAMPLES = 20
    SEED = 42

    X = generate_lhs_samples(param_space, n_samples=N_SAMPLES, seed=SEED)

    print(f"Generated {X.shape[0]} LHS samples, shape: {X.shape}")
    print(f"\nFirst 5 samples:")
    print(f"  {'vio_exp':>10s}  {'vio_trl':>10s}  {'mec_conc':>10s}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for _i in range(min(5, len(X))):
        print(f"  {X[_i, 0]:>10.4f}  {X[_i, 1]:>10.4f}  {X[_i, 2]:>10.4f}")

    # Verify samples are within bounds
    _bounds = np.array(param_space.parameter_bounds)
    assert np.all(X >= _bounds[:, 0] - 1e-10), "Samples below lower bound"
    assert np.all(X <= _bounds[:, 1] + 1e-10), "Samples above upper bound"
    print(f"\nAll samples within parameter bounds.")

    return N_SAMPLES, SEED, X, generate_lhs_samples


# =============================================================================
# STEP 3: EXPORT BATCH CONFIGS (MODE C)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Export Batch Configs

    This is the core of **Mode C**.  For each LHS sample `x[i]`:

    1. `param_space.sample_to_params(x)` converts the numpy vector to
       `UQInputParametersVecoli`
    2. `.to_simulation_config()` produces variant param dicts
       (e.g., `new_gene_internal_shift_variable_strength`, `mecillinam_timeline`)
    3. The variant dicts are applied to a deep-copied baseline `sim_data`
       using the real `ecoli.variants.<name>.apply_variant()` functions
    4. The mutated sim_data is pickled to `sim_data/{i}.cPickle`
    5. A JSON config is written to `configs/{i}.json`

    **Key insight:** variant application happens up front, so each Nextflow
    job just runs a plain EcoliSim with no variant logic.

    Since we don't have real `sim_data` in this tutorial, we'll demonstrate
    the parameter conversion and config structure without actually writing pickles.
    """)
    return


@app.cell
def _(X, param_space):
    import json

    # Show what export-configs does for a single sample
    _x = X[0]
    _uq_params = param_space.sample_to_params(_x)
    _sim_config = _uq_params.to_simulation_config()

    print("=== Sample 0: Parameter Conversion ===")
    print(f"x vector: {_x}")
    print(f"\nUQInputParametersVecoli:")
    print(f"  vio.expression: {_uq_params.vio.expression}")
    print(f"  vio.translation_efficiency: {_uq_params.vio.translation_efficiency}")
    print(f"  mecillinam.concentrations: {_uq_params.mecillinam.concentrations}")
    print(f"\nSimulation config (variant dicts):")
    print(json.dumps(_sim_config, indent=2, default=str))

    return (json,)


@app.cell
def _(json, mo):
    # Show the per-sample JSON config structure that export-configs writes
    _example_config = {
        "experiment_id": "uq_sample_0000",
        "sim_data_path": "/batch/sim_data/0.cPickle",
        "generations": 1,
        "emitter": "parquet",
        "variants": {},
    }

    mo.md(f"""
    ### Per-Sample Config Structure

    Each `configs/{{i}}.json` looks like this:

    ```json
    {json.dumps(_example_config, indent=2)}
    ```

    Note `"variants": {{}}` — the variants are baked into the pickled `sim_data`,
    not in the JSON config.  This means Nextflow just calls
    `EcoliSim.from_file(config)` with no variant dispatch needed.

    ### Batch Directory Layout

    ```
    batch/
      metadata.json          # Sample-to-parameter mapping
      configs/
        0.json               # EcoliSim config for sample 0
        1.json               # EcoliSim config for sample 1
        ...
      sim_data/
        0.cPickle            # Variant-applied sim_data for sample 0
        1.cPickle            # Variant-applied sim_data for sample 1
        ...
    ```
    """)
    return


@app.cell
def _(N_SAMPLES, X, json, np, param_space):
    # Demonstrate metadata.json structure (what export-configs writes)
    _metadata = {
        "parameter_names": param_space.parameter_names,
        "n_samples": N_SAMPLES,
        "n_params": param_space.n_parameters,
        "bounds": np.array(param_space.parameter_bounds).tolist(),
        "samples": {},
    }

    for _i in range(min(3, N_SAMPLES)):
        _metadata["samples"][str(_i)] = {
            "x": X[_i].tolist(),
            "config": f"batch/configs/{_i}.json",
            "sim_data": f"batch/sim_data/{_i}.cPickle",
        }

    _metadata["samples"]["..."] = "..."

    print("=== metadata.json (first 3 samples) ===")
    print(json.dumps(_metadata, indent=2))
    print(f"\nThis file is the contract between export-configs and collect-results.")
    print(f"It records which parameter vector x produced each sample config.")

    return


# =============================================================================
# STEP 4: SIMULATE (HPC)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Run Simulations on HPC

    After `export-configs`, you submit the batch to your HPC infrastructure.
    Each sample runs independently — perfect for embarrassingly parallel execution.

    **Nextflow example:**
    ```bash
    # Run each sample config
    for config in batch/configs/*.json; do
        nextflow run sim.nf --config "$config"
    done
    ```

    **SLURM array job example:**
    ```bash
    #!/bin/bash
    #SBATCH --array=0-199
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G

    CONFIG="batch/configs/${SLURM_ARRAY_TASK_ID}.json"
    uv run python -c "
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    sim = EcoliSim.from_file('$CONFIG')
    sim.build_ecoli()
    sim.run()
    "
    ```

    Each job produces Parquet output under `uq_sample_NNNN/history/`.
    """)
    return


# =============================================================================
# STEP 5: COLLECT RESULTS INTO PRECOMPUTED CACHE
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Collect Results

    After all HPC jobs complete, `collect-results` reads the Parquet outputs
    and assembles them into a `PrecomputedCache`.

    For each sample `i`:
    1. Read Parquet from `{output_dir}/uq_sample_{i:04d}/history/`
    2. Extract observable columns (dry_mass, cell_mass, volume, growth)
    3. Compute time-mean aggregation -> one `y` vector per sample
    4. Stack all `(x, y)` pairs into `X.npy` and `Y.npy`

    **CLI:**
    ```bash
    uv run uq collect-results ./batch /path/to/outputs --cache-dir ./cache
    ```

    Since we don't have real Parquet outputs here, let's demonstrate the
    `PrecomputedCache` format directly with synthetic data.
    """)
    return


@app.cell
def _(N_SAMPLES, SEED, X, np, param_space):
    import tempfile
    from pathlib import Path

    from uq.sampling import PrecomputedCache

    # Simulate what collect-results produces: synthetic Y matrix
    _rng = np.random.default_rng(SEED)
    _bounds = np.array(param_space.parameter_bounds)
    _X_norm = (X - _bounds[:, 0]) / (_bounds[:, 1] - _bounds[:, 0])

    # Synthetic linear response: Y = A @ x_norm + noise
    _A = _rng.standard_normal((4, param_space.n_parameters))
    _Y_base = _X_norm @ _A.T
    _Y = _Y_base + 0.1 * _rng.standard_normal(_Y_base.shape)

    # Create the cache (same format as collect-results output)
    _cache_dir = Path(tempfile.mkdtemp()) / "uq_cache"

    cache = PrecomputedCache(
        cache_dir=_cache_dir,
        X=X,
        Y=_Y,
        parameter_names=param_space.parameter_names,
        metadata={
            "bounds": _bounds.tolist(),
            "seed": SEED,
        },
    )
    cache.save()

    print(f"=== PrecomputedCache ===")
    print(f"  X shape: {cache.X.shape}  (n_samples x n_params)")
    print(f"  Y shape: {cache.Y.shape}  (n_samples x n_outputs)")
    print(f"  Parameters: {cache.parameter_names}")
    print(f"  Saved to: {cache.cache_dir}")
    print(f"\nFiles:")
    for _f in sorted(cache.cache_dir.iterdir()):
        print(f"  {_f.name}  ({_f.stat().st_size:,} bytes)")

    return Path, PrecomputedCache, cache, tempfile


@app.cell
def _(cache, json):
    # Show what the metadata looks like on disk
    _meta = json.loads((cache.cache_dir / "metadata.json").read_text())
    print("=== metadata.json (on disk) ===")
    print(json.dumps(_meta, indent=2))

    return


# =============================================================================
# STEP 6: FEED INTO PIPELINE (STAGE 2)
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Step 6: Analyze with `quantify`

    The `PrecomputedCache` from any mode (A, B, or C) feeds directly into
    Stage 2 analysis.  No simulations are re-run.

    **CLI:**
    ```bash
    uv run uq quantify exp1 /sims --precomputed-path ./cache --export-path ./results
    ```

    **Python API:**
    ```python
    from uq import handlers

    pipeline = handlers.pipeline(
        experiment_ids=["mecillinam"],
        sim_base_path="/sims",
        precomputed_path="./cache",
        export_path="./results",
    )
    ```

    Below we demonstrate loading the cache and verifying it's pipeline-ready.
    """)
    return


@app.cell
def _(cache, np):
    from uq.sampling import PrecomputedCache as _PC

    # Round-trip: load from disk
    _loaded = _PC.load(cache.cache_dir)

    print("=== Cache Round-Trip Verification ===")
    print(f"  X matches: {np.allclose(_loaded.X, cache.X)}")
    print(f"  Y matches: {np.allclose(_loaded.Y, cache.Y)}")
    print(f"  Parameter names match: {_loaded.parameter_names == cache.parameter_names}")
    print(f"  Bounds present: {'bounds' in _loaded.metadata}")

    # The critical invariant: parameter_names and bounds must match the
    # pipeline's parameter space. This is what quantify checks.
    _cached_names = _loaded.parameter_names
    _cached_bounds = np.array(_loaded.metadata["bounds"])
    print(f"\n  Cached params: {_cached_names}")
    print(f"  Cached bounds shape: {_cached_bounds.shape}")
    print(f"  All lower < upper: {np.all(_cached_bounds[:, 0] < _cached_bounds[:, 1])}")

    return


# =============================================================================
# MODE B DEMO: LOCAL PARALLEL
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Bonus: Mode B — Local Parallel Evaluation

    For moderate sample counts on a multi-core machine, `evaluate_batch()`
    accepts `max_workers` to parallelize via `ProcessPoolExecutor`.

    ```python
    from uq.generators.vecoli import VecoliSimulationFunc

    sim_func = VecoliSimulationFunc(
        baseline_sim_data=sim_data,
        param_space=param_space,
    )

    # Sequential (default):
    Y = sim_func.evaluate_batch(X)

    # Parallel (4 workers):
    Y = sim_func.evaluate_batch(X, max_workers=4)
    ```

    **Important:** Each EcoliSim instance consumes significant memory.
    Set `max_workers` conservatively — typically 2-4 on a workstation
    with 32GB+ RAM.

    The `generate-samples` CLI command passes this through:

    ```bash
    uv run uq generate-samples exp1 /sims ./cache \\
        --n-samples 50 --max-workers 4
    ```
    """)
    return


# =============================================================================
# CLI QUICK REFERENCE
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## CLI Quick Reference

    ### Mode A: Local Sequential
    ```bash
    uv run uq generate-samples exp1 exp2 /sims ./cache --n-samples 200
    uv run uq quantify exp1 exp2 /sims --precomputed-path ./cache
    ```

    ### Mode B: Local Parallel
    ```bash
    uv run uq generate-samples exp1 exp2 /sims ./cache \\
        --n-samples 200 --max-workers 4
    uv run uq quantify exp1 exp2 /sims --precomputed-path ./cache
    ```

    ### Mode C: HPC Batch
    ```bash
    # 1. Export configs (fast, local)
    uv run uq export-configs /path/to/simData.cPickle ./batch \\
        --n-samples 200 --include-vio --include-mecillinam

    # 2. Run on HPC (slow, distributed)
    # ... submit to Nextflow / SLURM ...

    # 3. Collect results (fast, local)
    uv run uq collect-results ./batch /path/to/outputs --cache-dir ./cache

    # 4. Analyze (fast, local)
    uv run uq quantify exp1 /sims --precomputed-path ./cache \\
        --export-path ./results
    ```

    ### Inspect batch state
    ```bash
    # Check what's in the batch metadata
    python -c "import json; print(json.dumps(
        json.load(open('batch/metadata.json')),
        indent=2))" | head -30

    # Check cache contents
    python -c "
    import numpy as np, json
    X = np.load('cache/X.npy')
    Y = np.load('cache/Y.npy')
    meta = json.load(open('cache/metadata.json'))
    print(f'X: {X.shape}, Y: {Y.shape}')
    print(f'Params: {meta[\"parameter_names\"]}')
    "
    ```
    """)
    return


# =============================================================================
# SUMMARY
# =============================================================================
@app.cell
def _(mo):
    mo.md("""
    ## Summary

    | Step | Function / CLI | What it does |
    |------|---------------|--------------|
    | **LHS sampling** | `generate_lhs_samples()` | Generates sample points in parameter space |
    | **Variant application** | `_apply_variants()` | Calls real `ecoli.variants.<name>.apply_variant()` |
    | **Config export** | `export_batch_configs()` / `uq export-configs` | Writes per-sample JSON + pickled sim_data |
    | **HPC execution** | Nextflow / SLURM | Runs EcoliSim per sample (embarrassingly parallel) |
    | **Result collection** | `collect_batch_results()` / `uq collect-results` | Reads Parquet, aggregates, builds cache |
    | **Analysis** | `handlers.pipeline()` / `uq quantify` | PCE + Sobol from cached (X, Y) |

    The key design choice: **variants are baked into the pickled sim_data**
    at export time.  This means:

    - HPC jobs don't need the UQ codebase — just vEcoli
    - Each job is fully self-contained (config + sim_data pickle)
    - No variant dispatch logic runs on the cluster
    - The complex `internal_shift_dict` / `field_timeline` / expression
      adjustments happen once, up front, on your local machine
    """)
    return


if __name__ == "__main__":
    app.run()
