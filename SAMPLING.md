# SAMPLING — how `uq sample` uses the vEcoli variants API

`uq sample` is a thin, vEcoli-specific implementation of steps 1–3 of the
**PyTUQ UQPC forward-UQ workflow**
(<https://sandialabs.github.io/pytuq/apps/uqpc.html>). It delegates every
heavy step to either PyTUQ (input PC, germ sampling) or vEcoli (parameter
mutation, simulation, Parquet emission). There is no hand-rolled parameter
evaluator, scheduler, or variant expander anywhere in this repository.

This document walks through that delegation end-to-end and cites the exact
functions and line numbers in both repos.

---

## 1. Setup inputs (UQPC step 1)

An input PC representation is built from per-parameter bounds. PyTUQ owns
the polynomial basis and the affine domain transform.

- `uq/workflow.py::_setup_input_pc` (lines 110–145) constructs a
  `pytuq.rv.pcrv.PCRV` with `pctype="LU"` (Legendre, uniform priors, order 1)
  whose coefficients encode the affine map `xi ∈ [-1, 1] → [lb, ub]`.
- The parameter list comes from `libuq/pipeline/param_loader.py` —
  `ParameterDataset.to_parameter_space(...)` projects scalar `SimulationDataEcoli`
  attributes into generic `SimDataParameter` specs. The defaults are the six
  parameters in `DEFAULT_SIM_DATA_PARAMETERS`.

**No hand-rolled samplers, CDFs, or bound checks** — PyTUQ's `PCRV` is the
single source of truth for the parameter space.

---

## 2. Generate samples (UQPC step 2)

`uq/cli.py::sample` draws samples from the germ measure using PyTUQ
directly, then lifts them back into physical space with `evalPC`.

- `uq/cli.py` lines ~128–141:
  ```python
  input_pc, _, _ = _setup_input_pc(bounds)
  np.random.seed(seed)
  germ_train = input_pc.sampleGerm(n_samples)
  X_train   = input_pc.evalPC(germ_train)
  if n_test > 0:
      np.random.seed(seed + 1)
      germ_test = input_pc.sampleGerm(n_test)
      X_test    = input_pc.evalPC(germ_test)
  ```

This is the UQPC `--sampl rand` path, with `--ntst = n_test` for held-out
validation (PyTUQ's surrogate-quality cross-check).

`X_test` is concatenated onto `X_train` before evaluation, so vEcoli runs
**one** workflow that covers both train and validation — the split is
preserved in the cache and consumed by `uq quantify` for test relative
errors.

---

## 3. Evaluate the model (UQPC step 3) — vEcoli variants API

Every sample becomes one vEcoli variant through the **`sim_data_setattr`**
variant function that lives in vEcoli itself (not in this repo), and the
whole batch is executed by vEcoli's Nextflow workflow. There is no shortcut
around Parca, no in-process `EcoliSim`, no hand-rolled mutation code.

### 3a. Variant encoding — `sim_data_setattr`

For each germ sample, we emit one dictionary of attribute-path mutations.
The variant list (one entry per sample) is handed to vEcoli's variant system
via the `sim_data_setattr` variant function:

- `uq/tui.py::_build_variants_from_samples` (lines 127–142):
  ```python
  def _build_variants_from_samples(X, param_specs):
      mutations_list = []
      for i in range(X.shape[0]):
          mutations = {}
          for j, spec in enumerate(param_specs):
              val = float(X[i, j])
              if spec.index is not None:
                  mutations[spec.attr_path] = {"__index__": spec.index, "__value__": val}
              else:
                  mutations[spec.attr_path] = val
          mutations_list.append(mutations)
      return {"sim_data_setattr": {"mutations": {"value": mutations_list}}}
  ```

The `"value": [...]` form is what vEcoli's variant parser recognises as a
"make one variant per list element" expansion — this is the canonical API
documented at <https://covertlab.github.io/vEcoli/workflows.html#variants>
and implemented in `../vEcoli/runscripts/create_variants.py::parse_variants`
(the `if param_type == "value"` branch).

The actual mutation logic lives in vEcoli:

- `../vEcoli/ecoli/variants/sim_data_setattr.py::apply_variant` (full file,
  58 lines) walks each dot-path with `getattr` and writes the new value
  (or indexed slot) directly onto the `SimulationDataEcoli` object before
  the sim runs. **This lives in vEcoli, not here**, so every uqEcoli run is
  driven by the official variant pipeline.

We contribute **zero** custom simulation-data editing code — the only thing
this repo does is translate a PyTUQ germ sample into the dict format
`apply_variant` expects.

### 3b. Config assembly — the vEcoli workflow config

- `uq/tui.py::_build_config` (lines 100–124) builds a minimal, standard
  vEcoli workflow config:
  ```python
  {
      "sim_data_path": sim_data_path,   # pre-computed simData.cPickle
      "experiment_id": "uqpc_batch",
      "emitter": "parquet",             # official vEcoli Parquet emitter
      "emitter_arg": {"out_dir": output_dir},
      "max_duration": max_duration,
      "n_init_sims": n_init_sims,       # lineage seeds — strategy 3
      "generations": generations,       # generations — strategy 2
      "single_daughters": True,
      "suffix_time": False,
      "variants": variants_section,     # {"sim_data_setattr": ...}
  }
  ```

  The config is written to
  `<cache_dir>/_batch/workflow_config.json` and passed verbatim to
  vEcoli. Notice that we pass `sim_data_path` so vEcoli **skips Parca** —
  it reuses the pre-computed pickle we already loaded in step 1.

### 3c. Execution — `runscripts/workflow.py --config`

`uq/cli.py` lines 175–182 spawn vEcoli's own Nextflow-backed workflow
entry point as a subprocess:

```python
workflow_script = os.path.join(vecoli_root, "runscripts", "workflow.py")
cmd = [sys.executable, workflow_script, "--config", str(config_path)]
proc = subprocess.Popen(cmd, ..., cwd=vecoli_root, env=env)
```

This is the exact command documented in vEcoli's workflow manual. vEcoli
then:

1. Runs `runscripts/create_variants.py` to materialize
   `variant_sim_data_0000.cPickle`, `..._0001.cPickle`, … from our mutation
   list (one baseline + one per sample).
2. Dispatches Nextflow processes for each `(variant, lineage_seed, generation)`
   triple.
3. Emits Parquet files in the official hive-partitioned layout:
   `history/experiment_id=.../variant=.../lineage_seed=.../generation=.../agent_id=.../NNN.pq`

While this runs, `uq/cli.py::sample` drives a rich-progress bar by scraping
stdout and polling `_count_completed_variants` (`uq/tui.py:145`), which in
turn walks the hive path looking for completed `variant=N` partitions.

### 3d. Result collection — hive-partitioned Parquet

- `uq/tui.py::_collect_variant_timeseries` (lines 164–214) reads the
  Parquet tree with `polars.read_parquet(..., hive_partitioning=True)` —
  the `variant`, `lineage_seed`, `generation`, and `time` columns come
  directly from vEcoli's emitter.
- For each variant index it extracts the observable columns, computes the
  per-variant timeseries mean (→ aggregated `Y`), and preserves the
  per-row `generation` / `lineage_seed` arrays. That per-row metadata
  is what enables RFC006 strategies 2 (by generation) and 3 (by lineage
  seed) in `uq quantify`.

The train/test split is restored by slicing the collected arrays at index
`n_samples` — the first `n_samples` variants are training, any remaining
variants are the UQPC `--ntst` validation set. Everything is stored in
`libuq.sampling.PrecomputedCache` (including new `X_test.npy`, `Y_test.npy`,
and `timeseries_test/`).

---

## 4. What's in this repo vs. what's in vEcoli / PyTUQ

| Concern | Owner | Where |
| --- | --- | --- |
| Polynomial basis, germ sampling, PC evaluation | **PyTUQ** | `pytuq.rv.pcrv.PCRV` |
| Input PC construction (Legendre affine map) | PyTUQ (invoked) | `uq/workflow.py::_setup_input_pc` |
| Variant function (attribute mutation) | **vEcoli** | `ecoli/variants/sim_data_setattr.py` |
| Variant expansion (`"value": [...]`) | **vEcoli** | `runscripts/create_variants.py::parse_variants` |
| Parca / `simData` build | **vEcoli** | handled by `runscripts/workflow.py` |
| Simulation dispatch + Nextflow orchestration | **vEcoli** | `runscripts/workflow.py` |
| Parquet emission + hive partitioning | **vEcoli** | vEcoli parquet emitter |
| Mutation dict building | uqEcoli | `uq/tui.py::_build_variants_from_samples` |
| Config assembly | uqEcoli | `uq/tui.py::_build_config` |
| Subprocess driver + progress UI | uqEcoli | `uq/cli.py::sample`, `uq/tui.py` |
| Hive Parquet → `(X, Y, meta)` reshape | uqEcoli | `uq/tui.py::_collect_variant_timeseries` |
| Cache format (`PrecomputedCache`) | uqEcoli | `libuq/sampling.py` |

The uqEcoli rows are intentionally thin. Every numerically-meaningful step
is implemented in vEcoli or PyTUQ; we only adapt their I/O formats.

---

## 5. Verification

You can verify each link in this chain yourself:

```bash
# The variant function we use is shipped by vEcoli, not this repo
ls ../vEcoli/ecoli/variants/sim_data_setattr.py

# Every parameter mutation goes through vEcoli's apply_variant
grep -n apply_variant ../vEcoli/ecoli/variants/sim_data_setattr.py

# The CLI actually spawns vEcoli's workflow.py
grep -n "runscripts/workflow.py\|runscripts.workflow" uq/cli.py uq/tui.py

# Germ sampling uses PyTUQ's PCRV
grep -n "sampleGerm\|evalPC" uq/workflow.py uq/cli.py

# Variant expansion is vEcoli's — we only emit the list form it parses
grep -n '"value"' uq/tui.py ../vEcoli/runscripts/create_variants.py
```

---

## 6. The novelty

This repo's value-add is not in re-implementing UQ primitives; it is in the
**end-user entrypoints** that glue PyTUQ and vEcoli together:

- `uq sample` / `uq quantify` — the two-stage CLI (Typer + rich).
- `uq tui` — a Textual terminal UI for the same workflow.
- `uq gui` / `uq dashboard` — marimo and tkinter GUIs over the cached
  results (`app/gui.py`, `app/dashboard_simple.py`, `app/uq_daw_simple.py`).
- `QuantifyResult.export(...)` — writes a dashboard-ready artifact
  directory (`population_surrogate/`, `growth_stratified_surrogate/`,
  per-strategy Sobol `.npy` files, `uq_results.json`).
- RFC006's four aggregation strategies (uniform, by generation, by lineage
  seed, growth-stratified) as a first-class feature of `quantify` — each
  still runs through PyTUQ's `PCRV`/`lreg` stack and the shared
  `run_uqpc` function in `uq/workflow.py`.

Everything else — the polynomial algebra, the sampling, the simulation —
belongs to the upstream projects.
