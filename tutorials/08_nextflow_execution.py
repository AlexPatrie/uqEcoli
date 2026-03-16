"""Tutorial 08: Nextflow Pipeline Execution via process-bigraph

This tutorial demonstrates how to run the RFC006 UQ pipeline through
**Nextflow** orchestration, wrapped in a **process-bigraph Composite**.

Instead of calling ``execute_pipeline()`` directly (as in Tutorial 07),
we construct a bigraph composite document that wires pipeline parameters
into a ``NextflowUQProcess``, which launches the Nextflow workflow as a
subprocess. Nextflow then distributes the pipeline steps (data loading,
aggregation, Phase 1 GSA, Phase 2 cell cycle GSA) across its execution
engine — enabling parallelism, provenance tracking, and HPC/cloud
scheduling.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  process-bigraph Composite                                   │
    │                                                              │
    │  ┌─────────────┐        ┌───────────────────────┐           │
    │  │ Shared State │───────►│ NextflowUQProcess     │           │
    │  │              │ reads  │                       │           │
    │  │ experiment_id│        │ launches:             │           │
    │  │ sim_base_path│        │  nextflow run         │           │
    │  │ param bounds │        │    uq_pipeline.nf     │           │
    │  │ PCE settings │        │    --experiment_id=... │           │
    │  │              │        │    --n_samples=...     │           │
    │  │              │◄───────│                       │           │
    │  │ pipeline_    │ writes │ reads result JSON     │           │
    │  │   result     │        └───────────────────────┘           │
    │  │ return_code  │                                            │
    │  └──────┬───────┘                                            │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌─────────────┐                                             │
    │  │ RAMEmitter   │  observes pipeline_result + return_code    │
    │  └─────────────┘                                             │
    └──────────────────────────────────────────────────────────────┘

    Inside Nextflow (uq_pipeline.nf):
    ┌─────────────────────────────────────────────────────────────┐
    │  LOAD_DATA → AGGREGATE → ┬─ PHASE1_GSA ────┬→ ASSEMBLE    │
    │                          └─ PHASE2_CELL_CYCLE─┘             │
    └─────────────────────────────────────────────────────────────┘

Run with: uv run marimo run tutorials/08_nextflow_execution.py
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
    # Tutorial 08: Nextflow Pipeline Execution via process-bigraph

    This tutorial walks through the **programmatic Python API** for running
    the UQ pipeline via Nextflow — the same thing the `python -m flow` CLI does,
    but step by step so you can see how the pieces fit together.

    **Key concepts:**
    1. **process-bigraph `Process`** — a time-driven computation that declares
       typed input/output ports and returns deltas (never mutates state directly)
    2. **Composite** — a hierarchical state container that wires Processes to
       shared state via port→path mappings
    3. **Nextflow** — a workflow engine that distributes pipeline steps across
       local/HPC/cloud executors with automatic parallelism and provenance
    4. **`NextflowUQProcess`** — our Process subclass that bridges the two:
       reads pipeline params from bigraph state → launches `nextflow run` →
       writes the result back

    The tutorial proceeds in 6 steps:
    1. Import and inspect the process-bigraph types
    2. Define pipeline parameters
    3. Build the composite state document (the "bigraph")
    4. Inspect the wiring (how ports connect to state)
    5. Run the composite (which triggers Nextflow)
    6. Read results from composite state
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: The process-bigraph building blocks

    In process-bigraph, everything is built from three primitives:

    | Concept | What it is | Our usage |
    |---------|-----------|-----------|
    | **Process** | Time-driven computation with `update(state, interval) → delta` | `NextflowUQProcess` — launches Nextflow |
    | **Step** | Dependency-triggered computation with `update(state) → delta` | `NextflowRunStep` — one-shot alternative |
    | **Composite** | State container + wiring + runtime | Holds pipeline params, runs the process |

    A Process declares **ports** — typed connection points:
    - `inputs()` → dict of port_name → bigraph type string (what it reads)
    - `outputs()` → dict of port_name → bigraph type string (what it writes)

    **Wires** map each port to a path in the shared state tree.
    The runtime reads state at input paths, passes it to `update()`,
    and merges the returned delta at output paths.

    Let's look at our `NextflowUQProcess`:
    """)
    return


@app.cell
def _(mo):
    from flow.nextflow.processes import NextflowUQProcess, NextflowRunStep


    def _format_ports(ports: dict) -> str:
        """Format port dict for display."""
        _lines = []
        for _name, _type in ports.items():
            _lines.append(f"    '{_name}': '{_type}'")
        return "{\n" + ",\n".join(_lines) + "\n}"

    
    # Show port declarations
    _core_for_inspection = __import__("flow").get_core()
    _proc = NextflowUQProcess(config={}, core=_core_for_inspection)

    _input_ports = _proc.inputs()
    _output_ports = _proc.outputs()

    mo.md(f"""
    ### NextflowUQProcess port declarations

    **Input ports** (what the process reads from composite state):
    ```python
    {_format_ports(_input_ports)}
    ```

    **Output ports** (what the process writes back):
    ```python
    {_format_ports(_output_ports)}
    ```

    Each port has a **bigraph-schema type** (e.g., `"string"`, `"integer"`,
    `"tree[string]"`). The type system governs how updates are merged —
    `"integer"` values are *added* by default, `"tree[string]"` does a
    deep dict merge, etc.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Define pipeline parameters

    These are the same parameters you'd pass to `execute_pipeline()` in
    Tutorial 07, or to `python -m flow` on the command line.

    The key difference: instead of calling a function directly, we'll put
    these values into **composite state** where the Process can read them
    through its wired input ports.
    """)
    return


@app.cell
def _():
    # ── Pipeline parameters ──
    # In a real run, point these at your simulation data:
    EXPERIMENT_ID = "mecillinam"
    SIM_BASE_PATH = "/path/to/vEcoli/api_integration/sims"
    OUTPUT_DIR = "./uq_results"

    # Parameter space bounds (vio pathway + mecillinam)
    INCLUDE_VIO = True
    INCLUDE_MECILLINAM = True
    VIO_EXPRESSION_BOUNDS = (0.0, 5.0)
    MECILLINAM_CONC_BOUNDS = (0.0, 10.0)

    # PCE / pipeline tuning
    POLYNOMIAL_ORDER = 3
    N_SAMPLES = 200
    N_BINS = 10
    EXPECTED_CYCLE_TIME = 3600.0
    PRESCREEN = False
    return (
        EXPECTED_CYCLE_TIME,
        EXPERIMENT_ID,
        INCLUDE_MECILLINAM,
        INCLUDE_VIO,
        MECILLINAM_CONC_BOUNDS,
        N_BINS,
        N_SAMPLES,
        OUTPUT_DIR,
        POLYNOMIAL_ORDER,
        PRESCREEN,
        SIM_BASE_PATH,
        VIO_EXPRESSION_BOUNDS,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Build the composite state document

    The **state document** is the heart of process-bigraph. It's a nested dict
    that contains:

    1. **Shared state** — the pipeline parameters (experiment_id, bounds, etc.)
       and output placeholders (pipeline_result, return_code)
    2. **Process spec** — a dict with `_type: "process"`, an address pointing to
       our registered `NextflowUQProcess`, config, and **wire mappings**
    3. **Emitter spec** (optional) — a Step that observes result state

    `build_state()` constructs this document from keyword arguments:
    """)
    return


@app.cell
def _(
    EXPECTED_CYCLE_TIME,
    EXPERIMENT_ID,
    INCLUDE_MECILLINAM,
    INCLUDE_VIO,
    MECILLINAM_CONC_BOUNDS,
    N_BINS,
    N_SAMPLES,
    OUTPUT_DIR,
    POLYNOMIAL_ORDER,
    PRESCREEN,
    SIM_BASE_PATH,
    VIO_EXPRESSION_BOUNDS,
    mo,
):
    import json as _json

    from flow.nextflow.composite import build_state

    state_document = build_state(
        experiment_id=EXPERIMENT_ID,
        sim_base_path=SIM_BASE_PATH,
        output_dir=OUTPUT_DIR,
        include_vio=INCLUDE_VIO,
        include_mecillinam=INCLUDE_MECILLINAM,
        vio_expression_bounds=VIO_EXPRESSION_BOUNDS,
        mecillinam_conc_bounds=MECILLINAM_CONC_BOUNDS,
        polynomial_order=POLYNOMIAL_ORDER,
        n_samples=N_SAMPLES,
        n_bins=N_BINS,
        expected_cycle_time=EXPECTED_CYCLE_TIME,
        prescreen=PRESCREEN,
        # use_step=False → time-driven Process (default)
        # with_emitter=True → attach RAMEmitter (default)
    )

    mo.md(f"""
    ### The state document

    ```json
    {_json.dumps(state_document, indent=2, default=str)}
    ```

    **Key things to notice:**

    - The `"nf_pipeline"` entry has `_type: "process"` — this tells the runtime
      to instantiate a `NextflowUQProcess` at this location in the state tree
    - `"address": "local:NextflowUQProcess"` — resolved via the Core's link registry
    - `"inputs"` and `"outputs"` are the **wire mappings**: each port name maps
      to a path in the state tree (e.g., `"experiment_id": ["experiment_id"]`)
    - `"interval": 1.0` — the process fires every 1.0 time units
    - The `"emitter"` is a Step that captures `pipeline_result` and `return_code`
    """)
    return (state_document,)


@app.cell
def _(mo, state_document):
    _nf = state_document["nf_pipeline"]

    _input_wires = _nf["inputs"]
    _output_wires = _nf["outputs"]

    # Build a readable wiring diagram
    _lines = ["| Port | Direction | State Path |", "|------|-----------|------------|"]
    for _port, _path in _input_wires.items():
        _lines.append(f"| `{_port}` | input ← | `state[{_path[0]!r}]` |")
    for _port, _path in _output_wires.items():
        _lines.append(f"| `{_port}` | → output | `state[{_path[0]!r}]` |")

    mo.md(f"""
    ## Step 4: Inspect the wiring

    Wires connect process ports to locations in the shared state tree.
    When the runtime invokes the process, it:

    1. **Reads** state values at each input wire path
    2. Passes them to `NextflowUQProcess.update(state, interval)`
    3. The process launches Nextflow, waits, reads the result JSON
    4. Returns a delta dict keyed by output port names
    5. The runtime **merges** the delta at each output wire path

    ### Wire table for `nf_pipeline`

    {"chr(10)".join(_lines)}

    This is the fundamental abstraction: the Process doesn't know *where*
    in the state tree its data lives — it only sees port names. The wiring
    is what makes it composable.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Construct and run the Composite

    Now we assemble the pieces:

    1. **`get_core()`** — allocates a bigraph-schema `Core` with our process
       types registered in the link registry
    2. **`Composite({"state": ...}, core=core)`** — the runtime parses the state
       document, instantiates all Processes/Steps, computes the step dependency
       DAG, and is ready to execute
    3. **`composite.run(interval)`** — advances `global_time` by `interval`;
       any Process whose scheduled time falls within the interval fires

    When `NextflowUQProcess` fires, it:
    - Builds a `nextflow run uq_pipeline.nf --experiment_id=... --n_samples=...` command
    - Launches it as a subprocess
    - Waits for completion
    - Reads `pipeline_result.json` from the output directory
    - Returns it as the output delta

    ```python
    from flow.nextflow.composite import get_core, build_state
    from process_bigraph import Composite

    core = get_core()
    state = build_state(experiment_id="mecillinam", sim_base_path="/data/sims")
    composite = Composite({"state": state}, core=core)

    # This triggers: state → NextflowUQProcess → nextflow run → result → state
    composite.run(1.0)
    ```

    > **Note:** Running `composite.run(1.0)` below will actually launch Nextflow.
    > If you don't have simulation data at `SIM_BASE_PATH`, the Nextflow processes
    > will fail — but the composite machinery still works (you'll see return_code != 0).
    > To do a dry run without executing, see the `--print-document` flag in the CLI.
    """)
    return


@app.cell
def _(mo, state_document):
    from flow.nextflow.composite import get_core
    from process_bigraph import Composite

    # 1. Allocate Core with our types registered
    core = get_core()

    # 2. Construct the Composite from the state document
    composite = Composite({"state": state_document}, core=core)

    mo.md(f"""
    ### Composite constructed

    The runtime has parsed the state document and instantiated:
    - **1 Process**: `NextflowUQProcess` at path `['nf_pipeline']`
    - **1 Step**: `RAMEmitter` at path `['emitter']`

    State keys: `{sorted(composite.state.keys())}`

    The composite is ready. Calling `composite.run(1.0)` will trigger the
    Nextflow process.

    > **Uncomment the cell below to actually run it.**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Executing the pipeline

    Uncomment the code in the next cell to actually launch Nextflow.
    The process will:

    1. Build `nextflow run uq_pipeline.nf --experiment_id=mecillinam --sim_base_path=...`
    2. Nextflow orchestrates: `LOAD_DATA → AGGREGATE → (PHASE1 || PHASE2) → ASSEMBLE`
    3. Result JSON is read back into composite state

    ```python
    # Uncomment to run:
    # composite.run(1.0)
    # print(f"Return code: {composite.state['return_code']}")
    # print(f"Pipeline result keys: {list(composite.state.get('pipeline_result', {}).keys())}")
    ```
    """)
    return


@app.cell
def _():
    # ── Uncomment to execute the Nextflow pipeline ──
    # composite.run(1.0)
    #
    # # Check results
    # _rc = composite.state.get("return_code", -1)
    # _result = composite.state.get("pipeline_result", {})
    # print(f"Return code: {_rc}")
    # print(f"Result keys: {list(_result.keys())}")
    #
    # # Population Sobol indices
    # _pop = _result.get("population", {})
    # if _pop:
    #     _sobol = _pop.get("sobol_indices", {})
    #     print(f"Population Sobol: {_sobol}")
    pass
    return


@app.cell
def _(
    EXPERIMENT_ID,
    MECILLINAM_CONC_BOUNDS,
    N_BINS,
    N_SAMPLES,
    OUTPUT_DIR,
    POLYNOMIAL_ORDER,
    SIM_BASE_PATH,
    VIO_EXPRESSION_BOUNDS,
    mo,
):
    import json as _json2

    from flow.nextflow.composite import build_state as _build_state2

    _step_state = _build_state2(
        experiment_id=EXPERIMENT_ID,
        sim_base_path=SIM_BASE_PATH,
        output_dir=OUTPUT_DIR,
        vio_expression_bounds=VIO_EXPRESSION_BOUNDS,
        mecillinam_conc_bounds=MECILLINAM_CONC_BOUNDS,
        polynomial_order=POLYNOMIAL_ORDER,
        n_samples=N_SAMPLES,
        n_bins=N_BINS,
        use_step=True,  # ← Step instead of Process
    )

    mo.md(f"""
    ## Step 6: The Step variant — one-shot execution

    process-bigraph also has **Steps** — dependency-triggered computations
    that fire once when their inputs are satisfied (no time interval).

    `NextflowRunStep` is the Step counterpart to `NextflowUQProcess`.
    Use `build_state(use_step=True)` to get a Step-based document:

    ```json
    {_json2.dumps(_step_state["nf_pipeline"], indent=2)}
    ```

    Notice:
    - `"_type": "step"` instead of `"process"`
    - No `"interval"` field
    - The Step fires immediately when the Composite is constructed
      (since all its inputs are already present in the initial state)

    **When to use which:**

    | Variant | `_type` | Fires when | Use case |
    |---------|---------|-----------|----------|
    | `NextflowUQProcess` | `process` | Every `interval` time units | Repeated/iterative runs |
    | `NextflowRunStep` | `step` | Input dependencies satisfied | One-shot pipeline execution |
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path as _Path

    # Resolve relative to this tutorial file, or fall back to the process module location
    _nf_file = _Path(__file__).resolve().parent.parent / "flow" / "workflows" / "nextflow" / "uq_pipeline.nf"
    if not _nf_file.exists():
        from flow.nextflow.processes import _WORKFLOW_FILE
        _nf_file = _WORKFLOW_FILE
    try:
        _nf_source = _nf_file.read_text()
        # Show just the workflow block (the orchestration logic)
        _wf_start = _nf_source.find("workflow UQ_PIPELINE")
        _wf_end = _nf_source.find("\nworkflow {", _wf_start)
        _wf_block = _nf_source[_wf_start:_wf_end].strip()
    except Exception:
        _wf_block = "(Could not read uq_pipeline.nf — run from the repo root)"

    mo.md(f"""
    ## Inside the Nextflow workflow

    The `.nf` file defines 5 Nextflow processes that map to the RFC006 pipeline:

    | Nextflow Process | RFC006 Step | What it does |
    |-----------------|-------------|--------------|
    | `LOAD_DATA` | Steps 1-2 | DuckDB + OutputExtractor → timeseries parquet |
    | `AGGREGATE` | Steps 3-4 | 3 aggregation strategies + variance decomposition |
    | `PHASE1_GSA` | Steps 5a-7a | Morris → PCE → Sobol (population-level) |
    | `PHASE2_CELL_CYCLE` | Steps 5b-7b | GSA obs → Koopman → Strategy4 → per-stage Sobol |
    | `ASSEMBLE_RESULTS` | — | Merge everything into `pipeline_result.json` |

    **Phase 1 and Phase 2 run in parallel** — they both depend only on
    `AGGREGATE` output, so Nextflow schedules them concurrently.

    ### The workflow orchestration block

    ```groovy
    {_wf_block}
    ```

    Each Nextflow process runs its Python logic in a `script:` block that
    imports from the `uq` package — the same functions used by `execute_pipeline()`.
    Nextflow handles scheduling, caching, and provenance.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Advanced: Nextflow configuration via process-bigraph config

    The `NextflowUQProcess` accepts a `config` dict that controls
    Nextflow-specific behavior:

    ```python
    state = build_state(
        experiment_id="mecillinam",
        sim_base_path="/data/sims",
        process_config={
            "workflow_file": "/custom/path/to/uq_pipeline.nf",
            "profile": "slurm",       # Nextflow profile for HPC
            "work_dir": "/scratch/nf", # Nextflow work directory
            "python": "uv run python", # Python for script blocks
            "cleanup": True,           # Remove work dir after completion
        },
    )
    ```

    This separation is intentional:
    - **Pipeline parameters** (experiment_id, n_samples, etc.) live in
      **shared state** → readable by any process in the composite
    - **Infrastructure config** (Nextflow profile, work dir) lives in
      **process config** → private to `NextflowUQProcess`

    In process-bigraph terms: state is the **place graph** (data that flows
    between processes), config is per-edge metadata.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## CLI equivalence

    Everything in this tutorial can be done from the command line:

    ```bash
    # Run the pipeline (Process mode, single interval)
    uv run python -m flow \
        --experiment-id mecillinam \
        --sim-base-path /path/to/sims \
        --output-dir ./uq_results \
        --n-samples 200 \
        --polynomial-order 3 \
        --print-state

    # One-shot Step mode
    uv run python -m flow \
        --experiment-id mecillinam \
        --sim-base-path /path/to/sims \
        --use-step

    # Dry run: just print the composite document
    uv run python -m flow \
        --experiment-id mecillinam \
        --sim-base-path /path/to/sims \
        --print-document

    # With HPC profile
    uv run python -m flow \
        --experiment-id mecillinam \
        --sim-base-path /path/to/sims \
        --nf-profile slurm \
        --nf-work-dir /scratch/nf_work
    ```

    The CLI does exactly what this tutorial does:
    1. `parse_args()` → parameter values
    2. `build_state()` → composite state document
    3. `get_core()` → bigraph-schema Core with registered types
    4. `Composite({"state": state}, core=core)` → runtime
    5. `composite.run(interval)` → triggers NextflowUQProcess → Nextflow
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    | Layer | What | Where |
    |-------|------|-------|
    | **UQ pipeline** | RFC006 7-step analysis | `uq/pipeline/workflow.py` |
    | **Nextflow workflow** | Distributed execution of those steps | `flow/workflows/nextflow/uq_pipeline.nf` |
    | **process-bigraph Process** | Typed wrapper that launches Nextflow | `flow/nextflow/processes.py` |
    | **Composite document** | Wires params → Process → results | `flow/nextflow/composite.py` |
    | **CLI / this tutorial** | User-facing parameterization | `flow/__main__.py` / this file |

    The process-bigraph layer adds:
    - **Type safety** — all ports and state have bigraph-schema types
    - **Composability** — the Nextflow process can be nested inside larger composites
    - **Provenance** — the RAMEmitter records state snapshots
    - **Separation of concerns** — pipeline params (state) vs infra config (process config)

    The Nextflow layer adds:
    - **Parallelism** — Phase 1 and Phase 2 run concurrently
    - **Caching** — Nextflow's `-resume` skips completed steps
    - **Portability** — same workflow runs locally, on SLURM, or in the cloud
    - **Provenance** — Nextflow logs every process execution with inputs/outputs
    """)
    return


if __name__ == "__main__":
    app.run()
