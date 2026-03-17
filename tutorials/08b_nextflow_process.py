"""Tutorial 08b: Running any Nextflow workflow with NextflowProcess

This tutorial demonstrates the **generalized** NextflowProcess — a
process-bigraph Process that can run any arbitrary `.nf` workflow file.

We use a simple example workflow (hello_stats.nf) that:
  1. Generates random numbers
  2. Computes summary statistics
  3. Writes a result JSON

The tutorial shows step by step how to:
  - Wire a NextflowProcess into a process-bigraph Composite
  - Pass parameters from composite state → Nextflow --params
  - Read the result JSON back into composite state
  - Actually execute it (no simulation data needed!)

Run with: uv run marimo run tutorials/08b_nextflow_process.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


# ── Cell 1: Imports ─────────────────────────────────────────────────────────────


@app.cell
def _():
    import marimo as mo

    return (mo,)


# ── Cell 2: Overview ────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    # Tutorial 08b: Running any Nextflow workflow via process-bigraph

    `NextflowProcess` is a **generalized** process-bigraph `Process` that
    can run **any** `.nf` workflow file. You configure it with:

    - `config.workflow_file` — path to the `.nf` file
    - `config.result_file` — JSON file to read back as output
    - `config.output_dir_port` — which param key holds the output directory

    It reads a `params` dict from its input port, forwards every key as a
    `--key=value` Nextflow argument, waits for completion, and returns
    the result JSON + stdout/stderr/return_code.

    ```
    ┌─────────────────────────────────────────────────────┐
    │  Composite State                                     │
    │                                                      │
    │  params: {greeting: "hi", count: "10", ...}          │
    │       │                                              │
    │       ▼  (input port: params)                        │
    │  ┌─────────────────────┐                             │
    │  │  NextflowProcess    │                             │
    │  │                     │  nextflow run hello_stats.nf│
    │  │  config:            │  --greeting=hi --count=10   │
    │  │    workflow_file: …  │         │                   │
    │  │    result_file: …   │         ▼                   │
    │  │                     │  reads result.json          │
    │  └──────────┬──────────┘                             │
    │             │  (output ports)                         │
    │             ▼                                         │
    │  result: '{"greeting":"hi","statistics":{...}}'      │
    │  return_code: 0                                      │
    │  stdout: "..."                                       │
    └─────────────────────────────────────────────────────┘
    ```
    """)
    return


# ── Cell 3: The example workflow ────────────────────────────────────────────────


@app.cell
def _(mo):
    from pathlib import Path as _Path

    from flow.nextflow.processes import _WORKFLOW_FILE

    _hello_nf = _WORKFLOW_FILE.parent / "hello_stats.nf"
    _src = _hello_nf.read_text()

    # Extract just the workflow block
    _wf_idx = _src.find("workflow {")
    _proc_section = _src[_src.find("process GENERATE_DATA") : _wf_idx].strip()
    _wf_block = _src[_wf_idx:].strip()

    mo.md(f"""
    ## The example workflow: `hello_stats.nf`

    A minimal 3-step Nextflow pipeline that needs nothing but Python:

    ```
    GENERATE_DATA → COMPUTE_STATS → WRITE_RESULT
    ```

    | Process | What it does |
    |---------|-------------|
    | `GENERATE_DATA` | Generates `count` random numbers with `seed` |
    | `COMPUTE_STATS` | Computes mean, std, min, max |
    | `WRITE_RESULT` | Writes `result.json` with greeting + statistics |

    **Parameters:** `--greeting`, `--count`, `--seed`, `--output_dir`

    **Output:** `result.json` in `output_dir`:
    ```json
    {{"greeting": "hi", "count": 10, "statistics": {{"n": 10, "mean": ..., "std": ...}}, "status": "complete"}}
    ```

    The workflow file lives at:
    `{_hello_nf}`
    """)

    return (_hello_nf,)


# ── Cell 4: Step 1 — Set up parameters ─────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Define the Nextflow parameters

    These become `--key=value` arguments passed to `nextflow run`.
    They live in composite state under the `params` key — the
    NextflowProcess reads them through its `params` input port.
    """)
    return


@app.cell
def _():
    import tempfile as _tempfile

    # Nextflow params — these become --greeting=... --count=... etc.
    GREETING = "hello from process-bigraph"
    COUNT = 15
    SEED = 42
    OUTPUT_DIR = _tempfile.mkdtemp(prefix="nf_hello_")

    return COUNT, GREETING, OUTPUT_DIR, SEED


# ── Cell 5: Step 2 — Build the composite state document ────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Build the composite state document

    The state document is a plain dict with:
    1. **`params`** — a flat dict of Nextflow parameters (all values as strings)
    2. **Output placeholders** — `result`, `stdout`, `stderr`, `return_code`
    3. **Process spec** — `_type: "process"`, address, config, and wire mappings

    The **wires** connect ports to state paths:
    - Input: `{"params": ["params"]}` — read the `params` dict
    - Output: `{"result": ["result"], ...}` — write results back

    The **config** tells NextflowProcess which `.nf` file to run and
    which JSON file to read back:
    """)
    return


@app.cell
def _(COUNT, GREETING, OUTPUT_DIR, SEED, _hello_nf, mo):
    import json as _json

    # The composite state document
    state_doc = {
        # ── Nextflow parameters (input to the process) ──
        "params": {
            "greeting": GREETING,
            "count": str(COUNT),
            "seed": str(SEED),
            "output_dir": OUTPUT_DIR,
        },
        # ── Output placeholders (written by the process) ──
        "result": "",
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        # ── The NextflowProcess wired into the state tree ──
        "runner": {
            "_type": "process",
            "address": "local:NextflowProcess",
            "config": {
                # Which .nf file to run
                "workflow_file": str(_hello_nf),
                # Which JSON file to read from the output dir
                "result_file": "result.json",
                # Which param key holds the output directory path
                "output_dir_port": "output_dir",
            },
            "interval": 1.0,
            # Wires: port → state path
            "inputs": {
                "params": ["params"],
            },
            "outputs": {
                "result": ["result"],
                "stdout": ["stdout"],
                "stderr": ["stderr"],
                "return_code": ["return_code"],
            },
        },
    }

    mo.md(f"""
    ### The state document

    ```json
    {_json.dumps(state_doc, indent=2)}
    ```

    **Key things:**
    - `"params"` values are **strings** — Nextflow receives them as CLI args
    - `"config.workflow_file"` points to `hello_stats.nf`
    - `"config.result_file"` tells the process to read `result.json`
      from whatever directory `params.output_dir` points to
    - `"result"` starts as `""` — it will be filled with a JSON string
      after the workflow runs
    """)

    return (state_doc,)


# ── Cell 6: Step 3 — Construct the Composite ───────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Construct the Composite

    Three lines:
    1. `get_core()` — allocates bigraph-schema Core with `NextflowProcess` registered
    2. `Composite({"state": ...}, core=core)` — parses the document, instantiates
       the NextflowProcess, resolves wiring
    3. `composite.run(1.0)` — advances time by 1.0, triggering the process

    When the process fires it:
    1. Reads `state["params"]` via the input wire
    2. Builds: `nextflow run hello_stats.nf --greeting="hello from process-bigraph" --count=15 --seed=42 --output_dir=/tmp/...`
    3. Runs it as a subprocess, waits for completion
    4. Reads `/tmp/.../result.json` → serializes as JSON string
    5. Returns `{"result": "<json>", "stdout": "...", "return_code": 0}`
    6. Runtime merges the delta into state at the output wire paths
    """)
    return


@app.cell
def _(mo, state_doc):
    from flow import NextflowProcess, get_core
    from process_bigraph import Composite

    # 1. Allocate Core with NextflowProcess registered
    core = get_core()

    # 2. Build the Composite from our state document
    composite = Composite({"state": state_doc}, core=core)

    mo.md(f"""
    ### Composite ready

    State keys: `{sorted(composite.state.keys())}`

    The NextflowProcess is instantiated at `state["runner"]` and wired to
    read `params` and write `result`, `stdout`, `stderr`, `return_code`.
    """)

    return Composite, composite, core, get_core


# ── Cell 7: Step 4 — Run it! ───────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Run the Composite

    `composite.run(1.0)` triggers the NextflowProcess, which launches
    `hello_stats.nf`. This actually executes — no mocks, no stubs.

    The Nextflow workflow runs three processes in sequence
    (GENERATE_DATA → COMPUTE_STATS → WRITE_RESULT), then the
    NextflowProcess reads `result.json` and writes it back into
    composite state.
    """)
    return


@app.cell
def _(composite, mo):
    import json as _json2

    # Actually run it — this launches Nextflow!
    composite.run(1.0)

    # Read results from composite state
    _rc = composite.state["return_code"]
    _result_str = composite.state["result"]
    _result = _json2.loads(_result_str) if _result_str else {}

    _status_icon = "OK" if _rc == 0 else "FAILED"

    mo.md(f"""
    ### Result: {_status_icon} (return code: {_rc})

    ```json
    {_json2.dumps(_result, indent=2)}
    ```

    **What happened:**
    1. process-bigraph read `params` from state → passed to `NextflowProcess.update()`
    2. NextflowProcess built and ran: `nextflow run hello_stats.nf --greeting=... --count=15 ...`
    3. Nextflow executed: `GENERATE_DATA` → `COMPUTE_STATS` → `WRITE_RESULT`
    4. NextflowProcess read `result.json` from the output dir
    5. Returned it as a JSON string via the `result` output port
    6. process-bigraph merged the delta into composite state

    The result tells us:
    - **Greeting:** `{_result.get("greeting", "?")}`
    - **Count:** `{_result.get("count", "?")}` random numbers generated
    - **Mean:** `{_result.get("statistics", {}).get("mean", "?")}`
    - **Std:** `{_result.get("statistics", {}).get("std", "?")}`
    """)

    return


# ── Cell 8: Inspect stdout ──────────────────────────────────────────────────────


@app.cell
def _(composite, mo):
    _stdout = composite.state.get("stdout", "")
    _stderr = composite.state.get("stderr", "")

    mo.md(f"""
    ### Nextflow stdout

    ```
    {_stdout[:2000] if _stdout else "(empty)"}
    ```

    {"### Nextflow stderr" if _stderr else ""}
    {"```" if _stderr else ""}
    {_stderr[:1000] if _stderr else ""}
    {"```" if _stderr else ""}
    """)

    return


# ── Cell 9: How to use with your own workflow ──────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## Using NextflowProcess with your own workflow

    To run any `.nf` file, just change the config:

    ```python
    from flow import NextflowProcess, get_core
    from process_bigraph import Composite

    core = get_core()

    state = {
        # Your Nextflow params (all values as strings)
        "params": {
            "reads": "/data/reads/*.fq.gz",
            "genome": "/ref/hg38.fa",
            "outdir": "./results",
        },
        # Output placeholders
        "result": "",
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        # The process
        "nf_runner": {
            "_type": "process",
            "address": "local:NextflowProcess",
            "config": {
                "workflow_file": "/pipelines/rnaseq.nf",
                "result_file": "multiqc/summary.json",  # relative to outdir
                "output_dir_port": "outdir",             # which param is the output dir
                "profile": "slurm",                      # Nextflow profile
                "resume": True,                          # -resume for caching
            },
            "interval": 1.0,
            "inputs": {"params": ["params"]},
            "outputs": {
                "result": ["result"],
                "stdout": ["stdout"],
                "stderr": ["stderr"],
                "return_code": ["return_code"],
            },
        },
    }

    comp = Composite({"state": state}, core=core)
    comp.run(1.0)

    import json
    result = json.loads(comp.state["result"])
    print(result)
    ```

    **Config reference:**

    | Config key | Type | Default | Purpose |
    |-----------|------|---------|---------|
    | `workflow_file` | str | *required* | Path to the `.nf` file |
    | `result_file` | str | `""` | JSON file to read back (relative to output dir) |
    | `output_dir_port` | str | `"output_dir"` | Which param key is the output directory |
    | `profile` | str | `""` | Nextflow profile (`"docker"`, `"slurm"`, etc.) |
    | `resume` | bool | `false` | Pass `-resume` for cached re-runs |
    | `cleanup` | bool | `true` | Remove Nextflow work dir after completion |
    | `nextflow_bin` | str | `"nextflow"` | Path to nextflow binary |
    | `extra_args` | str | `""` | Additional CLI args (space-separated) |
    | `env` | str | `""` | JSON-encoded extra env vars |
    """)
    return


# ── Cell 10: Comparison with NextflowUQProcess ─────────────────────────────────


@app.cell
def _(mo):
    mo.md("""
    ## NextflowProcess vs NextflowUQProcess

    | | `NextflowProcess` | `NextflowUQProcess` |
    |-|-------------------|---------------------|
    | **Scope** | Any `.nf` workflow | UQ pipeline only (`uq_pipeline.nf`) |
    | **Input ports** | `params: tree[string]` (generic dict) | Typed: `experiment_id: string`, `n_samples: integer`, etc. |
    | **Config** | `workflow_file` required | Defaults to bundled `uq_pipeline.nf` |
    | **Result** | JSON string (generic) | `tree[string]` (UQ-specific) |
    | **Use case** | Wrap any Nextflow pipeline | Run the RFC006 UQ pipeline specifically |

    `NextflowUQProcess` (Tutorial 08) is a **specialization** with
    domain-specific typed ports. `NextflowProcess` (this tutorial) is the
    **general-purpose** building block — use it to wrap any Nextflow
    workflow into a process-bigraph Composite.
    """)
    return


if __name__ == "__main__":
    app.run()
