"""
process-bigraph Process implementations for Nextflow-orchestrated workflows.

Provides:
    NextflowProcess — a generalized Process that runs any arbitrary .nf
        workflow file. Input ports are forwarded as ``--key=value`` params
        to Nextflow; outputs are read from a configurable result JSON file.

    NextflowUQProcess — a specialization of NextflowProcess pre-wired for
        the RFC006 UQ pipeline (uq_pipeline.nf).

    NextflowRunStep — a dependency-triggered Step that fires a single
        Nextflow run and returns immediately (useful for one-shot execution
        inside a larger Composite).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from process_bigraph import Process, Step

# ─── Helpers ────────────────────────────────────────────────────────────────────

_WORKFLOW_FILE = Path(__file__).resolve().parent.parent / "workflows" / "nextflow" / "uq_pipeline.nf"


def _build_nextflow_cmd(
    workflow_file: str | Path,
    params: dict[str, Any],
    work_dir: str | Path | None = None,
    profile: str | None = None,
) -> list[str]:
    """Build a ``nextflow run`` command list from a parameter dict."""
    cmd = ["nextflow", "run", str(workflow_file)]
    if work_dir:
        cmd.extend(["-work-dir", str(work_dir)])
    if profile:
        cmd.extend(["-profile", profile])
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cmd.append(f"--{key}={str(value).lower()}")
        else:
            cmd.append(f"--{key}={value}")
    return cmd


def _read_result(output_dir: str | Path) -> dict[str, Any]:
    """Read the assembled pipeline_result.json produced by the workflow."""
    result_path = Path(output_dir) / "pipeline_result.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return {"error": f"Result file not found at {result_path}"}


# ─── NextflowProcess (generalized) ──────────────────────────────────────────────


class NextflowProcess(Process):
    """
    A generalized process-bigraph Process that runs **any** Nextflow workflow.

    Every key in the input state dict is forwarded to Nextflow as a
    ``--key=value`` CLI parameter. After the workflow completes, a result
    JSON file is read from the output directory and returned as the
    ``result`` output port, along with ``stdout``, ``stderr``, and
    ``return_code``.

    Config:
        workflow_file (str): **Required.** Absolute path to the ``.nf`` file.
        work_dir (str): Nextflow work directory. Empty string → auto tempdir.
        profile (str): Nextflow profile (e.g. ``"docker"``, ``"slurm"``).
            Empty string → none.
        result_file (str): Filename (relative to the output dir given by
            the ``output_dir`` input port) to read as structured JSON
            output. Empty string → skip JSON reading.
        output_dir_port (str): Name of the input port whose value is the
            output directory path. Defaults to ``"output_dir"``.
        resume (bool): Pass ``-resume`` to Nextflow for cached re-runs.
        cleanup (bool): Remove the Nextflow work dir after completion.
        nextflow_bin (str): Path to the ``nextflow`` binary.
        extra_args (str): Additional raw CLI arguments appended to the
            Nextflow command (space-separated string).
        env (str): JSON-encoded dict of extra environment variables for
            the Nextflow subprocess. Empty string → inherit parent env.

    Ports:
        **inputs** — ``{"params": "tree[string]"}`` by default.
            Override ``inputs()`` in a subclass for stricter typing, or
            wire a ``tree[string]`` port whose value is a flat dict of
            Nextflow params.
        **outputs** — ``result`` (string — JSON-serialized dict),
            ``stdout`` (string), ``stderr`` (string),
            ``return_code`` (integer).
            Call ``json.loads(state["result"])`` to unpack.

    Example composite state (runs a custom workflow)::

        {
            "params": {"input": "/data/reads.fq", "outdir": "./results"},
            "result": "",
            "stdout": "",
            "stderr": "",
            "return_code": 0,
            "my_nf": {
                "_type": "process",
                "address": "local:NextflowProcess",
                "config": {
                    "workflow_file": "/pipelines/rnaseq.nf",
                    "result_file": "results/summary.json",
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
    """

    config_schema = {
        "workflow_file": {"_type": "string", "_default": ""},
        "work_dir": {"_type": "string", "_default": ""},
        "profile": {"_type": "string", "_default": ""},
        "result_file": {"_type": "string", "_default": ""},
        "output_dir_port": {"_type": "string", "_default": "output_dir"},
        "resume": {"_type": "boolean", "_default": False},
        "cleanup": {"_type": "boolean", "_default": True},
        "nextflow_bin": {"_type": "string", "_default": "nextflow"},
        "extra_args": {"_type": "string", "_default": ""},
        "env": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {"params": "tree[string]"}

    def outputs(self):
        return {
            "result": "string",
            "stdout": "string",
            "stderr": "string",
            "return_code": "integer",
        }

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve_workflow(self) -> str:
        wf = self.config.get("workflow_file", "")
        if not wf:
            raise ValueError("NextflowProcess requires config.workflow_file to be set (absolute path to the .nf file).")
        return str(wf)

    def _build_cmd(
        self,
        workflow_file: str,
        nf_params: dict[str, Any],
    ) -> list[str]:
        nf_bin = self.config.get("nextflow_bin", "nextflow") or "nextflow"
        cmd = [nf_bin, "run", workflow_file]

        work_dir = self.config.get("work_dir")
        if work_dir:
            cmd.extend(["-work-dir", str(work_dir)])

        profile = self.config.get("profile")
        if profile:
            cmd.extend(["-profile", str(profile)])

        if self.config.get("resume", False):
            cmd.append("-resume")

        extra = self.config.get("extra_args", "")
        if extra:
            cmd.extend(extra.split())

        for key, value in nf_params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                cmd.append(f"--{key}={str(value).lower()}")
            else:
                cmd.append(f"--{key}={value}")

        return cmd

    def _resolve_env(self) -> dict[str, str] | None:
        env_str = self.config.get("env", "")
        if not env_str:
            return None
        extra = json.loads(env_str)
        merged = {**os.environ, **extra}
        return merged

    def _read_result_file(self, nf_params: dict[str, Any]) -> dict[str, Any]:
        result_file = self.config.get("result_file", "")
        if not result_file:
            return {}

        output_dir_port = self.config.get("output_dir_port", "output_dir")
        base = nf_params.get(output_dir_port, ".")
        path = Path(base) / result_file
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"error": f"Result file not found: {path}"}

    # ── update ──────────────────────────────────────────────────────────────

    def update(self, state: dict, interval: float) -> dict:
        """
        Launch an arbitrary Nextflow workflow with params from input state.

        The ``params`` input port should be a flat dict whose keys become
        Nextflow ``--key=value`` arguments.
        """
        workflow_file = self._resolve_workflow()

        # Accept either a nested "params" dict or the raw state itself
        nf_params: dict[str, Any] = state.get("params", state)
        if not isinstance(nf_params, dict):
            nf_params = {}

        work_dir = self.config.get("work_dir") or tempfile.mkdtemp(prefix="nf_")

        cmd = self._build_cmd(workflow_file, nf_params)
        env = self._resolve_env()

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(workflow_file).parent),
            env=env,
        )

        result = self._read_result_file(nf_params)

        if self.config.get("cleanup", True) and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

        # Serialize result as JSON string to avoid deep-merge issues
        # with heterogeneous nested dicts (bigraph tree merge expects
        # uniform leaf types).  Consumers call json.loads() to unpack.
        return {
            "result": json.dumps(result),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "return_code": proc.returncode,
        }


# ─── NextflowUQProcess ─────────────────────────────────────────────────────────


class NextflowUQProcess(Process):
    """
    A process-bigraph Process that executes the full RFC006 UQ pipeline
    via Nextflow.

    On each ``update()`` call the process:
      1. Reads pipeline parameters from its input ports
      2. Launches ``nextflow run uq_pipeline.nf`` with those parameters
      3. Waits for completion
      4. Reads the assembled result JSON
      5. Returns the result as an output delta

    Config:
        workflow_file: Path to the .nf file (defaults to bundled uq_pipeline.nf)
        work_dir: Nextflow work directory (defaults to a tempdir)
        profile: Nextflow profile name (e.g. "docker", "slurm")
        python: Python executable for Nextflow script blocks
        cleanup: Whether to clean up the Nextflow work dir after completion
    """

    config_schema = {
        "workflow_file": {"_type": "string", "_default": str(_WORKFLOW_FILE)},
        "work_dir": {"_type": "string", "_default": ""},
        "profile": {"_type": "string", "_default": ""},
        "python": {"_type": "string", "_default": "python"},
        "cleanup": {"_type": "boolean", "_default": True},
    }

    def inputs(self):
        return {
            "experiment_id": "string",
            "sim_base_path": "string",
            "output_dir": "string",
            "include_vio": "boolean",
            "include_mecillinam": "boolean",
            "vio_expression_bounds": "tuple[float,float]",
            "mecillinam_conc_bounds": "tuple[float,float]",
            "polynomial_order": "integer",
            "n_samples": "integer",
            "n_bins": "integer",
            "expected_cycle_time": "float",
            "prescreen": "boolean",
        }

    def outputs(self):
        return {
            "pipeline_result": "tree[string]",
            "return_code": "integer",
            "output_dir": "string",
        }

    def update(self, state: dict, interval: float) -> dict:
        """Launch Nextflow, wait for completion, return result."""
        # Map bigraph state to Nextflow params
        vio_bounds = state.get("vio_expression_bounds", (0.0, 5.0))
        mec_bounds = state.get("mecillinam_conc_bounds", (0.0, 10.0))

        nf_params = {
            "experiment_id": state.get("experiment_id", ""),
            "sim_base_path": state.get("sim_base_path", ""),
            "output_dir": state.get("output_dir", "./uq_results"),
            "include_vio": state.get("include_vio", True),
            "include_mecillinam": state.get("include_mecillinam", True),
            "vio_expression_lo": vio_bounds[0] if isinstance(vio_bounds, (list, tuple)) else 0.0,
            "vio_expression_hi": vio_bounds[1] if isinstance(vio_bounds, (list, tuple)) else 5.0,
            "mecillinam_conc_lo": mec_bounds[0] if isinstance(mec_bounds, (list, tuple)) else 0.0,
            "mecillinam_conc_hi": mec_bounds[1] if isinstance(mec_bounds, (list, tuple)) else 10.0,
            "polynomial_order": state.get("polynomial_order", 3),
            "n_samples": state.get("n_samples", 200),
            "n_bins": state.get("n_bins", 10),
            "expected_cycle_time": state.get("expected_cycle_time", 3600.0),
            "prescreen": state.get("prescreen", False),
            "python": self.config.get("python", "python"),
        }

        workflow_file = self.config.get("workflow_file", str(_WORKFLOW_FILE))
        work_dir = self.config.get("work_dir") or tempfile.mkdtemp(prefix="nf_uq_")
        profile = self.config.get("profile") or None

        cmd = _build_nextflow_cmd(workflow_file, nf_params, work_dir, profile)

        # Execute Nextflow
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(workflow_file).parent),
        )

        output_dir = nf_params["output_dir"]
        result = _read_result(output_dir)

        # Cleanup work dir if configured
        if self.config.get("cleanup", True) and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "pipeline_result": result,
            "return_code": proc.returncode,
            "output_dir": output_dir,
        }


# ─── NextflowRunStep ───────────────────────────────────────────────────────────


class NextflowRunStep(Step):
    """
    A dependency-triggered Step that fires a Nextflow run once.

    Unlike NextflowUQProcess (time-driven), this Step executes exactly
    once when its input dependencies are satisfied, making it suitable
    for inclusion in a DAG of Steps within a Composite.
    """

    config_schema = {
        "workflow_file": {"_type": "string", "_default": str(_WORKFLOW_FILE)},
        "work_dir": {"_type": "string", "_default": ""},
        "profile": {"_type": "string", "_default": ""},
        "python": {"_type": "string", "_default": "python"},
    }

    def inputs(self):
        return {
            "experiment_id": "string",
            "sim_base_path": "string",
            "output_dir": "string",
            "include_vio": "boolean",
            "include_mecillinam": "boolean",
            "polynomial_order": "integer",
            "n_samples": "integer",
            "n_bins": "integer",
        }

    def outputs(self):
        return {
            "pipeline_result": "tree[string]",
            "return_code": "integer",
        }

    def update(self, state: dict) -> dict:
        """Fire Nextflow and return the result."""
        nf_params = {
            "experiment_id": state.get("experiment_id", ""),
            "sim_base_path": state.get("sim_base_path", ""),
            "output_dir": state.get("output_dir", "./uq_results"),
            "include_vio": state.get("include_vio", True),
            "include_mecillinam": state.get("include_mecillinam", True),
            "polynomial_order": state.get("polynomial_order", 3),
            "n_samples": state.get("n_samples", 200),
            "n_bins": state.get("n_bins", 10),
            "python": self.config.get("python", "python"),
        }

        workflow_file = self.config.get("workflow_file", str(_WORKFLOW_FILE))
        work_dir = self.config.get("work_dir") or None
        profile = self.config.get("profile") or None

        cmd = _build_nextflow_cmd(workflow_file, nf_params, work_dir, profile)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(workflow_file).parent),
        )

        result = _read_result(nf_params["output_dir"])

        return {
            "pipeline_result": result,
            "return_code": proc.returncode,
        }
