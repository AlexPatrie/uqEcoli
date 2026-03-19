"""
vEcoli simulation function for UQ pipeline: f(x) -> y.

This module provides a parameter-agnostic simulation wrapper that:
1. Accepts x as a numpy array positionally aligned with XSpace parameters
2. Converts x -> UQInputParametersVecoli -> variant param dicts via XSpace
3. Applies vEcoli variant functions (apply_variant) to a deep-copied sim_data
4. Runs vEcoli simulations as **subprocesses** via ecoli_master_sim CLI
   with the Parquet emitter (no in-process EcoliSim)
5. Collects Parquet outputs and returns timeseries arrays

The simulation function does NOT know which parameters are being varied --
that is controlled by the XSpace / parameter space layer. Morris prescreening
can reduce the parameter set between phases without changing f.

Variant application uses the real vEcoli apply_variant() functions, which
handle complex multi-step sim_data mutations (e.g., new gene expression
adjustment, internal_shift_dict setup, condition/timeline changes) that
cannot be reduced to simple setattr calls.
"""
import abc
import copy
import importlib
import json as _json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from uq.common import BaseClass
from uq.io import get_bucket
from uq.pipeline.models import SimulationConfig

logger = logging.getLogger(__name__)

# -- Helpers ----------------------------------------------------------------


def _get_nested_dict(d: dict, keys: list[str]) -> Any:
    """Traverse a nested dict by a list of keys."""
    for k in keys:
        d = d[k]
    return d


def _fix_new_gene_rel_adj(
    sim_data: SimulationDataEcoli,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Broadcast rel_adj lists to match the actual new gene counts in sim_data.

    The ``new_gene_internal_shift_variable_strength`` variant asserts::

        len(new_gene_rna_indices) == len(rel_exp_adj_list)
        len(new_monomer_indices)  == len(rel_trl_eff_adj_list)

    ``VioPathwayParams`` defaults both to ``[1.0]``, but violacein sim_data
    may have N monomers in a single operon (1 RNA, N monomers).  This
    helper broadcasts short lists to the correct length.
    """
    if "rel_adj" not in params:
        return params

    from ecoli.variants.new_gene_internal_shift_variable_strength import (
        get_new_gene_ids_and_indices,
    )

    _, rna_indices, _, monomer_indices = get_new_gene_ids_and_indices(sim_data)
    n_rna = len(rna_indices)
    n_mono = len(monomer_indices)

    rel_adj = params["rel_adj"]
    exp_list = rel_adj.get("rel_exp_adj_list", [1.0])
    trl_list = rel_adj.get("rel_trl_eff_adj_list", [1.0])

    # Broadcast length-1 lists to match actual gene counts
    if len(exp_list) == 1 and n_rna > 1:
        exp_list = exp_list * n_rna
    if len(trl_list) == 1 and n_mono > 1:
        trl_list = trl_list * n_mono

    params = {
        **params,
        "rel_adj": {
            "rel_exp_adj_list": exp_list,
            "rel_trl_eff_adj_list": trl_list,
        },
    }
    return params


def _apply_variants(
    sim_data: SimulationDataEcoli,
    variants_config: dict[str, list[dict[str, Any]]],
) -> SimulationDataEcoli:
    """Apply vEcoli variant functions to sim_data.

    Uses the real ``ecoli.variants.<name>.apply_variant()`` functions,
    which correctly handle complex multi-step mutations (expression
    adjustment, internal_shift_dict, condition/timeline changes, etc.).

    Args:
        sim_data: The SimulationDataEcoli to mutate (already deep-copied).
        variants_config: Dict mapping variant function names to lists of
            param dicts, as produced by
            ``UQInputParametersVecoli.to_simulation_config()["variants"]``.
            Example::

                {
                    "new_gene_internal_shift_variable_strength": [{
                        "condition": "basal",
                        "induction_gen": 1,
                        "exp_trl_eff": {"exp": 1.5, "trl_eff": 0.8},
                    }],
                    "mecillinam_timeline": [{
                        "times": [0.0],
                        "concentrations": [5.0],
                    }],
                }

    Returns:
        The mutated sim_data (same object, mutated in place).
    """
    for variant_name, param_dicts in variants_config.items():
        variant_mod = importlib.import_module(f"ecoli.variants.{variant_name}")
        for params in param_dicts:
            # Fix rel_adj list lengths for new gene variants
            if variant_name == "new_gene_internal_shift_variable_strength":
                params = _fix_new_gene_rel_adj(sim_data, params)
            sim_data = variant_mod.apply_variant(sim_data, params)
    return sim_data


# -- Parquet column mapping --------------------------------------------------

# Map short observable names (used in output_keys) to Parquet column names
_SHORT_TO_PARQUET: dict[str, str] = {
    "cell_mass": "listeners__mass__cell_mass",
    "dry_mass": "listeners__mass__dry_mass",
    "volume": "listeners__mass__volume",
    "growth": "listeners__mass__growth",
    "growth_rate": "listeners__mass__growth",
}

# Default observable columns to extract from Parquet output
_DEFAULT_PARQUET_COLUMNS = [
    "listeners__mass__dry_mass",
    "listeners__mass__cell_mass",
    "listeners__mass__volume",
    "listeners__mass__growth",
]


def _resolve_parquet_columns(output_keys: list[str] | None) -> list[str]:
    """Convert short output key names to full Parquet column names."""
    if output_keys is None:
        return list(_DEFAULT_PARQUET_COLUMNS)
    resolved = []
    for key in output_keys:
        resolved.append(_SHORT_TO_PARQUET.get(key, key))
    return resolved


def _read_parquet_timeseries(
    output_dir: Path,
    experiment_id: str,
    observable_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Read hive-partitioned Parquet output and extract timeseries.

    Args:
        output_dir: Root output directory (the ``out_dir`` passed to emitter).
        experiment_id: The experiment ID used in the simulation config.
        observable_columns: Full Parquet column names to extract.

    Returns:
        Tuple of:
        - timeseries array of shape ``(n_timesteps, n_obs)``
        - list of observable column names
    """
    # Parquet emitter writes to:
    # {out_dir}/{experiment_id}/history/experiment_id={id}/variant=.../.../*.pq
    history_base = output_dir / experiment_id / "history"

    # Try both .pq and .parquet extensions
    pq_files = list(history_base.rglob("*.pq"))
    if not pq_files:
        pq_files = list(history_base.rglob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(
            f"No Parquet files found under {history_base}. "
            f"Check that the simulation completed successfully."
        )

    df = pl.read_parquet(
        [str(p) for p in pq_files],
        hive_partitioning=True,
    )

    # Filter to available observable columns
    available = [c for c in observable_columns if c in df.columns]
    if not available:
        raise ValueError(
            f"None of {observable_columns} found in Parquet columns: "
            f"{df.columns[:20]}..."
        )

    # Sort by time if available
    if "time" in df.columns:
        df = df.sort("time")

    obs_df = df.select(available).fill_null(0.0)
    ts_array = obs_df.to_numpy().astype(np.float64)

    # Use short names for the observable labels
    parquet_to_short = {v: k for k, v in _SHORT_TO_PARQUET.items()}
    obs_names = [parquet_to_short.get(c, c) for c in available]

    return ts_array, obs_names


# -- Subprocess simulation runner -------------------------------------------


def _build_sim_config(
    sim_data_path: str,
    experiment_id: str,
    output_dir: str,
    max_duration: float,
    variant_index: int = 0,
    seed: int = 0,
    generations: int = 1,
    base_config_path: str | None = None,
) -> dict[str, Any]:
    """Build a vEcoli simulation config dict for subprocess execution."""
    config: dict[str, Any] = {}
    if base_config_path is not None:
        config = _json.loads(Path(base_config_path).read_text())

    config.update({
        "sim_data_path": str(sim_data_path),
        "experiment_id": experiment_id,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": str(output_dir)},
        "max_duration": max_duration,
        "generations": generations,
        "seed": seed,
        "variant": variant_index,
        "n_init_sims": 1,
        "lineage_seed": 0,
        "divide": False,
        "variants": {},  # variants already baked into sim_data
        "raw_output": True,
    })
    return config


def _run_sim_subprocess(
    config_path: Path,
    cwd: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run ecoli_master_sim.py as a subprocess.

    Args:
        config_path: Path to the JSON config file.
        cwd: Working directory for the subprocess.
        timeout: Timeout in seconds (None = no timeout).

    Returns:
        CompletedProcess result.

    Raises:
        RuntimeError: If the subprocess exits with non-zero status.
    """
    cmd = [
        sys.executable,
        "-m", "ecoli.experiments.ecoli_master_sim",
        "--config", str(config_path),
    ]

    logger.info("Running vEcoli simulation: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"vEcoli simulation failed (exit code {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr:\n{result.stderr[-2000:]}"
        )

    return result


# -- Simulation wrapper class -----------------------------------------------


@dataclass
class ITimeseriesBatchProcessor(BaseClass, abc.ABC):
    """
    Methods:
        _apply_parameter_mutations(x:np.ndarray) -> Any
        _run_batch(X, config) -> np.ndarray
    """
    @abc.abstractmethod
    def _apply_parameter_mutations(self, x: np.ndarray) -> Any:
        """Deep-copy baseline sim_data and apply variant mutations from x.

        Flow:
            x (numpy array)
            -> param_space.sample_to_params(x) -> UQInputParameters
            -> .to_simulation_config() -> {"variants": {name: [params]}, ...}
            -> _apply_variants(sim_data, variants) using real apply_variant() (for example, in vecoli)
        """
        pass

    @abc.abstractmethod
    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """Run simulations for a batch of parameter vectors.

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Max parallel subprocesses. None = sequential.
            batch_dir: Working directory for batch artifacts. If None,
                a temporary directory is created.

        Returns:
            Tuple of:
            - Y_aggregated: (n_samples, n_outputs) time-mean array
            - Y_timeseries: list of per-sample timeseries arrays, or None
        """
        pass


@dataclass
class TimeseriesGenerator(ITimeseriesBatchProcessor):
    param_space: Any  # XSpace — avoid circular import
    max_duration: float
    output_keys: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    def _apply_parameter_mutations(self, x: np.ndarray) -> Any:
        pass

    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        raise NotImplementedError("TODO: implement this to pass general func/callable")

    def evaluate_batch(self, X: np.ndarray, max_workers: int | None = None) -> np.ndarray:
        Y_agg, _ = self._run_batch(X, max_workers=max_workers)
        return Y_agg


@dataclass
class TimeseriesGeneratorVecoli(ITimeseriesBatchProcessor):
    """Parameter-agnostic vEcoli simulation wrapper for the UQ pipeline.

    Implements the dual interface required by the ``./uq`` workflow:

    - ``__call__(x)`` -> raw timeseries ``(n_timesteps, n_obs)``
      (used by ``Strategy4Wrapper`` in Phase 2)
    - ``evaluate_batch(X)`` -> aggregated outputs ``(n_samples, n_outputs)``
      (used by ``SensitivityAnalyzer`` in Phase 1)

    All simulations are executed as **subprocesses** via
    ``ecoli_master_sim.py`` with the Parquet emitter. No ``EcoliSim``
    object is held in the UQ process memory.

    The parameter vector ``x`` is a 1-D numpy array whose elements
    correspond positionally to ``param_space.parameter_names``.  The
    ``param_space`` (XSpace) converts ``x`` into
    ``UQInputParametersVecoli`` via ``sample_to_params(x)``, which is
    then converted to vEcoli variant param dicts.  The real
    ``ecoli.variants.<name>.apply_variant()`` functions handle the
    complex sim_data mutations (expression adjustment, internal shifts,
    condition/timeline changes).

    Args:
        baseline_sim_data: The unperturbed SimulationDataEcoli object.
        param_space: XSpace instance that converts x -> UQInputParametersVecoli.
        sim_config_path: Path to a base vEcoli JSON config file (optional).
        max_duration: Simulation duration in seconds.
        output_keys: Which outputs to extract from the Parquet timeseries.
            If None, extracts default mass listeners.
    """

    baseline_sim_data: SimulationDataEcoli
    param_space: Any  # XSpace — avoid circular import
    sim_config_path: str | None = None
    max_duration: float = 10800.0
    output_keys: list[str] | None = None
    _obs_names: list[str] | None = field(default=None, init=False, repr=False)

    @property
    def parameter_names(self) -> list[str]:
        """UQ parameter names (for Sobol index labeling)."""
        return self.param_space.parameter_names

    @property
    def obs_names(self) -> list[str]:
        """Observable names determined after the first simulation run."""
        if self._obs_names is None:
            raise RuntimeError("Observable names not yet known. Run at least one simulation to populate obs_names.")
        return self._obs_names

    def _apply_parameter_mutations(self, x: np.ndarray) -> SimulationDataEcoli:
        """Deep-copy baseline sim_data and apply variant mutations from x.

        Flow:
            x (numpy array)
            -> param_space.sample_to_params(x) -> UQInputParametersVecoli
            -> .to_simulation_config() -> {"variants": {name: [params]}, ...}
            -> _apply_variants(sim_data, variants) using real apply_variant()
        """
        sd = copy.deepcopy(self.baseline_sim_data)

        # x -> UQInputParametersVecoli -> simulation config with variants
        uq_params = self.param_space.sample_to_params(x)
        sim_config = uq_params.to_simulation_config()
        variants = sim_config.get("variants", {})

        if variants:
            sd = _apply_variants(sd, variants)

        return sd

    def _run_subprocess(
        self,
        x: np.ndarray,
        work_dir: Path | None = None,
        experiment_id: str = "uq_single",
        variant_index: int = 0,
    ) -> tuple[np.ndarray, list[str]]:
        """Run a single simulation as a subprocess and return timeseries.

        Args:
            x: Parameter vector, shape ``(n_params,)``.
            work_dir: Working directory. If None, creates a temp dir.
            experiment_id: Experiment ID for Parquet output structure.
            variant_index: Variant index for hive partitioning.

        Returns:
            Tuple of (timeseries array, observable names).
        """
        cleanup = work_dir is None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="uq_sim_"))

        try:
            # 1. Apply parameter mutations and pickle variant sim_data
            sd = self._apply_parameter_mutations(x)
            pickle_path = work_dir / "sim_data.cPickle"
            with open(pickle_path, "wb") as f:
                pickle.dump(sd, f)

            # 2. Build config JSON
            output_dir = work_dir / "output"
            output_dir.mkdir(exist_ok=True)
            config = _build_sim_config(
                sim_data_path=str(pickle_path),
                experiment_id=experiment_id,
                output_dir=str(output_dir),
                max_duration=self.max_duration,
                variant_index=variant_index,
                base_config_path=self.sim_config_path,
            )
            config_path = work_dir / "config.json"
            config_path.write_text(_json.dumps(config, indent=2))

            # 3. Run simulation as subprocess
            _run_sim_subprocess(config_path)

            # 4. Read Parquet output
            parquet_cols = _resolve_parquet_columns(self.output_keys)
            timeseries, obs_names = _read_parquet_timeseries(
                output_dir=output_dir,
                experiment_id=experiment_id,
                observable_columns=parquet_cols,
            )

            return timeseries, obs_names

        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run simulation for a single parameter vector.

        Args:
            x: Parameter values, shape ``(n_params,)``.

        Returns:
            Raw timeseries array of shape ``(n_timesteps, n_obs)``.
            This is the format expected by ``Strategy4Wrapper`` in Phase 2.
        """
        timeseries, obs_names = self._run_subprocess(x)
        self._obs_names = obs_names
        return timeseries

    def _run_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
        batch_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """Run simulations for a batch of parameter vectors via subprocesses.

        For each row in X:
        1. Deep-copy baseline sim_data and apply variant mutations
        2. Pickle variant sim_data to ``{batch_dir}/sim_data/{i}.cPickle``
        3. Write a per-sample JSON config
        4. Run ``ecoli_master_sim.py`` as a subprocess with Parquet emitter
        5. Collect Parquet output

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Max parallel subprocesses. None = sequential.
            batch_dir: Working directory for batch artifacts. If None,
                a temporary directory is created and cleaned up after.

        Returns:
            Tuple of:
            - Y_aggregated: (n_samples, n_outputs) time-mean array
            - Y_timeseries: list of per-sample raw timeseries arrays
        """
        cleanup = batch_dir is None
        if batch_dir is None:
            batch_dir = Path(tempfile.mkdtemp(prefix="uq_batch_"))

        n_samples = X.shape[0]
        sim_data_dir = batch_dir / "sim_data"
        configs_dir = batch_dir / "configs"
        output_dir = batch_dir / "output"
        sim_data_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_id = "uq_batch"
        parquet_cols = _resolve_parquet_columns(self.output_keys)

        try:
            # --- Phase A: Create all variant pickles and configs ---
            config_paths: list[Path] = []
            for i in range(n_samples):
                sd = self._apply_parameter_mutations(X[i])
                pickle_path = sim_data_dir / f"{i:04d}.cPickle"
                with open(pickle_path, "wb") as f:
                    pickle.dump(sd, f)

                config = _build_sim_config(
                    sim_data_path=str(pickle_path),
                    experiment_id=experiment_id,
                    output_dir=str(output_dir),
                    max_duration=self.max_duration,
                    variant_index=i,
                    seed=i,
                    base_config_path=self.sim_config_path,
                )
                config_path = configs_dir / f"{i:04d}.json"
                config_path.write_text(_json.dumps(config, indent=2))
                config_paths.append(config_path)

            logger.info(
                "Created %d variant pickles and configs in %s",
                n_samples, batch_dir,
            )

            # --- Phase B: Run simulations ---
            if max_workers is not None and max_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                errors: list[tuple[int, Exception]] = []
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_idx = {
                        pool.submit(_run_sim_subprocess, config_paths[i]): i
                        for i in range(n_samples)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            future.result()
                            logger.info("Sample %d/%d complete", idx + 1, n_samples)
                        except Exception as e:
                            errors.append((idx, e))
                            logger.error("Sample %d failed: %s", idx, e)

                if errors:
                    failed = [str(i) for i, _ in errors]
                    raise RuntimeError(
                        f"Batch simulation failed for samples: {', '.join(failed)}. "
                        f"First error: {errors[0][1]}"
                    )
            else:
                for i in range(n_samples):
                    logger.info("Running sample %d/%d...", i + 1, n_samples)
                    _run_sim_subprocess(config_paths[i])

            # --- Phase C: Collect Parquet outputs ---
            # All samples write to the same output_dir with different
            # variant indices. The hive structure is:
            # {output_dir}/{experiment_id}/history/experiment_id={id}/variant={i}/...
            history_base = output_dir / experiment_id / "history"

            pq_files = list(history_base.rglob("*.pq"))
            if not pq_files:
                pq_files = list(history_base.rglob("*.parquet"))
            if not pq_files:
                raise FileNotFoundError(
                    f"No Parquet files found under {history_base}"
                )

            df = pl.read_parquet(
                [str(p) for p in pq_files],
                hive_partitioning=True,
            )

            # Filter to available observable columns
            available = [c for c in parquet_cols if c in df.columns]
            if not available:
                raise ValueError(
                    f"None of {parquet_cols} found in Parquet columns: "
                    f"{df.columns[:20]}..."
                )

            parquet_to_short = {v: k for k, v in _SHORT_TO_PARQUET.items()}
            obs_names = [parquet_to_short.get(c, c) for c in available]
            self._obs_names = obs_names

            # Sort by time within each variant
            sort_cols = []
            if "variant" in df.columns:
                sort_cols.append("variant")
            if "time" in df.columns:
                sort_cols.append("time")
            if sort_cols:
                df = df.sort(sort_cols)

            Y_list: list[np.ndarray] = []
            Y_timeseries: list[np.ndarray] = []

            for i in range(n_samples):
                if "variant" in df.columns:
                    sample_df = df.filter(pl.col("variant") == i)
                else:
                    # Fallback: no variant column (shouldn't happen)
                    sample_df = df

                obs_df = sample_df.select(available).fill_null(0.0)
                ts_array = obs_df.to_numpy().astype(np.float64)
                Y_timeseries.append(ts_array)
                Y_list.append(ts_array.mean(axis=0))

            Y_aggregated = np.vstack(Y_list)

            # Write batch metadata for potential later collection
            meta = {
                "parameter_names": self.parameter_names,
                "n_samples": n_samples,
                "n_params": int(X.shape[1]),
                "observable_columns": available,
                "obs_names": obs_names,
                "bounds": np.array(self.param_space.parameter_bounds).tolist(),
                "samples": {
                    str(i): {"x": X[i].tolist()} for i in range(n_samples)
                },
            }
            (batch_dir / "metadata.json").write_text(
                _json.dumps(meta, indent=2)
            )

            return Y_aggregated, Y_timeseries

        finally:
            if cleanup:
                shutil.rmtree(batch_dir, ignore_errors=True)

    def evaluate_batch(
        self,
        X: np.ndarray,
        max_workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate simulation for a batch of parameter vectors.

        Args:
            X: Parameter array of shape ``(n_samples, n_params)``.
            max_workers: Max parallel subprocesses. None = sequential.

        Returns:
            Aggregated (time-mean) output array of shape
            ``(n_samples, n_outputs)``.
        """
        Y_agg, _ = self._run_batch(X, max_workers=max_workers)
        return Y_agg


# -- Batch config export / collection --------------------------------------


def export_batch_configs(
    sim_func: "TimeseriesGeneratorVecoli",
    X: np.ndarray,
    batch_dir: str | os.PathLike,
    base_config_path: str | None = None,
    generations: int = 1,
    emitter: str = "parquet",
) -> Path:
    """Convert LHS samples into per-sample vEcoli configs for Nextflow/HPC.

    For each row in *X*, this function:
    1. Converts x -> variant param dicts via ``param_space.sample_to_params()``
    2. Deep-copies baseline sim_data, applies variants, pickles to
       ``{batch_dir}/sim_data/{i}.cPickle``
    3. Writes a per-sample JSON config to ``{batch_dir}/configs/{i}.json``
    4. Writes ``{batch_dir}/metadata.json`` mapping sample indices to
       parameter vectors (for result collection)

    The resulting directory can be submitted to Nextflow:

    .. code-block:: bash

        nextflow run sim.nf --config_dir <batch_dir>/configs \\
                            --sim_data_dir <batch_dir>/sim_data

    Args:
        sim_func: VecoliSimulationFunc with baseline_sim_data and param_space.
        X: LHS sample array, shape ``(n_samples, n_params)``.
        batch_dir: Output directory for the batch.
        base_config_path: Optional JSON config template.  If None, a
            minimal config is generated.
        generations: Number of generations per sample sim.
        emitter: Emitter type (``"parquet"`` for Nextflow collection).

    Returns:
        Path to batch_dir (for chaining).
    """
    batch_dir = Path(batch_dir)
    config_dir = batch_dir / "configs"
    sim_data_dir = batch_dir / "sim_data"
    config_dir.mkdir(parents=True, exist_ok=True)
    sim_data_dir.mkdir(parents=True, exist_ok=True)

    # Load base config template if provided
    base_config: dict = {}
    if base_config_path is not None:
        base_config = _json.loads(Path(base_config_path).read_text())

    sample_metadata: dict = {
        "parameter_names": sim_func.parameter_names,
        "n_samples": int(X.shape[0]),
        "n_params": int(X.shape[1]),
        "bounds": np.array(sim_func.param_space.parameter_bounds).tolist(),
        "samples": {},
    }

    for i, x in enumerate(X):
        # x -> variant config
        uq_params = sim_func.param_space.sample_to_params(x)
        sim_config = uq_params.to_simulation_config()
        variants = sim_config.get("variants", {})

        # Deep-copy and apply variants to sim_data, then pickle
        sd = copy.deepcopy(sim_func.baseline_sim_data)
        if variants:
            sd = _apply_variants(sd, variants)

        pickle_path = sim_data_dir / f"{i}.cPickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(sd, f)

        # Build per-sample JSON config
        sample_config = {**base_config}
        sample_config["experiment_id"] = f"uq_sample_{i:04d}"
        sample_config["sim_data_path"] = str(pickle_path)
        sample_config["generations"] = generations
        sample_config["emitter"] = emitter
        sample_config["variants"] = {}  # already baked into sim_data

        config_path = config_dir / f"{i}.json"
        config_path.write_text(_json.dumps(sample_config, indent=2))

        sample_metadata["samples"][str(i)] = {
            "x": x.tolist(),
            "config": str(config_path),
            "sim_data": str(pickle_path),
        }

    (batch_dir / "metadata.json").write_text(_json.dumps(sample_metadata, indent=2))

    return batch_dir


def collect_batch_results(
    batch_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    observable_columns: list[str] | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> "PrecomputedCache":
    """Collect Parquet outputs from a completed batch run into PrecomputedCache.

    After Nextflow/HPC runs complete, each sample's output lives under
    ``{output_dir}/uq_sample_NNNN/``.  This function reads the hive-
    partitioned Parquet files, aggregates (time-mean), and assembles
    the ``(X, Y)`` cache.

    Args:
        batch_dir: Directory produced by :func:`export_batch_configs`
            (must contain ``metadata.json``).
        output_dir: Root directory containing per-sample Parquet outputs.
            Expected structure: ``{output_dir}/uq_sample_NNNN/history/...``
        observable_columns: Which columns to extract.  Defaults to
            mass listeners (dry_mass, cell_mass, volume, growth).
        cache_dir: Where to save the PrecomputedCache.  Defaults to
            ``{batch_dir}/cache``.

    Returns:
        PrecomputedCache with X, Y assembled from the batch outputs.
    """
    from uq.sampling import PrecomputedCache

    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)

    meta = _json.loads((batch_dir / "metadata.json").read_text())
    n_samples = meta["n_samples"]
    parameter_names = meta["parameter_names"]
    bounds = meta["bounds"]

    if observable_columns is None:
        observable_columns = list(_DEFAULT_PARQUET_COLUMNS)

    X_list = []
    Y_list = []
    Y_timeseries = []

    for i in range(n_samples):
        sample_meta = meta["samples"][str(i)]
        x = np.array(sample_meta["x"])
        X_list.append(x)

        # Read Parquet output for this sample
        experiment_id = f"uq_sample_{i:04d}"
        history_dir = output_dir / experiment_id / "history"

        if not history_dir.exists():
            raise FileNotFoundError(
                f"No output found for sample {i} at {history_dir}. Ensure Nextflow/HPC run completed successfully."
            )

        # Read hive-partitioned parquet
        pq_files = list(history_dir.rglob("*.pq"))
        if not pq_files:
            pq_files = list(history_dir.rglob("*.parquet"))

        df = pl.read_parquet(
            [str(p) for p in pq_files],
            hive_partitioning=True,
        )

        # Extract observable columns that exist in the data
        available = [c for c in observable_columns if c in df.columns]
        if not available:
            raise ValueError(f"None of {observable_columns} found in Parquet columns: {df.columns}")

        obs_df = df.select(available).fill_null(0.0)
        ts_array = obs_df.to_numpy()  # (n_rows, n_obs)
        Y_timeseries.append(ts_array)

        # Aggregate: time-mean
        y_mean = ts_array.mean(axis=0)
        Y_list.append(y_mean)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    if cache_dir is None:
        cache_dir = batch_dir / "cache"

    cache = PrecomputedCache(
        cache_dir=Path(cache_dir),
        X=X,
        Y=Y,
        parameter_names=parameter_names,
        metadata={"bounds": bounds, "seed": -1, "source": "batch_collection"},
        Y_timeseries=Y_timeseries,
    )
    cache.save()
    return cache


# -- Legacy config classes (kept for backward compatibility) -----------------


class NextflowProfile(StrEnum):
    STANDARD = "standard"
    AWS = "aws"
    CCAM = "ccam"


@dataclass
class OutputEmitterConfig(BaseClass):
    type: Literal["parquet", "timeseries", "xarray"] = "parquet"
    out_dir: str | None = None
    out_uri: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.out_uri is None and self.out_dir is None:
            self.out_uri = get_bucket()


@dataclass
class VariantVecoli(BaseClass):
    id: Literal["new_gene_internal_shift_variable_strength", "condition", "mecillinam_timeline"]
    config: dict[str, Any]


@dataclass
class SimulationConfigVecoli(SimulationConfig):
    """Vecoli simulation config (JSON), 1:1"""

    experiment_id: str
    sim_data_path: str | None = None
    n_init_sims: int = field(default=1)
    generations: int = field(default=1)
    variants: list[VariantVecoli] = field(default_factory=list)
    emitter_arg: OutputEmitterConfig = field(default=OutputEmitterConfig)

    def validate(self) -> bool:
        return True

    def model_dump(self) -> dict[str, Any]:
        attrs = ["experiment_id", "sim_data_path", "n_init_sims", "generations"]
        config = dict(zip(attrs, [getattr(self, attr) for attr in attrs]))
        config.update(self._format_emitter())
        config.update({"variants": {variant.id: variant.config for variant in self.variants}})
        return config

    def _format_emitter(self) -> dict[str, Any]:
        outdir_key = "out_dir"
        path = self.emitter_arg.out_dir or self.emitter_arg.out_uri
        if path == self.emitter_arg.out_uri:
            outdir_key = "out_uri"

        return {"emitter": self.emitter_arg.type, "emitter_arg": {outdir_key: path}}


# -- Example / smoke test ---------------------------------------------------


def example_simulation_func(
    sim_data_path: str,
    max_duration: float = 10.0,
    include_vio: bool = False,
    include_mecillinam: bool = False,
) -> dict[str, Any]:
    """End-to-end example: load sim_data, build f(x)->y, run one sample.

    Demonstrates the full mutation -> simulate -> extract chain using
    subprocess-based execution (no in-process EcoliSim).

    Args:
        sim_data_path: Path to a simData.cPickle file.
        max_duration: Short duration for testing (default 10s).
        include_vio: Include vio pathway parameters.
        include_mecillinam: Include mecillinam parameters.

    Returns:
        Dict with keys:
            sim_func: TimeseriesGeneratorVecoli instance (pass to pipeline)
            param_space: XSpaceVecoli
            x_sample: the parameter vector used
            uq_params: UQInputParametersVecoli from sample_to_params
            sim_config: variant config dict sent to apply_variant
            timeseries: (n_timesteps, n_obs) raw output from __call__
            aggregated: (n_obs,) time-mean output from evaluate_batch
            obs_names: list of observable column names

    Example::

        from uq.generators.vecoli import example_simulation_func

        result = example_simulation_func(
            sim_data_path="/path/to/vEcoli/out/parca/kb/simData.cPickle",
            max_duration=10.0,
        )

        # What the pipeline receives:
        sim_func = result["sim_func"]       # pass to run_phase1 / run_phase2
        param_space = result["param_space"] # pass to SensitivityAnalyzer

        # Phase 1 interface (SensitivityAnalyzer calls this):
        X = np.random.uniform(size=(5, param_space.n_parameters))
        Y = sim_func.evaluate_batch(X)   # (5, n_obs)

        # Phase 2 interface (Strategy4Wrapper calls this):
        ts = sim_func(X[0])              # (n_timesteps, n_obs)
    """
    from uq.pipeline.param_loader import ParameterDataset

    # --- 1. Load baseline sim_data and build parameter space ---
    ds = ParameterDataset(sim_data_path=sim_data_path)
    param_space = ds.to_parameter_space(
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
    )

    print(f"Parameter space: {param_space.parameter_names}")
    print(f"Bounds: {param_space.parameter_bounds}")

    if param_space.n_parameters == 0:
        raise ValueError(
            "Parameter space is empty. Set include_vio=True and/or "
            "include_mecillinam=True, or use sim_data that contains "
            "new gene entries (violacein)."
        )

    # --- 2. Instantiate TimeseriesGeneratorVecoli ---
    sim_func = TimeseriesGeneratorVecoli(
        baseline_sim_data=ds.sim_data,
        param_space=param_space,
        max_duration=max_duration,
    )

    # --- 3. Create a sample x vector (midpoint of bounds) ---
    bounds = param_space.bounds_array
    x_sample = (bounds[:, 0] + bounds[:, 1]) / 2.0
    print(f"Sample x: {dict(zip(param_space.parameter_names, x_sample))}")

    # --- 4. Trace the mutation chain (without running sim) ---
    uq_params = param_space.sample_to_params(x_sample)
    sim_config = uq_params.to_simulation_config()
    print(f"Variant config: {sim_config.get('variants', {})}")

    # --- 5. Run single-cell simulation via __call__ (Phase 2 interface) ---
    print(f"Running vEcoli subprocess (max_duration={max_duration}s)...")
    timeseries = sim_func(x_sample)  # (n_timesteps, n_obs)
    print(f"Timeseries shape: {timeseries.shape}")
    print(f"Observable names: {sim_func.obs_names}")

    # --- 6. Run evaluate_batch (Phase 1 interface) ---
    X_batch = x_sample.reshape(1, -1)  # (1, n_params)
    Y_aggregated = sim_func.evaluate_batch(X_batch)  # (1, n_obs)
    print(f"Aggregated output shape: {Y_aggregated.shape}")
    print(f"Aggregated values: {dict(zip(sim_func.obs_names, Y_aggregated[0]))}")

    return {
        "sim_func": sim_func,
        "param_space": param_space,
        "x_sample": x_sample,
        "uq_params": uq_params,
        "sim_config": sim_config,
        "timeseries": timeseries,
        "aggregated": Y_aggregated[0],
        "obs_names": sim_func.obs_names,
    }
