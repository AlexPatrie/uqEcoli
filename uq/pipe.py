"""
Uncertainty Quantification framework execution pipeline (as proposed by RFC006)

Workflow:
      Inputs: experiment_id: str, hpc_sim_base_path: Path, param_space: XSpaceVecoli, f: Callable[[np.ndarray], np.ndarray]

      1. Define Parameter Space — param_space = XSpaceVecoli(include_vio=True, include_mecillinam=True) → n parameters with bounds
      2. Load Simulation Data — df = load_dataset(experiment_id, hpc_sim_base_path) → Polars DataFrame from hive-partitioned Parquet
      3. Aggregation Strategies 1-3 — aggregator runs 3 strategies → 3 × AggregatedOutput
        - 3a. Strategy 1: UNIFORM — mean, std across all cells/times
        - 3b. Strategy 2: BY_GENERATION — per-gen stats (convergence)
        - 3c. Strategy 3: BY_LINEAGE_SEED — per-seed stats (exogenous variance)
      4. Variance Decomposition — compute_variance_decomposition(agg_uniform, agg_by_gen, agg_by_seed) → per observable:
         generation_fraction, seed_fraction, residual_fraction (cell-cycle-related, key output)

      Branches into Phase 1 and Phase 2 (parallel):

      Phase 1 (Strategies 1-3 GSA):

      5a. Morris Prescreening → n params → K params (K << n) → selected: list[Parameter], MorrisIndices
      6a. PCE Surrogate (Strategies 1-3) → PCESurrogate, PCEFitResult
      7a. Sobol Indices (from PCE) — S_i, S_Ti from PCE coefficients → SobolIndices

      Phase 2 (Strategy 4 — Cell Cycle):

      5b. GSA-Informed Observable Selection — identify_cell_cycle_relevant_observables(decomp)
      6b. Koopman DMD — KoopmanCellCycleVariable(observable_columns=relevant_obs) → θ(x) ∈ [0, 1]
      6c. Strategy 4 Aggregation — CellCycleAggregator — bin by θ → per-stage mean, std
      6d. Strategy4Wrapper — f_stage4(params): raw = f(params), θ = koopman(raw), bin by θ, return stage_means
      7b. PCE + Sobol on Strategy 4 → per-stage PCESurrogate, list[SobolIndices]

      Phases converge:

      Pipeline Outputs:
      - From Phase 1: AggregatedOutput × 3, variance decomposition, MorrisIndices, PCESurrogate (bulk), SobolIndices (bulk)
      - From Phase 2: CellCycleResult (θ, stages), per-stage statistics, PCESurrogate (phenotypic), list[SobolIndices] (phenotypic)
      - Feedback loop: Step 4 residual_fraction → Step 5b observable selection → Step 6b Koopman

      Final user-facing outputs:
        1. Phase 1 Sobol: "vio_expression drives 60% of bulk mass variance, mecillinam_conc drives 25%, ..."
        2. Phase 2 Sobol: "During C-period (DNA replication), mecillinam_conc drives 80% of variance; during D-period, vio_expression dominates"

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  PIPELINE OUTPUTS                                                                   │
  │                                                                                     │
  │  From Phase 1 (Strategies 1-3):              From Phase 2 (Strategy 4):             │
  │  ├─ AggregatedOutput × 3                     ├─ CellCycleResult (θ, stages)         │
  │  ├─ variance decomposition (fractions)       ├─ per-stage statistics                │
  │  ├─ MorrisIndices (parameter screening)      ├─ PCESurrogate (phenotypic)           │
  │  ├─ PCESurrogate (bulk)                      └─ list[SobolIndices] (phenotypic)     │
  │  └─ SobolIndices (bulk)                                                             │
  │                                                                                     │
  │  ┌───────────────────────────────────────────────────────────────────────────┐       │
  │  │ FEEDBACK LOOP (variance decomp → Phase 2):                               │       │
  │  │                                                                           │       │
  │  │  Step 4 residual_fraction ──► Step 5b observable selection ──► Step 6b   │       │
  │  │                                                                           │       │
  │  │  "Strategies 1-3 tell you WHICH observables to give to Koopman"          │       │
  │  └───────────────────────────────────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import polars
import pytest

from uq import (
    AggregatedOutput,
    SensitivityAnalyzer,
    SobolIndices,
    XSpaceVecoli,
    compute_variance_decomposition,
    identify_cell_cycle_relevant_observables,
)
from uq import (
    calculate_cell_cycle as _cell_cycle,
)
from uq.inputs import XSpace, XSpaceInterface
from uq.pce.models import PCEParameterSelectionConfig
from uq.pce.surrogate import generate_surrogate as _generate_surrogate
from uq.pipeline.models import PipelineConfig, PipelineResult, StratificationLens, UqProfile
from uq.pipeline.output_loader import OutputVariables, TimeseriesDataset, TimeseriesLoaderParquet, load_timeseries
from uq.pipeline.workflow import aggregate_timeseries, get_variance_decomposition, run_phase1, run_phase2
from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices

if TYPE_CHECKING:
    from uq.sensitivity import PCESurrogate

from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

from uq.outputs import OutputExtractor, OutputType
from uq.pipeline.param_loader import ParameterDataset

# === Pipeline Orchestration === #
"""
. Observable Selection (uq/sensitivity.py:858-928)

  identify_cell_cycle_relevant_observables currently ranks observables by residual variance fraction
  (greedy threshold). This is a natural graph partitioning problem:

  - Nodes = observables (e.g., mRNA counts, protein counts, fluxes)
  - Edges = co-variance or mutual information between observable pairs
  - Max-cut partitions into "cell-cycle-relevant" vs "non-relevant" sets, maximizing the total
  dissimilarity (edge weight) across the cut — ensuring the two groups are maximally distinct in their
  variance structure
"""


@dataclass
class DatasetMultiExperiment:
    """
    Attributes:
        experiment_ids: list[str]
        x: list[ParameterDataset]
    """

    experiment_ids: list[str]
    x: list[ParameterDataset]
    y: TimeseriesDataset
    parameter_space: XSpace = field(init=False)

    def __post_init__(self):
        parameter_datasets = self.x
        n_param_ds = len(parameter_datasets)
        if not n_param_ds == len(self.experiment_ids):
            raise ValueError(
                f"You must pass 1 parameter dataset for each experiment id. "
                f"Expected: {self.experiment_ids}, Got: {n_param_ds}"
            )
        if n_param_ds == 1:
            self.parameter_space = parameter_datasets[0].to_parameter_space()
        else:
            self.parameter_space = ParameterDataset.merge_to_parameter_space(*parameter_datasets)


def initialize_data(
    experiment_ids: str | list[str],
    sim_base_path: str | Path,
    observable_columns: list[str] | None = None,
    generation_lower_bound: int | None = 2,
    time_lower_bound: float | None = 100.0,
) -> DatasetMultiExperiment:
    """
    Initialize Datasets

    Args:
        experiment_ids: Experiment identifier(s) for data loading.
        sim_base_path: Root directory containing simulation outputs.
        observable_columns: Column names of observables to aggregate/analyze.
        generation_lower_bound: Skip initial generations (default: 2).
        time_lower_bound: Skip transient period in seconds (default: 100.0).

    Returns:
        PipelineResult with all RFC006 pipeline outputs.
    """
    sim_data_paths = [
        (expid, (Path(sim_base_path) / expid / "parca" / "kb" / "simData.cPickle")) for expid in experiment_ids
    ]
    # --- Step 1: Build parameter space from sim_data ---
    parameter_datasets = [ParameterDataset(sim_data_path=p, experiment_id=expid) for expid, p in sim_data_paths]

    if len(parameter_datasets) == 1:
        param_space = parameter_datasets[0].to_parameter_space()
    else:
        param_space = ParameterDataset.merge_to_parameter_space(*parameter_datasets)

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    # --- Step 2: Load simulation data via DuckDB + OutputExtractor ---
    # Load timeseries once, then derive typed outputs from it.
    timeseries_dataset = load_timeseries(
        sim_base_path=sim_base_path,
        experiment_ids=experiment_ids,
        observables=observable_columns,
        lb_generation=generation_lower_bound,
        lb_time=time_lower_bound,
    )

    return DatasetMultiExperiment(experiment_ids=experiment_ids, x=parameter_datasets, y=timeseries_dataset)


def test_initialize_data():
    experiments = [
        "api_simulation_default",
        # 'mecillinam',
        "test_violacein_with_metabolism",
    ]
    base_path = Path("/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims")
    ds = initialize_data(experiment_ids=experiments, sim_base_path=base_path)
    print()


async def pipeline(
    experiment_ids: str | list[str],
    sim_base_path: str | Path,
    simulation_func: Callable,
    observable_columns: list[str] | None = None,
    lb_generation: int | None = 2,
    lb_time: float | None = 100.0,
    n_bins: int = 10,
    polynomial_order: int = 3,
    n_samples: int = 200,
    expected_cycle_time: float = 3600.0,
    prescreen_config: PCEParameterSelectionConfig | None = None,
    export_path: Path | None = None,
) -> PipelineResult:
    """
    Async version of execute_pipeline — runs Phase 1 and Phase 2 concurrently.

    Same interface and outputs as execute_pipeline but uses asyncio to run
    the two independent phases in parallel after the shared steps 1-4.

    Args:
        simulation_func: Callable with evaluate_batch(X) → Y.
        experiment_ids: Experiment identifier(s) for data loading.
        sim_base_path: Root directory containing simulation outputs.
        observable_columns: Column names of observables to aggregate/analyze.
        lb_generation: Skip initial generations (default: 2).
        lb_time: Skip transient period in seconds (default: 100.0).
        n_bins: Number of cell cycle stage bins for Phase 2.
        polynomial_order: PCE polynomial order for both phases.
        n_samples: Number of LHS samples for PCE fitting.
        expected_cycle_time: Expected cell cycle period in seconds.
        prescreen_config: Optional Morris prescreening config for Phase 1.
        export_path: If provided, export surrogates and results here.

    Returns:
        PipelineResult with all RFC006 pipeline outputs.
    """
    # --- Step 1: Load x and y for given experiment ids ---
    ds = initialize_data(
        experiment_ids=experiment_ids,
        sim_base_path=sim_base_path,
        observable_columns=observable_columns,
        generation_lower_bound=lb_generation,
        time_lower_bound=lb_time,
    )
    param_space = ds.parameter_space
    timeseries = ds.y

    # --- Steps 3-4: Aggregation + Variance Decomposition (shared) ---
    agg_result = aggregate_timeseries(timeseries, observable_columns)
    decomp = get_variance_decomposition(agg_result)

    # --- Phase 1 || Phase 2 (parallel) ---
    loop = asyncio.get_running_loop()

    phase1_future = loop.run_in_executor(
        None,
        lambda: run_phase1(
            param_space=param_space,
            simulation_func=simulation_func,
            polynomial_order=polynomial_order,
            n_samples=n_samples,
            prescreen_config=prescreen_config,
            export_path=export_path,
        ),
    )

    phase2_future = loop.run_in_executor(
        None,
        lambda: run_phase2(
            param_space=param_space,
            simulation_func=simulation_func,
            agg_result=agg_result,
            observable_names=observable_columns,
            n_bins=n_bins,
            polynomial_order=polynomial_order,
            n_samples=n_samples,
            expected_cycle_time=expected_cycle_time,
            export_path=export_path,
        ),
    )

    (sobol_bulk, surrogate_bulk, morris_indices), (per_stage_sobol, surrogate_cc, cc_relevance) = await asyncio.gather(
        phase1_future, phase2_future
    )

    # --- Assemble PipelineResult ---
    result = PipelineResult(
        population=UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[sobol_bulk],
            surrogate=surrogate_bulk,
        ),
        cell_cycle=UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=per_stage_sobol,
            surrogate=surrogate_cc,
        ),
        variance_decomposition=decomp,
        aggregation=agg_result,
        morris_indices=morris_indices,
        cell_cycle_relevance=cc_relevance,
    )

    if export_path is not None:
        result.export(export_path)

    return result
