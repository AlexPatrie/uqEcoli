"""
Uncertainty Quantification framework execution pipeline (as proposed by RFC006)

Workflow:
      Inputs: experiment_id: str, hpc_sim_base_path: Path, param_space: XSpaceVecoli

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

import copy
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars
from pydantic import BaseModel, ConfigDict
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

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
from uq.common import BaseClass, get_repo_root
from uq.generators.vecoli import VecoliSimulationFunc
from uq.inputs import XSpace, XSpaceInterface
from uq.pce.models import PCEParameterSelectionConfig
from uq.pce.surrogate import generate_surrogate as _generate_surrogate
from uq.pipeline.models import PipelineResult, StratificationLens, UqProfile
from uq.pipeline.output_loader import OutputVariables, TimeseriesDataset, TimeseriesLoaderParquet, load_timeseries
from uq.pipeline.workflow import (
    AggregationResult,
    aggregate_timeseries,
    get_variance_decomposition,
    run_phase1,
    run_phase2,
)
from uq.sampling import PrecomputedCache
from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices
from uq.wrappers import DataDrivenWrapper

if TYPE_CHECKING:
    from uq.sensitivity import PCESurrogate

from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

from uq.models import PipelineConfig
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
        y: polars.DataFrame — timeseries data from load_timeseries()
        observables: list[str] — resolved observable column names
    """

    experiment_ids: list[str]
    x: list[ParameterDataset]
    y: polars.DataFrame
    parameter_space: XSpace = field(init=False)
    observables: list[str] = field(init=False)

    _METADATA_COLS: set[str] = field(
        default_factory=lambda: {"experiment_id", "variant", "lineage_seed", "generation", "agent_id", "time"},
        init=False,
        repr=False,
    )

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

        # Resolve observable columns from the timeseries DataFrame
        self.observables = [c for c in self.y.columns if c not in self._METADATA_COLS]


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
        DatasetMultiExperiment with parameter datasets and timeseries.
    """
    # Normalize to list BEFORE iterating
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    sim_data_paths = [
        (expid, (Path(sim_base_path) / expid / "parca" / "kb" / "simData.cPickle")) for expid in experiment_ids
    ]
    # --- Step 1: Build parameter space from sim_data ---
    parameter_datasets = [ParameterDataset(sim_data_path=p, experiment_id=expid) for expid, p in sim_data_paths]

    # --- Step 2: Load simulation data via DuckDB + OutputExtractor ---
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
        "mecillinam",
        "test_violacein_with_metabolism",
    ]
    base_path = Path("/Users/alexanderpatrie/sms/vEcoli-private/api_integration/sims")
    ds = initialize_data(experiment_ids=experiments, sim_base_path=base_path)
    print()


@dataclass
class System(BaseClass):
    dataset: DatasetMultiExperiment
    observable_cols: list[str]
    simulation_func: VecoliSimulationFunc | DataDrivenWrapper
    aggregation: AggregationResult
    decomposition: dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]]
    baseline_sim_data: SimulationDataEcoli | None = None


@dataclass
class Pipeline(BaseClass):
    """
    Object which represents a full ./uq workflow.

    Attributes:
        experiment_ids: list[str]
        sim_base_path: str | Path
        observable_columns: list[str] | None = None
        lb_generation: int | None = 2
        lb_time: float | None = 100.0
        n_bins: int = 10
        polynomial_order: int = 3
        n_samples: int = 200
        expected_cycle_time: float = 3600.0
        max_duration: float = 10800.0
        prescreen_config: PCEParameterSelectionConfig | None = None
        export_path: Path | None = None
        precomputed_path: Path | str | None = None
        sim_config_path: str | None = None
        init: bool = True
        cache: PrecomputedCache | None = field(init=False, default=None)
        system: System | None = field(init=False, default=None)
        result: PipelineResult | None = field(init=False, default=None)
        _ds: DatasetMultiExperiment | None = field(init=False, default=None)
    """

    experiment_ids: list[str]
    sim_base_path: str | Path
    observable_columns: list[str] | None = None
    lb_generation: int | None = 2
    lb_time: float | None = 100.0
    n_bins: int = 10
    polynomial_order: int = 3
    n_samples: int = 200
    expected_cycle_time: float = 3600.0
    max_duration: float = 10800.0
    prescreen_config: PCEParameterSelectionConfig | None = None
    export_path: Path | None = None
    precomputed_path: Path | str | None = None
    sim_config_path: str | None = None
    init: bool = True
    cache: PrecomputedCache | None = field(init=False, default=None)
    system: System | None = field(init=False, default=None)
    result: PipelineResult | None = field(init=False, default=None)
    _ds: DatasetMultiExperiment | None = field(init=False, default=None)

    def __post_init__(self):
        self.initialize()
        if self.system is None:
            warnings.warn("Warning: You must first run initialize_system before use!")

    def run(self):
        """
        Execute full pipeline initialization and gsa execution
        """
        self.result = self.gsa()

    def initialize(self) -> None:
        ds = self._initialize_data()
        if self.init:
            self.initialize_system(ds)
            del self._ds
        else:
            self.init = True

    def _initialize_data(self) -> tuple[DatasetMultiExperiment]:
        # --- Step 1: Load x and y for given experiment ids ---
        dataset = initialize_data(
            experiment_ids=self.experiment_ids,
            sim_base_path=self.sim_base_path,
            observable_columns=self.observable_columns,
            generation_lower_bound=self.lb_generation,
            time_lower_bound=self.lb_time,
        )
        # param_space: XSpace = ds.parameter_space
        # timeseries: polars.DataFrame = ds.y

        # Resolve observable columns from loaded data if not provided
        obs_cols: list[str] = self.observable_columns if self.observable_columns is not None else dataset.observables
        self.observable_columns = obs_cols

        # --- Resolve simulation data source ---
        cache = None
        if self.precomputed_path is not None:
            from uq.sampling import PrecomputedCache

            cache = PrecomputedCache.load(self.precomputed_path)
        self._ds = dataset
        self.cache = cache
        return dataset

    def initialize_system(self, ds: DatasetMultiExperiment | None = None):
        if ds is None:
            ds = copy.deepcopy(self._ds)

        if ds is None:
            raise ValueError()

        obs_cols: list[str] = self.observable_columns

        # --- Steps 3-4: Aggregation + Variance Decomposition (shared) ---
        agg_result: AggregationResult = aggregate_timeseries(timeseries=ds.y, observable_columns=obs_cols)
        decomp: dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]] = get_variance_decomposition(agg_result)

        # --- Instantiate simulation function from loaded sim_data ---
        # ds.x is list[ParameterDataset], each with a .sim_data attribute.
        # For live simulation mode, use the first experiment's sim_data as
        # the baseline (single-cell runs are per-experiment).
        simulation_func: Any = None
        baseline_sim_data: SimulationDataEcoli | None = None
        if self.cache is None:
            baseline_sim_data = next(filter(lambda ds_i: "baseline" in ds_i.experiment_id, ds.x))
            baseline_sim_data = ds.x[0].sim_data if ds.x else None
            if baseline_sim_data is not None:
                simulation_func = VecoliSimulationFunc(
                    baseline_sim_data=baseline_sim_data,
                    param_space=ds.parameter_space,
                    sim_config_path=self.sim_config_path,
                    max_duration=self.max_duration,
                    output_keys=obs_cols,
                )
            else:
                # Fallback: synthetic response surface for demos
                simulation_func = DataDrivenWrapper(
                    parameter_space=ds.parameter_space,
                    observable_means=agg_result.uniform.mean,
                    observable_stds=agg_result.uniform.std,
                )
        self.system = System(
            dataset=ds,
            observable_cols=obs_cols,
            simulation_func=simulation_func,
            baseline_sim_data=baseline_sim_data,
            aggregation=agg_result,
            decomposition=decomp,
        )

    def gsa(self) -> PipelineResult:
        sobol_bulk, surrogate_bulk, morris_indices = self.gsa_bulk()
        per_stage_sobol, surrogate_cc, cc_relevance = self.gsa_cell(sobol_bulk, surrogate_bulk, morris_indices)
        return self.assemble_results(
            sobol_bulk, surrogate_bulk, morris_indices, per_stage_sobol, surrogate_cc, cc_relevance
        )

    def gsa_bulk(self) -> tuple[SobolIndices, PCESurrogate, MorrisIndices | None]:
        if self.system is None:
            raise RuntimeError("First run .initialize_system()!")

        param_space = self.system.dataset.parameter_space
        f = self.system.simulation_func

        # --- Phase 1 then Phase 2 (sequential to avoid OOM) ---
        return run_phase1(
            param_space=param_space,
            simulation_func=f,
            polynomial_order=self.polynomial_order,
            n_samples=self.n_samples,
            prescreen_config=self.prescreen_config if self.cache is None else None,
            export_path=self.export_path,
            precomputed_samples=self.cache.X if self.cache else None,
            precomputed_outputs=self.cache.Y if self.cache else None,
        )

    def gsa_cell(
        self, sobol_bulk, surrogate_bulk, morris_indices
    ) -> tuple[list[SobolIndices], PCESurrogate, CellCycleRelevanceResult]:
        param_space = self.system.dataset.parameter_space
        f = self.system.simulation_func
        return run_phase2(
            param_space=param_space,
            simulation_func=f,
            agg_result=self.system.aggregation,
            observable_names=self.system.observable_cols,
            n_bins=self.n_bins,
            polynomial_order=self.polynomial_order,
            n_samples=self.n_samples,
            expected_cycle_time=self.expected_cycle_time,
            export_path=self.export_path,
            precomputed_samples=self.cache.X if self.cache else None,
            precomputed_timeseries=self.cache.Y_timeseries if self.cache else None,
        )

    def assemble_results(
        self, sobol_bulk, surrogate_bulk, morris_indices, per_stage_sobol, surrogate_cc, cc_relevance
    ) -> PipelineResult:
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
            variance_decomposition=self.system.decomposition,
            aggregation=self.system.aggregation,
            morris_indices=morris_indices,
            cell_cycle_relevance=cc_relevance,
        )
        if self.export_path is not None:
            result.export(self.export_path)
        return result

    def _gsa(self) -> PipelineResult:
        param_space = self.system.dataset.parameter_space
        f = self.system.simulation_func

        # --- Phase 1 then Phase 2 (sequential to avoid OOM) ---
        sobol_bulk, surrogate_bulk, morris_indices = run_phase1(
            param_space=param_space,
            simulation_func=f,
            polynomial_order=self.polynomial_order,
            n_samples=self.n_samples,
            prescreen_config=self.prescreen_config if self.cache is None else None,
            export_path=self.export_path,
            precomputed_samples=self.cache.X if self.cache else None,
            precomputed_outputs=self.cache.Y if self.cache else None,
        )
        per_stage_sobol, surrogate_cc, cc_relevance = run_phase2(
            param_space=param_space,
            simulation_func=f,
            agg_result=self.system.aggregation,
            observable_names=self.system.observable_cols,
            n_bins=self.n_bins,
            polynomial_order=self.polynomial_order,
            n_samples=self.n_samples,
            expected_cycle_time=self.expected_cycle_time,
            export_path=self.export_path,
            precomputed_samples=self.cache.X if self.cache else None,
            precomputed_timeseries=self.cache.Y_timeseries if self.cache else None,
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
            variance_decomposition=self.system.decomposition,
            aggregation=self.system.aggregation,
            morris_indices=morris_indices,
            cell_cycle_relevance=cc_relevance,
        )

        if self.export_path is not None:
            result.export(self.export_path)

        return result


def execute_pipeline(
    experiment_ids: str | list[str],
    sim_base_path: str | Path,
    observable_columns: list[str] | None = None,
    lb_generation: int | None = 2,
    lb_time: float | None = 100.0,
    n_bins: int = 10,
    polynomial_order: int = 3,
    n_samples: int = 200,
    expected_cycle_time: float = 3600.0,
    max_duration: float = 10800.0,
    prescreen_config: PCEParameterSelectionConfig | None = None,
    export_path: Path | None = None,
    precomputed_path: Path | str | None = None,
    sim_config_path: str | None = None,
) -> PipelineResult:
    """
    Execute the RFC006 UQ pipeline.

    The simulation function ``f(x) -> y`` is constructed internally from
    the ``SimulationDataEcoli`` pickle(s) found at
    ``{sim_base_path}/{experiment_id}/parca/kb/simData.cPickle`` and the
    parameter space derived from those pickles.  Callers do not need to
    provide a simulation function.

    Supports three modes for Phase 1/2 evaluation:

    1. **Live simulation** (default): instantiates a
       ``VecoliSimulationFunc`` that runs single-cell ``EcoliSim`` with
       the timeseries emitter for each LHS sample.
    2. **precomputed_path provided**: loads cached (X, Y) from a prior
       ``uq generate-samples`` run — no simulation calls needed.
    3. **Neither live nor precomputed and no sim_data available**: builds
       a ``DataDrivenWrapper`` (synthetic linear response surface from
       aggregated statistics) for quick demos.

    Args:
        experiment_ids: Experiment identifier(s) for data loading.
        sim_base_path: Root directory containing simulation outputs.
        observable_columns: Column names of observables to aggregate/analyze.
        lb_generation: Skip initial generations (default: 2).
        lb_time: Skip transient period in seconds (default: 100.0).
        n_bins: Number of cell cycle stage bins for Phase 2.
        polynomial_order: PCE polynomial order for both phases.
        n_samples: Number of LHS samples for PCE fitting.
        expected_cycle_time: Expected cell cycle period in seconds.
        max_duration: EcoliSim max_duration in seconds for live mode.
        prescreen_config: Optional Morris prescreening config for Phase 1.
        export_path: If provided, export surrogates and results here.
        precomputed_path: Path to cached (X, Y) from ``generate-samples``.
            When provided, PCE is fit directly to cached data.
        sim_config_path: Optional path to EcoliSim JSON config file.
            If None, uses the vEcoli default config.

    Returns:
        PipelineResult with all RFC006 pipeline outputs.
    """
    from uq.generators.vecoli import VecoliSimulationFunc
    from uq.wrappers import DataDrivenWrapper

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

    # Resolve observable columns from loaded data if not provided
    obs_cols = observable_columns if observable_columns is not None else ds.observables

    # --- Steps 3-4: Aggregation + Variance Decomposition (shared) ---
    agg_result = aggregate_timeseries(timeseries, obs_cols)
    decomp = get_variance_decomposition(agg_result)

    # --- Resolve simulation data source ---
    cache = None
    if precomputed_path is not None:
        from uq.sampling import PrecomputedCache

        cache = PrecomputedCache.load(precomputed_path)

    # --- Instantiate simulation function from loaded sim_data ---
    # ds.x is list[ParameterDataset], each with a .sim_data attribute.
    # For live simulation mode, use the first experiment's sim_data as
    # the baseline (single-cell runs are per-experiment).
    simulation_func: Any = None
    if cache is None:
        baseline_sim_data = ds.x[0].sim_data if ds.x else None
        if baseline_sim_data is not None:
            simulation_func = VecoliSimulationFunc(
                baseline_sim_data=baseline_sim_data,
                param_space=param_space,
                sim_config_path=sim_config_path,
                max_duration=max_duration,
                output_keys=obs_cols if observable_columns is not None else None,
            )
        else:
            # Fallback: synthetic response surface for demos
            simulation_func = DataDrivenWrapper(
                parameter_space=param_space,
                observable_means=agg_result.uniform.mean,
                observable_stds=agg_result.uniform.std,
            )

    # --- Phase 1 then Phase 2 (sequential to avoid OOM) ---
    sobol_bulk, surrogate_bulk, morris_indices = run_phase1(
        param_space=param_space,
        simulation_func=simulation_func,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
        prescreen_config=prescreen_config if cache is None else None,
        export_path=export_path,
        precomputed_samples=cache.X if cache else None,
        precomputed_outputs=cache.Y if cache else None,
    )

    per_stage_sobol, surrogate_cc, cc_relevance = run_phase2(
        param_space=param_space,
        simulation_func=simulation_func,
        agg_result=agg_result,
        observable_names=obs_cols,
        n_bins=n_bins,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
        expected_cycle_time=expected_cycle_time,
        export_path=export_path,
        precomputed_samples=cache.X if cache else None,
        precomputed_timeseries=cache.Y_timeseries if cache else None,
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
