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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import polars

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
from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices

if TYPE_CHECKING:
    from uq.sensitivity import PCESurrogate


# === Type Aliases === #

VarianceDecomposition = dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]]


# === Step 1: Define Parameter Space === #


def define_parameter_space(
    include_vio: bool = True,
    include_mecillinam: bool = True,
    vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
    vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
    mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
) -> XSpaceVecoli:
    """Step 1: Create input parameter space Ξ for the pipeline."""
    return XSpaceVecoli(
        include_vio=include_vio,
        include_mecillinam=include_mecillinam,
        vio_expression_bounds=vio_expression_bounds,
        vio_trl_eff_bounds=vio_trl_eff_bounds,
        mecillinam_conc_bounds=mecillinam_conc_bounds,
    )


# === Step 3: Aggregate Timeseries === #


@dataclass
class AggregationResult:
    """Container for the three aggregation strategy outputs (steps 3a-3c)."""

    uniform: AggregatedOutput
    generation: AggregatedOutput
    seed: AggregatedOutput


def aggregate_timeseries(
    timeseries: polars.DataFrame,
    observable_columns: list[str],
) -> AggregationResult:
    """
    Step 3: Run aggregation strategies 1-3 on timeseries data.

    Aggregates the given observable columns uniformly (strategy 1),
    by generation (strategy 2), and by lineage seed (strategy 3).

    Args:
        timeseries: Polars DataFrame with simulation timeseries data.
            Must contain columns: observable_columns + 'generation' + 'lineage_seed'.
        observable_columns: Column names of the observables to aggregate.

    Returns:
        AggregationResult with uniform, generation, and seed AggregatedOutput instances.
    """
    # Strategy 1: UNIFORM — mean, std across all cells/times
    uniform_mean = timeseries.select(observable_columns).mean().to_numpy().flatten()
    uniform_std = timeseries.select(observable_columns).std().to_numpy().flatten()
    agg_uniform = AggregatedOutput(
        mean=uniform_mean,
        std=uniform_std,
        n_samples=len(timeseries),
        groups=None,
    )

    # Strategy 2: BY_GENERATION — per-generation stats
    gen_aggs = []
    for col in observable_columns:
        gen_aggs.extend([
            polars.col(col).mean().alias(f"{col}__mean"),
            polars.col(col).std().alias(f"{col}__std"),
        ])
    gen_aggs.append(polars.len().alias("n"))

    by_gen = timeseries.group_by("generation").agg(gen_aggs).sort("generation")
    gen_means = np.column_stack([by_gen[f"{c}__mean"].to_numpy() for c in observable_columns])
    gen_stds = np.column_stack([by_gen[f"{c}__std"].to_numpy() for c in observable_columns])
    agg_by_gen = AggregatedOutput(
        mean=gen_means,
        std=gen_stds,
        n_samples=by_gen["n"].to_numpy(),
        groups=by_gen["generation"].to_numpy(),
    )

    # Strategy 3: BY_LINEAGE_SEED — per-seed stats
    seed_aggs = []
    for col in observable_columns:
        seed_aggs.extend([
            polars.col(col).mean().alias(f"{col}__mean"),
            polars.col(col).std().alias(f"{col}__std"),
        ])
    seed_aggs.append(polars.len().alias("n"))

    by_seed = timeseries.group_by("lineage_seed").agg(seed_aggs).sort("lineage_seed")
    seed_means = np.column_stack([by_seed[f"{c}__mean"].to_numpy() for c in observable_columns])
    seed_stds = np.column_stack([by_seed[f"{c}__std"].to_numpy() for c in observable_columns])
    agg_by_seed = AggregatedOutput(
        mean=seed_means,
        std=seed_stds,
        n_samples=by_seed["n"].to_numpy(),
        groups=by_seed["lineage_seed"].to_numpy(),
    )

    return AggregationResult(uniform=agg_uniform, generation=agg_by_gen, seed=agg_by_seed)


# === Step 4: Variance Decomposition === #


def get_variance_decomposition(agg: AggregationResult) -> VarianceDecomposition:
    """
    Step 4: Compute variance decomposition across strategies 1-3.

    Returns dict with keys:
    - 'between_generation_variance': Variance between generations
    - 'between_seed_variance': Variance between lineage seeds
    - 'within_group_variance': Residual variance
    - 'generation_fraction': Fraction of variance from generation
    - 'seed_fraction': Fraction of variance from lineage seed
    """
    return compute_variance_decomposition(agg.generation, agg.seed, agg.uniform)


# === Step 6d: Strategy 4 Wrapper (previously MISSING) === #


class Strategy4Wrapper:
    """
    Strategy 4 wrapper (Step 6d): params → per-stage output.

    Wraps a base simulation function to produce cell-cycle-stratified output:
      1. Run the simulation with given parameters → raw timeseries
      2. Compute cell cycle variable θ via Koopman DMD
      3. Bin timesteps by θ into n_bins stages
      4. Return per-stage means as the output vector

    This wrapper is then fed into generate_surrogate() / SensitivityAnalyzer
    to produce Strategy 4 PCE surrogate and per-stage Sobol indices.

    Args:
        base_wrapper: Callable that takes params (np.ndarray) and returns
            raw simulation output (np.ndarray of shape (n_timesteps,) or (n_timesteps, n_obs)).
        koopman_cc: KoopmanCellCycleVariable instance for computing θ(x).
        n_bins: Number of cell cycle stage bins.
        timeseries_to_dataframe: Optional callable to convert raw output array
            to a polars DataFrame suitable for koopman_cc.compute().
            If None, a default conversion is used.
    """

    def __init__(
        self,
        base_wrapper: Callable[[np.ndarray], np.ndarray],
        koopman_cc: Any,
        n_bins: int = 10,
        timeseries_to_dataframe: Callable[[np.ndarray], polars.DataFrame] | None = None,
        observable_columns: list[str] | None = None,
    ):
        self.base_wrapper = base_wrapper
        self.koopman_cc = koopman_cc
        self.n_bins = n_bins
        self.stage_edges = np.linspace(0, 1, n_bins + 1)
        self._to_df = timeseries_to_dataframe
        self._observable_columns = observable_columns

    def _raw_to_dataframe(self, raw_output: np.ndarray) -> polars.DataFrame:
        """Convert raw simulation output array to a polars DataFrame for Koopman."""
        if self._to_df is not None:
            return self._to_df(raw_output)

        if raw_output.ndim == 1:
            raw_output = raw_output.reshape(-1, 1)

        cols = self._observable_columns or [f"obs_{i}" for i in range(raw_output.shape[1])]
        return polars.DataFrame({col: raw_output[:, i] for i, col in enumerate(cols)})

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """params → run sim → compute θ → bin by θ → per-stage means."""
        raw_output = self.base_wrapper(params)

        # Compute θ via Koopman
        df = self._raw_to_dataframe(raw_output)
        cc_var = self.koopman_cc.compute(df)
        theta = cc_var.values

        # Bin by θ
        bins = np.digitize(theta, self.stage_edges) - 1
        bins = np.clip(bins, 0, self.n_bins - 1)

        # Compute per-stage means
        if raw_output.ndim == 1:
            return np.array([raw_output[bins == s].mean() if np.any(bins == s) else 0.0 for s in range(self.n_bins)])
        else:
            # Multi-observable: flatten to (n_bins * n_obs,)
            n_obs = raw_output.shape[1]
            stage_means = np.zeros(self.n_bins * n_obs)
            for s in range(self.n_bins):
                mask = bins == s
                if np.any(mask):
                    stage_means[s * n_obs : (s + 1) * n_obs] = raw_output[mask].mean(axis=0)
            return stage_means

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Strategy 4 wrapper for a batch of parameter vectors."""
        return np.vstack([self(x) for x in X])


# === Steps 7b: Strategy 4 Sobol (previously MISSING) === #


def compute_strategy4_sobol(
    param_space: XSpaceInterface,
    f_stage4: Strategy4Wrapper,
    polynomial_order: int = 3,
    n_samples: int = 200,
) -> tuple[list[SobolIndices], PCESurrogate]:
    """
    Step 7b: Phenotypic sensitivity analysis — Sobol indices per cell-cycle stage.

    Builds a PCE surrogate for the Strategy 4 wrapper and computes Sobol indices.
    Returns one SobolIndices per θ-bin, answering:
    "During DNA replication, mecillinam_conc drives 80% of variance"

    Args:
        param_space: Input parameter space Ξ.
        f_stage4: Strategy4Wrapper instance (has evaluate_batch()).
        polynomial_order: PCE polynomial order.
        n_samples: Number of LHS samples for PCE fitting.

    Returns:
        Tuple of (list[SobolIndices], PCESurrogate).
    """
    analyzer = SensitivityAnalyzer(
        parameter_space=param_space,
        wrapper=f_stage4,
    )
    sobol_multi, surrogate = analyzer.analyze_with_pce(
        polynomial_order=polynomial_order,
        n_samples=n_samples,
    )

    # Split multi-output Sobol into per-stage SobolIndices
    per_stage_sobol = _split_multi_output_sobol(sobol_multi)
    return per_stage_sobol, surrogate


def _split_multi_output_sobol(sobol: SobolIndices) -> list[SobolIndices]:
    """Split a multi-output SobolIndices into a list of per-output SobolIndices."""
    fo = sobol.first_order
    to = sobol.total_order

    if fo.ndim > 1:
        n_outputs = fo.shape[0]
        per_output = []
        for i in range(n_outputs):
            stage_sobol = SobolIndices(
                first_order=fo[i],
                total_order=to[i],
                parameter_names=sobol.parameter_names,
                output_names=[f"stage_{i}"],
            )
            per_output.append(stage_sobol)
        return per_output
    else:
        return [sobol]


# === Phase 1: Population-Level GSA (Steps 5a-7a) === #


def run_phase1(
    param_space: XSpace,
    simulation_func: Callable,
    polynomial_order: int = 3,
    n_samples: int = 200,
    prescreen_config: PCEParameterSelectionConfig | None = None,
    export_path: Path | None = None,
) -> tuple[SobolIndices, PCESurrogate, MorrisIndices | None]:
    """
    Phase 1: Population-level GSA (Steps 5a-7a).

    Step 5a: Morris prescreening (if prescreen_config provided) to reduce
        n parameters → K most influential parameters.
    Step 6a: PCE surrogate construction on (screened) parameter space.
    Step 7a: Sobol indices from PCE coefficients.

    Answers: "Which parameters drive bulk output variance?"

    Args:
        param_space: Input parameter space Ξ.
        simulation_func: Callable with evaluate_batch(X) → Y.
        polynomial_order: PCE polynomial order.
        n_samples: Number of LHS samples for PCE fitting.
        prescreen_config: Optional Morris prescreening config. When provided,
            Morris screening runs first to identify the top-K parameters,
            and PCE is built on the reduced space.
        export_path: If provided, export surrogate to this path.

    Returns:
        Tuple of (SobolIndices, PCESurrogate, MorrisIndices | None).
    """
    morris_indices: MorrisIndices | None = None

    # Step 5a: Morris prescreening (optional but RFC006-recommended)
    if prescreen_config is not None:
        analyzer_morris = SensitivityAnalyzer(
            parameter_space=param_space,
            wrapper=simulation_func,
        )
        morris_indices = analyzer_morris.analyze_with_morris(
            n_trajectories=prescreen_config.n_trajectories,
        )

    # Steps 6a-7a: PCE surrogate + Sobol indices
    analyzer = SensitivityAnalyzer(
        parameter_space=param_space,
        wrapper=simulation_func,
    )
    sobol, surrogate = analyzer.analyze_with_pce(
        polynomial_order=polynomial_order,
        n_samples=n_samples,
    )

    if export_path is not None:
        surrogate.export(export_path / "population_surrogate")

    return sobol, surrogate, morris_indices


# === Phase 2: Cell-Cycle-Stratified GSA (Steps 5b-7b) === #


def run_phase2(
    param_space: XSpace,
    simulation_func: Callable,
    agg_result: AggregationResult,
    observable_names: list[str],
    n_bins: int = 10,
    polynomial_order: int = 3,
    n_samples: int = 200,
    expected_cycle_time: float = 3600.0,
    export_path: Path | None = None,
) -> tuple[list[SobolIndices], PCESurrogate, CellCycleRelevanceResult]:
    """
    Phase 2: Cell-cycle-stratified GSA (Steps 5b-7b).

    Step 5b: GSA-informed observable selection — uses variance decomposition
        residuals to identify observables driven by cell cycle dynamics.
    Step 6b: Koopman cell cycle variable θ from selected observables.
    Step 6d: Strategy 4 wrapper (params → per-stage means via θ-binning).
    Step 7b: PCE + Sobol on Strategy 4 → per-stage sensitivity.

    Answers: "Which parameters drive variance WITHIN each cell cycle stage?"

    Args:
        param_space: Input parameter space Ξ.
        simulation_func: Base simulation callable (params → raw output).
        agg_result: AggregationResult from step 3.
        observable_names: Names of observables in the simulation output.
        n_bins: Number of cell cycle stage bins.
        polynomial_order: PCE polynomial order.
        n_samples: Number of LHS samples for PCE fitting.
        expected_cycle_time: Expected cell cycle period in seconds.
        export_path: If provided, export surrogate to this path.

    Returns:
        Tuple of (list[SobolIndices], PCESurrogate, CellCycleRelevanceResult).
    """
    from uq.cell_cycle import KoopmanCellCycleVariable

    # Step 5b: GSA-informed observable selection
    relevance = identify_cell_cycle_relevant_observables(
        aggregated_uniform=agg_result.uniform,
        aggregated_by_gen=agg_result.generation,
        aggregated_by_seed=agg_result.seed,
        observable_names=observable_names,
    )

    # Use relevant observables, or fall back to all observables
    selected_obs = relevance.relevant_observables or observable_names

    # Step 6b: Koopman cell cycle variable
    koopman_cc = KoopmanCellCycleVariable(
        observable_columns=selected_obs,
        expected_cycle_time=expected_cycle_time,
    )

    # Step 6d: Strategy 4 wrapper
    f_stage4 = Strategy4Wrapper(
        base_wrapper=simulation_func,
        koopman_cc=koopman_cc,
        n_bins=n_bins,
        observable_columns=selected_obs,
    )

    # Step 7b: PCE + Sobol on Strategy 4
    per_stage_sobol, surrogate = compute_strategy4_sobol(
        param_space=param_space,
        f_stage4=f_stage4,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
    )

    if export_path is not None:
        surrogate.export(export_path / "cell_cycle_surrogate")

        # Export Koopman spectrum visualization (Step 6b artifact)
        if koopman_cc.spectrum is not None:
            from uq.viz import plot_koopman_spectrum

            fig = plot_koopman_spectrum(
                spectrum=koopman_cc.spectrum,
                expected_cycle_time=expected_cycle_time,
                observable_names=selected_obs,
            )
            fig.write_image(str(export_path / "koopman_spectrum.pdf"))

    return per_stage_sobol, surrogate, relevance


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


def execute_pipeline(
    sim_data_path: str | Path | list[str | Path],
    simulation_func: Callable,
    experiment_ids: str | list[str],
    sim_base_path: str | Path,
    observable_columns: list[str] | None = None,
    output_types: list[str] | None = None,
    generation_lower_bound: int | None = 2,
    time_lower_bound: float | None = 100.0,
    n_bins: int = 10,
    polynomial_order: int = 3,
    n_samples: int = 200,
    expected_cycle_time: float = 3600.0,
    prescreen_config: PCEParameterSelectionConfig | None = None,
    export_path: Path | None = None,
) -> PipelineResult:
    """
    Execute the full RFC006 UQ pipeline.

    Step 1 builds the input parameter space Ξ programmatically from
    ``SimulationDataEcoli`` pickle(s).  Steps 2-4 load simulation
    timeseries via DuckDB, aggregate, and decompose variance.  Then
    Phase 1 (population-level GSA) and Phase 2 (cell-cycle GSA) run.

    All intermediate outputs specified by RFC006 are captured in the
    returned PipelineResult.

    Args:
        sim_data_path: Path(s) to ``simData.cPickle`` file(s).  A single
            path produces an ``XSpaceVecoli`` for one condition; a list
            merges parameters from multiple conditions (e.g. violacein +
            mecillinam).
        simulation_func: Callable with evaluate_batch(X) → Y.
        experiment_ids: Experiment identifier(s) for data loading (e.g.,
            ``"mecillinam"`` or ``["mecillinam", "violacein"]``).
            Passed to ``dataset_sql``.
        sim_base_path: Root directory containing simulation outputs
            (e.g., ``/path/to/vEcoli/api_integration/sims``).
        observable_columns: Column names of observables to aggregate/analyze.
            If None, defaults to higher-order properties
            (dry_mass, growth_rate, volume, cell_mass).
        output_types: List of OutputType values to extract (e.g.,
            ["transcriptome", "exchange_fluxes"]). If None, extracts
            higher-order properties only.
        generation_lower_bound: Skip initial generations (default: 2).
        time_lower_bound: Skip transient period in seconds (default: 100.0).
        n_bins: Number of cell cycle stage bins for Phase 2.
        polynomial_order: PCE polynomial order for both phases.
        n_samples: Number of LHS samples for PCE fitting.
        expected_cycle_time: Expected cell cycle period in seconds.
        prescreen_config: Optional Morris prescreening config for Phase 1
            (Step 5a). When provided, Morris screening identifies the most
            influential parameters before PCE construction.
        export_path: If provided, export surrogates and results here.

    Returns:
        PipelineResult with all RFC006 pipeline outputs.
    """
    from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

    from uq.outputs import OutputExtractor, OutputType
    from uq.pipeline.param_loader import ParameterDataset

    # --- Step 1: Build parameter space from sim_data ---
    if isinstance(sim_data_path, (str, Path)):
        datasets = [ParameterDataset(sim_data_path=sim_data_path)]
    else:
        datasets = [ParameterDataset(sim_data_path=p) for p in sim_data_path]

    if len(datasets) == 1:
        param_space = datasets[0].to_parameter_space()
    else:
        param_space = ParameterDataset.merge_to_parameter_space(*datasets)

    # Normalize experiment_ids to list
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    # --- Step 2: Load simulation data via DuckDB + OutputExtractor ---
    conn = create_duckdb_conn()
    history_sql, config_sql, _ = dataset_sql(str(sim_base_path), experiment_ids)
    extractor = OutputExtractor(conn, history_sql, config_sql)

    # Load timeseries once — the single source of truth for all data.
    timeseries = extractor.load_timeseries(
        columns=observable_columns,
        generation_lower_bound=generation_lower_bound,
        time_lower_bound=time_lower_bound,
    )

    # Extract typed outputs from the already-loaded timeseries (no re-query).
    if output_types is not None:
        extract_types = [OutputType(t) for t in output_types]
    else:
        extract_types = [OutputType.HIGHER_ORDER_PROPERTIES]

    outputs = extractor.extract_all(
        output_types=extract_types,
        generation_lower_bound=generation_lower_bound,
        time_lower_bound=time_lower_bound,
        timeseries=timeseries,
    )

    # Resolve observable columns from extracted outputs if not specified
    if observable_columns is None:
        observable_columns = list(outputs.higher_order_properties.keys())
        if not observable_columns:
            # Fall back to available numeric columns in the timeseries
            metadata_cols = {"experiment_id", "variant", "lineage_seed", "generation", "agent_id", "time"}
            observable_columns = [c for c in timeseries.columns if c not in metadata_cols]

    # --- Steps 3-4: Aggregation + Variance Decomposition (shared) ---
    agg_result = aggregate_timeseries(timeseries, observable_columns)
    decomp = get_variance_decomposition(agg_result)

    # --- Phase 1: Population-level GSA (Steps 5a-7a) ---
    sobol_bulk, surrogate_bulk, morris_indices = run_phase1(
        param_space=param_space,
        simulation_func=simulation_func,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
        prescreen_config=prescreen_config,
        export_path=export_path,
    )

    # --- Phase 2: Cell-cycle-stratified GSA (Steps 5b-7b) ---
    per_stage_sobol, surrogate_cc, cc_relevance = run_phase2(
        param_space=param_space,
        simulation_func=simulation_func,
        agg_result=agg_result,
        observable_names=observable_columns,
        n_bins=n_bins,
        polynomial_order=polynomial_order,
        n_samples=n_samples,
        expected_cycle_time=expected_cycle_time,
        export_path=export_path,
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


async def execute_pipeline_async(
    sim_data_path: str | Path | list[str | Path],
    simulation_func: Callable,
    experiment_ids: str | list[str],
    sim_base_path: str | Path,
    observable_columns: list[str] | None = None,
    output_types: list[str] | None = None,
    generation_lower_bound: int | None = 2,
    time_lower_bound: float | None = 100.0,
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
        sim_data_path: Path(s) to ``simData.cPickle`` file(s).
        simulation_func: Callable with evaluate_batch(X) → Y.
        experiment_ids: Experiment identifier(s) for data loading.
        sim_base_path: Root directory containing simulation outputs.
        observable_columns: Column names of observables to aggregate/analyze.
        output_types: List of OutputType values to extract.
        generation_lower_bound: Skip initial generations (default: 2).
        time_lower_bound: Skip transient period in seconds (default: 100.0).
        n_bins: Number of cell cycle stage bins for Phase 2.
        polynomial_order: PCE polynomial order for both phases.
        n_samples: Number of LHS samples for PCE fitting.
        expected_cycle_time: Expected cell cycle period in seconds.
        prescreen_config: Optional Morris prescreening config for Phase 1.
        export_path: If provided, export surrogates and results here.

    Returns:
        PipelineResult with all RFC006 pipeline outputs.
    """
    from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

    from uq.outputs import OutputExtractor, OutputType
    from uq.pipeline.param_loader import ParameterDataset

    # --- Step 1: Build parameter space from sim_data ---
    if isinstance(sim_data_path, (str, Path)):
        datasets = [ParameterDataset(sim_data_path=sim_data_path)]
    else:
        datasets = [ParameterDataset(sim_data_path=p) for p in sim_data_path]

    if len(datasets) == 1:
        param_space = datasets[0].to_parameter_space()
    else:
        param_space = ParameterDataset.merge_to_parameter_space(*datasets)

    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    # --- Step 2: Load simulation data via DuckDB + OutputExtractor ---
    conn = create_duckdb_conn()
    history_sql, config_sql, _ = dataset_sql(str(sim_base_path), experiment_ids)
    extractor = OutputExtractor(conn, history_sql, config_sql)

    # Load timeseries once, then derive typed outputs from it.
    timeseries = extractor.load_timeseries(
        columns=observable_columns,
        generation_lower_bound=generation_lower_bound,
        time_lower_bound=time_lower_bound,
    )

    if output_types is not None:
        extract_types = [OutputType(t) for t in output_types]
    else:
        extract_types = [OutputType.HIGHER_ORDER_PROPERTIES]

    outputs = extractor.extract_all(
        output_types=extract_types,
        generation_lower_bound=generation_lower_bound,
        time_lower_bound=time_lower_bound,
        timeseries=timeseries,
    )

    if observable_columns is None:
        observable_columns = list(outputs.higher_order_properties.keys())
        if not observable_columns:
            metadata_cols = {"experiment_id", "variant", "lineage_seed", "generation", "agent_id", "time"}
            observable_columns = [c for c in timeseries.columns if c not in metadata_cols]

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


# === Convenience: create_surrogate === #


def create_surrogate(
    full_space: XSpaceInterface,
    f: Callable,
    sample_size: int,
    config: PCEParameterSelectionConfig | None = None,
    export: bool = True,
    **kwargs,
) -> PCESurrogate:
    """Build a PCE surrogate from a parameter space and simulation function."""
    surrogate = _generate_surrogate(space=full_space, generator=f, sample_size=sample_size, config=config)
    if export:
        path = kwargs.get("path", Path(os.getcwd()).absolute())
        surrogate.export(path)
    return surrogate


# === Convenience: calculate_cell_cycle === #


def calculate_cell_cycle(
    experiment_id: str,
    outdir_root: str,
    variable_type: Literal["mass_based", "dna_replication", "cell_angle", "koopman"] = "mass_based",
    n_bins: int = 10,
    output_column: str = "listeners__mass__dry_mass",
    verbose: bool = True,
):
    """Convenience wrapper for cell cycle computation."""
    result = _cell_cycle(
        experiment_id=experiment_id,
        outdir_root=outdir_root,
        variable_type=variable_type,
        n_bins=n_bins,
        output_column=output_column,
        verbose=verbose,
    )
    if verbose:
        result.print_summary()
    return result
