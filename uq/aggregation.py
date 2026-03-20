"""
Aggregation strategies for uncertainty quantification.

This module implements the four aggregation strategies specified in the UQ framework:

1. Uniform aggregation: Across all simulated cells and times (baseline)
2. Stratified by generation: Control of convergence towards steady-state growth
3. Stratified by lineage seed: Control of exogenous variance
4. Stratified by cell cycle stage: Time course within a cell's lifespan (Phase 2)

These strategies enable deconvolution of different types of uncertainty and
accurate modeling of the relationship between "bulk" and "single-cell" attributes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


class AggregationStrategy(str, Enum):
    """Available aggregation strategies for UQ analysis."""

    UNIFORM = "uniform"
    BY_GENERATION = "by_generation"
    BY_LINEAGE_SEED = "by_lineage_seed"
    BY_CELL_CYCLE = "by_cell_cycle"


@dataclass
class AggregatedOutput:
    """
    Container for aggregated output statistics.

    Attributes:
        mean: Mean values across the aggregation dimension
        std: Standard deviation across the aggregation dimension
        n_samples: Number of samples in each aggregation group
        groups: Group labels (generation, lineage_seed, etc.) if stratified
        raw_data: Optional raw data before aggregation
    """

    mean: np.ndarray
    std: np.ndarray
    n_samples: int | np.ndarray
    groups: Optional[np.ndarray] = None
    raw_data: Optional[np.ndarray] = None


class Aggregator:
    """
    Performs aggregation of simulation outputs according to specified strategies.

    This class extracts subsampled time points and variables from emitted
    simulation trajectories and aggregates them into output variables Y
    for sensitivity analysis.
    """

    # Column names
    ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]

    def __init__(
        self,
        conn: DuckDBPyConnection,
        history_sql: str,
        config_sql: str,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ):
        """
        Initialize the aggregator.

        Args:
            conn: DuckDB connection
            history_sql: SQL subquery for history (timeseries) data
            config_sql: SQL subquery for configuration data
            sim_data: Optional SimulationDataEcoli for metadata lookup
        """
        self.conn = conn
        self.history_sql = history_sql
        self.config_sql = config_sql
        self.sim_data = sim_data

    def aggregate(
        self,
        data: np.ndarray,
        strategy: AggregationStrategy,
        group_labels: Optional[np.ndarray] = None,
    ) -> AggregatedOutput:
        """
        Aggregate data according to the specified strategy.

        Args:
            data: Input data array of shape (n_samples, n_features)
            strategy: Aggregation strategy to use
            group_labels: Group labels for stratified aggregation

        Returns:
            AggregatedOutput containing statistics
        """
        if strategy == AggregationStrategy.UNIFORM:
            return self._aggregate_uniform(data)
        elif strategy in (
            AggregationStrategy.BY_GENERATION,
            AggregationStrategy.BY_LINEAGE_SEED,
            AggregationStrategy.BY_CELL_CYCLE,
        ):
            if group_labels is None:
                raise ValueError(f"group_labels required for {strategy.value} aggregation")
            return self._aggregate_stratified(data, group_labels)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _aggregate_uniform(self, data: np.ndarray) -> AggregatedOutput:
        """
        Aggregate uniformly across all samples (Strategy 1).

        This is the baseline aggregation that computes mean and std
        across all cells and all timepoints.

        Args:
            data: Input data array of shape (n_samples, n_features)

        Returns:
            AggregatedOutput with single mean/std values per feature
        """
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        return AggregatedOutput(
            mean=mean,
            std=std,
            n_samples=data.shape[0],
            raw_data=data,
        )

    def _aggregate_stratified(
        self,
        data: np.ndarray,
        group_labels: np.ndarray,
    ) -> AggregatedOutput:
        """
        Aggregate stratified by group labels (Strategies 2, 3, 4).

        This computes mean and std within each group, enabling
        analysis of variance attributable to different factors.

        Args:
            data: Input data array of shape (n_samples, n_features)
            group_labels: Array of group labels for each sample

        Returns:
            AggregatedOutput with per-group statistics
        """
        unique_groups = np.unique(group_labels)
        n_groups = len(unique_groups)
        n_features = data.shape[1] if len(data.shape) > 1 else 1

        means = np.zeros((n_groups, n_features))
        stds = np.zeros((n_groups, n_features))
        n_samples = np.zeros(n_groups, dtype=int)

        for i, group in enumerate(unique_groups):
            mask = group_labels == group
            group_data = data[mask]
            means[i] = np.nanmean(group_data, axis=0)
            stds[i] = np.nanstd(group_data, axis=0)
            n_samples[i] = np.sum(mask)

        return AggregatedOutput(
            mean=means,
            std=stds,
            n_samples=n_samples,
            groups=unique_groups,
            raw_data=data,
        )

    def aggregate_transcriptome(
        self,
        strategy: AggregationStrategy,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[AggregatedOutput, list[str]]:
        """
        Aggregate transcriptome data according to the specified strategy.

        Args:
            strategy: Aggregation strategy to use
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (AggregatedOutput, cistron_ids)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        mrna_col = "listeners__rna_counts__mRNA_cistron_counts"
        mrna_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=mrna_col,
        )

        history_subquery = read_stacked_columns(
            self.history_sql,
            [f"{mrna_col} AS mrna_counts"],
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        group_col = self._get_group_column(strategy)

        if strategy == AggregationStrategy.UNIFORM:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(mrna_counts) AS mrna_count,
                        generate_subscripts(mrna_counts, 1) AS idx
                    FROM filtered
                )
                SELECT
                    idx,
                    AVG(mrna_count) AS mean_count,
                    STDDEV(mrna_count) AS std_count,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx
                ORDER BY idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    mrna_ids,
                )

            return (
                AggregatedOutput(
                    mean=result["mean_count"].to_numpy(),
                    std=result["std_count"].to_numpy(),
                    n_samples=int(result["n_samples"][0]),
                ),
                mrna_ids,
            )
        else:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(mrna_counts) AS mrna_count,
                        generate_subscripts(mrna_counts, 1) AS idx,
                        {group_col}
                    FROM filtered
                )
                SELECT
                    idx,
                    {group_col} AS group_label,
                    AVG(mrna_count) AS mean_count,
                    STDDEV(mrna_count) AS std_count,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx, {group_col}
                ORDER BY {group_col}, idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    mrna_ids,
                )

            # Reshape to (n_groups, n_features)
            groups = result["group_label"].unique().sort().to_numpy()
            n_features = len(mrna_ids)
            means = np.zeros((len(groups), n_features))
            stds = np.zeros((len(groups), n_features))
            n_samples = np.zeros(len(groups), dtype=int)

            for i, group in enumerate(groups):
                group_data = result.filter(pl.col("group_label") == group).sort("idx")
                means[i] = group_data["mean_count"].to_numpy()
                stds[i] = group_data["std_count"].to_numpy()
                n_samples[i] = int(group_data["n_samples"][0])

            return (
                AggregatedOutput(
                    mean=means,
                    std=stds,
                    n_samples=n_samples,
                    groups=groups,
                ),
                mrna_ids,
            )

    def aggregate_proteome(
        self,
        strategy: AggregationStrategy,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[AggregatedOutput, list[str]]:
        """
        Aggregate proteome data according to the specified strategy.

        Args:
            strategy: Aggregation strategy to use
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (AggregatedOutput, monomer_ids)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        monomer_col = "listeners__monomer_counts"
        monomer_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=monomer_col,
        )

        history_subquery = read_stacked_columns(
            self.history_sql,
            [f"{monomer_col} AS monomer_counts"],
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        group_col = self._get_group_column(strategy)

        if strategy == AggregationStrategy.UNIFORM:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(monomer_counts) AS monomer_count,
                        generate_subscripts(monomer_counts, 1) AS idx
                    FROM filtered
                )
                SELECT
                    idx,
                    AVG(monomer_count) AS mean_count,
                    STDDEV(monomer_count) AS std_count,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx
                ORDER BY idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    monomer_ids,
                )

            return (
                AggregatedOutput(
                    mean=result["mean_count"].to_numpy(),
                    std=result["std_count"].to_numpy(),
                    n_samples=int(result["n_samples"][0]),
                ),
                monomer_ids,
            )
        else:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(monomer_counts) AS monomer_count,
                        generate_subscripts(monomer_counts, 1) AS idx,
                        {group_col}
                    FROM filtered
                )
                SELECT
                    idx,
                    {group_col} AS group_label,
                    AVG(monomer_count) AS mean_count,
                    STDDEV(monomer_count) AS std_count,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx, {group_col}
                ORDER BY {group_col}, idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    monomer_ids,
                )

            groups = result["group_label"].unique().sort().to_numpy()
            n_features = len(monomer_ids)
            means = np.zeros((len(groups), n_features))
            stds = np.zeros((len(groups), n_features))
            n_samples = np.zeros(len(groups), dtype=int)

            for i, group in enumerate(groups):
                group_data = result.filter(pl.col("group_label") == group).sort("idx")
                means[i] = group_data["mean_count"].to_numpy()
                stds[i] = group_data["std_count"].to_numpy()
                n_samples[i] = int(group_data["n_samples"][0])

            return (
                AggregatedOutput(
                    mean=means,
                    std=stds,
                    n_samples=n_samples,
                    groups=groups,
                ),
                monomer_ids,
            )

    def aggregate_fluxes(
        self,
        strategy: AggregationStrategy,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
        exchange_only: bool = False,
    ) -> tuple[AggregatedOutput, list[str]]:
        """
        Aggregate metabolic flux data according to the specified strategy.

        Args:
            strategy: Aggregation strategy to use
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards
            exchange_only: If True, only return exchange reaction fluxes

        Returns:
            Tuple of (AggregatedOutput, reaction_ids)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        flux_col = "listeners__fba_results__base_reaction_fluxes"
        mass_col = "listeners__mass__cell_mass"
        dry_mass_col = "listeners__mass__dry_mass"

        rxn_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=flux_col,
        )

        # Get cell density
        cell_density = 1100.0  # Default g/L
        if self.sim_data is not None:
            from wholecell.utils import units

            cd = self.sim_data.constants.cell_density
            cell_density = cd.asNumber(units.g / units.L)

        history_subquery = read_stacked_columns(
            self.history_sql,
            [
                f"{flux_col} AS fluxes",
                f"{mass_col} AS cell_mass",
                f"{dry_mass_col} AS dry_mass",
            ],
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        group_col = self._get_group_column(strategy)

        if strategy == AggregationStrategy.UNIFORM:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *,
                        dry_mass / cell_mass * {cell_density} AS conversion_coeff
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(fluxes) / conversion_coeff AS flux,
                        generate_subscripts(fluxes, 1) AS idx
                    FROM filtered
                )
                SELECT
                    idx,
                    AVG(flux) AS mean_flux,
                    STDDEV(flux) AS std_flux,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx
                ORDER BY idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    rxn_ids,
                )

            mean = result["mean_flux"].to_numpy()
            std = result["std_flux"].to_numpy()

            if exchange_only:
                exchange_mask = np.array(["EX_" in rxn_id for rxn_id in rxn_ids])
                rxn_ids = [rxn_id for rxn_id in rxn_ids if "EX_" in rxn_id]
                mean = mean[exchange_mask]
                std = std[exchange_mask]

            return (
                AggregatedOutput(
                    mean=mean,
                    std=std,
                    n_samples=int(result["n_samples"][0]),
                ),
                rxn_ids,
            )
        else:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *,
                        dry_mass / cell_mass * {cell_density} AS conversion_coeff
                    FROM history
                    {filter_clause}
                ),
                exploded AS (
                    SELECT
                        unnest(fluxes) / conversion_coeff AS flux,
                        generate_subscripts(fluxes, 1) AS idx,
                        {group_col}
                    FROM filtered
                )
                SELECT
                    idx,
                    {group_col} AS group_label,
                    AVG(flux) AS mean_flux,
                    STDDEV(flux) AS std_flux,
                    COUNT(*) AS n_samples
                FROM exploded
                GROUP BY idx, {group_col}
                ORDER BY {group_col}, idx
            """
            result = self.conn.sql(query).pl()

            if result.is_empty():
                return (
                    AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    ),
                    rxn_ids,
                )

            groups = result["group_label"].unique().sort().to_numpy()
            n_features = len(rxn_ids)
            means = np.zeros((len(groups), n_features))
            stds = np.zeros((len(groups), n_features))
            n_samples = np.zeros(len(groups), dtype=int)

            for i, group in enumerate(groups):
                group_data = result.filter(pl.col("group_label") == group).sort("idx")
                means[i] = group_data["mean_flux"].to_numpy()
                stds[i] = group_data["std_flux"].to_numpy()
                n_samples[i] = int(group_data["n_samples"][0])

            if exchange_only:
                exchange_mask = np.array(["EX_" in rxn_id for rxn_id in rxn_ids])
                rxn_ids = [rxn_id for rxn_id in rxn_ids if "EX_" in rxn_id]
                means = means[:, exchange_mask]
                stds = stds[:, exchange_mask]

            return (
                AggregatedOutput(
                    mean=means,
                    std=stds,
                    n_samples=n_samples,
                    groups=groups,
                ),
                rxn_ids,
            )

    def aggregate_higher_order_properties(
        self,
        strategy: AggregationStrategy,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> dict[str, AggregatedOutput]:
        """
        Aggregate higher-order properties according to the specified strategy.

        Args:
            strategy: Aggregation strategy to use
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Dictionary mapping property names to AggregatedOutput
        """
        from ecoli.library.parquet_emitter import read_stacked_columns

        property_cols = [
            "listeners__mass__cell_mass AS cell_mass",
            "listeners__mass__dry_mass AS dry_mass",
            "listeners__mass__volume AS volume",
        ]

        history_subquery = read_stacked_columns(
            self.history_sql,
            property_cols,
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        group_col = self._get_group_column(strategy)

        results: dict[str, AggregatedOutput] = {}

        for prop_name in ["cell_mass", "dry_mass", "volume"]:
            if strategy == AggregationStrategy.UNIFORM:
                query = f"""
                    WITH history AS ({history_subquery}),
                    filtered AS (
                        SELECT *
                        FROM history
                        {filter_clause}
                    )
                    SELECT
                        AVG({prop_name}) AS mean_val,
                        STDDEV({prop_name}) AS std_val,
                        COUNT(*) AS n_samples
                    FROM filtered
                """
                result = self.conn.sql(query).pl()

                if result.is_empty():
                    results[prop_name] = AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    )
                else:
                    results[prop_name] = AggregatedOutput(
                        mean=np.array([result["mean_val"][0]]),
                        std=np.array([result["std_val"][0]]),
                        n_samples=int(result["n_samples"][0]),
                    )
            else:
                query = f"""
                    WITH history AS ({history_subquery}),
                    filtered AS (
                        SELECT *
                        FROM history
                        {filter_clause}
                    )
                    SELECT
                        {group_col} AS group_label,
                        AVG({prop_name}) AS mean_val,
                        STDDEV({prop_name}) AS std_val,
                        COUNT(*) AS n_samples
                    FROM filtered
                    GROUP BY {group_col}
                    ORDER BY {group_col}
                """
                result = self.conn.sql(query).pl()

                if result.is_empty():
                    results[prop_name] = AggregatedOutput(
                        mean=np.array([]),
                        std=np.array([]),
                        n_samples=0,
                    )
                else:
                    results[prop_name] = AggregatedOutput(
                        mean=result["mean_val"].to_numpy(),
                        std=result["std_val"].to_numpy(),
                        n_samples=result["n_samples"].to_numpy(),
                        groups=result["group_label"].to_numpy(),
                    )

        return results

    def _get_group_column(self, strategy: AggregationStrategy) -> str:
        """Get the SQL column name for grouping based on strategy."""
        if strategy == AggregationStrategy.BY_GENERATION:
            return "generation"
        elif strategy == AggregationStrategy.BY_LINEAGE_SEED:
            return "lineage_seed"
        elif strategy == AggregationStrategy.BY_CELL_CYCLE:
            # This will be defined by the cell cycle variable
            # For now, return a placeholder
            return "cell_cycle_stage"
        else:
            return ""

    def _build_filter_clause(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> str:
        """Build SQL WHERE clause for filtering by generation and time."""
        filters = []
        if generation_lower_bound is not None:
            filters.append(f"generation >= {generation_lower_bound}")
        if time_lower_bound is not None:
            filters.append(f"time >= {float(time_lower_bound)}")

        if filters:
            return "WHERE " + " AND ".join(filters)
        return ""


def compute_variance_decomposition(
    aggregated_by_gen: AggregatedOutput,
    aggregated_by_seed: AggregatedOutput,
    aggregated_uniform: AggregatedOutput,
) -> dict[str, np.ndarray]:
    """
    Compute variance decomposition across different stratification levels.

    This decomposes total variance into components attributable to:
    - Between-generation variance (convergence effects)
    - Between-seed variance (exogenous stochasticity)
    - Within-group variance (intrinsic stochasticity)

    Args:
        aggregated_by_gen: Output aggregated by generation
        aggregated_by_seed: Output aggregated by lineage seed
        aggregated_uniform: Uniform aggregation (baseline)

    Returns:
        Dictionary with variance components:
        - 'total_variance': Total variance from uniform aggregation
        - 'between_generation_variance': Variance between generations
        - 'between_seed_variance': Variance between lineage seeds
        - 'within_group_variance': Residual variance
        - 'generation_fraction': Fraction of variance from generation
        - 'seed_fraction': Fraction of variance from lineage seed
    """
    total_var = np.nan_to_num(aggregated_uniform.std, nan=0.0)**2

    # Between-group variance is variance of group means
    gen_between_var = np.var(np.nan_to_num(aggregated_by_gen.mean, nan=0.0), axis=0)
    seed_between_var = np.var(np.nan_to_num(aggregated_by_seed.mean, nan=0.0), axis=0)

    # Within-group variance is mean of group variances
    gen_within_var = np.mean(np.nan_to_num(aggregated_by_gen.std, nan=0.0)**2, axis=0)
    seed_within_var = np.mean(np.nan_to_num(aggregated_by_seed.std, nan=0.0)**2, axis=0)

    # Avoid division by zero
    total_var_safe = np.where(total_var > 0, total_var, 1.0)

    return {
        "total_variance": total_var,
        "between_generation_variance": gen_between_var,
        "between_seed_variance": seed_between_var,
        "within_generation_variance": gen_within_var,
        "within_seed_variance": seed_within_var,
        "generation_fraction": gen_between_var / total_var_safe,
        "seed_fraction": seed_between_var / total_var_safe,
    }
