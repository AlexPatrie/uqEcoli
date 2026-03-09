"""
Cell cycle stratification for uncertainty quantification (Phase 2).

This module implements aggregation strategy (4): stratified by cell cycle stage.
It defines the framework for computing low-dimensional "cell cycle variables"
that deterministically bin simulation data into cell stages.

The cell cycle variable should:
1. Be approximately cyclic with respect to the cell cycle
2. Have a quantitative relationship with omics measurements from IV&V
3. Be computed from process variables inside vEcoli

Examples of cell cycle variables:
- Cell angle (established in literature)
- Mass-based progression
- DNA replication progress
- Custom composite variables

This module provides an extensible framework where specific cell cycle
variables can be implemented as subclasses or registered functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


class CellCyclePhase(str, Enum):
    """Standard cell cycle phases for E. coli."""

    B_PERIOD = "B_period"  # Pre-initiation (birth to replication initiation)
    C_PERIOD = "C_period"  # DNA replication
    D_PERIOD = "D_period"  # Post-replication to division
    UNKNOWN = "unknown"


@dataclass
class CellCycleVariable:
    """
    Container for cell cycle variable values.

    Attributes:
        values: The computed cell cycle variable values, shape (n_timepoints,)
        phase_labels: Cell cycle phase labels for each timepoint
        normalized: Whether values are normalized to [0, 1]
        variable_name: Name of the cell cycle variable
        metadata: Additional metadata about the computation
    """

    values: np.ndarray
    phase_labels: Optional[np.ndarray] = None
    normalized: bool = True
    variable_name: str = "cell_cycle_variable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_stage_bins(self, n_bins: int = 10) -> np.ndarray:
        """
        Bin the cell cycle variable into discrete stages.

        Args:
            n_bins: Number of bins/stages

        Returns:
            Array of bin indices (0 to n_bins-1)
        """
        if self.normalized:
            bins = np.linspace(0, 1, n_bins + 1)
        else:
            bins = np.linspace(self.values.min(), self.values.max(), n_bins + 1)

        return np.digitize(self.values, bins) - 1


class CellCycleVariableComputer(ABC):
    """
    Abstract base class for computing cell cycle variables.

    Implementations should provide methods to compute a low-dimensional
    (preferably scalar) variable that tracks cell cycle progression.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this cell cycle variable."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """List of Parquet column names required for computation."""
        pass

    @abstractmethod
    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute the cell cycle variable from simulation data.

        Args:
            data: Polars DataFrame with required columns
            sim_data: Optional SimulationDataEcoli for additional metadata

        Returns:
            CellCycleVariable instance
        """
        pass


class MassBasedCellCycleVariable(CellCycleVariableComputer):
    """
    Cell cycle variable based on cell mass progression.

    This computes a normalized cell cycle position based on the ratio
    of current dry mass to the expected mass at division.

    The mass-based approach is simple and robust, assuming exponential
    growth during the cell cycle.
    """

    @property
    def name(self) -> str:
        return "mass_based"

    @property
    def required_columns(self) -> list[str]:
        return [
            "listeners__mass__dry_mass",
            "generation",
            "agent_id",
            "time",
        ]

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute mass-based cell cycle variable.

        For each cell, computes: (log(M) - log(M_birth)) / (log(M_div) - log(M_birth))
        where M is current mass, M_birth is mass at birth, M_div is mass at division.

        This normalizes the cell cycle to [0, 1] assuming exponential growth.
        """
        # Get initial and final mass for each cell
        cell_stats = data.group_by(["generation", "agent_id"]).agg([
            pl.col("listeners__mass__dry_mass").first().alias("mass_birth"),
            pl.col("listeners__mass__dry_mass").last().alias("mass_div"),
        ])

        # Join back to get per-timepoint values
        data_with_stats = data.join(
            cell_stats,
            on=["generation", "agent_id"],
            how="left",
        )

        # Compute normalized cell cycle position
        log_mass = np.log(data_with_stats["listeners__mass__dry_mass"].to_numpy())
        log_birth = np.log(data_with_stats["mass_birth"].to_numpy())
        log_div = np.log(data_with_stats["mass_div"].to_numpy())

        # Avoid division by zero
        denom = log_div - log_birth
        denom = np.where(denom > 0, denom, 1.0)

        values = (log_mass - log_birth) / denom
        values = np.clip(values, 0, 1)

        return CellCycleVariable(
            values=values,
            normalized=True,
            variable_name=self.name,
            metadata={
                "method": "log_mass_ratio",
            },
        )


class DNAReplicationCellCycleVariable(CellCycleVariableComputer):
    """
    Cell cycle variable based on DNA replication progress.

    This tracks the progress through the cell cycle using DNA mass
    as a proxy for replication progress. E. coli cells initiate DNA
    replication at a characteristic mass per origin.
    """

    @property
    def name(self) -> str:
        return "dna_replication"

    @property
    def required_columns(self) -> list[str]:
        return [
            "listeners__mass__dna_mass",
            "listeners__mass__dry_mass",
            "generation",
            "agent_id",
            "time",
        ]

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute DNA replication-based cell cycle variable.

        Uses the ratio of DNA mass to dry mass as an indicator of
        replication progress. Higher ratios indicate ongoing replication.
        """
        dna_mass = data["listeners__mass__dna_mass"].to_numpy()
        dry_mass = data["listeners__mass__dry_mass"].to_numpy()

        # DNA/mass ratio
        dna_ratio = dna_mass / np.where(dry_mass > 0, dry_mass, 1.0)

        # Normalize within each cell
        cell_groups = data.group_by(["generation", "agent_id"]).agg([
            pl.col("listeners__mass__dna_mass").min().alias("min_dna"),
            pl.col("listeners__mass__dna_mass").max().alias("max_dna"),
        ])

        data_with_stats = data.join(
            cell_groups,
            on=["generation", "agent_id"],
            how="left",
        )

        min_dna = data_with_stats["min_dna"].to_numpy()
        max_dna = data_with_stats["max_dna"].to_numpy()

        denom = max_dna - min_dna
        denom = np.where(denom > 0, denom, 1.0)

        values = (dna_mass - min_dna) / denom
        values = np.clip(values, 0, 1)

        # Assign phases based on DNA content
        phases = np.full(len(values), CellCyclePhase.UNKNOWN.value)
        # B period: low DNA content (< 0.2 of range)
        phases[values < 0.2] = CellCyclePhase.B_PERIOD.value
        # C period: increasing DNA (0.2 - 0.8)
        phases[(values >= 0.2) & (values < 0.8)] = CellCyclePhase.C_PERIOD.value
        # D period: high DNA content (> 0.8)
        phases[values >= 0.8] = CellCyclePhase.D_PERIOD.value

        return CellCycleVariable(
            values=values,
            phase_labels=phases,
            normalized=True,
            variable_name=self.name,
            metadata={
                "method": "dna_mass_normalization",
            },
        )


class CellAngleCellCycleVariable(CellCycleVariableComputer):
    """
    Cell angle-based cell cycle variable.

    The "cell angle" is a well-established approach in the literature
    for representing cell cycle state. It uses PCA or similar dimensionality
    reduction on omics data to find a circular (periodic) coordinate.

    This implementation uses a simplified version based on mass and
    growth rate that approximates the cell angle concept.
    """

    @property
    def name(self) -> str:
        return "cell_angle"

    @property
    def required_columns(self) -> list[str]:
        return [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "generation",
            "agent_id",
            "time",
        ]

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute cell angle from simulation data.

        Uses a 2D projection based on:
        - x: normalized mass (log-scaled)
        - y: growth rate

        The angle in this 2D space approximates cell cycle progression.
        """
        # Get mass data
        dry_mass = data["listeners__mass__dry_mass"].to_numpy()
        time = data["time"].to_numpy()

        # Compute growth rate using windowed difference
        # Group by cell to compute per-cell growth rates
        growth_rates = np.zeros_like(dry_mass)

        # Sort by time within each cell group
        sorted_data = data.sort(["generation", "agent_id", "time"])
        dry_mass_sorted = sorted_data["listeners__mass__dry_mass"].to_numpy()

        # Simple growth rate: d(log(mass))/dt
        log_mass = np.log(np.maximum(dry_mass_sorted, 1e-10))
        growth_rates = np.gradient(log_mass)

        # Normalize both dimensions
        log_mass_norm = (log_mass - log_mass.min()) / (log_mass.max() - log_mass.min() + 1e-10)
        growth_norm = (growth_rates - growth_rates.min()) / (growth_rates.max() - growth_rates.min() + 1e-10)

        # Compute angle in the (mass, growth) plane
        # Shift to center
        x = log_mass_norm - 0.5
        y = growth_norm - 0.5

        angles = np.arctan2(y, x)

        # Normalize to [0, 1]
        values = (angles + np.pi) / (2 * np.pi)

        return CellCycleVariable(
            values=values,
            normalized=True,
            variable_name=self.name,
            metadata={
                "method": "mass_growth_angle",
                "description": "Angle in (log_mass, growth_rate) space",
            },
        )


class CompositeCellCycleVariable(CellCycleVariableComputer):
    """
    Composite cell cycle variable from multiple omics measurements.

    This allows defining custom cell cycle variables as functions of
    multiple simulation outputs, providing flexibility for exploring
    different approaches.
    """

    def __init__(
        self,
        name: str,
        required_columns: list[str],
        compute_func: Callable[[pl.DataFrame], np.ndarray],
    ):
        """
        Initialize with a custom computation function.

        Args:
            name: Name for this variable
            required_columns: List of required column names
            compute_func: Function that takes a DataFrame and returns
                         an array of cell cycle values
        """
        self._name = name
        self._required_columns = required_columns
        self._compute_func = compute_func

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_columns(self) -> list[str]:
        return self._required_columns

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """Compute using the provided function."""
        values = self._compute_func(data)
        return CellCycleVariable(
            values=values,
            normalized=True,
            variable_name=self.name,
        )


class CellCycleAggregator:
    """
    Aggregates simulation data by cell cycle stage.

    This class implements aggregation strategy (4) from the UQ framework,
    enabling "phenotypic" sensitivity analysis across the physiological
    time dimension.
    """

    # Available cell cycle variable implementations
    VARIABLES: dict[str, type[CellCycleVariableComputer]] = {
        "mass_based": MassBasedCellCycleVariable,
        "dna_replication": DNAReplicationCellCycleVariable,
        "cell_angle": CellAngleCellCycleVariable,
    }

    def __init__(
        self,
        conn: DuckDBPyConnection,
        history_sql: str,
        config_sql: str,
        variable_type: str = "mass_based",
        n_stages: int = 10,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ):
        """
        Initialize the cell cycle aggregator.

        Args:
            conn: DuckDB connection
            history_sql: SQL subquery for history data
            config_sql: SQL subquery for configuration data
            variable_type: Type of cell cycle variable to use
            n_stages: Number of cell cycle stages for binning
            sim_data: Optional SimulationDataEcoli for metadata
        """
        self.conn = conn
        self.history_sql = history_sql
        self.config_sql = config_sql
        self.n_stages = n_stages
        self.sim_data = sim_data

        if variable_type not in self.VARIABLES:
            raise ValueError(
                f"Unknown cell cycle variable type: {variable_type}. "
                f"Available: {list(self.VARIABLES.keys())}"
            )
        self.variable_computer = self.VARIABLES[variable_type]()

    def compute_cell_cycle_variable(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> CellCycleVariable:
        """
        Compute the cell cycle variable for all data points.

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            CellCycleVariable instance with values for all timepoints
        """
        from ecoli.library.parquet_emitter import read_stacked_columns

        columns = self.variable_computer.required_columns

        history_subquery = read_stacked_columns(
            self.history_sql,
            columns,
            order_results=False,
        )

        filter_clause = self._build_filter_clause(
            generation_lower_bound, time_lower_bound
        )

        query = f"""
            WITH history AS ({history_subquery}),
            filtered AS (
                SELECT *
                FROM history
                {filter_clause}
            )
            SELECT *
            FROM filtered
            ORDER BY generation, agent_id, time
        """

        data = self.conn.sql(query).pl()

        if data.is_empty():
            return CellCycleVariable(
                values=np.array([]),
                variable_name=self.variable_computer.name,
            )

        return self.variable_computer.compute(data, self.sim_data)

    def aggregate_by_cell_cycle(
        self,
        output_column: str,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate an output variable by cell cycle stage.

        Args:
            output_column: Name of the output column to aggregate
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (stage_means, stage_stds, stage_labels)
        """
        from ecoli.library.parquet_emitter import read_stacked_columns

        # Get both cell cycle variable columns and output column
        columns = self.variable_computer.required_columns + [output_column]
        columns = list(set(columns))  # Remove duplicates

        history_subquery = read_stacked_columns(
            self.history_sql,
            columns,
            order_results=False,
        )

        filter_clause = self._build_filter_clause(
            generation_lower_bound, time_lower_bound
        )

        query = f"""
            WITH history AS ({history_subquery}),
            filtered AS (
                SELECT *
                FROM history
                {filter_clause}
            )
            SELECT *
            FROM filtered
            ORDER BY generation, agent_id, time
        """

        data = self.conn.sql(query).pl()

        if data.is_empty():
            return np.array([]), np.array([]), np.array([])

        # Compute cell cycle variable
        cc_var = self.variable_computer.compute(data, self.sim_data)

        # Bin into stages
        stage_labels = cc_var.to_stage_bins(self.n_stages)

        # Get output values
        output_values = data[output_column].to_numpy()

        # Handle array-valued outputs
        if output_values.dtype == object:
            # Assume it's a list of arrays
            output_values = np.vstack(output_values)
            n_features = output_values.shape[1]

            stage_means = np.zeros((self.n_stages, n_features))
            stage_stds = np.zeros((self.n_stages, n_features))

            for stage in range(self.n_stages):
                mask = stage_labels == stage
                if np.sum(mask) > 0:
                    stage_data = output_values[mask]
                    stage_means[stage] = np.nanmean(stage_data, axis=0)
                    stage_stds[stage] = np.nanstd(stage_data, axis=0)
        else:
            stage_means = np.zeros(self.n_stages)
            stage_stds = np.zeros(self.n_stages)

            for stage in range(self.n_stages):
                mask = stage_labels == stage
                if np.sum(mask) > 0:
                    stage_data = output_values[mask]
                    stage_means[stage] = np.nanmean(stage_data)
                    stage_stds[stage] = np.nanstd(stage_data)

        return stage_means, stage_stds, np.arange(self.n_stages)

    def get_cell_cycle_profile(
        self,
        output_column: str,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """
        Get the full cell cycle profile for an output variable.

        This returns the aggregated mean and std for each cell cycle stage,
        along with the stage labels.

        Args:
            output_column: Name of the output column
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Dictionary with 'mean', 'std', 'stage', and 'n_samples' arrays
        """
        means, stds, stages = self.aggregate_by_cell_cycle(
            output_column,
            generation_lower_bound,
            time_lower_bound,
        )

        return {
            "mean": means,
            "std": stds,
            "stage": stages,
            "cell_cycle_variable": self.variable_computer.name,
            "n_stages": self.n_stages,
        }

    def _build_filter_clause(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> str:
        """Build SQL WHERE clause for filtering."""
        filters = []
        if generation_lower_bound is not None:
            filters.append(f"generation >= {generation_lower_bound}")
        if time_lower_bound is not None:
            filters.append(f"time >= {float(time_lower_bound)}")

        if filters:
            return "WHERE " + " AND ".join(filters)
        return ""


def register_cell_cycle_variable(
    name: str,
    computer: CellCycleVariableComputer,
) -> None:
    """
    Register a custom cell cycle variable implementation.

    Args:
        name: Name for the variable (used in CellCycleAggregator)
        computer: CellCycleVariableComputer instance
    """
    CellCycleAggregator.VARIABLES[name] = type(computer)
