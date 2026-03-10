"""
Cell cycle stratification for uncertainty quantification (Phase 2).

This module implements aggregation strategy (4): stratified by cell cycle stage.
It defines the framework for computing low-dimensional "cell cycle variables"
that deterministically bin simulation data into cell stages.

The cell cycle variable should:
1. Be approximately cyclic with respect to the cell cycle
2. Have a quantitative relationship with omics measurements from IV&V
3. Be computed from process variables inside vEcoli

Available cell cycle variable implementations:
- **Koopman eigenfunction phase** (recommended): Uses Dynamic Mode Decomposition
  to identify the cell cycle mode and extract its eigenfunction phase as the
  cell cycle coordinate. This data-driven approach automatically captures
  periodic dynamics without assumptions about the underlying mechanism.
- Mass-based progression: Normalized log-mass ratio
- DNA replication progress: DNA content normalization
- Cell angle: 2D projection in (mass, growth_rate) space
- Custom composite variables: User-defined functions

The **Koopman approach** is the preferred method because:
1. It is purely data-driven (no mechanistic assumptions)
2. It automatically identifies periodic cell cycle dynamics
3. The eigenfunction phase naturally wraps [0, 1] once per cycle
4. It is robust to noise and individual cell variations

Per RFC006 Section 1.3, this implements the fourth aggregation strategy
for phenotypic sensitivity analysis across the physiological time dimension.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from uq.koopman import (
    CellCycleKoopmanAnalyzer,
    DynamicModeDecomposition,
    ExtendedDMD,
    KoopmanMode,
)

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

    from uq.aggregation import AggregatedOutput
    from uq.sensitivity import CellCycleRelevanceResult


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


class KoopmanCellCycleVariable(CellCycleVariableComputer):
    """
    Koopman eigenfunction-based cell cycle variable.

    This class uses Dynamic Mode Decomposition (DMD) to identify the cell cycle
    mode from simulation data, then extracts the phase of the corresponding
    Koopman eigenfunction as the cell cycle variable.

    The Koopman approach offers several advantages over heuristic methods:
    1. **Data-driven**: No assumptions about growth mechanism or cycle structure
    2. **Spectral identification**: Automatically finds periodic dynamics at the
       expected cell cycle frequency
    3. **Phase extraction**: The eigenfunction phase naturally wraps [0, 1] once
       per cycle, providing a deterministic mapping from state to cycle position
    4. **Robust**: Captures the dominant periodic structure even with noise

    Per RFC006 Section 1.3, this implements aggregation strategy #4 (cell cycle
    stratification) by providing a principled way to define the cell cycle
    variable through Koopman spectral analysis.

    Mathematical Background:
        For an oscillatory Koopman mode with eigenvalue λ = |λ|e^(iω),
        the eigenfunction φ(x) maps states to complex numbers. The phase
        angle θ = arg(φ(x)) / (2π) provides a [0,1]-valued coordinate
        that advances uniformly with the oscillation.

    Example:
        >>> koopman_cc = KoopmanCellCycleVariable(expected_cycle_time=3600.0)
        >>> cc_var = koopman_cc.compute(trajectory_data)
        >>> bins = cc_var.to_stage_bins(n_bins=10)
    """

    def __init__(
        self,
        expected_cycle_time: float = 3600.0,
        frequency_tolerance: float = 0.3,
        dt: float = 1.0,
        use_edmd: bool = True,
        observable_columns: Optional[list[str]] = None,
    ):
        """
        Initialize Koopman-based cell cycle variable computer.

        Args:
            expected_cycle_time: Expected cell cycle duration in seconds.
                Default is 3600s (1 hour), typical for fast-growing E. coli.
            frequency_tolerance: Fractional tolerance for matching the cell
                cycle frequency. Default 0.3 means modes with frequency within
                30% of expected are considered cell cycle modes.
            dt: Timestep between data points in seconds. Used for converting
                eigenvalues to frequencies.
            use_edmd: Whether to use Extended DMD with polynomial dictionary.
                EDMD can better capture nonlinear dynamics but is slower.
            observable_columns: Optional list of additional observable columns
                to include in DMD analysis. If None, uses default mass-based
                observables.
        """
        self._expected_cycle_time = expected_cycle_time
        self._frequency_tolerance = frequency_tolerance
        self._dt = dt
        self._use_edmd = use_edmd

        # Default observables for DMD if not specified
        self._observable_columns = observable_columns or [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
        ]

        # Store the identified cell cycle mode for inspection
        self._cell_cycle_mode: Optional[KoopmanMode] = None

    @property
    def name(self) -> str:
        return "koopman"

    @property
    def required_columns(self) -> list[str]:
        return self._observable_columns + [
            "generation",
            "agent_id",
            "time",
        ]

    @property
    def cell_cycle_mode(self) -> Optional[KoopmanMode]:
        """The identified cell cycle Koopman mode (available after compute())."""
        return self._cell_cycle_mode

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute Koopman eigenfunction phase as cell cycle variable.

        This method:
        1. Builds a trajectory matrix from the observable columns
        2. Fits DMD/EDMD to extract Koopman modes
        3. Identifies the cell cycle mode (oscillatory mode near expected frequency)
        4. Projects each data point onto the mode's eigenvector
        5. Extracts the phase angle and normalizes to [0, 1]

        Args:
            data: Polars DataFrame with required columns
            sim_data: Optional SimulationDataEcoli (not used, for API consistency)

        Returns:
            CellCycleVariable with phase values in [0, 1]
        """
        # Build trajectory matrix from observable columns
        obs_cols = [c for c in self._observable_columns if c in data.columns]
        if len(obs_cols) < 1:
            raise ValueError(f"No observable columns found in data. Required: {self._observable_columns}")

        # Extract data as numpy array
        X = data.select(obs_cols).to_numpy().astype(np.float64)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize observables for better DMD convergence
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True) + 1e-10
        X_normalized = (X - X_mean) / X_std

        if len(X_normalized) < 10:
            # Not enough data for DMD - fall back to linear progression
            return self._fallback_compute(data)

        # Fit DMD or EDMD
        try:
            if self._use_edmd:
                dmd = ExtendedDMD(rank=min(10, len(obs_cols) * 2), dt=self._dt)
            else:
                dmd = DynamicModeDecomposition(rank=min(10, len(obs_cols)), dt=self._dt)

            dmd.fit(X_normalized)
            spectrum = dmd.get_spectrum(obs_cols)
        except Exception as e:
            # If DMD fails, fall back to simple method
            return self._fallback_compute(data, error=str(e))

        # Identify cell cycle mode using the analyzer
        analyzer = CellCycleKoopmanAnalyzer(
            expected_cycle_time=self._expected_cycle_time,
            frequency_tolerance=self._frequency_tolerance,
            dt=self._dt,
        )

        cell_cycle_modes = analyzer.identify_cell_cycle_modes(spectrum)

        if not cell_cycle_modes:
            # No cell cycle mode found - use dominant oscillatory mode
            oscillatory = spectrum.get_oscillatory_modes()
            if oscillatory:
                self._cell_cycle_mode = oscillatory[0]
            else:
                # No oscillatory modes - fall back
                return self._fallback_compute(data, error="No oscillatory modes found")
        else:
            # Use the fundamental cell cycle mode
            self._cell_cycle_mode = min(
                cell_cycle_modes,
                key=lambda m: np.abs(m.frequency - 1.0 / self._expected_cycle_time),
            )

        # Project data onto the cell cycle mode's eigenvector
        # The projection gives us a complex number at each time point
        mode_vector = self._cell_cycle_mode.mode
        mode_vector_norm = mode_vector / (np.linalg.norm(mode_vector) + 1e-10)

        # Project each observation onto the mode
        projections = X_normalized @ mode_vector_norm.real

        # For a true eigenfunction, we need the complex phase
        # Reconstruct using the eigenvalue evolution
        # φ(x_t) ≈ φ(x_0) * λ^t, so phase = angle of projection * λ^t
        eigenvalue = self._cell_cycle_mode.eigenvalue
        n_points = len(projections)

        # Compute phase by tracking eigenvalue evolution
        phases = np.zeros(n_points)
        for t in range(n_points):
            # Complex phase from eigenvalue evolution
            complex_val = projections[t] * (eigenvalue**t)
            phases[t] = np.angle(complex_val)

        # Normalize phase to [0, 1]
        # angle returns values in [-π, π], normalize to [0, 1]
        values = (phases + np.pi) / (2 * np.pi)
        values = np.clip(values, 0, 1)

        # Determine cell cycle phases based on position
        phase_labels = np.full(len(values), CellCyclePhase.UNKNOWN.value)
        phase_labels[values < 0.33] = CellCyclePhase.B_PERIOD.value
        phase_labels[(values >= 0.33) & (values < 0.67)] = CellCyclePhase.C_PERIOD.value
        phase_labels[values >= 0.67] = CellCyclePhase.D_PERIOD.value

        return CellCycleVariable(
            values=values,
            phase_labels=phase_labels,
            normalized=True,
            variable_name=self.name,
            metadata={
                "method": "koopman_eigenfunction_phase",
                "eigenvalue": complex(self._cell_cycle_mode.eigenvalue),
                "frequency": self._cell_cycle_mode.frequency,
                "period": self._cell_cycle_mode.period,
                "growth_rate": self._cell_cycle_mode.growth_rate,
                "expected_cycle_time": self._expected_cycle_time,
                "use_edmd": self._use_edmd,
            },
        )

    def _fallback_compute(
        self,
        data: pl.DataFrame,
        error: Optional[str] = None,
    ) -> CellCycleVariable:
        """
        Fallback computation when DMD cannot be performed.

        Uses simple time-based progression within each cell.
        """
        # Get time within each cell normalized to [0, 1]
        cell_groups = data.group_by(["generation", "agent_id"]).agg([
            pl.col("time").min().alias("t_start"),
            pl.col("time").max().alias("t_end"),
        ])

        data_with_bounds = data.join(
            cell_groups,
            on=["generation", "agent_id"],
            how="left",
        )

        t = data_with_bounds["time"].to_numpy()
        t_start = data_with_bounds["t_start"].to_numpy()
        t_end = data_with_bounds["t_end"].to_numpy()

        denom = t_end - t_start
        denom = np.where(denom > 0, denom, 1.0)

        values = (t - t_start) / denom
        values = np.clip(values, 0, 1)

        metadata = {
            "method": "time_fallback",
            "fallback_reason": error or "insufficient_data",
        }

        return CellCycleVariable(
            values=values,
            normalized=True,
            variable_name=self.name,
            metadata=metadata,
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
        "koopman": KoopmanCellCycleVariable,
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
                f"Unknown cell cycle variable type: {variable_type}. Available: {list(self.VARIABLES.keys())}"
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

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

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

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

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


class GSAInformedCellCycleVariable(CellCycleVariableComputer):
    """
    GSA-informed cell cycle variable per RFC006 Section 3.

    This class implements the full RFC006-compliant workflow:
    1. Run GSA for aggregation strategies 1-3 (uniform, by_generation, by_lineage_seed)
    2. Analyze variance decomposition to identify cell-cycle-relevant observables
    3. Use those observables to compute the Koopman cell cycle variable

    Per RFC006: "The choice of the 'cell cycle variable' will be informed by
    the sensitivity analyses (1-3), will be explored through dedicated
    visualisations, and will be discussed with all subteams."

    This class automates the "informed by sensitivity analyses" requirement.

    Example:
        >>> from uq import GSAInformedCellCycleVariable
        >>>
        >>> # Create with pre-computed aggregation results
        >>> gsa_cc = GSAInformedCellCycleVariable(
        ...     aggregated_uniform=agg_uniform,
        ...     aggregated_by_gen=agg_by_gen,
        ...     aggregated_by_seed=agg_by_seed,
        ...     observable_names=observable_names,
        ...     expected_cycle_time=3600.0,
        ... )
        >>>
        >>> # Compute cell cycle variable
        >>> cc_var = gsa_cc.compute(trajectory_data)
        >>>
        >>> # Access which observables were selected
        >>> print(gsa_cc.selected_observables)
        >>> print(gsa_cc.relevance_result)
    """

    def __init__(
        self,
        aggregated_uniform: "AggregatedOutput",
        aggregated_by_gen: "AggregatedOutput",
        aggregated_by_seed: "AggregatedOutput",
        observable_names: list[str],
        expected_cycle_time: float = 3600.0,
        min_observables: int = 2,
        max_observables: int = 10,
        min_residual_fraction: float = 0.1,
        dt: float = 1.0,
        use_edmd: bool = True,
    ):
        """
        Initialize GSA-informed cell cycle variable.

        Args:
            aggregated_uniform: Results from UNIFORM aggregation strategy
            aggregated_by_gen: Results from BY_GENERATION aggregation strategy
            aggregated_by_seed: Results from BY_LINEAGE_SEED aggregation strategy
            observable_names: Names of all available observables
            expected_cycle_time: Expected cell cycle duration in seconds
            min_observables: Minimum number of observables to select
            max_observables: Maximum number of observables to select
            min_residual_fraction: Minimum residual variance fraction for relevance
            dt: Timestep of trajectory data
            use_edmd: Whether to use Extended DMD
        """
        self._aggregated_uniform = aggregated_uniform
        self._aggregated_by_gen = aggregated_by_gen
        self._aggregated_by_seed = aggregated_by_seed
        self._observable_names = observable_names
        self._expected_cycle_time = expected_cycle_time
        self._min_observables = min_observables
        self._max_observables = max_observables
        self._min_residual_fraction = min_residual_fraction
        self._dt = dt
        self._use_edmd = use_edmd

        # These are populated after compute()
        self._relevance_result: Optional[CellCycleRelevanceResult] = None
        self._selected_observables: list[str] = []
        self._koopman_cc: Optional[KoopmanCellCycleVariable] = None

    @property
    def name(self) -> str:
        return "gsa_informed"

    @property
    def required_columns(self) -> list[str]:
        # Return selected observables if available, otherwise all
        if self._selected_observables:
            return self._selected_observables + ["generation", "agent_id", "time"]
        return self._observable_names + ["generation", "agent_id", "time"]

    @property
    def selected_observables(self) -> list[str]:
        """Observables selected by GSA analysis."""
        return self._selected_observables

    @property
    def relevance_result(self) -> Optional["CellCycleRelevanceResult"]:
        """Full GSA relevance analysis result."""
        return self._relevance_result

    @property
    def koopman_variable(self) -> Optional[KoopmanCellCycleVariable]:
        """The underlying Koopman cell cycle variable (after compute)."""
        return self._koopman_cc

    def _run_gsa_selection(self) -> None:
        """Run GSA to select relevant observables."""
        from uq.sensitivity import identify_cell_cycle_relevant_observables

        self._relevance_result = identify_cell_cycle_relevant_observables(
            aggregated_uniform=self._aggregated_uniform,
            aggregated_by_gen=self._aggregated_by_gen,
            aggregated_by_seed=self._aggregated_by_seed,
            observable_names=self._observable_names,
            min_residual_fraction=self._min_residual_fraction,
            top_n=self._max_observables,
        )

        # Get selected observables
        self._selected_observables = self._relevance_result.relevant_observables

        # Ensure minimum number
        if len(self._selected_observables) < self._min_observables:
            all_ranked = sorted(
                self._relevance_result.relevance_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            self._selected_observables = [name for name, _ in all_ranked[: self._min_observables]]

    def compute(
        self,
        data: pl.DataFrame,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ) -> CellCycleVariable:
        """
        Compute GSA-informed cell cycle variable.

        This method:
        1. Analyzes variance decomposition from strategies 1-3
        2. Selects observables with high residual (cell-cycle-related) variance
        3. Computes Koopman cell cycle variable using selected observables

        Args:
            data: Polars DataFrame with observable time series
            sim_data: Optional SimulationDataEcoli (not used)

        Returns:
            CellCycleVariable with GSA-informed computation
        """
        # Step 1: Run GSA selection if not already done
        if not self._selected_observables:
            self._run_gsa_selection()

        # Step 2: Create Koopman CC with selected observables
        # Filter to observables that exist in data
        available_obs = [col for col in self._selected_observables if col in data.columns]
        if len(available_obs) < 2:
            # Fall back to any numeric columns
            available_obs = [
                col
                for col in data.columns
                if col not in ["generation", "agent_id", "time", "lineage_seed"]
                and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ][: self._max_observables]

        self._koopman_cc = KoopmanCellCycleVariable(
            expected_cycle_time=self._expected_cycle_time,
            dt=self._dt,
            use_edmd=self._use_edmd,
            observable_columns=available_obs,
        )

        # Step 3: Compute cell cycle variable
        result = self._koopman_cc.compute(data, sim_data)

        # Add GSA metadata
        result.metadata["gsa_informed"] = True
        result.metadata["selected_observables"] = available_obs
        if self._relevance_result is not None:
            result.metadata["relevance_scores"] = {
                obs: self._relevance_result.relevance_scores.get(obs, 0.0) for obs in available_obs
            }
            result.metadata["variance_by_strategy"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self._relevance_result.variance_by_strategy.items()
            }

        return result

    def get_relevance_summary(self) -> dict:
        """
        Get a summary of the GSA relevance analysis.

        Returns:
            Dict with summary statistics and selected observables
        """
        if self._relevance_result is None:
            return {"error": "GSA analysis not yet run. Call compute() first."}

        return {
            "selected_observables": self._selected_observables,
            "n_selected": len(self._selected_observables),
            "top_relevance_scores": {
                obs: self._relevance_result.relevance_scores.get(obs, 0.0) for obs in self._selected_observables[:5]
            },
            "mean_residual_fraction": float(
                np.mean(self._relevance_result.residual_variance_fraction)
                if self._relevance_result.residual_variance_fraction is not None
                else 0.0
            ),
        }


# Register the GSA-informed variable
CellCycleAggregator.VARIABLES["gsa_informed"] = GSAInformedCellCycleVariable
