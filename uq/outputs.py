"""
Output variable extraction for uncertainty quantification.

This module defines functions to extract the scientifically most relevant
output variables from vEcoli simulations:
- Transcriptome (mRNA counts)
- Proteome (protein/monomer counts)
- Metabolic fluxes (particularly exchange fluxes)
- Higher-order properties (mass, growth rate, etc.)

These outputs are extracted from Parquet-emitted simulation data using DuckDB.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from duckdb import DuckDBPyConnection

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import (
        SimulationDataEcoli,  # TODO: ecoli.library.sim_data.LoadSimData.sim_data instead!
    )


class OutputType(str, Enum):
    """Types of output variables that can be extracted."""

    TRANSCRIPTOME = "transcriptome"
    PROTEOME = "proteome"
    METABOLIC_FLUXES = "metabolic_fluxes"
    EXCHANGE_FLUXES = "exchange_fluxes"
    HIGHER_ORDER_PROPERTIES = "higher_order_properties"
    METABOLOME = "metabolome"


@dataclass
class OutputVariables:
    """
    Container for extracted output variables.

    Attributes:
        transcriptome: mRNA counts per cistron, shape (n_timepoints, n_cistrons)
        proteome: Protein monomer counts, shape (n_timepoints, n_monomers)
        metabolic_fluxes: All reaction fluxes, shape (n_timepoints, n_reactions)
        exchange_fluxes: Exchange reaction fluxes, shape (n_timepoints, n_exchange)
        higher_order_properties: Dict of scalar time series (mass, growth_rate, etc.)
        metadata: Additional metadata (gene IDs, reaction IDs, etc.)
    """

    transcriptome: Optional[np.ndarray] = None
    proteome: Optional[np.ndarray] = None
    metabolic_fluxes: Optional[np.ndarray] = None
    exchange_fluxes: Optional[np.ndarray] = None
    higher_order_properties: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_array(self) -> np.ndarray:
        """
        Flatten all outputs into a single 1D array for sensitivity analysis.

        Returns:
            1D numpy array containing all output values
        """
        arrays = []
        if self.transcriptome is not None:
            arrays.append(self.transcriptome.flatten())
        if self.proteome is not None:
            arrays.append(self.proteome.flatten())
        if self.metabolic_fluxes is not None:
            arrays.append(self.metabolic_fluxes.flatten())
        if self.exchange_fluxes is not None:
            arrays.append(self.exchange_fluxes.flatten())
        for prop in self.higher_order_properties.values():
            arrays.append(prop.flatten())

        if not arrays:
            return np.array([])
        return np.concatenate(arrays)


class OutputExtractor:
    """
    Extracts output variables from simulation data stored in Parquet format.

    This class uses DuckDB to efficiently query and aggregate simulation outputs.
    """

    # Column names in Parquet files
    TRANSCRIPTOME_COL = "listeners__rna_counts__mRNA_cistron_counts"
    PROTEOME_COL = "listeners__monomer_counts"
    FLUX_COL = "listeners__fba_results__base_reaction_fluxes"
    CELL_MASS_COL = "listeners__mass__cell_mass"
    DRY_MASS_COL = "listeners__mass__dry_mass"
    VOLUME_COL = "listeners__mass__volume"
    DNA_MASS_COL = "listeners__mass__dna_mass"
    RNA_MASS_COL = "listeners__mass__rna_mass"
    PROTEIN_MASS_COL = "listeners__mass__protein_mass"

    # ID columns for grouping
    ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]

    def __init__(
        self,
        conn: DuckDBPyConnection,
        history_sql: str,
        config_sql: str,
        sim_data: Optional["SimulationDataEcoli"] = None,
    ):
        """
        Initialize the output extractor.

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

    def extract_transcriptome(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract mRNA cistron counts from simulation data.

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (counts array, cistron IDs)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        mrna_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=self.TRANSCRIPTOME_COL,
        )

        history_subquery = read_stacked_columns(
            self.history_sql,
            [f"{self.TRANSCRIPTOME_COL} AS mrna_counts"],
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
            SELECT
                mrna_counts,
                {", ".join(self.ID_COLS)},
                time
            FROM filtered
            ORDER BY {", ".join(self.ID_COLS)}, time
        """

        result = self.conn.sql(query).pl()

        if result.is_empty():
            return np.array([]), mrna_ids

        # Convert list column to numpy array
        counts = np.vstack(result["mrna_counts"].to_numpy())
        return counts, mrna_ids

    def extract_proteome(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract protein monomer counts from simulation data.

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (counts array, monomer IDs)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        monomer_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=self.PROTEOME_COL,
        )

        history_subquery = read_stacked_columns(
            self.history_sql,
            [f"{self.PROTEOME_COL} AS monomer_counts"],
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
            SELECT
                monomer_counts,
                {", ".join(self.ID_COLS)},
                time
            FROM filtered
            ORDER BY {", ".join(self.ID_COLS)}, time
        """

        result = self.conn.sql(query).pl()

        if result.is_empty():
            return np.array([]), monomer_ids

        counts = np.vstack(result["monomer_counts"].to_numpy())
        return counts, monomer_ids

    def extract_metabolic_fluxes(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
        normalize_by_mass: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract metabolic reaction fluxes from simulation data.

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards
            normalize_by_mass: If True, normalize fluxes by dry mass

        Returns:
            Tuple of (flux array, reaction IDs)
        """
        from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns

        rxn_ids = field_metadata(
            conn=self.conn,
            config_subquery=self.config_sql,
            field=self.FLUX_COL,
        )

        columns = [f"{self.FLUX_COL} AS fluxes"]
        if normalize_by_mass:
            columns.extend([
                f"{self.CELL_MASS_COL} AS cell_mass",
                f"{self.DRY_MASS_COL} AS dry_mass",
            ])

        history_subquery = read_stacked_columns(
            self.history_sql,
            columns,
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        if normalize_by_mass:
            # Get cell density from sim_data if available
            cell_density = 1100.0  # Default g/L
            if self.sim_data is not None:
                from wholecell.utils import units

                cd = self.sim_data.constants.cell_density
                cell_density = cd.asNumber(units.g / units.L)

            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *,
                        dry_mass / cell_mass * {cell_density} AS conversion_coeff
                    FROM history
                    {filter_clause}
                )
                SELECT
                    list_transform(fluxes, x -> x / conversion_coeff) AS fluxes,
                    {", ".join(self.ID_COLS)},
                    time
                FROM filtered
                ORDER BY {", ".join(self.ID_COLS)}, time
            """
        else:
            query = f"""
                WITH history AS ({history_subquery}),
                filtered AS (
                    SELECT *
                    FROM history
                    {filter_clause}
                )
                SELECT
                    fluxes,
                    {", ".join(self.ID_COLS)},
                    time
                FROM filtered
                ORDER BY {", ".join(self.ID_COLS)}, time
            """

        result = self.conn.sql(query).pl()

        if result.is_empty():
            return np.array([]), rxn_ids

        fluxes = np.vstack(result["fluxes"].to_numpy())
        return fluxes, rxn_ids

    def extract_exchange_fluxes(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract exchange reaction fluxes from simulation data.

        Exchange reactions are identified by their reaction IDs (typically
        containing "EX_" prefix).

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Tuple of (flux array for exchange reactions, exchange reaction IDs)
        """
        fluxes, rxn_ids = self.extract_metabolic_fluxes(generation_lower_bound, time_lower_bound)

        if fluxes.size == 0:
            return np.array([]), []

        # Find exchange reaction indices
        exchange_mask = np.array(["EX_" in rxn_id for rxn_id in rxn_ids])
        exchange_ids = [rxn_id for rxn_id in rxn_ids if "EX_" in rxn_id]

        return fluxes[:, exchange_mask], exchange_ids

    def extract_higher_order_properties(
        self,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract higher-order properties (mass, volume, etc.) from simulation data.

        Args:
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            Dictionary mapping property names to time series arrays
        """
        from ecoli.library.parquet_emitter import read_stacked_columns

        property_cols = [
            f"{self.CELL_MASS_COL} AS cell_mass",
            f"{self.DRY_MASS_COL} AS dry_mass",
            f"{self.VOLUME_COL} AS volume",
        ]

        history_subquery = read_stacked_columns(
            self.history_sql,
            property_cols,
            order_results=False,
        )

        filter_clause = self._build_filter_clause(generation_lower_bound, time_lower_bound)

        # Calculate growth rate from dry mass
        query = f"""
            WITH history AS ({history_subquery}),
            filtered AS (
                SELECT *
                FROM history
                {filter_clause}
            ),
            with_growth AS (
                SELECT
                    *,
                    (dry_mass - LAG(dry_mass) OVER (
                        PARTITION BY {", ".join(self.ID_COLS)}
                        ORDER BY time
                    )) / dry_mass AS growth_rate
                FROM filtered
            )
            SELECT
                cell_mass,
                dry_mass,
                volume,
                COALESCE(growth_rate, 0) AS growth_rate,
                {", ".join(self.ID_COLS)},
                time
            FROM with_growth
            ORDER BY {", ".join(self.ID_COLS)}, time
        """

        result = self.conn.sql(query).pl()

        if result.is_empty():
            return {}

        return {
            "cell_mass": result["cell_mass"].to_numpy(),
            "dry_mass": result["dry_mass"].to_numpy(),
            "volume": result["volume"].to_numpy(),
            "growth_rate": result["growth_rate"].to_numpy(),
        }

    def extract_all(
        self,
        output_types: Optional[list[OutputType]] = None,
        generation_lower_bound: Optional[int] = None,
        time_lower_bound: Optional[float] = None,
    ) -> OutputVariables:
        """
        Extract all specified output types into an OutputVariables container.

        Args:
            output_types: List of output types to extract. If None, extracts all.
            generation_lower_bound: Only include data from this generation onwards
            time_lower_bound: Only include data from this time onwards

        Returns:
            OutputVariables container with extracted data
        """
        if output_types is None:
            output_types = list(OutputType)

        outputs = OutputVariables()

        if OutputType.TRANSCRIPTOME in output_types:
            counts, ids = self.extract_transcriptome(generation_lower_bound, time_lower_bound)
            outputs.transcriptome = counts
            outputs.metadata["cistron_ids"] = ids

        if OutputType.PROTEOME in output_types:
            counts, ids = self.extract_proteome(generation_lower_bound, time_lower_bound)
            outputs.proteome = counts
            outputs.metadata["monomer_ids"] = ids

        if OutputType.METABOLIC_FLUXES in output_types:
            fluxes, ids = self.extract_metabolic_fluxes(generation_lower_bound, time_lower_bound)
            outputs.metabolic_fluxes = fluxes
            outputs.metadata["reaction_ids"] = ids

        if OutputType.EXCHANGE_FLUXES in output_types:
            fluxes, ids = self.extract_exchange_fluxes(generation_lower_bound, time_lower_bound)
            outputs.exchange_fluxes = fluxes
            outputs.metadata["exchange_reaction_ids"] = ids

        if OutputType.HIGHER_ORDER_PROPERTIES in output_types:
            outputs.higher_order_properties = self.extract_higher_order_properties(
                generation_lower_bound, time_lower_bound
            )

        return outputs

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


def get_output_variable_info(
    sim_data: "SimulationDataEcoli",
) -> dict[str, dict[str, Any]]:
    """
    Get information about available output variables from sim_data.

    Args:
        sim_data: SimulationDataEcoli instance

    Returns:
        Dictionary with information about each output type
    """
    info: dict[str, dict[str, Any]] = {}

    # Transcriptome info
    cistron_data = sim_data.process.transcription.cistron_data.struct_array
    info["transcriptome"] = {
        "n_cistrons": len(cistron_data),
        "cistron_ids": cistron_data["id"].tolist(),
    }

    # Proteome info
    monomer_data = sim_data.process.translation.monomer_data.struct_array
    info["proteome"] = {
        "n_monomers": len(monomer_data),
        "monomer_ids": monomer_data["id"].tolist(),
    }

    # Flux info
    rxn_ids = sim_data.process.metabolism.reaction_stoich().keys()
    exchange_ids = [rxn_id for rxn_id in rxn_ids if "EX_" in rxn_id]
    info["metabolic_fluxes"] = {
        "n_reactions": len(rxn_ids),
        "n_exchange_reactions": len(exchange_ids),
    }

    # Higher-order properties
    info["higher_order_properties"] = {
        "properties": ["cell_mass", "dry_mass", "volume", "growth_rate"],
    }

    return info
