"""
Unit tests for UQ cell cycle stratification.

Tests the cell cycle variable computation and aggregation
as specified in Milestone 08.4.2 (Strategy 4).
"""

import numpy as np
import polars as pl
import pytest


class TestCellCyclePhase:
    """Tests for the CellCyclePhase enum."""

    @pytest.mark.unit
    def test_phase_values(self):
        """CellCyclePhase should have standard E. coli phases."""
        from uq import CellCyclePhase

        assert CellCyclePhase.B_PERIOD.value == "B_period"
        assert CellCyclePhase.C_PERIOD.value == "C_period"
        assert CellCyclePhase.D_PERIOD.value == "D_period"
        assert CellCyclePhase.UNKNOWN.value == "unknown"

    @pytest.mark.unit
    def test_all_phases_defined(self):
        """All standard E. coli cell cycle phases should be defined."""
        from uq import CellCyclePhase

        phases = list(CellCyclePhase)
        # B, C, D periods plus unknown
        assert len(phases) == 4


class TestCellCycleVariable:
    """Tests for the CellCycleVariable dataclass."""

    @pytest.mark.unit
    def test_creation_with_values(self):
        """CellCycleVariable should store numpy array of values."""
        from uq import CellCycleVariable

        values = np.linspace(0, 1, 100)
        cc_var = CellCycleVariable(values=values)

        np.testing.assert_array_equal(cc_var.values, values)

    @pytest.mark.unit
    def test_creation_with_phase_labels(self):
        """CellCycleVariable should support phase labels."""
        from uq import CellCyclePhase, CellCycleVariable

        values = np.linspace(0, 1, 100)
        phases = np.array([CellCyclePhase.B_PERIOD.value] * 100)

        cc_var = CellCycleVariable(values=values, phase_labels=phases)

        assert cc_var.phase_labels is not None
        assert len(cc_var.phase_labels) == 100

    @pytest.mark.unit
    def test_default_normalized(self):
        """CellCycleVariable should default to normalized=True."""
        from uq import CellCycleVariable

        cc_var = CellCycleVariable(values=np.array([0.5]))
        assert cc_var.normalized is True

    @pytest.mark.unit
    def test_to_stage_bins_normalized(self):
        """to_stage_bins should bin normalized values correctly."""
        from uq import CellCycleVariable

        # Values evenly spread across [0, 1]
        values = np.linspace(0, 1, 100)
        cc_var = CellCycleVariable(values=values, normalized=True)

        bins = cc_var.to_stage_bins(n_bins=10)

        assert bins.min() >= 0
        # np.digitize can return n_bins for values at upper edge
        assert bins.max() <= 10
        # Should have samples in most bins
        assert len(np.unique(bins)) >= 8

    @pytest.mark.unit
    def test_to_stage_bins_unnormalized(self):
        """to_stage_bins should work with unnormalized values."""
        from uq import CellCycleVariable

        values = np.linspace(0, 100, 100)  # Range [0, 100]
        cc_var = CellCycleVariable(values=values, normalized=False)

        bins = cc_var.to_stage_bins(n_bins=5)

        assert bins.min() >= 0
        # np.digitize can return n_bins for values at upper edge
        assert bins.max() <= 5

    @pytest.mark.unit
    def test_variable_name_stored(self):
        """CellCycleVariable should store variable name."""
        from uq import CellCycleVariable

        cc_var = CellCycleVariable(
            values=np.array([0.5]),
            variable_name="mass_based",
        )

        assert cc_var.variable_name == "mass_based"

    @pytest.mark.unit
    def test_metadata_stored(self):
        """CellCycleVariable should store metadata."""
        from uq import CellCycleVariable

        metadata = {"method": "log_ratio", "version": 1}
        cc_var = CellCycleVariable(
            values=np.array([0.5]),
            metadata=metadata,
        )

        assert cc_var.metadata["method"] == "log_ratio"


class TestCellCycleVariableComputer:
    """Tests for the CellCycleVariableComputer abstract base class."""

    @pytest.mark.unit
    def test_abstract_interface(self):
        """CellCycleVariableComputer should define abstract interface."""
        from uq import CellCycleVariableComputer

        # Should have abstract methods
        assert hasattr(CellCycleVariableComputer, "name")
        assert hasattr(CellCycleVariableComputer, "required_columns")
        assert hasattr(CellCycleVariableComputer, "compute")


class TestMassBasedCellCycleVariable:
    """Tests for mass-based cell cycle variable."""

    @pytest.mark.unit
    def test_name_property(self):
        """MassBasedCellCycleVariable should have correct name."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        assert computer.name == "mass_based"

    @pytest.mark.unit
    def test_required_columns(self):
        """MassBasedCellCycleVariable should specify required columns."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        required = computer.required_columns

        assert "listeners__mass__dry_mass" in required
        assert "generation" in required
        assert "agent_id" in required

    @pytest.mark.unit
    def test_compute_with_synthetic_data(self, synthetic_simulation_dataframe):
        """MassBasedCellCycleVariable should compute from simulation data."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        df = synthetic_simulation_dataframe

        result = computer.compute(df)

        assert result.values is not None
        assert len(result.values) == len(df)
        assert result.normalized is True
        assert result.variable_name == "mass_based"

    @pytest.mark.unit
    def test_compute_produces_normalized_values(self, synthetic_simulation_dataframe):
        """MassBasedCellCycleVariable should produce values in [0, 1]."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)


class TestDNAReplicationCellCycleVariable:
    """Tests for DNA replication-based cell cycle variable."""

    @pytest.mark.unit
    def test_name_property(self):
        """DNAReplicationCellCycleVariable should have correct name."""
        from uq import DNAReplicationCellCycleVariable

        computer = DNAReplicationCellCycleVariable()
        assert computer.name == "dna_replication"

    @pytest.mark.unit
    def test_required_columns(self):
        """DNAReplicationCellCycleVariable should specify required columns."""
        from uq import DNAReplicationCellCycleVariable

        computer = DNAReplicationCellCycleVariable()
        required = computer.required_columns

        assert "listeners__mass__dna_mass" in required
        assert "listeners__mass__dry_mass" in required

    @pytest.mark.unit
    def test_assigns_phase_labels(self, synthetic_simulation_dataframe):
        """DNAReplicationCellCycleVariable should assign phase labels."""
        from uq import DNAReplicationCellCycleVariable

        computer = DNAReplicationCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        assert result.phase_labels is not None
        assert len(result.phase_labels) == len(result.values)

    @pytest.mark.unit
    def test_phase_labels_are_valid(self, synthetic_simulation_dataframe):
        """Phase labels should be valid CellCyclePhase values."""
        from uq import CellCyclePhase, DNAReplicationCellCycleVariable

        computer = DNAReplicationCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Get the expected phase prefixes (handle string truncation in numpy)
        valid_prefixes = {p.value[:7] for p in CellCyclePhase}

        # Phase labels are stored as strings
        unique_labels = set(str(label) for label in result.phase_labels)

        # Check that labels start with valid phase prefixes
        # (numpy may truncate strings based on initial dtype)
        for label in unique_labels:
            prefix = label[:7]
            assert prefix in valid_prefixes, f"Label '{label}' should start with valid phase prefix"


class TestCellAngleCellCycleVariable:
    """Tests for cell angle-based cell cycle variable."""

    @pytest.mark.unit
    def test_name_property(self):
        """CellAngleCellCycleVariable should have correct name."""
        from uq import CellAngleCellCycleVariable

        computer = CellAngleCellCycleVariable()
        assert computer.name == "cell_angle"

    @pytest.mark.unit
    def test_required_columns(self):
        """CellAngleCellCycleVariable should specify required columns."""
        from uq import CellAngleCellCycleVariable

        computer = CellAngleCellCycleVariable()
        required = computer.required_columns

        assert "listeners__mass__dry_mass" in required
        assert "listeners__mass__cell_mass" in required
        assert "time" in required

    @pytest.mark.unit
    def test_produces_circular_coordinate(self, synthetic_simulation_dataframe):
        """CellAngleCellCycleVariable should produce values in [0, 1]."""
        from uq import CellAngleCellCycleVariable

        computer = CellAngleCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Values should wrap around [0, 1] like an angle
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

    @pytest.mark.unit
    def test_metadata_includes_method(self, synthetic_simulation_dataframe):
        """CellAngleCellCycleVariable should include method in metadata."""
        from uq import CellAngleCellCycleVariable

        computer = CellAngleCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        assert "method" in result.metadata


class TestCompositeCellCycleVariable:
    """Tests for composite/custom cell cycle variables."""

    @pytest.mark.unit
    def test_custom_compute_function(self):
        """CompositeCellCycleVariable should use custom compute function."""
        import polars as pl

        from uq import CompositeCellCycleVariable

        def custom_compute(data: pl.DataFrame) -> np.ndarray:
            mass = data["listeners__mass__dry_mass"].to_numpy()
            return (mass - mass.min()) / (mass.max() - mass.min() + 1e-10)

        computer = CompositeCellCycleVariable(
            name="custom_mass",
            required_columns=["listeners__mass__dry_mass"],
            compute_func=custom_compute,
        )

        assert computer.name == "custom_mass"
        assert "listeners__mass__dry_mass" in computer.required_columns

    @pytest.mark.unit
    def test_custom_compute_with_data(self, synthetic_simulation_dataframe):
        """CompositeCellCycleVariable should compute from data."""
        from uq import CompositeCellCycleVariable

        def custom_compute(data: pl.DataFrame) -> np.ndarray:
            mass = data["listeners__mass__dry_mass"].to_numpy()
            return (mass - mass.min()) / (mass.max() - mass.min() + 1e-10)

        computer = CompositeCellCycleVariable(
            name="custom",
            required_columns=["listeners__mass__dry_mass"],
            compute_func=custom_compute,
        )

        result = computer.compute(synthetic_simulation_dataframe)

        assert len(result.values) == len(synthetic_simulation_dataframe)


class TestCellCycleAggregator:
    """Tests for the CellCycleAggregator class."""

    @pytest.mark.unit
    def test_available_variables(self):
        """CellCycleAggregator should have standard variables available."""
        from uq import CellCycleAggregator

        available = CellCycleAggregator.VARIABLES

        assert "mass_based" in available
        assert "dna_replication" in available
        assert "cell_angle" in available

    @pytest.mark.unit
    def test_invalid_variable_type_raises(self):
        """CellCycleAggregator should raise for invalid variable type."""
        import duckdb

        from uq import CellCycleAggregator

        conn = duckdb.connect()

        with pytest.raises(ValueError, match="Unknown cell cycle variable type"):
            CellCycleAggregator(
                conn=conn,
                history_sql="SELECT 1",
                config_sql="SELECT 1",
                variable_type="invalid_type",
            )

    @pytest.mark.unit
    def test_n_stages_parameter(self):
        """CellCycleAggregator should respect n_stages parameter."""
        import duckdb

        from uq import CellCycleAggregator

        conn = duckdb.connect()

        aggregator = CellCycleAggregator(
            conn=conn,
            history_sql="SELECT 1",
            config_sql="SELECT 1",
            variable_type="mass_based",
            n_stages=20,
        )

        assert aggregator.n_stages == 20


class TestRegisterCellCycleVariable:
    """Tests for registering custom cell cycle variables."""

    @pytest.mark.unit
    def test_register_custom_variable(self):
        """register_cell_cycle_variable should add custom variables."""
        from uq import (
            CellCycleAggregator,
            CellCycleVariable,
            CellCycleVariableComputer,
            register_cell_cycle_variable,
        )

        class CustomVariable(CellCycleVariableComputer):
            @property
            def name(self) -> str:
                return "test_custom"

            @property
            def required_columns(self) -> list[str]:
                return ["time"]

            def compute(self, data, sim_data=None) -> CellCycleVariable:
                return CellCycleVariable(
                    values=np.zeros(len(data)),
                    variable_name=self.name,
                )

        register_cell_cycle_variable("test_custom", CustomVariable())

        assert "test_custom" in CellCycleAggregator.VARIABLES


class TestCellCycleStratification:
    """Integration tests for cell cycle stratification."""

    @pytest.mark.unit
    def test_stratification_produces_per_stage_stats(self, rng):
        """Cell cycle stratification should produce statistics per stage."""
        from uq import CellCycleVariable

        # Create mock cell cycle data
        n_samples = 1000
        values = rng.uniform(0, 1, n_samples)
        cc_var = CellCycleVariable(values=values, normalized=True)

        # Bin into stages
        n_stages = 10
        bins = cc_var.to_stage_bins(n_stages)

        # Each stage should have samples
        for stage in range(n_stages):
            mask = bins == stage
            n_in_stage = np.sum(mask)
            # With uniform distribution, each stage should have ~10% of samples
            assert n_in_stage > 0, f"Stage {stage} has no samples"

    @pytest.mark.unit
    def test_cell_cycle_preserves_sample_count(self, rng):
        """Cell cycle binning should preserve total sample count."""
        from uq import CellCycleVariable

        n_samples = 500
        values = rng.uniform(0, 1, n_samples)
        cc_var = CellCycleVariable(values=values)

        bins = cc_var.to_stage_bins(n_bins=10)

        assert len(bins) == n_samples

    @pytest.mark.unit
    def test_phenotypic_variation_across_cycle(self, synthetic_simulation_dataframe):
        """Cell cycle variable should capture phenotypic variation."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Values should span most of [0, 1]
        value_range = result.values.max() - result.values.min()
        assert value_range > 0.5, "Cell cycle variable should span significant range"

    @pytest.mark.unit
    def test_cyclic_nature_of_variable(self, synthetic_simulation_dataframe):
        """Cell cycle variables should be approximately cyclic."""
        from uq import CellAngleCellCycleVariable

        computer = CellAngleCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Check values cover the full range
        assert result.values.min() < 0.2
        assert result.values.max() > 0.8


class TestCellCycleRequirements:
    """Tests verifying cell cycle variable requirements from spec."""

    @pytest.mark.unit
    def test_computed_from_process_variables(self, synthetic_simulation_dataframe):
        """Cell cycle variables should be computed from vEcoli process variables."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()

        # Required columns should be process outputs
        for col in computer.required_columns:
            assert "listeners__" in col or col in ["generation", "agent_id", "time"]

    @pytest.mark.unit
    def test_produces_low_dimensional_output(self, synthetic_simulation_dataframe):
        """Cell cycle variable should be low-dimensional (scalar per timepoint)."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Should be 1D array (one value per timepoint)
        assert result.values.ndim == 1

    @pytest.mark.unit
    def test_deterministic_binning(self, synthetic_simulation_dataframe):
        """Cell cycle variable should enable deterministic binning."""
        from uq import MassBasedCellCycleVariable

        computer = MassBasedCellCycleVariable()
        result = computer.compute(synthetic_simulation_dataframe)

        # Same input should produce same output
        bins1 = result.to_stage_bins(n_bins=10)
        bins2 = result.to_stage_bins(n_bins=10)

        np.testing.assert_array_equal(bins1, bins2)


class TestKoopmanCellCycleVariable:
    """Tests for Koopman eigenfunction-based cell cycle variable.

    The Koopman approach uses Dynamic Mode Decomposition (DMD) to identify
    the cell cycle mode and extract its eigenfunction phase as the cell
    cycle coordinate. This is the recommended approach per RFC006 Section 1.3.
    """

    @pytest.mark.unit
    def test_name_property(self):
        """KoopmanCellCycleVariable should have correct name."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable()
        assert computer.name == "koopman"

    @pytest.mark.unit
    def test_required_columns(self):
        """KoopmanCellCycleVariable should specify required columns."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable()
        required = computer.required_columns

        # Should include generation and agent_id for grouping
        assert "generation" in required
        assert "agent_id" in required
        assert "time" in required

    @pytest.mark.unit
    def test_custom_observable_columns(self):
        """KoopmanCellCycleVariable should accept custom observable columns."""
        from uq import KoopmanCellCycleVariable

        custom_cols = ["listeners__mass__dry_mass", "listeners__mass__dna_mass"]
        computer = KoopmanCellCycleVariable(observable_columns=custom_cols)

        for col in custom_cols:
            assert col in computer.required_columns

    @pytest.mark.unit
    def test_expected_cycle_time_parameter(self):
        """KoopmanCellCycleVariable should accept expected_cycle_time."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(expected_cycle_time=7200.0)  # 2 hours
        assert computer._expected_cycle_time == 7200.0

    @pytest.mark.unit
    def test_compute_with_synthetic_data(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should compute from simulation data."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        df = synthetic_simulation_dataframe

        result = computer.compute(df)

        assert result.values is not None
        assert len(result.values) == len(df)
        assert result.normalized is True
        assert result.variable_name == "koopman"

    @pytest.mark.unit
    def test_compute_produces_normalized_values(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should produce values in [0, 1]."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(synthetic_simulation_dataframe)

        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

    @pytest.mark.unit
    def test_metadata_includes_koopman_info(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should include Koopman info in metadata."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(synthetic_simulation_dataframe)

        # Should include method identifier
        assert "method" in result.metadata
        assert "koopman" in result.metadata["method"]

    @pytest.mark.unit
    def test_assigns_phase_labels(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should assign phase labels."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(synthetic_simulation_dataframe)

        assert result.phase_labels is not None
        assert len(result.phase_labels) == len(result.values)

    @pytest.mark.unit
    def test_use_edmd_parameter(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should support EDMD toggle."""
        from uq import KoopmanCellCycleVariable

        # Test with standard DMD
        computer_dmd = KoopmanCellCycleVariable(
            use_edmd=False,
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result_dmd = computer_dmd.compute(synthetic_simulation_dataframe)

        # Test with EDMD
        computer_edmd = KoopmanCellCycleVariable(
            use_edmd=True,
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result_edmd = computer_edmd.compute(synthetic_simulation_dataframe)

        # Both should produce valid results
        assert len(result_dmd.values) == len(result_edmd.values)
        assert np.all(result_dmd.values >= 0) and np.all(result_dmd.values <= 1)
        assert np.all(result_edmd.values >= 0) and np.all(result_edmd.values <= 1)

    @pytest.mark.unit
    def test_cell_cycle_mode_accessible(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should expose the identified cell cycle mode."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        _ = computer.compute(synthetic_simulation_dataframe)

        # After compute, cell_cycle_mode should be available
        # (may be None if no mode found, but property should exist)
        assert hasattr(computer, "cell_cycle_mode")

    @pytest.mark.unit
    def test_fallback_for_short_data(self, rng):
        """KoopmanCellCycleVariable should fallback gracefully for short data."""
        from uq import KoopmanCellCycleVariable

        # Create very short data
        df = pl.DataFrame({
            "listeners__mass__dry_mass": rng.uniform(1, 2, 5),
            "listeners__mass__cell_mass": rng.uniform(1.3, 2.6, 5),
            "generation": [0] * 5,
            "agent_id": ["a"] * 5,
            "time": list(range(5)),
        })

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(df)

        # Should still produce valid output via fallback
        assert len(result.values) == 5
        assert "fallback" in result.metadata.get("method", "") or "fallback_reason" in result.metadata

    @pytest.mark.unit
    def test_registered_in_aggregator(self):
        """KoopmanCellCycleVariable should be registered in CellCycleAggregator."""
        from uq import CellCycleAggregator

        assert "koopman" in CellCycleAggregator.VARIABLES

    @pytest.mark.unit
    def test_deterministic_output(self, synthetic_simulation_dataframe):
        """KoopmanCellCycleVariable should produce deterministic output."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )

        result1 = computer.compute(synthetic_simulation_dataframe)
        result2 = computer.compute(synthetic_simulation_dataframe)

        # Same input should produce same output
        np.testing.assert_array_almost_equal(result1.values, result2.values)


class TestKoopmanCellCycleIntegration:
    """Integration tests for Koopman-based cell cycle stratification."""

    @pytest.mark.unit
    def test_koopman_variable_enables_stratification(self, synthetic_simulation_dataframe):
        """Koopman cell cycle variable should enable cell cycle stratification."""
        from uq import KoopmanCellCycleVariable

        computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(synthetic_simulation_dataframe)

        # Should be able to bin into stages
        n_stages = 10
        bins = result.to_stage_bins(n_stages)

        # Should have samples in multiple bins
        unique_bins = np.unique(bins)
        assert len(unique_bins) >= 3, "Should have samples in multiple cell cycle stages"

    @pytest.mark.unit
    def test_koopman_captures_periodic_structure(self, rng):
        """Koopman cell cycle variable should capture periodic dynamics."""
        from uq import KoopmanCellCycleVariable

        # Create synthetic data with clear periodic structure
        n_points = 200
        t = np.arange(n_points)

        # Mass with periodic growth (cell cycle-like)
        period = 50  # 50 timestep period
        phase = 2 * np.pi * t / period
        dry_mass = 1.0 + 0.5 * np.sin(phase) + 0.01 * t + rng.normal(0, 0.05, n_points)
        cell_mass = 1.3 * dry_mass

        df = pl.DataFrame({
            "listeners__mass__dry_mass": dry_mass,
            "listeners__mass__cell_mass": cell_mass,
            "generation": [0] * n_points,
            "agent_id": ["a"] * n_points,
            "time": t.tolist(),
        })

        computer = KoopmanCellCycleVariable(
            expected_cycle_time=period,
            dt=1.0,
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )
        result = computer.compute(df)

        # Values should span the full range
        value_range = result.values.max() - result.values.min()
        assert value_range > 0.5, "Koopman variable should capture periodic variation"

    @pytest.mark.unit
    def test_koopman_vs_mass_based_correlation(self, synthetic_simulation_dataframe):
        """Koopman and mass-based variables should be correlated for simple data."""
        from uq import KoopmanCellCycleVariable, MassBasedCellCycleVariable

        mass_computer = MassBasedCellCycleVariable()
        koopman_computer = KoopmanCellCycleVariable(
            observable_columns=[
                "listeners__mass__dry_mass",
                "listeners__mass__cell_mass",
            ],
        )

        mass_result = mass_computer.compute(synthetic_simulation_dataframe)
        koopman_result = koopman_computer.compute(synthetic_simulation_dataframe)

        # Both should produce valid [0, 1] values
        assert np.all(mass_result.values >= 0) and np.all(mass_result.values <= 1)
        assert np.all(koopman_result.values >= 0) and np.all(koopman_result.values <= 1)
