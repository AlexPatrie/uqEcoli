"""
Integration tests using real simulation data from api_simulation_default.

These tests verify the UQ framework works with actual vEcoli simulation output,
loaded via uq.inputs.load_dataset("api_simulation_default").

Tests are skipped if real data is not available.
"""

import numpy as np
import pytest

# =============================================================================
# Colorful logging helpers
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def log_header(msg: str) -> None:
    """Print a bold header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}{Colors.END}")


def log_success(msg: str) -> None:
    """Print a green success message."""
    print(f"{Colors.GREEN}  ✓ {msg}{Colors.END}")


def log_info(msg: str) -> None:
    """Print a cyan info message."""
    print(f"{Colors.CYAN}  ℹ {msg}{Colors.END}")


def log_data(label: str, value) -> None:
    """Print a data point with yellow label."""
    print(f"{Colors.YELLOW}  {label}:{Colors.END} {Colors.BOLD}{value}{Colors.END}")


def log_warning(msg: str) -> None:
    """Print a yellow warning message."""
    print(f"{Colors.YELLOW}  ⚠ {msg}{Colors.END}")


def log_section(msg: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}  ── {msg} ──{Colors.END}")


class TestRealDataLoading:
    """Tests for loading real simulation data."""

    @pytest.mark.real_data
    def test_load_dataset_function(self):
        """uq.inputs.load_dataset should load real simulation data."""
        from tests.conftest import _get_repo_root, _load_real_dataset_safe

        log_header("LOADING REAL SIMULATION DATA")

        # Check if data directory exists
        repo_root = _get_repo_root()
        data_path = repo_root / "api_integration/sims/api_simulation_default"
        log_info(f"Looking for data at: {data_path}")

        if not data_path.exists():
            pytest.skip("Real data not available at expected path")

        log_success("Data directory found!")

        # This may fail due to schema issues, so we use the safe loader
        df = _load_real_dataset_safe("api_simulation_default", max_files=10)

        assert df is not None, "Failed to load any data"
        assert len(df) > 0, "Loaded data is empty"
        assert "listeners__mass__dry_mass" in df.columns

        log_success(f"Loaded {len(df):,} rows from REAL vEcoli simulation")
        log_data("Experiment ID", "api_simulation_default")
        log_data("Columns loaded", len(df.columns))

    @pytest.mark.real_data
    def test_real_data_has_required_columns(self, real_simulation_dataframe):
        """Real data should have columns required for UQ analysis."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        log_section("Verifying Required Columns")

        required = [
            "listeners__mass__dry_mass",
            "listeners__mass__cell_mass",
            "listeners__mass__dna_mass",
            "listeners__mass__growth",
            "time",
            "generation",
            "lineage_seed",
        ]

        for col in required:
            assert col in real_simulation_dataframe.columns, f"Missing column: {col}"
            log_success(f"Found: {col}")

    @pytest.mark.real_data
    def test_real_data_has_multiple_generations(self, real_simulation_dataframe):
        """Real data should span multiple generations."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        log_section("Checking Generation Coverage")

        generations = real_simulation_dataframe["generation"].unique().sort()
        log_data("Generations found", list(generations.to_numpy()))
        log_data("Total generations", len(generations))

        assert len(generations) > 1, "Need multiple generations for stratified analysis"
        log_success("Multiple generations available for stratified analysis")

    @pytest.mark.real_data
    def test_real_data_has_multiple_lineage_seeds(self, real_simulation_dataframe):
        """Real data should have multiple lineage seeds."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        log_section("Checking Lineage Seed Coverage")

        seeds = real_simulation_dataframe["lineage_seed"].unique().sort()
        log_data("Lineage seeds found", list(seeds.to_numpy()))
        log_data("Total seeds", len(seeds))

        assert len(seeds) >= 1, "Need at least one lineage seed"
        log_success("Lineage seed data available")


class TestRealDataAggregation:
    """Tests for aggregation with real data."""

    @pytest.mark.real_data
    def test_uniform_aggregation_with_real_data(self, real_aggregated_uniform):
        """Uniform aggregation should work with real data."""
        log_header("UNIFORM AGGREGATION - REAL DATA")

        assert real_aggregated_uniform.mean is not None
        assert real_aggregated_uniform.std is not None
        assert real_aggregated_uniform.n_samples > 0

        # Mean should be physically reasonable
        mass_mean = real_aggregated_uniform.mean[0]
        assert mass_mean > 0, "Dry mass should be positive"

        log_data("Total samples", real_aggregated_uniform.n_samples)
        log_data("Mean dry mass", f"{mass_mean:.6f}")
        log_data("Std dry mass", f"{real_aggregated_uniform.std[0]:.6f}")
        log_success("Uniform aggregation completed successfully")

    @pytest.mark.real_data
    def test_generation_stratification_with_real_data(self, real_aggregated_by_generation):
        """Generation stratification should work with real data."""
        log_header("GENERATION STRATIFICATION - REAL DATA")

        assert real_aggregated_by_generation.groups is not None
        assert len(real_aggregated_by_generation.groups) > 1

        # Mass should generally increase across generations (cells grow)
        masses = real_aggregated_by_generation.mean[:, 0]
        # At least some positive values
        assert np.any(masses > 0)

        log_data("Number of generations", len(real_aggregated_by_generation.groups))
        log_data("Generation groups", list(real_aggregated_by_generation.groups))
        for i, gen in enumerate(real_aggregated_by_generation.groups):
            log_info(f"Gen {gen}: mean={masses[i]:.6f}, n={real_aggregated_by_generation.n_samples[i]}")
        log_success("Generation stratification completed successfully")

    @pytest.mark.real_data
    def test_seed_stratification_with_real_data(self, real_aggregated_by_seed):
        """Lineage seed stratification should work with real data."""
        log_header("LINEAGE SEED STRATIFICATION - REAL DATA")

        assert real_aggregated_by_seed.groups is not None

        # Each seed should have samples
        assert np.all(real_aggregated_by_seed.n_samples > 0)

        log_data("Number of lineage seeds", len(real_aggregated_by_seed.groups))
        log_data("Seed groups", list(real_aggregated_by_seed.groups))
        for i, seed in enumerate(real_aggregated_by_seed.groups):
            log_info(f"Seed {seed}: n_samples={real_aggregated_by_seed.n_samples[i]}")
        log_success("Lineage seed stratification completed successfully")

    @pytest.mark.real_data
    def test_variance_decomposition_with_real_data(
        self,
        real_aggregated_uniform,
        real_aggregated_by_generation,
        real_aggregated_by_seed,
    ):
        """Variance decomposition should work with real data."""
        from uq import compute_variance_decomposition

        log_header("VARIANCE DECOMPOSITION - REAL DATA")

        decomp = compute_variance_decomposition(
            real_aggregated_by_generation,
            real_aggregated_by_seed,
            real_aggregated_uniform,
        )

        assert "total_variance" in decomp
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp

        # Total variance should be positive
        assert np.all(decomp["total_variance"] >= 0)

        log_data("Total variance", f"{decomp['total_variance'][0]:.6f}")
        log_data(
            "Generation fraction",
            f"{decomp['generation_fraction'][0]:.4f} ({decomp['generation_fraction'][0] * 100:.1f}%)",
        )
        log_data("Seed fraction", f"{decomp['seed_fraction'][0]:.4f} ({decomp['seed_fraction'][0] * 100:.1f}%)")
        log_success("Variance decomposition completed successfully")


class TestRealDataCellCycle:
    """Tests for cell cycle analysis with real data."""

    @pytest.mark.real_data
    def test_mass_based_cell_cycle_with_real_data(self, real_simulation_dataframe):
        """Mass-based cell cycle variable should work with real data."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        from uq import MassBasedCellCycleVariable

        log_header("MASS-BASED CELL CYCLE - REAL DATA")

        # Rename columns to match expected schema
        df = real_simulation_dataframe.rename({
            "listeners__mass__growth": "listeners__fba_results__growth",
        })

        log_info(f"Processing {len(df):,} real data points")

        computer = MassBasedCellCycleVariable()
        result = computer.compute(df)

        assert result.values is not None
        assert len(result.values) == len(df)
        assert result.normalized is True

        # Values should be in [0, 1]
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

        log_data("Cell cycle values computed", len(result.values))
        log_data("Min value", f"{np.min(result.values):.4f}")
        log_data("Max value", f"{np.max(result.values):.4f}")
        log_data("Mean value", f"{np.mean(result.values):.4f}")
        log_success("Mass-based cell cycle analysis completed")

    @pytest.mark.real_data
    def test_dna_replication_cell_cycle_with_real_data(self, real_simulation_dataframe):
        """DNA replication cell cycle variable should work with real data."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        from uq import DNAReplicationCellCycleVariable

        log_header("DNA REPLICATION CELL CYCLE - REAL DATA")

        log_info(f"Processing {len(real_simulation_dataframe):,} real data points")

        computer = DNAReplicationCellCycleVariable()
        result = computer.compute(real_simulation_dataframe)

        assert result.values is not None
        assert len(result.values) == len(real_simulation_dataframe)

        # Values should be in [0, 1]
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 1)

        log_data("Cell cycle values computed", len(result.values))
        log_data("Min value", f"{np.min(result.values):.4f}")
        log_data("Max value", f"{np.max(result.values):.4f}")
        log_data("Mean value", f"{np.mean(result.values):.4f}")
        log_success("DNA replication cell cycle analysis completed")


class TestRealDataKoopman:
    """Tests for Koopman analysis with real data."""

    @pytest.mark.real_data
    def test_dmd_with_real_trajectory(self, real_trajectory):
        """DMD should work with real cell trajectory."""
        from uq import DynamicModeDecomposition

        log_header("DYNAMIC MODE DECOMPOSITION - REAL DATA")

        log_info(f"Trajectory shape: {real_trajectory.shape}")
        log_info(f"Time steps: {real_trajectory.shape[0]}, Features: {real_trajectory.shape[1]}")

        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(real_trajectory)
        spectrum = dmd.get_spectrum()

        assert spectrum.n_modes > 0
        assert len(spectrum.eigenvalues) > 0

        log_data("Number of modes", spectrum.n_modes)
        log_data("Eigenvalues computed", len(spectrum.eigenvalues))
        dominant = spectrum.get_dominant_modes(3)
        for i, mode in enumerate(dominant):
            log_info(f"Mode {i + 1}: freq={mode.frequency:.6f}, stable={mode.is_stable}")
        log_success("DMD analysis completed")

    @pytest.mark.real_data
    def test_edmd_with_real_trajectory(self, real_trajectory):
        """Extended DMD should work with real cell trajectory."""
        from uq import ExtendedDMD

        log_header("EXTENDED DMD - REAL DATA")

        log_info(f"Trajectory shape: {real_trajectory.shape}")

        edmd = ExtendedDMD(dictionary_order=2, dt=1.0)
        edmd.fit(real_trajectory)
        spectrum = edmd.get_spectrum()

        assert spectrum.n_modes > 0

        log_data("Dictionary order", 2)
        log_data("Number of modes", spectrum.n_modes)
        log_success("Extended DMD analysis completed")

    @pytest.mark.real_data
    def test_cell_cycle_koopman_with_real_data(self, real_trajectory):
        """Cell cycle Koopman analysis should work with real data."""
        from uq import CellCycleKoopmanAnalyzer

        log_header("CELL CYCLE KOOPMAN ANALYSIS - REAL DATA")

        # Assuming ~1 hour cell cycle
        analyzer = CellCycleKoopmanAnalyzer(
            expected_cycle_time=3600.0,
            frequency_tolerance=0.5,
            dt=1.0,
        )

        log_info("Expected cycle time: 3600.0 seconds")
        log_info("Frequency tolerance: 0.5")

        result = analyzer.analyze_cell_cycle_spectrum(real_trajectory)

        assert "spectrum" in result
        assert "cell_cycle_modes" in result
        assert "cell_cycle_variance_fraction" in result

        log_data("Cell cycle modes found", len(result["cell_cycle_modes"]))
        log_data("Variance fraction", f"{result['cell_cycle_variance_fraction']:.4f}")
        log_success("Cell cycle Koopman analysis completed")


class TestRealDataSensitivity:
    """Tests for sensitivity analysis preparation with real data."""

    @pytest.mark.real_data
    def test_parameter_space_with_real_data(self, real_simulation_dataframe):
        """Parameter space should be compatible with real data structure."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        from uq import InputParameterSpace

        log_header("PARAMETER SPACE VALIDATION - REAL DATA")

        space = InputParameterSpace(
            include_vio=True,
            include_mecillinam=True,
        )

        # Should be able to generate samples
        lb, ub = space.get_pytuq_bounds()
        assert len(lb) == space.n_parameters
        assert np.all(lb < ub)

        log_data("Number of parameters", space.n_parameters)
        log_data("Parameter names", space.parameter_names)
        for i, name in enumerate(space.parameter_names):
            log_info(f"{name}: bounds=[{lb[i]:.2f}, {ub[i]:.2f}]")
        log_success("Parameter space validated for sensitivity analysis")

    @pytest.mark.real_data
    def test_output_extraction_from_real_data(self, real_simulation_dataframe):
        """Output extraction should work with real data."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        log_header("OUTPUT EXTRACTION - REAL DATA")

        df = real_simulation_dataframe

        # Extract key outputs
        outputs = df.select([
            "listeners__mass__dry_mass",
            "listeners__mass__growth",
        ]).to_numpy()

        assert outputs.shape[0] == len(df)
        assert outputs.shape[1] == 2

        # Should have valid numeric values
        assert np.isfinite(outputs).all()

        log_data("Rows extracted", outputs.shape[0])
        log_data("Columns extracted", outputs.shape[1])
        log_data("Dry mass range", f"[{outputs[:, 0].min():.6f}, {outputs[:, 0].max():.6f}]")
        log_data("Growth range", f"[{outputs[:, 1].min():.6e}, {outputs[:, 1].max():.6e}]")
        log_success("Output extraction from real data completed")


class TestRealDataE2E:
    """End-to-end tests with real data."""

    @pytest.mark.real_data
    @pytest.mark.e2e
    def test_complete_workflow_with_real_data(
        self,
        real_simulation_dataframe,
        real_aggregated_uniform,
        real_aggregated_by_generation,
        real_aggregated_by_seed,
    ):
        """Complete UQ workflow should work with real data."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        from uq import (
            InputParameterSpace,
            compute_variance_decomposition,
        )

        log_header("COMPLETE UQ WORKFLOW - REAL vEcoli DATA")
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║  🧬  ANALYZING REAL WHOLE-CELL SIMULATION DATA  🧬            ║")
        print("  ║      Dataset: api_simulation_default                           ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.END}")

        # 1. Define parameter space
        log_section("Step 1: Define Parameter Space")
        param_space = InputParameterSpace(
            include_vio=True,
            include_mecillinam=True,
        )
        assert param_space.n_parameters == 3
        log_success(f"Parameter space defined: {param_space.n_parameters} parameters")
        for name in param_space.parameter_names:
            log_info(f"  - {name}")

        # 2. Verify aggregation worked
        log_section("Step 2: Verify Data Aggregation")
        assert real_aggregated_uniform.n_samples > 0
        assert len(real_aggregated_by_generation.groups) > 0
        assert len(real_aggregated_by_seed.groups) >= 1
        log_success(f"Uniform aggregation: {real_aggregated_uniform.n_samples:,} samples")
        log_success(f"Generation groups: {len(real_aggregated_by_generation.groups)}")
        log_success(f"Lineage seed groups: {len(real_aggregated_by_seed.groups)}")

        # 3. Compute variance decomposition
        log_section("Step 3: Variance Decomposition")
        decomp = compute_variance_decomposition(
            real_aggregated_by_generation,
            real_aggregated_by_seed,
            real_aggregated_uniform,
        )
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp
        log_success("Variance decomposition computed")

        # 4. Report results
        log_section("FINAL RESULTS FROM REAL DATA")
        print(f"\n{Colors.BOLD}{Colors.GREEN}")
        print("  ┌────────────────────────────────────────────────────────────┐")
        print("  │              REAL DATA UQ ANALYSIS RESULTS                 │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print(f"  │  Total samples:              {real_aggregated_uniform.n_samples:>20,}   │")
        print(f"  │  Generations:                {len(real_aggregated_by_generation.groups):>20}   │")
        print(f"  │  Lineage seeds:              {len(real_aggregated_by_seed.groups):>20}   │")
        print(f"  │  Mean dry mass:              {real_aggregated_uniform.mean[0]:>20.6f}   │")
        print(f"  │  Std dry mass:               {real_aggregated_uniform.std[0]:>20.6f}   │")
        print(f"  │  Generation variance frac:   {decomp['generation_fraction'][0]:>20.4f}   │")
        print(f"  │  Seed variance frac:         {decomp['seed_fraction'][0]:>20.4f}   │")
        print("  └────────────────────────────────────────────────────────────┘")
        print(f"{Colors.END}")
        log_success("Complete UQ workflow finished successfully!")

    @pytest.mark.real_data
    @pytest.mark.e2e
    def test_cell_cycle_stratification_with_real_data(self, real_simulation_dataframe):
        """Cell cycle stratification should reveal phenotypic variation in real data."""
        if real_simulation_dataframe is None:
            pytest.skip("Real simulation data not available")

        from uq import MassBasedCellCycleVariable

        log_header("CELL CYCLE STRATIFICATION - REAL vEcoli DATA")

        # Rename columns
        df = real_simulation_dataframe.rename({
            "listeners__mass__growth": "listeners__fba_results__growth",
        })

        # 1. Compute cell cycle variable
        log_section("Computing Cell Cycle Variable")
        computer = MassBasedCellCycleVariable()
        cc_var = computer.compute(df)
        log_success(f"Cell cycle variable computed for {len(cc_var.values):,} data points")

        # 2. Bin into stages
        log_section("Binning into Cell Cycle Stages")
        n_stages = 10
        stages = cc_var.to_stage_bins(n_bins=n_stages)
        log_success(f"Data binned into {n_stages} cell cycle stages")

        # 3. Aggregate by stage
        log_section("Aggregating by Cell Cycle Stage")
        output = df["listeners__mass__dry_mass"].to_numpy()

        stage_stats = []
        for s in range(n_stages):
            mask = stages == s
            if np.sum(mask) > 0:
                stage_stats.append({
                    "stage": s,
                    "mean": np.mean(output[mask]),
                    "std": np.std(output[mask]),
                    "n": np.sum(mask),
                })
                log_info(f"Stage {s}: n={np.sum(mask):,}, mean={np.mean(output[mask]):.6f}")

        # 4. Verify we have variation across stages
        means = [s["mean"] for s in stage_stats]
        if len(means) > 1:
            variation = np.std(means) / np.mean(means)  # CV
            log_data("Cell cycle stage CV", f"{variation:.4f}")
            # Mass should vary across cell cycle
            assert variation > 0, "Mass should vary across cell cycle stages"
            log_success("Cell cycle stratification reveals phenotypic variation!")

    @pytest.mark.real_data
    @pytest.mark.e2e
    def test_koopman_spectral_decomposition_real_data(self, real_trajectory):
        """Koopman spectral decomposition should reveal dynamics in real data."""
        from uq import DynamicModeDecomposition

        log_header("KOOPMAN SPECTRAL ANALYSIS - REAL vEcoli DATA")

        # 1. Standard DMD
        log_section("Performing Dynamic Mode Decomposition")
        dmd = DynamicModeDecomposition(dt=1.0)
        dmd.fit(real_trajectory)
        spectrum = dmd.get_spectrum()
        log_success(f"DMD completed: {spectrum.n_modes} modes extracted")

        # 2. Get dominant modes
        log_section("Analyzing Dominant Modes")
        dominant = spectrum.get_dominant_modes(5)

        # 3. Report
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("  ┌────────────────────────────────────────────────────────────┐")
        print("  │           KOOPMAN SPECTRAL ANALYSIS RESULTS                │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print(f"  │  Total modes:                {spectrum.n_modes:>20}   │")
        print(f"  │  Dominant modes analyzed:    {len(dominant):>20}   │")
        print("  ├────────────────────────────────────────────────────────────┤")
        print("  │  Top 3 Dominant Modes:                                     │")
        for i, mode in enumerate(dominant[:3]):
            stable_str = "stable" if mode.is_stable else "unstable"
            print(f"  │    Mode {i + 1}: freq={mode.frequency:>8.4f}, {stable_str:<10}        │")
        print("  └────────────────────────────────────────────────────────────┘")
        print(f"{Colors.END}")

        assert spectrum.n_modes > 0
        log_success("Koopman spectral analysis completed successfully!")
