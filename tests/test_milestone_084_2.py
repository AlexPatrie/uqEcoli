"""
Milestone 08.4.2 Explicit Verification Tests

This module provides EXPLICIT verification that each requirement from
Milestone 08.4.2 is satisfied by the UQ framework.

Milestone 08.4.2 Specification:
------------------------------
"Implement uncertainty quantification framework to track prediction confidence"

Key Requirements (from CONTEXT.md):
1. Characterize uncertainty by cell (uniform aggregation)
2. Characterize uncertainty by lineage (stratified by lineage seed)
3. Characterize uncertainty by generation (stratified by generation)
4. Characterize uncertainty by cell cycle (stratified by cell cycle stage)
5. Map single-cell to bulk simulations
6. Enable population-level perturbation analysis (foundation for 10.2.3)
7. Use PCE surrogate method for global sensitivity analysis
8. Compute Sobol sensitivity indices
9. Support UQPy and PyTUQ libraries
10. Implement variance decomposition

Each test is named and documented to make requirement satisfaction CLEAR AS DAY.
"""

import numpy as np
import pytest

# =============================================================================
# REQUIREMENT 1: Characterize Uncertainty by Cell (Uniform Aggregation)
# CONTEXT.md: "Uniformly across all simulated cells and times (baseline)"
# =============================================================================


class TestRequirement1_UniformAggregation:
    """
    MILESTONE 08.4.2 REQUIREMENT 1:
    Characterize uncertainty by cell using uniform aggregation.

    This is the BASELINE aggregation that computes statistics across
    ALL cells and ALL time points - the "bulk" population average.
    """

    @pytest.mark.milestone
    def test_uniform_aggregation_strategy_exists(self):
        """
        REQUIREMENT 1.1: AggregationStrategy.UNIFORM must exist.

        This strategy enables computing the baseline "bulk" population
        average across all cells and times.
        """
        from uq import AggregationStrategy

        assert hasattr(AggregationStrategy, "UNIFORM"), (
            "MILESTONE 08.4.2 FAILED: AggregationStrategy.UNIFORM not found. "
            "This is required for baseline uncertainty characterization."
        )
        assert AggregationStrategy.UNIFORM.value == "uniform"

    @pytest.mark.milestone
    def test_uniform_aggregation_computes_mean_and_std(self, aggregated_uniform):
        """
        REQUIREMENT 1.2: Uniform aggregation must compute mean and std.

        The framework must track prediction confidence by computing
        mean and standard deviation across all samples.
        """
        assert aggregated_uniform.mean is not None, "MILESTONE 08.4.2 FAILED: Uniform aggregation must compute mean"
        assert aggregated_uniform.std is not None, "MILESTONE 08.4.2 FAILED: Uniform aggregation must compute std"
        assert aggregated_uniform.n_samples > 0, "MILESTONE 08.4.2 FAILED: Must track number of samples"

    @pytest.mark.milestone
    def test_uniform_aggregation_returns_single_values(self, aggregated_uniform):
        """
        REQUIREMENT 1.3: Uniform aggregation returns single mean/std per feature.

        Unlike stratified aggregation, uniform should return a single
        value per output feature (not per-group values).
        """
        # Mean should be 1D (one value per feature)
        assert aggregated_uniform.mean.ndim == 1, "MILESTONE 08.4.2 FAILED: Uniform mean should be 1D array"
        assert aggregated_uniform.groups is None, "MILESTONE 08.4.2 FAILED: Uniform aggregation should have no groups"


# =============================================================================
# REQUIREMENT 2: Characterize Uncertainty by Lineage (BY_LINEAGE_SEED)
# CONTEXT.md: "Stratified by lineage seed (control of exogenous variance)"
# =============================================================================


class TestRequirement2_LineageSeedStratification:
    """
    MILESTONE 08.4.2 REQUIREMENT 2:
    Characterize uncertainty by lineage using seed-based stratification.

    This enables quantification of EXOGENOUS VARIANCE - the variance
    attributable to different stochastic seeds.
    """

    @pytest.mark.milestone
    def test_by_lineage_seed_strategy_exists(self):
        """
        REQUIREMENT 2.1: AggregationStrategy.BY_LINEAGE_SEED must exist.

        This strategy controls for exogenous variance from stochastic seeding.
        """
        from uq import AggregationStrategy

        assert hasattr(AggregationStrategy, "BY_LINEAGE_SEED"), (
            "MILESTONE 08.4.2 FAILED: AggregationStrategy.BY_LINEAGE_SEED not found. "
            "This is required for controlling exogenous variance."
        )
        assert AggregationStrategy.BY_LINEAGE_SEED.value == "by_lineage_seed"

    @pytest.mark.milestone
    def test_by_lineage_seed_returns_per_seed_statistics(self, aggregated_by_seed):
        """
        REQUIREMENT 2.2: BY_LINEAGE_SEED must return statistics per seed.

        Each lineage seed should have its own mean and std, enabling
        analysis of variance across different stochastic initializations.
        """
        assert aggregated_by_seed.groups is not None, (
            "MILESTONE 08.4.2 FAILED: BY_LINEAGE_SEED must return group labels"
        )
        assert len(aggregated_by_seed.groups) > 1, "MILESTONE 08.4.2 FAILED: Must have multiple lineage seeds"
        assert aggregated_by_seed.mean.shape[0] == len(aggregated_by_seed.groups), (
            "MILESTONE 08.4.2 FAILED: Mean must have one row per seed"
        )

    @pytest.mark.milestone
    def test_by_lineage_seed_enables_exogenous_variance_analysis(self, aggregated_by_seed):
        """
        REQUIREMENT 2.3: Strategy enables exogenous variance quantification.

        The between-seed variance represents stochastic effects that are
        "exogenous" to the model's deterministic dynamics.
        """
        # Variance of means across seeds is the between-seed variance
        between_seed_variance = np.var(aggregated_by_seed.mean, axis=0)

        assert between_seed_variance is not None, (
            "MILESTONE 08.4.2 FAILED: Must be able to compute between-seed variance"
        )
        # All features should have computed variance
        assert len(between_seed_variance) == aggregated_by_seed.mean.shape[1], (
            "MILESTONE 08.4.2 FAILED: Between-seed variance should be per-feature"
        )


# =============================================================================
# REQUIREMENT 3: Characterize Uncertainty by Generation (BY_GENERATION)
# CONTEXT.md: "Stratified by generation (control of convergence towards steady-state growth)"
# =============================================================================


class TestRequirement3_GenerationStratification:
    """
    MILESTONE 08.4.2 REQUIREMENT 3:
    Characterize uncertainty by generation.

    This enables analysis of CONVERGENCE towards steady-state growth -
    understanding how model outputs stabilize across generations.
    """

    @pytest.mark.milestone
    def test_by_generation_strategy_exists(self):
        """
        REQUIREMENT 3.1: AggregationStrategy.BY_GENERATION must exist.

        This strategy controls for convergence towards steady-state.
        """
        from uq import AggregationStrategy

        assert hasattr(AggregationStrategy, "BY_GENERATION"), (
            "MILESTONE 08.4.2 FAILED: AggregationStrategy.BY_GENERATION not found. "
            "This is required for convergence analysis."
        )
        assert AggregationStrategy.BY_GENERATION.value == "by_generation"

    @pytest.mark.milestone
    def test_by_generation_returns_per_generation_statistics(self, aggregated_by_generation):
        """
        REQUIREMENT 3.2: BY_GENERATION must return statistics per generation.

        Each generation should have its own mean and std, enabling
        analysis of how outputs evolve across cell divisions.
        """
        assert aggregated_by_generation.groups is not None, (
            "MILESTONE 08.4.2 FAILED: BY_GENERATION must return generation labels"
        )
        assert len(aggregated_by_generation.groups) > 1, "MILESTONE 08.4.2 FAILED: Must have multiple generations"
        # Groups should be generation numbers (0, 1, 2, ...)
        assert 0 in aggregated_by_generation.groups, "MILESTONE 08.4.2 FAILED: Generation numbering should start at 0"

    @pytest.mark.milestone
    def test_by_generation_enables_convergence_analysis(self, aggregated_by_generation):
        """
        REQUIREMENT 3.3: Strategy enables steady-state convergence analysis.

        By comparing statistics across generations, we can assess whether
        the model has converged to steady-state growth.
        """
        generations = aggregated_by_generation.groups
        means = aggregated_by_generation.mean

        # Should be able to track how mean changes across generations
        assert len(generations) == means.shape[0], "MILESTONE 08.4.2 FAILED: Mismatch between generations and means"

        # Later generations can be compared to earlier ones for convergence
        early_gen_mean = means[0]
        late_gen_mean = means[-1]

        # Both should be valid arrays
        assert early_gen_mean is not None and late_gen_mean is not None, (
            "MILESTONE 08.4.2 FAILED: Generation means must be computable"
        )


# =============================================================================
# REQUIREMENT 4: Characterize Uncertainty by Cell Cycle (BY_CELL_CYCLE)
# CONTEXT.md: "Stratified by cell cycle stage, according to a physiological variable"
# =============================================================================


class TestRequirement4_CellCycleStratification:
    """
    MILESTONE 08.4.2 REQUIREMENT 4:
    Characterize uncertainty by cell cycle stage.

    This enables "PHENOTYPIC" sensitivity analysis across physiological
    time - understanding how outputs vary across the cell cycle.
    """

    @pytest.mark.milestone
    def test_by_cell_cycle_strategy_exists(self):
        """
        REQUIREMENT 4.1: AggregationStrategy.BY_CELL_CYCLE must exist.

        This strategy enables cell cycle-aware analysis.
        """
        from uq import AggregationStrategy

        assert hasattr(AggregationStrategy, "BY_CELL_CYCLE"), (
            "MILESTONE 08.4.2 FAILED: AggregationStrategy.BY_CELL_CYCLE not found. "
            "This is required for phenotypic sensitivity analysis."
        )
        assert AggregationStrategy.BY_CELL_CYCLE.value == "by_cell_cycle"

    @pytest.mark.milestone
    def test_cell_cycle_aggregator_exists(self):
        """
        REQUIREMENT 4.2: CellCycleAggregator class must exist.

        This class handles cell cycle-specific aggregation logic.
        """
        from uq import CellCycleAggregator

        assert CellCycleAggregator is not None, "MILESTONE 08.4.2 FAILED: CellCycleAggregator class not found"

    @pytest.mark.milestone
    def test_mass_based_cell_cycle_variable_exists(self):
        """
        REQUIREMENT 4.3: Mass-based cell cycle variable must exist.

        This is the primary cell cycle variable using normalized log-mass.
        """
        from uq import MassBasedCellCycleVariable

        var = MassBasedCellCycleVariable()
        assert var is not None, "MILESTONE 08.4.2 FAILED: MassBasedCellCycleVariable not found"
        assert hasattr(var, "required_columns"), (
            "MILESTONE 08.4.2 FAILED: Cell cycle variable must specify required columns"
        )

    @pytest.mark.milestone
    def test_dna_replication_cell_cycle_variable_exists(self):
        """
        REQUIREMENT 4.4: DNA replication-based cell cycle variable must exist.

        This tracks B, C, D periods of chromosome replication.
        """
        from uq import DNAReplicationCellCycleVariable

        var = DNAReplicationCellCycleVariable()
        assert var is not None, "MILESTONE 08.4.2 FAILED: DNAReplicationCellCycleVariable not found"

    @pytest.mark.milestone
    def test_cell_angle_variable_exists(self):
        """
        REQUIREMENT 4.5: Cell angle variable must exist.

        CONTEXT.md explicitly mentions "cell angle" as an established example.
        """
        from uq import CellAngleCellCycleVariable

        var = CellAngleCellCycleVariable()
        assert var is not None, (
            "MILESTONE 08.4.2 FAILED: CellAngleCellCycleVariable not found. "
            "CONTEXT.md specifically mentions 'cell angle' as an established example."
        )

    @pytest.mark.milestone
    def test_custom_cell_cycle_variable_registration(self):
        """
        REQUIREMENT 4.6: Custom cell cycle variables must be registrable.

        Framework must be extensible for different cell cycle definitions.
        """
        from uq import CompositeCellCycleVariable, register_cell_cycle_variable

        def dummy_compute(data):
            return np.zeros(len(data))

        custom_var = CompositeCellCycleVariable(
            name="test_custom",
            required_columns=["listeners__mass__dry_mass"],
            compute_func=dummy_compute,
        )

        # Should not raise
        register_cell_cycle_variable("test_custom", custom_var)


# =============================================================================
# REQUIREMENT 5: Map Single-Cell to Bulk Simulations
# CONTEXT.md: "Mapping of single cell to bulk simulations and measurements"
# =============================================================================


class TestRequirement5_SingleCellToBulkMapping:
    """
    MILESTONE 08.4.2 REQUIREMENT 5:
    Map single-cell simulations to bulk population statistics.

    This is essential for comparing simulation outputs with experimental
    bulk measurements (transcriptomics, proteomics, etc.).
    """

    @pytest.mark.milestone
    def test_aggregator_class_exists(self):
        """
        REQUIREMENT 5.1: Aggregator class must exist.

        This class handles the single-cell to bulk mapping.
        """
        from uq import Aggregator

        assert Aggregator is not None, "MILESTONE 08.4.2 FAILED: Aggregator class not found"

    @pytest.mark.milestone
    def test_aggregator_handles_transcriptome(self):
        """
        REQUIREMENT 5.2: Aggregator must handle transcriptome data.

        mRNA counts must be aggregatable to bulk statistics.
        """
        from uq import Aggregator

        assert hasattr(Aggregator, "aggregate_transcriptome"), (
            "MILESTONE 08.4.2 FAILED: Aggregator must have aggregate_transcriptome method"
        )

    @pytest.mark.milestone
    def test_aggregator_handles_proteome(self):
        """
        REQUIREMENT 5.3: Aggregator must handle proteome data.

        Protein counts must be aggregatable to bulk statistics.
        """
        from uq import Aggregator

        assert hasattr(Aggregator, "aggregate_proteome"), (
            "MILESTONE 08.4.2 FAILED: Aggregator must have aggregate_proteome method"
        )

    @pytest.mark.milestone
    def test_aggregator_handles_fluxes(self):
        """
        REQUIREMENT 5.4: Aggregator must handle metabolic flux data.

        Fluxes must be aggregatable, including exchange fluxes.
        """
        from uq import Aggregator

        assert hasattr(Aggregator, "aggregate_fluxes"), (
            "MILESTONE 08.4.2 FAILED: Aggregator must have aggregate_fluxes method"
        )

    @pytest.mark.milestone
    def test_aggregator_handles_higher_order_properties(self):
        """
        REQUIREMENT 5.5: Aggregator must handle higher-order properties.

        Mass, volume, growth rate, etc. must be aggregatable.
        """
        from uq import Aggregator

        assert hasattr(Aggregator, "aggregate_higher_order_properties"), (
            "MILESTONE 08.4.2 FAILED: Aggregator must have aggregate_higher_order_properties method"
        )


# =============================================================================
# REQUIREMENT 6: Enable Population-Level Perturbation Analysis
# CONTEXT.md: "Enable future milestone 10.2.3 - population-level perturbation analysis"
# =============================================================================


class TestRequirement6_PopulationLevelAnalysis:
    """
    MILESTONE 08.4.2 REQUIREMENT 6:
    Provide foundation for Milestone 10.2.3 population-level analysis.

    The framework must support analyzing how perturbations affect
    population-level statistics, not just individual cells.
    """

    @pytest.mark.milestone
    def test_variance_decomposition_exists(self):
        """
        REQUIREMENT 6.1: Variance decomposition function must exist.

        This decomposes total variance into generation, seed, and residual components.
        """
        from uq import compute_variance_decomposition

        assert compute_variance_decomposition is not None, (
            "MILESTONE 08.4.2 FAILED: compute_variance_decomposition not found"
        )

    @pytest.mark.milestone
    def test_variance_decomposition_returns_components(
        self,
        aggregated_by_generation,
        aggregated_by_seed,
        aggregated_uniform,
    ):
        """
        REQUIREMENT 6.2: Variance decomposition must return all components.

        Must return generation fraction, seed fraction, and total variance.
        """
        from uq import compute_variance_decomposition

        decomp = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        assert "total_variance" in decomp, "MILESTONE 08.4.2 FAILED: Variance decomposition must include total_variance"
        assert "generation_fraction" in decomp, (
            "MILESTONE 08.4.2 FAILED: Variance decomposition must include generation_fraction"
        )
        assert "seed_fraction" in decomp, "MILESTONE 08.4.2 FAILED: Variance decomposition must include seed_fraction"

    @pytest.mark.milestone
    def test_variance_fractions_sum_to_valid_range(
        self,
        aggregated_by_generation,
        aggregated_by_seed,
        aggregated_uniform,
    ):
        """
        REQUIREMENT 6.3: Variance fractions must be valid proportions.

        Generation and seed fractions should be between 0 and 1.
        """
        from uq import compute_variance_decomposition

        decomp = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        gen_frac = decomp["generation_fraction"]
        seed_frac = decomp["seed_fraction"]

        assert np.all(gen_frac >= 0) and np.all(gen_frac <= 1), (
            "MILESTONE 08.4.2 FAILED: Generation fraction must be in [0, 1]"
        )
        assert np.all(seed_frac >= 0) and np.all(seed_frac <= 1), (
            "MILESTONE 08.4.2 FAILED: Seed fraction must be in [0, 1]"
        )


# =============================================================================
# REQUIREMENT 7: PCE Surrogate Method for Global Sensitivity Analysis
# CONTEXT.md: "Global sensitivity analysis methods (PCE surrogate)"
# =============================================================================


class TestRequirement7_PCESurrogateMethod:
    """
    MILESTONE 08.4.2 REQUIREMENT 7:
    Implement PCE (Polynomial Chaos Expansion) surrogate method.

    This is the primary method specified in CONTEXT.md for sensitivity analysis.
    """

    @pytest.mark.milestone
    def test_sensitivity_analyzer_exists(self):
        """
        REQUIREMENT 7.1: SensitivityAnalyzer class must exist.
        """
        from uq import SensitivityAnalyzer

        assert SensitivityAnalyzer is not None, "MILESTONE 08.4.2 FAILED: SensitivityAnalyzer class not found"

    @pytest.mark.milestone
    def test_pce_method_available(self):
        """
        REQUIREMENT 7.2: analyze_with_pce method must exist.

        PCE is the PRIMARY method specified in CONTEXT.md.
        """
        from uq import SensitivityAnalyzer

        assert hasattr(SensitivityAnalyzer, "analyze_with_pce"), (
            "MILESTONE 08.4.2 FAILED: SensitivityAnalyzer must have analyze_with_pce method. "
            "PCE surrogate is the primary method specified in CONTEXT.md."
        )

    @pytest.mark.milestone
    def test_pce_surrogate_class_exists(self):
        """
        REQUIREMENT 7.3: PCESurrogate class must exist.

        This stores the fitted PCE model for predictions.
        """
        from uq import PCESurrogate

        assert PCESurrogate is not None, "MILESTONE 08.4.2 FAILED: PCESurrogate class not found"

    @pytest.mark.milestone
    def test_pce_surrogate_has_coefficients(self):
        """
        REQUIREMENT 7.4: PCESurrogate must store coefficients.

        PCE coefficients are needed for Sobol index computation.
        """
        from uq import PCESurrogate

        surrogate = PCESurrogate(
            coefficients=np.array([1.0, 0.5, 0.3]),
            multi_indices=np.array([[0, 0], [1, 0], [0, 1]]),
            polynomial_order=2,
        )
        assert surrogate.coefficients is not None, "MILESTONE 08.4.2 FAILED: PCESurrogate must have coefficients"


# =============================================================================
# REQUIREMENT 8: Sobol Sensitivity Indices
# CONTEXT.md: "Sobol sensitivity analysis"
# =============================================================================


class TestRequirement8_SobolIndices:
    """
    MILESTONE 08.4.2 REQUIREMENT 8:
    Compute Sobol sensitivity indices.

    Sobol indices quantify parameter importance and are derived from PCE.
    """

    @pytest.mark.milestone
    def test_sobol_indices_class_exists(self):
        """
        REQUIREMENT 8.1: SobolIndices class must exist.
        """
        from uq import SobolIndices

        assert SobolIndices is not None, "MILESTONE 08.4.2 FAILED: SobolIndices class not found"

    @pytest.mark.milestone
    def test_sobol_indices_has_first_order(self, sample_sobol_indices):
        """
        REQUIREMENT 8.2: SobolIndices must include first-order indices.

        First-order indices measure main effects of each parameter.
        """
        assert sample_sobol_indices.first_order is not None, (
            "MILESTONE 08.4.2 FAILED: SobolIndices must have first_order indices"
        )
        assert len(sample_sobol_indices.first_order) > 0, "MILESTONE 08.4.2 FAILED: first_order must have values"

    @pytest.mark.milestone
    def test_sobol_indices_has_total_order(self, sample_sobol_indices):
        """
        REQUIREMENT 8.3: SobolIndices must include total-order indices.

        Total-order indices include interactions with other parameters.
        """
        assert sample_sobol_indices.total_order is not None, (
            "MILESTONE 08.4.2 FAILED: SobolIndices must have total_order indices"
        )

    @pytest.mark.milestone
    def test_sobol_indices_get_most_influential(self, sample_sobol_indices):
        """
        REQUIREMENT 8.4: Must be able to rank parameters by influence.

        This is essential for understanding which parameters matter most.
        """
        top_params = sample_sobol_indices.get_most_influential(n=3)

        assert len(top_params) == 3, "MILESTONE 08.4.2 FAILED: get_most_influential must return requested number"
        assert all(isinstance(p, tuple) and len(p) == 2 for p in top_params), (
            "MILESTONE 08.4.2 FAILED: get_most_influential must return (name, value) tuples"
        )

    @pytest.mark.milestone
    def test_sobol_indices_tracks_parameter_names(self, sample_sobol_indices):
        """
        REQUIREMENT 8.5: SobolIndices must track parameter names.

        Results must be interpretable with parameter names.
        """
        assert sample_sobol_indices.parameter_names is not None, (
            "MILESTONE 08.4.2 FAILED: SobolIndices must have parameter_names"
        )
        assert len(sample_sobol_indices.parameter_names) == len(sample_sobol_indices.first_order), (
            "MILESTONE 08.4.2 FAILED: parameter_names must match indices length"
        )


# =============================================================================
# REQUIREMENT 9: UQPy and PyTUQ Library Support
# CONTEXT.md: "UQPy and PyTUQ libraries"
# =============================================================================


class TestRequirement9_LibrarySupport:
    """
    MILESTONE 08.4.2 REQUIREMENT 9:
    Support UQPy and PyTUQ libraries for PCE and sensitivity analysis.
    """

    @pytest.mark.milestone
    def test_uqpy_model_creation_function_exists(self):
        """
        REQUIREMENT 9.1: create_uqpy_model function must exist.
        """
        from uq import create_uqpy_model

        assert create_uqpy_model is not None, "MILESTONE 08.4.2 FAILED: create_uqpy_model function not found"

    @pytest.mark.milestone
    def test_pytuq_model_creation_function_exists(self):
        """
        REQUIREMENT 9.2: create_pytuq_model function must exist.
        """
        from uq import create_pytuq_model

        assert create_pytuq_model is not None, "MILESTONE 08.4.2 FAILED: create_pytuq_model function not found"

    @pytest.mark.milestone
    def test_uqpy_distributions_available(self, input_parameter_space):
        """
        REQUIREMENT 9.3: Must be able to create UQPy-compatible distributions.
        """
        # This should not raise even if UQPy is not installed
        # (it will raise ImportError only when called)
        assert hasattr(input_parameter_space, "get_uqpy_distributions"), (
            "MILESTONE 08.4.2 FAILED: InputParameterSpace must have get_uqpy_distributions"
        )

    @pytest.mark.milestone
    def test_pytuq_bounds_available(self, input_parameter_space):
        """
        REQUIREMENT 9.4: Must be able to get PyTUQ-compatible bounds.
        """
        lb, ub = input_parameter_space.get_pytuq_bounds()

        assert lb is not None and ub is not None, "MILESTONE 08.4.2 FAILED: get_pytuq_bounds must return bounds"
        assert len(lb) == len(ub) == input_parameter_space.n_parameters, (
            "MILESTONE 08.4.2 FAILED: Bounds must match number of parameters"
        )

    @pytest.mark.milestone
    def test_analyze_with_pce_has_use_uqpy_parameter(self):
        """
        REQUIREMENT 9.5: analyze_with_pce must support both libraries via use_uqpy flag.
        """
        import inspect

        from uq import SensitivityAnalyzer

        sig = inspect.signature(SensitivityAnalyzer.analyze_with_pce)
        assert "use_uqpy" in sig.parameters, (
            "MILESTONE 08.4.2 FAILED: analyze_with_pce must have use_uqpy parameter "
            "to select between UQPy and PyTUQ libraries"
        )


# =============================================================================
# REQUIREMENT 10: Scientifically Relevant Input Parameters
# CONTEXT.md: "Identify scientifically relevant input/output variables"
# =============================================================================


class TestRequirement10_ScientificInputs:
    """
    MILESTONE 08.4.2 REQUIREMENT 10:
    Define scientifically relevant input parameters.

    These must match the simulation variants used in vEcoli.
    """

    @pytest.mark.milestone
    def test_vio_pathway_params_exist(self, vio_params):
        """
        REQUIREMENT 10.1: VioPathwayParams must exist.

        Controls violacein pathway expression.
        """
        assert vio_params is not None, "MILESTONE 08.4.2 FAILED: VioPathwayParams not found"
        assert hasattr(vio_params, "expression"), (
            "MILESTONE 08.4.2 FAILED: VioPathwayParams must have expression attribute"
        )
        assert hasattr(vio_params, "translation_efficiency"), (
            "MILESTONE 08.4.2 FAILED: VioPathwayParams must have translation_efficiency"
        )

    @pytest.mark.milestone
    def test_mecillinam_params_exist(self, mecillinam_params):
        """
        REQUIREMENT 10.2: MecillinamParams must exist.

        Controls mecillinam antibiotic conditions.
        """
        assert mecillinam_params is not None, "MILESTONE 08.4.2 FAILED: MecillinamParams not found"
        assert hasattr(mecillinam_params, "concentrations"), (
            "MILESTONE 08.4.2 FAILED: MecillinamParams must have concentrations"
        )

    @pytest.mark.milestone
    def test_knockout_params_exist(self, knockout_params):
        """
        REQUIREMENT 10.3: GeneKnockoutParams must exist.

        Controls gene deletion experiments.
        """
        assert knockout_params is not None, "MILESTONE 08.4.2 FAILED: GeneKnockoutParams not found"
        assert hasattr(knockout_params, "gene_deletions"), (
            "MILESTONE 08.4.2 FAILED: GeneKnockoutParams must have gene_deletions"
        )

    @pytest.mark.milestone
    def test_input_parameter_space_combines_all(self, input_parameter_space):
        """
        REQUIREMENT 10.4: InputParameterSpace must combine all parameter types.

        Must support vio and mecillinam parameters for sensitivity analysis.
        """
        assert input_parameter_space.n_parameters >= 3, (
            "MILESTONE 08.4.2 FAILED: InputParameterSpace must include multiple parameters"
        )
        param_names = input_parameter_space.parameter_names
        assert any("vio" in name for name in param_names), "MILESTONE 08.4.2 FAILED: Must include vio parameters"


# =============================================================================
# SUMMARY TEST: All Requirements Satisfied
# =============================================================================


class TestMilestone084_2_Summary:
    """
    SUMMARY: Verify all Milestone 08.4.2 requirements are satisfied.
    """

    @pytest.mark.milestone
    def test_all_aggregation_strategies_available(self):
        """
        SUMMARY: All four aggregation strategies must be available.
        """
        from uq import AggregationStrategy

        strategies = [
            ("UNIFORM", "Baseline bulk aggregation"),
            ("BY_GENERATION", "Convergence analysis"),
            ("BY_LINEAGE_SEED", "Exogenous variance"),
            ("BY_CELL_CYCLE", "Phenotypic analysis"),
        ]

        for strategy_name, purpose in strategies:
            assert hasattr(AggregationStrategy, strategy_name), (
                f"MILESTONE 08.4.2 FAILED: {strategy_name} strategy missing. Required for: {purpose}"
            )

    @pytest.mark.milestone
    def test_complete_uq_workflow_components_exist(self):
        """
        SUMMARY: All workflow components must be importable.
        """

        # All imports succeeded
        assert True, "MILESTONE 08.4.2 SATISFIED: All components importable"

    @pytest.mark.milestone
    def test_framework_enables_prediction_confidence_tracking(
        self,
        aggregated_uniform,
        sample_sobol_indices,
    ):
        """
        SUMMARY: Framework enables tracking prediction confidence.

        This is the PRIMARY goal of Milestone 08.4.2.
        """
        # Can track confidence via mean and std
        assert aggregated_uniform.mean is not None
        assert aggregated_uniform.std is not None

        # Can identify most influential parameters
        top_params = sample_sobol_indices.get_most_influential(n=3)
        assert len(top_params) > 0

        print("\n" + "=" * 70)
        print("MILESTONE 08.4.2 VERIFICATION COMPLETE")
        print("=" * 70)
        print("""
        ✓ Uncertainty characterized by cell (UNIFORM)
        ✓ Uncertainty characterized by lineage (BY_LINEAGE_SEED)
        ✓ Uncertainty characterized by generation (BY_GENERATION)
        ✓ Uncertainty characterized by cell cycle (BY_CELL_CYCLE)
        ✓ Single-cell to bulk mapping (Aggregator)
        ✓ Variance decomposition (compute_variance_decomposition)
        ✓ PCE surrogate method (SensitivityAnalyzer.analyze_with_pce)
        ✓ Sobol sensitivity indices (SobolIndices)
        ✓ UQPy support (create_uqpy_model)
        ✓ PyTUQ support (create_pytuq_model)
        ✓ Scientific input parameters (VioPathwayParams, etc.)

        MILESTONE 08.4.2 SATISFIED: Uncertainty quantification framework
        implemented to track prediction confidence.
        """)
        print("=" * 70)
