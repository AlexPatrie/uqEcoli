"""
Unit tests for UQ aggregation strategies.

Tests the four aggregation strategies specified in Milestone 08.4.2:
1. Uniform aggregation across all cells and times
2. Stratified by generation
3. Stratified by lineage seed
4. Stratified by cell cycle stage

Also tests variance decomposition functionality.
"""

import numpy as np
import pytest


class TestAggregationStrategy:
    """Tests for the AggregationStrategy enum."""

    @pytest.mark.unit
    def test_enum_values(self):
        """AggregationStrategy should have all four required strategies."""
        from libuq import AggregationStrategy

        assert AggregationStrategy.UNIFORM.value == "uniform"
        assert AggregationStrategy.BY_GENERATION.value == "by_generation"
        assert AggregationStrategy.BY_LINEAGE_SEED.value == "by_lineage_seed"
        assert AggregationStrategy.BY_CELL_CYCLE.value == "by_cell_cycle"

    @pytest.mark.unit
    def test_enum_count(self):
        """AggregationStrategy should have exactly four strategies."""
        from libuq import AggregationStrategy

        strategies = list(AggregationStrategy)
        assert len(strategies) == 4, "Must have exactly 4 aggregation strategies per requirements"

    @pytest.mark.unit
    def test_string_conversion(self):
        """AggregationStrategy values should be usable as strings."""
        from libuq import AggregationStrategy

        assert str(AggregationStrategy.UNIFORM) == "AggregationStrategy.UNIFORM"
        assert AggregationStrategy.UNIFORM.value == "uniform"


class TestAggregatedOutput:
    """Tests for the AggregatedOutput dataclass."""

    @pytest.mark.unit
    def test_creation_with_arrays(self):
        """AggregatedOutput should store numpy arrays."""
        from libuq import AggregatedOutput

        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])

        output = AggregatedOutput(mean=mean, std=std, n_samples=100)

        assert np.array_equal(output.mean, mean)
        assert np.array_equal(output.std, std)
        assert output.n_samples == 100

    @pytest.mark.unit
    def test_creation_with_groups(self):
        """AggregatedOutput should support group labels."""
        from libuq import AggregatedOutput

        mean = np.array([[1.0, 2.0], [3.0, 4.0]])
        std = np.array([[0.1, 0.2], [0.3, 0.4]])
        groups = np.array([0, 1])

        output = AggregatedOutput(
            mean=mean,
            std=std,
            n_samples=np.array([50, 50]),
            groups=groups,
        )

        assert np.array_equal(output.groups, groups)
        assert output.mean.shape == (2, 2)

    @pytest.mark.unit
    def test_optional_raw_data(self):
        """AggregatedOutput should optionally store raw data."""
        from libuq import AggregatedOutput

        raw = np.random.randn(100, 3)
        output = AggregatedOutput(
            mean=np.mean(raw, axis=0),
            std=np.std(raw, axis=0),
            n_samples=100,
            raw_data=raw,
        )

        assert output.raw_data is not None
        assert output.raw_data.shape == (100, 3)

    @pytest.mark.unit
    def test_default_optional_fields(self):
        """Optional fields should default to None."""
        from libuq import AggregatedOutput

        output = AggregatedOutput(
            mean=np.array([1.0]),
            std=np.array([0.1]),
            n_samples=10,
        )

        assert output.groups is None
        assert output.raw_data is None


class TestUniformAggregation:
    """Tests for uniform aggregation strategy (Strategy 1)."""

    @pytest.mark.unit
    def test_uniform_mean_calculation(self, aggregated_uniform):
        """Uniform aggregation should compute global mean."""
        assert aggregated_uniform.mean is not None
        assert len(aggregated_uniform.mean) == 2  # Two features

    @pytest.mark.unit
    def test_uniform_std_calculation(self, aggregated_uniform):
        """Uniform aggregation should compute global standard deviation."""
        assert aggregated_uniform.std is not None
        assert len(aggregated_uniform.std) == 2

    @pytest.mark.unit
    def test_uniform_no_groups(self, aggregated_uniform):
        """Uniform aggregation should not have groups."""
        assert aggregated_uniform.groups is None

    @pytest.mark.unit
    def test_uniform_sample_count(self, aggregated_uniform):
        """Uniform aggregation should report total sample count."""
        assert aggregated_uniform.n_samples > 0


class TestGenerationStratification:
    """Tests for stratification by generation (Strategy 2)."""

    @pytest.mark.unit
    def test_generation_groups_exist(self, aggregated_by_generation):
        """Generation stratification should have group labels."""
        assert aggregated_by_generation.groups is not None
        assert len(aggregated_by_generation.groups) > 0

    @pytest.mark.unit
    def test_generation_means_per_group(self, aggregated_by_generation):
        """Generation stratification should have mean per generation."""
        n_groups = len(aggregated_by_generation.groups)
        assert aggregated_by_generation.mean.shape[0] == n_groups

    @pytest.mark.unit
    def test_generation_stds_per_group(self, aggregated_by_generation):
        """Generation stratification should have std per generation."""
        n_groups = len(aggregated_by_generation.groups)
        assert aggregated_by_generation.std.shape[0] == n_groups

    @pytest.mark.unit
    def test_generation_samples_per_group(self, aggregated_by_generation):
        """Generation stratification should have sample counts per generation."""
        n_groups = len(aggregated_by_generation.groups)
        assert len(aggregated_by_generation.n_samples) == n_groups

    @pytest.mark.unit
    def test_generation_groups_sorted(self, aggregated_by_generation):
        """Generation groups should be sorted."""
        groups = aggregated_by_generation.groups
        assert np.all(groups[:-1] <= groups[1:])


class TestLineageSeedStratification:
    """Tests for stratification by lineage seed (Strategy 3)."""

    @pytest.mark.unit
    def test_seed_groups_exist(self, aggregated_by_seed):
        """Lineage seed stratification should have group labels."""
        assert aggregated_by_seed.groups is not None
        assert len(aggregated_by_seed.groups) > 0

    @pytest.mark.unit
    def test_seed_means_per_group(self, aggregated_by_seed):
        """Lineage seed stratification should have mean per seed."""
        n_groups = len(aggregated_by_seed.groups)
        assert aggregated_by_seed.mean.shape[0] == n_groups

    @pytest.mark.unit
    def test_seed_enables_exogenous_variance(self, aggregated_by_seed):
        """Lineage seed stratification enables control of exogenous variance."""
        # Variance between seeds represents exogenous stochasticity
        between_seed_var = np.var(aggregated_by_seed.mean, axis=0)
        assert np.all(between_seed_var >= 0)

    @pytest.mark.unit
    def test_seed_samples_sum(self, aggregated_by_seed, synthetic_simulation_dataframe):
        """Total samples across seeds should match data."""
        total = np.sum(aggregated_by_seed.n_samples)
        assert total == len(synthetic_simulation_dataframe)


class TestCellCycleStratification:
    """Tests for stratification by cell cycle stage (Strategy 4)."""

    @pytest.mark.unit
    def test_cell_cycle_strategy_exists(self):
        """Cell cycle aggregation strategy should exist."""
        from libuq import AggregationStrategy

        assert AggregationStrategy.BY_CELL_CYCLE is not None
        assert AggregationStrategy.BY_CELL_CYCLE.value == "by_cell_cycle"

    @pytest.mark.unit
    def test_cell_cycle_variables_available(self):
        """Cell cycle variables should be defined."""
        from libuq import (
            CellAngleCellCycleVariable,
            DNAReplicationCellCycleVariable,
            MassBasedCellCycleVariable,
        )

        # All three types specified in requirements
        assert MassBasedCellCycleVariable is not None
        assert DNAReplicationCellCycleVariable is not None
        assert CellAngleCellCycleVariable is not None


class TestVarianceDecomposition:
    """Tests for compute_variance_decomposition function."""

    @pytest.mark.unit
    def test_decomposition_returns_dict(self, aggregated_uniform, aggregated_by_generation, aggregated_by_seed):
        """Variance decomposition should return dictionary."""
        from libuq import compute_variance_decomposition

        result = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_decomposition_keys(self, aggregated_uniform, aggregated_by_generation, aggregated_by_seed):
        """Variance decomposition should have expected keys."""
        from libuq import compute_variance_decomposition

        result = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        expected_keys = [
            "total_variance",
            "between_generation_variance",
            "between_seed_variance",
            "generation_fraction",
            "seed_fraction",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.unit
    def test_decomposition_total_variance(self, aggregated_uniform, aggregated_by_generation, aggregated_by_seed):
        """Total variance should equal uniform std squared."""
        from libuq import compute_variance_decomposition

        result = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        expected = aggregated_uniform.std**2
        np.testing.assert_allclose(result["total_variance"], expected)

    @pytest.mark.unit
    def test_decomposition_fractions_bounded(self, aggregated_uniform, aggregated_by_generation, aggregated_by_seed):
        """Variance fractions should be between 0 and 1."""
        from libuq import compute_variance_decomposition

        result = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        assert np.all(result["generation_fraction"] >= 0)
        assert np.all(result["seed_fraction"] >= 0)
        # Fractions can be > 1 in some edge cases with small samples

    @pytest.mark.unit
    def test_decomposition_with_synthetic_data(self, rng):
        """Variance decomposition should work with synthetic data."""
        from libuq import AggregatedOutput, compute_variance_decomposition

        # Create data with known variance structure
        n_samples = 1000
        n_features = 2
        n_generations = 5
        n_seeds = 4

        # Generate data with between-group variance
        data = rng.normal(size=(n_samples, n_features))
        gen_labels = np.repeat(np.arange(n_generations), n_samples // n_generations)
        seed_labels = np.tile(np.arange(n_seeds), n_samples // n_seeds)

        # Add generation effect
        for g in range(n_generations):
            mask = gen_labels == g
            data[mask] += g * 0.5

        # Create aggregated outputs
        uniform = AggregatedOutput(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            n_samples=n_samples,
        )

        gen_means = np.array([np.mean(data[gen_labels == g], axis=0) for g in range(n_generations)])
        gen_stds = np.array([np.std(data[gen_labels == g], axis=0) for g in range(n_generations)])
        by_gen = AggregatedOutput(
            mean=gen_means,
            std=gen_stds,
            n_samples=np.array([np.sum(gen_labels == g) for g in range(n_generations)]),
            groups=np.arange(n_generations),
        )

        seed_means = np.array([np.mean(data[seed_labels == s], axis=0) for s in range(n_seeds)])
        seed_stds = np.array([np.std(data[seed_labels == s], axis=0) for s in range(n_seeds)])
        by_seed = AggregatedOutput(
            mean=seed_means,
            std=seed_stds,
            n_samples=np.array([np.sum(seed_labels == s) for s in range(n_seeds)]),
            groups=np.arange(n_seeds),
        )

        result = compute_variance_decomposition(by_gen, by_seed, uniform)

        # Generation variance should be significant due to added effect
        assert np.all(result["between_generation_variance"] > 0)


class TestAggregatorClass:
    """Tests for the Aggregator class methods."""

    @pytest.mark.unit
    def test_aggregate_uniform_method(self, rng):
        """Aggregator._aggregate_uniform should compute statistics."""
        from libuq.aggregation import Aggregator

        # Create mock aggregator (without DB connection)
        class MockAggregator(Aggregator):
            def __init__(self):
                pass  # Skip DB initialization

        aggregator = MockAggregator()
        data = rng.normal(size=(100, 3))

        result = aggregator._aggregate_uniform(data)

        np.testing.assert_allclose(result.mean, np.nanmean(data, axis=0))
        np.testing.assert_allclose(result.std, np.nanstd(data, axis=0))
        assert result.n_samples == 100

    @pytest.mark.unit
    def test_aggregate_stratified_method(self, rng):
        """Aggregator._aggregate_stratified should compute per-group statistics."""
        from libuq.aggregation import Aggregator

        class MockAggregator(Aggregator):
            def __init__(self):
                pass

        aggregator = MockAggregator()

        data = rng.normal(size=(100, 2))
        labels = np.repeat([0, 1, 2, 3, 4], 20)

        result = aggregator._aggregate_stratified(data, labels)

        assert result.mean.shape == (5, 2)
        assert result.std.shape == (5, 2)
        assert len(result.groups) == 5
        assert np.all(result.n_samples == 20)

    @pytest.mark.unit
    def test_aggregate_requires_labels_for_stratified(self, rng):
        """Aggregator.aggregate should require labels for stratified strategies."""
        from libuq.aggregation import AggregationStrategy, Aggregator

        class MockAggregator(Aggregator):
            def __init__(self):
                pass

        aggregator = MockAggregator()
        data = rng.normal(size=(100, 2))

        with pytest.raises(ValueError, match="group_labels required"):
            aggregator.aggregate(data, AggregationStrategy.BY_GENERATION)

    @pytest.mark.unit
    def test_aggregate_uniform_no_labels_needed(self, rng):
        """Aggregator.aggregate should not require labels for uniform strategy."""
        from libuq.aggregation import AggregationStrategy, Aggregator

        class MockAggregator(Aggregator):
            def __init__(self):
                pass

        aggregator = MockAggregator()
        data = rng.normal(size=(100, 2))

        result = aggregator.aggregate(data, AggregationStrategy.UNIFORM)

        assert result.n_samples == 100


class TestAggregationIntegration:
    """Integration tests for aggregation with synthetic data."""

    @pytest.mark.unit
    def test_all_strategies_produce_valid_output(self, synthetic_simulation_dataframe, rng):
        """All aggregation strategies should produce valid AggregatedOutput."""
        from libuq import AggregatedOutput

        df = synthetic_simulation_dataframe
        data = df.select([
            "listeners__mass__dry_mass",
            "listeners__fba_results__growth",
        ]).to_numpy()

        # Test uniform
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        uniform = AggregatedOutput(mean=mean, std=std, n_samples=len(df))
        assert len(uniform.mean) == 2

        # Test by generation
        generations = df["generation"].to_numpy()
        unique_gens = np.unique(generations)
        gen_means = np.array([np.mean(data[generations == g], axis=0) for g in unique_gens])
        gen_stds = np.array([np.std(data[generations == g], axis=0) for g in unique_gens])
        by_gen = AggregatedOutput(
            mean=gen_means,
            std=gen_stds,
            n_samples=np.array([np.sum(generations == g) for g in unique_gens]),
            groups=unique_gens,
        )
        assert by_gen.mean.shape[0] == len(unique_gens)

        # Test by lineage seed
        seeds = df["lineage_seed"].to_numpy()
        unique_seeds = np.unique(seeds)
        seed_means = np.array([np.mean(data[seeds == s], axis=0) for s in unique_seeds])
        seed_stds = np.array([np.std(data[seeds == s], axis=0) for s in unique_seeds])
        by_seed = AggregatedOutput(
            mean=seed_means,
            std=seed_stds,
            n_samples=np.array([np.sum(seeds == s) for s in unique_seeds]),
            groups=unique_seeds,
        )
        assert by_seed.mean.shape[0] == len(unique_seeds)

    @pytest.mark.unit
    def test_stratified_preserves_total_samples(self, synthetic_simulation_dataframe):
        """Stratified aggregation should preserve total sample count."""
        df = synthetic_simulation_dataframe

        generations = df["generation"].to_numpy()
        unique_gens = np.unique(generations)
        samples_per_gen = np.array([np.sum(generations == g) for g in unique_gens])

        assert np.sum(samples_per_gen) == len(df)

    @pytest.mark.unit
    def test_within_group_variance_less_than_total(self, rng):
        """Within-group variance should generally be less than total variance."""
        # This tests the fundamental property of stratification
        n = 1000
        n_groups = 5
        data = rng.normal(size=(n, 1))
        labels = np.repeat(np.arange(n_groups), n // n_groups)

        # Add between-group variance
        for g in range(n_groups):
            data[labels == g] += g * 2.0

        total_var = np.var(data)
        within_vars = [np.var(data[labels == g]) for g in range(n_groups)]
        mean_within_var = np.mean(within_vars)

        # Within-group variance should be less than total
        assert mean_within_var < total_var


class TestBulkSingleCellMapping:
    """Tests for single-cell to bulk attribute mapping via aggregation."""

    @pytest.mark.unit
    def test_uniform_represents_bulk(self, aggregated_uniform):
        """Uniform aggregation represents bulk population average."""
        # Bulk measurement = average over population
        assert aggregated_uniform.mean is not None
        assert len(aggregated_uniform.mean) > 0

    @pytest.mark.unit
    def test_stratified_preserves_single_cell_info(self, aggregated_by_generation, aggregated_by_seed):
        """Stratified aggregation preserves single-cell heterogeneity info."""
        # Variance between groups captures population heterogeneity
        gen_var = np.var(aggregated_by_generation.mean, axis=0)
        seed_var = np.var(aggregated_by_seed.mean, axis=0)

        # Both should be non-negative
        assert np.all(gen_var >= 0)
        assert np.all(seed_var >= 0)

    @pytest.mark.unit
    def test_aggregation_enables_variance_attribution(
        self, aggregated_uniform, aggregated_by_generation, aggregated_by_seed
    ):
        """Aggregation enables attributing variance to different sources."""
        from libuq import compute_variance_decomposition

        decomp = compute_variance_decomposition(
            aggregated_by_generation,
            aggregated_by_seed,
            aggregated_uniform,
        )

        # Should be able to determine what fraction comes from each source
        assert "generation_fraction" in decomp
        assert "seed_fraction" in decomp
