Aggregation Strategies
======================

The UQ framework implements four aggregation strategies for characterizing different
types of uncertainty in vEcoli simulations. Each strategy enables analysis of
variance from different sources.

Overview
--------

.. list-table:: Aggregation Strategies
   :header-rows: 1
   :widths: 20 30 50

   * - Strategy
     - Groups By
     - Purpose
   * - Uniform
     - None (all data)
     - Baseline "bulk" population average
   * - By Generation
     - Cell generation number
     - Control for convergence towards steady-state
   * - By Lineage Seed
     - Stochastic seed value
     - Control for exogenous variance
   * - By Cell Cycle
     - Cell cycle stage
     - Analyze variance across physiological time

Strategy 1: Uniform Aggregation
-------------------------------

The **uniform** strategy averages across all cells and all time points, producing
the baseline "bulk" population average that corresponds to experimental measurements.

.. code-block:: python

   from uq import Aggregator, AggregationStrategy

   aggregator = Aggregator(conn, history_sql, config_sql)
   result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

   print(f"Mean shape: {result.mean.shape}")  # (n_features,)
   print(f"Std shape: {result.std.shape}")    # (n_features,)
   print(f"N samples: {result.n_samples}")

**Use case**: Comparing simulation predictions to bulk experimental measurements
(e.g., population-averaged RNA-seq or proteomics data).

Strategy 2: By Generation
-------------------------

The **by generation** strategy stratifies data by cell generation number, enabling
analysis of how outputs evolve as the simulation converges towards steady-state growth.

.. code-block:: python

   result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

   print(f"Mean shape: {result.mean.shape}")  # (n_generations, n_features)
   print(f"Generations: {result.groups}")     # [0, 1, 2, ...]

   # Analyze convergence
   import numpy as np
   gen_variance = np.var(result.mean, axis=0)
   print(f"Variance across generations: {gen_variance.mean():.4f}")

**Use case**:

* Verifying simulation has reached steady-state
* Identifying outputs that vary with cell age
* Filtering out initial transient behavior

Strategy 3: By Lineage Seed
---------------------------

The **by lineage seed** strategy stratifies data by the stochastic seed used to
initialize each lineage, enabling analysis of exogenous variance (variance due
to random initial conditions).

.. code-block:: python

   result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

   print(f"Mean shape: {result.mean.shape}")  # (n_seeds, n_features)
   print(f"Seeds: {result.groups}")           # [0, 1, 2, ...]

   # Analyze seed-to-seed variance
   seed_variance = np.var(result.mean, axis=0)
   print(f"Variance across seeds: {seed_variance.mean():.4f}")

**Use case**:

* Quantifying stochastic variability
* Determining how many lineages are needed for robust statistics
* Separating intrinsic from extrinsic noise

Strategy 4: By Cell Cycle
-------------------------

The **by cell cycle** strategy stratifies data by cell cycle stage, enabling
"phenotypic" sensitivity analysis across the physiological time dimension
within a cell's lifespan.

.. code-block:: python

   from uq import CellCycleAggregator

   cc_agg = CellCycleAggregator(
       conn, history_sql, config_sql,
       variable_type="mass_based",  # or "dna_replication", "cell_angle"
       n_stages=10,
   )

   profile = cc_agg.get_cell_cycle_profile(
       "listeners__rna_counts__mRNA_cistron_counts"
   )

   print(f"Stages: {profile['stage']}")
   print(f"Mean per stage: {profile['mean'].shape}")  # (n_stages, n_features)

**Use case**:

* Analyzing cell cycle-dependent gene expression
* Understanding temporal dynamics within cell division
* Preparing for CD2 population aggregation requirements

Cell Cycle Variables
^^^^^^^^^^^^^^^^^^^^

Three built-in cell cycle variable implementations:

1. **Mass-based** (default): Normalized log-mass ratio

   .. math::

      \\phi = \\frac{\\log(M) - \\log(M_{birth})}{\\log(M_{div}) - \\log(M_{birth})}

2. **DNA replication-based**: DNA mass normalized within each cell

3. **Cell angle**: 2D angle in (log-mass, growth-rate) space

See :doc:`cell_cycle` for details on cell cycle variables.

Variance Decomposition
----------------------

The framework can decompose total variance into components attributable to
different factors:

.. code-block:: python

   from uq import compute_variance_decomposition

   # Get aggregated results
   agg_uniform, _ = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
   agg_by_gen, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)
   agg_by_seed, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

   # Decompose variance
   decomposition = compute_variance_decomposition(
       agg_by_gen,
       agg_by_seed,
       agg_uniform,
   )

   print(f"Total variance: {decomposition['total_variance'].mean():.4f}")
   print(f"Generation fraction: {decomposition['generation_fraction'].mean():.2%}")
   print(f"Seed fraction: {decomposition['seed_fraction'].mean():.2%}")

The decomposition returns:

* ``total_variance``: Total variance from uniform aggregation
* ``between_generation_variance``: Variance of generation means
* ``between_seed_variance``: Variance of seed means
* ``generation_fraction``: Fraction attributable to generation
* ``seed_fraction``: Fraction attributable to stochastic seeding

Filtering Data
--------------

All aggregation methods support filtering by generation and time:

.. code-block:: python

   # Skip initial generations (transient)
   result, ids = aggregator.aggregate_transcriptome(
       AggregationStrategy.UNIFORM,
       generation_lower_bound=2,
   )

   # Skip initial time within each cell
   result, ids = aggregator.aggregate_transcriptome(
       AggregationStrategy.BY_GENERATION,
       time_lower_bound=100.0,  # seconds
   )

   # Combine both filters
   result, ids = aggregator.aggregate_fluxes(
       AggregationStrategy.BY_LINEAGE_SEED,
       generation_lower_bound=2,
       time_lower_bound=100.0,
       exchange_only=True,  # Only exchange fluxes
   )

Choosing a Strategy
-------------------

.. list-table:: Strategy Selection Guide
   :header-rows: 1
   :widths: 40 60

   * - If you want to...
     - Use this strategy
   * - Compare with bulk experimental data
     - Uniform
   * - Check simulation convergence
     - By Generation
   * - Quantify stochastic variability
     - By Lineage Seed
   * - Analyze cell cycle dynamics
     - By Cell Cycle
   * - Decompose variance sources
     - All strategies + ``compute_variance_decomposition``

See Also
--------

* :doc:`api/aggregation` - Full API reference
* :doc:`tutorials/variance_decomposition` - Tutorial on variance analysis
* :doc:`cell_cycle` - Cell cycle stratification details
