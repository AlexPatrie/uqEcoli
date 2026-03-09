Aggregation API
===============

.. module:: uq.aggregation

This module implements the four aggregation strategies for uncertainty quantification:

1. Uniform: Across all cells and times (baseline)
2. Stratified by generation
3. Stratified by lineage seed
4. Stratified by cell cycle stage

Classes
-------

AggregationStrategy
^^^^^^^^^^^^^^^^^^^

.. autoclass:: AggregationStrategy
   :members:
   :undoc-members:
   :show-inheritance:

AggregatedOutput
^^^^^^^^^^^^^^^^

.. autoclass:: AggregatedOutput
   :members:
   :undoc-members:
   :show-inheritance:

Aggregator
^^^^^^^^^^

.. autoclass:: Aggregator
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

compute_variance_decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: compute_variance_decomposition

Usage Examples
--------------

Basic Aggregation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import Aggregator, AggregationStrategy
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

   aggregator = Aggregator(conn, history_sql, config_sql)

   # Uniform aggregation
   result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)

   # By generation
   result, ids = aggregator.aggregate_fluxes(
       AggregationStrategy.BY_GENERATION,
       exchange_only=True,
   )

   # By lineage seed
   result, ids = aggregator.aggregate_proteome(
       AggregationStrategy.BY_LINEAGE_SEED,
       generation_lower_bound=2,
   )

Variance Decomposition
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import compute_variance_decomposition

   agg_uniform, _ = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
   agg_by_gen, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)
   agg_by_seed, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

   decomposition = compute_variance_decomposition(
       agg_by_gen, agg_by_seed, agg_uniform
   )

   print(f"Generation fraction: {decomposition['generation_fraction'].mean():.2%}")
   print(f"Seed fraction: {decomposition['seed_fraction'].mean():.2%}")

Working with Results
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   result, ids = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)

   # Access statistics
   print(f"Mean shape: {result.mean.shape}")  # (n_groups, n_features)
   print(f"Std shape: {result.std.shape}")
   print(f"Groups: {result.groups}")          # Generation numbers
   print(f"Samples per group: {result.n_samples}")

   # Convert to DataFrame
   import polars as pl

   df = pl.DataFrame({
       "generation": result.groups,
       "mean": result.mean.tolist(),
       "std": result.std.tolist(),
   })
