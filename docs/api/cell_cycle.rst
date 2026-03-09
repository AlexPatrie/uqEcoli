Cell Cycle API
==============

.. module:: uq.cell_cycle

This module implements cell cycle stratification (aggregation strategy 4) for
analyzing variance across physiological time within a cell's lifespan.

Classes
-------

CellCyclePhase
^^^^^^^^^^^^^^

.. autoclass:: CellCyclePhase
   :members:
   :undoc-members:
   :show-inheritance:

CellCycleVariable
^^^^^^^^^^^^^^^^^

.. autoclass:: CellCycleVariable
   :members:
   :undoc-members:
   :show-inheritance:

CellCycleVariableComputer
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CellCycleVariableComputer
   :members:
   :undoc-members:
   :show-inheritance:

MassBasedCellCycleVariable
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MassBasedCellCycleVariable
   :members:
   :undoc-members:
   :show-inheritance:

DNAReplicationCellCycleVariable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DNAReplicationCellCycleVariable
   :members:
   :undoc-members:
   :show-inheritance:

CellAngleCellCycleVariable
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CellAngleCellCycleVariable
   :members:
   :undoc-members:
   :show-inheritance:

CompositeCellCycleVariable
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CompositeCellCycleVariable
   :members:
   :undoc-members:
   :show-inheritance:

CellCycleAggregator
^^^^^^^^^^^^^^^^^^^

.. autoclass:: CellCycleAggregator
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

register_cell_cycle_variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: register_cell_cycle_variable

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from uq import CellCycleAggregator
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

   cc_agg = CellCycleAggregator(
       conn=conn,
       history_sql=history_sql,
       config_sql=config_sql,
       variable_type="mass_based",
       n_stages=10,
   )

   profile = cc_agg.get_cell_cycle_profile(
       "listeners__rna_counts__mRNA_cistron_counts"
   )

Custom Variables
^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import CompositeCellCycleVariable, register_cell_cycle_variable

   def compute_custom(data):
       mass = data["listeners__mass__dry_mass"].to_numpy()
       return (mass - mass.min()) / (mass.max() - mass.min())

   custom = CompositeCellCycleVariable(
       name="custom",
       required_columns=["listeners__mass__dry_mass", "generation", "agent_id", "time"],
       compute_func=compute_custom,
   )

   register_cell_cycle_variable("custom", custom)

Available Variables
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Description
   * - ``mass_based``
     - Normalized log-mass ratio (default)
   * - ``dna_replication``
     - DNA mass progression with phase labels
   * - ``cell_angle``
     - 2D angle in (mass, growth) space
