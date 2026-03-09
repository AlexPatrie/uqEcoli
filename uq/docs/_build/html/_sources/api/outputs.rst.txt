Output Variables API
====================

.. module:: uq.outputs

This module provides functions to extract output variables from vEcoli simulations:

* Transcriptome (mRNA counts)
* Proteome (protein counts)
* Metabolic fluxes (particularly exchange fluxes)
* Higher-order properties (mass, volume, growth rate)

Classes
-------

OutputType
^^^^^^^^^^

.. autoclass:: OutputType
   :members:
   :undoc-members:
   :show-inheritance:

OutputVariables
^^^^^^^^^^^^^^^

.. autoclass:: OutputVariables
   :members:
   :undoc-members:
   :show-inheritance:

OutputExtractor
^^^^^^^^^^^^^^^

.. autoclass:: OutputExtractor
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

get_output_variable_info
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: get_output_variable_info

Usage Examples
--------------

Extracting Outputs
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import OutputExtractor, OutputType
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

   extractor = OutputExtractor(conn, history_sql, config_sql)

   # Extract transcriptome
   counts, cistron_ids = extractor.extract_transcriptome(
       generation_lower_bound=2,
       time_lower_bound=100.0,
   )

   # Extract exchange fluxes
   fluxes, rxn_ids = extractor.extract_exchange_fluxes()

   # Extract all outputs
   outputs = extractor.extract_all(
       output_types=[
           OutputType.TRANSCRIPTOME,
           OutputType.EXCHANGE_FLUXES,
           OutputType.HIGHER_ORDER_PROPERTIES,
       ],
   )

Output Column Names
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Output Type
     - Parquet Column
   * - Transcriptome
     - ``listeners__rna_counts__mRNA_cistron_counts``
   * - Proteome
     - ``listeners__monomer_counts``
   * - Metabolic Fluxes
     - ``listeners__fba_results__base_reaction_fluxes``
   * - Cell Mass
     - ``listeners__mass__cell_mass``
   * - Dry Mass
     - ``listeners__mass__dry_mass``
   * - Volume
     - ``listeners__mass__volume``
