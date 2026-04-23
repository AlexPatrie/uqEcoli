Cell Cycle Stratification
=========================

The cell cycle stratification module (Phase 2) enables aggregation strategy 4:
stratifying simulation data by cell cycle stage for "phenotypic" sensitivity
analysis across the physiological time dimension.

Overview
--------

E. coli cells progress through a cell cycle with distinct phases:

* **B period**: Birth to DNA replication initiation
* **C period**: DNA replication
* **D period**: Post-replication to cell division

The UQ framework provides methods to compute "cell cycle variables" that map
continuous simulation data to discrete cell cycle stages.

Cell Cycle Variables
--------------------

A cell cycle variable is a low-dimensional (typically scalar) function of
simulation outputs that:

1. Has approximately cyclic behavior across the cell cycle
2. Can be computed deterministically from process variables
3. Has a quantitative relationship with measurable omics data

Built-in Variables
^^^^^^^^^^^^^^^^^^

The framework provides five built-in implementations:

**Mass-Based** (default)

Uses the normalized log-mass ratio:

.. math::

   \\phi = \\frac{\\log(M) - \\log(M_{birth})}{\\log(M_{div}) - \\log(M_{birth})}

.. code-block:: python

   from uq import MassBasedCellCycleVariable

   cc_var = MassBasedCellCycleVariable()
   # Returns values in [0, 1] representing cell cycle progression

**DNA Replication-Based**

Tracks DNA mass as a proxy for replication progress:

.. code-block:: python

   from uq import DNAReplicationCellCycleVariable

   cc_var = DNAReplicationCellCycleVariable()
   # Also assigns phase labels: B_period, C_period, D_period

**Cell Angle**

2D projection in (log-mass, growth-rate) space:

.. code-block:: python

   from uq import CellAngleCellCycleVariable

   cc_var = CellAngleCellCycleVariable()
   # Approximates established "cell angle" approach from literature

**Growth-Stratified** (default in ``uq quantify``)

The simplified pipeline uses a growth-stratified cell cycle variable
directly in ``uq/growth.py``:

.. math::

   \theta = \frac{\log(m(t)) - \log(m_{birth})}{\log(m_{div}) - \log(m_{birth})}

This monotonic surrogate avoids spectral decomposition entirely and is
used by ``uq quantify`` strategy 4 to bin cells into θ-stages.

Using the Cell Cycle Aggregator
-------------------------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from uq import CellCycleAggregator
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   # Connect to simulation data
   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./output_dir", ["experiment_id"])

   # Create aggregator
   cc_agg = CellCycleAggregator(
       conn=conn,
       history_sql=history_sql,
       config_sql=config_sql,
       variable_type="mass_based",  # or "dna_replication", "cell_angle"
       n_stages=10,                 # Number of cell cycle bins
   )

Computing Cell Cycle Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get profile for a specific output
   profile = cc_agg.get_cell_cycle_profile(
       output_column="listeners__rna_counts__mRNA_cistron_counts",
       generation_lower_bound=2,
       time_lower_bound=100.0,
   )

   print(f"Cell cycle variable: {profile['cell_cycle_variable']}")
   print(f"Number of stages: {profile['n_stages']}")
   print(f"Stage labels: {profile['stage']}")
   print(f"Mean per stage: {profile['mean'].shape}")  # (n_stages, n_features)
   print(f"Std per stage: {profile['std'].shape}")

Direct Aggregation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Aggregate a specific output by cell cycle
   means, stds, stages = cc_agg.aggregate_by_cell_cycle(
       output_column="listeners__mass__dry_mass",
       generation_lower_bound=2,
   )

   # Plot cell cycle profile
   import matplotlib.pyplot as plt

   plt.errorbar(stages, means, yerr=stds, capsize=3)
   plt.xlabel("Cell Cycle Stage")
   plt.ylabel("Dry Mass")
   plt.title("Dry Mass Across Cell Cycle")
   plt.savefig("cell_cycle_profile.png")

Computing Raw Cell Cycle Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the raw cell cycle variable for all data points
   cc_variable = cc_agg.compute_cell_cycle_variable(
       generation_lower_bound=2,
   )

   print(f"Values shape: {cc_variable.values.shape}")
   print(f"Value range: [{cc_variable.values.min():.3f}, {cc_variable.values.max():.3f}]")
   print(f"Normalized: {cc_variable.normalized}")

   # Convert to discrete stages
   stage_bins = cc_variable.to_stage_bins(n_bins=10)

Custom Cell Cycle Variables
---------------------------

Register custom variables for domain-specific analysis:

Using CompositeCellCycleVariable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import CompositeCellCycleVariable, register_cell_cycle_variable
   import numpy as np

   def compute_custom_variable(data):
       """
       Custom cell cycle variable based on RNA/protein ratio.

       Args:
           data: Polars DataFrame with simulation columns

       Returns:
           Normalized cell cycle values (0-1)
       """
       rna_mass = data["listeners__mass__rna_mass"].to_numpy()
       protein_mass = data["listeners__mass__protein_mass"].to_numpy()

       ratio = rna_mass / np.maximum(protein_mass, 1e-10)

       # Normalize to [0, 1]
       return (ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-10)

   custom_var = CompositeCellCycleVariable(
       name="rna_protein_ratio",
       required_columns=[
           "listeners__mass__rna_mass",
           "listeners__mass__protein_mass",
           "generation",
           "agent_id",
           "time",
       ],
       compute_func=compute_custom_variable,
   )

   # Register for use with CellCycleAggregator
   register_cell_cycle_variable("rna_protein_ratio", custom_var)

   # Now use it
   cc_agg = CellCycleAggregator(
       conn, history_sql, config_sql,
       variable_type="rna_protein_ratio",
   )

Implementing CellCycleVariableComputer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control, implement the abstract base class:

.. code-block:: python

   from uq.cell_cycle import CellCycleVariableComputer, CellCycleVariable
   import numpy as np
   import polars as pl

   class MyCustomVariable(CellCycleVariableComputer):

       @property
       def name(self) -> str:
           return "my_custom"

       @property
       def required_columns(self) -> list[str]:
           return [
               "listeners__mass__dry_mass",
               "listeners__mass__dna_mass",
               "generation",
               "agent_id",
               "time",
           ]

       def compute(self, data: pl.DataFrame, sim_data=None) -> CellCycleVariable:
           # Your computation logic
           dry_mass = data["listeners__mass__dry_mass"].to_numpy()
           dna_mass = data["listeners__mass__dna_mass"].to_numpy()

           # Example: DNA/mass ratio normalized
           values = dna_mass / np.maximum(dry_mass, 1e-10)
           values = (values - values.min()) / (values.max() - values.min() + 1e-10)

           return CellCycleVariable(
               values=values,
               normalized=True,
               variable_name=self.name,
               metadata={"method": "dna_mass_ratio"},
           )

   # Register
   from uq import CellCycleAggregator
   CellCycleAggregator.VARIABLES["my_custom"] = MyCustomVariable

Integration with Sensitivity Analysis
-------------------------------------

Cell cycle profiles can be used as outputs for sensitivity analysis:

.. code-block:: python

   from uq import (
       CellCycleAggregator,
       InputParameterSpace,
       SensitivityAnalyzer,
   )
   import numpy as np

   # Extract cell cycle profiles as output vector
   def extract_cell_cycle_outputs(output_dir):
       conn = create_duckdb_conn()
       history_sql, config_sql, _ = dataset_sql(output_dir, ["*"])

       cc_agg = CellCycleAggregator(
           conn, history_sql, config_sql,
           variable_type="mass_based",
           n_stages=5,
       )

       profile = cc_agg.get_cell_cycle_profile(
           "listeners__rna_counts__mRNA_cistron_counts"
       )

       # Flatten profile means as output vector
       return profile["mean"].flatten()

   # Use in sensitivity analysis wrapper
   # (See wrappers documentation for full integration)

Visualization
-------------

Plot Cell Cycle Profiles
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   profile = cc_agg.get_cell_cycle_profile("listeners__mass__dry_mass")

   fig, ax = plt.subplots(figsize=(8, 5))
   stages = profile["stage"]
   means = profile["mean"].flatten()
   stds = profile["std"].flatten()

   ax.fill_between(stages, means - stds, means + stds, alpha=0.3)
   ax.plot(stages, means, 'o-', linewidth=2)
   ax.set_xlabel("Cell Cycle Stage")
   ax.set_ylabel("Dry Mass")
   ax.set_title("Dry Mass Profile Across Cell Cycle")
   plt.savefig("cell_cycle_dry_mass.png", dpi=150)

Compare Variables
^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig, axes = plt.subplots(1, 3, figsize=(12, 4))

   for ax, var_type in zip(axes, ["mass_based", "dna_replication", "cell_angle"]):
       cc_agg = CellCycleAggregator(
           conn, history_sql, config_sql,
           variable_type=var_type,
           n_stages=10,
       )
       profile = cc_agg.get_cell_cycle_profile("listeners__mass__dry_mass")

       ax.errorbar(profile["stage"], profile["mean"].flatten(),
                   yerr=profile["std"].flatten(), capsize=3)
       ax.set_title(var_type)
       ax.set_xlabel("Stage")
       ax.set_ylabel("Dry Mass")

   plt.tight_layout()
   plt.savefig("compare_cell_cycle_variables.png", dpi=150)

Best Practices
--------------

1. **Choose appropriate variable**: Mass-based is robust; DNA-based captures replication
2. **Select n_stages carefully**: 5-10 stages balances resolution with statistical power
3. **Filter early generations**: Use ``generation_lower_bound`` to skip transients
4. **Validate with known biology**: Check that profiles match expected cell cycle behavior
5. **Consider multiple variables**: Compare results across different cell cycle definitions

Future Directions
-----------------

The consensus RFC (Activity 6) will determine the recommended default cell cycle
variable for CD2 evaluation. The framework already supports extensibility via
``register_cell_cycle_variable()``, so new definitions can be added without
modifying core code.

See Also
--------

* :doc:`api/cell_cycle` - Full API reference
* :doc:`tutorials/cell_cycle_analysis` - Hands-on tutorial
* :doc:`aggregation_strategies` - Overview of all aggregation strategies
