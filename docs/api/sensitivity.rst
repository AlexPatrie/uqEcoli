Sensitivity Analysis API
========================

.. module:: uq.sensitivity

This module implements global sensitivity analysis using PCE (Polynomial Chaos
Expansion) surrogate methods and Sobol indices.

Classes
-------

SensitivityMethod
^^^^^^^^^^^^^^^^^

.. autoclass:: SensitivityMethod
   :members:
   :undoc-members:
   :show-inheritance:

SobolIndices
^^^^^^^^^^^^

.. autoclass:: SobolIndices
   :members:
   :undoc-members:
   :show-inheritance:

PCESurrogate
^^^^^^^^^^^^

.. autoclass:: PCESurrogate
   :members:
   :undoc-members:
   :show-inheritance:

SensitivityAnalyzer
^^^^^^^^^^^^^^^^^^^

.. autoclass:: SensitivityAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

run_sensitivity_analysis
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: run_sensitivity_analysis

analyze_precomputed_results
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: analyze_precomputed_results

Usage Examples
--------------

PCE-Based Analysis
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import SensitivityAnalyzer, InputParameterSpace

   param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
   analyzer = SensitivityAnalyzer(param_space, wrapper)

   sobol, pce = analyzer.analyze_with_pce(
       polynomial_order=3,
       n_samples=100,
       use_uqpy=True,
   )

   # Get most influential parameters
   for name, value in sobol.get_most_influential(n=5):
       print(f"{name}: {value:.4f}")

Direct Sobol Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sobol = analyzer.analyze_with_sobol(
       n_samples=1024,
       calc_second_order=True,
   )

   print(f"First-order: {sobol.first_order}")
   print(f"Total-order: {sobol.total_order}")
   print(f"Second-order: {sobol.second_order}")

Convenience Functions
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import run_sensitivity_analysis, AggregationStrategy

   sobol, pce = run_sensitivity_analysis(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_analysis",
       aggregation_strategy=AggregationStrategy.UNIFORM,
       polynomial_order=3,
       include_vio=True,
       include_mecillinam=True,
   )

Precomputed Results
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import analyze_precomputed_results

   sobol, pce = analyze_precomputed_results(
       data_dir="./simulation_outputs",
       aggregation_strategy=AggregationStrategy.BY_GENERATION,
       polynomial_order=3,
   )

Working with Results
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Access indices
   print(f"Parameters: {sobol.parameter_names}")
   print(f"First-order: {sobol.first_order}")
   print(f"Total-order: {sobol.total_order}")

   # Check for interactions
   interactions = sobol.total_order - sobol.first_order
   print(f"Interaction effects: {interactions}")

   # Multi-output case
   if sobol.first_order.ndim > 1:
       avg_s1 = np.mean(sobol.first_order, axis=0)
       print(f"Average first-order: {avg_s1}")
