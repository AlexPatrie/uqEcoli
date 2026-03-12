Getting Started
===============

This guide will help you get started with the vEcoli UQ Framework for uncertainty
quantification in whole-cell simulations.

Prerequisites
-------------

Before using the UQ framework, ensure you have:

1. A working vEcoli installation
2. Simulation data in Parquet format (from ``ParquetEmitter``)
3. A ``sim_data`` pickle file from ParCa

Installation
------------

The UQ framework is included with vEcoli. Install the optional UQ dependencies:

.. code-block:: bash

   # Install vEcoli with UQ dependencies
   pip install -e ".[uq]"

   # Or install UQPy separately
   pip install UQpy>=4.1.0

Basic Concepts
--------------

The UQ framework is built around four key concepts:

Input Parameters
^^^^^^^^^^^^^^^^

Input parameters define the experimental conditions being varied:

* **Violacein (vio) pathway**: New gene expression parameters
* **Mecillinam**: Antibiotic stress conditions
* **Gene knockouts**: Gene deletion experiments

.. code-block:: python

   from uq import InputParameterSpaceVecoli

   param_space = InputParameterSpaceVecoli(
       include_vio=True,
       include_mecillinam=True,
       vio_expression_bounds=(0.0, 5.0),
       mecillinam_conc_bounds=(0.0, 10.0),
   )

Output Variables
^^^^^^^^^^^^^^^^

Output variables are the simulation results being analyzed:

* **Transcriptome**: mRNA counts per cistron
* **Proteome**: Protein monomer counts
* **Metabolic fluxes**: Reaction rates (especially exchange fluxes)
* **Higher-order properties**: Mass, volume, growth rate

.. code-block:: python

   from uq import OutputType

   output_types = [
       OutputType.TRANSCRIPTOME,
       OutputType.EXCHANGE_FLUXES,
       OutputType.HIGHER_ORDER_PROPERTIES,
   ]

Aggregation Strategies
^^^^^^^^^^^^^^^^^^^^^^

Four strategies for aggregating simulation data:

1. **Uniform**: Average across all cells and times (baseline)
2. **By Generation**: Stratify by cell generation
3. **By Lineage Seed**: Stratify by stochastic seed
4. **By Cell Cycle**: Stratify by cell cycle stage

.. code-block:: python

   from uq import AggregationStrategy

   strategy = AggregationStrategy.BY_GENERATION

Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^

PCE (Polynomial Chaos Expansion) surrogate method for computing Sobol indices:

.. code-block:: python

   from uq import SensitivityAnalyzer

   analyzer = SensitivityAnalyzer(param_space, wrapper)
   sobol, pce = analyzer.analyze_with_pce(polynomial_order=3)

Morris Screening
^^^^^^^^^^^^^^^^

For high-dimensional parameter spaces, Morris screening provides an efficient pre-screening step:

.. code-block:: python

   from uq import SensitivityAnalyzer

   morris = analyzer.analyze_with_morris(n_trajectories=20)
   top_params = morris.get_screening_candidates(top_n=5)

Your First Analysis
-------------------

Here's a complete example of running a sensitivity analysis:

.. code-block:: python

   from uq import (
       InputParameterSpace,
       WrapperConfig,
       SimulationWrapper,
       SensitivityAnalyzer,
       AggregationStrategy,
       OutputType,
   )

   # Step 1: Define parameter space
   param_space = InputParameterSpace(
       include_vio=True,
       include_mecillinam=True,
   )

   # Step 2: Configure the wrapper
   config = WrapperConfig(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_analysis",
       cache_dir="./uq_cache",
       generations=8,
       aggregation_strategy=AggregationStrategy.UNIFORM,
       output_types=[
           OutputType.EXCHANGE_FLUXES,
           OutputType.HIGHER_ORDER_PROPERTIES,
       ],
       generation_lower_bound=2,  # Skip initial generations
   )

   # Step 3: Create wrapper and analyzer
   wrapper = SimulationWrapper(config, param_space)
   analyzer = SensitivityAnalyzer(param_space, wrapper)

   # Step 4: Run sensitivity analysis
   sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
       polynomial_order=3,
       n_samples=100,
   )

   # Step 5: Interpret results
   print("Parameter sensitivity ranking:")
   for name, value in sobol_indices.get_most_influential(n=5):
       print(f"  {name}: {value:.4f}")

   print(f"\nFirst-order indices: {sobol_indices.first_order}")
   print(f"Total-order indices: {sobol_indices.total_order}")

Using Precomputed Results
-------------------------

If you already have simulation results, you can analyze them directly:

.. code-block:: python

   from uq import analyze_precomputed_results, AggregationStrategy

   sobol, surrogate = analyze_precomputed_results(
       data_dir="./simulation_outputs",
       aggregation_strategy=AggregationStrategy.BY_GENERATION,
       polynomial_order=3,
   )

Next Steps
----------

* Learn about :doc:`aggregation_strategies` in detail
* Explore :doc:`sensitivity_analysis` methods
* Try the :doc:`tutorials/basic_sensitivity` tutorial
* See the :doc:`api/inputs` API reference
* See :doc:`../uq/PIPELINE` for the complete 7-step workflow
