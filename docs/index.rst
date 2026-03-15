.. vEcoli UQ Framework documentation master file

vEcoli Uncertainty Quantification Framework
===========================================

A comprehensive framework for tracking prediction confidence in vEcoli whole-cell
simulations, implementing **RFC006** requirements for **Milestone 08.4.2** (extensibility)
and laying groundwork for **Milestone 10.2.3** (population-level perturbation analysis).

.. note::
   This package implements the UQ framework specified in **RFC006** (``uq/RFC006.md``).
   See ``uq/RFC006_VERIFICATION.md`` for the compliance analysis and implementation status.

Overview
--------

The UQ framework addresses the need to:

* **Track prediction confidence** in whole-cell model outputs
* **Deconvolve uncertainty types**: by cell, by lineage, by generation, and across the cell cycle
* **Map single-cell to bulk** simulations for comparison with experimental measurements
* **Enable population-level analysis** for CD2 evaluation requirements

Key Features
------------

**Input Parameters** (:doc:`api/inputs`)
   Define scientifically relevant inputs: violacein pathway, mecillinam conditions,
   and gene knockouts.

**Output Extraction** (:doc:`api/outputs`)
   Extract transcriptome, proteome, metabolic fluxes, and higher-order properties
   from simulation data.

**Aggregation Strategies** (:doc:`api/aggregation`)
   Four strategies: uniform, by generation, by lineage seed, and by cell cycle stage.

**Sensitivity Analysis** (:doc:`api/sensitivity`)
   PCE surrogate method with Sobol indices using UQPy or PyTUQ libraries.

**Koopman Spectral Analysis** (:doc:`api/koopman`)
   Dynamic Mode Decomposition for extracting dynamical modes and cell cycle harmonics.

Quick Start
-----------

.. code-block:: python

   from uq import (
       InputParameterSpace,
       WrapperConfig,
       SimulationWrapper,
       SensitivityAnalyzer,
       AggregationStrategy,
   )

   # Define parameter space
   param_space = InputParameterSpace(
       include_vio=True,
       include_mecillinam=True,
   )

   # Configure wrapper
   config = WrapperConfig(
       sim_data_path="/path/to/sim_data.cPickle",
       output_dir="./uq_outputs",
       aggregation_strategy=AggregationStrategy.UNIFORM,
   )

   # Run sensitivity analysis
   wrapper = SimulationWrapper(config, param_space)
   analyzer = SensitivityAnalyzer(param_space, wrapper)
   sobol_indices, pce_surrogate = analyzer.analyze_with_pce(polynomial_order=3)

   # Get most influential parameters
   for name, value in sobol_indices.get_most_influential(n=5):
       print(f"{name}: {value:.4f}")

Installation
------------

The UQ framework is included with vEcoli. For sensitivity analysis features,
install the optional UQ dependencies:

.. code-block:: bash

   pip install -e ".[uq]"

This installs `UQPy <https://uqpyproject.readthedocs.io/>`_, the primary library
used for PCE surrogate construction and Sobol sensitivity analysis.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   aggregation_strategies
   sensitivity_analysis
   cell_cycle
   koopman

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_sensitivity
   tutorials/variance_decomposition
   tutorials/cell_cycle_analysis
   tutorials/koopman_analysis
   tutorials/full_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/inputs
   api/outputs
   api/aggregation
   api/wrappers
   api/sensitivity
   api/cell_cycle
   api/koopman

.. toctree::
   :maxdepth: 1
   :caption: Development

   design_document
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

* `UQPy Documentation <https://uqpyproject.readthedocs.io/>`_
* `PyTUQ Documentation <https://sandialabs.github.io/pytuq/>`_
* Sobol, I.M. (2001). "Global sensitivity indices for nonlinear mathematical models"
* Xiu, D. & Karniadakis, G.E. (2002). "The Wiener-Askey polynomial chaos for stochastic differential equations"
