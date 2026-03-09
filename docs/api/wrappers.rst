Wrappers API
============

.. module:: uq.wrappers

This module provides wrapper functions that interface vEcoli simulations with
UQPy and PyTUQ uncertainty quantification libraries.

Classes
-------

WrapperConfig
^^^^^^^^^^^^^

.. autoclass:: WrapperConfig
   :members:
   :undoc-members:
   :show-inheritance:

SimulationWrapper
^^^^^^^^^^^^^^^^^

.. autoclass:: SimulationWrapper
   :members:
   :undoc-members:
   :show-inheritance:

PrecomputedWrapper
^^^^^^^^^^^^^^^^^^

.. autoclass:: PrecomputedWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

create_uqpy_model
^^^^^^^^^^^^^^^^^

.. autofunction:: create_uqpy_model

create_pytuq_model
^^^^^^^^^^^^^^^^^^

.. autofunction:: create_pytuq_model

Usage Examples
--------------

SimulationWrapper
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import (
       InputParameterSpace,
       WrapperConfig,
       SimulationWrapper,
       AggregationStrategy,
       OutputType,
   )

   param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)

   config = WrapperConfig(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_outputs",
       cache_dir="./uq_cache",
       generations=8,
       aggregation_strategy=AggregationStrategy.UNIFORM,
       output_types=[OutputType.EXCHANGE_FLUXES],
       generation_lower_bound=2,
       use_cache=True,
   )

   wrapper = SimulationWrapper(config, param_space)

   # Evaluate single sample
   import numpy as np
   x = np.array([2.0, 1.0, 5.0])  # vio_exp, vio_trl, mec_conc
   y = wrapper(x)

   # Evaluate batch
   X = np.random.uniform(0, 1, size=(10, 3))
   Y = wrapper.evaluate_batch(X)

PrecomputedWrapper
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import PrecomputedWrapper, AggregationStrategy

   wrapper = PrecomputedWrapper(
       data_dir="./simulation_outputs",
       parameter_space=param_space,
       aggregation_strategy=AggregationStrategy.BY_GENERATION,
   )

   # Get all samples and outputs
   X, Y = wrapper.get_samples_and_outputs()
   print(f"Samples: {X.shape}, Outputs: {Y.shape}")

UQPy Integration
^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import create_uqpy_model

   model = create_uqpy_model(config, param_space)

   # Use with UQPy
   from UQpy.sampling import LatinHypercubeSampling
   from UQpy.distributions import JointIndependent

   distributions = param_space.get_uqpy_distributions()
   joint = JointIndependent(marginals=distributions)
   lhs = LatinHypercubeSampling(distributions=joint, nsamples=50)

PyTUQ Integration
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import create_pytuq_model

   model_func = create_pytuq_model(config, param_space)
   lb, ub = param_space.get_pytuq_bounds()

   # model_func(X) returns outputs for parameter array X

Caching
^^^^^^^

The ``SimulationWrapper`` caches results to avoid redundant simulations:

.. code-block:: python

   config = WrapperConfig(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_outputs",
       cache_dir="./uq_cache",
       use_cache=True,  # Enable caching (default)
   )

   wrapper = SimulationWrapper(config, param_space)

   # First call runs simulation
   y1 = wrapper(x)

   # Second call loads from cache
   y2 = wrapper(x)  # Fast!
