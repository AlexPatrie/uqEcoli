Input Parameters API
====================

.. module:: uq.inputs

This module defines the scientifically most relevant input variables for
uncertainty quantification:

* Violacein (vio) pathway presence
* Mecillinam antibiotic condition
* Gene knockouts

Classes
-------

MediaCondition
^^^^^^^^^^^^^^

.. autoclass:: MediaCondition
   :members:
   :undoc-members:
   :show-inheritance:

VioPathwayParams
^^^^^^^^^^^^^^^^

.. autoclass:: VioPathwayParams
   :members:
   :undoc-members:
   :show-inheritance:

MecillinamParams
^^^^^^^^^^^^^^^^

.. autoclass:: MecillinamParams
   :members:
   :undoc-members:
   :show-inheritance:

GeneKnockoutParams
^^^^^^^^^^^^^^^^^^

.. autoclass:: GeneKnockoutParams
   :members:
   :undoc-members:
   :show-inheritance:

UQInputParameters
^^^^^^^^^^^^^^^^^

.. autoclass:: UQInputParameters
   :members:
   :undoc-members:
   :show-inheritance:

InputParameterSpace
^^^^^^^^^^^^^^^^^^^

.. autoclass:: InputParameterSpace
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Defining Input Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import VioPathwayParams, MecillinamParams, UQInputParameters

   # Violacein pathway
   vio = VioPathwayParams(
       enabled=True,
       induction_gen=1,
       expression=2.5,
       translation_efficiency=1.2,
   )

   # Mecillinam condition
   mec = MecillinamParams(
       times=[0.0, 3600.0],
       concentrations=[0.0, 5.0],
   )

   # Combined parameters
   params = UQInputParameters(vio=vio, mecillinam=mec, seed=42, generations=8)

Creating Parameter Spaces
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import InputParameterSpace

   param_space = InputParameterSpace(
       vio_expression_bounds=(0.0, 5.0),
       vio_trl_eff_bounds=(0.0, 2.0),
       mecillinam_conc_bounds=(0.0, 10.0),
       include_vio=True,
       include_mecillinam=True,
   )

   # Get bounds for UQPy/PyTUQ
   print(f"Parameters: {param_space.parameter_names}")
   print(f"Bounds: {param_space.bounds_array}")
