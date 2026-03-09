Design Document
===============

This page documents the design rationale and implementation status of the
vEcoli UQ Framework.

.. note::
   This package implements **RFC006** (``uq/RFC006.md``), the authoritative
   specification for uncertainty quantification in vEcoli.

   For compliance status, see ``uq/RFC006_VERIFICATION.md``.

   For the Claude context document, see :download:`CONTEXT.md <../CONTEXT.md>`.

Vision
------

The UQ framework satisfies **Milestone 08.4.2**: Implement uncertainty quantification
framework to track prediction confidence, as specified in RFC006.

Key goals:

1. Characterize different types of uncertainty
2. Map single-cell to bulk simulations
3. Enable population-level perturbation analysis (Milestone 10)
4. Support CD2 evaluation requirements

RFC006 Compliance
-----------------

The package implements all requirements specified in RFC006:

**Phase 1 (MS-08.4.2)**

.. list-table::
   :header-rows: 1
   :widths: 10 50 20 20

   * - #
     - Activity
     - Status
     - Module
   * - 1
     - Identify input/output variables
     - Complete
     - ``inputs.py``, ``outputs.py``
   * - 2
     - Enable output via emitter
     - Deviation (ParquetEmitter)
     - ``outputs.py``
   * - 3
     - Implement wrapper functions
     - Complete
     - ``wrappers.py``
   * - 4
     - Implement sensitivity analysis (PCE)
     - Complete
     - ``sensitivity.py``
   * - 5
     - Apply to representative simulations
     - Pending
     - Framework ready

**Phase 2 (CD2/Milestone 10)**

.. list-table::
   :header-rows: 1
   :widths: 10 50 20 20

   * - #
     - Activity
     - Status
     - Module
   * - 6
     - Cell cycle stratification strategy
     - Framework Complete
     - ``cell_cycle.py``
   * - 7
     - Cell cycle variable analysis
     - Framework Complete
     - ``cell_cycle.py``

Four Aggregation Strategies
---------------------------

As specified in RFC006 Section 1:

1. **Uniform**: Across all simulated cells and times (baseline)
2. **By Generation**: Stratified by cell generation
3. **By Lineage Seed**: Stratified by stochastic seed
4. **By Cell Cycle**: Stratified by cell cycle stage

Architecture
------------

The implementation parametrizes three steps (RFC006 Section 4):

A. **Selection/Extraction**: ``outputs.py`` - Extract variables from simulation trajectories
B. **Temporal Aggregation**: ``aggregation.py`` - Aggregate into output variables Y
C. **Sensitivity Analysis**: ``sensitivity.py`` - Apply PCE-based methods

Additional module for complementary analysis:

D. **Koopman Spectral Analysis**: ``koopman.py`` - Dynamic mode decomposition for cell cycle harmonics

Libraries
---------

* **UQPy**: Primary library for PCE and Sobol analysis
* **PyTUQ**: Alternative compatible library

Koopman Spectral Analysis
-------------------------

The ``koopman.py`` module provides a complementary "harmonic" view of simulation
dynamics using Dynamic Mode Decomposition (DMD). This enables:

* Extraction of dominant frequencies and growth rates
* Identification of cell cycle harmonics
* Spectral sensitivity analysis

See :doc:`koopman` for details.

Key Documents
-------------

* ``uq/RFC006.md`` - Authoritative specification
* ``uq/RFC006_VERIFICATION.md`` - Compliance analysis
* ``uq/CONTEXT.md`` - Claude context document

Future Directions
-----------------

* Experimental data ingestion
* Strain design optimization
* ML surrogate models
* XarrayEmitter integration (when available)
