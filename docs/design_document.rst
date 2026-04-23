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
     - Pending — framework ready, needs real data application and report
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
     - Software Complete — consensus RFC not yet written
     - ``cell_cycle.py``
   * - 7
     - Cell cycle variable analysis + per-stage GSA
     - Complete — Strategy 4 wrapper + per-stage PCE/Sobol demonstrated
     - ``cell_cycle.py``, ``examples/uq_pipeline.py``

Four Aggregation Strategies
---------------------------

As specified in RFC006 Section 1:

1. **Uniform**: Across all simulated cells and times (baseline)
2. **By Generation**: Stratified by cell generation
3. **By Lineage Seed**: Stratified by stochastic seed
4. **By Cell Cycle**: Stratified by cell cycle stage

GSA → Cell Cycle Feedback Loop
-------------------------------

Variance decomposition from strategies 1–3 (Uniform, By Generation, By Lineage
Seed) identifies observables with high residual variance — variance not explained
by input parameters — which are candidates for cell-cycle-related dynamics.

Architecture
------------

The implementation parametrizes three steps (RFC006 Section 4):

A. **Selection/Extraction**: ``outputs.py`` - Extract variables from simulation trajectories
B. **Temporal Aggregation**: ``aggregation.py`` - Aggregate into output variables Y
C. **Sensitivity Analysis**: ``sensitivity.py`` - Apply PCE-based methods

Libraries
---------

* **PyTUQ**: Primary library for PCE and Sobol analysis (Sandia National Labs)

Key Documents
-------------

* ``readmes/RFC006.md`` - Authoritative specification
* ``SAMPLING.md`` - How ``uq sample`` delegates to PyTUQ + vEcoli

Full Pipeline Example
---------------------

The ``uq`` CLI demonstrates the complete RFC006 pipeline:

* **Phase 1** (strategies 1–3): PCE surrogate → Sobol indices
  ("Which parameters drive bulk output variance?")
* **Phase 2** (strategy 4): Growth-stratified θ →
  per-stage PCE → per-stage Sobol indices
  ("Which parameters drive variance WITHIN each cell cycle stage?")

Both phases produce a ``UqProfile``, assembled into a ``PipelineResult``.

.. code-block:: bash

   uv run python examples/uq_pipeline.py --output-dir ./my_results

Future Directions
-----------------

* Experimental data ingestion
* Strain design optimization
* ML surrogate models
* XarrayEmitter integration (when available)
