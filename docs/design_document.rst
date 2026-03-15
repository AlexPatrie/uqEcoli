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
Seed) selects observables for Koopman DMD cell cycle mode extraction. Observables
with high residual variance — variance not explained by input parameters — are
candidates for cell-cycle-related dynamics. The ``GSAInformedCellCycleVariable``
class implements this feedback loop: it takes aggregated sensitivity results,
identifies observables whose variance is dominated by cell cycle effects, and
feeds those observables into Koopman DMD to extract cell cycle modes.

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

Apollo Package
--------------

The Apollo package provides sonification of UQ results:

* **Layer 1**: Koopman modes → musical notes (existing ``apollo/`` module)
* **Layer 2**: UQ pipeline outputs → musical score (``uq_score.py``) — maps
  Sobol indices, variance decomposition, and per-stage sensitivity to Western
  musical notation

Key Documents
-------------

* ``uq/RFC006.md`` - Authoritative specification
* ``uq/RFC006_VERIFICATION.md`` - Compliance analysis
* ``uq/PIPELINE.md`` - Complete 7-step UQ workflow
* ``apollo/README.md`` - Apollo sonification package
* ``readmes/CONTEXT.md`` - Claude context document

Full Pipeline Example
---------------------

The ``examples/uq_pipeline.py`` script demonstrates the complete RFC006 pipeline:

* **Phase 1** (Steps 5a–7a): Morris prescreening → PCE surrogate → Sobol indices
  ("Which parameters drive bulk output variance?")
* **Phase 2** (Steps 5b–7b): GSA-informed observable selection → Koopman θ →
  Strategy 4 wrapper → per-stage PCE → per-stage Sobol indices
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
