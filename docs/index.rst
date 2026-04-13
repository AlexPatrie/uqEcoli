uqEcoli — UQ for the vEcoli whole-cell model
============================================

``uqEcoli`` is a thin glue layer between **vEcoli**
(<https://covertlab.github.io/vEcoli/>) and **PyTUQ's UQPC workflow**
(<https://sandialabs.github.io/pytuq/apps/uqpc.html>).  It turns the
six-step UQPC pipeline (setup → sample → evaluate → surrogate → error →
Sobol) into a two-command CLI backed by vEcoli's Nextflow-driven
simulations.

.. important::

   Every numerically-meaningful step lives in PyTUQ or vEcoli.  This
   repository only adapts their I/O formats and adds end-user entry
   points (CLI, TUI, GUI, marimo dashboard).  See ``SAMPLING.md`` at the
   repo root for a full accounting of *what lives where*.

Two-stage workflow
------------------

.. code-block:: bash

   # Stage 1 — UQPC steps 1-3: germ sampling + vEcoli evaluation
   uv run uq sample /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --n-samples 200 \
       --n-test 40 \
       --generations 2 \
       --observables higher_order \
       --observables exchange_fluxes \
       --observables transcriptome \
       --generation-lower-bound 2

   # Stage 2 — UQPC steps 4-5: PCE surrogate fit + Sobol decomposition
   uv run uq quantify /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --export-path ./uq_results \
       --polynomial-order 3 \
       --regression lsq

See :doc:`cli_reference` for every flag, and :doc:`tutorial_workflow` for
the full mathematical walkthrough.

Why two stages?
---------------

The expensive operation is step 3 (running vEcoli).  Caching its output
means you can iterate freely on PCE order, regression backend, or
aggregation strategy without re-simulating.  The cache directory
contains ``X.npy``, ``Y.npy``, ``germ_train.npy``, optional
``X_test.npy``/``Y_test.npy`` (UQPC ``--ntst``), and per-sample Parquet
timeseries — everything ``quantify`` needs.

User-facing entry points
------------------------

All four clients expose the same functionality as different shells over
the workflow in ``uq.workflow``:

============  =====================================  =========================================
Client        Command                                Best for
============  =====================================  =========================================
CLI (Rich)    ``uv run uq sample`` / ``quantify``    Headless runs, scripts, CI
TUI           ``uv run uq tui``                      Terminal dashboards with live progress
GUI (marimo)  ``uv run uq gui``                      Reactive browser notebook
Dashboard     ``uv run uq dashboard``                Draggable DAW-style result exploration
============  =====================================  =========================================

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   cli_reference
   tutorial_workflow

.. toctree::
   :maxdepth: 2
   :caption: Topics

   aggregation_strategies
   sensitivity_analysis
   cell_cycle
   koopman

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

* PyTUQ UQPC workflow: https://sandialabs.github.io/pytuq/apps/uqpc.html
* vEcoli workflows + variants: https://covertlab.github.io/vEcoli/workflows.html
* Sudret, B. (2008). *Global sensitivity analysis using polynomial chaos expansions*.
  Reliability Engineering & System Safety 93(7), 964-979.
* Xiu, D. & Karniadakis, G.E. (2002). *The Wiener–Askey polynomial chaos for
  stochastic differential equations*. SIAM J. Sci. Comput. 24(2), 619-644.
* Macklin, D. N. *et al.* (2020). *Simultaneous cross-evaluation of heterogeneous
  E. coli datasets via mechanistic simulation*. Science 369(6502).
* Ahn-Horst, T. A. *et al.* (2022). *An expanded whole-cell model of E. coli
  links cellular physiology with mechanisms of growth rate control*. npj
  Systems Biology and Applications 8:30.
