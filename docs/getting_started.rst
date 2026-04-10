Getting Started
===============

This guide walks through the shortest path from "I have vEcoli and a
``simData.cPickle``" to "I have PCE Sobol indices for five sim_data
parameters."

Prerequisites
-------------

1. A working **vEcoli** checkout at ``../vEcoli`` (editable install).
2. A pre-computed ``simData.cPickle`` produced by vEcoli's Parca
   (for example ``../vEcoli/reconstruction/sim_data/kb/simData.cPickle``).
3. ``uv`` installed (https://docs.astral.sh/uv/).

.. code-block:: bash

   git clone https://github.com/.../uqEcoli.git
   cd uqEcoli
   uv sync --all-groups --all-extras

This repository depends on vEcoli as an editable package, so no further
installation of vEcoli is needed as long as the path above is correct.

The two-stage workflow
----------------------

``uq`` exposes the PyTUQ UQPC workflow as two commands:

* ``uq sample`` — UQPC steps **1-3**: build the input PC, draw germ
  samples, evaluate vEcoli, cache ``(X, Y, timeseries)`` to disk.
* ``uq quantify`` — UQPC steps **4-5**: fit the PCE surrogate, compute
  Sobol indices for all four RFC006 aggregation strategies, report
  surrogate relative errors, and export a dashboard-ready artifact
  directory.

Stage 1 — sample
^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run uq sample /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --n-samples 50 \
       --n-test 10 \
       --generations 2

What happens:

1. ``ParameterDataset`` loads ``simData.cPickle`` and projects the six
   default parameters from :py:data:`libuq.pipeline.param_loader.DEFAULT_SIM_DATA_PARAMETERS`
   into a generic ``XSpaceVecoli`` parameter space.
2. ``_setup_input_pc`` in :py:mod:`uq.workflow` builds a
   ``pytuq.rv.pcrv.PCRV`` (Legendre basis, uniform priors) that encodes
   the affine map from germ space ``[-1, 1]`` to physical bounds.
3. ``input_pc.sampleGerm(n_samples)`` draws 50 germ samples and
   ``evalPC`` maps them into physical parameter space.  With
   ``--n-test 10`` an additional 10 held-out validation samples are
   drawn (UQPC's ``--ntst``).
4. Each row of ``X`` becomes one vEcoli variant via the **upstream**
   ``sim_data_setattr`` variant function.  See ``SAMPLING.md`` for the
   exact mapping.
5. ``runscripts/workflow.py`` (from vEcoli) is spawned as a subprocess
   with the generated workflow config.  Nextflow manages Parca skipping,
   variant instantiation, simulation, and Parquet emission.
6. When vEcoli finishes, hive-partitioned Parquet is collected into
   ``libuq.sampling.PrecomputedCache`` and saved to ``./uq_cache``.
   Test samples are sliced off and stored as ``X_test.npy`` /
   ``Y_test.npy``.

Stage 2 — quantify
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run uq quantify /path/to/simData.cPickle \
       --cache-dir ./uq_cache \
       --export-path ./uq_results \
       --polynomial-order 3 \
       --regression lsq

What happens:

1. The cache is loaded, parameter space is reconstructed, and germ
   samples are read back from ``germ_train.npy``.
2. ``run_uqpc`` fits a multi-output PCE per aggregation strategy using
   PyTUQ's ``pcrv.evalBases``/``lsq.fita`` stack.
3. Sobol main, total, and joint indices are computed analytically from
   the PCE coefficients via ``PCRV.computeSens``/``computeTotSens``
   (Sudret 2008).
4. Relative training errors (and test errors if ``X_test``/``Y_test``
   are in the cache) are computed per output and displayed in the Rich
   report.
5. A ``QuantifyResult`` is exported to ``./uq_results/`` as a dashboard
   schema, per-strategy Sobol ``.npy`` files, and the two PCE surrogates
   (population + growth-stratified).

Interactive clients
-------------------

All four clients wrap the same ``uq.workflow`` functions:

.. code-block:: bash

   uv run uq tui         # Textual TUI
   uv run uq gui         # marimo browser GUI
   uv run uq dashboard   # tkinter DAW-style dashboard

See :doc:`cli_reference` for a per-flag breakdown of every command and
:doc:`tutorial_workflow` for the underlying math.
