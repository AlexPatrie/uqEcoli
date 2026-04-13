CLI Reference
=============

``uq`` is a `Typer <https://typer.tiangolo.com/>`_ application.  Every
command is also available programmatically through :py:mod:`uq.workflow`.

Top-level
---------

.. code-block:: text

   uv run uq [OPTIONS] COMMAND [ARGS]...

   Commands:
     help       Show help for a specific subcommand, or the main CLI.
     sample     UQPC Steps 1-3: sample via PCRV.sampleGerm(), run vEcoli workflow.py.
     quantify   UQPC Steps 4-5: fit PCE surrogates, compute Sobol (all 4 strategies).
     dashboard  Launch the uq interactive dashboard.
     tui        Launch the UQPC interactive terminal UI (Textual).
     gui        Launch the UQPC interactive GUI (marimo).

``uq sample``
-------------

UQPC steps 1-3.  Sets up the input PC, draws germ samples, and evaluates
vEcoli through ``runscripts/workflow.py``.

.. code-block:: text

   uv run uq sample SIM_DATA_PATH [OPTIONS]

Arguments:

* ``SIM_DATA_PATH`` — absolute or relative path to a pre-computed
  ``simData.cPickle`` produced by vEcoli's Parca.

Options:

``--cache-dir PATH``
    Directory to write cached ``X.npy``, ``Y.npy``, ``germ_train.npy``,
    ``timeseries/``, and (if ``--n-test > 0``) ``X_test.npy`` /
    ``Y_test.npy`` / ``timeseries_test/``.  Default ``./uq_cache``.

``--n-samples INTEGER``
    Number of training samples drawn from the germ measure.  Equivalent
    to the UQPC ``--nqd`` flag in random-sampling mode.  Default ``20``.

``--n-test INTEGER``
    Number of **held-out validation samples** (PyTUQ UQPC ``--ntst``).
    Concatenated onto the training samples and evaluated in the same
    vEcoli workflow; split back out before caching.  Consumed by
    ``quantify`` to compute test relative errors.  Default ``0``.

``--seed INTEGER``
    RNG seed for ``PCRV.sampleGerm``.  Default ``42``.

``--generations INTEGER``
    Generations per variant/lineage-seed pair.  Set ``>= 2`` to enable
    RFC006 aggregation strategy 2 (by generation).  Default ``1``.

``--n-init-sims INTEGER``
    Initial lineage seeds per variant.  Set ``>= 2`` to enable RFC006
    aggregation strategy 3 (by lineage seed).  Default ``1``.

``--max-duration FLOAT``
    Per-simulation wall-clock limit, seconds.  Default ``10800``.

``--params-file PATH``
    Optional JSON file of ``SimDataParameter`` dicts overriding the
    default six sim_data parameters.  See
    ``examples/uq_artifacts/params/params_demo.json``.

``--observables TEXT`` (repeatable)
    Observable presets to extract from vEcoli Parquet output, matching
    the cd1 analysis modules used for Vegas/Bermuda CD1 deliverables.
    Pass multiple times to compose presets.  Default ``mass``.

    ================  ================================================  ========
    Preset            cd1 module equivalent                              Features
    ================  ================================================  ========
    ``mass``          cd1_higher_order_properties (raw scalars)          5
    ``higher_order``  cd1_higher_order_properties (derived metrics)      6
    ``exchange_fluxes`` cd1_exchange_fluxes                              ~87
    ``transcriptome`` cd1_transcriptomics                                ~4,300
    ``proteome``      cd1_proteomics                                     ~4,300
    ``fluxome``       cd1_fluxomics (dry-mass normalized)                ~2,800
    ================  ================================================  ========

    Example — all CD1 analyses at once::

        uv run uq sample simData.cPickle \
            --observables higher_order \
            --observables exchange_fluxes \
            --observables transcriptome \
            --observables proteome \
            --observables fluxome

``--generation-lower-bound INTEGER``
    Skip generations below this value when aggregating observables.
    Mirrors the cd1 ``generation_lower_bound`` parameter — filters
    early transient dynamics so the sensitivity analysis focuses on
    steady-state growth.  Default ``0`` (keep all).

``uq quantify``
---------------

UQPC steps 4-5.  Fits PCE surrogates per aggregation strategy, computes
Sobol indices, reports relative errors.

.. code-block:: text

   uv run uq quantify SIM_DATA_PATH [OPTIONS]

Options:

``--cache-dir PATH``
    Path to the cache produced by ``uq sample``.  Default ``./uq_cache``.

``--export-path PATH``
    Where to write ``uq_results.json``, per-strategy Sobol ``.npy``
    files, and the two exported PCE surrogates.  Default ``./uq_results``.

``--n-bins INTEGER``
    Number of growth-stratified bins (strategy 4).  Bins are uniform in
    ``θ = [log(mass) − log(mass_birth)] / [log(mass_div) − log(mass_birth)]``.
    Default ``10``.

``--polynomial-order INTEGER``
    Output PCE polynomial degree (UQPC ``--outord``).  Default ``2``.

``--regression [lsq|bcs|anl]``
    PyTUQ regression backend:

    * ``lsq`` — least squares (default)
    * ``bcs`` — Bayesian Compressed Sensing, sparse
    * ``anl`` — analytical Bayesian projection

``--tol FLOAT``
    BCS sparsity tolerance (UQPC ``--tol``).  Only used when
    ``--regression=bcs``.  Default ``1e-3``.

Report
^^^^^^

The Rich report now includes (from top to bottom):

1. **Surrogate quality panel** — per-output training relative error and,
   if the cache contains ``X_test``/``Y_test``, test relative error.
2. **Strategy 1** — population-averaged Sobol indices (bulk).
3. **Strategy 2** — per-generation Sobol (requires ``--generations >= 2``
   at sample time).
4. **Strategy 3** — per-lineage-seed Sobol (requires ``--n-init-sims >= 2``).
5. **Strategy 4** — growth-stratified Sobol across ``n-bins`` cell-cycle
   stages.

``uq show-config``
------------------

Generates the full vEcoli workflow config JSON that ``uq sample`` would
pass to ``runscripts/workflow.py``, **without running anything**.  Useful
for stakeholder review, debugging, or manual execution.

.. code-block:: text

   uv run uq show-config SIM_DATA_PATH [OPTIONS]

Options:

``--n-samples INTEGER``
    Number of variants to include in the config.  Default ``5``.

``--output-file PATH``
    Write JSON to a file instead of printing to stdout.

All other flags (``--seed``, ``--generations``, ``--n-init-sims``,
``--max-duration``, ``--params-file``) mirror ``uq sample``.

``uq dashboard``
----------------

.. code-block:: text

   uv run uq dashboard [--results-path PATH] [--run-mode tk|mo]

``--run-mode tk``
    Tkinter DAW-style dashboard with draggable parameter markers
    (:py:mod:`app.uq_daw_simple`).  Default.

``--run-mode mo``
    Launches ``app/dashboard_simple.py`` as a marimo notebook via
    ``marimo edit``.

``uq tui``
----------

Textual terminal UI — the same workflow with live progress, tabbed
strategy tables, and interactive parameter selection.

.. code-block:: text

   uv run uq tui

``uq gui``
----------

Launches ``app/gui.py`` as a marimo application:

.. code-block:: text

   uv run uq gui

Programmatic API
----------------

Everything in the CLI delegates to two functions:

.. code-block:: python

   from uq.workflow import sample, quantify

   cache = sample(
       sim_data_path="/path/to/simData.cPickle",
       cache_dir="./uq_cache",
       n_samples=200,
       generations=2,
   )
   result = quantify(
       cache_dir="./uq_cache",
       sim_data_path="/path/to/simData.cPickle",
       polynomial_order=3,
       regression="lsq",
       export_path="./uq_results",
   )

See :py:mod:`uq.workflow` for the full signatures.
