- [x] 1. Please verify robustly that the sampling->uq workflow exposed by uq.cli is essentially simply a vEcoli-specific (../vEcoli & https://covertlab.github.io/vEcoli/index.html) implementation
    of the workflow/logic presented here: https://sandialabs.github.io/pytuq/apps/uqpc.html . If there are any missing elements, please implement and explain them.
    > Verified: `uq/workflow.py` maps 1:1 onto the UQPC 5-step pipeline (`_setup_input_pc`, `_generate_training_samples`,
    > `_evaluate_model_online`, `_fit_surrogate`, `_compute_relative_errors`, `_compute_sobol`).
    > Missing pieces implemented: (a) UQPC `--ntst` held-out validation — `uq sample --n-test N` draws extra
    > germ samples, runs them through the same vEcoli workflow, splits and caches them
    > (`X_test.npy`/`Y_test.npy`/`timeseries_test/`); `quantify` passes them into `run_uqpc` and reports test
    > relative errors. (b) UQPC `--tol` surfaced as `uq quantify --tol` for BCS sparsity.
    > (c) UQPC step 5 relative errors are now rendered in the Rich report (train + test).

- [x] 2. Explain in depth how the sample command of uq.cli optimially uses the vEcoli variants API (https://covertlab.github.io/vEcoli/workflows.html#variants) in a file called ./SAMPLING.md. Make sure to include specific references in the
    code that verifies this utilization. If it doesnt completely, adjust accordingly. We want as little hand rolling as possible in terms of the pipeline: vEcoli and PyTUQ should essentially be all we need for that. What this repo's novelty
    should be other than connecting those two tools, is very innovative, creative, easy to use, robust, novel end-user entrypoints (CLI, TUI, GUI, etc).
    > See `./SAMPLING.md` — walks through the full delegation chain with file:line references and an ownership
    > table showing PyTUQ (germ sampling), vEcoli (`sim_data_setattr` + `parse_variants` +
    > `runscripts/workflow.py`), and the thin adapter code in this repo.

- [x] 3. Please update the docs/ to reflect the current repo state once 2 and 3 are finished. NOTE: keep in mind that these docs should be end-user-facing, so we need a robust CLI reference, as well as a detailed tutorial which explains
    why/how everything is done the way it is, and how that is mathematically accurate/conceptually accurate
    > Rewrote `docs/index.rst` and `docs/getting_started.rst` for the two-stage CLI workflow; added
    > `docs/cli_reference.rst` (every flag on `sample`/`quantify`/`dashboard`/`tui`/`gui`) and
    > `docs/tutorial_workflow.rst` (6-step UQPC walkthrough with the Legendre/PCE math and the four
    > RFC006 aggregation strategies).

- [ ] 4. I noticed that the app.uq_daw has a nice feature that tracks a dot moving through the spectrogram heatmap as one changes the PCE response curves. Can we have that in app.uq_daw_simple? I want it to be just like that, even with a different pipeline
