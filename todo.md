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

- [x] 4. I noticed that the app.uq_daw has a nice feature that tracks a dot moving through the spectrogram heatmap as one changes the PCE response curves. Can we have that in app.uq_daw_simple? I want it to be just like that, even with a different pipeline
    > Added to `app/uq_daw_simple.py::HeatmapCanvas`:
    > (a) **Tracking dots** — one white dot per parameter row, positioned at the slider's
    > normalized [0,1] position. Dots move left/right as the user drags sliders, showing
    > WHERE in the sensitivity landscape the current setting is. Selected param gets a
    > larger dot (r=6 vs r=4) with colored outline.
    > (b) **Peak stage indicator** — the max-prediction stage gets a highlighted dot with
    > a `θN=value` label, matching `uq_daw.py`'s behavior.
    > (c) **Y-axis ticks + grid** on the prediction curve for readability.
    > 4 headless tests in `tests/test_heatmap_tracking_dot.py`, all passing.

- [x] 5a. show me the full config json used that is passed to vEcoli when uq sample is run.
    > Added `uq show-config` CLI command — generates the full vEcoli workflow config JSON
    > (sim_data_path, emitter, variants section with per-sample sim_data_setattr mutations,
    > generations, n_init_sims, max_duration) without running anything. Syntax-highlighted
    > with line numbers via Rich. Optionally writes to file with `--output-file`.
    > Also: `uq sample` now prints the config path (`Config JSON: <path>`) during step 3.

- [x] 6. **Observable selector in DAW** — the current pipeline exports only one variance-weighted
    aggregate PCE per strategy (not per-output coefficients). `population_surrogate/coefficients.npy`
    is shape `(28,)` — a single scalar output. Adding per-observable switching (e.g. "show me the
    PCE for gene X" when `--observables transcriptome` was used) requires changing
    `QuantifyResult.export()` to save per-output coefficient arrays from `_fit_surrogate`'s
    `linregs` list. The per-output coefficients already exist in memory during `quantify` —
    they just aren't exported individually.
    > Fixed: `QuantifyResult.export()` now writes `coefficients_per_output.npy` — shape `(n_obs, n_basis)` —
    > alongside the existing `coefficients.npy` (backward compatible). DAW adds an OBSERVABLE dropdown
    > that switches the response curves, Ŷ readout, and local sensitivity to evaluate the selected
    > observable's own PCE instead of the variance-weighted aggregate.
    > 7 tests in `tests/test_daw_observable_selector.py`, all passing.

- [x] 7. **Per-stage surrogate evaluation in DAW** — same root cause: `growth_stratified_surrogate/
    coefficients.npy` is shape `(28,)` (one output column), not `(28, n_stages*n_obs)`. The
    per-stage prediction curve currently uses a first-order linear approximation
    `pop_y + Σ(∂Ŷ/∂x_i × deviation × S_Ti^(k))` instead of the actual per-stage PCE.
    Fix: export the full per-output coefficient matrix from the combined strategy-4 fit,
    then evaluate `legendre_eval` per-stage in the DAW.
    > Fixed: `QuantifyResult.export()` now writes `coefficients_per_output.npy` in
    > `growth_stratified_surrogate/` — shape `(n_stages * n_obs, n_basis)`. The DAW loads it and
    > evaluates the exact per-stage PCE for the selected observable, replacing the linear approximation.
    > Falls back to the old linear approximation if the file is missing (backward compatible).
    > Verified: per-stage predictions are finite, varying across stages, and the observable selector
    > picks the correct column index `stage * n_obs + obs_idx`.

- [x] 8. yes, but i also want (maybe togglable view, but the following as default view) in that area the same type of line-graph curve which
  responds to changes in the parameter knobs, but is the actual predicted output for the selected observables (shows all of them together if
  none are explictly selected)...in other words, i want users to be able to see the direct effects on output observables for any given
  coordinates of param settings...I want users to be able to use the uq_daw_simple (which leverages this repo) to be able to experiment with
  differnet parameter vals and be able to directly see its effect on the familiar timeseries output, in other words and for example, say a
  user has an imagined desired ecoli state (healthy, or even something like with a defect), but does not know which parameter config vals
  would produce this outcome. Thus, since the user knows exactly what their desired state looks like in terms of the timeseries output (they
  hypothesized the desired outcome's timeseries in the familiar format and plotted it as such. Thus, in this example, the user can adjust the
  parameter knobs and in tern their values to use in vEcoli numerically until the predicted timeseries matches the desired one from the
  user. Essenially, users should be able to "tune" parameter vals (that is, vals to mutate in SimulationDataEcoli for a given simulation) to
  a desired observables outcome (leveraging the predictions from the surrogate). NOTE: do not introduce any functionality that would not be "RFC"-approved!
    > Added `PredictedProfileCanvas` to `app/uq_daw_simple.py`:
    > - New panel below the spectrogram showing **per-observable predicted profiles across the cell cycle**
    >   (θ = growth progress, X-axis = θ-bins, Y-axis = predicted value from per-stage PCE)
    > - When an observable is selected: bold line with actual values + units on Y-axis, baseline reference
    >   line, other observables shown as dimmed shape-guides rescaled to the same Y range
    > - When "(aggregate)" is selected: all observables shown as **% deviation from baseline** on a common axis
    > - Updates live on every slider drag — biologists can tune parameters and watch the predicted
    >   observable profile change in real time until it matches their desired cell state
    > - Uses the exact per-stage PCE (`growth_stratified_surrogate/coefficients_per_output.npy`)
    >   not the linear approximation — same PCE that `uq quantify` fits via PyTUQ
    > - Legend shows color-coded observable names; X-axis labeled with θ% ranges
    > - 5 tests in `tests/test_daw_predicted_profile.py`, all passing

- [ ] 9. **Deploy public docs via GitHub Pages (free, private source repo)**
    Create a separate public repo for rendered HTML only:
    ```
    # 1. Create empty public repo "uqEcoli-docs" on GitHub (no README)

    # 2. Build + deploy:
    cd docs && uv run sphinx-build -b html . _build/html && cd ..
    cd docs/_build/html
    git init
    git remote add origin https://github.com/<you>/uqEcoli-docs.git
    touch .nojekyll
    git add -A
    git commit -m "docs build"
    git push -f origin main

    # 3. Go to uqEcoli-docs repo → Settings → Pages → Source →
    #    "Deploy from branch" → main → Save
    ```
    Docs will be at `https://<you>.github.io/uqEcoli-docs/`.
    Source code stays private in `uqEcoli`. To update: re-run step 2.
    Makefile target already added: `make docs` builds the HTML.

- [x] 10. For sampling, we are currently running simulations with ../vEcoli. Let's now add the ability to pass a flag to sample, that has the actual compute come from
    the SMS-API (../sms-api). Default base URL: http://localhost:8080 (stanford-test namespace via port-forward).
    > Implemented in `uq/remote.py` and `uq/cli.py`:
    > - `SmsApiClient` — httpx wrapper for SMS-API REST endpoints with ALB 502/504 retry logic
    > - `uq sample --api-url http://localhost:8080 --simulator-id 11` — remote execution mode
    >   Steps 1-2 (PCRV.sampleGerm) remain local; Step 3 submits to SMS-API; Step 4 downloads
    >   cd1 analysis TSVs instead of collecting raw Parquet
    > - `uq fetch <sim_id>` — download + parse cd1 outputs from a completed simulation
    > - cd1 module → UQ observable mapping:
    >   cd1_transcriptomics → transcriptome (4345 genes)
    >   cd1_proteomics → proteome (4309 monomers)
    >   cd1_fluxomics → fluxome (2820 reactions)
    >   cd1_metabolomics → exchange_fluxes (165 compounds)
    >   cd1_higher_order_properties → higher_order (5 properties)
    > - Verified: `uq fetch 48` against stanford-test → 11,644 observables
    > - 14 tests in `tests/test_remote_client.py`, all passing

- [x] 11. Stakeholders specifically want a human-readable/sleek/modern/clear/conscice yet informative html report of what exists as the main output artifact (a la /Users/alexanderpatrie/sms/uqEcoli/uq_results_e2e_verify/uq_results.json). Please make this.
    > Implemented `uq/report.py` + `uq report` CLI command. Self-contained HTML (zero deps):
    > - All 4 RFC006 strategies visualized with SVG bar charts, heatmaps, ranking tables
    > - Cross-strategy comparison with automatic insight callouts
    > - Interactive PCE Explorer: client-side Legendre evaluation, multi-select observable
    >   toggle pills, real-time cell-cycle profile chart from embedded surrogate coefficients
    > - Experimental Design section: parameter specs table with SimData dot-paths,
    >   perturbation bounds, and biological role descriptions
    > - Simulation Design summary, Provenance (git SHAs, packages, CLI command)
    > - `sobol_first_order` now exported for all strategies (was S1 only)
    > - `parameter_specs` embedded in manifest.json for biological context

- [ ] 12. Let's find unused content in this repo and together work to do some housekeeping in the following way: for each item found, 1. you tell me the filepath 2. you summarize its content and try to decipher and explain why it exists in the first place 3. i say either "keep" (do nothing: we want to keep this content) or "toss" (delete this content: we want to remove it.).NOTE: DO NOT INCLUDE output artifacts from any cli (`uq` cli) commands/calls. 

- [ ] 13. PR number 7 AND ALL FUTURE WORK/PRS MUST BE ON THE vivarium-collective origin (github.com/vivarium-collective/uqEcoli). It would be nice if any work done to this remote (vivarium-collective) can automatically be streamed/added to the AlexPatrie remote (github.com/AlexPatrie/uqEcoli). I hope this doesnt screw anything up.

- [ ] 14. Create Release for version `v0.0.1` in github.com/vivarium-collective/uqEcoli AND github.com/AlexPatrie/uqEcoli.Ensure that ALL documentation/human-readable references are up to date
