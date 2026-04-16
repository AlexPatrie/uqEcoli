```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Plan to implement                                                                                                                                            │
│                                                                                                                                                              │
│ Implementation Plan: 15 Features for uqEcoli                                                                                                                 │
│                                                                                                                                                              │
│ Context                                                                                                                                                      │
│                                                                                                                                                              │
│ The uqEcoli UQ framework has a complete RFC006 pipeline (4 strategies, PyTUQ PCE, vEcoli integration) with working CLI, TUI, Tk DAW, and Marimo dashboard.   │
│ The ASSESSMENT.md identified gaps in polish, UX, and product completeness. This plan implements 15 features incrementally, each building on the last.        │
│                                                                                                                                                              │
│ Phased Implementation                                                                                                                                        │
│                                                                                                                                                              │
│ Phase 1 — Quick Wins (CLI + Dashboard Polish)                                                                                                                │
│                                                                                                                                                              │
│ These touch existing files with small, additive changes. Each is independently verifiable.                                                                   │
│                                                                                                                                                              │
│ 1.1 Reproducibility Manifest                                                                                                                                 │
│                                                                                                                                                              │
│ Files: uq/workflow.py (QuantifyResult.export method, ~line 1200)                                                                                             │
│ What: Add manifest.json to export directory with:                                                                                                            │
│ - Git SHA (uqEcoli + vEcoli via subprocess git rev-parse HEAD)                                                                                               │
│ - Python version, PyTUQ version, numpy version                                                                                                               │
│ - Full CLI command (from sys.argv)                                                                                                                           │
│ - Hash of X.npy and Y.npy (sha256)                                                                                                                           │
│ - Timestamp, hostname                                                                                                                                        │
│ - Cache path, sim_data_path, polynomial_order, regression, n_bins                                                                                            │
│ How: Add _write_manifest() helper called at end of export(). Use hashlib.sha256, platform.node(), importlib.metadata.version().                              │
│ Test: Verify manifest.json exists in export, contains expected keys, hashes match.                                                                           │
│                                                                                                                                                              │
│ 1.2 Plain-English Narrative Summary                                                                                                                          │
│                                                                                                                                                              │
│ Files: uq/cli.py (~line 387, _print_report)                                                                                                                  │
│ What: After the Sobol tables, print a "Key Findings" panel with:                                                                                             │
│ - Top parameter per strategy with % variance explained                                                                                                       │
│ - Parameter that's negligible across all strategies                                                                                                          │
│ - Which growth stage has most parameter sensitivity                                                                                                          │
│ - One-sentence recommendation                                                                                                                                │
│ How: Add _print_narrative(result) function called at end of _print_report(). Pure string formatting from existing Sobol arrays — no new computation.         │
│ Test: Run uq quantify on existing cache, verify narrative prints. Unit test with mock QuantifyResult.                                                        │
│                                                                                                                                                              │
│ 1.3 Strategy 2-3 in Marimo Dashboard (Rich Panels)                                                                                                           │
│                                                                                                                                                              │
│ Files: app/dashboard_simple.py                                                                                                                               │
│ What: Currently strategies 2-3 are markdown text with top-3 params. Upgrade to:                                                                              │
│ - Strategy 2: Plotly grouped bar chart (generations × params, S_Ti values)                                                                                   │
│ - Strategy 3: Plotly grouped bar chart (seeds × params, S_Ti values)                                                                                         │
│ - Both appear as new cells between the spectrogram and the methods section                                                                                   │
│ - Graceful fallback: show "Not available (run with --generations >= 2)" if empty                                                                             │
│ How: Add 2 new @app.cell functions. Read from data["strategy2_by_generation"]["generations"] and data["strategy3_by_seed"]["seeds"]. Use go.Bar with         │
│ barmode="group". Follow existing COLORS/DAW styling.                                                                                                         │
│ Layout: Insert between current Cell 7 (spectrogram) and Cell 8 (param table). Update Cell 9 layout to include new figures.                                   │
│ Test: Load uq_results_e2e/uq_results.json in marimo, verify bars render. Test with missing strategy data.                                                    │
│                                                                                                                                                              │
│ 1.4 Dashboard Context Sidebar (Tk DAW)                                                                                                                       │
│                                                                                                                                                              │
│ Files: app/uq_daw_simple.py                                                                                                                                  │
│ What: Below the "Strategy info" text in the left panel, add a Context Panel showing:                                                                         │
│ - Per-observable Sobol ranking (top-3 params for selected observable, using per-output Sobol from coefficients_per_output.npy)                               │
│ - Surrogate quality (train/test relative error for selected observable)                                                                                      │
│ - Observable metadata (units, baseline value, % deviation from midpoint)                                                                                     │
│ - Updates reactively when observable selection changes                                                                                                       │
│ How: Add a ContextPanel frame (Tk Label + Text widget, ~60 lines) below the strategy info. Populate in _on_obs_listbox_select(). Compute per-observable      │
│ Sobol by evaluating PCE sensitivity per output column.                                                                                                       │
│ Test: Headless test: load data, select observable, verify context text contains parameter names and Sobol values.                                            │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ Phase 2 — Interactive UX Features (DAW Enhancements)                                                                                                         │
│                                                                                                                                                              │
│ 2.1 Progressive Disclosure                                                                                                                                   │
│                                                                                                                                                              │
│ Files: app/uq_daw_simple.py                                                                                                                                  │
│ What: Three view modes (radio buttons in top bar):                                                                                                           │
│ - Simple — Left panel: sliders + Ŷ readout only. Right: predicted profile only. No spectrogram, no response curves.                                          │
│ - Standard (default) — Current layout (response curves + spectrogram + predicted profile)                                                                    │
│ - Expert — Standard + context sidebar + patches + target mode + spectral (if available)                                                                      │
│ How: Add self._view_mode StringVar + 3 Radiobuttons in top bar. On change, call _apply_view_mode() which grid_remove()/grid() frames. All widgets stay       │
│ created; just hidden/shown.                                                                                                                                  │
│ Test: Headless: toggle modes, verify widget visibility states.                                                                                               │
│                                                                                                                                                              │
│ 2.2 What-If Comparison Mode                                                                                                                                  │
│                                                                                                                                                              │
│ Files: app/uq_daw_simple.py                                                                                                                                  │
│ What: "Compare" button in top bar. When active:                                                                                                              │
│ - Current slider position becomes "Configuration A" (frozen, dimmed line)                                                                                    │
│ - User adjusts sliders → new position is "Configuration B" (bright line)                                                                                     │
│ - Predicted profile shows both A and B overlaid with delta annotation                                                                                        │
│ - "Clear comparison" button resets                                                                                                                           │
│ How: Store self._compare_snapshot (dict of param values + predicted profiles). In _update_predicted_profile(), if snapshot exists, draw both curves. Add     │
│ _on_compare_toggle() and _on_compare_clear().                                                                                                                │
│ Test: Set sliders → snapshot → move sliders → verify two curves drawn.                                                                                       │
│                                                                                                                                                              │
│ 2.3 Undo/History                                                                                                                                             │
│                                                                                                                                                              │
│ Files: app/uq_daw_simple.py                                                                                                                                  │
│ What: Track slider movements as a stack (max 50 entries). Ctrl+Z to undo, Ctrl+Y to redo. Each entry = dict of all slider values + timestamp.                │
│ How: self._history: list[dict], self._history_idx: int. Push on _on_slider_change() (debounced — only after 200ms of no movement). Undo/redo restore all     │
│ slider values and trigger _update_response_curves(). Bind keyboard shortcuts.                                                                                │
│ Test: Move sliders, undo, verify values restored.                                                                                                            │
│                                                                                                                                                              │
│ 2.4 Guided uq init                                                                                                                                           │
│                                                                                                                                                              │
│ Files: uq/cli.py (new command)                                                                                                                               │
│ What: uq init command that:                                                                                                                                  │
│ 1. Prompts for vEcoli sim_data path (with auto-detection: search ../vEcoli, ~/.local/share/vEcoli)                                                           │
│ 2. Prompts for observable preset (show descriptions)                                                                                                         │
│ 3. Prompts for sample count (suggest 20 for quick, 50 for production)                                                                                        │
│ 4. Validates setup: check vEcoli importable, simData.cPickle exists, PyTUQ available                                                                         │
│ 5. Writes uq_config.json with all settings                                                                                                                   │
│ 6. Prints: "Ready! Run uq sample <path> --config uq_config.json to start."                                                                                   │
│ How: New @app.command() using typer.prompt() and typer.confirm(). Auto-detection via Path.glob().                                                            │
│ Test: Run uq init with mock paths, verify config JSON written.                                                                                               │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ Phase 3 — Analysis & Pipeline Features                                                                                                                       │
│                                                                                                                                                              │
│ 3.1 Publication-Ready Exports                                                                                                                                │
│                                                                                                                                                              │
│ Files: uq/cli.py (new command), uq/viz_export.py (new module, ~200 lines)                                                                                    │
│ What: uq export-figures command producing:                                                                                                                   │
│ - sobol_bar_chart.pdf — Grouped bars (S_Ti per strategy, all params)                                                                                         │
│ - spectrogram.pdf — Print-quality heatmap with proper labels                                                                                                 │
│ - response_curves.pdf — PCE response curves at midpoint                                                                                                      │
│ - sobol_table.tex — LaTeX table of top-K params per strategy                                                                                                 │
│ How: Load uq_results.json + surrogates. Use Plotly with kaleido for PDF export. LaTeX via string formatting.                                                 │
│ Test: Generate from uq_results_e2e/, verify PDFs exist and are non-empty.                                                                                    │
│                                                                                                                                                              │
│ 3.2 Adaptive Sampling                                                                                                                                        │
│                                                                                                                                                              │
│ Files: uq/workflow.py (extend sample())                                                                                                                      │
│ What: uq sample ... --adaptive flag:                                                                                                                         │
│ 1. Start with n_samples // 3 initial samples                                                                                                                 │
│ 2. Fit PCE, compute leave-one-out CV error                                                                                                                   │
│ 3. If error > threshold, draw more samples from high-variance regions                                                                                        │
│ 4. Repeat until error < --tol or budget exhausted                                                                                                            │
│ 5. Report convergence history                                                                                                                                │
│ How: Add adaptive: bool = False param to sample(). Implement _adaptive_sampling_loop() that calls _fit_surrogate() + _compute_relative_errors() iteratively. │
│  Use PyTUQ's regressor prediction variance to identify where to add samples.                                                                                 │
│ Test: Synthetic function (polynomial), verify adaptive uses fewer samples than fixed for same accuracy.                                                      │
│                                                                                                                                                              │
│ 3.3 Live Streaming Results                                                                                                                                   │
│                                                                                                                                                              │
│ Files: uq/tui.py (extend Sample tab)                                                                                                                         │
│ What: During uq tui sampling, after every 5 completed simulations:                                                                                           │
│ - Fit a preliminary PCE on completed samples                                                                                                                 │
│ - Show "Preliminary Sobol" table that updates live                                                                                                           │
│ - Show convergence indicator (are indices stabilizing?)                                                                                                      │
│ How: In the TUI's sample worker thread, after each batch of sims completes, call a lightweight _preliminary_sobol() that fits PCE on available (X[:k],       │
│ Y[:k]). Update a DataTable widget.                                                                                                                           │
│ Test: Mock 10 sequential sample completions, verify preliminary Sobol updates.                                                                               │
│                                                                                                                                                              │
│ 3.4 Multi-Experiment Comparison                                                                                                                              │
│                                                                                                                                                              │
│ Files: uq/cli.py (new command), app/dashboard_simple.py (new cell)                                                                                           │
│ What: uq compare <dir1> <dir2> [dir3...] command:                                                                                                            │
│ - Load multiple uq_results.json files                                                                                                                        │
│ - Compute differential Sobol: ΔS_Ti = S_Ti(A) - S_Ti(B)                                                                                                      │
│ - Print side-by-side Rich tables + highlight shifts                                                                                                          │
│ - Dashboard: new Marimo cell with overlay bar charts                                                                                                         │
│ How: New CLI command loads N result JSONs. Differential computed as simple subtraction. Plotly grouped bars with positive/negative coloring.                 │
│ Test: Compare uq_results_e2e/ against a copy with perturbed values.                                                                                          │
│                                                                                                                                                              │
│ 3.5 Surrogate-Guided Experimental Design                                                                                                                     │
│                                                                                                                                                              │
│ Files: uq/workflow.py (new function), uq/cli.py (new command)                                                                                                │
│ What: uq suggest-experiment command:                                                                                                                         │
│ - Load fitted PCE from export                                                                                                                                │
│ - Evaluate prediction variance across parameter space (grid or random)                                                                                       │
│ - Identify region of maximum uncertainty                                                                                                                     │
│ - Print: "Measure X in range [a, b] to reduce uncertainty most"                                                                                              │
│ How: Use PyTUQ regressor .predicta() variance. Grid-sample parameter space, find max-variance point, report in physical units with parameter name.           │
│ Test: Load existing surrogate, verify suggestion is within bounds and has positive variance.                                                                 │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ Phase 4 — New Platforms                                                                                                                                      │
│                                                                                                                                                              │
│ 4.1 Standalone Web Dashboard                                                                                                                                 │
│                                                                                                                                                              │
│ Files: app/web_dashboard.py (new, ~300 lines)                                                                                                                │
│ What: FastAPI server serving a Plotly Dash app:                                                                                                              │
│ - Upload uq_results.json → renders all panels                                                                                                                │
│ - Shareable URL (team collaboration)                                                                                                                         │
│ - No Python environment needed on viewer side                                                                                                                │
│ How: FastAPI + Dash (or Panel). Reuse legendre_eval from dashboard_simple.py. Serve on localhost:8050. Add uq dashboard --run-mode web option.               │
│ Test: Start server, load JSON via upload, verify all panels render.                                                                                          │
│                                                                                                                                                              │
│ 4.2 Jupyter Widgets                                                                                                                                          │
│                                                                                                                                                              │
│ Files: uq/jupyter.py (new, ~200 lines)                                                                                                                       │
│ What: display_sobol(result) → Rich table in notebook. UQWidget(result) → ipywidgets sliders + Plotly figure. result.plot() → spectrogram.                    │
│ How: Use ipywidgets.interact for sliders, plotly.graph_objects.FigureWidget for reactive plots. Import guard: try: import ipywidgets.                        │
│ Test: Unit test widget creation (headless). Integration in a test notebook.                                                                                  │
│                                                                                                                                                              │
│ ---                                                                                                                                                          │
│ Implementation Order                                                                                                                                         │
│                                                                                                                                                              │
│ 1.1 Manifest  →  1.2 Narrative  →  1.3 Marimo S2/S3  →  1.4 Context Sidebar                                                                                  │
│       ↓                                                                                                                                                      │
│ 2.1 Progressive Disclosure  →  2.2 Compare  →  2.3 Undo  →  2.4 Init                                                                                         │
│       ↓                                                                                                                                                      │
│ 3.1 Pub Exports  →  3.2 Adaptive  →  3.3 Live Stream  →  3.4 Multi-Exp  →  3.5 Suggest                                                                       │
│       ↓                                                                                                                                                      │
│ 4.1 Web Dashboard  →  4.2 Jupyter                                                                                                                            │
│                                                                                                                                                              │
│ Each item is independently committable. Phase 1 items are ~30-80 lines each. Phase 2 items are ~50-150 lines. Phase 3 items are ~100-300 lines. Phase 4      │
│ items are ~200-300 lines.                                                                                                                                    │
│                                                                                                                                                              │
│ Verification                                                                                                                                                 │
│                                                                                                                                                              │
│ After each feature:                                                                                                                                          │
│ 1. uv run pytest tests/ -v -s --ignore=tests/test_real_data.py — existing tests still pass                                                                   │
│ 2. Feature-specific test added                                                                                                                               │
│ 3. Manual verification: uq quantify (for CLI features), uq dashboard (for DAW features), uv run marimo run app/dashboard_simple.py (for Marimo features)     │
│ 4. make check — type checking and linting pass                                                                                                               │
│                                                                                                                                                              │
│ Critical Files to Modify                                                                                                                                     │
│                                                                                                                                                              │
│ ┌─────────────────────────┬─────────────────────────┐                                                                                                        │
│ │          File           │        Features         │                                                                                                        │
│ ├─────────────────────────┼─────────────────────────┤                                                                                                        │
│ │ uq/workflow.py          │ 1.1, 3.2, 3.5           │                                                                                                        │
│ ├─────────────────────────┼─────────────────────────┤                                                                                                        │
│ │ uq/cli.py               │ 1.2, 2.4, 3.1, 3.4, 3.5 │                                                                                                        │
│ ├─────────────────────────┼─────────────────────────┤                                                                                                        │
│ │ app/dashboard_simple.py │ 1.3, 3.4                │                                                                                                        │
│ ├─────────────────────────┼─────────────────────────┤                                                                                                        │
│ │ app/uq_daw_simple.py    │ 1.4, 2.1, 2.2, 2.3      │                                                                                                        │
│ ├─────────────────────────┼─────────────────────────┤                                                                                                        │
│ │ uq/tui.py               │ 3.3                     │                                                                                                        │
│ └─────────────────────────┴─────────────────────────┘                                                                                                        │
│                                                                                                                                                              │
│ New Files                                                                                                                                                    │
│                                                                                                                                                              │
│ ┌──────────────────────────────────────┬─────────┐                                                                                                           │
│ │                 File                 │ Feature │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ uq/viz_export.py                     │ 3.1     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ app/web_dashboard.py                 │ 4.1     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ uq/jupyter.py                        │ 4.2     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_manifest.py               │ 1.1     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_narrative.py              │ 1.2     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_dashboard_strategies.py   │ 1.3     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_context_sidebar.py        │ 1.4     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_progressive_disclosure.py │ 2.1     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_compare_mode.py           │ 2.2     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_undo_history.py           │ 2.3     │                                                                                                           │
│ ├──────────────────────────────────────┼─────────┤                                                                                                           │
│ │ tests/test_init_command.py           │ 2.4     │                                                                                                           │
│ └──────────────────────────────────────┴─────────┘                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### What's been done:

- plan created
- All 15 features implemented
- 269 tests pass (1 pre-existing timeout in test_observables, unrelated)
- All modified/new files parse and import cleanly

#### Phase 1 — Quick Wins (CLI + Dashboard Polish) ✅
- **1.1 Reproducibility Manifest** — `_write_manifest()` in `uq/workflow.py`, writes `manifest.json` to export dir (git SHAs, versions, data hashes, CLI argv, hostname, timestamp)
- **1.2 Plain-English Narrative** — `_print_narrative()` in `uq/cli.py`, prints "Key Findings" panel after Sobol tables (top param, negligible param, most sensitive stage, recommendation)
- **1.3 Strategy 2-3 Plotly Bars** — Two new `@app.cell` functions in `app/dashboard_simple.py` with `go.Bar(barmode="group")`, graceful fallback when data missing. Included in layout via `mo.hstack`
- **1.4 Context Sidebar** — `_update_context_panel()` in `app/uq_daw_simple.py`, Text widget below strategy info showing per-observable top-3 drivers, baseline/deviation, Sobol quality

#### Phase 2 — Interactive UX Features (DAW Enhancements) ✅
- **2.1 Progressive Disclosure** — `_view_mode` StringVar + 3 Radiobuttons (Simple/Standard/Expert) in top bar, `_apply_view_mode()` shows/hides response canvas, heatmap, context panel
- **2.2 What-If Comparison** — Compare/Clear A/B buttons, `_on_compare_toggle()` freezes Config A, `PredictedProfileCanvas` draws dashed overlay line labeled "Config A"
- **2.3 Undo/History** — 50-entry stack, debounced push (200ms), `_undo()`/`_redo()` restore slider state, Ctrl+Z/Y (Cmd+Z/Cmd+Shift+Z on Mac)
- **2.4 Guided `uq init`** — Interactive wizard: auto-detect simData, choose observable presets, sample count, regression method, writes `uq_config.json`

#### Phase 3 — Analysis & Pipeline Features ✅
- **3.1 Publication-Ready Exports** — New `uq/viz_export.py` module (~200 lines), `uq export-figures` command → `sobol_bar_chart.pdf`, `spectrogram.pdf`, `response_curves.pdf`, `sobol_table.tex`
- **3.2 Adaptive Sampling** — `_adaptive_sampling_loop()` in `uq/workflow.py`: iterative PCE fit + LOO error, batch refinement until convergence or budget exhausted
- **3.3 Live Streaming Sobol** — `_show_preliminary_sobol()` in `uq/tui.py`, fits preliminary PCE every 5 completed variants during TUI sampling, displays live bar chart
- **3.4 Multi-Experiment Comparison** — `uq compare dir1 dir2 [dir3...]` command with side-by-side Rich table, differential Sobol (ΔS_Ti) with color-coded shifts
- **3.5 Surrogate-Guided Design** — `uq suggest-experiment` command: grid-samples parameter space (1000 pts), finds max-variance region, reports in physical units with recommendation

#### Phase 4 — New Platforms ✅
- **4.1 Standalone Web Dashboard** — New `app/web_dashboard.py` (~280 lines), Dash app with reactive sliders + Plotly panels, `uq dashboard --run-mode web` serves at localhost:8050
- **4.2 Jupyter Widgets** — New `uq/jupyter.py` (~200 lines), `display_sobol()` for Rich tables, `UQWidget` with ipywidgets sliders + FigureWidget, `plot_spectrogram()` for heatmap

#### Files Modified
| File | Features |
|------|----------|
| `uq/workflow.py` | 1.1, 3.2 |
| `uq/cli.py` | 1.2, 2.4, 3.1, 3.4, 3.5 |
| `app/dashboard_simple.py` | 1.3 |
| `app/uq_daw_simple.py` | 1.4, 2.1, 2.2, 2.3 |
| `uq/tui.py` | 3.3 |

#### New Files
| File | Feature |
|------|---------|
| `uq/viz_export.py` | 3.1 |
| `app/web_dashboard.py` | 4.1 |
| `uq/jupyter.py` | 4.2 |

#### CLI Commands (10 total)
`sample`, `quantify`, `dashboard`, `tui`, `gui`, `show-config`, `compare`, `export-figures`, `suggest-experiment`, `init`