# PR2: Cross-Condition Global Sensitivity Analysis via Multi-ParCa

## Scientific Motivation

RFC006 currently answers: *"Which kinetic parameters most influence whole-cell model outputs?"* — but only for a single growth condition (the ParCa dataset baked into `simData.cPickle`). This is a limitation because:

1. **Sensitivity rankings are condition-dependent.** A parameter like `secretion_penalty_coeff` might dominate in minimal glucose media (where overflow metabolism matters) but be irrelevant in rich media (where amino acids are imported). If you only test one condition, you mistake a context-specific finding for a universal one.

2. **Model-form uncertainty is invisible.** The current pipeline treats the ParCa output as ground truth. But ParCa fits kinetic parameters from RNA-seq data, and different RNA-seq datasets (different labs, strains, growth phases) produce different fits. The Sobol indices you compute are conditional on that single fit — they don't reflect uncertainty in the fit itself.

3. **Multi-condition experiments are standard in systems biology.** Wet-lab validation of model predictions typically involves testing across conditions. A UQ tool that only reports single-condition sensitivity is incomplete for experimental design.

### What multi-parca-aws enables

vEcoli's `multi-parca-aws` branch solves the infrastructure problem: multiple RNA-seq datasets can be processed through ParCa in a single Nextflow workflow, with offset-based variant indexing so all outputs land in one history tree. This means we can run the *same* UQ parameter sweep under *multiple growth conditions* without any manual orchestration.

### The new question we can answer

> "How do parameter sensitivities vary across growth conditions, and which parameters are universally important vs. condition-specific?"

This is scientifically novel for whole-cell modeling. It bridges:
- **Parameter sensitivity** (RFC006 strategies 1-4) — "which knobs matter?"
- **Condition robustness** — "do the same knobs matter everywhere?"
- **Experimental design** — "which parameter should I measure next, given I care about multiple conditions?"

---

## Scope

### Tier 0: Safety Net (prerequisite, ~1 day)

Ensure existing UQ functionality is not broken by multi-parca-aws, and add guardrails for the new output structure.

### Tier 1: Cross-Condition GSA (core feature, ~3-5 days)

Enable `uq sample --conditions <id1> <id2> ...` to run multi-condition UQ, and extend `quantify` + dashboards to display per-condition Sobol comparisons.

### Explicitly out of scope (Tier 2, future)

- Treating RNA-seq dataset choice as a *categorical* uncertainty input alongside continuous parameters (requires mixed-type PCE or ensemble methods)
- Interaction between condition choice and parameter sensitivity (would need a joint Sobol framework)
- Campaign-style perturbation operators (log-normal noise, gene zeroing) — these perturb RNA-seq *input data*, not simData parameters, and answer a different question

---

## Tier 0: Safety Net

### 0.1 Observable Dimension Guard

**Problem:** If different ParCa datasets produce different gene sets (e.g., one dataset has 4,300 genes, another has 4,280), `collect_observables()` will silently produce misaligned arrays. UQ assumes all variants have shape `(n_timesteps, n_obs)` with the same `n_obs`.

**Files:** `uq/observables.py`

**What:** After collecting all variant DataFrames, verify that every variant has the same column set. If not:
- Log a warning with the differing columns
- Intersect to the common column set (safe default)
- Optionally raise if `--strict` flag is passed

**Implementation:**
```python
def _validate_observable_alignment(
    dfs: list[pl.DataFrame],
    n_variants: int,
) -> list[str]:
    """Return the intersection of column sets across all variants.

    Warns if any variant has a different column set.
    """
    col_sets = [set(df.columns) for df in dfs]
    common = set.intersection(*col_sets) if col_sets else set()
    union = set.union(*col_sets) if col_sets else set()
    if common != union:
        missing = union - common
        logger.warning(
            "Observable dimension mismatch: %d columns not present in all variants: %s",
            len(missing), sorted(missing)[:10],
        )
    return sorted(common)
```

Call this in `collect_observables()` before stacking arrays.

**Test:** Create two mock Parquet directories with slightly different column sets, verify warning is logged and common intersection is returned.

---

### 0.2 Multi-ParCa Config Passthrough

**Problem:** The UQ config generator (`_build_config()` in `uq/tui.py`) doesn't know about `parca_variants`. If a user has a multi-parca base config, UQ should preserve it rather than silently dropping it.

**Files:** `uq/tui.py` (`_build_config()`), `uq/cli.py` (sample command)

**What:** Add optional `base_config_path` parameter. If provided, load the base config and merge UQ-specific fields (variants, emitter, output_dir) on top, preserving `parca_variants`, `analysis_options`, and any other multi-parca keys.

**Implementation:**
```python
def _build_config(
    sim_data_path: str,
    output_dir: str,
    variants_section: dict[str, Any],
    experiment_id: str = "uqpc_batch",
    n_init_sims: int = 1,
    generations: int = 1,
    max_duration: float = 10800.0,
    base_config_path: str | None = None,  # NEW
) -> dict[str, Any]:
    if base_config_path:
        base = json.loads(Path(base_config_path).read_text())
    else:
        base = {}

    # UQ fields override base; everything else preserved
    base.update({
        "sim_data_path": sim_data_path,
        "experiment_id": experiment_id,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": output_dir},
        "max_duration": max_duration,
        "n_init_sims": n_init_sims,
        "generations": generations,
        "single_daughters": True,
        "suffix_time": False,
        "variants": variants_section,
    })
    return base
```

**CLI addition:**
```
--base-config PATH  Base vEcoli config to merge with (preserves parca_variants, analysis_options, etc.)
```

**Test:** Build config with and without base_config_path, verify parca_variants preserved when present.

---

### 0.3 Manifest Records Multi-ParCa Info

**Problem:** When running on multi-parca-aws, the manifest.json should record which ParCa datasets were used, so results are fully reproducible.

**Files:** `uq/workflow.py` (`_write_manifest()`)

**What:** If the workflow config contains `parca_variants`, include it in the manifest alongside the existing fields.

**Implementation:** In `_write_manifest()`, after building the manifest dict:
```python
# If a base config was used, record parca_variants for provenance
config_path = cache.cache_dir / "_batch" / "workflow_config.json"
if config_path.exists():
    wf_config = _json.loads(config_path.read_text())
    if "parca_variants" in wf_config:
        manifest["parca_variants"] = wf_config["parca_variants"]
```

**Test:** Run with a config containing parca_variants, verify they appear in manifest.json.

---

## Tier 1: Cross-Condition GSA

### Architecture Overview

```
uq sample --conditions glucose_minimal glucose_rich \
    --sim-data-path /path/to/simData.cPickle \
    --cache-dir ./uq_cache_multi \
    --n-samples 20

                    ┌─────────────────────────────────┐
                    │  PCRV.sampleGerm() → N samples   │
                    │  (same parameter space for all)   │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┼──────────────────────┐
              ▼                  ▼                       ▼
    ┌─────────────────┐ ┌─────────────────┐   ┌─────────────────┐
    │  ParCa 0         │ │  ParCa 1         │   │  ParCa 2         │
    │  glucose_minimal │ │  glucose_rich    │   │  precise_wt_glc  │
    │  variants 0..N   │ │  variants N+1..2N│   │  variants 2N+1.. │
    └────────┬────────┘ └────────┬────────┘   └────────┬────────┘
             │                   │                      │
             ▼                   ▼                      ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Single Nextflow workflow → hive-partitioned Parquet     │
    │  history/experiment_id=X/variant=V/.../*.pq              │
    └────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                   ▼
    ┌──────────────┐ ┌──────────────┐   ┌──────────────┐
    │ Cache cond=0  │ │ Cache cond=1  │   │ Cache cond=2  │
    │ X, Y, meta    │ │ X, Y, meta    │   │ X, Y, meta    │
    └──────┬───────┘ └──────┬───────┘   └──────┬───────┘
           │                │                   │
           ▼                ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │  quantify → per-condition Sobol + cross-condition    │
    │  comparison (ΔS_Ti heatmap, rank stability, etc.)    │
    └─────────────────────────────────────────────────────┘
```

The key insight: **all conditions share the same X (parameter samples)**, so Sobol indices are directly comparable across conditions. The only thing that changes is Y.

---

### 1.1 Multi-Condition Sampling

**Files:** `uq/cli.py` (sample command), `uq/tui.py` (`_build_config()`)

**What:** Add `--conditions` flag to `uq sample`. Each condition is an `rnaseq_basal_dataset_id` string. When provided:
1. Build a multi-parca config with `parca_variants` populated
2. Each ParCa gets the same `sim_data_setattr` mutations (same X)
3. Offset indexing is handled by vEcoli's Nextflow template automatically
4. After workflow completes, split outputs by condition into separate cache subdirectories

**CLI:**
```
uq sample /path/to/simData.cPickle \
    --cache-dir ./uq_cache_multi \
    --n-samples 20 \
    --conditions vecoli_m9_glucose_minus_aas \
    --conditions vecoli_m9_glucose_plus_aas \
    --conditions precise_control:wt_glc
```

**Config generation changes:**
```python
def _build_multi_condition_config(
    sim_data_path: str,
    output_dir: str,
    variants_section: dict[str, Any],
    conditions: list[str],
    **kwargs,
) -> dict[str, Any]:
    """Build a multi-parca workflow config for cross-condition UQ."""
    config = _build_config(sim_data_path, output_dir, variants_section, **kwargs)
    config["parca_variants"] = [
        {"rnaseq_basal_dataset_id": cond_id}
        for cond_id in conditions
    ]
    return config
```

**Cache structure:**
```
uq_cache_multi/
├── condition_0_glucose_minimal/
│   ├── X.npy          # Same X for all conditions
│   ├── Y.npy          # Y from ParCa 0 variants
│   ├── germ_train.npy
│   └── metadata.json
├── condition_1_glucose_rich/
│   ├── X.npy          # Same X (symlink or copy)
│   ├── Y.npy          # Y from ParCa 1 variants
│   └── ...
├── conditions.json    # Maps condition index → dataset_id
└── X.npy              # Shared X (canonical copy)
```

**Output splitting logic:**
After workflow completes, determine which variant indices belong to which ParCa by reading the merged metadata.json or by using the known offset formula:
```python
n_variants_per_parca = n_samples + 1  # +1 for baseline
for cond_idx, cond_id in enumerate(conditions):
    offset = cond_idx * n_variants_per_parca
    # Variants offset+1 .. offset+n_samples belong to this condition
    # (offset+0 = baseline for this parca)
```

**Test:** Mock a 2-condition config, verify parca_variants populated, verify output splitting assigns correct variant ranges.

---

### 1.2 Per-Condition Quantification

**Files:** `uq/workflow.py` (new `quantify_multi_condition()`), `uq/cli.py`

**What:** Extend `quantify` to detect multi-condition caches and run all 4 strategies per condition, then compute cross-condition comparisons.

**New result type:**
```python
@dataclass
class MultiConditionResult:
    """Cross-condition UQ results."""
    conditions: list[str]                          # dataset IDs
    per_condition: dict[str, QuantifyResult]        # condition → full result

    # Cross-condition derived quantities
    rank_stability: np.ndarray                     # (n_params,) — how stable is each param's rank
    condition_specific: dict[str, list[str]]        # condition → params that are uniquely important there
    universal_drivers: list[str]                    # params important in ALL conditions

    def export(self, export_dir: str | Path) -> Path:
        """Export per-condition results + cross-condition summary."""
        ...
```

**Cross-condition metrics:**

1. **Rank stability** — For each parameter, compute its S_Ti rank in each condition. Stability = 1 - (std of ranks / max possible std). A parameter with rank [1, 1, 2] across 3 conditions is highly stable.

2. **Universal drivers** — Parameters where S_Ti > threshold (e.g., 0.1) in ALL conditions.

3. **Condition-specific drivers** — Parameters where S_Ti > threshold in one condition but < threshold/2 in others.

4. **Differential Sobol matrix** — `(n_conditions, n_conditions, n_params)` array of pairwise ΔS_Ti.

**Implementation:**
```python
def quantify_multi_condition(
    cache_dir: str | Path,
    sim_data_path: str | Path,
    polynomial_order: int = 3,
    n_bins: int = 10,
    regression: str = "lsq",
    export_path: str | Path | None = None,
) -> MultiConditionResult:
    cache_dir = Path(cache_dir)
    conditions_meta = json.loads((cache_dir / "conditions.json").read_text())
    conditions = conditions_meta["conditions"]

    per_condition = {}
    for cond_id in conditions:
        cond_cache = cache_dir / f"condition_{cond_id}"
        per_condition[cond_id] = quantify(
            cache_dir=str(cond_cache),
            sim_data_path=sim_data_path,
            polynomial_order=polynomial_order,
            n_bins=n_bins,
            regression=regression,
        )

    # Compute cross-condition metrics
    ...

    result = MultiConditionResult(
        conditions=conditions,
        per_condition=per_condition,
        rank_stability=_compute_rank_stability(per_condition),
        condition_specific=_find_condition_specific(per_condition),
        universal_drivers=_find_universal_drivers(per_condition),
    )

    if export_path:
        result.export(export_path)

    return result
```

**CLI:**
```
uq quantify /path/to/simData.cPickle \
    --cache-dir ./uq_cache_multi \
    --export-path ./uq_results_multi
```

Auto-detects multi-condition cache by presence of `conditions.json`.

**Test:** Create 2 synthetic per-condition caches with known Sobol differences, verify rank stability and condition-specific detection.

---

### 1.3 Cross-Condition CLI Report

**Files:** `uq/cli.py` (`_print_multi_condition_report()`)

**What:** When `quantify` detects multi-condition results, print:

1. **Per-condition Sobol tables** (reuse existing `_print_report`)
2. **Cross-condition comparison table:**
```
┌─────────────────────────────────────────────────────────────┐
│ CROSS-CONDITION SENSITIVITY COMPARISON                       │
├─────────────────────┬───────────┬───────────┬───────────────┤
│ PARAMETER           │ glucose-  │ glucose+  │ RANK STABILITY│
├─────────────────────┼───────────┼───────────┼───────────────┤
│ kinetic_obj_weight   │ 42.1%     │ 38.7%     │ ●●●●● (1.00) │
│ secretion_penalty    │ 31.2%     │  8.4%     │ ●●○○○ (0.40) │
│ rnap_free            │ 12.0%     │ 28.1%     │ ●●○○○ (0.40) │
│ ...                  │ ...       │ ...       │ ...           │
└─────────────────────┴───────────┴───────────┴───────────────┘
```
3. **Key Findings narrative:**
```
┌───────────────────────────────────────────────────────────────┐
│ KEY FINDINGS — CROSS-CONDITION                                │
│                                                               │
│ • kinetic_objective_weight is the top driver in ALL conditions │
│   (universal — measure this first regardless of media)        │
│                                                               │
│ • secretion_penalty_coeff is condition-specific: 31.2% in     │
│   glucose- but only 8.4% in glucose+ (overflow metabolism     │
│   matters more in minimal media)                              │
│                                                               │
│ • Recommendation: prioritize kinetic_objective_weight for     │
│   cross-condition robustness; investigate secretion_penalty    │
│   if minimal-media predictions are the target.                │
└───────────────────────────────────────────────────────────────┘
```

**Test:** Mock MultiConditionResult, verify formatted output contains expected strings.

---

### 1.4 Cross-Condition Dashboard Panels

**Files:** `app/dashboard_simple.py` (new cells), `app/uq_daw_simple.py` (new canvas)

**What:** When `uq_results.json` contains a `"conditions"` key:

**Marimo dashboard:**
- New cell: Plotly grouped bar chart — conditions × params, S_Ti values (similar to strategy 2-3 bars)
- New cell: Rank stability indicator (dot plot or radar chart)
- Condition selector dropdown → switches which condition's spectrogram/response curves are shown

**Tk DAW:**
- New `ConditionBarCanvas` (reuse `StrategyBarCanvas` pattern)
- Condition selector combobox in left panel
- When condition changes, reload that condition's surrogate coefficients and update response curves

**Export JSON extension:**
```json
{
    "conditions": ["glucose_minimal", "glucose_rich"],
    "cross_condition": {
        "rank_stability": {"kinetic_objective_weight": 1.0, "secretion_penalty_coeff": 0.4, ...},
        "universal_drivers": ["kinetic_objective_weight"],
        "condition_specific": {
            "glucose_minimal": ["secretion_penalty_coeff"],
            "glucose_rich": ["rnap_free"]
        },
        "differential_sobol": {
            "glucose_minimal_vs_glucose_rich": {
                "kinetic_objective_weight": 0.034,
                "secretion_penalty_coeff": 0.228,
                ...
            }
        }
    },
    "per_condition": {
        "glucose_minimal": { ... existing uq_results.json schema ... },
        "glucose_rich": { ... }
    }
}
```

**Test:** Load multi-condition JSON in Marimo, verify grouped bars render. Headless Tk test: select condition, verify response curves update.

---

### 1.5 TUI Multi-Condition Support

**Files:** `uq/tui.py`

**What:** Add a `Conditions` input field in the sidebar (comma-separated dataset IDs). When non-empty:
- Sample step builds a multi-parca config
- Progress bar shows per-parca status
- Quantify step runs per-condition
- Results tab shows cross-condition comparison table

**Implementation:** Primarily UI wiring — the `_build_config` and `quantify_multi_condition` do the work.

**Test:** Headless: set conditions field, verify config includes parca_variants.

---

## Implementation Order

```
0.1 Dimension Guard  →  0.2 Config Passthrough  →  0.3 Manifest Extension
         ↓
1.1 Multi-Condition Sampling  →  1.2 Per-Condition Quantification
         ↓
1.3 CLI Report  →  1.4 Dashboard Panels  →  1.5 TUI Support
```

Tier 0 items are prerequisites — they can be merged independently.
Tier 1 items build sequentially: sampling → quantification → visualization.

---

## Files Modified

| File | Items |
|------|-------|
| `uq/observables.py` | 0.1 |
| `uq/tui.py` | 0.2, 1.1, 1.5 |
| `uq/workflow.py` | 0.3, 1.2 |
| `uq/cli.py` | 1.1, 1.3 |
| `app/dashboard_simple.py` | 1.4 |
| `app/uq_daw_simple.py` | 1.4 |

## New Files

| File | Item |
|------|------|
| `uq/multi_condition.py` | 1.2 (cross-condition metrics) |
| `tests/test_dimension_guard.py` | 0.1 |
| `tests/test_multi_condition.py` | 1.1, 1.2 |
| `tests/test_cross_condition_report.py` | 1.3 |

---

## Testing Strategy

### Unit tests (fast, no vEcoli)
- Dimension guard with mock DataFrames
- Config generation with/without conditions
- Cross-condition metric computation (rank stability, universal drivers) from synthetic Sobol arrays
- CLI report formatting from mock MultiConditionResult
- Dashboard JSON schema validation

### Integration tests (need vEcoli, slow)
- 2-condition sampling with 5 samples each on `vecoli_m9_glucose_minus_aas` + `vecoli_m9_glucose_plus_aas`
- Verify cache structure, per-condition X identity, Y shapes
- Full quantify → export → dashboard load cycle

### Known constraints
- Requires `multi-parca-aws` branch in vEcoli (editable install)
- Requires `ecoli-sources` repo with RNA-seq manifests accessible
- Integration tests need ~2h wall time for 2×5 = 10 samples + 2 parcas
- ParCa failures on some datasets are expected — tests should use known-good datasets only

---

## What's been done

- Plan created (2026-04-16)
