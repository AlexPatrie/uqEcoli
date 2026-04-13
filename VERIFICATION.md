# Observable Coverage Verification

This document verifies that the UQ pipeline (`uq sample → uq quantify`)
supports both categories of observables requested for review:

1. **Public-repo general metrics** — doubling time, biomass composition
2. **CD1 deliverable analyses** — higher-order properties, exchange fluxes,
   transcriptome, proteome (as reported for Vegas + Bermuda)

All numbers below were verified on real vEcoli Parquet output (20 variants,
baseline run from `sim_data/baseline/kb/simData.cPickle`).

---

## 1. Public-repo general metrics

Available via `--observables higher_order`:

| Metric | Preset feature name | Verified value (mean) | Source |
|---|---|---|---|
| **Doubling time** | `doubling_time_hours` | 0.75 h (~45 min) | `ln(2) / mean(instantaneous_growth_rate) / 3600` |
| **Growth rate** | `growth_rate_per_hour` | 0.93 h⁻¹ | `instantaneous_growth_rate × 3600` |
| **DNA fraction** | `dna_fraction_g_per_g_dw` | 0.019 g/g DW | `dna_mass / dry_mass` |
| **RNA fraction** | `rna_fraction_g_per_g_dw` | 0.132 g/g DW | `rna_mass / dry_mass` |
| **Dry mass fraction** | `dry_mass_fraction` | 0.287 | `dry_mass / cell_mass` |
| **Cell volume** | `cell_volume_um3` | 1.47 μm³ | direct from `listeners__mass__volume` |

These are the standard biomass composition metrics. The doubling time and
growth rate are the canonical public-repo benchmarks.

```bash
uv run uq sample /path/to/simData.cPickle --observables higher_order
```

---

## 2. CD1 analyses (Vegas + Bermuda deliverables)

Each cd1 analysis module in `vEcoli-private/ecoli/analysis/multiseed/`
has a corresponding observable preset in `uq/observables.py`. The
extraction uses the **same Parquet columns and aggregation** (time-average
per variant, optional generation filtering).

| cd1 module | UQ preset | `--observables` flag | Features | Verified |
|---|---|---|---|---|
| `cd1_higher_order_properties.py` | `higher_order` | `--observables higher_order` | 6 derived metrics | ✅ |
| `cd1_exchange_fluxes.py` | `exchange_fluxes` | `--observables exchange_fluxes` | 87 exchange metabolites | ✅ |
| `cd1_transcriptomics.py` | `transcriptome` | `--observables transcriptome` | 4,345 mRNA cistron counts | ✅ |
| `cd1_proteomics.py` | `proteome` | `--observables proteome` | 4,309 protein monomer counts | ✅ |
| `cd1_fluxomics.py` | `fluxome` | `--observables fluxome` | 2,797 reaction fluxes (dry-mass normalized) | ✅ |
| `cd1_metabolomics.py` | *(planned)* | — | ~200 metabolites | requires `sim_data` for `conc_dict` index lookup |

### Full CD1 run (all at once):

```bash
uv run uq sample /path/to/simData.cPickle \
    --n-samples 200 \
    --observables higher_order \
    --observables exchange_fluxes \
    --observables transcriptome \
    --observables proteome \
    --observables fluxome \
    --generation-lower-bound 2

uv run uq quantify /path/to/simData.cPickle \
    --polynomial-order 2 \
    --regression lsq
```

This produces a Y matrix with **11,544 features** and runs PCE + Sobol
independently for each — answering *"which sim_data parameters drive
variance in gene X / protein Y / flux Z"* across the entire multi-omics
output space.

### Generation filtering

```bash
--generation-lower-bound 2
```

Mirrors the cd1 `generation_lower_bound` parameter: skips transient
initialization dynamics and focuses the sensitivity analysis on
steady-state growth. Same SQL `WHERE generation >= N` logic the cd1
modules use.

---

## 3. What the PCE/Sobol pipeline adds beyond cd1

The cd1 modules compute **descriptive statistics** (mean, std across
cells). The UQ pipeline adds **causal attribution**:

- **cd1:** "the mean doubling time is 45 minutes"
- **UQ:** "48% of the doubling-time variance is driven by
  `fraction_active_rnap_free`, 30% by `basal_elongation_rate`, ..."

This works because the PCE surrogate approximates each observable as a
polynomial function of the input parameters, and the Sobol indices
decompose that polynomial's variance analytically (Sudret 2008).

---

## 4. Test coverage

`tests/test_observables.py` — 10 tests, all passing on real data:

- Each preset individually (mass, exchange_fluxes, transcriptome,
  proteome, fluxome)
- Preset composition (mass + exchange_fluxes → concatenated Y)
- Generation lower-bound filtering
- Timeseries + metadata extraction
- Unknown preset error handling

---

## 5. Architecture note

Observable extraction (`uq/observables.py`) is decoupled from both
sampling and quantification. The presets are composable and additive —
adding a new preset (e.g., metabolomics when `conc_dict` support is
ready) requires only a new entry in the `PRESETS` dict + an extraction
function. No changes to the PCE/Sobol pipeline.
