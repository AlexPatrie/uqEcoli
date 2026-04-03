# PyTUQ Integration — What's From PyTUQ vs What's Ours

*Last updated: 2026-04-03*

## The UQPC Workflow

Our `uq.workflow` implements PyTUQ's 5-step UQPC workflow
(https://sandialabs.github.io/pytuq/apps/uqpc.html):

| Step | UQPC | Our code | PyTUQ calls |
|------|------|----------|-------------|
| 1 | Setup inputs | `_setup_input_pc()` | `PCRV(pdim, sdim, "LU", mi=mi, cfs=cfs.T)`, `get_mi(order, dim)` |
| 2 | Generate samples | `PCRV.sampleGerm() → evalPC()` | `PCRV.sampleGerm()`, `PCRV.evalPC()` |
| 3 | Evaluate model | `TimeseriesGeneratorVecoli` | (vEcoli, not PyTUQ) |
| 4 | Build surrogate | `_fit_surrogate()` | `PCRV.evalBases()`, `lsq/bcs/anl().fita()`, `PCRV.setMiCfs()`, `PCRV.setFunction()` |
| 5 | Post-process | `_compute_sobol()` | `PCRV.computeSens()`, `computeTotSens()`, `computeJointSens()` |

**Every PCE math operation calls PyTUQ directly. Zero handrolled UQ math.**

## The One Semi-Handrolled Piece: `_fit_surrogate()`

This replicates `pytuq.workflows.fits.pc_fit()` — but only because the
installed version of `pc_fit()` returns just `PCRV` and discards the
per-output linear regression objects. We need those `linreg` objects for
`predicta()` (prediction variance estimation).

The loop body is identical to `pc_fit`'s source — every line calls PyTUQ:

```python
lreg_obj = lsq()                              # pytuq.lreg.lreg.lsq
lreg_obj.fita(Amat, Y_train[:, j])            # pytuq fita()
mindices_list.append(mindex[lreg_obj.used, :]) # pytuq .used
cfs_list.append(lreg_obj.cf)                   # pytuq .cf
pcrv.setMiCfs(mindices_list, cfs_list)         # pytuq setMiCfs
pcrv.setFunction()                             # pytuq setFunction
```

If PyTUQ updates `pc_fit()` to return `(pcrv, linregs)`, we can delete
`_fit_surrogate()` and call `pc_fit()` directly.

## What's Handrolled (vEcoli Domain, Not UQ Math)

1. **Parameter loading** — `ParameterDataset` reads `simData.cPickle`
2. **Parameter space** — `SimDataParameter` specs with dot-paths
3. **Model evaluation** — `TimeseriesGeneratorVecoli._run_batch()` →
   `runscripts/workflow.py` subprocess
4. **Aggregation strategies 2-4** — group-by-generation, group-by-seed,
   growth-stratified θ binning
5. **Disk caching** — `PrecomputedCache` (`.npy` + `metadata.json`)

These are the extension points `uq_pc.py` expects users to customize.

## Critical Gotcha: `setCfs` After Fitting

After `PCE.build(regression='lsq')` or manual `lsq().fita()`, you
**must** call `pcrv.setCfs([lreg.cf])` before computing Sobol indices.
`build()` does NOT sync coefficients to PCRV for `lsq`/`anl` — only
`bcs` does this internally. Our `_fit_surrogate()` uses `setMiCfs()`
which handles this correctly.

## Regression Methods

| Method | PyTUQ class | When to use |
|--------|-------------|-------------|
| `lsq` | `pytuq.lreg.lreg.lsq` | Default. Fast, exact on noiseless polynomial data. |
| `bcs` | `pytuq.lreg.bcs.bcs` | Sparse PCE. Retains only significant terms. |
| `anl` | `pytuq.lreg.anl.anl` | Analytical Bayesian. Calibrated uncertainty. |

## `sampleGerm` Does Not Accept `seed`

`PCRV.sampleGerm(nsam)` — no `seed` parameter. Set `np.random.seed()`
before calling it.
