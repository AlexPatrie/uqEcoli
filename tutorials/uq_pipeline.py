"""Interactive walkthrough: the `uq sample → uq quantify` UQPC pipeline.

Run with:  uv run marimo run tutorials/uq_pipeline.py
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # `uq sample` → `uq quantify`

    This notebook walks through the **two-stage UQ pipeline** cell by cell.
    Every PyTUQ call maps 1:1 to the
    [UQPC reference workflow](https://sandialabs.github.io/pytuq/apps/uqpc.html).
    Every aggregation strategy maps to
    [RFC006 §1](readmes/start/tools/RFC006.md).

    | Stage | CLI | UQPC steps | What it does |
    |---|---|---|---|
    | 1 | `uq sample` | 1-3 | Setup input PC, draw germ samples, run vEcoli |
    | 2 | `uq quantify` | 4-5 | Fit PCE surrogate, compute Sobol indices |
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np

    # ── Edit these before running ──
    SIM_DATA_PATH = "sim_data/baseline/kb/simData.cPickle"
    CACHE_DIR = "./uq_cache"
    EXPORT_DIR = "./uq_results"
    N_SAMPLES = 20
    POLYNOMIAL_ORDER = 2
    REGRESSION = "lsq"  # lsq | bcs | anl
    SEED = 42
    GENERATIONS = 1
    N_INIT_SIMS = 1
    N_BINS = 10

    # Observable presets (cd1 analysis modules).
    # Options: mass, higher_order, exchange_fluxes, transcriptome, proteome, fluxome
    OBSERVABLES = ["mass"]
    GENERATION_LOWER_BOUND = 0  # skip first N generations (cd1 pattern)

    sim_data_path = str(Path(SIM_DATA_PATH).resolve())
    cache_dir = str(Path(CACHE_DIR).resolve())
    export_dir = str(Path(EXPORT_DIR).resolve())
    return (
        GENERATION_LOWER_BOUND,
        GENERATIONS,
        N_BINS,
        N_INIT_SIMS,
        N_SAMPLES,
        OBSERVABLES,
        POLYNOMIAL_ORDER,
        REGRESSION,
        SEED,
        cache_dir,
        export_dir,
        np,
        sim_data_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Stage 1: `uq sample`

    **UQPC steps 1-3** — setup the Legendre input PC from parameter bounds,
    draw germ samples via `PCRV.sampleGerm()`, evaluate vEcoli as the
    black-box model, extract observables via cd1 presets.

    This is the expensive stage (runs Nextflow simulations).  Once cached,
    you never need to re-run it.

    Observable presets (edit `OBSERVABLES` above):

    | Preset | cd1 module | Features |
    |---|---|---|
    | `mass` | cd1_higher_order_properties (raw) | 5 |
    | `higher_order` | cd1_higher_order_properties (derived) | 6 |
    | `exchange_fluxes` | cd1_exchange_fluxes | ~87 |
    | `transcriptome` | cd1_transcriptomics | ~4,300 |
    | `proteome` | cd1_proteomics | ~4,300 |
    | `fluxome` | cd1_fluxomics | ~2,800 |
    """)
    return


@app.cell
def _(
    GENERATION_LOWER_BOUND,
    GENERATIONS,
    N_INIT_SIMS,
    N_SAMPLES,
    OBSERVABLES,
    SEED,
    cache_dir,
    sim_data_path,
):
    from uq.workflow import sample

    cache = sample(
        sim_data_path=sim_data_path,
        cache_dir=cache_dir,
        n_samples=N_SAMPLES,
        seed=SEED,
        generations=GENERATIONS,
        n_init_sims=N_INIT_SIMS,
        observable_columns=None,  # overridden by CLI --observables; here we use workflow default
    )

    print(f"X = {cache.X.shape}  (N_samples × N_params)")
    print(f"Y = {cache.Y.shape}  (N_samples × N_outputs)")
    print(f"Observables: {OBSERVABLES}, gen_lower_bound={GENERATION_LOWER_BOUND}")
    if cache.Y_timeseries:
        print(f"Timeseries: {len(cache.Y_timeseries)} samples")
    return (cache,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Under the hood: what `sample()` does

    `sample()` is a thin wrapper around three PyTUQ calls + one vEcoli
    subprocess spawn.  Here they are individually, with the equivalent
    `uq_pc.py` flags annotated:

    ```
    UQPC step 1  (--pdom, --pctype LU, --pcord 1)
        → PCRV(ndim, npc, "LU", mi=get_mi(1,d), cfs=affine_map)

    UQPC step 2  (--sampl rand, --nqd N, --seed 42)
        → germ = PCRV.sampleGerm(N)
        → X    = PCRV.evalPC(germ)

    UQPC step 3  (--regime online_bb)
        → each row of X → one sim_data_setattr variant
        → subprocess: runscripts/workflow.py --config batch.json
        → collect Parquet → Y.npy + timeseries/
    ```

    The cell below reproduces steps 1-2 so you can inspect the objects.
    """)
    return


@app.cell
def _(cache, np):
    from pytuq.rv.pcrv import PCRV
    from pytuq.utils.mindex import get_mi

    # Step 1: reconstruct the input PC from the cached bounds
    bounds = np.array(cache.metadata["bounds"])
    ndim = bounds.shape[0]
    midpoints = 0.5 * (bounds[:, 1] + bounds[:, 0])
    half_ranges = 0.5 * (bounds[:, 1] - bounds[:, 0])
    pcf = np.vstack((midpoints, np.diag(half_ranges)))
    mi = get_mi(1, ndim)
    input_pc = PCRV(ndim, ndim, "LU", mi=mi, cfs=pcf.T)  # Legendre, order 1

    # Step 2: verify the germ ↔ physical mapping
    germ = np.load(str(cache.cache_dir) + "/germ_train.npy")
    x_reconstructed = input_pc.evalPC(germ)
    print("Step 1: input PCRV constructed (Legendre, LU, order 1)")
    print(f"  {ndim} parameters, germ ∈ [-1,1]^{ndim}")
    print(f"\nStep 2: germ_train.shape = {germ.shape}")
    print(f"  X reconstruction matches cache: {np.allclose(x_reconstructed, cache.X)}")
    print(f"\nParameters:")
    for _i in range(ndim):
        print(f"  {cache.parameter_names[_i]}: [{bounds[_i,0]:.4g}, {bounds[_i,1]:.4g}]")
    return (bounds,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Stage 2: `uq quantify`

    **UQPC steps 4-5** — fit a Legendre PCE surrogate, compute Sobol
    indices analytically from the coefficients (Sudret 2008).

    Runs the same PCE/Sobol pipeline **four times** (one per RFC006
    aggregation strategy).  No vEcoli calls — pure matrix algebra on
    the cached `(X, Y)`.
    """)
    return


@app.cell
def _(
    N_BINS,
    POLYNOMIAL_ORDER,
    REGRESSION,
    cache_dir,
    export_dir,
    sim_data_path,
):
    from uq.workflow import quantify

    result = quantify(
        cache_dir=cache_dir,
        sim_data_path=sim_data_path,
        polynomial_order=POLYNOMIAL_ORDER,
        n_bins=N_BINS,
        regression=REGRESSION,
        export_path=export_dir,
    )

    print(f"Strategy 1 (uniform):           {len(result.parameter_names)} params")
    print(f"Strategy 2 (by generation):     {len(result.strategy2)} groups")
    print(f"Strategy 3 (by lineage seed):   {len(result.strategy3)} groups")
    print(f"Strategy 4 (growth-stratified): {len(result.strategy4_per_stage)} stages")
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Under the hood: what `quantify()` does

    For each strategy, `quantify()` runs:

    ```
    UQPC step 4  (--method lsq, --outord 2)
        mi   = get_mi(order, ndim)
        pcrv = PCRV(n_out, ndim, "LU", mi=mi)
        A    = pcrv.evalBases(germ, 0)          # Legendre basis matrix
        for each output j:
            reg = lsq()                          # or bcs() / anl()
            reg.fita(A, Y[:, j])                 # fit coefficients
        pcrv.setMiCfs(mindices, coefficients)
        pcrv.setFunction()

    UQPC step 5
        Y_hat = pcrv.function(germ)              # surrogate prediction
        relerr = ||Y - Y_hat|| / ||Y||           # quality check
        S_i    = pcrv.computeSens()               # first-order Sobol
        S_Ti   = pcrv.computeTotSens()            # total-order Sobol
        S_ij   = pcrv.computeJointSens()          # second-order Sobol
    ```

    The cell below reproduces this for **Strategy 1** so you can
    inspect every object.
    """)
    return


@app.cell
def _(bounds, cache, np, result):
    from pytuq.lreg.lreg import lsq as Lsq
    from pytuq.rv.pcrv import PCRV as Pcrv
    from pytuq.utils.mindex import get_mi as GetMi

    # Load inputs
    germ_train = np.load(str(cache.cache_dir) + "/germ_train.npy")
    Y = cache.Y
    ndim_s4 = bounds.shape[0]

    # Step 4: build PCE and fit
    mi_out = GetMi(2, ndim_s4)
    pcrv = Pcrv(Y.shape[1], ndim_s4, "LU", mi=mi_out)
    A = pcrv.evalBases(germ_train, 0)  # basis matrix
    print(f"Step 4: basis matrix A = {A.shape}  (samples × terms)")

    mlist, clist = [], []
    for _j in range(Y.shape[1]):
        _r = Lsq()
        _r.fita(A, Y[:, _j])  # ← uq_pc.py: reg.fita(Amat, ytrain[:,j])
        mlist.append(mi_out[_r.used, :])
        clist.append(_r.cf)
    pcrv.setMiCfs(mlist, clist)
    pcrv.setFunction()

    # Step 5: predict + Sobol
    Y_hat = pcrv.function(germ_train)
    norms = np.maximum(np.linalg.norm(Y, axis=0), 1e-12)
    relerr = np.linalg.norm(Y - Y_hat, axis=0) / norms
    S_main = pcrv.computeSens()   # (n_out, n_params)
    S_total = pcrv.computeTotSens()

    print(f"\nStep 5: training relerr = {relerr}")
    print(f"\nSobol total-order (output 0):")
    for _i in range(ndim_s4):
        print(f"  {cache.parameter_names[_i]:<40s} {S_total[0, _i]:.1%}")

    # Verify matches quantify() output
    print(f"\nMatches quantify() strategy 1: "
          f"{np.allclose(relerr, result.strategy1.relerr_train, atol=1e-6)}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Results: RFC006 strategies 1-4

    The PCE machinery is **identical** across all strategies — only the Y
    matrix changes:

    | # | RFC006 §1 | Y aggregation |
    |---|---|---|
    | 1 | "Uniformly across all cells and times" | Mean over all timesteps |
    | 2 | "Stratified by generation" | Mean per generation |
    | 3 | "Stratified by lineage seed" | Mean per seed |
    | 4 | "Stratified by cell cycle stage" | Mean per θ-bin |
    """)
    return


@app.cell
def _(result):
    print("═══ Strategy 1: Population-averaged ═══")
    print(f"Training relerr: {result.strategy1.relerr_train}\n")
    for _i, _p in enumerate(result.parameter_names):
        _s = result.strategy1.sobol.total_order[_i]
        print(f"  {_p:<40s} S_Ti = {_s:>6.1%}  {'█' * int(_s * 40)}")
    return


@app.cell
def _(result):
    if result.strategy2:
        print("═══ Strategy 2: By generation ═══")
        for _g, _r in sorted(result.strategy2.items()):
            print(f"\n  Generation {_g}:")
            for _i, _p in enumerate(result.parameter_names):
                print(f"    {_p:<38s} S_Ti = {_r.sobol.total_order[_i]:.1%}")
    else:
        print("Strategy 2: N/A  (set GENERATIONS >= 2)")
    return


@app.cell
def _(result):
    if result.strategy3:
        print("═══ Strategy 3: By lineage seed ═══")
        for _s, _r in sorted(result.strategy3.items()):
            print(f"\n  Seed {_s}:")
            for _i, _p in enumerate(result.parameter_names):
                print(f"    {_p:<38s} S_Ti = {_r.sobol.total_order[_i]:.1%}")
    else:
        print("Strategy 3: N/A  (set N_INIT_SIMS >= 2)")
    return


@app.cell
def _(result):
    if result.strategy4_per_stage:
        _ns = len(result.strategy4_per_stage)
        print(f"═══ Strategy 4: Growth-stratified ({_ns} θ-bins) ═══")
        print(f"  θ = [log m(t) - log m_birth] / [log m_div - log m_birth]\n")
        _header = f"  {'θ range':<12s}"
        for _p in result.parameter_names:
            _header += f" {_p.split('.')[-1][:10]:>10s}"
        print(_header)
        print("  " + "─" * len(_header))
        for _si, _r in enumerate(result.strategy4_per_stage):
            _row = f"  {_si/_ns:.0%}–{(_si+1)/_ns:.0%}       "
            for _v in _r.sobol.total_order:
                _row += f" {_v:>9.1%}"
            print(_row)
    else:
        print("Strategy 4: N/A  (no timeseries in cache)")
    return


@app.cell
def _(export_dir, mo):
    from pathlib import Path as _Path

    _p = _Path(export_dir)
    if _p.exists():
        _files = "\n".join(
            f"    {f.relative_to(_p)}" for f in sorted(_p.rglob("*")) if f.is_file()
        )
        mo.md(f"""
        ---
        ## Exported to `{export_dir}/`

        ```
    {_files}
        ```

        Open the dashboard:
        ```bash
        uv run uq dashboard {export_dir}/uq_results.json
        ```
        """)
    else:
        mo.md(f"*(export directory `{export_dir}` not found)*")
    return


if __name__ == "__main__":
    app.run()
