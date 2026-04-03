"""Tutorial 6: Calculate Cell Cycle Variable from Timeseries

This tutorial demonstrates how to use the uq package to compute a scalar
cell cycle variable from simulation timeseries output using Koopman
spectral analysis.

The cell cycle variable theta in [0, 1] tells you where a cell is in its
division cycle - purely from data, without mechanistic assumptions.

Run with: marimo run 06_calculate_cell_cycle.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import polars as pl

    alt.data_transformers.enable("vegafusion")
    return alt, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Calculate Cell Cycle Variable from Timeseries

    Given a timeseries output `y(t)` from a simulation, compute a **scalar**
    cell cycle variable `theta(t)` that:

    - Ranges from 0 (birth) to 1 (division)
    - Wraps once per cell cycle
    - Is computed purely from data (no mechanistic assumptions)

    ## The Koopman Approach

    We use **Dynamic Mode Decomposition (DMD)** to:
    1. Find the dominant oscillatory mode at the cell cycle frequency
    2. Extract the **eigenfunction phase** as the cell cycle coordinate

    This is the **recommended approach** per RFC006.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Generate (or Load) Timeseries Data

    For demonstration, we generate synthetic cell trajectory data.
    In practice, this would be vEcoli simulation output.
    """)
    return


@app.cell
def _(np, pl):
    from libuq.inputs import load_dataset
    from pathlib import Path

    def _generate_cell_trajectory(
        n_timesteps: int = 200,
        dt: float = 1.0,
        cell_cycle_time: float = 60.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Generate synthetic cell trajectory with cell cycle dynamics.

        Returns:
            X: Timeseries array of shape (n_timesteps, n_observables)
            t: Time array
            observable_names: List of observable names
        """
        rng = np.random.default_rng(seed)
        t = np.arange(n_timesteps) * dt

        # Cell cycle angular frequency
        omega = 2 * np.pi / cell_cycle_time

        # Observable 1: Dry mass (exponential growth with oscillations)
        growth_rate = 0.01
        dry_mass = 100 * np.exp(growth_rate * t) * (1 + 0.05 * np.sin(omega * t))
        dry_mass += 2 * rng.normal(size=n_timesteps)

        # Observable 2: Cell mass (slightly larger than dry mass)
        cell_mass = dry_mass * 1.3 + 1 * rng.normal(size=n_timesteps)

        # Observable 3: Growth rate (oscillates around mean)
        gr = growth_rate * (1 + 0.3 * np.cos(omega * t))
        gr += 0.002 * rng.normal(size=n_timesteps)

        # Stack into trajectory matrix
        X = np.column_stack([dry_mass, cell_mass, gr])
        observable_names = ["dry_mass", "cell_mass", "growth_rate"]

        return X, t, observable_names

    # Generate trajectory
    # X, t, obs_names = generate_cell_trajectory(
    #     n_timesteps=200,
    #     dt=1.0,
    #     cell_cycle_time=60.0,  # 60 time units per cycle
    # )

    def load_trajectory():
        import json

        with open("baseline_observables.json", "r") as f:
            obs = [col for col in json.load(f) if col.startswith("listener")]
        obs.append("time")
        X = load_dataset(
            experiment_id="api_simulation_default",
            outdir_root=Path("/Users/alexanderpatrie/sms/sms-api/artifacts/sims"),
            observables=obs,
        )

        observables = []
        schema = X.collect_schema()
        for obs_i in obs:
            coltype = schema[obs_i]

            if isinstance(coltype, pl.Float64):
                observables.append(obs_i)

        X = X.select(observables)
        return X

    X = load_trajectory()
    t = X.select("time").to_numpy()
    return X, t


@app.cell
def _(X):
    X.head()
    return


@app.cell
def _(X, alt, mo, pl, t):
    # Visualize the raw timeseries
    obs_names = X.columns
    traj_data = []
    for i, name in enumerate(obs_names):
        vals = X[:, i]
        # Normalize for visualization
        vals_norm = (vals - vals.mean()) / vals.std()
        for j, (time, val) in enumerate(zip(t, vals_norm)):
            traj_data.append({
                "time": time,
                "value": val,
                "observable": name,
            })

    traj_df = pl.DataFrame(traj_data)

    timeseries_chart = (
        alt.Chart(traj_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="Time"),
            y=alt.Y("value:Q", title="Normalized Value"),
            color=alt.Color("observable:N", legend=alt.Legend(title="Observable")),
        )
        .properties(width=600, height=300, title="Input Timeseries (Normalized)")
    )

    mo.md(f"""
    ### Raw Timeseries Data

    - **Shape:** {X.shape} (timesteps x observables)
    - **Observables:** {obs_names}
    - **Duration:** {t[-1]} time units
    - **Expected cell cycle:** ~60 time units

    The oscillations visible in the data are the cell cycle dynamics we want to capture.
    """)
    return (timeseries_chart,)


@app.cell
def _(timeseries_chart):
    timeseries_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Compute Cell Cycle Variable with Koopman

    The `KoopmanCellCycleVariable` class:
    1. Fits DMD to the trajectory
    2. Identifies the cell cycle mode (oscillatory mode near expected frequency)
    3. Extracts the eigenfunction phase as theta in [0, 1]
    """)
    return


@app.cell
def _(X, np, pl):
    from libuq import KoopmanCellCycleVariable

    # Create the Koopman cell cycle variable computer
    koopman_cc = KoopmanCellCycleVariable(
        expected_cycle_time=60.0,  # Expected cell cycle duration
        frequency_tolerance=0.3,  # 30% tolerance for matching frequency
        dt=1.0,  # Timestep of data
        use_edmd=True,  # Use Extended DMD for nonlinearity
        observable_columns=[  # Which columns to use (must match data)
            "dry_mass",
            "cell_mass",
        ],
    )

    # Convert numpy array to Polars DataFrame (required by compute())
    # Add required metadata columns
    n_timesteps = X.shape[0]
    data_df = pl.DataFrame({
        "dry_mass": X[:, 0],
        "cell_mass": X[:, 1],
        "growth_rate": X[:, 2],
        "time": np.arange(n_timesteps).astype(float),
        "generation": np.zeros(n_timesteps, dtype=int),
        "agent_id": ["cell_0"] * n_timesteps,
    })

    # Compute cell cycle variable
    cc_result = koopman_cc.compute(data_df)
    return cc_result, koopman_cc


@app.cell
def _(cc_result, koopman_cc, mo):
    # Display results
    mode = koopman_cc.cell_cycle_mode

    mode_info = "No cell cycle mode identified (fallback used)"
    if mode is not None:
        mode_info = f"""
    **Identified Cell Cycle Mode:**
    | Property | Value |
    |----------|-------|
    | Eigenvalue | {mode.eigenvalue:.4f} |
    | Frequency | {mode.frequency:.6f} Hz |
    | Period | {mode.period:.2f} time units |
    | Growth rate | {mode.growth_rate:.6f} |
    | Is oscillatory | {mode.is_oscillatory} |
    | Is stable | {mode.is_stable} |
    """

    mo.md(f"""
    ### Cell Cycle Variable Computed

    **Output:** `cc_result.values` - numpy array of shape ({len(cc_result.parameters)},)

    | Statistic | Value |
    |-----------|-------|
    | Min | {cc_result.parameters.min():.4f} |
    | Max | {cc_result.parameters.max():.4f} |
    | Mean | {cc_result.parameters.mean():.4f} |
    | Std | {cc_result.parameters.std():.4f} |

    **Method:** `{cc_result.metadata.get("method", "unknown")}`

    {mode_info}
    """)
    return (mode,)


@app.cell
def _(alt, cc_result, mo, pl, t):
    # Visualize the cell cycle variable
    theta = cc_result.parameters

    cc_df = pl.DataFrame({
        "time": t,
        "theta": theta,
    })

    theta_chart = (
        alt.Chart(cc_df)
        .mark_line(strokeWidth=2, color="#e74c3c")
        .encode(
            x=alt.X("time:Q", title="Time"),
            y=alt.Y("theta:Q", title="Cell Cycle Variable (theta)", scale=alt.Scale(domain=[0, 1])),
        )
        .properties(width=600, height=250, title="Cell Cycle Variable Over Time")
    )

    # Add horizontal lines at 0, 0.5, 1
    hlines = (
        alt.Chart(pl.DataFrame({"y": [0.0, 0.5, 1.0]}))
        .mark_rule(
            strokeDash=[5, 5],
            opacity=0.3,
        )
        .encode(y="y:Q")
    )

    mo.md("""
    ### Cell Cycle Variable (theta)

    The computed cell cycle variable theta:
    - **0** = cell birth
    - **0.5** = mid-cycle
    - **1** = cell division

    The oscillations show the cell progressing through its cycle repeatedly.
    """)
    return hlines, theta_chart


@app.cell
def _(hlines, theta_chart):
    theta_chart + hlines
    return


@app.cell
def _(alt, cc_result, mo, pl):
    # Histogram of cell cycle variable
    theta_hist_df = pl.DataFrame({"theta": cc_result.parameters})

    theta_hist = (
        alt.Chart(theta_hist_df)
        .mark_bar(opacity=0.7, color="#3498db")
        .encode(
            x=alt.X("theta:Q", bin=alt.Bin(maxbins=20), title="Cell Cycle Variable (theta)"),
            y=alt.Y("count():Q", title="Count"),
        )
        .properties(width=500, height=250, title="Distribution of Cell Cycle Variable")
    )

    mo.md("""
    ### Distribution of theta

    A uniform distribution indicates good coverage of the cell cycle.
    Peaks indicate where cells spend more time (e.g., G1 phase).
    """)
    return (theta_hist,)


@app.cell
def _(theta_hist):
    theta_hist
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Bin into Discrete Stages (Optional)

    For aggregation strategy #4 (RFC006), bin the continuous theta
    into discrete cell cycle stages.
    """)
    return


@app.cell
def _(alt, cc_result, mo, pl):
    # Bin into discrete stages
    n_bins = 10
    stage_bins = cc_result.to_stage_bins(n_bins=n_bins)

    stage_df = pl.DataFrame({
        "stage": stage_bins,
        "theta": cc_result.parameters,
    })

    # Count per stage
    stage_counts = stage_df.group_by("stage").agg(pl.count().alias("count")).sort("stage")

    stage_bar = (
        alt.Chart(stage_counts)
        .mark_bar(color="#2ecc71")
        .encode(
            x=alt.X("stage:O", title="Cell Cycle Stage"),
            y=alt.Y("count:Q", title="Count"),
        )
        .properties(width=500, height=250, title=f"Data Points per Cell Cycle Stage (n_bins={n_bins})")
    )

    mo.md(f"""
    ### Discrete Cell Cycle Stages

    **`cc_result.to_stage_bins(n_bins={n_bins})`** returns integer stage indices.

    - Stage 0 = theta in [0.0, 0.1)
    - Stage 1 = theta in [0.1, 0.2)
    - ...
    - Stage 9 = theta in [0.9, 1.0]

    This enables aggregation by cell cycle stage for phenotypic sensitivity analysis.
    """)
    return (stage_bar,)


@app.cell
def _(stage_bar):
    stage_bar
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Inspect the Koopman Mode (Advanced)

    The identified cell cycle mode contains rich information about
    the periodic dynamics captured by DMD.
    """)
    return


@app.cell
def _(alt, koopman_cc, mo, mode, np, pl):
    _mode = koopman_cc.cell_cycle_mode

    if _mode is not None:
        # Eigenvalue in complex plane
        eigenvalue = mode.eigenvalue

        # Unit circle
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        circle_df = pl.DataFrame({
            "x": np.cos(theta_circle),
            "y": np.sin(theta_circle),
        })

        unit_circle = (
            alt.Chart(circle_df)
            .mark_line(
                strokeDash=[5, 5],
                color="gray",
                opacity=0.5,
            )
            .encode(x="x:Q", y="y:Q")
        )

        eig_df = pl.DataFrame({
            "real": [eigenvalue.real],
            "imag": [eigenvalue.imag],
            "magnitude": [np.abs(eigenvalue)],
        })

        eig_point = (
            alt.Chart(eig_df)
            .mark_point(
                size=300,
                color="#e74c3c",
                filled=True,
            )
            .encode(
                x=alt.X("real:Q", title="Real", scale=alt.Scale(domain=[-1.5, 1.5])),
                y=alt.Y("imag:Q", title="Imaginary", scale=alt.Scale(domain=[-1.5, 1.5])),
                tooltip=["real", "imag", "magnitude"],
            )
        )

        eig_label = (
            alt.Chart(eig_df)
            .mark_text(
                dx=15,
                dy=-10,
                fontSize=12,
            )
            .encode(
                x="real:Q",
                y="imag:Q",
                text=alt.value("Cell Cycle Mode"),
            )
        )

        eigenvalue_chart = (unit_circle + eig_point + eig_label).properties(
            width=350, height=350, title="Cell Cycle Mode Eigenvalue"
        )

        display_chart = eigenvalue_chart
        interpretation = f"""
    **Eigenvalue interpretation:**
    - **Magnitude |lambda| = {np.abs(eigenvalue):.4f}**: {"< 1 (decaying)" if np.abs(eigenvalue) < 1 else ">= 1 (persistent/growing)"}
    - **Angle = {np.angle(eigenvalue):.4f} rad**: Determines oscillation frequency
    - **On unit circle**: Persistent oscillation (cell cycle!)
    """
    else:
        display_chart = None
        interpretation = "No cell cycle mode was identified."

    mo.md(f"""
    ### Cell Cycle Mode Eigenvalue

    The eigenvalue lambda determines the mode's dynamics:
    - **|lambda| < 1**: Mode decays over time
    - **|lambda| = 1**: Mode persists (perfect oscillation)
    - **|lambda| > 1**: Mode grows over time

    {interpretation}
    """)
    return (display_chart,)


@app.cell
def _(display_chart):
    display_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Complete Code Summary

    Here's the minimal code to compute the cell cycle variable:

    ```python
    import numpy as np
    import polars as pl
    from uq import KoopmanCellCycleVariable

    # Your timeseries data: shape (n_timesteps, n_observables)
    X = np.array(...)  # e.g., from vEcoli simulation

    # Convert to DataFrame with required columns
    data_df = pl.DataFrame({
        "dry_mass": X[:, 0],
        "cell_mass": X[:, 1],
        "time": np.arange(len(X)).astype(float),
        "generation": np.zeros(len(X), dtype=int),
        "agent_id": ["cell_0"] * len(X),
    })

    # Create Koopman cell cycle variable computer
    koopman_cc = KoopmanCellCycleVariable(
        expected_cycle_time=3600.0,  # seconds (1 hour)
        dt=1.0,                       # timestep
    )

    # Compute cell cycle variable
    result = koopman_cc.compute(data_df)

    # Access the scalar cell cycle variable
    theta = result.values  # numpy array, theta in [0, 1]

    # Bin into discrete stages (optional)
    stages = result.to_stage_bins(n_bins=10)  # integer stage indices
    ```

    **Key outputs:**
    - `theta`: Scalar cell cycle variable, shape `(n_timesteps,)`, range `[0, 1]`
    - `stages`: Discrete stage indices, shape `(n_timesteps,)`, range `[0, n_bins-1]`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## What Makes This Impressive

    1. **Data-driven**: No assumptions about growth mechanism, DNA replication timing, etc.

    2. **Automatic frequency detection**: DMD finds the cell cycle oscillation automatically

    3. **Principled math**: Koopman eigenfunctions are the "right" coordinates for periodic systems

    4. **Scalar output**: Reduces high-dimensional state to single theta in [0, 1]

    5. **Generalizable**: Same approach works for any quasi-periodic biological process

    The cell cycle variable is extracted purely from the data's spectral structure -
    this is the novel contribution that RFC006 calls for.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## RFC006-Compliant: GSA-Informed Cell Cycle Variable

    Per RFC006 Section 3: *"The choice of the 'cell cycle variable' will be informed by the sensitivity analyses (1-3)."*

    The `GSAInformedCellCycleVariable` class implements this requirement by:
    1. Analyzing variance decomposition from strategies 1-3
    2. Identifying observables with high **residual variance** (cell-cycle-related)
    3. Using those observables to compute the Koopman cell cycle variable

    This closes the loop between GSA and the cell cycle variable as required by RFC006.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### GSA-Informed Workflow

    ```python
    from uq import (
        GSAInformedCellCycleVariable,
        Aggregator,
        AggregationStrategy,
        identify_cell_cycle_relevant_observables,
    )

    # Step 1: Run aggregation for strategies 1-3
    aggregator = Aggregator(conn, history_sql, config_sql)
    agg_uniform, _ = aggregator.aggregate_transcriptome(AggregationStrategy.UNIFORM)
    agg_by_gen, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_GENERATION)
    agg_by_seed, _ = aggregator.aggregate_transcriptome(AggregationStrategy.BY_LINEAGE_SEED)

    # Step 2: Create GSA-informed cell cycle variable
    gsa_cc = GSAInformedCellCycleVariable(
        aggregated_uniform=agg_uniform,
        aggregated_by_gen=agg_by_gen,
        aggregated_by_seed=agg_by_seed,
        observable_names=observable_names,
        expected_cycle_time=3600.0,  # Expected cell cycle in seconds
    )

    # Step 3: Compute - automatically selects relevant observables
    cc_var = gsa_cc.compute(trajectory_data)

    # Step 4: Inspect GSA selection results
    print(f"Selected observables: {gsa_cc.selected_observables}")
    print(f"Relevance summary: {gsa_cc.get_relevance_summary()}")

    # The result includes GSA metadata
    print(f"GSA informed: {cc_var.metadata['gsa_informed']}")
    print(f"Relevance scores: {cc_var.metadata['relevance_scores']}")
    ```

    ### How it works

    ```
    Total Variance = Generation Variance + Seed Variance + Residual Variance
                                                            ↑
                                                     Cell-cycle-related!

    Observables with high residual variance → selected for Koopman CC
    ```

    ### Key Classes

    | Class | Purpose |
    |-------|---------|
    | `GSAInformedCellCycleVariable` | RFC006-compliant cell cycle variable |
    | `CellCycleRelevanceResult` | Results from GSA relevance analysis |
    | `identify_cell_cycle_relevant_observables()` | Identify CC-relevant observables |
    | `run_gsa_informed_cell_cycle_analysis()` | All-in-one workflow function |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Summary

    | Approach | When to Use |
    |----------|-------------|
    | **`GSAInformedCellCycleVariable`** | RFC006-compliant analysis (production) |
    | **`KoopmanCellCycleVariable`** | Quick prototyping, manual observable selection |
    | **Heuristic methods** | Comparison, specialized use cases |

    For full RFC006 compliance, use `GSAInformedCellCycleVariable` which ensures
    the cell cycle variable choice is informed by sensitivity analyses (1-3).
    """)
    return


if __name__ == "__main__":
    app.run()
