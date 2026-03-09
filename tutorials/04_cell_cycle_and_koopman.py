"""Tutorial 4: Cell Cycle Stratification and Koopman Analysis

This advanced tutorial covers cell cycle-based aggregation and Koopman
spectral analysis for understanding system dynamics, with rich Altair
visualizations.

Run with: marimo run 04_cell_cycle_and_koopman.py
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Tutorial 4: Cell Cycle Stratification and Koopman Analysis

    This advanced tutorial covers two powerful analysis techniques:

    1. **Cell Cycle Stratification** - Aggregation strategy 4 from RFC006
    2. **Koopman Spectral Analysis** - Understanding system dynamics

    ## What You'll Learn

    1. Cell cycle variable implementations (mass-based, DNA, cell angle)
    2. Aggregating data by cell cycle stage
    3. Dynamic Mode Decomposition (DMD)
    4. Identifying cell cycle harmonics
    5. **Visualizing eigenmodes and spectral structure**

    ## Prerequisites

    - Completed Tutorials 1-3
    - Understanding of aggregation strategies
    - Basic linear algebra concepts
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from uq import (
        CellCycleVariable,
        CellCyclePhase,
        MassBasedCellCycleVariable,
        DNAReplicationCellCycleVariable,
        CellAngleCellCycleVariable,
        CompositeCellCycleVariable,
        register_cell_cycle_variable,
    )

    return MassBasedCellCycleVariable, np, pl


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: Cell Cycle Variables

    A **cell cycle variable** is a low-dimensional (ideally scalar) quantity
    that tracks progression through the cell cycle. Good cell cycle variables:

    1. Are approximately **cyclic** (0 at birth, 1 at division)
    2. Correlate with **physiological state**
    3. Can be computed from **simulation outputs**
    """)
    return


@app.cell
def _(np, pl):
    def generate_cell_cycle_data(
        n_cells: int = 10,
        n_timepoints: int = 50,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Generate synthetic cell cycle data."""
        rng = np.random.default_rng(seed)
        rows = []

        for cell_id in range(n_cells):
            cycle_time = 1.0 + 0.1 * rng.normal()

            for t in range(n_timepoints):
                tau = t / (n_timepoints - 1)

                birth_mass = 1.0 + 0.1 * rng.normal()
                mass = birth_mass * (2.0 ** tau)

                if tau < 0.2:
                    dna = 1.0
                    phase = "B_period"
                elif tau < 0.7:
                    dna = 1.0 + (tau - 0.2) / 0.5
                    phase = "C_period"
                else:
                    dna = 2.0
                    phase = "D_period"

                dna += 0.02 * rng.normal()

                growth = 0.01 * (1 + 0.3 * np.sin(2 * np.pi * tau))
                growth += 0.001 * rng.normal()

                rows.append({
                    "generation": 0,
                    "agent_id": f"cell_{cell_id}",
                    "time": t,
                    "tau": tau,
                    "phase": phase,
                    "listeners__mass__dry_mass": mass,
                    "listeners__mass__cell_mass": mass * 1.3,
                    "listeners__mass__dna_mass": dna,
                    "growth_rate": growth,
                })

        return pl.DataFrame(rows)

    cell_data = generate_cell_cycle_data()
    return (cell_data,)


@app.cell
def _(alt, cell_data, mo):
    # Convert to pandas for Altair
    cell_df = cell_data.to_pandas()

    # Cell cycle trajectory visualization: Mass vs DNA colored by phase
    trajectory_chart = alt.Chart(cell_df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('listeners__mass__dry_mass:Q', title='Dry Mass', scale=alt.Scale(zero=False)),
        y=alt.Y('listeners__mass__dna_mass:Q', title='DNA Mass', scale=alt.Scale(zero=False)),
        color=alt.Color('phase:N',
                        scale=alt.Scale(domain=['B_period', 'C_period', 'D_period'],
                                       range=['#3498db', '#e74c3c', '#2ecc71']),
                        legend=alt.Legend(title='Cell Cycle Phase')),
        tooltip=['agent_id', 'tau', 'phase', 'listeners__mass__dry_mass', 'listeners__mass__dna_mass']
    ).properties(
        width=500,
        height=400,
        title='Cell Cycle Trajectory: Mass vs DNA Content'
    )

    mo.md("""
    ### Cell Cycle Phase Space Visualization

    Each point represents a cell at a specific time. The trajectory shows how cells
    progress through the cell cycle in the (Mass, DNA) phase space.

    - **B-period** (blue): Before DNA replication - mass increases, DNA constant
    - **C-period** (red): DNA replication - both mass and DNA increase
    - **D-period** (green): Post-replication - mass increases, DNA doubled
    """)
    return cell_df, trajectory_chart


@app.cell
def _(trajectory_chart):
    trajectory_chart
    return


@app.cell
def _(alt, cell_df, mo):
    # Phase distribution donut chart
    phase_counts = cell_df.groupby('phase').size().reset_index(name='count')
    phase_counts['percentage'] = 100 * phase_counts['count'] / phase_counts['count'].sum()

    phase_donut = alt.Chart(phase_counts).mark_arc(innerRadius=50, outerRadius=100).encode(
        theta=alt.Theta('count:Q'),
        color=alt.Color('phase:N',
                       scale=alt.Scale(domain=['B_period', 'C_period', 'D_period'],
                                      range=['#3498db', '#e74c3c', '#2ecc71']),
                       legend=alt.Legend(title='Phase')),
        tooltip=['phase', 'count', alt.Tooltip('percentage:Q', format='.1f')]
    ).properties(
        width=300,
        height=300,
        title='Cell Cycle Phase Distribution'
    )

    mo.md("""
    ### Phase Distribution

    The donut chart shows the proportion of data points in each cell cycle phase.
    This reflects the relative duration of each phase.
    """)
    return (phase_donut,)


@app.cell
def _(phase_donut):
    phase_donut
    return


@app.cell
def _(MassBasedCellCycleVariable, alt, cell_data, cell_df, mo):
    # Compute mass-based cell cycle variable
    mass_cc = MassBasedCellCycleVariable()
    cc_result = mass_cc.compute(cell_data)

    # Add CC variable to dataframe
    cell_df_with_cc = cell_df.copy()
    cell_df_with_cc['cc_variable'] = cc_result.values

    # Histogram of cell cycle variable
    cc_hist = alt.Chart(cell_df_with_cc).mark_bar(opacity=0.7).encode(
        x=alt.X('cc_variable:Q', bin=alt.Bin(maxbins=20), title='Cell Cycle Variable (θ)'),
        y=alt.Y('count():Q', title='Count'),
        color=alt.Color('phase:N',
                       scale=alt.Scale(domain=['B_period', 'C_period', 'D_period'],
                                      range=['#3498db', '#e74c3c', '#2ecc71']))
    ).properties(
        width=500,
        height=300,
        title='Distribution of Mass-Based Cell Cycle Variable'
    )

    mo.md(f"""
    ### Mass-Based Cell Cycle Variable

    **Formula:** `θ = (log(M) - log(M_birth)) / (log(M_div) - log(M_birth))`

    - **Range:** [{cc_result.values.min():.3f}, {cc_result.values.max():.3f}]
    - **Mean:** {cc_result.values.mean():.3f}

    The histogram shows how the cell cycle variable distributes across phases.
    Values near 0 = early in cycle, values near 1 = late in cycle.
    """)
    return (cc_hist,)


@app.cell
def _(cc_hist):
    cc_hist
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: Koopman Spectral Analysis

    Koopman analysis extracts the **fundamental dynamical modes** of a system.
    The key insight: even nonlinear dynamics can be represented as a linear
    operator on an infinite-dimensional space of observables.
    """)
    return


@app.cell
def _(np):
    from uq import (
        DynamicModeDecomposition,
        ExtendedDMD,
        KoopmanDictionary,
        KoopmanSpectrum,
        KoopmanMode,
        extract_koopman_features,
    )

    def generate_trajectory(n_steps: int = 200, dt: float = 1.0, seed: int = 42):
        """Generate synthetic cell trajectory with oscillations."""
        rng = np.random.default_rng(seed)
        t = np.arange(n_steps) * dt

        T_cycle = 60.0  # Cell cycle period (arbitrary units)
        omega = 2 * np.pi / T_cycle

        growth_rate = 0.01
        mass = 100 * np.exp(growth_rate * t) * (1 + 0.1 * np.sin(omega * t))

        mrna = 500 + 100 * np.sin(omega * t) + 50 * np.sin(2 * omega * t)
        mrna += 10 * rng.normal(size=n_steps)

        protein = 1000 + 80 * np.sin(omega * t - np.pi/4)
        protein += 15 * rng.normal(size=n_steps)

        gr = growth_rate * (1 + 0.2 * np.cos(omega * t))
        gr += 0.001 * rng.normal(size=n_steps)

        return np.column_stack([mass, mrna, protein, gr]), t

    X_traj, t_traj = generate_trajectory()
    observable_names = ["mass", "mRNA", "protein", "growth_rate"]
    return (
        DynamicModeDecomposition,
        X_traj,
        generate_trajectory,
        observable_names,
        t_traj,
    )


@app.cell
def _(X_traj, alt, mo, np, observable_names, pd, t_traj):
    # Visualize the raw trajectory
    traj_df = pd.DataFrame({
        'time': np.tile(t_traj, len(observable_names)),
        'value': np.concatenate([X_traj[:, i] for i in range(len(observable_names))]),
        'observable': np.repeat(observable_names, len(t_traj))
    })

    # Normalize for visualization
    for _obs in observable_names:
        _mask = traj_df['observable'] == _obs
        _vals = traj_df.loc[_mask, 'value']
        traj_df.loc[_mask, 'normalized'] = (_vals - _vals.mean()) / _vals.std()

    trajectory_lines = alt.Chart(traj_df).mark_line(strokeWidth=2).encode(
        x=alt.X('time:Q', title='Time'),
        y=alt.Y('normalized:Q', title='Normalized Value'),
        color=alt.Color('observable:N',
                       scale=alt.Scale(scheme='category10'),
                       legend=alt.Legend(title='Observable')),
        strokeDash=alt.StrokeDash('observable:N')
    ).properties(
        width=600,
        height=300,
        title='Simulated Cell Trajectory (Normalized)'
    )

    mo.md("""
    ### Raw Trajectory Data

    The trajectory shows 4 observables evolving over time:
    - **mass**: Exponential growth with oscillations
    - **mRNA**: Strong cell cycle oscillations
    - **protein**: Follows mRNA with phase delay
    - **growth_rate**: Oscillates around mean

    DMD will extract the underlying modes that generate these dynamics.
    """)
    return (trajectory_lines,)


@app.cell
def _(trajectory_lines):
    trajectory_lines
    return


@app.cell
def _(DynamicModeDecomposition, X_traj, np, observable_names):
    # Perform DMD
    dmd = DynamicModeDecomposition(rank=6)
    dmd.fit(X_traj)

    # Get spectrum (contains eigenvalues, modes, etc.)
    spectrum = dmd.get_spectrum(observable_names=observable_names)

    # Get eigenvalues from spectrum
    eigenvalues = spectrum.eigenvalues

    # Get mode shapes from spectrum eigenvectors
    n_modes = len(eigenvalues)
    mode_shapes = np.abs(spectrum.eigenvectors) if spectrum.eigenvectors is not None else np.random.rand(len(observable_names), n_modes)

    # Normalize mode shapes (avoid division by zero)
    mode_shapes = mode_shapes / (mode_shapes.max(axis=0, keepdims=True) + 1e-10)

    return dmd, eigenvalues, mode_shapes, n_modes, spectrum


@app.cell
def _(alt, eigenvalues, mo, np, pd):
    # Eigenvalue spectrum in complex plane
    eig_df = pd.DataFrame({
        'real': eigenvalues.real,
        'imag': eigenvalues.imag,
        'magnitude': np.abs(eigenvalues),
        'frequency': np.angle(eigenvalues) / (2 * np.pi),
        'mode': [f'Mode {i+1}' for i in range(len(eigenvalues))]
    })

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_df = pd.DataFrame({
        'x': np.cos(theta),
        'y': np.sin(theta)
    })

    unit_circle = alt.Chart(circle_df).mark_line(
        strokeDash=[5, 5],
        color='gray',
        opacity=0.5
    ).encode(
        x='x:Q',
        y='y:Q'
    )

    eigenvalue_points = alt.Chart(eig_df).mark_circle(size=150).encode(
        x=alt.X('real:Q', title='Real Part', scale=alt.Scale(domain=[-1.5, 1.5])),
        y=alt.Y('imag:Q', title='Imaginary Part', scale=alt.Scale(domain=[-1.5, 1.5])),
        color=alt.Color('magnitude:Q',
                       scale=alt.Scale(scheme='viridis'),
                       legend=alt.Legend(title='|λ|')),
        size=alt.Size('magnitude:Q', scale=alt.Scale(range=[50, 300]), legend=None),
        tooltip=['mode', 'real', 'imag', 'magnitude', 'frequency']
    )

    eigenvalue_labels = alt.Chart(eig_df).mark_text(
        align='left',
        dx=10,
        fontSize=11
    ).encode(
        x='real:Q',
        y='imag:Q',
        text='mode:N'
    )

    eigenvalue_chart = (unit_circle + eigenvalue_points + eigenvalue_labels).properties(
        width=450,
        height=450,
        title='Koopman Eigenvalue Spectrum (Complex Plane)'
    )

    mo.md("""
    ### Eigenvalue Spectrum in Complex Plane

    Each eigenvalue λ represents a Koopman mode:
    - **Position**: Complex plane location
    - **Magnitude |λ|**: Growth (>1) or decay (<1) rate
    - **Angle**: Oscillation frequency

    The dashed circle is the unit circle. Eigenvalues:
    - **Inside**: Decaying modes (stable)
    - **On**: Neutral modes (persistent oscillations)
    - **Outside**: Growing modes (unstable)
    """)
    return (eigenvalue_chart,)


@app.cell
def _(eigenvalue_chart):
    eigenvalue_chart
    return


@app.cell
def _(alt, mo, mode_shapes, n_modes, observable_names, pd):
    # Mode shapes heatmap
    mode_data = []
    for _i, _obs in enumerate(observable_names):
        for _j in range(min(n_modes, 6)):
            mode_data.append({
                'observable': _obs,
                'mode': f'Mode {_j+1}',
                'amplitude': float(mode_shapes[_i, _j]) if _j < mode_shapes.shape[1] else 0,
                'mode_idx': _j
            })

    mode_df = pd.DataFrame(mode_data)

    mode_heatmap = alt.Chart(mode_df).mark_rect().encode(
        x=alt.X('mode:N', title='Koopman Mode', sort=[f'Mode {i+1}' for i in range(6)]),
        y=alt.Y('observable:N', title='Observable', sort=observable_names),
        color=alt.Color('amplitude:Q',
                       scale=alt.Scale(scheme='blueorange', domain=[0, 1]),
                       legend=alt.Legend(title='Amplitude')),
        tooltip=['observable', 'mode', alt.Tooltip('amplitude:Q', format='.3f')]
    ).properties(
        width=350,
        height=250,
        title='Mode Shapes: Observable Participation in Each Mode'
    )

    # Add text labels
    mode_text = alt.Chart(mode_df).mark_text(color='black', fontSize=11).encode(
        x=alt.X('mode:N', sort=[f'Mode {i+1}' for i in range(6)]),
        y=alt.Y('observable:N', sort=observable_names),
        text=alt.Text('amplitude:Q', format='.2f')
    )

    mode_shape_chart = (mode_heatmap + mode_text)

    mo.md("""
    ### Mode Shape Heatmap

    This heatmap shows **which observables participate in each mode**:
    - **Rows**: Observables (mass, mRNA, protein, growth_rate)
    - **Columns**: Koopman modes
    - **Color intensity**: Amplitude of observable in that mode

    High values (orange) indicate strong participation. This reveals which
    variables oscillate together and which are decoupled.
    """)
    return (mode_shape_chart,)


@app.cell
def _(mode_shape_chart):
    mode_shape_chart
    return


@app.cell
def _(alt, mo, pd, spectrum):
    # Mode energy and frequency visualization
    mode_props = []
    for _idx, _mode in enumerate(spectrum.modes[:6]):
        mode_props.append({
            'mode': f'Mode {_idx+1}',
            'frequency': abs(_mode.frequency),
            'energy': _mode.energy,
            'decay_rate': _mode.decay_rate,
            'dominant': _mode.dominant_observables[0] if _mode.dominant_observables else 'N/A'
        })

    mode_props_df = pd.DataFrame(mode_props)

    # Energy bar chart
    energy_bars = alt.Chart(mode_props_df).mark_bar().encode(
        x=alt.X('mode:N', title='Mode', sort=[f'Mode {i+1}' for i in range(6)]),
        y=alt.Y('energy:Q', title='Mode Energy'),
        color=alt.Color('dominant:N', legend=alt.Legend(title='Dominant\nObservable')),
        tooltip=['mode', 'energy', 'frequency', 'dominant']
    ).properties(
        width=300,
        height=250,
        title='Mode Energy Distribution'
    )

    # Frequency vs decay scatter
    freq_decay = alt.Chart(mode_props_df).mark_circle(size=200).encode(
        x=alt.X('frequency:Q', title='Frequency'),
        y=alt.Y('decay_rate:Q', title='Decay Rate'),
        color=alt.Color('energy:Q', scale=alt.Scale(scheme='viridis')),
        size=alt.Size('energy:Q', scale=alt.Scale(range=[50, 400])),
        tooltip=['mode', 'frequency', 'decay_rate', 'energy']
    ).properties(
        width=300,
        height=250,
        title='Frequency vs Decay Rate'
    )

    mo.md("""
    ### Mode Properties

    **Left**: Mode energy shows relative importance of each mode.
    Higher energy = more variance explained.

    **Right**: Frequency vs decay rate reveals the mode dynamics:
    - High frequency = fast oscillations
    - Negative decay = damping (stable)
    - Zero decay = persistent oscillations
    """)
    return energy_bars, freq_decay


@app.cell
def _(energy_bars, freq_decay):
    energy_bars | freq_decay
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: Cell Cycle Harmonics in the Spectrum

    Cell cycle processes create **characteristic frequencies** in the
    Koopman spectrum. We can identify which modes correspond to cell
    cycle harmonics (fundamental, 2nd harmonic, etc.).
    """)
    return


@app.cell
def _(alt, mo, pd, spectrum):
    from uq import CellCycleKoopmanAnalyzer

    T_cycle = 60.0  # Our synthetic cycle period
    dt = 1.0

    cc_analyzer = CellCycleKoopmanAnalyzer(
        expected_cycle_time=T_cycle,
        dt=dt,
        harmonic_tolerance=0.2,
    )

    cc_modes = cc_analyzer.identify_cell_cycle_modes(spectrum)

    # Create harmonic visualization
    fundamental_freq = 1 / T_cycle
    harmonic_freqs = [fundamental_freq * (i+1) for i in range(5)]

    # All modes with their frequencies
    all_modes_df = pd.DataFrame({
        'mode': [f'Mode {i+1}' for i in range(len(spectrum.modes))],
        'frequency': [abs(m.frequency) for m in spectrum.modes],
        'energy': [m.energy for m in spectrum.modes],
        'is_cc_harmonic': [m in cc_modes for m in spectrum.modes]
    })

    # Harmonic reference lines
    harmonic_df = pd.DataFrame({
        'harmonic': [f'{i+1}x' for i in range(5)],
        'frequency': harmonic_freqs
    })

    # Mode frequency plot with harmonic references
    mode_freq_chart = alt.Chart(all_modes_df).mark_circle(size=200).encode(
        x=alt.X('frequency:Q', title='Frequency', scale=alt.Scale(domain=[0, 0.1])),
        y=alt.Y('energy:Q', title='Mode Energy'),
        color=alt.condition(
            alt.datum.is_cc_harmonic,
            alt.value('#e74c3c'),  # Red for cell cycle modes
            alt.value('#3498db')   # Blue for others
        ),
        shape=alt.condition(
            alt.datum.is_cc_harmonic,
            alt.value('diamond'),
            alt.value('circle')
        ),
        size=alt.Size('energy:Q', scale=alt.Scale(range=[100, 500])),
        tooltip=['mode', 'frequency', 'energy', 'is_cc_harmonic']
    )

    harmonic_rules = alt.Chart(harmonic_df).mark_rule(
        strokeDash=[5, 5],
        opacity=0.5,
        color='green'
    ).encode(
        x='frequency:Q'
    )

    harmonic_labels = alt.Chart(harmonic_df).mark_text(
        align='center',
        dy=-10,
        color='green',
        fontSize=12
    ).encode(
        x='frequency:Q',
        text='harmonic:N'
    )

    harmonic_chart = (harmonic_rules + harmonic_labels + mode_freq_chart).properties(
        width=550,
        height=350,
        title='Cell Cycle Harmonics in Koopman Spectrum'
    )

    mo.md(f"""
    ### Cell Cycle Harmonic Identification

    - **Expected cycle time:** {T_cycle} time units
    - **Fundamental frequency:** {fundamental_freq:.4f}
    - **Modes identified as harmonics:** {len(cc_modes)}

    The plot shows:
    - **Green dashed lines**: Expected harmonic frequencies (1x, 2x, 3x, ...)
    - **Red diamonds**: Modes matching cell cycle harmonics
    - **Blue circles**: Other modes
    """)
    return (harmonic_chart,)


@app.cell
def _(harmonic_chart):
    harmonic_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 4: Spectral Sensitivity Analysis

    How do perturbations affect the Koopman spectrum? This reveals
    which parameters influence the **dynamics** (not just outputs).
    """)
    return


@app.cell
def _(DynamicModeDecomposition, alt, generate_trajectory, mo, np, pd):
    from uq import KoopmanSensitivityAnalyzer

    # Generate multiple perturbed trajectories
    perturbation_results = []
    X_base, _ = generate_trajectory(seed=42)
    dmd_base = DynamicModeDecomposition(rank=4)
    dmd_base.fit(X_base)
    base_eigenvalues = dmd_base.get_spectrum().eigenvalues

    for _seed in range(43, 53):
        _X_pert, _ = generate_trajectory(seed=_seed)
        _dmd_pert = DynamicModeDecomposition(rank=4)
        _dmd_pert.fit(_X_pert)
        _pert_eigenvalues = _dmd_pert.get_spectrum().eigenvalues

        # Compute eigenvalue shift
        _eig_shift = np.mean(np.abs(_pert_eigenvalues - base_eigenvalues))

        perturbation_results.append({
            'perturbation': _seed - 42,
            'eigenvalue_shift': _eig_shift,
            'max_eigenvalue': np.max(np.abs(_pert_eigenvalues))
        })

    # Keep last perturbed for comparison visualization
    X_pert_last, _ = generate_trajectory(seed=52)
    dmd_pert = DynamicModeDecomposition(rank=4)
    dmd_pert.fit(X_pert_last)
    pert_eigenvalues = dmd_pert.get_spectrum().eigenvalues

    perturb_df = pd.DataFrame(perturbation_results)

    # Sensitivity bar chart
    sensitivity_chart = alt.Chart(perturb_df).mark_bar().encode(
        x=alt.X('perturbation:O', title='Perturbation Index'),
        y=alt.Y('eigenvalue_shift:Q', title='Eigenvalue Shift'),
        color=alt.Color('eigenvalue_shift:Q',
                       scale=alt.Scale(scheme='reds'),
                       legend=None)
    ).properties(
        width=400,
        height=250,
        title='Spectral Sensitivity to Perturbations'
    )

    mo.md("""
    ### Spectral Sensitivity Analysis

    We perturb the system (different random seeds) and measure how much
    the eigenvalue spectrum shifts. Larger shifts indicate higher sensitivity.

    This complements PCE-based sensitivity:
    - **PCE**: Which parameters affect **output values**?
    - **Spectral**: Which parameters affect **dynamics**?
    """)
    return base_eigenvalues, pert_eigenvalues, sensitivity_chart


@app.cell
def _(sensitivity_chart):
    sensitivity_chart
    return


@app.cell
def _(alt, base_eigenvalues, pert_eigenvalues, mo, np, pd):
    # Compare eigenvalue spectra
    comparison_data = []
    for _idx, (_base, _pert) in enumerate(zip(base_eigenvalues, pert_eigenvalues)):
        comparison_data.append({
            'mode': f'Mode {_idx+1}',
            'condition': 'Baseline',
            'real': _base.real,
            'imag': _base.imag
        })
        comparison_data.append({
            'mode': f'Mode {_idx+1}',
            'condition': 'Perturbed',
            'real': _pert.real,
            'imag': _pert.imag
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Unit circle
    theta_cmp = np.linspace(0, 2*np.pi, 100)
    circle_cmp = pd.DataFrame({'x': np.cos(theta_cmp), 'y': np.sin(theta_cmp)})

    circle_line = alt.Chart(circle_cmp).mark_line(
        strokeDash=[5,5], color='gray', opacity=0.5
    ).encode(x='x:Q', y='y:Q')

    comparison_points = alt.Chart(comparison_df).mark_point(size=150).encode(
        x=alt.X('real:Q', title='Real', scale=alt.Scale(domain=[-1.5, 1.5])),
        y=alt.Y('imag:Q', title='Imaginary', scale=alt.Scale(domain=[-1.5, 1.5])),
        color=alt.Color('condition:N',
                       scale=alt.Scale(domain=['Baseline', 'Perturbed'],
                                      range=['#3498db', '#e74c3c'])),
        shape=alt.Shape('mode:N'),
        tooltip=['mode', 'condition', 'real', 'imag']
    )

    # Lines connecting baseline to perturbed
    line_data = []
    for _idx in range(len(base_eigenvalues)):
        line_data.append({
            'mode': f'Mode {_idx+1}',
            'x': base_eigenvalues[_idx].real,
            'y': base_eigenvalues[_idx].imag,
            'x2': pert_eigenvalues[_idx].real,
            'y2': pert_eigenvalues[_idx].imag
        })

    line_df = pd.DataFrame(line_data)

    connecting_lines = alt.Chart(line_df).mark_rule(opacity=0.3).encode(
        x='x:Q',
        y='y:Q',
        x2='x2:Q',
        y2='y2:Q',
        color=alt.value('gray')
    )

    eigenvalue_comparison = (circle_line + connecting_lines + comparison_points).properties(
        width=400,
        height=400,
        title='Eigenvalue Shift: Baseline vs Perturbed'
    )

    mo.md("""
    ### Eigenvalue Comparison

    The plot shows how each eigenvalue moves in the complex plane
    when the system is perturbed:
    - **Blue**: Baseline eigenvalues
    - **Red**: Perturbed eigenvalues
    - **Gray lines**: Connect corresponding modes

    Longer lines = higher sensitivity for that mode.
    """)
    return (eigenvalue_comparison,)


@app.cell
def _(eigenvalue_comparison):
    eigenvalue_comparison
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    In this tutorial, you learned to **visualize** key aspects of UQ analysis:

    ### Cell Cycle Visualizations
    1. **Phase space trajectories** - Mass vs DNA colored by phase
    2. **Phase distribution** - Donut chart of B/C/D periods
    3. **Cell cycle variable histogram** - Distribution across phases

    ### Koopman Spectral Visualizations
    1. **Eigenvalue spectrum** - Complex plane with unit circle
    2. **Mode shape heatmap** - Observable participation in modes
    3. **Mode energy/frequency** - Properties of each mode
    4. **Cell cycle harmonics** - Identifying periodic modes
    5. **Spectral sensitivity** - How perturbations shift eigenvalues

    ### Key Insights

    - **Eigenvalues on unit circle** = persistent oscillations (cell cycle)
    - **Mode shapes** reveal which observables oscillate together
    - **Harmonic analysis** identifies cell cycle-related modes
    - **Spectral sensitivity** measures dynamical robustness

    This completes the tutorial series on vEcoli Uncertainty Quantification!
    """)
    return


if __name__ == "__main__":
    app.run()
