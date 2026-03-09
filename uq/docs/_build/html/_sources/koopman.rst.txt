Koopman Spectral Analysis
=========================

The UQ framework includes Koopman operator methods as a complementary approach to
PCE-based sensitivity analysis. While PCE builds polynomial surrogates to rank parameter
importance, Koopman analysis extracts the fundamental dynamical modes—the "harmonics"—of
the simulation.

What is Koopman Analysis?
-------------------------

The Koopman operator is an infinite-dimensional linear operator that governs the evolution
of observable functions in a dynamical system. Even for nonlinear dynamics like whole-cell
simulations, the Koopman operator is linear, enabling spectral analysis.

The key insight: **any dynamical system can be decomposed into a sum of oscillating/growing/decaying modes**.

.. math::

   g(x_t) = \sum_{j=1}^{\infty} \phi_j(x_0) \lambda_j^t v_j

where:

- :math:`\lambda_j` are Koopman eigenvalues (complex numbers encoding frequency and growth)
- :math:`\phi_j` are Koopman eigenfunctions (spatial structure of the mode)
- :math:`v_j` are mode coefficients

Why Use Koopman for Whole-Cell Simulations?
-------------------------------------------

Cell biology is inherently periodic. The cell cycle imposes a fundamental rhythm, and
Koopman analysis is ideal for extracting these periodic dynamics:

1. **Cell Cycle Detection**: Find modes at the cell cycle frequency and harmonics
2. **Growth Rate Analysis**: Identify exponential growth modes
3. **Transient Identification**: Separate initial transients from steady-state behavior
4. **Parameter Effects on Dynamics**: See how inputs change system frequencies

Dynamic Mode Decomposition (DMD)
--------------------------------

DMD is a data-driven method to approximate the Koopman operator from simulation snapshots.

Given a trajectory :math:`X = [x_1, x_2, \ldots, x_m]`, DMD finds the best-fit linear
operator :math:`A` such that :math:`x_{k+1} \approx A x_k`.

.. code-block:: python

   from uq import DynamicModeDecomposition

   # X has shape (n_timesteps, n_observables)
   dmd = DynamicModeDecomposition(rank=10)
   dmd.fit(X)

   spectrum = dmd.get_spectrum()

The ``rank`` parameter controls how many modes to extract. Higher rank captures more
detail but may include noise.

Extended DMD (EDMD)
-------------------

For nonlinear dynamics, Extended DMD lifts the observables using dictionary functions:

.. code-block:: python

   from uq import ExtendedDMD, KoopmanDictionary

   dictionary = KoopmanDictionary(
       polynomial_degree=2,    # Include x^2, xy terms
       include_fourier=True,   # Include sin/cos terms
       fourier_terms=3,
   )

   edmd = ExtendedDMD(dictionary=dictionary, rank=20)
   edmd.fit(X)

Available dictionary functions:

- **Polynomials**: Capture saturation and quadratic interactions
- **Fourier**: Ideal for periodic behavior
- **Radial Basis Functions** (RBF): Local nonlinear features

Interpreting the Spectrum
-------------------------

Each Koopman mode has an eigenvalue :math:`\lambda = e^{(\gamma + i\omega)\Delta t}` where:

- :math:`\omega` is the **frequency** (radians per time step)
- :math:`\gamma` is the **growth rate** (positive = growing, negative = decaying)

For whole-cell simulations:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode Type
     - Interpretation
   * - :math:`\lambda \approx 1`, real
     - Steady-state or slow drift (growth mode)
   * - :math:`|\lambda| < 1`, real
     - Decaying transient
   * - :math:`|\lambda| \approx 1`, complex
     - Cell cycle oscillation
   * - High frequency, :math:`|\lambda| < 1`
     - Noise or fast transients

Cell Cycle Harmonics
--------------------

The ``CellCycleKoopmanAnalyzer`` identifies modes corresponding to cell cycle periodicity:

.. code-block:: python

   from uq import CellCycleKoopmanAnalyzer

   analyzer = CellCycleKoopmanAnalyzer(
       expected_cycle_time=2400.0,  # 40 minutes in seconds
       dt=1.0,                       # Simulation time step
       harmonic_tolerance=0.15,      # Allow 15% frequency error
   )

   cc_modes = analyzer.identify_cell_cycle_modes(spectrum)

   for mode in cc_modes:
       harmonic_order = mode.frequency * 2400.0
       print(f"Found {harmonic_order:.1f}× cell cycle harmonic")

Harmonics to look for:

1. **1× fundamental**: Main cell cycle oscillation (mass doubling, division)
2. **2× harmonic**: Often related to DNA replication initiation/termination
3. **Higher harmonics**: Metabolic sub-cycles, gene expression bursts

Spectral Sensitivity
--------------------

Compare spectra between conditions to understand dynamical sensitivity:

.. code-block:: python

   from uq import KoopmanSensitivityAnalyzer

   analyzer = KoopmanSensitivityAnalyzer(rank=10)

   sensitivity = analyzer.spectral_sensitivity(
       X_baseline=X_wt,
       X_perturbed=X_vio_high,
       parameter_names=["vio_expression"],
   )

This reveals:

- **Eigenvalue shift**: Did growth rates or frequencies change?
- **Mode shape change**: Are different genes now dominant in each mode?
- **Frequency perturbation**: Did the cell cycle speed up or slow down?

Comparison with PCE Methods
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - PCE/Sobol
     - Koopman
   * - Question
     - "Which parameters matter?"
     - "What are the dynamics?"
   * - Output
     - Sensitivity indices
     - Frequencies, growth rates, modes
   * - Interpretation
     - Statistical
     - Dynamical systems
   * - For cell cycle
     - Stratify by stage
     - Extract harmonic modes
   * - Computation
     - Requires parameter variations
     - Works on single trajectory

**Use both together**: PCE identifies important parameters, Koopman explains how they
affect the system's dynamical behavior.

Best Practices
--------------

1. **Data preparation**: Normalize observables to similar scales
2. **Rank selection**: Start low (5-10), increase if reconstruction is poor
3. **Time resolution**: Ensure dt resolves the dynamics of interest
4. **Trajectory length**: Longer trajectories give better frequency resolution
5. **Validation**: Check reconstruction error and compare with known biology

See Also
--------

- :doc:`tutorials/koopman_analysis` - Step-by-step tutorial
- :doc:`api/koopman` - Full API reference
- :doc:`sensitivity_analysis` - PCE-based methods
- :doc:`cell_cycle` - Traditional cell cycle stratification
