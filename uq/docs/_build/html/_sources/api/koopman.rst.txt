Koopman Spectral Analysis (``uq.koopman``)
==========================================

.. module:: uq.koopman
   :synopsis: Koopman operator methods for spectral analysis of simulation dynamics

The ``koopman`` module provides Koopman operator methods as a complementary approach
to PCE-based sensitivity analysis. Where PCE builds polynomial surrogates, Koopman
analysis extracts the fundamental dynamical modes of the system—providing insight into
oscillatory behaviors, growth dynamics, and cell cycle periodicity.

Overview
--------

The Koopman operator is an infinite-dimensional linear operator that governs the evolution
of observable functions in a dynamical system. By approximating its spectrum using
Dynamic Mode Decomposition (DMD), we can:

- Identify dominant frequencies and growth/decay rates
- Understand how perturbations affect system dynamics
- Extract cell cycle harmonics from simulation trajectories
- Characterize the "musical" structure of whole-cell dynamics

Data Classes
------------

.. autoclass:: KoopmanMode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: KoopmanSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

Dictionary Functions
--------------------

.. autoclass:: KoopmanDictionary
   :members:
   :undoc-members:
   :show-inheritance:

DMD Methods
-----------

Dynamic Mode Decomposition (DMD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DynamicModeDecomposition
   :members:
   :undoc-members:
   :show-inheritance:

Extended DMD (EDMD)
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExtendedDMD
   :members:
   :undoc-members:
   :show-inheritance:

Sensitivity Analysis
--------------------

.. autoclass:: KoopmanSensitivityAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Cell Cycle Analysis
-------------------

.. autoclass:: CellCycleKoopmanAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: extract_koopman_features

Mathematical Background
-----------------------

The Koopman Operator
^^^^^^^^^^^^^^^^^^^^

For a discrete-time dynamical system :math:`x_{k+1} = F(x_k)`, the Koopman operator
:math:`\mathcal{K}` acts on observable functions :math:`g: \mathcal{X} \to \mathbb{C}`:

.. math::

   \mathcal{K}g(x) = g(F(x))

Despite the nonlinearity of :math:`F`, the Koopman operator is linear, enabling
spectral analysis of nonlinear dynamics.

Dynamic Mode Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^

DMD approximates the Koopman operator from data. Given snapshots :math:`X = [x_1, \ldots, x_{m-1}]`
and :math:`Y = [x_2, \ldots, x_m]`, DMD finds:

.. math::

   A = Y X^{\dagger}

where :math:`X^{\dagger}` is the pseudoinverse. The eigenvalues :math:`\lambda_j` of
:math:`A` approximate Koopman eigenvalues, yielding:

- **Frequencies**: :math:`\omega_j = \text{arg}(\lambda_j) / \Delta t`
- **Growth rates**: :math:`\gamma_j = \log|\lambda_j| / \Delta t`
- **Modes**: :math:`\phi_j` (eigenvectors of :math:`A`)

Extended DMD
^^^^^^^^^^^^

EDMD lifts the state space using dictionary functions :math:`\psi(x) = [\psi_1(x), \ldots, \psi_N(x)]`:

.. math::

   \mathcal{K} \approx G^{\dagger} A

where :math:`G_{ij} = \langle \psi_i, \psi_j \rangle` and :math:`A_{ij} = \langle \psi_i, \mathcal{K}\psi_j \rangle`.

Common dictionaries include polynomials, Fourier functions, and radial basis functions.

Interpretation for Cell Biology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In whole-cell simulations, Koopman modes reveal:

1. **Growth modes**: Dominant eigenvalues on the positive real axis indicate exponential growth
2. **Cell cycle harmonics**: Complex eigenvalues with :math:`|\lambda| \approx 1` correspond to
   periodic behavior at cell cycle frequency and its harmonics
3. **Transient modes**: Eigenvalues with :math:`|\lambda| < 1` decay away—these represent
   initial transients that disappear as the simulation converges

Example Usage
-------------

Basic DMD Analysis
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import DynamicModeDecomposition
   import numpy as np

   # Load trajectory: (n_timesteps, n_observables)
   X = np.load("trajectory.npy")

   # Fit DMD
   dmd = DynamicModeDecomposition(rank=10)
   dmd.fit(X)

   # Get spectrum
   spectrum = dmd.get_spectrum(observable_names=["mass", "growth_rate"])

   # Examine modes
   for mode in spectrum.modes[:5]:
       print(f"λ = {mode.eigenvalue:.4f}")
       print(f"  Frequency: {mode.frequency:.4f} Hz")
       print(f"  Decay rate: {mode.decay_rate:.4f}")
       print(f"  Energy: {mode.energy:.4f}")

EDMD with Polynomial Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import ExtendedDMD, KoopmanDictionary

   # Create polynomial + Fourier dictionary
   dictionary = KoopmanDictionary(
       polynomial_degree=3,
       include_fourier=True,
       fourier_terms=5,
   )

   # Fit EDMD
   edmd = ExtendedDMD(dictionary=dictionary, rank=20)
   edmd.fit(X)

   # Reconstruct trajectory
   X_reconstructed = edmd.reconstruct(n_steps=100)

Finding Cell Cycle Modes
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from uq import CellCycleKoopmanAnalyzer

   analyzer = CellCycleKoopmanAnalyzer(
       expected_cycle_time=3600.0,  # 1 hour in seconds
       dt=1.0,
       harmonic_tolerance=0.1,
   )

   cc_modes = analyzer.identify_cell_cycle_modes(spectrum)

   print(f"Found {len(cc_modes)} cell cycle harmonics")
   for mode in cc_modes:
       harmonic = mode.frequency * 3600.0
       print(f"  {harmonic:.1f}× fundamental, energy={mode.energy:.4f}")

See Also
--------

- :doc:`../sensitivity_analysis` - PCE-based sensitivity methods
- :doc:`../cell_cycle` - Cell cycle stratification
- :doc:`../tutorials/koopman_analysis` - Complete tutorial

References
----------

1. Kutz, J.N. et al. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*
2. Williams, M.O. et al. (2015). "A Data-Driven Approximation of the Koopman Operator"
3. Mezić, I. (2013). "Analysis of Fluid Flows via Spectral Properties of the Koopman Operator"
