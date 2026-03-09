Koopman Spectral Analysis Tutorial
===================================

This tutorial introduces Koopman spectral analysis as a complementary approach
to PCE-based sensitivity analysis. You'll learn how to extract dynamical modes
from simulation trajectories and identify cell cycle harmonics.

Introduction
------------

While PCE-based methods answer "which parameters matter most?", Koopman analysis
answers "what are the dominant dynamics of the system?". Think of it as finding
the "frequencies" or "harmonics" of the whole-cell simulation—similar to how a
Fourier transform reveals the frequency content of a signal.

Prerequisites
-------------

This tutorial assumes you have:

- Completed simulations with time-series data
- Basic familiarity with numpy arrays
- Understanding of the UQ framework basics (see :doc:`basic_sensitivity`)

Getting Trajectory Data
-----------------------

First, let's extract a trajectory from simulation outputs:

.. code-block:: python

   import numpy as np
   from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql

   # Connect to simulation data
   conn = create_duckdb_conn()
   history_sql, config_sql, _ = dataset_sql("./outputs", ["experiment_id"])

   # Query for time-series data
   query = f"""
   SELECT
       time,
       "listeners__mass__dry_mass" as mass,
       "listeners__mass__protein_mass" as protein,
       "listeners__fba_results__growth" as growth_rate
   FROM ({history_sql})
   WHERE experiment_id = 0 AND lineage_seed = 0
   ORDER BY time
   """

   result = conn.execute(query).fetchnumpy()

   # Stack into trajectory matrix: (n_timesteps, n_observables)
   X = np.column_stack([
       result["mass"],
       result["protein"],
       result["growth_rate"],
   ])

   observable_names = ["mass", "protein", "growth_rate"]
   print(f"Trajectory shape: {X.shape}")

Basic DMD Analysis
------------------

Dynamic Mode Decomposition extracts the dominant modes:

.. code-block:: python

   from uq import DynamicModeDecomposition

   # Create DMD object
   # rank controls how many modes to extract (higher = more detail)
   dmd = DynamicModeDecomposition(rank=10)

   # Fit to trajectory data
   dmd.fit(X)

   # Get the spectrum
   spectrum = dmd.get_spectrum(observable_names=observable_names)

   print(f"Extracted {len(spectrum.modes)} Koopman modes")

Understanding the Spectrum
--------------------------

Each Koopman mode has several properties:

.. code-block:: python

   for i, mode in enumerate(spectrum.modes[:5]):
       print(f"\nMode {i+1}:")
       print(f"  Eigenvalue: {mode.eigenvalue:.4f}")
       print(f"  Frequency: {mode.frequency:.6f} Hz")
       print(f"  Period: {1/abs(mode.frequency) if mode.frequency != 0 else 'inf':.1f} s")
       print(f"  Decay rate: {mode.decay_rate:.6f} /s")
       print(f"  Energy: {mode.energy:.4f}")
       if mode.dominant_observables:
           print(f"  Dominant observables: {mode.dominant_observables[:3]}")

Interpreting the results:

- **Frequency = 0**: Growth/decay modes (exponential behavior)
- **|Frequency| > 0**: Oscillatory modes (periodic behavior)
- **Decay rate > 0**: Mode grows over time
- **Decay rate < 0**: Mode decays (transient)
- **Decay rate ≈ 0**: Persistent mode (steady behavior)
- **Energy**: Relative importance of the mode

Visualizing Modes
-----------------

Create a simple visualization of mode energies:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Sort modes by energy
   energies = [m.energy for m in spectrum.modes]
   frequencies = [m.frequency for m in spectrum.modes]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   # Energy bar chart
   ax1.bar(range(len(energies)), energies)
   ax1.set_xlabel("Mode index")
   ax1.set_ylabel("Energy")
   ax1.set_title("Mode Energies")

   # Eigenvalue spectrum (complex plane)
   eigenvalues = [m.eigenvalue for m in spectrum.modes]
   ax2.scatter([e.real for e in eigenvalues], [e.imag for e in eigenvalues], s=50)
   circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
   ax2.add_patch(circle)
   ax2.set_xlabel("Real")
   ax2.set_ylabel("Imaginary")
   ax2.set_title("Koopman Eigenvalues")
   ax2.axis('equal')

   plt.tight_layout()
   plt.savefig("koopman_spectrum.png", dpi=150)

Extended DMD with Dictionaries
------------------------------

For nonlinear dynamics, Extended DMD uses dictionary functions:

.. code-block:: python

   from uq import ExtendedDMD, KoopmanDictionary

   # Create a dictionary with polynomial and Fourier features
   dictionary = KoopmanDictionary(
       polynomial_degree=2,    # x, x^2, xy, etc.
       include_fourier=True,   # sin(x), cos(x), etc.
       fourier_terms=3,        # Number of Fourier terms
   )

   print(f"Dictionary will lift {X.shape[1]} observables to ~{dictionary.get_output_dim(X.shape[1])} features")

   # Fit EDMD
   edmd = ExtendedDMD(dictionary=dictionary, rank=15)
   edmd.fit(X)

   # Get spectrum
   edmd_spectrum = edmd.get_spectrum()
   print(f"EDMD extracted {len(edmd_spectrum.modes)} modes")

   # Compare reconstruction error
   X_pred = edmd.predict(X[:-1])
   mse = np.mean((X_pred - X[1:])**2)
   print(f"One-step prediction MSE: {mse:.6f}")

Finding Cell Cycle Harmonics
----------------------------

The key application for whole-cell simulations is identifying cell cycle modes:

.. code-block:: python

   from uq import CellCycleKoopmanAnalyzer

   # Create analyzer with expected cell cycle time
   # For E. coli in rich media, ~40-60 minutes
   cc_analyzer = CellCycleKoopmanAnalyzer(
       expected_cycle_time=2400.0,  # 40 minutes in seconds
       dt=1.0,                       # Time step of your data
       harmonic_tolerance=0.15,      # Allow 15% frequency mismatch
   )

   # Find cell cycle modes
   cc_modes = cc_analyzer.identify_cell_cycle_modes(spectrum)

   print(f"\nFound {len(cc_modes)} cell cycle-related modes:")
   for mode in cc_modes:
       period = 1 / abs(mode.frequency) if mode.frequency != 0 else float('inf')
       harmonic_order = mode.frequency * 2400.0  # cycles per cell cycle
       print(f"  Period: {period:.1f}s ({harmonic_order:.1f}× fundamental)")
       print(f"    Energy: {mode.energy:.4f}")
       print(f"    Decay: {mode.decay_rate:.6f}")

What to look for:

1. **Fundamental mode** (~1× period): Main cell cycle oscillation
2. **Second harmonic** (~2× frequency): Often related to division events
3. **Higher harmonics**: DNA replication phases, metabolic cycles

Spectral Sensitivity Analysis
-----------------------------

Compare Koopman spectra between conditions to understand how parameters affect dynamics:

.. code-block:: python

   from uq import KoopmanSensitivityAnalyzer

   # Load baseline and perturbed trajectories
   X_baseline = np.load("trajectory_baseline.npy")
   X_perturbed = np.load("trajectory_vio_high.npy")

   # Create analyzer
   ks_analyzer = KoopmanSensitivityAnalyzer(rank=10)

   # Compute spectral sensitivity
   sensitivity = ks_analyzer.spectral_sensitivity(
       X_baseline=X_baseline,
       X_perturbed=X_perturbed,
       parameter_names=["vio_expression"],
   )

   print("\nSpectral Sensitivity:")
   print(f"  Eigenvalue shift: {sensitivity['eigenvalue_shift']:.4f}")
   print(f"  Mode shape change: {sensitivity['mode_shape_change']:.4f}")
   print(f"  Frequency perturbation: {sensitivity['frequency_perturbation']:.4f}")

Interpretation:

- **Eigenvalue shift**: How much the eigenvalues moved in the complex plane
- **Mode shape change**: How different the mode structures are
- **Frequency perturbation**: Change in dominant frequencies

Extracting Features for ML
--------------------------

Use Koopman features as inputs to machine learning models:

.. code-block:: python

   from uq import extract_koopman_features

   # Extract compact feature set
   features = extract_koopman_features(
       X=X,
       n_modes=10,
       include_frequencies=True,
       include_amplitudes=True,
       include_growth_rates=True,
   )

   print("Koopman Features:")
   print(f"  Frequencies: {features['frequencies']}")
   print(f"  Amplitudes: {features['amplitudes']}")
   print(f"  Growth rates: {features['growth_rates']}")
   print(f"  Mode energies: {features['mode_energies']}")

   # Concatenate for ML input
   feature_vector = np.concatenate([
       features['frequencies'],
       features['amplitudes'],
       features['growth_rates'],
   ])
   print(f"Feature vector shape: {feature_vector.shape}")

Combining with PCE Analysis
---------------------------

Koopman and PCE complement each other:

.. code-block:: python

   from uq import (
       InputParameterSpace,
       WrapperConfig,
       SensitivityAnalyzer,
       SimulationWrapper,
       DynamicModeDecomposition,
       extract_koopman_features,
   )
   import numpy as np

   # 1. Run PCE sensitivity analysis
   param_space = InputParameterSpace(include_vio=True)
   config = WrapperConfig(sim_data_path="./sim_data.cPickle", output_dir="./uq")
   wrapper = SimulationWrapper(config, param_space)
   analyzer = SensitivityAnalyzer(param_space, wrapper)

   sobol, pce = analyzer.analyze_with_pce(polynomial_order=3)
   print("PCE tells us which parameters matter:")
   for name, val in sobol.get_most_influential(3):
       print(f"  {name}: {val:.4f}")

   # 2. Use Koopman to understand dynamics
   X = wrapper.get_trajectory(param_values=[1.0, 0.5])  # Example params
   dmd = DynamicModeDecomposition(rank=5)
   dmd.fit(X)
   spectrum = dmd.get_spectrum()

   print("\nKoopman tells us how the system behaves:")
   for mode in spectrum.modes[:3]:
       print(f"  Period: {1/abs(mode.frequency):.1f}s, Energy: {mode.energy:.4f}")

   # 3. Combine insights
   # - Most sensitive parameter (from PCE): vio_expression
   # - Dominant dynamics (from Koopman): 40-minute cell cycle
   # - Conclusion: vio_expression affects the 40-minute growth cycle

Best Practices
--------------

1. **Choose rank carefully**: Too low misses important modes; too high adds noise
2. **Use EDMD for nonlinear systems**: Polynomial dictionaries capture saturation effects
3. **Normalize data**: DMD works better when observables are on similar scales
4. **Check reconstruction**: High reconstruction error suggests too few modes or dictionary
5. **Validate cell cycle modes**: Compare identified periods with known biology

Common Issues
-------------

**No modes found**

- Data may be too short or noisy
- Try reducing rank or using EDMD with smoother dictionaries

**Spurious oscillations**

- Can arise from numerical noise
- Filter modes with very low energy
- Use larger dt (downsample data)

**Cell cycle modes not identified**

- Adjust ``harmonic_tolerance`` in CellCycleKoopmanAnalyzer
- Check that expected_cycle_time matches your simulation
- Cell cycle may be masked by stronger dynamics

Next Steps
----------

- :doc:`variance_decomposition` - Decompose variance across aggregation strategies
- :doc:`cell_cycle_analysis` - Traditional cell cycle stratification
- :doc:`../api/koopman` - Full API reference
