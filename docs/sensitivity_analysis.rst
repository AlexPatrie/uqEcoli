Sensitivity Analysis
====================

The UQ framework implements global sensitivity analysis using Polynomial Chaos
Expansion (PCE) surrogate methods and Sobol indices, as specified in the
Milestone 08.4.2 requirements.

Overview
--------

Sensitivity analysis answers the question: **Which input parameters have the
greatest influence on simulation outputs?**

The framework uses:

* **PCE (Polynomial Chaos Expansion)**: Builds a polynomial surrogate model
* **Sobol indices**: Quantifies variance-based sensitivity
* **UQPy/PyTUQ**: Industry-standard uncertainty quantification libraries

Sobol Sensitivity Indices
-------------------------

Sobol indices decompose output variance into contributions from each input parameter:

First-Order Index (S₁)
^^^^^^^^^^^^^^^^^^^^^^

Measures the **main effect** of a single parameter:

.. math::

   S_i = \\frac{V[E[Y|X_i]]}{V[Y]}

where :math:`V[E[Y|X_i]]` is the variance of the expected output when :math:`X_i` is fixed.

Total-Order Index (Sₜ)
^^^^^^^^^^^^^^^^^^^^^^

Measures the **total effect** including all interactions:

.. math::

   S_{Ti} = 1 - \\frac{V[E[Y|X_{\\sim i}]]}{V[Y]}

where :math:`X_{\\sim i}` represents all parameters except :math:`X_i`.

**Interpretation**:

* :math:`S_i \\approx S_{Ti}`: Parameter has minimal interactions
* :math:`S_{Ti} >> S_i`: Parameter has significant interactions with others
* :math:`\\sum S_i < 1`: Significant interaction effects exist

PCE Surrogate Method
--------------------

The recommended method builds a PCE surrogate model from simulation data:

.. code-block:: python

   from uq import SensitivityAnalyzer, InputParameterSpace

   param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
   analyzer = SensitivityAnalyzer(param_space, wrapper)

   # Build PCE surrogate and compute Sobol indices
   sobol_indices, pce_surrogate = analyzer.analyze_with_pce(
       polynomial_order=3,  # Polynomial degree
       n_samples=100,       # Training samples
       use_uqpy=True,       # Use UQPy (default)
   )

**Advantages**:

* Efficient: Sobol indices computed analytically from PCE coefficients
* Accurate: Captures nonlinear relationships
* Interpretable: Surrogate can be used for rapid predictions

Polynomial Order Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

The polynomial order controls model complexity:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Order
     - Pros
     - Cons
   * - 1-2
     - Few samples needed, fast
     - May miss nonlinear effects
   * - 3-4
     - Good balance
     - Moderate sample requirements
   * - 5+
     - Captures complex behavior
     - Many samples needed, potential overfitting

**Rule of thumb**: Start with order 3, increase if :math:`R^2 < 0.9`.

Sample Size Selection
^^^^^^^^^^^^^^^^^^^^^

The number of samples should scale with input dimension and polynomial order:

.. math::

   N_{min} \\approx 2 \\times \\binom{d + p}{p}

where :math:`d` is the number of parameters and :math:`p` is the polynomial order.

For 3 parameters and order 3: :math:`N_{min} \\approx 2 \\times 20 = 40`

Using UQPy
----------

The framework integrates with UQPy for sensitivity analysis:

.. code-block:: python

   from uq import create_uqpy_model, InputParameterSpace, WrapperConfig

   param_space = InputParameterSpace(include_vio=True, include_mecillinam=True)
   config = WrapperConfig(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_outputs",
   )

   # Create UQPy-compatible model
   model = create_uqpy_model(config, param_space)

   # Use with UQPy directly
   from UQpy.distributions import Uniform, JointIndependent
   from UQpy.sampling import LatinHypercubeSampling
   from UQpy.surrogates.polynomial_chaos import PolynomialChaosExpansion

   distributions = param_space.get_uqpy_distributions()
   joint = JointIndependent(marginals=distributions)
   lhs = LatinHypercubeSampling(distributions=joint, nsamples=100)

Using PyTUQ
-----------

Alternative integration with PyTUQ:

.. code-block:: python

   from uq import create_pytuq_model

   model_func = create_pytuq_model(config, param_space)
   lb, ub = param_space.get_pytuq_bounds()

   # Use with PyTUQ
   # from pytuq.surrogates import PCE
   # pce = PCE(order=3, bounds=(lb, ub))
   # pce.fit(X, model_func(X))

Direct Sobol Analysis
---------------------

For cases where PCE assumptions may not hold, direct Monte Carlo Sobol analysis
is available:

.. code-block:: python

   sobol = analyzer.analyze_with_sobol(
       n_samples=1024,           # Base samples (total = N * (2d + 2))
       calc_second_order=True,   # Include interaction indices
   )

   print(f"First-order: {sobol.first_order}")
   print(f"Total-order: {sobol.total_order}")
   print(f"Second-order: {sobol.second_order}")  # If calc_second_order=True

**Note**: Direct Sobol analysis requires many more samples than PCE
(:math:`N \\times (2d + 2)` evaluations).

Morris Screening
----------------

For high-dimensional parameter spaces (>10 parameters), running detailed PCE
or Sobol analysis on all parameters is computationally prohibitive. Morris
screening provides an efficient **pre-screening step** to identify which
parameters are influential.

Overview
^^^^^^^^

Morris screening (Elementary Effects method) computes the change in output when
each parameter is perturbed one at a time. Statistics of these "elementary
effects" reveal parameter importance:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Metric
     - Formula
     - Interpretation
   * - μ (mu)
     - Mean of elementary effects
     - Overall influence (can cancel if non-monotonic)
   * - μ* (mu_star)
     - Mean of |elementary effects|
     - Robust measure of influence (preferred)
   * - σ (sigma)
     - Std dev of elementary effects
     - Indicates interactions or nonlinearity

**Classification**:

* High μ*, low σ: Linear effect (no interactions)
* High μ*, high σ: Nonlinear effect or interactions with other params
* Low μ*: Parameter has little influence (can be fixed)

Computational Cost
^^^^^^^^^^^^^^^^^^

Morris screening requires only :math:`O(r \\times (d + 1))` evaluations,
where :math:`r` is the number of trajectories and :math:`d` is the number
of parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Method
     - Evaluations (20 params)
     - Evaluations (50 params)
     - Scaling
   * - Morris (r=20)
     - 420
     - 1,020
     - O(d)
   * - PCE (order 2)
     - ~500
     - ~2,600
     - O(d²)
   * - Sobol (MC)
     - ~50,000
     - ~500,000
     - O(d²)

Usage
^^^^^

.. code-block:: python

   from uq import SensitivityAnalyzer, MorrisIndices

   # Run Morris screening
   morris = analyzer.analyze_with_morris(
       n_trajectories=20,  # More = more stable (typical: 10-50)
       n_levels=4,         # Grid resolution (typical: 4-8)
   )

   # View summary
   print(morris.summary())

   # Get most influential parameters
   top_params = morris.get_most_influential(n=5)
   for name, mu_star in top_params:
       print(f"{name}: μ*={mu_star:.4f}")

   # Get parameters for detailed analysis
   important = morris.get_screening_candidates(top_n=5)
   print(f"Focus PCE analysis on: {important}")

   # Classify parameters
   classification = morris.classify_parameters()
   print(f"Negligible (can fix): {classification['negligible']}")
   print(f"Linear effects: {classification['linear']}")
   print(f"Nonlinear/interactions: {classification['nonlinear']}")

Screening Workflow
^^^^^^^^^^^^^^^^^^

The recommended workflow for high-dimensional problems:

.. code-block:: python

   from uq import InputParameterSpace, SensitivityAnalyzer

   # Stage 1: Define full parameter space
   full_space = InputParameterSpace(
       include_vio=True,
       include_mecillinam=True,
       include_knockouts=True,  # Many parameters
   )

   # Stage 2: Morris screening (cheap)
   analyzer = SensitivityAnalyzer(full_space, wrapper)
   morris = analyzer.analyze_with_morris(n_trajectories=20)

   # Stage 3: Identify important parameters
   important = morris.get_screening_candidates(top_n=5)
   print(f"Important parameters: {important}")

   # Stage 4: Create reduced parameter space
   reduced_space = InputParameterSpace(
       include_vio="vio_expression" in important,
       include_mecillinam="mecillinam_conc" in important,
       # ... only include important params
   )

   # Stage 5: Detailed PCE analysis on subset
   reduced_analyzer = SensitivityAnalyzer(reduced_space, wrapper)
   sobol, pce = reduced_analyzer.analyze_with_pce(polynomial_order=3)

Integration with Tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^

Morris results can be exported to the reactive tutorial format:

.. code-block:: python

   # Convert to PARAMETER_CONFIG format for tutorial 03c
   PARAMETER_CONFIG = morris.to_parameter_config(
       parameter_bounds=param_space.parameter_bounds,
       top_n=5,
   )

   # Result is a list of dicts ready for the tutorial:
   # [{"name": "param_a", "bounds": [0, 10], "default": 5.0,
   #   "step": 0.2, "description": "Morris: μ*=0.82 (linear)"}, ...]

Interpreting Results
--------------------

Working with SobolIndices
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get most influential parameters
   top_params = sobol_indices.get_most_influential(n=5, index_type="total")
   for name, value in top_params:
       print(f"{name}: {value:.4f}")

   # Access raw indices
   print(f"First-order: {sobol_indices.first_order}")
   print(f"Total-order: {sobol_indices.total_order}")
   print(f"Parameters: {sobol_indices.parameter_names}")

Visualization
^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   names = sobol_indices.parameter_names
   s1 = sobol_indices.first_order
   st = sobol_indices.total_order

   x = np.arange(len(names))
   width = 0.35

   fig, ax = plt.subplots()
   ax.bar(x - width/2, s1, width, label='First-order')
   ax.bar(x + width/2, st, width, label='Total-order')
   ax.set_ylabel('Sobol Index')
   ax.set_xticks(x)
   ax.set_xticklabels(names, rotation=45)
   ax.legend()
   plt.tight_layout()
   plt.savefig('sensitivity_indices.png')

Multi-Output Analysis
^^^^^^^^^^^^^^^^^^^^^

When analyzing multiple outputs, indices have shape ``(n_outputs, n_params)``:

.. code-block:: python

   # Average across outputs
   avg_first_order = np.mean(sobol_indices.first_order, axis=0)
   avg_total_order = np.mean(sobol_indices.total_order, axis=0)

   # Or analyze specific outputs
   output_idx = 0  # First output
   s1_output0 = sobol_indices.first_order[output_idx]

Convenience Functions
---------------------

For quick analysis, use the convenience functions:

.. code-block:: python

   from uq import run_sensitivity_analysis, analyze_precomputed_results

   # Run complete analysis workflow
   sobol, pce = run_sensitivity_analysis(
       sim_data_path="./sim_data.cPickle",
       output_dir="./uq_analysis",
       aggregation_strategy=AggregationStrategy.UNIFORM,
       polynomial_order=3,
       include_vio=True,
       include_mecillinam=True,
   )

   # Analyze existing simulation results
   sobol, pce = analyze_precomputed_results(
       data_dir="./simulation_outputs",
       aggregation_strategy=AggregationStrategy.BY_GENERATION,
   )

Best Practices
--------------

1. **Start simple**: Begin with order 3 PCE and ~100 samples
2. **Check convergence**: Increase samples until indices stabilize
3. **Validate surrogate**: Check :math:`R^2` of PCE fit
4. **Use appropriate aggregation**: Match strategy to analysis goals
5. **Filter transients**: Use ``generation_lower_bound`` to skip initial behavior
6. **Consider interactions**: If :math:`S_{Ti} >> S_i`, explore interaction effects

``uq_simple`` — Direct PyTUQ PCE Integration
----------------------------------------------

The ``uq_simple`` package bypasses ``SensitivityAnalyzer`` and uses
`PyTUQ's PCE class <https://sandialabs.github.io/pytuq/autoapi/pytuq/surrogates/pce/index.html>`_
directly, following the `UQPC workflow <https://sandialabs.github.io/pytuq/apps/uqpc.html>`_:

.. code-block:: python

   from uq_simple.pipeline import run_pipeline
   from uq.sampling import PrecomputedCache
   from uq.pipe import initialize_datasets

   cache = PrecomputedCache.load("./uq_cache")
   ds = initialize_datasets(experiment_ids=["exp1"], sim_base_path="/path/to/sims")

   # lsq (default), bcs (sparse), or anl (analytical)
   result = run_pipeline(cache=cache, param_space=ds.parameter_space, regression="bcs")

Internally, ``_fit_pce_and_sobol()`` performs:

1. Scale X to germ space [-1, 1]
2. Per-output: ``PCE(dim, order, 'LU')`` → ``set_training_data()`` → ``build(regression=...)``
3. Sync coefficients: ``pce.pcrv.setCfs([pce.lreg.cf])``
4. Extract Sobol: ``pce.pcrv.computeSens()`` / ``computeTotSens()``

This supports three PyTUQ regression backends:

* **lsq** — Standard least squares (overdetermined system)
* **bcs** — Bayesian Compressed Sensing (sparse PCE with automatic term selection)
* **anl** — Analytical regression (posterior predictive with uncertainty estimates)

See Also
--------

* :doc:`api/sensitivity` - Full API reference
* :doc:`tutorials/basic_sensitivity` - Step-by-step tutorial
* :doc:`aggregation_strategies` - All four RFC006 strategies
* `PyTUQ UQPC Workflow <https://sandialabs.github.io/pytuq/apps/uqpc.html>`_
* `PyTUQ Documentation <https://sandialabs.github.io/pytuq/>`_
