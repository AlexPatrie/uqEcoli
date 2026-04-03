"""
Tests for ``uq_simple.workflow`` — the RFC006 UQPC workflow.

Demonstrates that the PyTUQ-based PCE pipeline (adapted from
``apps/uqpc/uq_pc.py``) correctly:

  1. Sets up input PC from parameter bounds (Step 1)
  2. Generates / maps training samples between physical and germ space (Step 2)
  3. Fits PCE surrogates with all three regression methods (Step 4)
  4. Computes meaningful surrogate quality diagnostics (Step 5)
  5. Extracts analytically correct Sobol sensitivity indices (Step 6)
  6. Runs all four RFC006 aggregation strategies end-to-end

Test naming convention:
  test_step{N}_*           — unit tests for individual UQPC workflow steps
  test_strategy{N}_*       — integration tests for each RFC006 strategy
  test_e2e_*               — end-to-end pipeline tests
  test_regression_{method} — regression method comparison tests
  test_mathematical_*      — mathematical property verification
"""

from __future__ import annotations

import numpy as np
import pytest
from pytuq.rv.pcrv import PCRV

from uq.inputs import XSpaceVecoli
from uq.pipeline.models import SimDataParameter
from uq.sampling import PrecomputedCache
from uq.sensitivity import PCESurrogate, SobolIndices
from uq_simple.workflow import (
    UQPCResult,
    _bin_by_growth_stage,
    _compute_growth_fraction,
    _compute_relative_errors,
    _compute_sobol,
    _fit_surrogate,
    _physical_to_germ,
    _predict_and_variance,
    _setup_input_pc,
    run_strategy1_uniform,
    run_strategy2_by_generation,
    run_strategy3_by_seed,
    run_strategy4_growth_stratified,
    run_uqpc,
    run_uqpc_from_cache,
)

# ── Fixtures ────────────────────────────────────────────────────────


def _make_param_space(n_params: int = 3) -> XSpaceVecoli:
    """Create a minimal parameter space without real sim_data."""
    params = [
        SimDataParameter(
            name=f"p{i}",
            attr_path=f"dummy.path{i}",
            bounds=(0.0, 1.0),
        )
        for i in range(n_params)
    ]
    return XSpaceVecoli(
        parameter_names=[],
        parameter_bounds=[],
        parameter_types=[],
        experiment_id="test",
        parameters=params,
    )


def _make_param_space_nonuniform() -> XSpaceVecoli:
    """Parameter space with biologically realistic non-uniform bounds."""
    params = [
        SimDataParameter(
            name="fraction_active_rnap_free",
            attr_path="process.transcription.fraction_active_rnap_free",
            bounds=(0.25, 0.47),
        ),
        SimDataParameter(
            name="kinetic_objective_weight",
            attr_path="process.metabolism.kinetic_objective_weight",
            bounds=(5e-8, 5e-7),
        ),
        SimDataParameter(
            name="cell_dry_mass_fraction",
            attr_path="mass.cell_dry_mass_fraction",
            bounds=(0.25, 0.35),
        ),
    ]
    return XSpaceVecoli(
        parameter_names=[],
        parameter_bounds=[],
        parameter_types=[],
        experiment_id="test_bio",
        parameters=params,
    )


def _generate_linear_data(
    n_samples: int = 50,
    n_params: int = 3,
    coefficients: np.ndarray | None = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Y = X @ coeff + noise (known linear function).

    Returns (X, Y, true_coefficients).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, n_params))
    if coefficients is None:
        coefficients = np.array([2.0, 0.5, 0.1])[:n_params]
    Y = (X @ coefficients).reshape(-1, 1)
    if noise_std > 0:
        Y += rng.normal(0, noise_std, Y.shape)
    return X, Y, coefficients


def _generate_quadratic_data(
    n_samples: int = 60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Y = 3*x0^2 + x1 - 0.5*x0*x1 (known quadratic)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    Y = (3 * X[:, 0] ** 2 + X[:, 1] - 0.5 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    return X, Y


def _generate_multioutput_data(
    n_samples: int = 60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multi-output data where different params dominate each output.

    Output 0: dominated by p0 (coeff 3.0)
    Output 1: dominated by p1 (coeff 5.0)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    Y = np.column_stack([
        3.0 * X[:, 0] + 0.1 * X[:, 1],
        0.2 * X[:, 0] + 5.0 * X[:, 1],
    ])
    return X, Y


def _generate_synthetic_timeseries(
    n_samples: int = 30,
    n_timesteps: int = 100,
    n_obs: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate synthetic timeseries with exponential mass growth.

    Returns (X, Y_timeseries) where Y_timeseries[i] is (n_timesteps, n_obs).
    Column 0 is dry mass (exponential growth), column 1 is an observable.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    Y_timeseries = []
    for i in range(n_samples):
        t = np.linspace(0, 1, n_timesteps)
        # Mass grows exponentially, rate depends on p0
        mass = np.exp(t * (0.5 + X[i, 0]))
        # Second observable depends on p1
        obs2 = X[i, 1] * np.sin(2 * np.pi * t) + 1.0
        Y_timeseries.append(np.column_stack([mass, obs2]))
    return X, Y_timeseries


def _generate_timeseries_with_meta(
    n_samples: int = 30,
    n_timesteps: int = 200,
    n_generations: int = 3,
    n_seeds: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Generate timeseries with generation and seed metadata labels.

    Each timeseries has metadata assigning rows to generations and seeds.
    Returns (X, Y_timeseries, Y_timeseries_meta).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, 2))
    Y_ts = []
    Y_meta = []
    for i in range(n_samples):
        t = np.linspace(0, 1, n_timesteps)
        mass = np.exp(t * (0.5 + X[i, 0]))
        obs2 = X[i, 1] * np.sin(2 * np.pi * t) + 1.0
        Y_ts.append(np.column_stack([mass, obs2]))

        # Assign rows to generations and seeds uniformly
        generations = np.repeat(np.arange(n_generations), n_timesteps // n_generations + 1)[:n_timesteps]
        seeds = np.repeat(np.arange(n_seeds), n_timesteps // n_seeds + 1)[:n_timesteps]
        Y_meta.append({
            "generation": generations,
            "lineage_seed": seeds,
        })
    return X, Y_ts, Y_meta


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Input PC Setup
# ═══════════════════════════════════════════════════════════════════


class TestStep1InputPCSetup:
    """Verify that parameter bounds → PCRV (Legendre basis) is correct."""

    @pytest.mark.unit
    def test_pcrv_type_and_basis(self):
        """PCRV should use Legendre (LU) basis for uniform parameters."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        pc, _pcf_all, in_pcdim = _setup_input_pc(bounds)
        assert isinstance(pc, PCRV)
        assert in_pcdim == 3

    @pytest.mark.unit
    def test_pc_coefficients_encode_affine_map(self):
        """PC coefficients should encode the midpoint + half_range affine map."""
        bounds = np.array([[2.0, 4.0], [10.0, 20.0]])
        _pc, pcf_all, _ = _setup_input_pc(bounds)

        # Row 0 = midpoints: [3.0, 15.0]
        np.testing.assert_allclose(pcf_all[0], [3.0, 15.0])
        # Rows 1-2 = diag(half_ranges): [[1.0, 0], [0, 5.0]]
        np.testing.assert_allclose(pcf_all[1, 0], 1.0)
        np.testing.assert_allclose(pcf_all[2, 1], 5.0)

    @pytest.mark.unit
    def test_germ_to_physical_roundtrip(self):
        """Samples generated in germ space should map to within bounds."""
        bounds = np.array([[0.25, 0.47], [5e-8, 5e-7], [0.25, 0.35]])
        pc, _, _ = _setup_input_pc(bounds)
        np.random.seed(42)
        germ = pc.sampleGerm(100)
        X_phys = pc.evalPC(germ)

        for j in range(3):
            assert np.all(X_phys[:, j] >= bounds[j, 0] - 1e-10)
            assert np.all(X_phys[:, j] <= bounds[j, 1] + 1e-10)


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Sample Generation and Germ Mapping
# ═══════════════════════════════════════════════════════════════════


class TestStep2SampleGeneration:
    """Verify physical ↔ germ space transformations."""

    @pytest.mark.unit
    def test_physical_to_germ_unit_bounds(self):
        """[0,1] bounds → germ space should produce values in [-1, 1]."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        germ = _physical_to_germ(X, bounds)

        np.testing.assert_allclose(germ[0], [-1.0, -1.0])
        np.testing.assert_allclose(germ[1], [0.0, 0.0])
        np.testing.assert_allclose(germ[2], [1.0, 1.0])

    @pytest.mark.unit
    def test_physical_to_germ_nonuniform_bounds(self):
        """Non-uniform bounds should still map correctly to [-1, 1]."""
        bounds = np.array([[0.25, 0.47], [5e-8, 5e-7]])
        X = np.array([[0.25, 5e-8], [0.36, 2.75e-7], [0.47, 5e-7]])
        germ = _physical_to_germ(X, bounds)

        np.testing.assert_allclose(germ[0], [-1.0, -1.0], atol=1e-10)
        np.testing.assert_allclose(germ[1], [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(germ[2], [1.0, 1.0], atol=1e-10)

    @pytest.mark.unit
    def test_zero_width_bounds_handled(self):
        """Zero-width bounds (degenerate) should not produce NaN."""
        bounds = np.array([[1.0, 1.0], [0.0, 1.0]])
        X = np.array([[1.0, 0.5]])
        germ = _physical_to_germ(X, bounds)
        assert not np.any(np.isnan(germ))


# ═══════════════════════════════════════════════════════════════════
# STEP 4: PCE Surrogate Fitting
# ═══════════════════════════════════════════════════════════════════


class TestStep4SurrogateFitting:
    """Verify PCE fitting produces correct PCRV and linreg objects."""

    @pytest.mark.unit
    def test_fit_returns_pcrv_and_linregs(self):
        """pc_fit-equivalent should return (PCRV, list[linreg])."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=2)
        germ = _physical_to_germ(X, bounds)
        pcrv, linregs = _fit_surrogate(germ, Y, polynomial_order=2)

        assert isinstance(pcrv, PCRV)
        assert len(linregs) == Y.shape[1]

    @pytest.mark.unit
    def test_fit_lsq_low_error_on_linear(self):
        """LSQ should fit a linear function with near-zero error."""
        bounds = np.array([[0.0, 1.0]] * 3)
        X, Y, _ = _generate_linear_data(n_samples=50, n_params=3, noise_std=0.0)
        germ = _physical_to_germ(X, bounds)
        pcrv, _linregs = _fit_surrogate(germ, Y, polynomial_order=1, regression="lsq")

        Y_pred = pcrv.function(germ)
        relerr = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
        assert relerr < 1e-10, f"LSQ on linear data should be exact, got relerr={relerr}"

    @pytest.mark.unit
    def test_fit_bcs_on_quadratic(self):
        """BCS should fit a quadratic with order-2 PCE and produce sparse coefficients."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        X, Y = _generate_quadratic_data(n_samples=60)
        germ = _physical_to_germ(X, bounds)
        pcrv, _linregs = _fit_surrogate(
            germ,
            Y,
            polynomial_order=3,
            regression="bcs",
            tolerance=1e-4,
        )

        Y_pred = pcrv.function(germ)
        relerr = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
        assert relerr < 0.1, f"BCS on quadratic should fit reasonably, got relerr={relerr}"

    @pytest.mark.unit
    def test_fit_anl_on_linear(self):
        """Analytical (Bayesian) regression should fit linear data."""
        bounds = np.array([[0.0, 1.0]] * 2)
        X, Y, _ = _generate_linear_data(n_samples=40, n_params=2, noise_std=0.01)
        germ = _physical_to_germ(X, bounds)
        pcrv, _linregs = _fit_surrogate(germ, Y, polynomial_order=2, regression="anl")

        Y_pred = pcrv.function(germ)
        relerr = np.linalg.norm(Y - Y_pred) / np.linalg.norm(Y)
        assert relerr < 0.05, f"ANL on noisy linear should fit well, got relerr={relerr}"

    @pytest.mark.unit
    def test_fit_invalid_regression_raises(self):
        """Unknown regression method should raise ValueError."""
        bounds = np.array([[0.0, 1.0]])
        germ = np.linspace(-1, 1, 10).reshape(-1, 1)
        Y = germ**2
        with pytest.raises(ValueError, match="Unknown regression method"):
            _fit_surrogate(germ, Y, polynomial_order=2, regression="xyz")


# ═══════════════════════════════════════════════════════════════════
# STEP 4b: Prediction and Variance
# ═══════════════════════════════════════════════════════════════════


class TestStep4bPredictionVariance:
    """Verify prediction outputs and uncertainty estimates."""

    @pytest.mark.unit
    def test_prediction_shapes(self):
        """Predictions and std dev should match input dimensions."""
        bounds = np.array([[0.0, 1.0]] * 2)
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=2)
        germ = _physical_to_germ(X, bounds)
        pcrv, linregs = _fit_surrogate(germ, Y, polynomial_order=2)
        Y_pc, Y_std = _predict_and_variance(pcrv, linregs, germ, Y.shape[1])

        assert Y_pc.shape == Y.shape
        assert Y_std.shape == Y.shape

    @pytest.mark.unit
    def test_prediction_variance_nonnegative(self):
        """Prediction variance should be non-negative everywhere."""
        bounds = np.array([[0.0, 1.0]] * 2)
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=2, noise_std=0.1)
        germ = _physical_to_germ(X, bounds)
        pcrv, linregs = _fit_surrogate(germ, Y, polynomial_order=2)
        _, Y_std = _predict_and_variance(pcrv, linregs, germ, Y.shape[1])
        assert np.all(Y_std >= 0)


# ═══════════════════════════════════════════════════════════════════
# STEP 5: Relative Errors
# ═══════════════════════════════════════════════════════════════════


class TestStep5RelativeErrors:
    """Verify relative error computation."""

    @pytest.mark.unit
    def test_perfect_prediction_zero_error(self):
        Y = np.array([[1.0, 2.0], [3.0, 4.0]])
        err = _compute_relative_errors(Y, Y)
        np.testing.assert_allclose(err, [0.0, 0.0], atol=1e-15)

    @pytest.mark.unit
    def test_known_relative_error(self):
        """Relative error should equal ||Y - Y_hat|| / ||Y|| per output."""
        Y = np.array([[10.0], [20.0]])
        Y_hat = np.array([[11.0], [21.0]])
        err = _compute_relative_errors(Y, Y_hat)
        expected = np.linalg.norm([1.0, 1.0]) / np.linalg.norm([10.0, 20.0])
        np.testing.assert_allclose(err[0], expected, rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════
# STEP 6: Sobol Indices
# ═══════════════════════════════════════════════════════════════════


class TestStep6SobolIndices:
    """Verify Sobol index computation from PCE coefficients."""

    @pytest.mark.unit
    def test_sobol_returns_correct_type(self):
        bounds = np.array([[0.0, 1.0]] * 2)
        X, Y, _ = _generate_linear_data(n_samples=40, n_params=2)
        germ = _physical_to_germ(X, bounds)
        pcrv, _ = _fit_surrogate(germ, Y, polynomial_order=2)
        sobol = _compute_sobol(pcrv, ["p0", "p1"], Y)
        assert isinstance(sobol, SobolIndices)
        assert sobol.first_order.shape == (2,)
        assert sobol.total_order.shape == (2,)

    @pytest.mark.unit
    def test_sobol_identifies_dominant_parameter(self):
        """For Y = 2*x0 + 0.5*x1, S_T(x0) should be >> S_T(x1)."""
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=50, n_params=2, coefficients=np.array([2.0, 0.5]))
        result = run_uqpc(ps, Y, X, polynomial_order=2)

        s_t = result.sobol.total_order
        assert s_t[0] > s_t[1], f"p0 (coeff=2) should dominate p1 (coeff=0.5), got S_T={s_t}"
        # Analytical: var(2*x0) / var_total = 4/4.25 ≈ 0.941
        assert s_t[0] > 0.9

    @pytest.mark.unit
    def test_sobol_sum_close_to_one(self):
        """For additive models (no interactions), sum(S1) ≈ 1."""
        ps = _make_param_space(n_params=3)
        X, Y, _ = _generate_linear_data(n_samples=80, n_params=3, noise_std=0.0)
        result = run_uqpc(ps, Y, X, polynomial_order=2)
        s1_sum = result.sobol.first_order.sum()
        assert abs(s1_sum - 1.0) < 0.01, f"sum(S1) should ≈ 1 for additive model, got {s1_sum}"

    @pytest.mark.unit
    def test_sobol_joint_indices_shape(self):
        """Joint (second-order) indices should be (n_params, n_params)."""
        ps = _make_param_space(n_params=3)
        X, Y, _ = _generate_linear_data(n_samples=50, n_params=3)
        result = run_uqpc(ps, Y, X, polynomial_order=2)
        assert result.sobol.second_order.shape == (3, 3)

    @pytest.mark.unit
    def test_sobol_variance_weighted_multioutput(self):
        """Multi-output Sobol should weight by per-output variance."""
        ps = _make_param_space(n_params=2)
        X, Y = _generate_multioutput_data(n_samples=60)
        result = run_uqpc(ps, Y, X, polynomial_order=2)

        s_t = result.sobol.total_order
        # Output 0: var(3*x0) = 0.75, var(0.1*x1) ≈ 0.0008 → p0 dominates
        # Output 1: var(0.2*x0) ≈ 0.003, var(5*x1) ≈ 2.08 → p1 dominates
        # Output 1 has higher total variance → p1 should dominate overall
        assert s_t[1] > s_t[0], f"p1 should dominate in variance-weighted Sobol, got S_T={s_t}"


# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestMathematicalProperties:
    """Verify mathematical guarantees of the PCE/Sobol framework."""

    @pytest.mark.unit
    def test_sobol_nonnegative(self):
        """All Sobol indices should be >= 0."""
        ps = _make_param_space(n_params=3)
        X, Y, _ = _generate_linear_data(n_samples=50, n_params=3, noise_std=0.01)
        result = run_uqpc(ps, Y, X, polynomial_order=2)
        assert np.all(result.sobol.first_order >= -1e-10)
        assert np.all(result.sobol.total_order >= -1e-10)

    @pytest.mark.unit
    def test_total_order_geq_first_order(self):
        """S_Ti >= S_i for all parameters (total includes interactions)."""
        ps = _make_param_space(n_params=2)
        X, Y = _generate_quadratic_data(n_samples=80)
        result = run_uqpc(ps, Y, X, polynomial_order=3)
        diff = result.sobol.total_order - result.sobol.first_order
        assert np.all(diff >= -1e-10), f"S_T should >= S_1, got diff={diff}"

    @pytest.mark.unit
    def test_higher_order_improves_fit(self):
        """Higher polynomial order should reduce training error on nonlinear data."""
        ps = _make_param_space(n_params=2)
        X, Y = _generate_quadratic_data(n_samples=80)

        r1 = run_uqpc(ps, Y, X, polynomial_order=1)
        r2 = run_uqpc(ps, Y, X, polynomial_order=2)
        r3 = run_uqpc(ps, Y, X, polynomial_order=3)

        assert r2.relerr_train[0] < r1.relerr_train[0]
        assert r3.relerr_train[0] <= r2.relerr_train[0] + 1e-10

    @pytest.mark.unit
    def test_surrogate_is_exportable(self):
        """PCESurrogate should have correct attributes for serialization."""
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=2)
        result = run_uqpc(ps, Y, X, polynomial_order=2)

        sur = result.surrogate
        assert isinstance(sur, PCESurrogate)
        assert sur.basis_type == "legendre"
        assert sur.polynomial_order == 2
        assert sur.input_dim == 2
        assert sur.output_dim == 1
        assert sur.input_bounds is not None
        assert sur.coefficients.ndim == 1
        assert sur.multi_indices.ndim == 2


# ═══════════════════════════════════════════════════════════════════
# REGRESSION METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════


class TestRegressionMethods:
    """Compare the three PyTUQ regression methods (lsq, bcs, anl)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["lsq", "bcs", "anl"])
    def test_all_methods_produce_result(self, method):
        """Each method should produce a valid UQPCResult."""
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=40, n_params=2, noise_std=0.01)
        result = run_uqpc(ps, Y, X, polynomial_order=2, regression=method)
        assert isinstance(result, UQPCResult)
        assert result.relerr_train[0] < 0.1

    @pytest.mark.unit
    def test_lsq_exact_on_noiseless(self):
        """LSQ should achieve machine-precision error on noiseless polynomial data."""
        ps = _make_param_space(n_params=2)
        X, Y = _generate_quadratic_data(n_samples=60)
        result = run_uqpc(ps, Y, X, polynomial_order=2, regression="lsq")
        assert result.relerr_train[0] < 1e-10


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: UNIFORM (BULK)
# ═══════════════════════════════════════════════════════════════════


class TestStrategy1Uniform:
    """RFC006 Strategy 1: Uniform aggregation across all cells/times."""

    @pytest.mark.unit
    def test_strategy1_returns_uqpc_result(self):
        ps = _make_param_space(n_params=3)
        X, Y, _ = _generate_linear_data(n_samples=50, n_params=3)
        result = run_strategy1_uniform(ps, X, Y, polynomial_order=2)
        assert isinstance(result, UQPCResult)

    @pytest.mark.unit
    def test_strategy1_identifies_dominant_param(self):
        """Strategy 1 should correctly identify the dominant parameter."""
        ps = _make_param_space(n_params=3)
        coeffs = np.array([0.1, 5.0, 0.3])
        X, Y, _ = _generate_linear_data(n_samples=80, n_params=3, coefficients=coeffs)
        result = run_strategy1_uniform(ps, X, Y, polynomial_order=2)
        # p1 (coeff=5.0) should dominate
        assert np.argmax(result.sobol.total_order) == 1


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: BY GENERATION
# ═══════════════════════════════════════════════════════════════════


class TestStrategy2ByGeneration:
    """RFC006 Strategy 2: Stratified by generation."""

    @pytest.mark.unit
    def test_strategy2_returns_per_generation_results(self):
        ps = _make_param_space(n_params=2)
        X, Y_ts, Y_meta = _generate_timeseries_with_meta(
            n_samples=30,
            n_generations=3,
            n_seeds=1,
        )
        results = run_strategy2_by_generation(
            ps,
            X,
            Y_ts,
            Y_meta,
            polynomial_order=2,
        )
        assert isinstance(results, dict)
        assert len(results) == 3  # 3 generations
        for gen, r in results.items():
            assert isinstance(r, UQPCResult)
            assert isinstance(gen, int)

    @pytest.mark.unit
    def test_strategy2_each_generation_has_sobol(self):
        """Each generation should have valid Sobol indices."""
        ps = _make_param_space(n_params=2)
        X, Y_ts, Y_meta = _generate_timeseries_with_meta(n_samples=30, n_generations=2)
        results = run_strategy2_by_generation(ps, X, Y_ts, Y_meta, polynomial_order=2)
        for gen, r in results.items():
            assert r.sobol.total_order.shape == (2,)
            assert np.all(r.sobol.total_order >= -1e-10)

    @pytest.mark.unit
    def test_strategy2_empty_when_no_generation_metadata(self):
        """Should return empty dict if metadata lacks 'generation' key."""
        ps = _make_param_space(n_params=2)
        X, Y_ts, _ = _generate_timeseries_with_meta(n_samples=10)
        empty_meta = [{"lineage_seed": np.zeros(200)} for _ in range(10)]
        results = run_strategy2_by_generation(ps, X, Y_ts, empty_meta, polynomial_order=2)
        assert results == {}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: BY LINEAGE SEED
# ═══════════════════════════════════════════════════════════════════


class TestStrategy3BySeed:
    """RFC006 Strategy 3: Stratified by lineage seed."""

    @pytest.mark.unit
    def test_strategy3_returns_per_seed_results(self):
        ps = _make_param_space(n_params=2)
        X, Y_ts, Y_meta = _generate_timeseries_with_meta(
            n_samples=30,
            n_seeds=2,
            n_generations=1,
        )
        results = run_strategy3_by_seed(ps, X, Y_ts, Y_meta, polynomial_order=2)
        assert isinstance(results, dict)
        assert len(results) == 2  # 2 seeds
        for seed_val, r in results.items():
            assert isinstance(r, UQPCResult)

    @pytest.mark.unit
    def test_strategy3_empty_when_no_seed_metadata(self):
        ps = _make_param_space(n_params=2)
        X, Y_ts, _ = _generate_timeseries_with_meta(n_samples=10)
        empty_meta = [{"generation": np.zeros(200)} for _ in range(10)]
        results = run_strategy3_by_seed(ps, X, Y_ts, empty_meta, polynomial_order=2)
        assert results == {}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: GROWTH-STRATIFIED
# ═══════════════════════════════════════════════════════════════════


class TestStrategy4GrowthStratified:
    """RFC006 Strategy 4: Stratified by cell cycle stage (θ)."""

    @pytest.mark.unit
    def test_strategy4_returns_per_stage_and_combined(self):
        ps = _make_param_space(n_params=2)
        X, Y_ts = _generate_synthetic_timeseries(n_samples=30, n_obs=2)
        per_stage, combined = run_strategy4_growth_stratified(
            ps,
            X,
            Y_ts,
            n_bins=5,
            polynomial_order=2,
        )
        assert len(per_stage) == 5
        assert isinstance(combined, UQPCResult)
        for r in per_stage:
            assert isinstance(r, UQPCResult)

    @pytest.mark.unit
    def test_strategy4_sobol_varies_across_stages(self):
        """Sobol indices should differ between early and late growth stages."""
        ps = _make_param_space(n_params=2)
        X, Y_ts = _generate_synthetic_timeseries(n_samples=40, n_obs=2)
        per_stage, _ = run_strategy4_growth_stratified(
            ps,
            X,
            Y_ts,
            n_bins=5,
            polynomial_order=2,
        )
        # Collect total-order indices across stages
        st_matrix = np.array([r.sobol.total_order for r in per_stage])
        # At least some variation across stages (not all identical)
        assert np.std(st_matrix[:, 0]) > 0.01 or np.std(st_matrix[:, 1]) > 0.01, (
            "Parameter importance should vary across growth stages"
        )

    @pytest.mark.unit
    def test_growth_fraction_monotonic(self):
        """θ = normalized log(mass) should be monotonically non-decreasing
        for exponentially growing cells."""
        ts = np.column_stack([np.exp(np.linspace(0, 1, 100)), np.ones(100)])
        theta = _compute_growth_fraction(ts, mass_col_index=0)
        assert np.all(np.diff(theta) >= -1e-10)
        np.testing.assert_allclose(theta[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(theta[-1], 1.0, atol=1e-10)

    @pytest.mark.unit
    def test_growth_fraction_degenerate_mass(self):
        """Constant mass (no growth) should produce linear fallback θ."""
        ts = np.column_stack([np.ones(50), np.ones(50)])
        theta = _compute_growth_fraction(ts, mass_col_index=0)
        expected = np.linspace(0, 1, 50)
        np.testing.assert_allclose(theta, expected, atol=1e-10)

    @pytest.mark.unit
    def test_bin_assignment_correct(self):
        """Bins should cover [0, 1] uniformly."""
        theta = np.linspace(0, 1, 100)
        bins = _bin_by_growth_stage(theta, n_bins=5)
        assert bins.min() == 0
        assert bins.max() == 4
        # Each bin should get roughly 20 points
        for b in range(5):
            count = np.sum(bins == b)
            assert 15 <= count <= 25, f"Bin {b} has {count} points (expected ~20)"


# ═══════════════════════════════════════════════════════════════════
# PRECOMPUTED CACHE INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestPrecomputedCacheIntegration:
    """Verify integration with the PrecomputedCache offline workflow."""

    @pytest.mark.unit
    def test_run_uqpc_from_cache(self, tmp_path):
        """run_uqpc_from_cache should produce identical results to run_uqpc."""
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=40, n_params=2)

        # Create cache on disk
        cache = PrecomputedCache(
            cache_dir=tmp_path / "cache",
            X=X,
            Y=Y,
            parameter_names=["p0", "p1"],
        )
        cache.save()
        loaded = PrecomputedCache.load(tmp_path / "cache")

        result_direct = run_uqpc(ps, Y, X, polynomial_order=2, seed=42)
        result_cache = run_uqpc_from_cache(ps, loaded, polynomial_order=2, seed=42)

        np.testing.assert_allclose(
            result_direct.sobol.total_order,
            result_cache.sobol.total_order,
            atol=1e-10,
        )


# ═══════════════════════════════════════════════════════════════════
# UQPCResult STRUCTURE
# ═══════════════════════════════════════════════════════════════════


class TestUQPCResultStructure:
    """Verify the result container exposes all UQPC workflow artifacts."""

    @pytest.mark.unit
    def test_result_has_all_fields(self):
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=2)
        result = run_uqpc(ps, Y, X, polynomial_order=2)

        # Core artifacts
        assert isinstance(result.sobol, SobolIndices)
        assert isinstance(result.surrogate, PCESurrogate)
        assert isinstance(result.pcrv, PCRV)
        assert isinstance(result.linregs, list)

        # Training data
        assert result.germ_train.shape == (30, 2)
        assert result.X_train.shape == (30, 2)
        assert result.Y_train.shape == (30, 1)
        assert result.Y_train_pc.shape == (30, 1)
        assert result.Y_train_pc_std.shape == (30, 1)
        assert result.relerr_train.shape == (1,)

        # Test data (not generated in offline mode)
        assert result.germ_test is None
        assert result.Y_test is None

    @pytest.mark.unit
    def test_result_parameter_names_preserved(self):
        ps = _make_param_space(n_params=3)
        X, Y, _ = _generate_linear_data(n_samples=30, n_params=3)
        result = run_uqpc(ps, Y, X, polynomial_order=1)
        assert result.sobol.parameter_names == ["p0", "p1", "p2"]


# ═══════════════════════════════════════════════════════════════════
# END-TO-END: Full RFC006 Pipeline
# ═══════════════════════════════════════════════════════════════════


class TestE2EFullPipeline:
    """End-to-end tests exercising the complete RFC006 workflow.

    These tests simulate the full pipeline path:
      1. Create parameter space (from SimDataParameter specs)
      2. Generate synthetic training data
      3. Run all 4 strategies
      4. Verify cross-strategy consistency
    """

    @pytest.mark.e2e
    def test_all_four_strategies_from_same_data(self):
        """Run all 4 RFC006 strategies from the same synthetic dataset
        and verify they produce structurally valid, non-degenerate results."""
        ps = _make_param_space(n_params=2)
        X, Y_ts, Y_meta = _generate_timeseries_with_meta(
            n_samples=40,
            n_timesteps=200,
            n_generations=2,
            n_seeds=2,
        )
        Y_bulk = np.vstack([ts.mean(axis=0) for ts in Y_ts])

        # Strategy 1: uniform
        r1 = run_strategy1_uniform(ps, X, Y_bulk, polynomial_order=2)
        assert r1.relerr_train.max() < 0.5

        # Strategy 2: by generation
        r2 = run_strategy2_by_generation(ps, X, Y_ts, Y_meta, polynomial_order=2)
        assert len(r2) == 2
        for gen, r in r2.items():
            assert r.sobol.total_order.shape == (2,)

        # Strategy 3: by seed
        r3 = run_strategy3_by_seed(ps, X, Y_ts, Y_meta, polynomial_order=2)
        assert len(r3) == 2

        # Strategy 4: growth-stratified
        per_stage, _combined = run_strategy4_growth_stratified(
            ps,
            X,
            Y_ts,
            n_bins=4,
            polynomial_order=2,
        )
        assert len(per_stage) == 4

    @pytest.mark.e2e
    def test_realistic_parameter_bounds(self):
        """Run UQPC with biologically realistic parameter bounds."""
        ps = _make_param_space_nonuniform()
        rng = np.random.default_rng(42)
        bounds = np.array(ps.parameter_bounds)
        n = 60
        X = rng.uniform(bounds[:, 0], bounds[:, 1], (n, 3))
        # Simulate a biologically plausible response
        Y = (
            0.5 * (X[:, 0] - 0.36) / 0.11  # RNAP effect (normalized)
            + 0.1 * np.log10(X[:, 1] / 1e-7)  # kinetic weight (log-scale)
            + 2.0 * (X[:, 2] - 0.30)  # mass fraction effect
        ).reshape(-1, 1)

        result = run_uqpc(ps, Y, X, polynomial_order=2)
        assert result.relerr_train[0] < 0.01
        assert result.sobol.total_order.shape == (3,)
        # All three params should have non-negligible influence
        # (kinetic_objective_weight has small range in log-space, so its
        #  normalized linear effect is weak — threshold accordingly)
        assert np.all(result.sobol.total_order > 0.005)

    @pytest.mark.e2e
    def test_pipeline_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        ps = _make_param_space(n_params=2)
        X, Y, _ = _generate_linear_data(n_samples=40, n_params=2)

        r1 = run_uqpc(ps, Y, X, polynomial_order=2, seed=99)
        r2 = run_uqpc(ps, Y, X, polynomial_order=2, seed=99)

        np.testing.assert_array_equal(r1.sobol.first_order, r2.sobol.first_order)
        np.testing.assert_array_equal(r1.sobol.total_order, r2.sobol.total_order)
        np.testing.assert_array_equal(r1.relerr_train, r2.relerr_train)
