"""
Generate a full PipelineResult with synthetic data and export all artifacts.

Run with:
    uv run pytest tests/test_export_artifacts.py -s -v

Artifacts are written to ./test_export_output/ and NOT cleaned up.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from uq.koopman import KoopmanMode, KoopmanSpectrum
from uq.pipeline.models import PipelineResult, StratificationLens, UqProfile
from uq.sensitivity import CellCycleRelevanceResult, MorrisIndices, PCESurrogate, SobolIndices
from uq.viz import plot_koopman_spectrum

OUTPUT_DIR = Path(__file__).parent.parent / "test_export_output"

PARAM_NAMES = ["vio_expression", "vio_trl_eff", "mecillinam_concentration"]
N_PARAMS = len(PARAM_NAMES)
N_BINS = 10
OBS_NAMES = ["listeners__mass__dry_mass", "listeners__mass__growth"]


def _make_synthetic_spectrum() -> KoopmanSpectrum:
    """Build a realistic synthetic KoopmanSpectrum with cell cycle mode."""
    rng = np.random.default_rng(42)
    n_modes = 8

    eigenvalues = []
    modes_list = []

    for i in range(n_modes):
        if i == 0:
            # Dominant cell cycle mode: frequency ~ 1/3600 Hz
            omega = 2 * np.pi / 3600.0
            eig = np.exp(complex(-0.001, omega))
            amp = complex(1.0, 0.2)
        elif i == 1:
            # Conjugate of cell cycle mode
            omega = 2 * np.pi / 3600.0
            eig = np.exp(complex(-0.001, -omega))
            amp = complex(1.0, -0.2)
        elif i == 2:
            # Second harmonic
            omega = 2 * 2 * np.pi / 3600.0
            eig = np.exp(complex(-0.005, omega))
            amp = complex(0.3, 0.1)
        elif i == 3:
            # Conjugate of second harmonic
            omega = 2 * 2 * np.pi / 3600.0
            eig = np.exp(complex(-0.005, -omega))
            amp = complex(0.3, -0.1)
        else:
            # Decaying non-oscillatory modes
            eig = complex(0.8 - 0.1 * i, 0.0)
            amp = complex(0.1 * rng.random(), 0.0)

        mode_vec = rng.normal(0, 1, len(OBS_NAMES)) + 1j * rng.normal(0, 0.1, len(OBS_NAMES))
        mode_vec = mode_vec / (np.linalg.norm(mode_vec) + 1e-10)

        eigenvalues.append(eig)
        modes_list.append(
            KoopmanMode(
                eigenvalue=eig,
                mode=mode_vec,
                amplitude=amp,
            )
        )

    eigenvalues_arr = np.array(eigenvalues)
    eigenvectors = np.column_stack([m.mode for m in modes_list])
    amplitudes = np.array([m.amplitude for m in modes_list])

    return KoopmanSpectrum(
        modes=modes_list,
        eigenvalues=eigenvalues_arr,
        eigenvectors=eigenvectors,
        amplitudes=amplitudes,
        reconstruction_error=0.02,
        dt=1.0,
        observable_names=OBS_NAMES,
    )


def _make_full_pipeline_result() -> tuple[PipelineResult, KoopmanSpectrum]:
    """Build a full PipelineResult with all fields populated."""
    rng = np.random.default_rng(42)

    # Population surrogate
    n_terms = 10
    pop_surrogate = PCESurrogate(
        coefficients=rng.normal(0, 1, n_terms),
        multi_indices=np.zeros((n_terms, N_PARAMS), dtype=int),
        basis_type="legendre",
        polynomial_order=3,
        input_dim=N_PARAMS,
        output_dim=len(OBS_NAMES),
        r_squared=0.92,
        input_bounds=np.array([[0.0, 5.0], [0.0, 2.0], [0.0, 10.0]]),
    )

    # Population Sobol
    pop_sobol = SobolIndices(
        first_order=np.array([0.45, 0.05, 0.40]),
        total_order=np.array([0.55, 0.08, 0.50]),
        parameter_names=PARAM_NAMES,
    )

    # Cell cycle surrogate
    cc_surrogate = PCESurrogate(
        coefficients=rng.normal(0, 1, n_terms),
        multi_indices=np.zeros((n_terms, N_PARAMS), dtype=int),
        basis_type="legendre",
        polynomial_order=2,
        input_dim=N_PARAMS,
        output_dim=N_BINS,
        r_squared=0.87,
        input_bounds=np.array([[0.0, 5.0], [0.0, 2.0], [0.0, 10.0]]),
    )

    # Per-stage Sobol with realistic cell-cycle-dependent patterns
    cc_sobols = []
    for k in range(N_BINS):
        theta_mid = (k + 0.5) / N_BINS
        # vio dominates early cycle, mecillinam dominates late cycle
        vio_s = 0.6 * (1 - theta_mid) + rng.normal(0, 0.03)
        trl_s = 0.05 + rng.normal(0, 0.01)
        mec_s = 0.5 * theta_mid + rng.normal(0, 0.03)
        # Clamp to valid range
        first_order = np.clip([vio_s, trl_s, mec_s], 0, 1)
        total_order = np.clip(first_order + rng.uniform(0.02, 0.08, N_PARAMS), 0, 1)
        cc_sobols.append(
            SobolIndices(
                first_order=first_order,
                total_order=total_order,
                parameter_names=PARAM_NAMES,
            )
        )

    # Variance decomposition
    decomp = {
        "total_variance": np.array([0.086, 2e-6]),
        "generation_fraction": np.array([0.0001, 0.0]),
        "seed_fraction": np.array([0.00005, 0.0]),
        "residual_fraction": np.array([0.9999, 1.0]),
        "between_generation_variance": np.array([8.6e-6, 0.0]),
        "between_seed_variance": np.array([4.3e-6, 0.0]),
        "within_group_variance": np.array([0.0859, 2e-6]),
    }

    # Morris indices
    morris = MorrisIndices(
        mu=np.array([0.42, 0.01, 0.38]),
        mu_star=np.array([0.45, 0.03, 0.40]),
        sigma=np.array([0.15, 0.01, 0.22]),
        parameter_names=PARAM_NAMES,
        elementary_effects=rng.normal(0, 0.3, (10, N_PARAMS)),
        n_trajectories=10,
        n_levels=4,
    )

    # Cell cycle relevance
    spectrum = _make_synthetic_spectrum()

    cc_relevance = CellCycleRelevanceResult(
        relevant_observables=OBS_NAMES,
        relevance_scores={
            "listeners__mass__dry_mass": 0.92,
            "listeners__mass__growth": 0.78,
        },
        variance_by_strategy={
            "uniform": np.array([0.086, 2e-6]),
            "by_generation": np.array([8.6e-6, 0.0]),
            "by_seed": np.array([4.3e-6, 0.0]),
        },
        residual_variance_fraction=np.array([0.9999, 1.0]),
    )
    # Attach the spectrum so export can find it
    cc_relevance.koopman_spectrum = spectrum

    # Cell cycle profile — realistic E. coli mass doubling and growth rate dip
    cc_profile = {
        "stages": list(range(N_BINS)),
        "dry_mass_mean": [
            1.027,
            1.104,
            1.179,
            1.263,
            1.350,
            1.443,
            1.543,
            1.653,
            1.767,
            1.908,
        ],
        "growth_mean": [
            0.01086,
            0.01138,
            0.01165,
            0.01145,
            0.01067,
            0.00971,
            0.00879,
            0.00833,
            0.00842,
            0.00894,
        ],
    }

    result = PipelineResult(
        population=UqProfile(
            stratification=StratificationLens.POPULATION,
            sobol_indices=[pop_sobol],
            surrogate=pop_surrogate,
        ),
        cell_cycle=UqProfile(
            stratification=StratificationLens.CELL_CYCLE,
            sobol_indices=cc_sobols,
            surrogate=cc_surrogate,
        ),
        variance_decomposition=decomp,
        morris_indices=morris,
        cell_cycle_relevance=cc_relevance,
        cell_cycle_profile=cc_profile,
    )

    return result, spectrum


class TestExportArtifacts:
    """Generate and export full PipelineResult artifacts for inspection."""

    def test_full_export(self):
        """Generate all export artifacts to ./test_export_output/."""
        result, spectrum = _make_full_pipeline_result()

        # Export via PipelineResult.export()
        result.export(OUTPUT_DIR)

        # Also write the Koopman spectrum PDF directly (since the auto-detection
        # path through cell_cycle_relevance may not find it)
        fig = plot_koopman_spectrum(
            spectrum,
            expected_cycle_time=3600.0,
            observable_names=OBS_NAMES,
        )
        fig.write_image(str(OUTPUT_DIR / "koopman_spectrum.pdf"))
        # Also write as HTML for interactive viewing
        fig.write_html(str(OUTPUT_DIR / "koopman_spectrum.html"))

        # Verify all expected artifacts exist
        assert (OUTPUT_DIR / "metadata.json").exists()
        assert (OUTPUT_DIR / "uq_results.json").exists()
        assert (OUTPUT_DIR / "variance_decomposition.json").exists()
        assert (OUTPUT_DIR / "population_surrogate").exists()
        assert (OUTPUT_DIR / "cell_cycle_surrogate").exists()
        assert (OUTPUT_DIR / "population_sobol").exists()
        assert (OUTPUT_DIR / "morris_indices").exists()
        assert (OUTPUT_DIR / "koopman_spectrum.pdf").exists()
        assert (OUTPUT_DIR / "koopman_spectrum.html").exists()

        for i in range(N_BINS):
            assert (OUTPUT_DIR / f"cell_cycle_sobol_stage_{i}").exists()

        # Validate uq_results.json content
        data = json.loads((OUTPUT_DIR / "uq_results.json").read_text())

        # Check top-level keys
        assert "parameter_names" in data
        assert "phase1_population_sobol" in data
        assert "phase2_cell_cycle_sobol_per_stage" in data
        assert "variance_decomposition" in data
        assert "morris_screening" in data
        assert "cell_cycle_relevance" in data
        assert "surrogates" in data

        # Check Phase 1 Sobol
        p1 = data["phase1_population_sobol"]
        assert set(p1["first_order"].keys()) == set(PARAM_NAMES)
        assert set(p1["total_order"].keys()) == set(PARAM_NAMES)

        # Check Phase 2 per-stage Sobol: must have n_bins entries
        p2 = data["phase2_cell_cycle_sobol_per_stage"]
        assert len(p2) == N_BINS
        for entry in p2:
            assert "stage" in entry
            assert "theta_range" in entry
            assert "first_order" in entry
            assert "total_order" in entry
            assert set(entry["first_order"].keys()) == set(PARAM_NAMES)

        # Check Morris
        assert data["morris_screening"]["parameter_names"] == PARAM_NAMES
        assert len(data["morris_screening"]["mu_star"]) == N_PARAMS

        # Check cell cycle profile
        assert data["cell_cycle_profile"] is not None
        assert data["cell_cycle_profile"]["stages"] == list(range(N_BINS))
        assert len(data["cell_cycle_profile"]["dry_mass_mean"]) == N_BINS
        assert len(data["cell_cycle_profile"]["growth_mean"]) == N_BINS
        # Mass should increase monotonically (healthy cell cycle)
        mass_means = data["cell_cycle_profile"]["dry_mass_mean"]
        assert all(mass_means[i] < mass_means[i + 1] for i in range(len(mass_means) - 1))

        # Check cell cycle relevance
        assert data["cell_cycle_relevance"]["relevant_observables"] == OBS_NAMES

        # Check surrogate summaries
        assert data["surrogates"]["population"]["r_squared"] == pytest.approx(0.92)
        assert data["surrogates"]["cell_cycle"]["r_squared"] == pytest.approx(0.87)

        print(f"\n{'=' * 60}")
        print(f"Export artifacts written to: {OUTPUT_DIR.resolve()}")
        print(f"{'=' * 60}")
        print("Files:")
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                print(f"  {f.relative_to(OUTPUT_DIR)}  ({size:,} bytes)")
        print(f"{'=' * 60}")
