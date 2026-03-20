"""
UQ Package Test Suite

This test suite provides comprehensive coverage of the UQ framework and
explicitly verifies compliance with Milestone 08.4.2 requirements.

Test Modules
------------
test_milestone_084_2.py
    Explicit verification of each Milestone 08.4.2 requirement
test_inputs.py
    Input parameter definitions (SimDataParameter specs)
test_aggregation.py
    Four aggregation strategies
test_sensitivity.py
    PCE-based sensitivity analysis
test_cell_cycle.py
    Cell cycle stratification
test_koopman.py
    Koopman spectral analysis
test_e2e.py
    End-to-end workflow tests with synthetic data
test_real_data.py
    Integration tests using real simulation data from api_simulation_default

Running Tests
-------------
From the repository root:

    # Run all UQ tests
    uv run pytest uq/tests/ -v

    # Run only milestone verification tests
    uv run pytest uq/tests/test_milestone_084_2.py -v

    # Run only real data tests
    uv run pytest uq/tests/test_real_data.py -v

    # Run with coverage
    uv run pytest uq/tests/ --cov=uq --cov-report=html

    # Run e2e tests only
    uv run pytest uq/tests/test_e2e.py -v

    # Run tests by marker
    uv run pytest uq/tests/ -m real_data      # Real data tests
    uv run pytest uq/tests/ -m milestone      # Milestone verification
    uv run pytest uq/tests/ -m e2e            # End-to-end tests
    uv run pytest uq/tests/ -m unit           # Unit tests

Real Data Loading
-----------------
Real simulation data is loaded from api_simulation_default using:

    from uq.inputs import load_dataset
    df = load_dataset("api_simulation_default")

The data is located at: api_integration/sims/api_simulation_default/history/
"""
