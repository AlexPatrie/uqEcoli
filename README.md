# vEcoli Uncertainty Quantification (UQ) Framework

Sensitivity analysis and uncertainty quantification for [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell simulations, built on Polynomial Chaos Expansion (PCE) surrogates and Sobol variance decomposition.

**RFC006 — Milestone 08.4.2**

## What It Does

The framework answers two questions about your simulation outputs:

1. **Population level** — Which input parameters drive the most variance in bulk outputs (e.g., dry mass, growth rate)?
2. **Cell cycle level** — How does parameter importance shift across cell cycle stages?

It does this through a 7-step pipeline:

```
Parameter Space → Load Data → Aggregate (4 strategies) → Variance Decomposition
                                                                │
Sobol Indices  ←  PCE Surrogate  ←  Morris Screening  ←────────┘
```

## Installation

```bash
uv sync
```

## Quick Start

```python
from uq import XSpaceVecoli
from uq.pipeline.workflow import execute_pipeline

# 1. Define input parameter space
param_space = XSpaceVecoli(
    include_vio=True,
    include_mecillinam=True,
    vio_expression_bounds=(0.0, 5.0),
    vio_trl_eff_bounds=(0.0, 2.0),
    mecillinam_conc_bounds=(0.0, 10.0),
)

# 2. Run the full pipeline — data loading, aggregation, variance
#    decomposition, PCE surrogate, and Sobol indices are all handled
#    internally. Just point it at your simulation output directory.
result = execute_pipeline(
    param_space=param_space,
    simulation_func=your_simulation_wrapper,  # callable with evaluate_batch(X) → Y
    experiment_id="mecillinam",
    sim_base_path="/path/to/vEcoli/api_integration/sims",
    output_types=["higher_order_properties", "exchange_fluxes"],
    generation_lower_bound=2,       # skip initial transient generations
    time_lower_bound=100.0,         # skip early transient timesteps
    polynomial_order=3,
    n_samples=200,
    export_path="./uq_results",
)

# 3. Inspect population-level results (Phase 1)
sobol = result.population.sobol_indices[0]
for name, value in sobol.select(n=5):
    print(f"{name}: {value:.4f}")

# 4. Variance decomposition (Step 4)
print(f"Generation: {result.variance_decomposition['generation_fraction']}")
print(f"Seed:       {result.variance_decomposition['seed_fraction']}")

# 5. Inspect per-cell-cycle-stage results (Phase 2)
for i, stage_sobol in enumerate(result.cell_cycle.sobol_indices):
    print(f"Stage {i}: {stage_sobol.select(n=3)}")

# 6. Reload results later
from uq.pipeline.models import PipelineResult
loaded = PipelineResult.from_export("./uq_results")
```

See [`examples/uq_pipeline.py`](examples/uq_pipeline.py) for a fully runnable end-to-end example.

## Pipeline Output

The pipeline produces a `PipelineResult` with two profiles:

| Profile | What it tells you | Key outputs |
|---------|-------------------|-------------|
| **Population** (Phase 1) | Which parameters drive bulk variance | 1 set of Sobol indices + PCE surrogate |
| **Cell Cycle** (Phase 2) | How sensitivity varies across the cell cycle | *n* sets of Sobol indices (one per stage) + PCE surrogate |

## Input Parameters

| Parameter | Description |
|-----------|-------------|
| Vio expression factor | Violacein pathway gene expression level |
| Vio translation efficiency | Translation efficiency for vio pathway |
| Mecillinam concentration | Antibiotic concentration timeline |
| Gene knockouts | Genes deleted or silenced at translation level |

## Key Outputs Analyzed

- **Transcriptome** — mRNA cistron counts
- **Proteome** — monomer counts
- **Metabolic fluxes** — FBA exchange/internal fluxes
- **Cell properties** — mass, volume, growth rate

## Documentation

| Resource | Location |
|----------|----------|
| Full RFC specification | [`readmes/RFC006.md`](readmes/RFC006.md) |
| Extended technical context | [`readmes/CONTEXT.md`](readmes/CONTEXT.md) |
| Sphinx docs | [`docs/`](docs/) |
| End-to-end example | [`examples/uq_pipeline.py`](examples/uq_pipeline.py) |
| Tutorials (Marimo) | [`tutorials/`](tutorials/) |

## Running Tests

```bash
uv run pytest tests/
```

## Dependencies

Built on [UQPy](https://uqpyproject.readthedocs.io/) and [PyTUQ](https://sandialabs.github.io/pytuq/) for PCE construction and Sobol analysis, with [Polars](https://pola.rs/) and [DuckDB](https://duckdb.org/) for data handling.
