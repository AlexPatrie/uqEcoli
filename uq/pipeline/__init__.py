from uq.pipeline.models import (
    Pipeline,
    PipelineConfig,
    PipelineResult,
    StratificationLens,
    UqProfile,
)


def __getattr__(name):
    """Lazy imports to avoid circular dependency with uq package."""
    _workflow_exports = {
        "AggregationResult",
        "Strategy4Wrapper",
        "aggregate_timeseries",
        "compute_strategy4_sobol",
        "create_surrogate",
        "execute_pipeline",
        "execute_pipeline_async",
        "get_variance_decomposition",
        "run_phase1",
        "run_phase2",
    }
    if name in _workflow_exports:
        from uq.pipeline import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module 'uq.pipeline' has no attribute {name!r}")


__all__ = [
    "AggregationResult",
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "StratificationLens",
    "Strategy4Wrapper",
    "UqProfile",
    "aggregate_timeseries",
    "compute_strategy4_sobol",
    "create_surrogate",
    "execute_pipeline",
    "execute_pipeline_async",
    "get_variance_decomposition",
    "run_phase1",
    "run_phase2",
]
