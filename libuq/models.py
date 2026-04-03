from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Re-export PCE and pipeline models so tests/consumers can import from uq.models
from libuq.pce.models import (
    Parameter,
    PCEConfig,
    PCEFitResult,
    PCEParameterSelectionConfig,
    PCEPreprocessingConfig,
    PCESolverConfig,
    PCESurrogateConfig,
)
from libuq.pipeline.models import (
    CellCyclePhase,
    CellCycleVariable,
    MediaCondition,
)


class _BaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SamplingConfig(_BaseModel):
    """
    Attributes:
        cache_dir: str | None = None
        n_samples: int = 200
        seed: int = 42
        max_workers: int | None = None
        generations: int = 1
        live: bool = True
    """

    cache_dir: str | None = None
    n_samples: int = 200
    seed: int = 42
    max_workers: int | None = None
    generations: int = 1
    live: bool = True


class XPrescreenConfig(_BaseModel):
    """
    Used to configure logic that selects most relevant params from x,
    particularly configuring Morris.

    Attributes:
        n_trajectories: n trajectories used in morris prescreening for param selection.
        n_top: number of most relevant parameters to select and return
    """

    n_trajectories: int | None = None
    n_top: int | None = None  # how many params to keep

    def init_defaults(self) -> None:
        if self.n_trajectories is None:
            self.n_trajectories = 20
        if self.n_top is None:
            self.n_top = 5


class PipelineConfig(_BaseModel):
    """
    Object which represents a full ./uq workflow.

    Attributes:
        experiment_ids: list[str]
        sim_base_path: str | Path
        samples: uq.models.SamplingConfig
        observable_columns: list[str] | None = None
        lb_generation: int | None = 2
        lb_time: float | None = 100.0
        n_bins: int = 10
        polynomial_order: int = 3
        n_samples: int = 200
        expected_cycle_time: float = 3600.0
        max_duration: float = 10800.0
        prescreen_config: uq.models.XPrescreenConfig | None = None
        export_path: Path | None = None
        precomputed_path: Path | str | None = None
        sim_config_path: str | None = None
        init: bool = True
    """

    experiment_ids: list[str]
    sim_base_path: str
    samples: SamplingConfig
    observable_columns: list[str] | None = None
    lb_generation: int | None = 2
    lb_time: float | None = 100.0
    n_bins: int = 10
    polynomial_order: int = 3
    n_samples: int = 200
    expected_cycle_time: float = 3600.0
    max_duration: float = 10800.0
    prescreen_config: XPrescreenConfig | None = None
    export_path: str | None = None
    precomputed_path: str | None = None
    sim_config_path: str | None = None
    init: bool = True
