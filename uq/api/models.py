import abc
import json
from os import PathLike
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseDatamodel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SystemComponent(BaseDatamodel):
    name: str
    attr_path: str
    bounds: tuple[float | int | complex | Any, float | int | complex | Any]
    description: str

    @classmethod
    def from_config(cls, p: PathLike) -> list["SystemComponent"]:
        with open(str(p), "r") as f:
            d: list[dict] = json.load(f)
        return [cls.from_dict(conf) for conf in d]

    @classmethod
    def from_dict(cls, d) -> "SystemComponent":
        """
        :param d: for example:
        {
            "name": "fraction_active_rnap_free",
            "attr_path": "process.transcription.fraction_active_rnap_free",
            "bounds": [0.25, 0.47],
            "description": "RNAP active fraction when ppGpp-free (baseline ~0.36)"
        }
        """
        return cls(**d)


class Parameter(SystemComponent):
    """Uq pipeline/sampling input.

    Attributes:
        name: (str)
        attr_path: (str) dot separated path expressing nesting with sim_data
        bounds: (tuple[float | int | complex, float | int | complex])
        description: (str)

    Methods:
        from_dict(:param d(dict):)
        from_config(:param p(PathLike):)
    """
    pass


class Observable(SystemComponent):
    """Uq pipeline timeseries observable

    Attributes:
        name: (str)
        attr_path: (str) dot separated path expressing nesting in timeseries output (cell state)
        bounds: (tuple[float | int | complex, float | int | complex])
        description: (str)
    Methods:
        from_dict(:param d(dict):)
        from_config(:param p(PathLike):)
        """
    pass


class SystemConfig(BaseDatamodel):
    parameters: list[Parameter] = []
    observables: list[Observable] | None = None


class UqConfig(BaseDatamodel):
    experiment_ids: list[str]
    system: SystemConfig
    n_samples: int
    seed: int
    max_duration: float
    generations: int
    max_workers: int


def test_params_from_config() -> None:
    p = "examples/uq_artifacts/params/params_demo.json"
    params = Parameter.from_config(p)
    print()