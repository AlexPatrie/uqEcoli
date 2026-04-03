"""
Input parameter definitions for uncertainty quantification.

This module defines the parameter space for UQ sensitivity analysis.
Parameters are specified as a list of ``SimDataParameter`` specs, each
identifying a scalar attribute in ``SimulationDataEcoli`` by dot-path.
No vEcoli variant functions required — sim_data attributes are set
directly via ``setattr``.
"""

import abc
from pathlib import Path
from typing import Any, Literal, override

import numpy as np
import polars
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from libuq.pce.models import Parameter
from libuq.pipeline.models import (
    GenericSimDataParams,
    SimDataParameter,
    UQInputParameters,
)

console = Console()


class XSpaceInterface(abc.ABC):
    """
    Interface whose implementations fulfill:
        - sample_to_params() -> UQInputParameters | GenericSimDataParams
        - params_to_sample() -> np.ndarray
    """

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs):
        pass

    @abc.abstractmethod
    def params_to_sample(self, params) -> np.ndarray:
        pass


class XSpace(XSpaceInterface):
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]
    experiment_id: str | None
    """
    Defines the parameter space for UQ sensitivity analysis.

    Attributes:
        parameter_names: Names of the parameters being varied
        parameter_bounds: Lower and upper bounds for each parameter
        parameter_types: Type of each parameter ('continuous', 'discrete', 'categorical')
    """

    def __init__(
        self,
        *args,
        parameter_names: list[str] | None = None,
        parameter_bounds: list[tuple[float, float]] | None = None,
        parameter_types: list[Literal["continuous", "discrete", "categorical"]] | None = None,
        experiment_id: str | None = None,
        **kwargs,
    ) -> None:
        self.parameter_names = parameter_names or []
        self.parameter_bounds = parameter_bounds or []
        self.parameter_types = parameter_types or []
        self.experiment_id = experiment_id

        self.implementation_init(*args, **kwargs)

    def implementation_init(self, *args, **kwargs) -> None:
        return None

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the space."""
        return len(self.parameter_names)

    @property
    def bounds_array(self) -> np.ndarray:
        """Parameter bounds as numpy array, shape (n_parameters, 2)."""
        return np.array(self.parameter_bounds)

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs):
        pass

    @abc.abstractmethod
    def params_to_sample(self, params) -> np.ndarray:
        pass

    def get_uqpy_distributions(self) -> list[Any]:
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]

    @property
    def parameters(self) -> dict[str, dict[list[float], str]]:
        params = {}
        if all(
            list(
                map(
                    lambda c: len(c) > 0 and len(c) == len(self.parameter_names),
                    [self.parameter_bounds, self.parameter_names, self.parameter_types],
                )
            )
        ):
            for i, name in enumerate(self.parameter_names):
                params[name] = {"bounds": self.parameter_bounds[i], "type": self.parameter_types[i]}
        return params

    def __repr__(self) -> str:
        self.show()
        return f"<ParameterSpace '{self.experiment_id}' n={self.n_parameters}>"

    def show(self):
        title = Text()
        title.append("INPUT PARAMETER SPACE", style="bold magenta")
        title.append("  //  ", style="dim cyan")
        title.append(self.experiment_id or "(no experiment)", style="bold cyan")

        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            pad_edge=False,
        )
        table.add_column("PARAM", style="bold yellow", no_wrap=True)
        table.add_column("BOUNDS", style="bright_white")
        table.add_column("TYPE", style="dim cyan")

        for name, val in self.parameters.items():
            table.add_row(name, str(val.get("bounds", "")), val.get("type", ""))

        panel = Panel(
            table,
            title=title,
            subtitle=Text(f"n = {self.n_parameters} parameters", style="bold green"),
            border_style="magenta",
            box=box.DOUBLE_EDGE,
            padding=(0, 1),
        )

        console.print(panel)


class XSpaceVecoli(XSpace):
    """Parameter space for UQ sensitivity analysis on vEcoli datasets.

    Accepts a list of ``SimDataParameter`` specs via the ``parameters``
    kwarg. Each spec identifies a scalar attribute in
    ``SimulationDataEcoli`` by dot-path. ``sample_to_params()`` returns
    ``GenericSimDataParams``, and mutations are applied directly via
    ``setattr``.

    Kwargs:
        parameters: list[SimDataParameter]
    """

    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]

    def implementation_init(self, *args, **kwargs) -> None:
        sim_data_parameters: list[SimDataParameter] = kwargs.get("parameters", [])
        self._sim_data_parameters = sim_data_parameters
        for p in sim_data_parameters:
            self.parameter_names.append(p.name)
            self.parameter_bounds.append(p.bounds)
            self.parameter_types.append("continuous")

    @override
    def sample_to_params(
        self,
        sample: np.ndarray,
        seed: int = 0,
        generations: int = 8,
        **kwargs,
    ) -> GenericSimDataParams:
        """Convert a sample vector to a ``GenericSimDataParams`` container."""
        values = {spec.name: float(sample[i]) for i, spec in enumerate(self._sim_data_parameters)}
        return GenericSimDataParams(
            parameter_specs=self._sim_data_parameters,
            values=values,
            seed=seed,
            generations=generations,
        )

    @override
    def params_to_sample(self, params: GenericSimDataParams) -> np.ndarray:
        """Convert a ``GenericSimDataParams`` container back to a sample array."""
        return np.array([params.values[spec.name] for spec in params.parameter_specs])

    def get_uqpy_distributions(self) -> list[Any]:
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]


# -- Dataset loading helpers ------------------------------------------------


def _get_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_dataset(
    experiment_id: str,
    outdir_root: Path | None = None,
    observables: list[str] | None = None,
    include_metadata: bool = True,
) -> polars.DataFrame:
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"
    lf = polars.scan_parquet(str(base_path / "**/*.pq"), hive_partitioning=include_metadata)

    if observables is not None:
        available = lf.collect_schema().names()
        valid_observables = [col for col in observables if col in available]

        if include_metadata:
            metadata_cols = ["variant", "lineage_seed", "generation", "agent_id"]
            for col in metadata_cols:
                if col in available and col not in valid_observables:
                    valid_observables.append(col)

        if valid_observables:
            lf = lf.select(valid_observables)

    return lf.collect()


def get_available_columns(
    experiment_id: str,
    outdir_root: Path | None = None,
) -> list[str]:
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"
    lf = polars.scan_parquet(str(base_path) + "/**/*.pq")
    return lf.collect_schema().names()
