"""
Input parameter definitions for uncertainty quantification.

This module defines the scientifically most relevant input variables for UQ:
- Violacein (vio) pathway presence
- Mecillinam condition
- Gene knockouts

These inputs are parametrized for use with UQPy/PyTUQ sensitivity analysis libraries.
"""

import abc
import pprint
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, override

import numpy as np
import polars
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uq.pce.models import Parameter
from uq.pipeline.models import UQInputParametersVecoli, UQInputParameters

if TYPE_CHECKING:
    pass


console = Console()


class XSpaceInterface(abc.ABC):
    """
    Interface whose implementations fulfill:
        - sample_to_params() -> UQInputParameters: Convert a sample from the model/simulation/function-specific parameter space to UQInputParameters.
        - params_to_sample() -> np.ndarray[ParameterValue]: Convert domain-specific input space-mapped UQInputParameters to an array of sample(perturbation) values 
            when N sample values (sample size) == N perturbations.
    """

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs) -> UQInputParameters:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            **kwargs: implementation-specific

        Returns:
            UQInputParameters instance
        """
        pass

    @abc.abstractmethod
    def params_to_sample(self, params: UQInputParameters) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        pass


class XSpace(XSpaceInterface):
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]
    experiment_id: str | None
    """
    Defines the parameter space for UQ sensitivity analysis.

    This class provides methods to sample input parameters and to convert
    between the UQ library format (numpy arrays) and UQInputParameters.

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
        # self.define_parameters()
        return None

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the space."""
        return len(self.parameter_names)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Parameter bounds as numpy array for UQPy.

        Returns:
            Array of shape (n_parameters, 2) with [lower, upper] bounds
        """
        return np.array(self.parameter_bounds)

    # @abc.abstractmethod
    # def define_parameters(self, *args, **kwargs) -> list[Parameter]:
    #     pass

    @abc.abstractmethod
    def sample_to_params(self, sample: np.ndarray, **kwargs) -> UQInputParameters:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            **kwargs: implementation-specific

        Returns:
            UQInputParameters instance
        """
        pass

    @abc.abstractmethod
    def params_to_sample(self, params: UQInputParametersVecoli) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        pass

    def get_uqpy_distributions(self) -> list[Any]:
        """
        Get UQPy distribution objects for this parameter space.

        Returns:
            List of UQPy Distribution objects (Uniform distributions)
        """
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get PyTUQ-compatible bounds arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]

    @property
    def parameters(self) -> dict[str, dict[list[float], str]]:
        params = {}
        # TODO: enable ragged shape, for now symmetry required for construction
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
        # Neon 90s header
        title = Text()
        title.append("⚡ ", style="bold yellow")
        title.append("INPUT PARAMETER SPACE", style="bold magenta")
        title.append("  //  ", style="dim cyan")
        title.append(self.experiment_id, style="bold cyan")

        # Parameter table
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            pad_edge=False,
        )
        table.add_column("PARAM", style="bold yellow", no_wrap=True)
        table.add_column("VALUE", style="bright_white")
        table.add_column("TYPE", style="dim cyan")

        for name, val in self.parameters.items():
            table.add_row(name, str(val), type(val).__name__)

        panel = Panel(
            table,
            title=title,
            subtitle=Text(f"n = {self.n_parameters} parameters", style="bold green"),
            border_style="magenta",
            box=box.DOUBLE_EDGE,
            padding=(0, 1),
        )

        console.print(panel)


class XSpaceVecoli(XSpaceInterface):
    """
    Defines the parameter space for UQ sensitivity analysis pipeline
    on vEcoli datasets.

    This class provides methods to sample input parameters and to convert
    between the UQ library format (numpy arrays) and UQInputParameters.

    Kwargs:
        `vio_expression_bounds: tuple[float, float] = (low, high)`
        `vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0)`
        `mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0)`
        `include_vio: bool = True`
        `include_mecillinam: bool = True`
        `knockout_genes: Optional[list[str]] = None`
    Attributes:
        parameter_names: Names of the parameters being varied
        parameter_bounds: Lower and upper bounds for each parameter
        parameter_types: Type of each parameter ('continuous', 'discrete', 'categorical')
    """

    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]

    def implementation_init(self, *args, **kwargs) -> None:
        if kwargs.get("include_vio"):
            self.parameter_names.extend(["vio_expression", "vio_trl_eff"])
            self.parameter_bounds.extend([
                kwargs.get("vio_expression_bounds", (0.0, 5.0)),
                kwargs.get("vio_trl_eff_bounds", (0.0, 2.0)),
            ])
            self.parameter_types.extend(["continuous", "continuous"])

        if kwargs.get("include_mecillinam"):
            self.parameter_names.append("mecillinam_concentration")
            self.parameter_bounds.append(kwargs.get("mecillinam_conc_bounds", (0.0, 10.0)))
            self.parameter_types.append("continuous")

        self.knockout_genes = kwargs.get("knockout_genes", [])
        self._include_vio = kwargs.get("include_vio", False)
        self._include_mecillinam = kwargs.get("include_mecillinam", False)

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the space."""
        return len(self.parameter_names)

    @property
    def bounds_array(self) -> np.ndarray:
        """
        Parameter bounds as numpy array for UQPy.

        Returns:
            Array of shape (n_parameters, 2) with [lower, upper] bounds
        """
        return np.array(self.parameter_bounds)

    @override
    def sample_to_params(
        self,
        sample: np.ndarray,
        seed: int = 0,
        generations: int = 8,
        knockouts: Optional[list[str]] = None,
    ) -> UQInputParametersVecoli:
        """
        Convert a sample from the parameter space to UQInputParameters.

        Args:
            sample: Array of parameter values in the same order as parameter_names
            seed: Random seed for the simulation
            generations: Number of generations to simulate
            knockouts: List of genes to knock out (optional)

        Returns:
            UQInputParameters instance
        """
        params = UQInputParametersVecoli(seed=seed, generations=generations)

        idx = 0
        if self._include_vio:
            params.vio.expression = float(sample[idx])
            params.vio.translation_efficiency = float(sample[idx + 1])
            params.vio.enabled = True
            idx += 2
        else:
            params.vio.enabled = False

        if self._include_mecillinam:
            # Apply concentration at time 0
            params.mecillinam.times = [0.0]
            params.mecillinam.concentrations = [float(sample[idx])]
            idx += 1

        if knockouts:
            params.knockouts.translation_knockouts = knockouts

        return params

    @override
    def params_to_sample(self, params: UQInputParametersVecoli) -> np.ndarray:
        """
        Convert UQInputParameters to a sample array.

        Args:
            params: UQInputParameters instance

        Returns:
            Array of parameter values
        """
        sample: list[float] = []

        if self._include_vio:
            sample.append(params.vio.expression)
            sample.append(params.vio.translation_efficiency)

        if self._include_mecillinam:
            # Use the first concentration value
            conc = params.mecillinam.concentrations[0] if params.mecillinam.concentrations else 0.0
            sample.append(conc)

        return np.array(sample)

    def get_uqpy_distributions(self) -> list[Any]:
        """
        Get UQPy distribution objects for this parameter space.

        Returns:
            List of UQPy Distribution objects (Uniform distributions)
        """
        try:
            from UQpy.distributions import Uniform
        except ImportError:
            raise ImportError("UQPy is required for sensitivity analysis. Install it with: pip install UQpy")

        distributions = []
        for lb, ub in self.parameter_bounds:
            distributions.append(Uniform(loc=lb, scale=ub - lb))
        return distributions

    def get_pytuq_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get PyTUQ-compatible bounds arrays.

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays
        """
        bounds = self.bounds_array
        return bounds[:, 0], bounds[:, 1]


def _get_repo_root() -> Path:
    """Get the repository root directory."""
    # Try to find repo root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


def load_dataset(
    experiment_id: str,
    outdir_root: Path | None = None,
    observables: list[str] | None = None,
    include_metadata: bool = True,
) -> polars.DataFrame:
    """
    Load simulation dataset from parquet files.

    Args:
        experiment_id: The experiment identifier (e.g., "api_simulation_default")
        outdir_root: Root directory for simulation outputs. Defaults to
                     {repo_root}/api_integration/sims
        observables: Optional list of column names to select
        include_metadata: If True, include hive partition columns (variant, lineage_seed,
                         generation, agent_id) from the directory structure

    Returns:
        Polars DataFrame with the simulation data
    """
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"

    # Scan nested parquet files with hive partitioning to extract metadata columns
    # (variant, lineage_seed, generation, agent_id) from directory structure
    lf = polars.scan_parquet(str(base_path / "**/*.pq"), hive_partitioning=include_metadata)

    if observables is not None:
        # Filter to only existing columns, but always include metadata if requested
        available = lf.collect_schema().names()
        valid_observables = [col for col in observables if col in available]

        # Add metadata columns if they exist and include_metadata is True
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
    """
    Get list of available columns in the dataset.

    Args:
        experiment_id: The experiment identifier
        outdir_root: Root directory for simulation outputs. Defaults to
                     {repo_root}/api_integration/sims

    Returns:
        List of column names
    """
    if outdir_root is None:
        outdir_root = _get_repo_root() / "api_integration" / "sims"

    base_path = Path(outdir_root) / experiment_id / "history" / f"experiment_id={experiment_id}"
    lf = polars.scan_parquet(str(base_path) + "/**/*.pq")
    return lf.collect_schema().names()
