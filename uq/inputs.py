"""
Input parameter definitions for uncertainty quantification.

This module defines the scientifically most relevant input variables for UQ:
- Violacein (vio) pathway presence
- Mecillinam condition
- Gene knockouts

These inputs are parametrized for use with UQPy/PyTUQ sensitivity analysis libraries.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, override

import numpy as np
import polars

from uq.models import UQInputParameters

if TYPE_CHECKING:
    pass


class InputParameterSpace(abc.ABC):
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]
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
        **kwargs,
    ) -> None:
        self.parameter_names = parameter_names or []
        self.parameter_bounds = parameter_bounds or []
        self.parameter_types = parameter_types or []
        self.implementation_init(*args, **kwargs)

    def implementation_init(self, *args, **kwargs) -> None:
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


class InputParameterSpaceVecoli(InputParameterSpace):
    parameter_names: list[str]
    parameter_bounds: list[tuple[float, float]]
    parameter_types: list[Literal["continuous", "discrete", "categorical"]]
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
        vio_expression_bounds: tuple[float, float] = (0.0, 5.0),
        vio_trl_eff_bounds: tuple[float, float] = (0.0, 2.0),
        mecillinam_conc_bounds: tuple[float, float] = (0.0, 10.0),
        include_vio: bool = True,
        include_mecillinam: bool = True,
        knockout_genes: Optional[list[str]] = None,
    ):
        """
        Initialize the input parameter space.

        Args:
            vio_expression_bounds: (min, max) bounds for vio expression factor
            vio_trl_eff_bounds: (min, max) bounds for vio translation efficiency
            mecillinam_conc_bounds: (min, max) bounds for mecillinam concentration
            include_vio: Whether to include vio parameters in the space
            include_mecillinam: Whether to include mecillinam parameters
            knockout_genes: List of genes that can be knocked out (discrete parameter)
        """
        self.parameter_names = []
        self.parameter_bounds = []
        self.parameter_types = []

        if include_vio:
            self.parameter_names.extend(["vio_expression", "vio_trl_eff"])
            self.parameter_bounds.extend([vio_expression_bounds, vio_trl_eff_bounds])
            self.parameter_types.extend(["continuous", "continuous"])

        if include_mecillinam:
            self.parameter_names.append("mecillinam_concentration")
            self.parameter_bounds.append(mecillinam_conc_bounds)
            self.parameter_types.append("continuous")

        self.knockout_genes = knockout_genes or []
        self._include_vio = include_vio
        self._include_mecillinam = include_mecillinam

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
    ) -> UQInputParameters:
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
        params = UQInputParameters(seed=seed, generations=generations)

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
    def params_to_sample(self, params: UQInputParameters) -> np.ndarray:
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
