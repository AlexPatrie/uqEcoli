"""SMS API client for TUI using httpx.AsyncClient."""

from enum import StrEnum
from typing import Any

import httpx
from sms_api.simulation.models import (
    HpcRun,
    RegisteredSimulators,
    Simulation,
    SimulationRun,
    Simulator,
    SimulatorVersion,
)


class BaseUrl(StrEnum):
    DEV = "https://sms-dev.cam.uchc.edu"
    LOCAL = "http://localhost:8888"


BASE_URL = BaseUrl.DEV


class SMSAPIClient:
    """Async client for SMS API endpoints."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        """Initialize the SMS API client.

        Args:
            base_url: The base URL of the SMS API server.
        """
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SMSAPIClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the httpx client, ensuring it's initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._client

    async def run_simulation(
        self,
        simulator_id: int,
        experiment_id: str,
        simulation_config_filename: str = "api_simulation_default.json",
        num_generations: int | None = None,
        num_seeds: int | None = None,
        description: str | None = None,
    ) -> Simulation:
        """Run a simulation via POST /api/v1/simulations.

        Args:
            simulator_id: Database ID of an existing simulator (required).
            experiment_id: Unique identifier for the experiment (required).
            simulation_config_filename: Config filename in vEcoli/configs/ on HPC.
            num_generations: Number of generations to simulate (optional, overrides config).
            num_seeds: Number of initial seeds/lineages (optional, overrides config).
            description: Description of the simulation (optional).

        Returns:
            Simulation object with job details.
        """
        params: dict[str, Any] = {
            "simulator_id": simulator_id,
            "experiment_id": experiment_id,
            "simulation_config_filename": simulation_config_filename,
        }

        if num_generations is not None:
            params["num_generations"] = num_generations
        if num_seeds is not None:
            params["num_seeds"] = num_seeds
        if description is not None:
            params["description"] = description

        response = await self.client.post("/api/v1/simulations", params=params)
        response.raise_for_status()
        return Simulation.model_validate(response.json())

    async def get_simulation_status(self, simulation_id: int) -> SimulationRun:
        """Get simulation status via GET /api/v1/simulations/{id}/status.

        Args:
            simulation_id: Database ID of the simulation.

        Returns:
            SimulationRun with id and status.
        """
        response = await self.client.get(f"/api/v1/simulations/{simulation_id}/status")
        response.raise_for_status()
        return SimulationRun.model_validate(response.json())

    async def get_simulation_data(self, simulation_id: int) -> list[dict[str, Any]]:
        """Get simulation data via POST /api/v1/simulations/{id}/data.

        Args:
            simulation_id: Database ID of the simulation.

        Returns:
            List of TSV output file data.
        """
        response = await self.client.post(f"/api/v1/simulations/{simulation_id}/data")
        response.raise_for_status()
        return response.json()

    async def list_simulations(self) -> list[Simulation]:
        """List all simulations via GET /api/v1/simulations.

        Returns:
            List of Simulation objects.
        """
        response = await self.client.get("/api/v1/simulations")
        response.raise_for_status()
        return [Simulation.model_validate(sim) for sim in response.json()]

    # =========================================================================
    # Simulator Endpoints (core router)
    # =========================================================================

    async def get_latest_simulator(
        self,
        git_repo_url: str = "https://github.com/vivarium-collective/vEcoli",
        git_branch: str = "ccam-nextflow",
    ) -> Simulator:
        """Get latest simulator version via GET /core/v1/simulator/latest.

        Args:
            git_repo_url: Git repository URL for the simulator.
            git_branch: Git branch name.

        Returns:
            Simulator with the latest commit hash.
        """
        response = await self.client.get(
            "/core/v1/simulator/latest",
            params={"git_repo_url": git_repo_url, "git_branch": git_branch},
        )
        response.raise_for_status()
        return Simulator.model_validate(response.json())

    async def upload_simulator(
        self,
        git_commit_hash: str,
        git_repo_url: str = "https://github.com/vivarium-collective/vEcoli",
        git_branch: str = "api-support",
    ) -> SimulatorVersion:
        """Upload a new simulator version via POST /core/v1/simulator/upload.

        Args:
            git_commit_hash: Git commit hash (first 7 characters).
            git_repo_url: Git repository URL.
            git_branch: Git branch name.

        Returns:
            SimulatorVersion with database ID.
        """
        payload = {
            "git_commit_hash": git_commit_hash,
            "git_repo_url": git_repo_url,
            "git_branch": git_branch,
        }
        response = await self.client.post("/core/v1/simulator/upload", json=payload)
        response.raise_for_status()
        return SimulatorVersion.model_validate(response.json())

    async def get_simulator_status(self, simulator_id: int) -> HpcRun:
        """Get simulator build status via GET /core/v1/simulator/status.

        Args:
            simulator_id: Database ID of the simulator.

        Returns:
            HpcRun with build status information.
        """
        response = await self.client.get(
            "/core/v1/simulator/status",
            params={"simulator_id": simulator_id},
        )
        response.raise_for_status()
        return HpcRun.model_validate(response.json())

    async def list_simulator_versions(self) -> RegisteredSimulators:
        """List all registered simulator versions via GET /core/v1/simulator/versions.

        Returns:
            RegisteredSimulators with list of available simulator versions.
        """
        response = await self.client.get("/core/v1/simulator/versions")
        response.raise_for_status()
        return RegisteredSimulators.model_validate(response.json())
