"""MCP tools for the Atlantis TUI that wrap SMS API endpoints via HTTP."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import FastMCP

from sms_api.tui.client import BASE_URL, SMSAPIClient

logger = logging.getLogger(__name__)

# Create the MCP server for TUI tools - uses HTTP client, not in-process FastAPI
mcp = FastMCP(
    name="Atlantis TUI Tools",
    instructions="""You are an assistant for the SMS (Simulating Microbial Systems) API.
You help users manage vEcoli whole-cell simulations including:
- Managing simulator versions (upload, check status, list)
- Running simulations and checking their status
- Running parameter calculations (parca)
- Retrieving simulation data

When asked to poll or monitor status until complete, check the status repeatedly
with appropriate delays until the job reaches COMPLETED or FAILED state.""",
)


def _format_result(data: Any) -> str:
    """Format API result as readable string."""
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=2, default=str)
    if isinstance(data, list):
        return json.dumps([d.model_dump() if hasattr(d, "model_dump") else d for d in data], indent=2, default=str)
    return json.dumps(data, indent=2, default=str)


class ToolContext:
    """Context for tool execution with API URL configuration."""

    def __init__(self, api_url: str = BASE_URL) -> None:
        self.api_url = api_url


# Global context - can be updated by the TUI
_context = ToolContext()


def set_api_url(url: str) -> None:
    """Set the API URL for all tools."""
    _context.api_url = url


def get_api_url() -> str:
    """Get the current API URL."""
    return _context.api_url


@mcp.tool()
async def get_latest_simulator(
    git_repo_url: str = "https://github.com/vivarium-collective/vEcoli",
    git_branch: str = "ccam-nextflow",
) -> str:
    """Get the latest simulator version (commit hash) from a git repository.

    Args:
        git_repo_url: The GitHub repository URL for vEcoli.
        git_branch: The branch name to check.

    Returns:
        JSON with the latest commit hash, repo URL, and branch.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.get_latest_simulator(git_repo_url=git_repo_url, git_branch=git_branch)
        return _format_result(result)


@mcp.tool()
async def upload_simulator(
    git_commit_hash: str,
    git_repo_url: str = "https://github.com/vivarium-collective/vEcoli",
    git_branch: str = "ccam-nextflow",
) -> str:
    """Upload a new simulator version to build its container image.

    This registers the simulator in the database and starts building the
    Singularity container image on the HPC cluster.

    Args:
        git_commit_hash: The git commit hash (first 7 characters) to build.
        git_repo_url: The GitHub repository URL for vEcoli.
        git_branch: The branch name.

    Returns:
        JSON with the new simulator's database ID and details.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.upload_simulator(
            git_commit_hash=git_commit_hash,
            git_repo_url=git_repo_url,
            git_branch=git_branch,
        )
        return _format_result(result)


@mcp.tool()
async def get_simulator_status(simulator_id: int) -> str:
    """Get the build status of a simulator by its database ID.

    The status indicates if the Singularity container build is:
    - PENDING: Not yet started
    - RUNNING: Build in progress
    - COMPLETED: Build finished successfully
    - FAILED: Build failed

    Args:
        simulator_id: The database ID of the simulator.

    Returns:
        JSON with the HPC run status including SLURM job ID and state.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.get_simulator_status(simulator_id)
        return _format_result(result)


@mcp.tool()
async def list_simulator_versions() -> str:
    """List all registered simulator versions.

    Returns:
        JSON array of all simulators with their database IDs, commit hashes,
        repo URLs, and branches.
    """
    async with SMSAPIClient(_context.api_url) as client:
        response = await client.client.get("/core/v1/simulator/versions")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


@mcp.tool()
async def run_ecoli_simulation(
    experiment_id: str,
    simulator_id: int,
    simulation_config_filename: str = "default.json",
    num_generations: int | None = None,
    num_seeds: int | None = None,
    description: str | None = None,
) -> str:
    """Run a new vEcoli simulation.

    Args:
        experiment_id: Unique identifier for this experiment.
        simulator_id: Database ID of the simulator to use.
        simulation_config_filename: Config filename in vEcoli/configs/ on HPC.
        num_generations: Number of generations to simulate (1-10).
        num_seeds: Number of initial seeds/lineages (1-100).
        description: Optional description of the simulation.

    Returns:
        JSON with the new simulation's database ID and details.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.run_simulation(
            experiment_id=experiment_id,
            simulator_id=simulator_id,
            num_generations=num_generations,
            num_seeds=num_seeds,
        )
        return _format_result(result)


@mcp.tool()
async def get_ecoli_simulation(simulation_id: int) -> str:
    """Get a simulation by its database ID.

    Args:
        simulation_id: The database ID of the simulation.

    Returns:
        JSON with the simulation details.
    """
    async with SMSAPIClient(_context.api_url) as client:
        response = await client.client.get(f"/api/v1/simulations/{simulation_id}")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, default=str)


@mcp.tool()
async def get_ecoli_simulation_status(simulation_id: int) -> str:
    """Get the status of a simulation by its database ID.

    The status indicates if the simulation is:
    - PENDING: Not yet started
    - RUNNING: Simulation in progress
    - COMPLETED: Simulation finished successfully
    - FAILED: Simulation failed

    Args:
        simulation_id: The database ID of the simulation.

    Returns:
        JSON with the simulation run status.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.get_simulation_status(simulation_id)
        return _format_result(result)


@mcp.tool()
async def list_ecoli_simulations() -> str:
    """List all simulations.

    Returns:
        JSON array of all simulations with their details.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.list_simulations()
        return _format_result(result)


@mcp.tool()
async def get_ecoli_simulation_data(simulation_id: int) -> str:
    """Get the output data from a completed simulation.

    Args:
        simulation_id: The database ID of the simulation.

    Returns:
        JSON with TSV output file contents.
    """
    async with SMSAPIClient(_context.api_url) as client:
        result = await client.get_simulation_data(simulation_id)
        return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def run_parca(
    simulator_id: int,
    cpus: int = 3,
) -> str:
    """Run a parameter calculation (parca) for a simulator.

    Parca generates the simulation dataset needed before running simulations.

    Args:
        simulator_id: Database ID of the simulator to use.
        cpus: Number of CPUs for the calculation (default: 3).

    Returns:
        JSON with the new parca dataset's database ID and details.
    """
    async with SMSAPIClient(_context.api_url) as client:
        # First get the simulator to get its details
        sim_response = await client.client.get("/core/v1/simulator/versions")
        sim_response.raise_for_status()
        simulators = sim_response.json().get("versions", [])

        simulator = None
        for s in simulators:
            if s.get("database_id") == simulator_id:
                simulator = s
                break

        if simulator is None:
            return json.dumps({"error": f"Simulator with ID {simulator_id} not found"})

        payload = {
            "simulator_version": {
                "git_commit_hash": simulator["git_commit_hash"],
                "git_repo_url": simulator["git_repo_url"],
                "git_branch": simulator["git_branch"],
            },
            "parca_config": {"cpus": cpus},
        }

        response = await client.client.post("/core/v1/simulation/parca", json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, default=str)


@mcp.tool()
async def get_parca_status(parca_id: int) -> str:
    """Get the status of a parca calculation by its database ID.

    Args:
        parca_id: The database ID of the parca dataset.

    Returns:
        JSON with the HPC run status.
    """
    async with SMSAPIClient(_context.api_url) as client:
        response = await client.client.get("/core/v1/simulation/parca/status", params={"parca_id": parca_id})
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, default=str)


@mcp.tool()
async def list_parca_datasets() -> str:
    """List all parca datasets.

    Returns:
        JSON array of all parca datasets.
    """
    async with SMSAPIClient(_context.api_url) as client:
        response = await client.client.get("/core/v1/simulation/parca/versions")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, default=str)


def _build_parameters_schema(func: Any) -> dict[str, Any]:
    """Build a JSON Schema parameters object from a function signature.

    Args:
        func: The function to introspect.

    Returns:
        JSON Schema object describing the function's parameters.
    """
    import inspect

    sig = inspect.signature(func)
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "ctx"):
            continue

        param_type = "string"
        description = ""

        if param.annotation is not inspect.Parameter.empty:
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"
            elif hasattr(param.annotation, "__origin__"):
                # Handle Optional types
                origin = param.annotation.__origin__
                if origin is int:
                    param_type = "integer"

        parameters["properties"][param_name] = {
            "type": param_type,
            "description": description,
        }

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    return parameters


def get_tools_for_ollama() -> list[dict[str, Any]]:
    """Get tool definitions formatted for Ollama's tool calling API.

    Returns:
        List of tool definitions in Ollama format.
    """
    tools = []
    for tool in mcp._tool_manager._tools.values():
        # Regular tools have an fn attribute with the function
        if hasattr(tool, "fn"):
            parameters = _build_parameters_schema(tool.fn)
        elif hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
            parameters = tool.parameters
        else:
            parameters = {"type": "object", "properties": {}, "required": []}

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        })

    return tools


def get_tools_for_huggingface() -> list[dict[str, Any]]:
    """Get tool definitions formatted for HuggingFace transformers (FunctionGemma).

    This format is compatible with tokenizer.apply_chat_template(tools=...).

    Returns:
        List of tool definitions in HuggingFace format.
    """
    tools = []
    for tool in mcp._tool_manager._tools.values():
        # Regular tools have an fn attribute with the function
        if hasattr(tool, "fn"):
            parameters = _build_parameters_schema(tool.fn)
        elif hasattr(tool, "parameters") and isinstance(tool.parameters, dict):
            parameters = tool.parameters
        else:
            parameters = {"type": "object", "properties": {}, "required": []}

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        })

    return tools


async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Dictionary of arguments to pass to the tool.

    Returns:
        The tool's result as a string.
    """
    tool = mcp._tool_manager._tools.get(tool_name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        # Regular tools have an fn attribute with the async function
        if hasattr(tool, "fn"):
            result: str = await tool.fn(**arguments)
            return result
        else:
            return json.dumps({"error": f"Tool {tool_name} has no callable method"})
    except Exception as e:
        logger.exception(f"Error executing tool {tool_name}")
        return json.dumps({"error": str(e)})
