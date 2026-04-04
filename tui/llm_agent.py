"""Ollama-based LLM agent with MCP tool support for the Atlantis TUI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import ollama
from ollama import AsyncClient
from sms_api.tui.client import BASE_URL
from sms_api.tui.mcp_tools import execute_tool, get_tools_for_ollama, set_api_url

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A chat message."""

    role: str  # "user", "assistant", or "tool"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # For tool messages


@dataclass
class ToolCallResult:
    """Result of a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result: str


def extract_function_call(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Extract function call from LLM text output.

    Handles multiple formats:
    1. JSON format: {"name": "...", "parameters": {...}}
    2. Simple JSON: {"tool": "...", "args": {...}}
    3. Markdown code blocks with JSON

    Args:
        text: The raw LLM output text.

    Returns:
        Tuple of (tool_name, args) or (None, None) if not found.
    """
    # Try to find JSON in markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
            # Handle {"name": "...", "parameters": {...}} format
            if "name" in data:
                return data["name"], data.get("parameters", {})
            # Handle {"tool": "...", "args": {...}} format
            if "tool" in data:
                return data["tool"], data.get("args", {})
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object with name/parameters or tool/args
    json_patterns = [
        r'\{\s*"name"\s*:\s*"([^"]+)"[^}]*"parameters"\s*:\s*(\{[^}]*\})',
        r'\{\s*"name"\s*:\s*"([^"]+)"[^}]*\}',
        r'\{\s*"tool"\s*:\s*"([^"]+)"[^}]*"args"\s*:\s*(\{[^}]*\})',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2)) if len(match.groups()) > 1 else {}
            except (json.JSONDecodeError, IndexError):
                args = {}
            return tool_name, args

    # Try to match tool name mentioned with list_* or get_* or run_* patterns
    tool_mention = re.search(
        r"(?:call|use|execute|calling|using)\s+(?:the\s+)?[`*]*(\w+_\w+)[`*]*",
        text,
        re.IGNORECASE,
    )
    if tool_mention:
        return tool_mention.group(1), {}

    return None, None


class AtlantisAgent:
    """Ollama-based agent with tool calling support for SMS API operations."""

    def __init__(
        self,
        model: str = "llama3.1",
        api_url: str = BASE_URL,
        ollama_host: str | None = None,
    ) -> None:
        """Initialize the Atlantis agent.

        Args:
            model: The Ollama model to use (must support tool calling).
            api_url: The SMS API URL.
            ollama_host: Optional Ollama host URL (default: http://localhost:11434).
        """
        self.model = model
        self.api_url = api_url
        self.ollama_host = ollama_host
        self.messages: list[dict[str, Any]] = []
        self.tools = get_tools_for_ollama()
        self._client: AsyncClient | None = None

        # Set the API URL for tools
        set_api_url(api_url)

        # System message for tool calling
        self.system_message = """You are Atlantis, an AI assistant for the SMS (Simulating Microbial Systems) API.
You help users manage vEcoli whole-cell biological simulations.

When users ask to perform operations, you MUST call the appropriate tool. \
Do not just describe what you would do - actually call the tool.

Available tools:
- list_ecoli_simulations: List all simulations
- run_ecoli_simulation: Run a new simulation (requires experiment_id, simulator_id)
- get_ecoli_simulation: Get simulation details by ID
- get_ecoli_simulation_status: Check simulation status
- get_ecoli_simulation_data: Get output data from a completed simulation
- get_latest_simulator: Get latest simulator version from GitHub
- upload_simulator: Upload/register a new simulator version
- get_simulator_status: Check simulator build status
- list_simulator_versions: List all registered simulators
- run_parca: Run parameter calculations for a simulator
- get_parca_status: Check parca calculation status
- list_parca_datasets: List all parca datasets

When you call a tool, I will execute it and give you the result. Then provide a helpful summary."""

        # System message for response generation
        self.response_system = """You are Atlantis, a helpful assistant for biological simulations.
Given an API result, provide a clear, user-friendly summary. Include:
1. The key information (IDs, status, counts)
2. What the result means in plain language
3. Suggested next steps if relevant

Be concise but informative. Use markdown formatting."""

    @property
    def client(self) -> AsyncClient:
        """Get the Ollama async client."""
        if self._client is None:
            self._client = AsyncClient(host=self.ollama_host)
        return self._client

    def set_api_url(self, url: str) -> None:
        """Update the API URL for tools."""
        self.api_url = url
        set_api_url(url)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.messages = []

    async def chat(
        self,
        message: str,
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> AsyncIterator[str]:
        """Send a message and get a streaming response.

        Args:
            message: The user's message.
            on_tool_call: Optional callback when a tool is called.
            on_tool_result: Optional callback when a tool returns.

        Yields:
            Text chunks from the assistant's response.
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": message})

        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_message}] + self.messages

        # Call Ollama with tools
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                tools=self.tools,
                stream=False,
            )
        except ollama.ResponseError as e:
            error_msg = f"Ollama error: {e}"
            logger.exception(error_msg)
            yield error_msg
            return
        except Exception as e:
            error_msg = f"Failed to connect to Ollama: {e}"
            logger.exception(error_msg)
            yield error_msg
            return

        # Handle the response
        assistant_message = response.get("message", {})
        content = assistant_message.get("content", "")
        tool_calls = assistant_message.get("tool_calls", [])

        # If native tool calls are present, execute them
        if tool_calls:
            async for chunk in self._execute_native_tool_calls(content, tool_calls, on_tool_call, on_tool_result):
                yield chunk
        elif content:
            # Try to extract function call from text output
            tool_name, args = extract_function_call(content)

            if tool_name:
                # Found a function call in the text - execute it
                async for chunk in self._execute_extracted_tool_call(
                    content, tool_name, args or {}, on_tool_call, on_tool_result
                ):
                    yield chunk
            else:
                # No tool call found, just return the content
                self.messages.append({"role": "assistant", "content": content})
                yield content
        else:
            yield "I didn't get a response. Please try again."

    async def _execute_native_tool_calls(
        self,
        initial_content: str,
        tool_calls: list[dict[str, Any]],
        on_tool_call: Any | None,
        on_tool_result: Any | None,
    ) -> AsyncIterator[str]:
        """Execute native Ollama tool calls and generate response.

        Args:
            initial_content: Initial assistant content before tool calls.
            tool_calls: List of tool calls from Ollama.
            on_tool_call: Optional callback when a tool is called.
            on_tool_result: Optional callback when a tool returns.

        Yields:
            Response chunks.
        """
        # Add assistant message with tool calls to history
        self.messages.append({
            "role": "assistant",
            "content": initial_content,
            "tool_calls": tool_calls,
        })

        all_results = []

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", {})

            if on_tool_call:
                on_tool_call(tool_name, arguments)

            yield f"**Calling:** `{tool_name}`"
            if arguments:
                yield f" with `{json.dumps(arguments)}`"
            yield "\n"

            # Execute the tool
            try:
                result = await execute_tool(tool_name, arguments)
            except Exception as e:
                result = json.dumps({"error": str(e)})
                logger.exception(f"Tool execution error: {tool_name}")

            if on_tool_result:
                on_tool_result(tool_name, result)

            # Add tool result to history
            self.messages.append({"role": "tool", "content": result})
            all_results.append({"tool": tool_name, "result": result})

        # Generate natural language response
        async for chunk in self._generate_response(all_results):
            yield chunk

    async def _execute_extracted_tool_call(
        self,
        initial_content: str,
        tool_name: str,
        arguments: dict[str, Any],
        on_tool_call: Any | None,
        on_tool_result: Any | None,
    ) -> AsyncIterator[str]:
        """Execute a tool call extracted from text and generate response.

        Args:
            initial_content: Initial assistant content.
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.
            on_tool_call: Optional callback when a tool is called.
            on_tool_result: Optional callback when a tool returns.

        Yields:
            Response chunks.
        """
        if on_tool_call:
            on_tool_call(tool_name, arguments)

        yield f"**Calling:** `{tool_name}`"
        if arguments:
            yield f" with `{json.dumps(arguments)}`"
        yield "\n"

        # Execute the tool
        try:
            result = await execute_tool(tool_name, arguments)
        except Exception as e:
            result = json.dumps({"error": str(e)})
            logger.exception(f"Tool execution error: {tool_name}")

        if on_tool_result:
            on_tool_result(tool_name, result)

        # Add to history
        self.messages.append({"role": "assistant", "content": initial_content})
        self.messages.append({"role": "tool", "content": result})

        # Generate natural language response
        async for chunk in self._generate_response([{"tool": tool_name, "result": result}]):
            yield chunk

    async def _generate_response(
        self,
        tool_results: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Generate a natural language response summarizing tool results.

        Args:
            tool_results: List of tool execution results.

        Yields:
            Response chunks.
        """
        # Format results for the prompt
        results_text = ""
        for tr in tool_results:
            results_text += f"\nTool: {tr['tool']}\nResult:\n{tr['result']}\n"

        # Create a focused prompt for response generation
        response_prompt = f"""Based on these API results, provide a helpful summary for the user:
{results_text}

Summarize the key information clearly. If there are IDs, status values, or counts, highlight them.
If appropriate, suggest what the user might want to do next."""

        messages = [
            {"role": "system", "content": self.response_system},
            {"role": "user", "content": response_prompt},
        ]

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
            )
            response_content = response.get("message", {}).get("content", "")

            if response_content:
                yield "\n\n"
                yield response_content
                # Add to history
                self.messages.append({"role": "assistant", "content": response_content})
            else:
                # Fallback: show raw results
                yield "\n\n**Results:**\n"
                for tr in tool_results:
                    yield f"```json\n{tr['result']}\n```\n"

        except Exception:
            logger.exception("Error generating response")
            yield "\n\n**Results:**\n"
            for tr in tool_results:
                yield f"```json\n{tr['result']}\n```\n"

    async def poll_status(
        self,
        check_func: str,
        resource_id: int,
        interval_seconds: float = 5.0,
        max_iterations: int = 100,
    ) -> AsyncIterator[str]:
        """Poll a status endpoint until completion.

        Args:
            check_func: The tool name to call (e.g., "get_simulator_status").
            resource_id: The ID to pass to the status function.
            interval_seconds: Seconds between checks.
            max_iterations: Maximum number of checks.

        Yields:
            Status updates as text.
        """
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}

        for i in range(max_iterations):
            yield f"\n**Check {i + 1}:** "

            try:
                result = await execute_tool(check_func, {"simulator_id": resource_id})
                result_data = json.loads(result)

                # Extract status
                status = result_data.get("status", {})
                if isinstance(status, dict):
                    status_value = status.get("value", "UNKNOWN")
                else:
                    status_value = str(status)

                yield f"Status = **{status_value}**\n"

                if status_value.upper() in terminal_states:
                    yield f"\n*Polling complete. Final status: {status_value}*"
                    return

                yield f"*Waiting {interval_seconds}s...*\n"
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                yield f"Error checking status: {e}\n"
                await asyncio.sleep(interval_seconds)

        yield "\n*Max iterations reached. Stopping poll.*"
