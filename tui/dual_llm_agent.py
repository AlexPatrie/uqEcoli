"""Two-LLM architecture agent for the Atlantis TUI.

This module implements a dual-LLM system:
1. ToolLLM (FunctionGemma): Detects user intent and generates function calls
2. ResponseLLM (Qwen2.5-7B): Generates natural language responses with scientific reasoning
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator, Generator
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import asyncio

from sms_api.tui.client import BASE_URL
from sms_api.tui.mcp_tools import execute_tool, get_tools_for_huggingface, set_api_url

logger = logging.getLogger(__name__)


# System prompt for the Tool LLM (function calling)
ROUTER_PROMPT = """You are a tool-calling assistant for the SMS (Simulating Microbial Systems) API.
When users ask to perform operations on biological simulations, select the appropriate tool.

Available operations:
- get_latest_simulator: Get latest vEcoli simulator version from a git repository
- upload_simulator: Upload/register a new simulator version to build its container
- get_simulator_status: Check simulator container build status
- list_simulator_versions: List all registered simulator versions
- run_simulation: Run a new E. coli whole-cell simulation
- get_simulation: Get simulation details by ID
- get_simulation_status: Check simulation run status
- get_simulation_data: Get output data from a completed simulation
- list_simulations: List all simulations
- run_parca: Run parameter calculations (parca) for a simulator
- get_parca_status: Check parca calculation status
- list_parca_datasets: List all parca datasets

When the user asks for an operation, call the appropriate tool with the correct parameters.
Extract parameters from the user's request (e.g., branch names, commit hashes, IDs, generation counts).
"""

# System prompt for the Response LLM (scientific explanation)
SCIENCE_RESPONSE_PROMPT = """You are Atlantis, a multiscale biological simulation assistant \
for E. coli whole-cell modeling using the vEcoli simulator.

Your role is to:
1. Explain API results in user-friendly language
2. Provide biological context when relevant:
   - "Parca" (parameter calculator) generates simulation parameters from BioCyc data
   - "Generations" refer to cell division cycles in the simulation
   - "Seeds" are independent simulation lineages with different random states
3. Highlight important values (database IDs, job IDs, status values)
4. Suggest logical next steps when appropriate

Be concise but scientifically accurate. Format output clearly with markdown.
When showing status, use clear indicators (e.g., RUNNING, COMPLETED, FAILED).
"""


class ToolLLM:
    """LLM specialized for function/tool calling using FunctionGemma."""

    MODEL_ID = "google/functiongemma-270m-it"

    def __init__(self, model_id: str | None = None, init_eager: bool = True) -> None:
        """Initialize the Tool LLM.

        Args:
            model_id: Optional model ID override.
            init_eager: If True, load the model immediately.
        """
        self.model_id = model_id or self.MODEL_ID
        self.model: Any = None
        self.tokenizer: Any = None
        self.initialized = False
        self._max_new_tokens = 256

        if init_eager:
            self.initialize()

    def initialize(self) -> None:
        """Load the model and tokenizer."""
        from sms_api.tui import model_factory

        logger.info(f"Initializing Tool LLM: {self.model_id}")
        self.model = model_factory.load_tool_llm(self.model_id)
        self.tokenizer = model_factory.load_tokenizer(self.model_id)
        self._max_new_tokens = model_factory.MAX_NEW_TOKENS
        self.initialized = True
        logger.info("Tool LLM initialized successfully")

    def prompt(
        self,
        user_prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        """Generate a response, yielding tokens as they are generated.

        Args:
            user_prompt: The user's message.
            system: Optional system prompt.
            tools: Optional list of tools in HuggingFace format.

        Yields:
            Generated tokens one at a time.
        """
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("ToolLLM not initialized. Call initialize() first.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})

        # Apply chat template with tools if supported
        try:
            inputs: dict[str, Any] = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            # Fallback if tokenizer doesn't support tools parameter
            logger.warning("Tokenizer doesn't support tools parameter, falling back")
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        yield from response


class ResponseLLM:
    """LLM for scientific reasoning and natural language responses using Qwen2.5."""

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model_id: str | None = None, init_eager: bool = True) -> None:
        """Initialize the Response LLM.

        Args:
            model_id: Optional model ID override.
            init_eager: If True, load the model immediately.
        """
        self.model_id = model_id or self.MODEL_ID
        self.model: Any = None
        self.tokenizer: Any = None
        self.initialized = False

        if init_eager:
            self.initialize()

    def initialize(self) -> None:
        """Load the model and tokenizer."""
        from sms_api.tui import model_factory

        logger.info(f"Initializing Response LLM: {self.model_id}")
        self.model = model_factory.load_response_llm(self.model_id)
        self.tokenizer = model_factory.load_tokenizer(self.model_id)
        self.initialized = True
        logger.info("Response LLM initialized successfully")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate a natural language response.

        Args:
            prompt: The prompt to respond to.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated response string.
        """
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("ResponseLLM not initialized. Call initialize() first.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        inputs: dict[str, Any] = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def extract_function_call(text: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Extract function call from LLM output.

    Handles multiple formats:
    1. JSON format: {"tool": "...", "args": {...}}
    2. Gemma native format with parens: <start_function_call>call:func(x=4, y=4)<end_function_call>
    3. Gemma native format with braces: <start_function_call>call:func{x:4, y:4}<end_function_call>

    Args:
        text: The raw LLM output text.

    Returns:
        Tuple of (tool_name, args, fallback_text).
        If a function call is found, fallback_text is None.
        If no function call is found, tool and args are None.
    """
    # Try JSON format first: {"tool": "square", "args": {"x": 4, "y": 4}}
    json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            tool = data.get("tool")
            args = data.get("args", {})
            if tool and tool != "NONE":
                return tool, args, None
        except json.JSONDecodeError:
            pass

    # Try Gemma native format with parens: <start_function_call>call:func_name(x=4, y=4)<end_function_call>
    gemma_paren_match = re.search(
        r"<start_function_call>call:(\w+)\((.*?)\)<end_function_call>",
        text,
        re.DOTALL,
    )
    if gemma_paren_match:
        tool = gemma_paren_match.group(1)
        args_str = gemma_paren_match.group(2).strip()
        args = _parse_key_value_pairs(args_str, separator="=")
        return tool, args, None

    # Try Gemma native format with braces: <start_function_call>call:func_name{x:4, y:4}<end_function_call>
    gemma_brace_match = re.search(
        r"<start_function_call>call:(\w+)\{(.*?)}<end_function_call>",
        text,
        re.DOTALL,
    )
    if gemma_brace_match:
        tool = gemma_brace_match.group(1)
        args_str = gemma_brace_match.group(2).strip()
        args = _parse_key_value_pairs(args_str, separator=":")
        return tool, args, None

    # No function call found, return the text as fallback
    return None, None, text


def _parse_key_value_pairs(args_str: str, separator: str = ":") -> dict[str, Any]:
    """Parse key-value pairs from a string like 'x:4, y:4' or 'x=4, y=4'.

    Args:
        args_str: String containing key-value pairs.
        separator: The separator between key and value.

    Returns:
        Dictionary of parsed arguments with appropriate types.
    """
    args: dict[str, Any] = {}
    if not args_str:
        return args

    for pair in args_str.split(","):
        if separator in pair:
            key, value = pair.split(separator, 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            # Try to parse value as JSON, then as number, then as string
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                try:
                    args[key] = float(value) if "." in value else int(value)
                except ValueError:
                    args[key] = value

    return args


class DualLLMAgent:
    """Two-LLM agent that separates tool calling from response generation.

    Architecture:
    1. User input -> ToolLLM (function call detection)
    2. Function call extraction
    3. Tool execution via MCP
    4. API result + user input -> ResponseLLM (scientific explanation)
    5. Final response to user
    """

    def __init__(
        self,
        api_url: str = BASE_URL,
        init_eager: bool = False,
        tool_model_id: str | None = None,
        response_model_id: str | None = None,
    ) -> None:
        """Initialize the dual-LLM agent.

        Args:
            api_url: The SMS API URL.
            init_eager: If True, load models immediately (slow but ready).
            tool_model_id: Optional override for the tool LLM model.
            response_model_id: Optional override for the response LLM model.
        """
        self.api_url = api_url
        self.tool_llm = ToolLLM(model_id=tool_model_id, init_eager=init_eager)
        self.response_llm = ResponseLLM(model_id=response_model_id, init_eager=init_eager)
        self._initialized = init_eager

        # Set the API URL for MCP tools
        set_api_url(api_url)

    def initialize(self) -> None:
        """Initialize both LLMs (lazy initialization)."""
        if not self._initialized:
            logger.info("Initializing DualLLMAgent models...")
            self.tool_llm.initialize()
            self.response_llm.initialize()
            self._initialized = True
            logger.info("DualLLMAgent initialization complete")

    async def initialize_async(self) -> None:
        """Initialize both LLMs (runs in Textual worker)."""
        if not self._initialized:
            logger.info("Initializing DualLLMAgent models...")
            self.initialize()

    @property
    def initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized

    def set_api_url(self, url: str) -> None:
        """Update the API URL for tools.

        Args:
            url: The new API URL.
        """
        self.api_url = url
        set_api_url(url)

    async def chat(self, message: str) -> AsyncIterator[str]:
        """Process a user message through the two-LLM pipeline.

        Args:
            message: The user's message.

        Yields:
            Response chunks for streaming display.
        """
        # Ensure models are initialized
        if not self._initialized:
            yield "Initializing models (this may take a moment)...\n\n"
            await self.initialize_async()

        # Phase 1: Tool selection using FunctionGemma
        yield "**Analyzing request...**\n"

        tools = get_tools_for_huggingface()

        # Run tool LLM directly (Textual's worker handles async properly)
        planning_response = ""
        for token in self.tool_llm.prompt(message, system=ROUTER_PROMPT, tools=tools):
            planning_response += token

        logger.debug(f"Tool LLM response: {planning_response}")

        # Phase 2: Extract function call
        tool_name, args, fallback_text = extract_function_call(planning_response)

        # Phase 3: Execute tool if detected
        if tool_name:
            yield f"**Calling tool:** `{tool_name}`\n"
            if args:
                yield f"**Arguments:** `{json.dumps(args)}`\n\n"

            try:
                result = await execute_tool(tool_name, args or {})
                yield f"**API Result:**\n```json\n{result}\n```\n\n"
            except Exception as e:
                result = json.dumps({"error": str(e)})
                logger.exception(f"Error executing tool {tool_name}")
                yield f"**Error:** {e}\n\n"
        else:
            result = fallback_text or "No specific action detected from your request."
            yield f"**Note:** {result}\n\n"

        # Phase 4: Generate scientific explanation using Qwen2.5
        yield "**Generating explanation...**\n\n"

        response_prompt = f"""User request: "{message}"

API Result:
{result}

Please explain this result clearly and suggest any relevant next steps."""

        try:
            final_response = self.response_llm.generate(
                response_prompt,
                system=SCIENCE_RESPONSE_PROMPT,
                max_tokens=512,
            )
            yield f"---\n\n{final_response}"
        except Exception as e:
            logger.exception("Error generating response")
            yield f"\n\n**Error generating explanation:** {e}"

    async def simple_chat(self, message: str) -> str:
        """Process a message and return the complete response.

        Args:
            message: The user's message.

        Returns:
            The complete response string.
        """
        response_parts = []
        async for chunk in self.chat(message):
            response_parts.append(chunk)
        return "".join(response_parts)
