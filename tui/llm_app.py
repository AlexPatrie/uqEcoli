"""LLM-powered TUI for SMS API using a lightweight Hugging Face model."""

import asyncio
import json
import logging
import re
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, Markdown, Static

from sms_api.tui.client import SMSAPIClient

logger = logging.getLogger(__name__)

# Default API URL
# DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_API_URL = "https://sms.cam.uchc.edu"

# System prompt for the LLM
SYSTEM_PROMPT = """You are an extremely helpful multiscale biological simulation assistant \
for the SMS (Simulating Microbial Systems) API.

You help users run and manage E. coli whole-cell simulations using the vEcoli model. You can:
1. List all simulations - show existing simulations in the system
2. Run a new simulation - start a new E. coli simulation with specified parameters
3. Check simulation status - get the current status of a simulation by ID
4. Get simulation data - retrieve output data from a completed simulation
5. Get latest simulator - find the latest vEcoli simulator version
6. Upload a simulator - register a new simulator version
7. Check simulator build status - monitor simulator build progress

When the user asks to do something, identify which action they want and respond helpfully.
Always be concise but informative about biological simulations.
"""

# Intent mapping for API actions
INTENTS = {
    "list_simulations": {
        "keywords": ["list", "show", "all simulations", "what simulations", "simulations"],
        "description": "List all simulations",
        "method": "list_simulations",
        "params": [],
    },
    "run_simulation": {
        "keywords": ["run", "start", "new simulation", "launch", "execute"],
        "description": "Run a new simulation",
        "method": "run_simulation",
        "params": ["experiment_id", "generations", "num_seeds", "simulator_id"],
    },
    "check_status": {
        "keywords": ["status", "check", "how is", "progress", "state"],
        "description": "Check simulation status",
        "method": "get_simulation_status",
        "params": ["simulation_id"],
    },
    "get_data": {
        "keywords": ["data", "output", "results", "get data", "download"],
        "description": "Get simulation data",
        "method": "get_simulation_data",
        "params": ["simulation_id"],
    },
    "get_latest_simulator": {
        "keywords": ["latest simulator", "newest", "current version", "simulator version"],
        "description": "Get latest simulator version",
        "method": "get_latest_simulator",
        "params": [],
    },
    "upload_latest_simulator": {
        "keywords": ["upload latest", "upload the latest", "register latest"],
        "description": "Upload the latest simulator version",
        "method": "upload_latest_simulator",
        "params": ["git_branch", "git_repo_url"],
    },
    "upload_simulator": {
        "keywords": ["upload", "register", "new simulator", "add simulator"],
        "description": "Upload a new simulator",
        "method": "upload_simulator",
        "params": ["git_commit_hash"],
    },
    "simulator_status": {
        "keywords": ["build status", "simulator status", "build progress"],
        "description": "Check simulator build status",
        "method": "get_simulator_status",
        "params": ["simulator_id"],
    },
}


def _extract_git_params(user_input: str) -> dict[str, Any]:
    """Extract git branch and repo parameters from natural language input.

    Handles patterns like:
    - "from the api-support branch" or "branch api-support"
    - "in the vivarium-collective/vEcoli repo" or "repo vivarium-collective/vEcoli"

    Args:
        user_input: The user's natural language query.

    Returns:
        Dictionary with git_branch and/or git_repo_url if found.
    """
    params: dict[str, Any] = {}

    # Extract branch: "from the X branch", "branch X", "on X branch", "the X branch"
    branch_patterns = [
        r"branch\s+([a-zA-Z0-9_.-]+)",  # "branch api-support" - must come first
        r"(?:from\s+)?(?:the\s+)?([a-zA-Z0-9_.-]+)\s+branch",  # "from the api-support branch"
        r"on\s+([a-zA-Z0-9_.-]+)\s+branch",  # "on api-support branch"
    ]
    for pattern in branch_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            branch = match.group(1).strip("'\"")
            # Skip common words that aren't branch names
            if branch.lower() not in ["the", "a", "from", "in", "on", "to"]:
                params["git_branch"] = branch
                break

    # Extract repo: "in the X repo", "repo X", "X/Y repo", "from X/Y"
    # Also handle full GitHub URLs
    repo_patterns = [
        r"(?:in\s+)?(?:the\s+)?(\S+/\S+)\s+repo",  # "in the vivarium-collective/vEcoli repo"
        r"repo\s+(\S+/\S+)",  # "repo vivarium-collective/vEcoli"
        r"(https?://github\.com/\S+)",  # full GitHub URL
        r"from\s+(\S+/\S+)(?:\s|$)",  # "from vivarium-collective/vEcoli"
    ]
    for pattern in repo_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            repo_value = match.group(1).strip("'\"")
            # Convert org/repo to full GitHub URL if not already a URL
            if not repo_value.startswith("http"):
                repo_value = f"https://github.com/{repo_value}"
            params["git_repo_url"] = repo_value
            break

    return params


def detect_intent(user_input: str) -> tuple[str | None, dict[str, Any]]:
    """Detect user intent from natural language input.

    Args:
        user_input: The user's natural language query.

    Returns:
        Tuple of (intent_name, extracted_params) or (None, {}) if no intent detected.
    """
    user_lower = user_input.lower()

    # Check for compound intents first (actions that combine multiple operations)
    # "upload" + "latest" should trigger upload_latest_simulator even if not adjacent
    if "upload" in user_lower and ("latest" in user_lower or "newest" in user_lower):
        params = _extract_git_params(user_input)
        return "upload_latest_simulator", params

    # Build a list of (keyword, intent_name, intent_info) tuples, sorted by keyword length descending
    # This ensures more specific multi-word keywords match before generic single-word ones
    keyword_matches: list[tuple[str, str, dict[str, Any]]] = []
    for intent_name, intent_info in INTENTS.items():
        for keyword in intent_info["keywords"]:
            keyword_matches.append((keyword, intent_name, intent_info))

    # Sort by keyword length descending so "build status" matches before "status"
    keyword_matches.sort(key=lambda x: len(x[0]), reverse=True)

    # Check for each keyword in order of specificity
    for keyword, intent_name, intent_info in keyword_matches:
        if keyword in user_lower:
            params: dict[str, Any] = {}

            # Look for experiment_id pattern
            if "experiment_id" in intent_info["params"]:
                # Default experiment ID
                params["experiment_id"] = "my_experiment"
                # Match patterns like "called X", "named X"
                name_match = re.search(r"(?:called|named)\s+([a-zA-Z0-9_-]+)", user_input, re.IGNORECASE)
                if name_match:
                    params["experiment_id"] = name_match.group(1).strip("'\"")

            # Extract generations pattern: "5 generation", "5 generations"
            if "generations" in intent_info["params"]:
                gen_match = re.search(r"(\d+)\s*generation", user_lower)
                if gen_match:
                    params["generations"] = int(gen_match.group(1))

            # Extract seeds pattern: "3 seed", "3 seeds"
            if "num_seeds" in intent_info["params"]:
                seed_match = re.search(r"(\d+)\s*seed", user_lower)
                if seed_match:
                    params["num_seeds"] = int(seed_match.group(1))

            # Extract simulation_id pattern: "simulation 5", "simulation id 5", "of simulation 5"
            if "simulation_id" in intent_info["params"]:
                sim_id_match = re.search(r"simulation\s*(?:id\s*)?(\d+)", user_lower)
                if sim_id_match:
                    params["simulation_id"] = int(sim_id_match.group(1))

            # Extract simulator_id pattern: "simulator 5", "simulator id 5", "with simulator 5"
            if "simulator_id" in intent_info["params"]:
                sim_id_match = re.search(r"simulator\s*(?:id\s*)?(\d+)", user_lower)
                if sim_id_match:
                    params["simulator_id"] = int(sim_id_match.group(1))

            return intent_name, params

    return None, {}


class LLMSimulationAssistant:
    """LLM-powered simulation assistant using Hugging Face transformers."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> None:
        """Initialize the LLM assistant.

        Args:
            model_name: Hugging Face model name to use.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the LLM model."""
        if self._loaded:
            return

        try:
            from transformers import pipeline

            logger.info(f"Loading model: {self.model_name}")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            self._loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._loaded = False

    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate a response using the LLM.

        Args:
            user_message: The user's message.
            context: Additional context (e.g., API results).

        Returns:
            The LLM's response.
        """
        if not self._loaded:
            # Fallback to simple response if model not loaded
            return self._fallback_response(user_message, context)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{user_message}\n\nContext: {context}" if context else user_message},
            ]

            outputs = self.pipeline(
                messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_message, context)

    def _fallback_response(self, user_message: str, context: str = "") -> str:
        """Generate a simple fallback response without the LLM."""
        if context:
            return f"Here are the results:\n\n{context}"
        return (
            "I understand you want to work with simulations. Please try one of these:\n"
            "- List all simulations\n"
            "- Check status of simulation [ID]\n"
            "- Get data for simulation [ID]\n"
            "- Run a new simulation\n"
            "- Get latest simulator"
        )


class LLMApp(App):
    """LLM-powered SMS API Terminal User Interface."""

    CSS = """
    Screen {
        background: $surface;
    }

    #chat-container {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary;
        overflow-y: auto;
    }

    #chat-log {
        height: auto;
        padding: 1;
    }

    .user-message {
        background: $primary-darken-2;
        padding: 1;
        margin: 1 0;
        border: round $primary;
    }

    .assistant-message {
        background: $surface-darken-2;
        padding: 1;
        margin: 1 0;
        border: round $accent;
    }

    .system-message {
        color: $text-muted;
        text-style: italic;
        padding: 1;
        text-align: center;
    }

    #input-container {
        height: auto;
        padding: 1;
        dock: bottom;
    }

    #user-input {
        width: 100%;
    }

    #status-bar {
        height: 1;
        background: $surface-darken-2;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    TITLE = "Atlantis AI"
    SUB_TITLE = "Multiscale Biological Simulation Assistant"

    def __init__(self, api_url: str, use_llm: bool = True) -> None:
        """Initialize the LLM TUI.

        Args:
            api_url: Base URL of the SMS API server.
            use_llm: Whether to use the LLM for responses.
        """
        super().__init__()
        self.api_url = api_url
        self.use_llm = use_llm
        self.assistant: LLMSimulationAssistant | None = None
        if use_llm:
            self.assistant = LLMSimulationAssistant()
        self.chat_history: list[dict[str, str]] = []
        self._reset_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header()

        with Vertical():
            yield Static(
                "Welcome to Atlantis AI - Your Multiscale Biological Simulation Assistant",
                id="status-bar",
            )

            with ScrollableContainer(id="chat-container"):
                yield Markdown("", id="chat-log")

            with Vertical(id="input-container"):
                yield Input(
                    placeholder="What would you like to do? (e.g., 'List simulations')",
                    id="user-input",
                )

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount - show welcome message."""
        welcome = """## Welcome to Atlantis AI

I'm your multiscale biological simulation assistant for the SMS API.

**What would you like to do?**

- **List simulations** - "Show me all simulations" or "List simulations"
- **Check status** - "Check status of simulation 1" or "How is simulation 1 doing?"
- **Get data** - "Get data for simulation 1" or "Download results from simulation 1"
- **Run simulation** - "Run a new simulation" or "Start a simulation called my_test"
- **Get latest simulator** - "Get the latest simulator version"
- **Upload latest simulator** - "Upload the latest simulator" (auto-fetches and uploads)
- **Upload simulator** - "Upload simulator with commit abc1234"
- **Simulator build status** - "Check build status for simulator 1"

Type your request below and press Enter.
"""
        await self._add_message("assistant", welcome)

        # Focus the input
        self.query_one("#user-input", Input).focus()

        # Load LLM in background if enabled
        if self.use_llm:
            self._load_llm_async()

    @work(exclusive=True, thread=True)
    def _load_llm_async(self) -> None:
        """Load the LLM model in background."""
        if self.assistant is None:
            return
        self.call_from_thread(self._update_status, "Loading AI model...")
        try:
            self.assistant.load_model()
            self.call_from_thread(self._update_status, f"Connected to {self.api_url}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.call_from_thread(self._update_status, "AI model not available - using simple mode")

    def _update_status(self, message: str) -> None:
        """Update the status bar."""
        status = self.query_one("#status-bar", Static)
        status.update(message)

    async def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat log.

        Args:
            role: 'user' or 'assistant'.
            content: The message content.
        """
        self.chat_history.append({"role": role, "content": content})

        # Rebuild the chat log
        chat_md = ""
        for msg in self.chat_history:
            if msg["role"] == "user":
                chat_md += f"\n\n**You:** {msg['content']}\n"
            else:
                chat_md += f"\n\n{msg['content']}\n"

        chat_log = self.query_one("#chat-log", Markdown)
        await chat_log.update(chat_md)

        # Scroll to bottom
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_end()

    @on(Input.Submitted, "#user-input")
    def on_user_input(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message:
            return

        # Clear input
        event.input.value = ""

        # Process the message
        self._process_message_async(user_message)

    @work(exclusive=True)
    async def _process_message_async(self, user_message: str) -> None:
        """Process user message and generate response."""
        # Add user message to chat
        await self._add_message("user", user_message)

        # Detect intent
        intent, params = detect_intent(user_message)

        if intent:
            # Execute the API call
            self._update_status(f"Executing: {INTENTS[intent]['description']}...")
            result = await self._execute_api_call(intent, params)

            # Always format and display the actual API result
            # The LLM should NOT override real API results
            response = self._format_result(intent, result)

            await self._add_message("assistant", response)
        else:
            # No intent detected - ask for clarification
            response = """I'm not sure what you'd like to do. Try one of these:

- **List simulations**: "Show me all simulations"
- **Check status**: "Check status of simulation 1"
- **Get data**: "Get data for simulation 1"
- **Run simulation**: "Run a new simulation"
- **Get latest simulator**: "Get the latest simulator"
"""
            await self._add_message("assistant", response)

        self._update_status(f"Connected to {self.api_url}")

    async def _execute_api_call(self, intent: str, params: dict[str, Any]) -> str:
        """Execute the API call based on intent.

        Args:
            intent: The detected intent.
            params: Extracted parameters.

        Returns:
            Result as a string.
        """
        try:
            async with SMSAPIClient(self.api_url) as client:
                if intent == "list_simulations":
                    result = await client.list_simulations()
                    sims = [s.model_dump(mode="json") for s in result]
                    return json.dumps(sims, indent=2)

                elif intent == "run_simulation":
                    experiment_id = params.get("experiment_id", "my_experiment")
                    generations = params.get("generations")
                    num_seeds = params.get("num_seeds")
                    simulator_id = params.get("simulator_id")

                    # If no simulator_id provided, get the latest available one
                    if simulator_id is None:
                        simulators = await client.list_simulator_versions()
                        if not simulators.versions:
                            return (
                                "Error: No simulators available. Please upload a simulator first:\n"
                                "- Say 'Upload the latest simulator' to register a new one"
                            )
                        # Use the most recent simulator (highest database_id)
                        latest_sim = max(simulators.versions, key=lambda s: s.database_id or 0)
                        simulator_id = latest_sim.database_id
                        if simulator_id is None:
                            return "Error: Could not determine simulator ID"

                    result = await client.run_simulation(
                        simulator_id=simulator_id,
                        experiment_id=experiment_id,
                        num_generations=generations,
                        num_seeds=num_seeds,
                    )
                    return f"Simulation created with ID: {result.database_id}, Job ID: {result.job_id}"

                elif intent == "check_status":
                    sim_id = params.get("simulation_id")
                    if not sim_id:
                        return "Error: Please specify a simulation ID (e.g., 'Check status of simulation 1')"
                    result = await client.get_simulation_status(sim_id)
                    return f"Simulation {result.id} status: {result.status.value}"

                elif intent == "get_data":
                    sim_id = params.get("simulation_id")
                    if not sim_id:
                        return "Error: Please specify a simulation ID"
                    result = await client.get_simulation_data(sim_id)
                    return f"Found {len(result)} data files"

                elif intent == "get_latest_simulator":
                    result = await client.get_latest_simulator()
                    return (
                        f"Latest simulator:\n"
                        f"- Commit: {result.git_commit_hash}\n"
                        f"- Branch: {result.git_branch}\n"
                        f"- Repo: {result.git_repo_url}"
                    )

                elif intent == "upload_latest_simulator":
                    # Use extracted params or defaults
                    git_repo_url = params.get("git_repo_url", "https://github.com/vivarium-collective/vEcoli")
                    git_branch = params.get("git_branch", "ccam-nextflow")
                    # First get the latest simulator info for the specified branch/repo
                    latest = await client.get_latest_simulator(
                        git_repo_url=git_repo_url,
                        git_branch=git_branch,
                    )
                    # Then upload it using the fetched info
                    result = await client.upload_simulator(
                        git_commit_hash=latest.git_commit_hash,
                        git_repo_url=latest.git_repo_url,
                        git_branch=latest.git_branch,
                    )
                    return (
                        f"Uploaded latest simulator:\n"
                        f"- Database ID: {result.database_id}\n"
                        f"- Commit: {latest.git_commit_hash}\n"
                        f"- Branch: {latest.git_branch}\n"
                        f"- Repo: {latest.git_repo_url}"
                    )

                elif intent == "upload_simulator":
                    commit_hash = params.get("git_commit_hash", "")
                    if not commit_hash:
                        return "Error: Please specify a commit hash"
                    result = await client.upload_simulator(git_commit_hash=commit_hash)
                    return f"Simulator uploaded with ID: {result.database_id}"

                elif intent == "simulator_status":
                    sim_id = params.get("simulator_id")
                    if not sim_id:
                        return "Error: Please specify a simulator ID"
                    result = await client.get_simulator_status(sim_id)
                    return f"Simulator {result.database_id} build status: {result.status.value}"

        except Exception as e:
            return f"Error: {e}"

        return "Unknown action"

    def _format_result(self, intent: str, result: str) -> str:
        """Format the API result for display with conversational text.

        Args:
            intent: The detected intent.
            result: The raw result string.

        Returns:
            Formatted markdown response.
        """
        # Provide conversational responses for each intent
        responses = {
            "list_simulations": f"Here are the current simulations:\n\n```json\n{result}\n```",
            "run_simulation": f"I've started a new simulation for you.\n\n**Result:** {result}",
            "check_status": f"Here's the status of your simulation:\n\n**{result}**",
            "get_data": f"Here's the simulation data:\n\n{result}",
            "get_latest_simulator": f"Here's the latest simulator version:\n\n{result}",
            "upload_latest_simulator": f"I've fetched and uploaded the latest simulator.\n\n{result}",
            "upload_simulator": f"I've uploaded the simulator.\n\n**Result:** {result}",
            "simulator_status": f"Here's the simulator build status:\n\n**{result}**",
        }

        return responses.get(intent, f"**Result:**\n\n{result}")

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        self.chat_history.clear()
        self._reset_task = asyncio.create_task(self._reset_chat())

    async def _reset_chat(self) -> None:
        """Reset the chat to initial state."""
        chat_log = self.query_one("#chat-log", Markdown)
        await chat_log.update("")
        await self.on_mount()


def main(api_url: str = DEFAULT_API_URL, use_llm: bool = True) -> None:
    """Run the LLM-powered TUI application.

    Args:
        api_url: Base URL of the SMS API server.
        use_llm: Whether to use the LLM for responses.
    """
    app = LLMApp(api_url=api_url, use_llm=use_llm)
    app.run()


if __name__ == "__main__":
    main()
