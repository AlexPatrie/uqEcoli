"""Two-LLM powered TUI for SMS API using FunctionGemma + Qwen2.5."""

from __future__ import annotations

import asyncio
import logging
import os

# Disable tokenizer parallelism to avoid multiprocessing issues on macOS
# Must be set before importing transformers (via dual_llm_agent)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Disable macOS fork safety check that causes "bad value(s) in fds_to_ke" errors
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from sms_api.tui.client import BASE_URL
from sms_api.tui.dual_llm_agent import DualLLMAgent
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, Markdown, Static

logger = logging.getLogger(__name__)


class DualLLMApp(App[None]):
    """Two-LLM powered SMS API Terminal User Interface.

    Uses:
    - FunctionGemma (270M) for tool calling and intent detection
    - Qwen2.5-7B for scientific reasoning and response generation
    """

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

    TITLE = "Atlantis AI (Dual-LLM)"
    SUB_TITLE = "FunctionGemma + Qwen2.5 Scientific Assistant"

    def __init__(self, api_url: str = BASE_URL) -> None:
        """Initialize the Dual-LLM TUI.

        Args:
            api_url: Base URL of the SMS API server.
        """
        super().__init__()
        self.api_url = api_url
        self.agent = DualLLMAgent(api_url=api_url, init_eager=False)
        self.chat_history: list[dict[str, str]] = []
        self._reset_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header()

        with Vertical():
            yield Static(
                "Atlantis AI - Two-LLM Scientific Simulation Assistant",
                id="status-bar",
            )

            with ScrollableContainer(id="chat-container"):
                yield Markdown("", id="chat-log")

            with Vertical(id="input-container"):
                yield Input(
                    placeholder="Ask Atlantis anything... (e.g., 'Run a 5 generation simulation')",
                    id="user-input",
                )

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount - show welcome message."""
        welcome = """## Welcome to Atlantis AI (Dual-LLM Mode)

I'm your multiscale biological simulation assistant powered by two specialized LLMs:
- **FunctionGemma**: Detects your intent and selects the right API operations
- **Qwen2.5-7B**: Provides scientific explanations and biological context

**What would you like to do?**

- **Run simulations**: "Run a 5 generation, 3 seed simulation called my_test"
- **Check status**: "Check status of simulation 1"
- **Get data**: "Get data for simulation 1"
- **Manage simulators**: "Get the latest simulator from the api-support branch"
- **Upload simulators**: "Upload the latest simulator"
- **Run parca**: "Run parameter calculations for simulator 1"

Type your request below and press Enter. Models will be loaded on first use.
"""
        await self._add_message("assistant", welcome)

        # Focus the input
        self.query_one("#user-input", Input).focus()

    def _update_status(self, message: str) -> None:
        """Update the status bar."""
        try:
            status = self.query_one("#status-bar", Static)
            status.update(message)
        except Exception:
            logger.debug("Could not update status bar")

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

        try:
            chat_log = self.query_one("#chat-log", Markdown)
            await chat_log.update(chat_md)

            # Scroll to bottom
            container = self.query_one("#chat-container", ScrollableContainer)
            container.scroll_end()
        except Exception:
            logger.debug("Could not update chat log")

    async def _update_last_message(self, content: str) -> None:
        """Update the last message in the chat log (for streaming).

        Args:
            content: The updated message content.
        """
        if self.chat_history:
            self.chat_history[-1]["content"] = content

        # Rebuild the chat log
        chat_md = ""
        for msg in self.chat_history:
            if msg["role"] == "user":
                chat_md += f"\n\n**You:** {msg['content']}\n"
            else:
                chat_md += f"\n\n{msg['content']}\n"

        try:
            chat_log = self.query_one("#chat-log", Markdown)
            await chat_log.update(chat_md)

            # Scroll to bottom
            container = self.query_one("#chat-container", ScrollableContainer)
            container.scroll_end()
        except Exception:
            logger.debug("Could not update last message")

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
        """Process user message through the dual-LLM pipeline."""
        # Add user message to chat
        await self._add_message("user", user_message)

        # Add placeholder for assistant response
        await self._add_message("assistant", "*Processing...*")

        self._update_status("Processing request...")

        try:
            # Stream the response
            response_text = ""
            async for chunk in self.agent.chat(user_message):
                response_text += chunk
                await self._update_last_message(response_text)

            self._update_status(f"Connected to {self.api_url}")

        except Exception as e:
            logger.exception("Error processing message")
            await self._update_last_message(f"**Error:** {e}")
            self._update_status(f"Error: {e}")

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        self.chat_history.clear()
        self._reset_task = asyncio.create_task(self._reset_chat())

    async def _reset_chat(self) -> None:
        """Reset the chat to initial state."""
        try:
            chat_log = self.query_one("#chat-log", Markdown)
            await chat_log.update("")
            await self.on_mount()
        except Exception:
            logger.debug("Could not reset chat")


def main(api_url: str = BASE_URL) -> None:
    """Run the Dual-LLM powered TUI application.

    Args:
        api_url: Base URL of the SMS API server.
    """
    app = DualLLMApp(api_url=api_url)
    app.run()


if __name__ == "__main__":
    main()
