"""SMS API Terminal User Interface built with Textual."""

from __future__ import annotations

import logging

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Input, Markdown, Static

from sms_api.tui.client import BaseUrl
from sms_api.tui.llm_agent import AtlantisAgent

# Default to localhost for local development
DEFAULT_API_URL = BaseUrl.LOCAL

logger = logging.getLogger(__name__)


class AtlantisApp(App):
    """SMS API Terminal User Interface - Chat-focused layout."""

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

    #chat-input {
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

    TITLE = "Atlantis"
    SUB_TITLE = "SMS API Chat Interface"

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        ollama_model: str = "llama3.1",
    ) -> None:
        """Initialize the SMS TUI.

        Args:
            api_url: Base URL of the SMS API server.
            ollama_model: Ollama model to use for chat (must support tool calling).
        """
        super().__init__()
        self.api_url = api_url
        self.ollama_model = ollama_model
        self._agent: AtlantisAgent | None = None
        self._chat_history: str = ""

    @property
    def agent(self) -> AtlantisAgent:
        """Get or create the LLM agent."""
        if self._agent is None:
            self._agent = AtlantisAgent(
                model=self.ollama_model,
                api_url=self.api_url,
            )
        return self._agent

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header()

        with Vertical():
            yield Static(
                f"Connected to {self.api_url}",
                id="status-bar",
            )

            with ScrollableContainer(id="chat-container"):
                yield Markdown("", id="chat-log")

            with Vertical(id="input-container"):
                yield Input(
                    placeholder="Ask Atlantis anything... (e.g., 'List all simulations')",
                    id="chat-input",
                )

        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount - show welcome message."""
        welcome = """## Welcome to Atlantis

I'm your AI assistant for the SMS (Simulating Microbial Systems) API.

**What can I help you with?**

- **List simulations** - "Show me all simulations"
- **Run simulation** - "Run a new simulation called my_test with 2 generations"
- **Check status** - "Check status of simulation 1"
- **Get data** - "Get data for simulation 1"
- **Get latest simulator** - "Get the latest simulator version"
- **Upload simulator** - "Upload the latest simulator from the api-support branch"
- **Simulator build status** - "Check build status for simulator 1"

Type your request below and press Enter.
"""
        self._chat_history = welcome
        await self._update_chat_display()

        # Focus the input
        self.query_one("#chat-input", Input).focus()

    def _update_status(self, message: str) -> None:
        """Update the status bar."""
        status = self.query_one("#status-bar", Static)
        status.update(message)

    async def _update_chat_display(self) -> None:
        """Update the chat display with current history."""
        chat_log = self.query_one("#chat-log", Markdown)
        await chat_log.update(self._chat_history)

        # Scroll to bottom
        container = self.query_one("#chat-container", ScrollableContainer)
        container.scroll_end()

    @on(Input.Submitted, "#chat-input")
    def on_chat_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in chat input."""
        message = event.value.strip()
        if message:
            event.input.value = ""
            self._chat_async(message)

    async def action_clear_chat(self) -> None:
        """Clear the chat history and reset the agent."""
        self._chat_history = ""
        if self._agent is not None:
            self._agent.clear_history()
        await self._update_chat_display()
        await self.on_mount()

    @work(exclusive=True)
    async def _chat_async(self, message: str) -> None:
        """Process a chat message with the LLM agent.

        Args:
            message: The user's message to send to the agent.
        """
        self._update_status("Thinking...")

        # Add user message to chat history
        self._chat_history += f"\n\n**You:** {message}\n\n"
        await self._update_chat_display()

        try:
            # Collect the full response
            response_text = ""
            async for chunk in self.agent.chat(message):
                response_text += chunk
                # Update display with streaming content
                temp_history = self._chat_history + f"**Atlantis:** {response_text}"
                chat_log = self.query_one("#chat-log", Markdown)
                await chat_log.update(temp_history)

            # Finalize chat history
            self._chat_history += f"**Atlantis:** {response_text}\n\n---"
            await self._update_chat_display()

        except Exception as e:
            error_msg = f"**Error:** {e}"
            logger.exception("Chat error")
            self._chat_history += f"{error_msg}\n\n---"
            await self._update_chat_display()
        finally:
            self._update_status(f"Connected to {self.api_url}")


def main(api_url: str = DEFAULT_API_URL) -> None:
    """Run the SMS TUI application.

    Args:
        api_url: Base URL of the SMS API server.
    """
    app = AtlantisApp(api_url=api_url)
    app.run()


if __name__ == "__main__":
    main()
