"""SMS API Terminal User Interface (TUI) package.

A colorful TUI built with Textual for interacting with SMS API endpoints.

Three modes are available:
- Standard mode: Form-based interface for direct API interaction
- AI mode: Chat interface with LLM-powered natural language understanding (Ollama)
- Dual-LLM mode: Two-LLM architecture (FunctionGemma + Qwen2.5) via transformers
"""

from sms_api.tui.__main__ import AgentMode, agent_cli
from sms_api.tui.app import AtlantisApp
from sms_api.tui.dual_llm_agent import DualLLMAgent
from sms_api.tui.dual_llm_app import DualLLMApp
from sms_api.tui.llm_app import LLMApp

__all__ = ["AtlantisApp", "DualLLMAgent", "DualLLMApp", "LLMApp", "agent_cli", "AgentMode"]
