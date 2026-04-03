"""Entry point for running the Atlantis TUI as a module.

Usage:
    atlantis                    # Standard form-based TUI
    atlantis --ai               # AI-powered chat interface (Ollama)
    atlantis --dual-llm         # Two-LLM architecture (transformers)
    atlantis --api-url URL      # Custom API URL
    python -m sms_api.tui
"""

# CRITICAL: Set environment variables BEFORE any other imports
# These must be set before torch/transformers/tokenizers are imported
import os
import sys

import typer
from typer import Option

from sms_api.common import StrEnumBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if sys.platform == "darwin":
    # Disable macOS fork safety check that causes "bad value(s) in fds_to_keep" errors
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import argparse

from sms_api.tui.app import DEFAULT_API_URL
from sms_api.tui.app import main as main_standard


agent_cli = typer.Typer()


class AgentMode(StrEnumBase):
    DUAL_LLM = "dual_llm"
    AI = "ai"


def cli() -> None:
    """CLI entry point for Atlantis TUI."""
    parser = argparse.ArgumentParser(
        description="Atlantis - SMS API Terminal User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run standard form-based TUI
    atlantis

    # Run AI-powered chat interface (Ollama-based)
    atlantis --ai

    # Run two-LLM architecture (transformers-based)
    atlantis --dual-llm

    # Run with custom API URL
    atlantis --api-url https://sms.cam.uchc.edu

    # AI mode without LLM (uses keyword matching)
    atlantis --ai --no-llm

Keyboard shortcuts (standard mode):
    q - Quit
    r - Focus Run Simulation tab
    s - Focus Status tab
    d - Focus Data tab
    i - Focus Simulator tab
    c - Clear results

AI mode commands:
    "List all simulations"
    "Check status of simulation 1"
    "Get data for simulation 1"
    "Run a new simulation"
    "Get the latest simulator"

Dual-LLM mode:
    Uses FunctionGemma for tool calling + Qwen2.5-7B for scientific reasoning.
    Requires ~15GB VRAM (or uses CPU offloading).
        """,
    )
    parser.add_argument(
        "--api-url",
        "-u",
        default=DEFAULT_API_URL,
        help=f"Base URL of the SMS API server (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Run AI-powered chat interface instead of form-based TUI",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM and use keyword matching only (faster startup, works offline)",
    )
    parser.add_argument(
        "--dual-llm",
        action="store_true",
        help="Use two-LLM architecture (FunctionGemma + Qwen2.5-7B via transformers)",
    )
    args = parser.parse_args()

    if args.dual_llm:
        from sms_api.tui.dual_llm_app import main as main_dual_llm

        main_dual_llm(api_url=args.api_url)
    elif args.ai:
        from sms_api.tui.llm_app import main as main_ai

        main_ai(api_url=args.api_url, use_llm=not args.no_llm)
    else:
        main_standard(api_url=args.api_url)


# if __name__ == "__main__":
#     cli()
