"""Device detection and model loading utilities for the two-LLM architecture."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from enum import StrEnum
from functools import lru_cache
from typing import Any

# Disable tokenizer parallelism to avoid multiprocessing issues on macOS
# This must be set BEFORE importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable macOS fork safety check that causes "bad value(s) in fds_to_ke" errors
# This is needed when using multiprocessing/threading with tokenizers on macOS
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Set multiprocessing start method to 'spawn' on macOS to avoid fork issues
# This must be done before any multiprocessing is used
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set - check if it's spawn (which is fine) or something else
        current = multiprocessing.get_start_method()
        if current != "spawn":
            import warnings

            warnings.warn(f"Multiprocessing start method already set to '{current}', not 'spawn'")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Model IDs
TOOL_LLM_ID: str = "google/functiongemma-270m-it"
RESPONSE_LLM_ID: str = "Qwen/Qwen2.5-7B-Instruct"

# Generation settings
MAX_NEW_TOKENS: int = 256
RESPONSE_MAX_TOKENS: int = 512


class Device(StrEnum):
    """Supported compute devices."""

    MPS = "mps"
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

    @classmethod
    def types(cls) -> list[str]:
        """Get all device type names."""
        return [a for a in dir(Device) if a.isupper()]


class DeviceMapping(dict[Any, Any]):
    """Dictionary wrapper for device mapping with convenience property."""

    @property
    def device(self) -> Any:
        """Get the first device value from the mapping."""
        return next(iter(self.values()))


def get_device(as_mapping: bool = False) -> DeviceMapping | str:
    """Auto-detect the best available compute device.

    Args:
        as_mapping: If True, return as DeviceMapping dict, otherwise as string.

    Returns:
        Device string or DeviceMapping for model loading.
    """
    dev = Device.MPS if torch.backends.mps.is_available() else Device.CUDA if torch.cuda.is_available() else Device.CPU
    return DeviceMapping({"": dev}) if as_mapping else dev


def get_torch_dtype() -> torch.dtype:
    """Get appropriate torch dtype for the current device."""
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=1)
def load_tool_llm_cpu(model_id: str | None = None) -> Any:
    """Load tool LLM on CPU."""
    mid = model_id or TOOL_LLM_ID
    logger.info(f"Loading tool LLM on CPU: {mid}")
    return AutoModelForCausalLM.from_pretrained(mid)


@lru_cache(maxsize=1)
def load_tool_llm_mps(model_id: str | None = None) -> Any:
    """Load tool LLM on Apple MPS."""
    mid = model_id or TOOL_LLM_ID
    logger.info(f"Loading tool LLM on MPS: {mid}")
    return AutoModelForCausalLM.from_pretrained(mid).to("mps")  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def load_tool_llm_cuda(model_id: str | None = None) -> Any:
    """Load tool LLM on CUDA."""
    mid = model_id or TOOL_LLM_ID
    logger.info(f"Loading tool LLM on CUDA: {mid}")
    return AutoModelForCausalLM.from_pretrained(
        mid,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )


@lru_cache(maxsize=1)
def load_response_llm_cpu(model_id: str | None = None) -> Any:
    """Load response LLM on CPU."""
    mid = model_id or RESPONSE_LLM_ID
    logger.info(f"Loading response LLM on CPU: {mid}")
    return AutoModelForCausalLM.from_pretrained(mid)


@lru_cache(maxsize=1)
def load_response_llm_mps(model_id: str | None = None) -> Any:
    """Load response LLM on Apple MPS."""
    mid = model_id or RESPONSE_LLM_ID
    logger.info(f"Loading response LLM on MPS: {mid}")
    return AutoModelForCausalLM.from_pretrained(mid).to("mps")  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def load_response_llm_cuda(model_id: str | None = None) -> Any:
    """Load response LLM on CUDA with automatic device mapping."""
    mid = model_id or RESPONSE_LLM_ID
    logger.info(f"Loading response LLM on CUDA: {mid}")
    return AutoModelForCausalLM.from_pretrained(
        mid,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )


def load_tool_llm(model_id: str | None = None) -> Any:
    """Load the tool LLM on the best available device.

    Args:
        model_id: Optional model ID override.

    Returns:
        Loaded model on appropriate device.
    """
    device = get_device()
    if device == Device.MPS:
        return load_tool_llm_mps(model_id)
    elif device == Device.CUDA:
        return load_tool_llm_cuda(model_id)
    else:
        return load_tool_llm_cpu(model_id)


def load_response_llm(model_id: str | None = None) -> Any:
    """Load the response LLM on the best available device.

    Args:
        model_id: Optional model ID override.

    Returns:
        Loaded model on appropriate device.
    """
    device = get_device()
    if device == Device.MPS:
        return load_response_llm_mps(model_id)
    elif device == Device.CUDA:
        return load_response_llm_cuda(model_id)
    else:
        return load_response_llm_cpu(model_id)


@lru_cache(maxsize=4)
def load_tokenizer(model_id: str) -> Any:
    """Load and configure a tokenizer for the given model.

    Args:
        model_id: The HuggingFace model ID.

    Returns:
        Configured tokenizer with padding token set.
    """
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]

    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# Lightweight chat model ID (~135M parameters)
LIGHTWEIGHT_MODEL_ID: str = "HuggingFaceTB/SmolLM-135M-Instruct"


@lru_cache(maxsize=1)
def _load_lightweight_model() -> tuple[Any, Any]:
    """Load the lightweight model and tokenizer (cached)."""
    device = get_device()
    logger.info(f"Loading lightweight model on {device}: {LIGHTWEIGHT_MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(LIGHTWEIGHT_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(LIGHTWEIGHT_MODEL_ID)

    if device == Device.MPS:
        model = model.to("mps")
    elif device == Device.CUDA:
        model = model.to("cuda")

    return model, tokenizer


def generate_response(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate a natural language response to a text prompt using a lightweight LLM.

    Uses SmolLM-135M-Instruct (~135M parameters) for fast, low-resource inference.

    Args:
        prompt: The user's text prompt (e.g., "who are you").
        max_new_tokens: Maximum number of tokens to generate. Defaults to 128.

    Returns:
        The model's natural language response as a string.
    """
    model, tokenizer = _load_lightweight_model()
    device = get_device()

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(chat_input, return_tensors="pt")
    if device in (Device.MPS, Device.CUDA):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (skip the input)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response.strip()
