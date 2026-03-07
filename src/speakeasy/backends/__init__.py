"""Backend model wrappers for LLM inference.

Supports OpenAI API and vLLM (local) backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speakeasy.backends.base import Backend


def get_backend(model_name: str) -> Backend:
    """Instantiate the appropriate backend from a ``source:model`` string.

    Args:
        model_name: A string in the format ``"source:model_id"`` where source is
            one of ``openai`` or ``vllm``.

    Returns:
        A backend instance with an ``infer_batch`` method.

    Raises:
        ValueError: If the source prefix is not recognized.

    Examples:
        >>> backend = get_backend("openai:gpt-4o")
        >>> backend = get_backend("vllm:meta-llama/Llama-3.3-70B-Instruct")
    """
    source = model_name.split(":")[0]
    model_id = model_name.split(":", 1)[-1] if ":" in model_name else model_name

    if source == "openai":
        from speakeasy.backends.openai import OpenAIBackend

        return OpenAIBackend(model_id)
    elif source == "vllm":
        from speakeasy.backends.vllm import VLLMBackend

        return VLLMBackend(model_id)
    else:
        raise ValueError(f"Unknown backend source '{source}'. Choose from: openai, vllm")
