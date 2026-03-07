"""Model loading and template utilities for TAP."""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastchat.conversation import get_conv_template
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto",
}


def get_template(
    model_name_or_path: str | None = None,
    chat_template: str | None = None,
    fschat_template: str | None = None,
    system_message: str | None = None,
    return_fschat_conv: bool = False,
    **kwargs: Any,
) -> Any:
    """Get the appropriate chat template for a model.

    Tries in order: FastChat template, legacy templates, tokenizer.apply_chat_template.
    """
    if fschat_template or return_fschat_conv:
        conv = _get_fschat_conv(model_name_or_path, fschat_template, system_message)
        if return_fschat_conv:
            return conv
        conv.append_message(conv.roles[0], "{instruction}")
        conv.append_message(conv.roles[1], None)
        return {"description": f"fschat template {conv.name}", "prompt": conv.get_prompt()}

    _TEMPLATES = {
        "vicuna": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
        "wizard": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
        "llama-2": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n{instruction} [/INST] ",
        "mistral": "[INST] {instruction} [/INST]",
        "mixtral": "[INST] {instruction} [/INST]",
        "qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    }

    if chat_template and chat_template in _TEMPLATES:
        return {"description": f"Template for {chat_template}", "prompt": _TEMPLATES[chat_template]}

    # Fall back to tokenizer.apply_chat_template
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        messages = [{"role": "user", "content": "{instruction}"}]
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token) :]
        return {"description": f"Template for {model_name_or_path}", "prompt": prompt}
    except Exception as e:
        raise ValueError(f"Cannot find template for {model_name_or_path}: {e}")


def _get_fschat_conv(
    model_name_or_path: str | None = None,
    fschat_template: str | None = None,
    system_message: str | None = None,
) -> Any:
    """Get a FastChat conversation template."""
    template_name = fschat_template or model_name_or_path
    if fschat_template:
        template = get_conv_template(fschat_template)
    else:
        template = get_conversation_template(template_name)

    if system_message:
        template.set_system_message(system_message)
    return template


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    revision: str | None = None,
    token: str | None = None,
    num_gpus: int = 1,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
    legacy: bool = False,
    pad_token: str | None = None,
    eos_token: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a HuggingFace model and tokenizer for TAP."""
    torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        legacy=legacy,
        padding_side=padding_side,
    )

    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer
