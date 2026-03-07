"""vLLM backend for local GPU inference."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from speakeasy.backends.base import Backend
from speakeasy.config import VLLMConfig

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """vLLM backend with automatic tensor parallelism across available GPUs.

    Args:
        model_name: HuggingFace model identifier or local path.
        config: Optional configuration override.
    """

    def __init__(self, model_name: str, config: VLLMConfig | None = None) -> None:
        from vllm import LLM, SamplingParams

        self.config = config or VLLMConfig()
        self.model_name = model_name
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            import ray

            ray.init(ignore_reinit_error=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=getattr(torch, self.config.dtype, torch.float16),
            max_model_len=self.config.max_model_len,
        )

    def infer_batch(
        self,
        inputs: list[str],
        save_dir: str | None = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> list[str]:
        """Generate responses for a batch of inputs.

        Args:
            inputs: List of prompt strings.
            save_dir: Optional path to save intermediate results for resumption.
            batch_size: Number of prompts per generation call.

        Returns:
            List of response strings.
        """
        responses: list[str] = []
        if save_dir and os.path.exists(save_dir):
            with open(save_dir) as f:
                responses = json.load(f)

        start_index = len(responses)
        for i in tqdm(range(start_index, len(inputs), batch_size), desc="Processing"):
            batch = inputs[i : i + batch_size]
            prompts = []
            for text in batch:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": text},
                ]
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            try:
                outputs = self.llm.generate(prompts, self.sampling_params)
            except Exception as e:
                logger.error("Generation error: %s", e)
                outputs = None

            for idx in range(len(batch)):
                if outputs and idx < len(outputs) and outputs[idx].outputs:
                    responses.append(outputs[idx].outputs[0].text)
                else:
                    responses.append("")

            if save_dir:
                with open(save_dir, "w") as f:
                    json.dump(responses, f, indent=4)

        return responses
