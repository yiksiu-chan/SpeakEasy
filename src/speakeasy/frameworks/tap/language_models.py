"""Language model wrappers used internally by the TAP framework."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import ray
import torch
from accelerate.utils import find_executable_batch_size

from speakeasy.frameworks.tap.model_utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)


class LanguageModel:
    """Base class for TAP language models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def batched_generate(
        self,
        prompts_list: list[Any],
        max_n_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError

    def is_initialized(self) -> None:
        logger.info("Initialized %s", self.model_name)

    def get_attribute(self, attr_name: str) -> Any:
        return getattr(self, attr_name, None)


class HuggingFace(LanguageModel):
    """HuggingFace Transformers model for TAP."""

    def __init__(self, model_name_or_path: str, **model_kwargs: Any) -> None:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path, **model_kwargs)
        self.model_name = model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_ids.append(self.tokenizer.encode("}", add_special_tokens=False)[0])
        self.generation_batch_size = 32

    def _batch_generate_bs(
        self,
        batch_size: int,
        inputs: list[str],
        **gen_kwargs: Any,
    ) -> list[str]:
        if batch_size != self.generation_batch_size:
            self.generation_batch_size = batch_size
        outputs = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding="longest")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded.to(self.model.device), **gen_kwargs
                ).cpu()
            if not self.model.config.is_encoder_decoder:
                output_ids = output_ids[:, encoded["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        return outputs

    def batched_generate(
        self,
        full_prompts_list: list[str],
        max_n_tokens: int,
        temperature: float,
        stop_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_n_tokens,
            "eos_token_id": self.eos_token_ids,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False

        gen_fn = find_executable_batch_size(self._batch_generate_bs, self.generation_batch_size)
        return gen_fn(full_prompts_list, **gen_kwargs)


class GPT(LanguageModel):
    """OpenAI-compatible API model for TAP.

    Works with OpenAI, or any OpenAI-compatible server (e.g. vLLM) by
    setting ``base_url``.
    """

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_MAX_RETRY = 5

    def __init__(
        self,
        model_name: str,
        token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        import openai

        self.model_name = model_name
        api_key = token or os.environ.get("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        conv: list[dict[str, str]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> str:
        import openai

        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = resp.choices[0].message.content or ""
                break
            except openai.RateLimitError:
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(0.5)
        return output

    def batched_generate(
        self,
        convs_list: list[list[dict[str, str]]],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> list[str]:
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class Gemini(LanguageModel):
    """Google Gemini model for TAP."""

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_MAX_RETRY = 5
    DEFAULT_REFUSAL = "I'm sorry, but I cannot assist with that request."

    def __init__(self, model_name: str, token: str | None = None) -> None:
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        self.model_name = model_name
        genai.configure(api_key=token or os.environ.get("GOOGLE_API_KEY", ""))
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.model = genai.GenerativeModel(model_name)
        self._genai = genai

    def generate(
        self,
        conv: Any,
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> str:
        output = self.API_ERROR_OUTPUT
        gen_config = self._genai.types.GenerationConfig(
            max_output_tokens=max_n_tokens, temperature=temperature, top_p=top_p
        )
        chat = self.model.start_chat(history=[])

        for _ in range(self.API_MAX_RETRY):
            try:
                completion = chat.send_message(
                    conv, generation_config=gen_config, safety_settings=self.safety_settings
                )
                output = completion.text
                break
            except (
                self._genai.types.BlockedPromptException,
                self._genai.types.StopCandidateException,
                ValueError,
            ):
                output = self.DEFAULT_REFUSAL
                break
            except Exception as e:
                logger.warning("Gemini error: %s", e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(1)
        return output

    def batched_generate(
        self,
        convs_list: list[Any],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> list[str]:
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


@ray.remote
class VLLM:
    """vLLM model managed as a Ray actor for TAP."""

    def __init__(self, model_name_or_path: str, num_gpus: int = 1, **model_kwargs: Any) -> None:
        from vllm import LLM

        self.model_name = model_name_or_path
        if num_gpus > 1:
            resources = ray.cluster_resources()
            available = ",".join(str(i) for i in range(int(resources.get("GPU", 0))))
            os.environ["CUDA_VISIBLE_DEVICES"] = available

        self.model = LLM(
            model=model_name_or_path,
            dtype=model_kwargs.get("dtype", "auto"),
            trust_remote_code=model_kwargs.get("trust_remote_code", False),
            tokenizer_mode="auto" if model_kwargs.get("use_fast_tokenizer", True) else "slow",
            tensor_parallel_size=num_gpus,
        )

    def batched_generate(
        self,
        full_prompts_list: list[str],
        max_n_tokens: int,
        temperature: float,
        stop_tokens: list[str] | None = None,
    ) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature, max_tokens=max_n_tokens, stop=stop_tokens or []
        )
        outputs = self.model.generate(full_prompts_list, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def is_initialized(self) -> None:
        logger.info("Initialized %s", self.model_name)
