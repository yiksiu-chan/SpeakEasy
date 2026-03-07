"""OpenAI API backend with multiprocessing support."""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from typing import Any

import openai
from openai import OpenAI
from tqdm import tqdm

from speakeasy.backends.base import Backend
from speakeasy.config import OpenAIConfig
from speakeasy.constants import CHAT_MODELS
from speakeasy.utils import partition

logger = logging.getLogger(__name__)


def _get_response(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    """Send a single prompt to the OpenAI API and return the response."""
    is_chat = any(name in model for name in CHAT_MODELS)
    if is_chat:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return resp.choices[0].message.content or ""
    else:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
        )
        return resp.choices[0].text or ""


def _query_worker(
    model: str,
    inputs: list[str | list[str]],
    process_id: int,
    lock: Any,
    temperature: float,
    max_tokens: int,
    top_p: float,
    api_key: str,
    base_url: str | None = None,
) -> list[str | list[str]]:
    """Worker function for multiprocessing inference."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    def _query_single(prompt: str) -> str:
        if prompt == "":
            return ""
        for _ in range(5):
            try:
                return _get_response(client, prompt, model, temperature, max_tokens, top_p)
            except openai.RateLimitError:
                logger.warning("Rate limited, retrying in 30s...")
                time.sleep(30)
            except openai.BadRequestError:
                logger.error("Bad request for prompt: %s", prompt[:100])
                return ""
            except Exception as e:
                logger.warning("API error: %s, retrying...", e)
                time.sleep(5)
        logger.error("Failed after retries for prompt: %s", prompt[:100])
        return ""

    with lock:
        bar = tqdm(
            desc=f"Process {process_id + 1}",
            total=len(inputs),
            position=process_id + 1,
            leave=False,
        )

    responses: list[str | list[str]] = []
    for instance in inputs:
        with lock:
            bar.update(1)
        if isinstance(instance, list):
            responses.append([_query_single(item) for item in instance])
        else:
            responses.append(_query_single(instance))

    with lock:
        bar.close()
    return responses


class OpenAIBackend(Backend):
    """OpenAI API backend with multiprocessing for parallel requests.

    Args:
        model_name: The OpenAI model identifier (e.g., ``"gpt-4o"``).
        config: Optional configuration override.
    """

    def __init__(self, model_name: str, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        self.model = model_name
        self.api_key = self.config.api_key or os.environ.get("SPEAKEASY_OPENAI_API_KEY", "")
        self.base_url = (
            self.config.base_url or os.environ.get("SPEAKEASY_OPENAI_BASE_URL", "") or None
        )

    def infer_batch(self, inputs: list[str], **kwargs: Any) -> list[str]:
        """Run batch inference using multiprocessing.

        Args:
            inputs: List of prompt strings.

        Returns:
            List of response strings.
        """
        num_procs = min(self.config.num_processes, len(inputs))
        if num_procs == 0:
            return []

        partitioned = partition(inputs, num_procs)
        manager = multiprocessing.Manager()
        lock = manager.Lock()
        pool = multiprocessing.Pool(processes=num_procs)

        try:
            futures = []
            for pid, chunk in enumerate(partitioned):
                args = (
                    self.model,
                    chunk,
                    pid,
                    lock,
                    self.config.temperature,
                    self.config.max_tokens,
                    self.config.top_p,
                    self.api_key,
                    self.base_url,
                )
                futures.append(pool.apply_async(_query_worker, args=args))

            pool.close()
            pool.join()

            results: list[str] = []
            for f in futures:
                results.extend(f.get())
            return results
        finally:
            manager.shutdown()
