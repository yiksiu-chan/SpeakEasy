"""Utility functions for the Speak Easy framework."""

from __future__ import annotations

import math
import re
from typing import Any

from speakeasy.constants import REFUSAL_WORDS


def partition(obj: list[Any], num_partitions: int) -> list[list[Any]]:
    """Split a list into ``num_partitions`` roughly equal chunks.

    >>> partition([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    chunk_size = len(obj) // num_partitions
    chunks: list[list[Any]] = []
    buf: list[Any] = []
    for item in obj:
        if len(buf) >= chunk_size and len(chunks) < num_partitions - 1:
            chunks.append(buf)
            buf = []
        buf.append(item)
    if buf:
        chunks.append(buf)
    return chunks


def check_format(data_dir: str) -> bool:
    """Validate that a JSON file contains a list of dicts with ``query`` and ``response`` keys."""
    import json
    import os

    if not os.path.exists(data_dir):
        return False
    try:
        with open(data_dir, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, list):
        return False
    return all(isinstance(item, dict) and "query" in item and "response" in item for item in data)


def contains_refusal(text: str) -> bool:
    """Check whether ``text`` contains any refusal phrases."""
    text_lower = text.lower()
    return any(word in text_lower for word in REFUSAL_WORDS)


def extract_subqueries(output: str, num_subqueries: int) -> list[str]:
    """Extract numbered sub-queries (questions ending with ``?``) from LLM output.

    >>> extract_subqueries("1. What are colors?\\n2. What is a rainbow?", 2)
    ['What are colors?', 'What is a rainbow?']
    """
    if not isinstance(num_subqueries, int) or num_subqueries <= 0:
        raise ValueError("num_subqueries must be a positive integer")
    pattern = r"(?:^|\n)\s*(?:\d+[\.)])\s*([^.]*?\?)"
    matches = re.findall(pattern, output, re.DOTALL)
    return [m.strip().strip('"') for m in matches[:num_subqueries]]


def reduce_repeated_phrases(text: str) -> str:
    """Collapse excessively repeated words and phrases in ``text``."""
    # Reduce single words repeated 5+ times
    text = re.sub(
        r"(\S+)(?:\s+\1){5,}",
        lambda m: m.group(1),
        text,
        flags=re.UNICODE,
    )
    # Reduce repeated phrases
    text = re.compile(r"\b(.+?)(?:\s+\1)+\b", re.DOTALL).sub(r"\1", text)
    return text


def truncate_strings(strings: list[str], tokenizer: Any, max_tokens: int = 256) -> list[str]:
    """Truncate each string to at most ``max_tokens`` using the given tokenizer."""
    result: list[str] = []
    for s in strings:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            result.append(tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True))
        else:
            result.append(s)
    return result


def geometric_mean(a: float, b: float) -> float:
    """Compute the geometric mean of two non-negative numbers."""
    return math.sqrt(a * b)


def detect_benchmark(save_dir: str) -> str:
    """Infer the benchmark name from the save directory path."""
    path_lower = save_dir.lower()
    if "sorry" in path_lower:
        return "sorrybench"
    elif "med" in path_lower:
        return "medharm"
    elif "harm" in path_lower:
        return "harmbench"
    elif "adv" in path_lower:
        return "advbench"
    return "unknown"
