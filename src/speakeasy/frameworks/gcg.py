"""GCG (Greedy Coordinate Gradient) baseline — appends adversarial suffixes."""

from __future__ import annotations

import json
import os
from typing import Any

from speakeasy.backends.base import Backend
from speakeasy.config import GCGSuffixes
from speakeasy.frameworks.base import Framework
from speakeasy.utils import detect_benchmark


class GCG(Framework):
    """Baseline that appends GCG adversarial suffixes to queries.

    The adversarial suffix is selected based on the benchmark inferred from
    the save directory path.

    Args:
        model: Backend for LLM inference.
        suffixes: Optional override for adversarial suffixes.
    """

    def __init__(self, model: Backend, suffixes: GCGSuffixes | None = None) -> None:
        super().__init__(model)
        self.suffixes = suffixes or GCGSuffixes()

    def _get_suffix(self, save_dir: str) -> str:
        """Select the adversarial suffix based on the benchmark."""
        benchmark = detect_benchmark(save_dir)
        suffix_map = {
            "sorrybench": self.suffixes.sorrybench,
            "medharm": self.suffixes.medharm,
            "harmbench": self.suffixes.harmbench,
            "advbench": self.suffixes.advbench,
        }
        return suffix_map.get(benchmark, self.suffixes.advbench)

    def infer(self, data: list[dict[str, Any]], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "combined_responses.json")

        if os.path.exists(save_path):
            return

        suffix = self._get_suffix(save_dir)
        prompts = [query["query"] + suffix for query in data]
        outputs = self.model.infer_batch(prompts, save_dir=save_path)

        qa_pairs = {}
        for query_dict, response in zip(data, outputs):
            qa_pairs[query_dict["query"]] = response if response else ""

        with open(save_path, "w") as f:
            json.dump(qa_pairs, f, indent=4)
