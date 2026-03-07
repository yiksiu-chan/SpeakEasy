"""Direct Request (DR) baseline — queries the model directly without modification."""

from __future__ import annotations

import json
import os
from typing import Any

from speakeasy.backends.base import Backend
from speakeasy.frameworks.base import Framework


class DirectRequest(Framework):
    """Baseline that sends queries directly to the target model.

    This is the simplest baseline: no adversarial modifications are applied.
    """

    def __init__(self, model: Backend) -> None:
        super().__init__(model)

    def infer(self, data: list[dict[str, Any]], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "combined_responses.json")

        if os.path.exists(save_path):
            return

        prompts = [instance["query"] for instance in data]
        outputs = self.model.infer_batch(prompts, save_dir=save_path)

        qa_pairs = {}
        for query_dict, response in zip(data, outputs):
            qa_pairs[query_dict["query"]] = response if response else ""

        with open(save_path, "w") as f:
            json.dump(qa_pairs, f, indent=4)
