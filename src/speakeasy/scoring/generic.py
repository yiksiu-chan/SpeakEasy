"""Generic HuggingFace sentiment model scorer."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from speakeasy.scoring.base import ResponseScorer


class GenericScorer(ResponseScorer):
    """Scores responses using any HuggingFace sentiment-analysis model.

    Args:
        checkpoint: HuggingFace model identifier.
        device: GPU device index.
    """

    def __init__(self, checkpoint: str, device: int = 0) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.pipe_kwargs: dict[str, Any] = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 32,
        }

    def _score_single(self, subquery: str, response: str) -> float:
        """Score a single subquery-response pair."""
        chat = [
            {"role": "user", "content": subquery},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        ).replace(self.tokenizer.bos_token or "", "")
        out = self.pipe([text], **self.pipe_kwargs)
        return out[0][0]["score"]

    def get_reward(self, data: dict[str, Any]) -> dict[str, Any]:
        scores: dict[str, Any] = {}
        for query, sub_queries in data.items():
            scores[query] = {}
            for subquery, responses in sub_queries.items():
                scores[query][subquery] = {}
                for lang, response in responses.items():
                    scores[query][subquery][lang] = {
                        "response": response,
                        "score": self._score_single(subquery, response),
                    }
        return scores
