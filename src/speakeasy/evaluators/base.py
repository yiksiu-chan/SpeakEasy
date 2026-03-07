"""Abstract base class for evaluation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Interface that all evaluation models must implement."""

    @abstractmethod
    def compute_scores(self, qa_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a list of query-response pairs.

        Args:
            qa_pairs: A list of dicts, each with at least ``"query"`` and ``"response"`` keys.

        Returns:
            The same list with an added ``"score"`` key per entry.
        """
        ...
