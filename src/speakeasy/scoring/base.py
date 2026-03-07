"""Abstract base class for response scoring models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ResponseScorer(ABC):
    """Interface for response selection/scoring models."""

    @abstractmethod
    def get_reward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Score all responses in a nested query -> subquery -> lang -> response dict.

        Args:
            data: Nested dict mapping query -> subquery -> language -> response text.

        Returns:
            Same structure with each response replaced by a dict containing
            ``"response"`` and ``"score"`` (and optionally sub-scores).
        """
        ...
