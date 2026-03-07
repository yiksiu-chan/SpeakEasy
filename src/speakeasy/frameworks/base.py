"""Abstract base class for attack frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from speakeasy.backends.base import Backend


class Framework(ABC):
    """Interface that all attack frameworks must implement."""

    def __init__(self, model: Backend) -> None:
        self.model = model

    @abstractmethod
    def infer(self, data: list[dict[str, Any]], save_dir: str) -> None:
        """Run the attack framework on the given data and save results.

        Args:
            data: List of dicts with at least a ``"query"`` key.
            save_dir: Directory to save results and intermediate files.
        """
        ...
