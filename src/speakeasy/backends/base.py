"""Abstract base class for all model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Interface that all model backends must implement."""

    @abstractmethod
    def infer_batch(self, inputs: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for a batch of input prompts.

        Args:
            inputs: A list of prompt strings.
            **kwargs: Backend-specific keyword arguments (e.g., ``save_dir``).

        Returns:
            A list of generated response strings, one per input.
        """
        ...
