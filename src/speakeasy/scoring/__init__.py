"""Response selection models for scoring and ranking multilingual responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speakeasy.scoring.base import ResponseScorer


def get_scorer(name: str, device: int = 0) -> ResponseScorer:
    """Instantiate a response scorer by name.

    Args:
        name: One of ``"harmscore"`` or ``"generic:model_name"``.
        device: GPU device index.

    Returns:
        A scorer instance with a ``get_reward`` method.

    Raises:
        ValueError: If the scorer name is not recognized.
    """
    if "harmscore" in name:
        from speakeasy.scoring.harmscore import HarmScoreScorer

        return HarmScoreScorer(device=device)
    elif "generic" in name:
        model_id = name.split(":", 1)[-1] if ":" in name else name
        from speakeasy.scoring.generic import GenericScorer

        return GenericScorer(model_id, device=device)
    else:
        raise ValueError(f"Unknown scorer '{name}'. Choose from: harmscore, generic:<model>")
