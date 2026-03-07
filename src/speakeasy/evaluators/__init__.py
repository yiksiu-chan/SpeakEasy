"""Evaluation models for scoring jailbreak responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speakeasy.evaluators.base import Evaluator


def get_evaluator(metric_name: str) -> Evaluator:
    """Instantiate an evaluation model by name.

    Args:
        metric_name: One of ``"gpt4judge"`` or ``"harmscore"``.

    Returns:
        An evaluator instance with a ``compute_scores`` method.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if metric_name == "gpt4judge":
        from speakeasy.evaluators.gpt4_judge import GPT4Judge

        return GPT4Judge()
    elif metric_name == "harmscore":
        from speakeasy.evaluators.harmscore import HarmScore

        return HarmScore()
    else:
        raise ValueError(f"Unknown evaluator '{metric_name}'. Choose from: gpt4judge, harmscore")
