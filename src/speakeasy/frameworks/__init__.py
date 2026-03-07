"""Attack frameworks: baselines (DR, GCG, TAP) and Speak Easy variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from speakeasy.backends.base import Backend
    from speakeasy.frameworks.base import Framework


def get_framework(
    name: str,
    model: Backend,
    device: int = 0,
    **kwargs: Any,
) -> Framework:
    """Instantiate an attack framework by name.

    Args:
        name: One of ``baseline_dr``, ``baseline_gcg``, ``baseline_tap``,
            ``speakeasy_dr``, ``speakeasy_gcg``, ``speakeasy_tap``.
        model: A backend instance for LLM inference.
        device: GPU device index (used by Speak Easy variants).

    Returns:
        A framework instance with an ``infer`` method.

    Raises:
        ValueError: If the framework name is not recognized.
    """
    if name == "baseline_dr":
        from speakeasy.frameworks.direct_request import DirectRequest

        return DirectRequest(model)

    elif name == "baseline_gcg":
        from speakeasy.frameworks.gcg import GCG

        return GCG(model)

    elif name == "baseline_tap":
        from speakeasy.frameworks.tap import TAP

        return TAP(model)

    elif name == "speakeasy_dr":
        from speakeasy.frameworks.direct_request import DirectRequest
        from speakeasy.frameworks.speakeasy import SpeakEasyPipeline

        return SpeakEasyPipeline(model, device=device, attack_method=None)

    elif name == "speakeasy_gcg":
        from speakeasy.frameworks.speakeasy import SpeakEasyPipeline

        return SpeakEasyPipeline(model, device=device, attack_method="gcg")

    elif name == "speakeasy_tap":
        from speakeasy.frameworks.speakeasy import SpeakEasyPipeline

        return SpeakEasyPipeline(model, device=device, attack_method="tap")

    else:
        raise ValueError(
            f"Unknown framework '{name}'. Choose from: "
            "baseline_dr, baseline_gcg, baseline_tap, "
            "speakeasy_dr, speakeasy_gcg, speakeasy_tap"
        )
