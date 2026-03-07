"""Translation backends for multilingual query processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speakeasy.translation.base import Translator


def get_translator(name: str) -> Translator:
    """Instantiate a translator backend by name.

    Args:
        name: One of ``"azure_translator"``, ``"google_cloud"``,
            or ``"deep_translator_google_translate"``.

    Returns:
        A translator instance.

    Raises:
        ValueError: If the translator name is not recognized.
    """
    if "azure_translator" in name:
        from speakeasy.translation.azure import AzureTranslator

        return AzureTranslator()
    elif "google_cloud" in name:
        from speakeasy.translation.google_cloud import GoogleCloudTranslator

        return GoogleCloudTranslator()
    elif "deep_translator" in name:
        from speakeasy.translation.deep import DeepTranslator

        return DeepTranslator()
    else:
        raise ValueError(
            f"Unknown translator '{name}'. "
            "Choose from: azure_translator, google_cloud, deep_translator_google_translate"
        )
