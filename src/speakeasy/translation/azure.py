"""Azure Cognitive Services translator backend."""

from __future__ import annotations

import json
import logging
import os

from speakeasy.translation.base import Translator

logger = logging.getLogger(__name__)


class AzureTranslator(Translator):
    """Translator using Azure Cognitive Services Translation API.

    Credentials are read from environment variables:
        - ``SPEAKEASY_AZURE_TRANSLATOR_KEY``
        - ``SPEAKEASY_AZURE_TRANSLATOR_REGION``

    Args:
        api_key: Azure API key (overrides env var).
        region: Azure region (overrides env var).
    """

    def __init__(
        self,
        api_key: str | None = None,
        region: str | None = None,
    ) -> None:
        from azure.ai.translation.text import TextTranslationClient
        from azure.core.credentials import AzureKeyCredential

        self.api_key = api_key or os.environ.get("SPEAKEASY_AZURE_TRANSLATOR_KEY", "")
        self.region = region or os.environ.get("SPEAKEASY_AZURE_TRANSLATOR_REGION", "eastus")
        self.client = TextTranslationClient(
            credential=AzureKeyCredential(self.api_key),
            region=self.region,
        )

    def _translate(self, sentence: str | list[str], target: str) -> str | list[str]:
        """Perform the raw translation call."""
        if isinstance(sentence, list):
            result = self.client.translate(body=sentence, to_language=[target])
            return [t.translations[0].text for t in result]
        result = self.client.translate(body=[sentence], to_language=[target])
        return result[0].translations[0].text

    def translate_multilingual(self, sentence: str, source: str, target: str) -> str:
        if not sentence:
            return ""
        if source == target == "en":
            return sentence
        try:
            return self._translate(sentence, target)
        except Exception as e:
            logger.error("Translation error: %s", e)
            raise

    def translate_to_english(
        self,
        sentence: str,
        source: str,
        target: str = "en",
        save_dir: str | None = None,
    ) -> str:
        if not sentence:
            return ""
        if source == target == "en":
            return sentence

        # Check cache
        cache: dict[str, str] = {}
        if save_dir and os.path.exists(save_dir):
            with open(save_dir) as f:
                cache = json.load(f)
        if sentence in cache:
            return cache[sentence]

        result = self._translate(sentence, target)
        translated = result if isinstance(result, str) else str(result)
        cache[sentence] = translated
        if save_dir:
            with open(save_dir, "w") as f:
                json.dump(cache, f, indent=4)
        return translated
