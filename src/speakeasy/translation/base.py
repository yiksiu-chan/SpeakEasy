"""Abstract base class for translation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Translator(ABC):
    """Interface that all translation backends must implement."""

    @abstractmethod
    def translate_multilingual(self, sentence: str, source: str, target: str) -> str:
        """Translate a sentence from source language to target language.

        Args:
            sentence: Text to translate.
            source: Source language code (e.g., ``"en"``).
            target: Target language code (e.g., ``"zh-Hans"``).

        Returns:
            The translated text.
        """
        ...

    @abstractmethod
    def translate_to_english(
        self,
        sentence: str,
        source: str,
        target: str = "en",
        save_dir: str | None = None,
    ) -> str:
        """Translate a sentence to English, with optional caching.

        Args:
            sentence: Text to translate.
            source: Source language code.
            target: Target language code (default ``"en"``).
            save_dir: Optional path to a JSON file for caching translations.

        Returns:
            The translated text.
        """
        ...
