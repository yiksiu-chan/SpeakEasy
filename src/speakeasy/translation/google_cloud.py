"""Google Cloud Translation API backend."""

from __future__ import annotations

import os

from speakeasy.translation.base import Translator


class GoogleCloudTranslator(Translator):
    """Translator using the Google Cloud Translation v2 API.

    Credentials are loaded from the path specified by the
    ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable or the
    ``SPEAKEASY_GOOGLE_CREDENTIALS_PATH`` environment variable.
    """

    def __init__(self, credentials_path: str | None = None) -> None:
        from google.cloud import translate_v2 as translate
        from google.oauth2 import service_account

        path = credentials_path or os.environ.get(
            "SPEAKEASY_GOOGLE_CREDENTIALS_PATH",
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        )
        credentials = service_account.Credentials.from_service_account_file(path)
        self.client = translate.Client(credentials=credentials)

    def translate_multilingual(self, sentence: str, source: str, target: str) -> str:
        if not sentence:
            return ""
        result = self.client.translate(sentence, source_language=source, target_language=target)
        return result["translatedText"]

    def translate_to_english(
        self,
        sentence: str,
        source: str,
        target: str = "en",
        save_dir: str | None = None,
    ) -> str:
        return self.translate_multilingual(sentence, source, target)
