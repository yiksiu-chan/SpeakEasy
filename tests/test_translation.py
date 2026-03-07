"""Tests for translation backends."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from speakeasy.translation import get_translator
from speakeasy.translation.deep import _AZURE_TO_GOOGLE, DeepTranslator

# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestGetTranslator:
    def test_deep_translator(self):
        t = get_translator("deep_translator_google_translate")
        assert isinstance(t, DeepTranslator)

    def test_azure_translator_import(self):
        """Factory selects AzureTranslator, but we mock __init__ to avoid credentials."""
        with patch("speakeasy.translation.azure.AzureTranslator.__init__", return_value=None):
            t = get_translator("azure_translator")
            from speakeasy.translation.azure import AzureTranslator

            assert isinstance(t, AzureTranslator)

    def test_google_cloud_import(self):
        with patch(
            "speakeasy.translation.google_cloud.GoogleCloudTranslator.__init__", return_value=None
        ):
            t = get_translator("google_cloud")
            from speakeasy.translation.google_cloud import GoogleCloudTranslator

            assert isinstance(t, GoogleCloudTranslator)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown translator"):
            get_translator("nonexistent")


# ---------------------------------------------------------------------------
# DeepTranslator tests
# ---------------------------------------------------------------------------


class TestDeepTranslator:
    def test_language_code_mapping(self):
        assert _AZURE_TO_GOOGLE["zh-Hans"] == "zh-CN"
        assert _AZURE_TO_GOOGLE["zh-Hant"] == "zh-TW"

    def test_empty_input(self):
        dt = DeepTranslator()
        assert dt.translate_multilingual("", "en", "fr") == ""

    def test_translate_to_english_delegates(self):
        dt = DeepTranslator()
        with patch.object(dt, "_translate", return_value="hello") as mock:
            result = dt.translate_to_english("bonjour", "fr", "en")
            mock.assert_called_once_with("bonjour", "fr", "en")
            assert result == "hello"

    def test_translate_maps_azure_codes(self):
        dt = DeepTranslator()
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "translated"

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            result = dt._translate("hello", "en", "zh-Hans")
            assert result == "translated"

    def test_translate_retry_on_none(self):
        dt = DeepTranslator()
        mock_translator = MagicMock()
        mock_translator.translate.side_effect = [None, None, "got it"]

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            result = dt._translate("test", "en", "fr")
            assert result == "got it"
            assert mock_translator.translate.call_count == 3

    def test_translate_all_retries_none(self):
        dt = DeepTranslator()
        mock_translator = MagicMock()
        mock_translator.translate.return_value = None

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            result = dt._translate("test", "en", "fr")
            assert result == ""
            assert mock_translator.translate.call_count == 5

    def test_translate_timeout_returns_empty(self):
        dt = DeepTranslator(timeout=1)

        def slow_translate(text):
            import time

            time.sleep(5)
            return text

        mock_translator = MagicMock()
        mock_translator.translate.side_effect = slow_translate

        with patch("deep_translator.GoogleTranslator", return_value=mock_translator):
            result = dt._translate("test", "en", "fr")
            assert result == ""


# ---------------------------------------------------------------------------
# AzureTranslator tests
# ---------------------------------------------------------------------------


class TestAzureTranslator:
    def _make_translator(self):
        """Create an AzureTranslator with mocked Azure client."""
        with patch("speakeasy.translation.azure.AzureTranslator.__init__", return_value=None):
            from speakeasy.translation.azure import AzureTranslator

            t = AzureTranslator()
            t.client = MagicMock()
            t.api_key = "test"
            t.region = "eastus"
            return t

    def test_empty_input(self):
        t = self._make_translator()
        assert t.translate_multilingual("", "en", "fr") == ""

    def test_same_language_passthrough(self):
        t = self._make_translator()
        assert t.translate_to_english("hello", "en", "en") == "hello"

    def test_translate_multilingual_calls_client(self):
        t = self._make_translator()
        mock_result = MagicMock()
        mock_result.translations = [MagicMock(text="bonjour")]
        t.client.translate.return_value = [mock_result]

        result = t.translate_multilingual("hello", "en", "fr")
        assert result == "bonjour"
        t.client.translate.assert_called_once()

    def test_translate_to_english_caching(self, tmp_path):
        t = self._make_translator()
        cache_file = str(tmp_path / "cache.json")

        mock_result = MagicMock()
        mock_result.translations = [MagicMock(text="hello")]
        t.client.translate.return_value = [mock_result]

        # First call: translates and caches
        result = t.translate_to_english("bonjour", "fr", "en", save_dir=cache_file)
        assert result == "hello"

        # Verify cache was written
        with open(cache_file) as f:
            cache = json.load(f)
        assert cache["bonjour"] == "hello"

        # Second call: reads from cache
        t.client.translate.reset_mock()
        result = t.translate_to_english("bonjour", "fr", "en", save_dir=cache_file)
        assert result == "hello"
        t.client.translate.assert_not_called()


# ---------------------------------------------------------------------------
# GoogleCloudTranslator tests
# ---------------------------------------------------------------------------


class TestGoogleCloudTranslator:
    def _make_translator(self):
        with patch(
            "speakeasy.translation.google_cloud.GoogleCloudTranslator.__init__", return_value=None
        ):
            from speakeasy.translation.google_cloud import GoogleCloudTranslator

            t = GoogleCloudTranslator()
            t.client = MagicMock()
            return t

    def test_empty_input(self):
        t = self._make_translator()
        assert t.translate_multilingual("", "en", "fr") == ""

    def test_translate_multilingual(self):
        t = self._make_translator()
        t.client.translate.return_value = {"translatedText": "bonjour"}
        result = t.translate_multilingual("hello", "en", "fr")
        assert result == "bonjour"

    def test_translate_to_english_delegates(self):
        t = self._make_translator()
        t.client.translate.return_value = {"translatedText": "hello"}
        result = t.translate_to_english("bonjour", "fr", "en")
        assert result == "hello"
