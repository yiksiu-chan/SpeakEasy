"""Tests for module factory functions (get_backend, get_framework)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speakeasy.backends import get_backend
from speakeasy.frameworks import get_framework

# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_openai_backend(self):
        with patch("speakeasy.backends.openai.OpenAIBackend.__init__", return_value=None):
            b = get_backend("openai:gpt-4o")
            from speakeasy.backends.openai import OpenAIBackend

            assert isinstance(b, OpenAIBackend)

    def test_vllm_backend(self):
        with patch("speakeasy.backends.vllm.VLLMBackend.__init__", return_value=None):
            b = get_backend("vllm:meta-llama/Llama-3.3-70B-Instruct")
            from speakeasy.backends.vllm import VLLMBackend

            assert isinstance(b, VLLMBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend source"):
            get_backend("nonexistent:model")

    def test_model_id_parsing(self):
        """Verify the model_id is extracted correctly from 'source:model'."""
        with patch("speakeasy.backends.openai.OpenAIBackend.__init__", return_value=None) as mock:
            get_backend("openai:gpt-4o-mini")
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# get_framework
# ---------------------------------------------------------------------------


class TestGetFramework:
    def _mock_model(self):
        return MagicMock()

    def test_baseline_dr(self):
        with patch("speakeasy.frameworks.direct_request.DirectRequest.__init__", return_value=None):
            fw = get_framework("baseline_dr", self._mock_model())
            from speakeasy.frameworks.direct_request import DirectRequest

            assert isinstance(fw, DirectRequest)

    def test_baseline_gcg(self):
        with patch("speakeasy.frameworks.gcg.GCG.__init__", return_value=None):
            fw = get_framework("baseline_gcg", self._mock_model())
            from speakeasy.frameworks.gcg import GCG

            assert isinstance(fw, GCG)

    def test_baseline_tap(self):
        with patch("speakeasy.frameworks.tap.TAP.__init__", return_value=None):
            fw = get_framework("baseline_tap", self._mock_model())
            from speakeasy.frameworks.tap import TAP

            assert isinstance(fw, TAP)

    def test_speakeasy_dr(self):
        with patch("speakeasy.frameworks.speakeasy.SpeakEasyPipeline.__init__", return_value=None):
            fw = get_framework("speakeasy_dr", self._mock_model())
            from speakeasy.frameworks.speakeasy import SpeakEasyPipeline

            assert isinstance(fw, SpeakEasyPipeline)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            get_framework("nonexistent", self._mock_model())
