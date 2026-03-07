"""Tests for response scoring modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speakeasy.scoring import get_scorer
from speakeasy.scoring.harmscore import _sigmoid

# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestGetScorer:
    def test_harmscore_import(self):
        with patch("speakeasy.scoring.harmscore.HarmScoreScorer.__init__", return_value=None):
            scorer = get_scorer("harmscore", device=0)
            from speakeasy.scoring.harmscore import HarmScoreScorer

            assert isinstance(scorer, HarmScoreScorer)

    def test_generic_import(self):
        with patch("speakeasy.scoring.generic.GenericScorer.__init__", return_value=None):
            scorer = get_scorer("generic:some-model", device=0)
            from speakeasy.scoring.generic import GenericScorer

            assert isinstance(scorer, GenericScorer)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            get_scorer("nonexistent")


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100) == pytest.approx(1.0)

    def test_large_negative(self):
        assert _sigmoid(-100) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HarmScoreScorer unit tests (mocked models)
# ---------------------------------------------------------------------------


class TestHarmScoreScorer:
    def _make_scorer(self):
        with patch("speakeasy.scoring.harmscore.HarmScoreScorer.__init__", return_value=None):
            from speakeasy.scoring.harmscore import HarmScoreScorer

            scorer = HarmScoreScorer()
            scorer.cfg = MagicMock()
            scorer.cfg.pad_token = "<pad>"
            scorer.cfg.actionable_model = "act-model"
            scorer.cfg.informative_model = "info-model"
            scorer.device = 0

            # Mock tokenizer
            scorer.tokenizer = MagicMock()
            scorer.tokenizer.apply_chat_template.return_value = "<bos>user: q\nassistant: r"
            scorer.tokenizer.bos_token = "<bos>"
            scorer.pipe_kwargs = {
                "return_all_scores": True,
                "top_k": None,
                "function_to_apply": "none",
                "batch_size": 32,
            }
            return scorer

    def test_prepare_text_normal(self):
        scorer = self._make_scorer()
        result = scorer._prepare_text("question", "answer")
        assert result == "user: q\nassistant: r"
        scorer.tokenizer.apply_chat_template.assert_called_once()

    def test_prepare_text_refusal(self):
        scorer = self._make_scorer()
        with patch("speakeasy.scoring.harmscore.contains_refusal", return_value=True):
            assert scorer._prepare_text("q", "I cannot help") is None

    def test_prepare_text_empty_response(self):
        scorer = self._make_scorer()
        assert scorer._prepare_text("q", "") is None

    def test_prepare_text_none_bos_token(self):
        scorer = self._make_scorer()
        scorer.tokenizer.bos_token = None
        scorer.tokenizer.apply_chat_template.return_value = "user: q\nassistant: r"
        result = scorer._prepare_text("q", "r")
        assert result == "user: q\nassistant: r"

    def test_get_reward_structure(self):
        scorer = self._make_scorer()

        # Mock _prepare_text to return a string
        scorer._prepare_text = MagicMock(return_value="formatted text")

        # Mock _load_pipe and _free_pipe
        mock_pipe = MagicMock()
        mock_pipe.return_value = [[{"score": 1.0}]]
        scorer._load_pipe = MagicMock(return_value=mock_pipe)
        scorer._free_pipe = MagicMock()

        data = {"q1": {"sq1": {"en": "response1", "fr": "response2"}}}
        result = scorer.get_reward(data)

        assert "q1" in result
        assert "sq1" in result["q1"]
        assert "en" in result["q1"]["sq1"]
        assert "fr" in result["q1"]["sq1"]
        for lang_result in result["q1"]["sq1"].values():
            assert "response" in lang_result
            assert "actionable_score" in lang_result
            assert "informative_score" in lang_result
            assert "score" in lang_result

    def test_get_reward_refusal_scores_zero(self):
        scorer = self._make_scorer()

        # Return None for all (refusal)
        scorer._prepare_text = MagicMock(return_value=None)

        mock_pipe = MagicMock()
        scorer._load_pipe = MagicMock(return_value=mock_pipe)
        scorer._free_pipe = MagicMock()

        data = {"q1": {"sq1": {"en": "I'm sorry I can't"}}}
        result = scorer.get_reward(data)
        assert result["q1"]["sq1"]["en"]["score"] == 0.0
        assert result["q1"]["sq1"]["en"]["actionable_score"] == 0.0
        assert result["q1"]["sq1"]["en"]["informative_score"] == 0.0

    def test_get_reward_geometric_mean(self):
        scorer = self._make_scorer()
        scorer._prepare_text = MagicMock(return_value="text")

        # actionable pipe returns score=2.0, informative returns score=0.0
        # sigmoid(2.0) ≈ 0.8808, sigmoid(0.0) = 0.5
        mock_act_pipe = MagicMock()
        mock_act_pipe.return_value = [[{"score": 2.0}]]
        mock_info_pipe = MagicMock()
        mock_info_pipe.return_value = [[{"score": 0.0}]]

        call_count = [0]

        def load_pipe_side_effect(model_name):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_act_pipe
            return mock_info_pipe

        scorer._load_pipe = MagicMock(side_effect=load_pipe_side_effect)
        scorer._free_pipe = MagicMock()

        data = {"q": {"sq": {"en": "response"}}}
        result = scorer.get_reward(data)

        act = _sigmoid(2.0)
        info = _sigmoid(0.0)
        expected = float(np.sqrt(act * info))
        assert result["q"]["sq"]["en"]["score"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# GenericScorer unit tests
# ---------------------------------------------------------------------------


class TestGenericScorer:
    def _make_scorer(self):
        with patch("speakeasy.scoring.generic.GenericScorer.__init__", return_value=None):
            from speakeasy.scoring.generic import GenericScorer

            scorer = GenericScorer.__new__(GenericScorer)
            scorer.tokenizer = MagicMock()
            scorer.tokenizer.apply_chat_template.return_value = "<bos>user: q\nassistant: r"
            scorer.tokenizer.bos_token = "<bos>"
            scorer.pipe = MagicMock()
            scorer.pipe.return_value = [[{"score": 0.75}]]
            scorer.pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 32,
            }
            return scorer

    def test_score_single(self):
        scorer = self._make_scorer()
        result = scorer._score_single("question", "answer")
        assert result == 0.75

    def test_score_single_none_bos(self):
        scorer = self._make_scorer()
        scorer.tokenizer.bos_token = None
        scorer.tokenizer.apply_chat_template.return_value = "user: q\nassistant: r"
        result = scorer._score_single("q", "r")
        assert result == 0.75

    def test_get_reward_structure(self):
        scorer = self._make_scorer()
        data = {"q1": {"sq1": {"en": "resp1"}, "sq2": {"fr": "resp2"}}}
        result = scorer.get_reward(data)

        assert "q1" in result
        assert "sq1" in result["q1"]
        assert "sq2" in result["q1"]
        assert result["q1"]["sq1"]["en"]["response"] == "resp1"
        assert result["q1"]["sq1"]["en"]["score"] == 0.75
