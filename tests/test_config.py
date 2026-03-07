"""Tests for speakeasy.config module."""

from speakeasy.config import (
    GCGSuffixes,
    GenerationConfig,
    HarmScoreEvalConfig,
    OpenAIConfig,
    SpeakEasyConfig,
    SpeakEasySettings,
    TAPConfig,
)


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.0
        assert cfg.top_p == 0.0
        assert cfg.max_tokens == 256


class TestOpenAIConfig:
    def test_inherits_generation(self):
        cfg = OpenAIConfig()
        assert cfg.num_processes == 10
        assert cfg.max_tokens == 256


class TestGCGSuffixes:
    def test_has_all_benchmarks(self):
        s = GCGSuffixes()
        assert s.harmbench
        assert s.advbench
        assert s.sorrybench
        assert s.medharm


class TestSpeakEasyConfig:
    def test_defaults(self):
        cfg = SpeakEasyConfig()
        assert cfg.num_subqueries == 3
        assert len(cfg.icl_examples) == 3

    def test_prompt_has_placeholders(self):
        cfg = SpeakEasyConfig()
        assert "{}" in cfg.subquery_prompt


class TestHarmScoreEvalConfig:
    def test_model_names(self):
        cfg = HarmScoreEvalConfig()
        assert "narutatsuri" in cfg.actionable_model
        assert "narutatsuri" in cfg.informative_model


class TestTAPConfig:
    def test_defaults(self):
        cfg = TAPConfig()
        assert cfg.depth == 10
        assert cfg.width == 10
        assert cfg.cutoff_score == 10


class TestSpeakEasySettings:
    def test_env_prefix(self):
        assert SpeakEasySettings.model_config["env_prefix"] == "SPEAKEASY_"
