"""Tests for speakeasy.utils module."""

import json
import os
import tempfile

import pytest

from speakeasy.utils import (
    check_format,
    contains_refusal,
    detect_benchmark,
    extract_subqueries,
    geometric_mean,
    partition,
    reduce_repeated_phrases,
)


class TestPartition:
    def test_even_split(self):
        result = partition([1, 2, 3, 4, 5, 6], 3)
        assert len(result) == 3
        assert sum(len(chunk) for chunk in result) == 6

    def test_uneven_split(self):
        result = partition([1, 2, 3, 4, 5], 2)
        assert len(result) == 2
        assert sum(len(chunk) for chunk in result) == 5

    def test_single_partition(self):
        result = partition([1, 2, 3], 1)
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = partition([], 1)
        assert result == []


class TestCheckFormat:
    def test_valid_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"query": "test", "response": "answer"}], f)
            f.flush()
            assert check_format(f.name) is True
        os.unlink(f.name)

    def test_missing_keys(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"query": "test"}], f)
            f.flush()
            assert check_format(f.name) is False
        os.unlink(f.name)

    def test_nonexistent_file(self):
        assert check_format("/nonexistent/path.json") is False

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            assert check_format(f.name) is False
        os.unlink(f.name)


class TestContainsRefusal:
    def test_refusal_detected(self):
        assert contains_refusal("I'm sorry, I cannot help with that.") is True

    def test_no_refusal(self):
        assert contains_refusal("Here is the information you requested.") is False

    def test_case_insensitive(self):
        assert contains_refusal("SORRY about that") is True

    def test_empty_string(self):
        assert contains_refusal("") is False


class TestExtractSubqueries:
    def test_basic_extraction(self):
        text = "1. What are colors?\n2. What is a rainbow?"
        result = extract_subqueries(text, 2)
        assert len(result) == 2
        assert "colors?" in result[0]

    def test_respects_limit(self):
        text = "1. Q1?\n2. Q2?\n3. Q3?"
        result = extract_subqueries(text, 2)
        assert len(result) == 2

    def test_invalid_num(self):
        with pytest.raises(ValueError):
            extract_subqueries("text", 0)

    def test_no_matches(self):
        result = extract_subqueries("no questions here", 3)
        assert result == []


class TestReduceRepeatedPhrases:
    def test_reduces_word_repeats(self):
        text = "hello hello hello hello hello hello"
        result = reduce_repeated_phrases(text)
        assert result.count("hello") < 6

    def test_no_change_for_normal_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert reduce_repeated_phrases(text) == text


class TestGeometricMean:
    def test_basic(self):
        assert geometric_mean(4, 9) == pytest.approx(6.0)

    def test_zero(self):
        assert geometric_mean(0, 5) == pytest.approx(0.0)


class TestDetectBenchmark:
    def test_sorrybench(self):
        assert detect_benchmark("results/sorry-bench/dr") == "sorrybench"

    def test_medharm(self):
        assert detect_benchmark("results/med-safety/gcg") == "medharm"

    def test_harmbench(self):
        assert detect_benchmark("results/harmbench/tap") == "harmbench"

    def test_advbench(self):
        assert detect_benchmark("results/advbench/dr") == "advbench"

    def test_unknown(self):
        assert detect_benchmark("results/other/dr") == "unknown"
