"""Unified Speak Easy pipeline that composes multilingual querying with any attack method.

This module consolidates the three Speak Easy variants (DR, GCG, TAP) into a single
class by parameterizing the attack method. The shared 3-step pipeline is:
    1. Generate sub-queries (via LLM + ICL examples)
    2. Translate, query in multiple languages, translate back
    3. Score and combine best responses
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from tqdm import tqdm

from speakeasy.backends.base import Backend
from speakeasy.config import GCGSuffixes, SpeakEasyConfig
from speakeasy.constants import LANGUAGE_LIST
from speakeasy.frameworks.base import Framework
from speakeasy.scoring import get_scorer
from speakeasy.translation import get_translator
from speakeasy.utils import (
    contains_refusal,
    detect_benchmark,
    extract_subqueries,
    reduce_repeated_phrases,
)

logger = logging.getLogger(__name__)


class SpeakEasyPipeline(Framework):
    """Unified Speak Easy attack pipeline.

    Combines sub-query decomposition, multilingual translation, and response
    scoring. Optionally composes with GCG suffixes or TAP adversarial prompts.

    Args:
        model: Backend for LLM inference.
        device: GPU device index for scoring models.
        attack_method: One of ``None`` (direct request), ``"gcg"``, or ``"tap"``.
        config: Optional configuration override.
        gcg_suffixes: Optional GCG suffix override.
    """

    def __init__(
        self,
        model: Backend,
        device: int = 0,
        attack_method: str | None = None,
        config: SpeakEasyConfig | None = None,
        gcg_suffixes: GCGSuffixes | None = None,
    ) -> None:
        super().__init__(model)
        self.config = config or SpeakEasyConfig()
        self.device = device
        self.attack_method = attack_method
        self.gcg_suffixes = gcg_suffixes or GCGSuffixes()
        self.translator = get_translator(self.config.translator)

    def _format_icl_examples(self) -> str:
        """Format in-context learning examples for sub-query generation."""
        parts: list[str] = []
        for key, examples in self.config.icl_examples.items():
            parts.append(f"{key}:\n" + "\n".join(f" {ex}" for ex in examples))
        return "\n".join(parts)

    def _generate_subqueries(
        self,
        data: list[dict[str, Any]],
        save_dir: str,
    ) -> dict[str, list[str]]:
        """Step 1: Generate sub-queries for each input query."""
        save_path = os.path.join(save_dir, "subqueries.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                return json.load(f)

        icl = self._format_icl_examples()
        prompts = [
            self.config.subquery_prompt.format(self.config.num_subqueries, inst["query"], icl)
            for inst in data
        ]
        outputs = self.model.infer_batch(prompts, save_dir=save_path)
        subqueries = {
            inst["query"]: extract_subqueries(out, self.config.num_subqueries)
            for inst, out in zip(data, outputs)
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(subqueries, f, ensure_ascii=False, indent=4)
        return subqueries

    def _apply_attack(
        self,
        subqueries: dict[str, list[str]],
        data: list[dict[str, Any]],
        save_dir: str,
    ) -> dict[str, dict[str, str]]:
        """Apply the attack method (GCG suffix or TAP) to sub-queries.

        Returns a dict mapping query -> {subquery: modified_prompt}.
        """
        if self.attack_method == "tap":
            return self._apply_tap(subqueries, data, save_dir)

        # For DR and GCG, the "prompt" is just the subquery (+ optional suffix)
        suffix = ""
        if self.attack_method == "gcg":
            benchmark = detect_benchmark(save_dir)
            suffix_map = {
                "sorrybench": self.gcg_suffixes.sorrybench,
                "medharm": self.gcg_suffixes.medharm,
                "harmbench": self.gcg_suffixes.harmbench,
                "advbench": self.gcg_suffixes.advbench,
            }
            suffix = " " + suffix_map.get(benchmark, self.gcg_suffixes.advbench)

        return {query: {sq: sq + suffix for sq in sqs} for query, sqs in subqueries.items()}

    def _apply_tap(
        self,
        subqueries: dict[str, list[str]],
        data: list[dict[str, Any]],
        save_dir: str,
    ) -> dict[str, dict[str, str]]:
        """Run TAP on each sub-query to generate adversarial prompts."""
        save_path = os.path.join(save_dir, "tap_test_cases.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                return json.load(f)

        from speakeasy.frameworks.tap import TAP

        tap = TAP(model=self.model)
        result: dict[str, dict[str, str]] = {}

        for query_dict in tqdm(data, desc="TAP sub-queries"):
            query = query_dict["query"]
            target = query_dict.get("target", "")
            result[query] = {}
            for sq in subqueries.get(query, []):
                result[query][sq] = tap.generate_single_test_case(sq, target)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result

    def _translate_and_query(
        self,
        prompts_by_query: dict[str, dict[str, str]],
        save_dir: str,
    ) -> dict[str, dict[str, dict[str, str]]]:
        """Step 2: Translate prompts to multiple languages, query model, translate back."""
        # 2a: Translate prompts
        translated_path = os.path.join(save_dir, "subqueries_translated.json")
        if os.path.exists(translated_path):
            with open(translated_path) as f:
                translated = json.load(f)
        else:
            translated: dict[str, dict[str, dict[str, str]]] = {}
            for query, sq_prompts in prompts_by_query.items():
                translated[query] = {}
                for sq, prompt in sq_prompts.items():
                    translated[query][sq] = {}
                    for lang_code in LANGUAGE_LIST.values():
                        text = f"{prompt} Keep your response within 120 words."
                        if lang_code == "en":
                            translated[query][sq][lang_code] = text
                        else:
                            translated[query][sq][lang_code] = (
                                self.translator.translate_multilingual(
                                    text, source="en", target=lang_code
                                )
                            )
            with open(translated_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=4)

        # 2b: Query model in all languages
        en_path = os.path.join(save_dir, "responses_en.json")
        if os.path.exists(en_path):
            with open(en_path) as f:
                return json.load(f)

        # Flatten for batch inference
        flat_inputs: list[str] = []
        keys: list[tuple[str, str, str]] = []
        for query, sqs in translated.items():
            for sq, langs in sqs.items():
                for lang, text in langs.items():
                    flat_inputs.append(text)
                    keys.append((query, sq, lang))

        ml_path = os.path.join(save_dir, "responses_multilingual.json")
        outputs = self.model.infer_batch(flat_inputs, save_dir=ml_path)

        # Reconstruct nested structure
        responses: dict[str, dict[str, dict[str, str]]] = {}
        for (query, sq, lang), output in zip(keys, outputs):
            responses.setdefault(query, {}).setdefault(sq, {})[lang] = output

        # 2c: Translate responses back to English
        translate_cache_path = os.path.join(save_dir, "responses_translate.json")
        responses_en: dict[str, dict[str, dict[str, str]]] = {}
        for query, sqs in tqdm(responses.items(), desc="Back-translation", ncols=100, leave=False):
            responses_en[query] = {}
            for sq, langs in sqs.items():
                responses_en[query][sq] = {}
                for lang, response in langs.items():
                    if lang == "en" or not response:
                        responses_en[query][sq][lang] = response
                    else:
                        cleaned = reduce_repeated_phrases(response)
                        responses_en[query][sq][lang] = self.translator.translate_to_english(
                            sentence=cleaned,
                            source=lang,
                            target="en",
                            save_dir=translate_cache_path,
                        )

        with open(en_path, "w", encoding="utf-8") as f:
            json.dump(responses_en, f, ensure_ascii=False, indent=4)
        return responses_en

    def _score_and_combine(
        self,
        responses_en: dict[str, dict[str, dict[str, str]]],
        save_dir: str,
    ) -> dict[str, str]:
        """Step 3: Score responses and select the best per sub-query."""
        reward_path = os.path.join(save_dir, "responses_with_reward.json")
        if os.path.exists(reward_path):
            with open(reward_path) as f:
                responses_with_reward = json.load(f)
        else:
            # Optionally filter refusals (for TAP variant)
            if self.attack_method == "tap":
                responses_en = {
                    q: {
                        sq: {
                            lang: resp for lang, resp in resps.items() if not contains_refusal(resp)
                        }
                        for sq, resps in sqs.items()
                        if any(not contains_refusal(r) for r in resps.values())
                    }
                    for q, sqs in responses_en.items()
                    if any(
                        any(not contains_refusal(r) for r in resps.values())
                        for resps in sqs.values()
                    )
                }

            scorer = get_scorer(self.config.resp_select_model, device=self.device)
            responses_with_reward = scorer.get_reward(responses_en)
            with open(reward_path, "w") as f:
                json.dump(responses_with_reward, f, indent=4)

        # Select best response per sub-query and combine
        combined: dict[str, str] = {}
        for query, sqs in responses_with_reward.items():
            parts: list[str] = []
            for sq, lang_responses in sqs.items():
                if not lang_responses:
                    continue
                best = max(lang_responses.items(), key=lambda item: item[1].get("score", 0))
                resp = best[1].get("response", "")
                parts.append(resp if isinstance(resp, str) else "".join(resp))
            combined[query] = "".join(parts)

        with open(os.path.join(save_dir, "combined_responses.json"), "w") as f:
            json.dump(combined, f, indent=4)
        return combined

    def infer(self, data: list[dict[str, Any]], save_dir: str) -> None:
        """Run the full Speak Easy pipeline.

        Steps:
            1. Generate sub-queries from each input query
            2. (Optional) Apply GCG suffix or TAP attack to sub-queries
            3. Translate sub-queries to multiple languages
            4. Query the target model in each language
            5. Translate responses back to English
            6. Score and select the best response per sub-query
            7. Combine into a final response per query
        """
        os.makedirs(save_dir, exist_ok=True)

        logger.info("Step 1: Generate sub-queries")
        subqueries = self._generate_subqueries(data, save_dir)

        logger.info("Step 2: Apply attack method (%s)", self.attack_method or "direct")
        prompts = self._apply_attack(subqueries, data, save_dir)

        logger.info("Step 3: Translate and query in multiple languages")
        responses_en = self._translate_and_query(prompts, save_dir)

        logger.info("Step 4: Score and combine best responses")
        self._score_and_combine(responses_en, save_dir)
