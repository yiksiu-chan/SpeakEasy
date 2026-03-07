"""TAP (Tree of Attacks with Pruning) attack framework.

This module preserves the original TAP algorithm while reorganizing it into
the unified Speak Easy package structure. The TAP utilities (conversers,
judges, language models, system prompts) are kept in this sub-package.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

import numpy as np
from tqdm import tqdm

from speakeasy.backends.base import Backend
from speakeasy.config import TAPConfig
from speakeasy.frameworks.base import Framework
from speakeasy.frameworks.tap.common import get_init_msg, process_target_response
from speakeasy.frameworks.tap.conversers import load_TAP_models
from speakeasy.frameworks.tap.system_prompts import get_attacker_system_prompt

logger = logging.getLogger(__name__)


class TAP(Framework):
    """TAP (Tree of Attacks with Pruning) attack framework.

    Implements the iterative tree-of-thoughts jailbreak approach from
    `Mehrotra et al. (2024) <https://arxiv.org/abs/2312.02119>`_.

    Args:
        model: Backend for LLM inference (used for final response generation).
        config: Optional TAP configuration override.
    """

    def __init__(self, model: Backend, config: TAPConfig | None = None) -> None:
        super().__init__(model)
        self.config = config or TAPConfig()

        self.attackLM, self.targetLM, self.judgeLM = load_TAP_models(
            attackLM_args=self.config.attack_model,
            targetLM_args=self.config.target_model,
            judgeLM_args=self.config.judge_model,
            attack_max_n_tokens=self.config.attack_max_n_tokens,
            target_max_n_tokens=self.config.target_max_n_tokens,
            judge_max_n_tokens=self.config.judge_max_n_tokens,
            max_retries=self.config.max_retries,
            attack_temp=self.config.attack_temp,
        )

    def generate_single_test_case(self, query: str, target: str) -> str:
        """Generate a single adversarial test case for the given query.

        Args:
            query: The jailbreak objective.
            target: Desired beginning of the target model's response.

        Returns:
            The best adversarial prompt found by TAP.
        """
        system_prompt = get_attacker_system_prompt(query, "", target)
        cfg = self.config

        init_msg = get_init_msg(query, target, "")
        processed_response_list = [init_msg for _ in range(cfg.n_streams)]
        convs_list = [copy.deepcopy(self.attackLM.template) for _ in range(cfg.n_streams)]
        for conv in convs_list:
            conv.set_system_message(system_prompt)

        best_adv_prompt = query
        highest_score = 0

        for step in range(cfg.depth):
            # Generate adversarial prompts
            extracted_attack_list: list[Any] = []
            convs_list_new: list[Any] = []

            for _ in range(cfg.branching_factor):
                convs_copy = copy.deepcopy(convs_list)
                extracted_attack_list.extend(
                    self.attackLM.get_attack(convs_copy, processed_response_list)
                )
                convs_list_new.extend(convs_copy)

            convs_list = copy.deepcopy(convs_list_new)
            extracted_attack_list, convs_list = _clean_attacks(extracted_attack_list, convs_list)

            adv_prompt_list = [a["prompt"] for a in extracted_attack_list]
            improv_list = [a["improvement"] for a in extracted_attack_list]

            # Prune phase 1: on-topic filtering
            on_topic_scores = self.judgeLM.on_topic_score(adv_prompt_list, query, target, "")
            (
                on_topic_scores,
                _,
                adv_prompt_list,
                improv_list,
                convs_list,
                _,
                extracted_attack_list,
            ) = _prune(
                on_topic_scores,
                None,
                adv_prompt_list,
                improv_list,
                convs_list,
                None,
                extracted_attack_list,
                sorting_score=on_topic_scores,
                width=cfg.width,
            )

            # Get target responses
            target_response_list = self.targetLM.get_response(adv_prompt_list, "")

            # Prune phase 2: judge score filtering
            judge_scores = self.judgeLM.score(
                adv_prompt_list, target_response_list, query, target, ""
            )
            (
                on_topic_scores,
                judge_scores,
                adv_prompt_list,
                improv_list,
                convs_list,
                target_response_list,
                extracted_attack_list,
            ) = _prune(
                on_topic_scores,
                judge_scores,
                adv_prompt_list,
                improv_list,
                convs_list,
                target_response_list,
                extracted_attack_list,
                sorting_score=judge_scores,
                width=cfg.width,
            )

            # Check for jailbreak
            jailbroken = False
            for conv, adv_prompt, score, judge_score, improv, target_resp in zip(
                convs_list,
                adv_prompt_list,
                on_topic_scores,
                judge_scores,
                improv_list,
                target_response_list,
            ):
                if judge_score >= cfg.cutoff_score:
                    jailbroken = True
                if judge_score > highest_score:
                    highest_score = judge_score
                    best_adv_prompt = adv_prompt
                conv.messages = conv.messages[-2 * cfg.keep_last_n :]

            if jailbroken:
                break

            processed_response_list = [
                process_target_response(resp, score, query, target, "")
                for resp, score in zip(target_response_list, judge_scores)
            ]

        return best_adv_prompt

    def get_all_test_cases(
        self,
        data: list[dict[str, Any]],
        test_case_path: str,
    ) -> list[str]:
        """Generate adversarial test cases for all queries."""
        if os.path.exists(test_case_path):
            with open(test_case_path) as f:
                return json.load(f)

        test_cases = []
        for query_dict in tqdm(data, desc="Generating TAP test cases"):
            test_case = self.generate_single_test_case(
                query_dict["query"], query_dict.get("target", "")
            )
            test_cases.append(test_case)

        with open(test_case_path, "w") as f:
            json.dump(test_cases, f, indent=4)
        return test_cases

    def infer(self, data: list[dict[str, Any]], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "combined_responses.json")

        prompts = self.get_all_test_cases(
            data, test_case_path=os.path.join(save_dir, "tap_attack_prompts.json")
        )

        if os.path.exists(save_path):
            return

        outputs = self.model.infer_batch(prompts, save_dir=save_path)
        qa_pairs = {}
        for query_dict, response in zip(data, outputs):
            qa_pairs[query_dict["query"]] = response if response else ""

        with open(save_path, "w") as f:
            json.dump(qa_pairs, f, indent=4)


# ---------------------------------------------------------------------------
# TAP helper functions
# ---------------------------------------------------------------------------


def _prune(
    on_topic_scores: list[Any] | None,
    judge_scores: list[Any] | None,
    adv_prompt_list: list[Any],
    improv_list: list[Any],
    convs_list: list[Any],
    target_response_list: list[Any] | None,
    extracted_attack_list: list[Any],
    sorting_score: list[Any],
    width: int,
) -> tuple:
    """Prune attacks based on sorting scores, keeping at most ``width`` items."""
    shuffled = list(enumerate(sorting_score))
    shuffled = [(s, i) for i, s in shuffled]
    np.random.shuffle(shuffled)
    shuffled.sort(reverse=True)

    def top_k(lst: list[Any] | None) -> list[Any] | None:
        if lst is None:
            return None
        if not shuffled or not lst:
            return lst[:1] if lst else []
        w = min(width, len(lst))
        result = [lst[shuffled[i][1]] for i in range(w) if shuffled[i][0] > 0]
        if not result:
            result = [lst[shuffled[0][1]]]
        return result

    return (
        top_k(on_topic_scores),
        top_k(judge_scores),
        top_k(adv_prompt_list),
        top_k(improv_list),
        top_k(convs_list),
        top_k(target_response_list),
        top_k(extracted_attack_list),
    )


def _clean_attacks(
    attack_list: list[Any],
    convs_list: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Remove failed attacks (None entries) and their corresponding conversations."""
    pairs = [(a, c) for a, c in zip(attack_list, convs_list) if a is not None]
    if not pairs:
        return [], []
    attacks, convs = zip(*pairs)
    return list(attacks), list(convs)
