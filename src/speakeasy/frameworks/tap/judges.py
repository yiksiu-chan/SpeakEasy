"""Judge models for TAP scoring and on-topic evaluation."""

from __future__ import annotations

import re
from typing import Any

import ray
from fastchat.model import get_conversation_template

from speakeasy.frameworks.tap.system_prompts import (
    get_judge_system_prompt_for_judge,
    get_judge_system_prompt_for_on_topic,
)


class JudgeBase:
    """Base class for TAP judge models."""

    def __init__(self, judge_name: str, max_n_tokens: int) -> None:
        self.max_n_tokens = max_n_tokens
        self.temperature = 0.0
        self.judge_name = judge_name

    def get_judge_prompt(
        self,
        attack_prompt: str,
        target_response: str,
        context_str: str,
    ) -> str:
        prompt = f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        if context_str:
            prompt = f"[CONTEXT]: {context_str}\n\n{prompt}"
        return prompt

    def get_judge_prompt_on_topic(self, attack_prompt: str, context_str: str) -> str:
        prompt = f"[PROMPT]:{attack_prompt}"
        if context_str:
            prompt = f"[CONTEXT]: {context_str}\n\n{prompt}"
        return prompt

    @staticmethod
    def process_output_judge_score(raw_output: str) -> int:
        """Extract a numeric score from ``[[N]]`` format."""
        match = re.search(r"\[\[(\d+)\]\]?", raw_output)
        return int(match.group(1)) if match else 1

    @staticmethod
    def process_output_on_topic_score(raw_output: str) -> bool:
        return "yes" in raw_output.lower()


class GPTJudge(JudgeBase):
    """GPT-based judge for TAP scoring."""

    def __init__(self, judge_name: str, max_n_tokens: int, judge_model: Any) -> None:
        super().__init__(judge_name, max_n_tokens)
        self.judge_model = judge_model

    def _create_conv(self, full_prompt: str, system_prompt: str) -> list[dict[str, str]]:
        conv = get_conversation_template(self.judge_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)
        return conv.to_openai_api_messages()

    def score(
        self,
        attack_prompt_list: list[str],
        target_response_list: list[str],
        behavior: str,
        target: str,
        context_str: str = "",
    ) -> list[int]:
        system_prompt = get_judge_system_prompt_for_judge(behavior, context_str)
        convs = [
            self._create_conv(self.get_judge_prompt(p, r, context_str), system_prompt)
            for p, r in zip(attack_prompt_list, target_response_list)
        ]
        raw = self.judge_model.batched_generate(
            convs, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )
        return [self.process_output_judge_score(o) for o in raw]

    def on_topic_score(
        self,
        attack_prompt_list: list[str],
        behavior: str,
        target: str,
        context_str: str = "",
    ) -> list[int]:
        system_prompt = get_judge_system_prompt_for_on_topic(behavior, context_str)
        convs = [
            self._create_conv(self.get_judge_prompt_on_topic(p, context_str), system_prompt)
            for p in attack_prompt_list
        ]
        raw = self.judge_model.batched_generate(
            convs, max_n_tokens=self.max_n_tokens, temperature=self.temperature
        )
        return [self.process_output_judge_score(o) for o in raw]


class OpenSourceJudge(JudgeBase):
    """Open-source (Ray-managed) judge for TAP scoring."""

    def __init__(self, judge_name: str, max_n_tokens: int, judge_model: Any) -> None:
        super().__init__(judge_name, max_n_tokens)
        self.judge_model = judge_model
        self.init_msg = "Rating: [["

    def _create_conv(self, full_prompt: str, system_prompt: str) -> str:
        full_prompt = system_prompt + "\n\n" + full_prompt
        conv = get_conversation_template(self.judge_name)
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], self.init_msg)
        return conv.get_prompt()[: -len(conv.sep2)]

    def score(
        self,
        attack_prompt_list: list[str],
        target_response_list: list[str],
        behavior: str,
        target: str,
        context_str: str = "",
    ) -> list[int]:
        system_prompt = get_judge_system_prompt_for_judge(behavior, context_str)
        convs = [
            self._create_conv(self.get_judge_prompt(p, r, context_str), system_prompt)
            for p, r in zip(attack_prompt_list, target_response_list)
        ]
        raw = ray.get(
            self.judge_model.batched_generate.remote(
                convs,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                stop_tokens=["]"],
            )
        )
        raw = [self.init_msg + o for o in raw]
        return [self.process_output_judge_score(o) for o in raw]

    def on_topic_score(
        self,
        attack_prompt_list: list[str],
        behavior: str,
        target: str,
        context_str: str = "",
    ) -> list[int]:
        system_prompt = get_judge_system_prompt_for_on_topic(behavior, context_str)
        convs = [
            self._create_conv(self.get_judge_prompt_on_topic(p, context_str), system_prompt)
            for p in attack_prompt_list
        ]
        raw = ray.get(
            self.judge_model.batched_generate.remote(
                convs, max_n_tokens=self.max_n_tokens, temperature=self.temperature
            )
        )
        raw = [self.init_msg + o for o in raw]
        return [self.process_output_on_topic_score(o) for o in raw]
