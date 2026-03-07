"""TAP attack/target model wrappers and model loading utilities."""

from __future__ import annotations

import copy
import logging
from typing import Any

import ray

from speakeasy.frameworks.tap.common import extract_json
from speakeasy.frameworks.tap.judges import GPTJudge, OpenSourceJudge
from speakeasy.frameworks.tap.language_models import GPT, VLLM, Gemini, HuggingFace
from speakeasy.frameworks.tap.model_utils import get_template

logger = logging.getLogger(__name__)


def load_TAP_models(
    attackLM_args: dict[str, Any],
    targetLM_args: dict[str, Any],
    judgeLM_args: dict[str, Any],
    attack_max_n_tokens: int,
    target_max_n_tokens: int,
    judge_max_n_tokens: int,
    max_retries: int,
    attack_temp: float,
) -> tuple[AttackLM, TargetLM, Any]:
    """Load attack, target, and judge models for TAP.

    Reuses model instances when the same model is used for multiple roles.
    """
    ray.init(ignore_reinit_error=True)

    a_name = attackLM_args.get("model_name_or_path", "")
    t_name = targetLM_args.get("model_name_or_path", "")
    j_name = judgeLM_args.get("model_name_or_path", "")

    # Default target to attack model if not specified.
    if not t_name:
        t_name = a_name
        targetLM_args = {**attackLM_args, **targetLM_args}
        targetLM_args["model_name_or_path"] = t_name

    attackLM = AttackLM(
        max_n_tokens=attack_max_n_tokens,
        max_retries=max_retries,
        temperature=attack_temp,
        **attackLM_args,
    )

    preloaded_target = attackLM.model if t_name == a_name else None
    if preloaded_target:
        targetLM_args = attackLM_args
    targetLM = TargetLM(
        max_n_tokens=target_max_n_tokens,
        temperature=0.0,
        preloaded_model=preloaded_target,
        **targetLM_args,
    )

    preloaded_judge = None
    if j_name == a_name:
        preloaded_judge = attackLM.model
        judgeLM_args = attackLM_args
    elif j_name == t_name:
        preloaded_judge = targetLM.model
        judgeLM_args = targetLM_args

    judgeLM = _load_judge(
        max_n_tokens=judge_max_n_tokens,
        preloaded_model=preloaded_judge,
        **judgeLM_args,
    )

    return attackLM, targetLM, judgeLM


class AttackLM:
    """Wrapper for the attacker language model in TAP."""

    def __init__(
        self,
        max_n_tokens: int,
        max_retries: int,
        temperature: float,
        **model_kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_kwargs["model_name_or_path"]
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_retries = max_retries
        self._uses_openai_api = (
            bool(model_kwargs.get("base_url")) or "gpt" in self.model_name_or_path.lower()
        )

        self.template = get_template(**model_kwargs, return_fschat_conv=True)
        self.model = _load_indiv_model(**model_kwargs)
        self.use_ray = isinstance(self.model, ray.actor.ActorHandle)

    def get_attack(
        self,
        convs_list: list[Any],
        prompts_list: list[str],
    ) -> list[dict[str, Any] | None]:
        """Generate adversarial attacks for the given conversations."""
        assert len(convs_list) == len(prompts_list)
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs: list[dict[str, Any] | None] = [None] * batchsize

        init_message = (
            '{"improvement": "","prompt": "'
            if len(convs_list[0].messages) == 0
            else '{"improvement": "'
        )

        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self._uses_openai_api:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[: -len(conv.sep2)])

        for _ in range(self.max_retries):
            subset = [full_prompts[i] for i in indices_to_regenerate]
            gen_fn = (
                self.model.batched_generate.remote if self.use_ray else self.model.batched_generate
            )
            outputs_list = gen_fn(
                subset,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                stop_tokens=["}"],
            )
            if self.use_ray:
                outputs_list = ray.get(outputs_list)

            new_indices: list[int] = []
            for i, full_output in enumerate(outputs_list):
                orig_idx = indices_to_regenerate[i]
                if not self._uses_openai_api:
                    full_output = init_message + full_output
                if full_output[-1] != "}":
                    full_output += "}"

                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_idx] = attack_dict
                    convs_list[orig_idx].update_last_message(json_str)
                else:
                    new_indices.append(orig_idx)

            indices_to_regenerate = new_indices
            if not indices_to_regenerate:
                break

        return valid_outputs


class TargetLM:
    """Wrapper for the target language model in TAP."""

    def __init__(
        self,
        max_n_tokens: int,
        temperature: float,
        preloaded_model: Any = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_kwargs.get("model_name_or_path", "")
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self._uses_openai_api = (
            bool(model_kwargs.get("base_url")) or "gpt" in self.model_name_or_path.lower()
        )

        self.template = get_template(**model_kwargs, return_fschat_conv=True)
        self.model = preloaded_model if preloaded_model else _load_indiv_model(**model_kwargs)
        self.use_ray = isinstance(self.model, ray.actor.ActorHandle)

    def get_response(self, prompts_list: list[str], context_str: str = "") -> list[str]:
        """Get target model responses for the given prompts."""
        batchsize = len(prompts_list)
        convs_list = [copy.deepcopy(self.template) for _ in range(batchsize)]
        full_prompts = []

        for conv, prompt in zip(convs_list, prompts_list):
            if context_str:
                prompt = f"{context_str}\n\n---\n\n{prompt}"
            conv.append_message(conv.roles[0], prompt)

            if self._uses_openai_api:
                full_prompts.append(conv.to_openai_api_messages())
            elif "gemini-" in self.model_name_or_path:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        gen_fn = (
            self.model.batched_generate if not self.use_ray else self.model.batched_generate.remote
        )
        outputs = gen_fn(
            full_prompts,
            max_n_tokens=self.max_n_tokens,
            temperature=self.temperature,
            stop_tokens=["}"],
        )
        return outputs if not self.use_ray else ray.get(outputs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_judge(
    model_name_or_path: str,
    max_n_tokens: int,
    preloaded_model: Any = None,
    token: str | None = None,
    base_url: str | None = None,
    **model_kwargs: Any,
) -> GPTJudge | OpenSourceJudge:
    """Load a judge model (OpenAI-compatible API or open-source)."""
    if base_url or "gpt" in model_name_or_path.lower():
        judge_model = GPT(model_name=model_name_or_path, token=token, base_url=base_url)
        return GPTJudge(model_name_or_path, max_n_tokens, judge_model)
    if preloaded_model is None:
        preloaded_model = _load_indiv_model(model_name_or_path=model_name_or_path, **model_kwargs)
    return OpenSourceJudge(model_name_or_path, max_n_tokens, judge_model=preloaded_model)


def _load_indiv_model(
    model_name_or_path: str,
    use_vllm: bool = False,
    token: str | None = None,
    base_url: str | None = None,
    **model_kwargs: Any,
) -> Any:
    """Load an individual model for TAP (OpenAI-compatible, Gemini, HuggingFace, or VLLM).

    To use any model served via an OpenAI-compatible API (including vLLM
    servers), pass ``base_url`` (e.g. ``"http://localhost:8000/v1"``).
    """
    if base_url or "gpt" in model_name_or_path.lower():
        return GPT(model_name_or_path, token=token, base_url=base_url)
    elif "gemini-" in model_name_or_path:
        return Gemini(model_name_or_path, token=token)
    else:
        num_gpus = model_kwargs.get("num_gpus", 1)
        if use_vllm:
            options = {"num_gpus": 1} if num_gpus == 1 else {}
            lm = VLLM.options(**options).remote(model_name_or_path, token=token, **model_kwargs)
        else:
            lm = (
                ray.remote(HuggingFace)
                .options(num_gpus=num_gpus)
                .remote(model_name_or_path, token=token, **model_kwargs)
            )
        ray.get(lm.is_initialized.remote())
        return lm
