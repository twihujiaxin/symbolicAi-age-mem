"""Strict reward profiles for the HotpotQA memory workflow.

M8 keeps reward design and credit assignment as separate experiment variables.
This module therefore contains no model, trainer, or network dependency: it
only validates a versioned reward-profile declaration and implements the
deterministic HotpotQA answer metrics used by the E1 terminal-only baseline.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping


REWARD_PROFILE_SCHEMA_VERSION = "agemem.reward_profile.v1"


class RewardProfileConfigError(ValueError):
    """Raised when a reward profile or its training contract is ambiguous."""


class RewardProfileName(str, Enum):
    E1_TERMINAL_ONLY = "e1_terminal_only"
    E2_AGEMEM_HEURISTIC = "e2_agemem_heuristic"


class TerminalMetric(str, Enum):
    # ``hotpotqa_official`` is the M8 config-facing name. As in the official
    # evaluator, EM and F1 are both recorded; answer F1 is the scalar reward.
    HOTPOTQA_OFFICIAL = "hotpotqa_official"
    ANSWER_EXACT_MATCH = "answer_exact_match"
    ANSWER_F1 = "answer_f1"


@dataclass(frozen=True)
class RewardProfile:
    """A fully resolved, immutable HotpotQA reward profile."""

    schema_version: str
    name: RewardProfileName
    terminal_metric: TerminalMetric | None = None
    milestone_reward_enabled: bool = False

    @property
    def is_terminal_only(self) -> bool:
        return self.name is RewardProfileName.E1_TERMINAL_ONLY

    @property
    def uses_llm_judge(self) -> bool:
        return self.name is RewardProfileName.E2_AGEMEM_HEURISTIC

    def heuristic_calculator_kwargs(self) -> Dict[str, float]:
        """Return the frozen legacy AgeMem weights for the E2 compatibility arm."""

        if self.name is not RewardProfileName.E2_AGEMEM_HEURISTIC:
            raise RewardProfileConfigError(
                "heuristic calculator weights are only defined for e2_agemem_heuristic"
            )
        return {
            "task_completion_weight": 0.5,
            "tool_efficiency_weight": 0.2,
            "context_management_weight": 0.15,
            "memory_management_weight": 0.15,
        }


@dataclass(frozen=True)
class HotpotAnswerScore:
    """Official-style deterministic answer metrics for one prediction."""

    exact_match: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True)
class RewardOutcome:
    """Scalar trajectory reward and an auditable component breakdown."""

    total: float
    breakdown: Dict[str, float]


def _strict_keys(
    payload: Mapping[str, object],
    *,
    required: set[str],
    allowed: set[str],
) -> None:
    missing = sorted(required.difference(payload))
    if missing:
        raise RewardProfileConfigError(
            f"reward profile is missing required field(s): {', '.join(missing)}"
        )
    unexpected = sorted(set(payload).difference(allowed))
    if unexpected:
        raise RewardProfileConfigError(
            f"reward profile contains unknown field(s): {', '.join(unexpected)}"
        )


def load_reward_profile(raw_profile: object) -> RewardProfile:
    """Parse a strict, versioned profile declaration from ``workflow_args``.

    There is intentionally no implicit default. A run whose reward arm is not
    named explicitly is not a valid M8 experiment.
    """

    if not isinstance(raw_profile, Mapping):
        raise RewardProfileConfigError(
            "workflow_args.reward_profile must be an explicit mapping"
        )
    payload = dict(raw_profile)
    _strict_keys(
        payload,
        required={"schema_version", "name"},
        allowed={"schema_version", "name", "terminal_metric"},
    )

    schema_version = payload["schema_version"]
    if schema_version != REWARD_PROFILE_SCHEMA_VERSION:
        raise RewardProfileConfigError(
            "unsupported reward profile schema_version "
            f"{schema_version!r}; expected {REWARD_PROFILE_SCHEMA_VERSION!r}"
        )

    try:
        name = RewardProfileName(payload["name"])
    except (TypeError, ValueError) as exc:
        valid = ", ".join(profile.value for profile in RewardProfileName)
        raise RewardProfileConfigError(
            f"unknown reward profile {payload['name']!r}; expected one of: {valid}"
        ) from exc

    if name is RewardProfileName.E1_TERMINAL_ONLY:
        _strict_keys(
            payload,
            required={"schema_version", "name", "terminal_metric"},
            allowed={"schema_version", "name", "terminal_metric"},
        )
        try:
            terminal_metric = TerminalMetric(payload["terminal_metric"])
        except (TypeError, ValueError) as exc:
            valid = ", ".join(metric.value for metric in TerminalMetric)
            raise RewardProfileConfigError(
                f"invalid E1 terminal_metric {payload['terminal_metric']!r}; "
                f"expected one of: {valid}"
            ) from exc
    else:
        _strict_keys(
            payload,
            required={"schema_version", "name"},
            allowed={"schema_version", "name"},
        )
        terminal_metric = None

    return RewardProfile(
        schema_version=REWARD_PROFILE_SCHEMA_VERSION,
        name=name,
        terminal_metric=terminal_metric,
        milestone_reward_enabled=False,
    )


def load_workflow_reward_profile(workflow_args: Mapping[str, object]) -> RewardProfile:
    """Resolve the flat reward keys used by Trinity workflow YAML files.

    Required E1 declaration::

        reward_profile: terminal_only
        terminal_reward_metric: hotpotqa_official
        milestone_reward_enabled: false

    E2 deliberately retains the old heuristic implementation, but still has
    to name that arm explicitly. Unknown names, implicit metrics and enabled
    milestone rewards fail closed rather than silently changing an ablation.
    """

    if not isinstance(workflow_args, Mapping):
        raise RewardProfileConfigError("workflow_args must be a mapping")
    required = {"reward_profile", "milestone_reward_enabled"}
    missing = sorted(key for key in required if key not in workflow_args)
    if missing:
        raise RewardProfileConfigError(
            "workflow_args is missing explicit reward field(s): " + ", ".join(missing)
        )

    milestone_enabled = workflow_args["milestone_reward_enabled"]
    if not isinstance(milestone_enabled, bool):
        raise RewardProfileConfigError("milestone_reward_enabled must be a boolean")
    if milestone_enabled:
        raise RewardProfileConfigError(
            "M8a E1/E2 profiles require milestone_reward_enabled=false"
        )

    raw_name = workflow_args["reward_profile"]
    aliases = {
        "terminal_only": RewardProfileName.E1_TERMINAL_ONLY,
        RewardProfileName.E1_TERMINAL_ONLY.value: RewardProfileName.E1_TERMINAL_ONLY,
        "agemem_heuristic": RewardProfileName.E2_AGEMEM_HEURISTIC,
        RewardProfileName.E2_AGEMEM_HEURISTIC.value: RewardProfileName.E2_AGEMEM_HEURISTIC,
    }
    if not isinstance(raw_name, str) or raw_name not in aliases:
        raise RewardProfileConfigError(
            "reward_profile must be one of: terminal_only, agemem_heuristic"
        )
    name = aliases[raw_name]

    raw_metric = workflow_args.get("terminal_reward_metric")
    if name is RewardProfileName.E1_TERMINAL_ONLY:
        if raw_metric is None:
            raise RewardProfileConfigError(
                "terminal_only requires terminal_reward_metric"
            )
        try:
            terminal_metric = TerminalMetric(raw_metric)
        except (TypeError, ValueError) as exc:
            valid = ", ".join(metric.value for metric in TerminalMetric)
            raise RewardProfileConfigError(
                f"invalid terminal_reward_metric {raw_metric!r}; expected one of: {valid}"
            ) from exc
    else:
        if raw_metric is not None:
            raise RewardProfileConfigError(
                "agemem_heuristic must not define terminal_reward_metric"
            )
        terminal_metric = None

    return RewardProfile(
        schema_version=REWARD_PROFILE_SCHEMA_VERSION,
        name=name,
        terminal_metric=terminal_metric,
        milestone_reward_enabled=False,
    )


def normalize_hotpot_answer(text: str) -> str:
    """Apply HotpotQA/SQuAD lowercase, punctuation and article normalization."""

    if not isinstance(text, str):
        raise TypeError("HotpotQA answers must be strings")
    lowered = text.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def score_hotpot_answer(predicted: str, expected: str) -> HotpotAnswerScore:
    """Compute deterministic HotpotQA answer EM, F1, precision and recall."""

    predicted_normalized = normalize_hotpot_answer(predicted)
    expected_normalized = normalize_hotpot_answer(expected)
    exact_match = float(predicted_normalized == expected_normalized)

    special_answers = {"yes", "no", "noanswer"}
    if (
        predicted_normalized in special_answers
        or expected_normalized in special_answers
    ) and predicted_normalized != expected_normalized:
        return HotpotAnswerScore(
            exact_match=exact_match,
            f1=0.0,
            precision=0.0,
            recall=0.0,
        )

    predicted_tokens = predicted_normalized.split()
    expected_tokens = expected_normalized.split()
    if not predicted_tokens or not expected_tokens:
        equal = float(predicted_tokens == expected_tokens)
        return HotpotAnswerScore(
            exact_match=exact_match,
            f1=equal,
            precision=equal,
            recall=equal,
        )

    overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return HotpotAnswerScore(
            exact_match=exact_match,
            f1=0.0,
            precision=0.0,
            recall=0.0,
        )
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return HotpotAnswerScore(
        exact_match=exact_match,
        f1=f1,
        precision=precision,
        recall=recall,
    )


def terminal_task_score(
    profile: RewardProfile,
    answer_score: HotpotAnswerScore,
) -> float:
    """Select the configured E1 terminal answer metric."""

    if not profile.is_terminal_only or profile.terminal_metric is None:
        raise RewardProfileConfigError(
            "terminal_task_score requires the e1_terminal_only profile"
        )
    if profile.terminal_metric is TerminalMetric.ANSWER_EXACT_MATCH:
        return answer_score.exact_match
    return answer_score.f1


def calculate_terminal_reward(
    profile: RewardProfile,
    *,
    task_score: float,
    found_answer: bool,
) -> RewardOutcome:
    """Return a true terminal-only E1 trajectory reward.

    No tool, memory, context, length, formatting or timeout component is added.
    A missing final answer is fail-closed to zero.
    """

    if not profile.is_terminal_only or profile.terminal_metric is None:
        raise RewardProfileConfigError(
            "calculate_terminal_reward requires the e1_terminal_only profile"
        )
    bounded_score = max(0.0, min(1.0, float(task_score)))
    total = bounded_score if found_answer else 0.0
    component = f"terminal_{profile.terminal_metric.value}"
    return RewardOutcome(
        total=total,
        breakdown={component: total, "total": total},
    )


def validate_e1_trajectory_credit_contract(
    profile: RewardProfile,
    *,
    algorithm_type: str,
    advantage_fn: str,
    repeat_times: int,
    require_group: bool = True,
) -> None:
    """Fail closed if an E1 config is not trajectory-level multi-step GRPO.

    Frozen bench diagnosis may use K=1 at T=0. GRPO training still requires a
    group of at least two rollouts.
    """

    if not profile.is_terminal_only:
        return
    if algorithm_type != "multi_step_grpo":
        raise RewardProfileConfigError("E1 requires algorithm_type='multi_step_grpo'")
    if advantage_fn != "step_wise_grpo":
        raise RewardProfileConfigError("E1 requires advantage_fn='step_wise_grpo'")
    minimum_repeat = 2 if require_group else 1
    if (
        isinstance(repeat_times, bool)
        or not isinstance(repeat_times, int)
        or repeat_times < minimum_repeat
    ):
        raise RewardProfileConfigError("E1 GRPO repeat_times must be at least 2")


__all__ = [
    "HotpotAnswerScore",
    "REWARD_PROFILE_SCHEMA_VERSION",
    "RewardOutcome",
    "RewardProfile",
    "RewardProfileConfigError",
    "RewardProfileName",
    "TerminalMetric",
    "calculate_terminal_reward",
    "load_reward_profile",
    "load_workflow_reward_profile",
    "normalize_hotpot_answer",
    "score_hotpot_answer",
    "terminal_task_score",
    "validate_e1_trajectory_credit_contract",
]
