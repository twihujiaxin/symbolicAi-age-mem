"""V2_T, V2_END, and V2_LIFE reward aggregation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import DynamicQueryPrivate, REWARD_VERSION, VALID_PROFILES
from .state_evaluator import DirectEvaluation


class DynamicRewardError(ValueError):
    pass


def canonical_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    value = match.group(1) if match else text
    return " ".join(value.casefold().strip().split())


def exact_match(answer_text: str, answers: Sequence[str]) -> float:
    value = canonical_answer(answer_text)
    return float(any(value == canonical_answer(answer) for answer in answers))


@dataclass(frozen=True)
class DynamicReward:
    profile: str
    reward_version: str
    task_reward: float
    end_score: float
    retention_score: float
    life_score: float
    semantic_component: float
    total_reward: float
    eligible_denominators: tuple[tuple[str, int], ...]
    no_retention_opportunity: tuple[str, ...]
    per_query_task: tuple[tuple[str, float], ...]
    per_query_end_memory: tuple[tuple[str, float], ...]
    per_query_exposure: tuple[tuple[str, float], ...]


def aggregate_dynamic_reward(
    *,
    profile: str,
    queries: Sequence[DynamicQueryPrivate],
    answer_text_by_query: Mapping[str, str],
    exposure_utility_by_query: Mapping[str, float],
    state_evaluation: DirectEvaluation,
    lambda_semantic: float = 0.25,
) -> DynamicReward:
    if profile not in VALID_PROFILES:
        raise DynamicRewardError(f"unknown profile: {profile}")
    if not queries:
        raise DynamicRewardError("at least one query is required")
    if not 0.0 <= lambda_semantic <= 1.0:
        raise DynamicRewardError("lambda_semantic must be in [0,1]")
    task = tuple(
        (
            query.query_id,
            exact_match(answer_text_by_query.get(query.query_id, ""), query.answers),
        )
        for query in queries
    )
    task_reward = sum(value for _, value in task) / len(task)
    utility_rows: dict[str, list[tuple[int, float]]] = {
        query.query_id: [] for query in queries
    }
    for query_id, checkpoint_index, utility in state_evaluation.checkpoint_utilities:
        utility_rows[query_id].append((checkpoint_index, utility))
    end_memory = []
    exposure = []
    retention_values = []
    denominators = []
    no_opportunity = []
    for query in queries:
        rows = sorted(utility_rows[query.query_id])
        if not rows:
            raise DynamicRewardError(f"query has no checkpoint state: {query.query_id}")
        end_memory.append((query.query_id, rows[-1][1]))
        exposed = float(exposure_utility_by_query.get(query.query_id, 0.0))
        if not 0.0 <= exposed <= 1.0:
            raise DynamicRewardError("exposure utility outside [0,1]")
        exposure.append((query.query_id, exposed))
        eligible = [
            value for index, value in rows if index >= query.answer_available_at
        ]
        denominators.append((query.query_id, len(eligible)))
        if not eligible:
            no_opportunity.append(query.query_id)
            retention_values.append(math.nan)
        else:
            retention_values.append(sum(eligible) / len(eligible))
    if no_opportunity:
        raise DynamicRewardError("no_retention_opportunity:" + ",".join(no_opportunity))
    end_score = sum(
        0.5 * memory_value + 0.5 * exposure_value
        for (_, memory_value), (_, exposure_value) in zip(end_memory, exposure)
    ) / len(queries)
    retention = sum(retention_values) / len(retention_values)
    life = 0.5 * end_score + 0.5 * retention
    semantic = (
        0.0
        if profile == "V2_T"
        else (life if profile in {"V2_LIFE", "V2_LIFE_MONITOR"} else end_score)
    )
    total = task_reward + lambda_semantic * semantic
    return DynamicReward(
        profile=profile,
        reward_version=REWARD_VERSION,
        task_reward=task_reward,
        end_score=end_score,
        retention_score=retention,
        life_score=life,
        semantic_component=semantic,
        total_reward=total,
        eligible_denominators=tuple(denominators),
        no_retention_opportunity=tuple(no_opportunity),
        per_query_task=task,
        per_query_end_memory=tuple(end_memory),
        per_query_exposure=tuple(exposure),
    )


__all__ = [
    "DynamicReward",
    "DynamicRewardError",
    "aggregate_dynamic_reward",
    "canonical_answer",
    "exact_match",
]
