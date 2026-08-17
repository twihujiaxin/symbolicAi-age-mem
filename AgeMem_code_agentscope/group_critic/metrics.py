"""Strict, deterministic metrics for the M7 offline Group Critic benchmark."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..memory_extraction.models import canonical_digest


M7_ACCEPTANCE_METRICS_SCHEMA_VERSION = "agemem.m7_acceptance_metrics.v1"
M7_REWARD_ERROR_METRICS_SCHEMA_VERSION = "agemem.m7_reward_error_metrics.v1"
M7_STRATUM_METRICS_SCHEMA_VERSION = "agemem.m7_stratum_metrics.v1"
M7_STABILITY_METRICS_SCHEMA_VERSION = "agemem.m7_stability_metrics.v1"
M7_USAGE_METRICS_SCHEMA_VERSION = "agemem.m7_usage_metrics.v1"


class AcceptanceObservation(BaseModel):
    """One terminal prediction joined to its M5 episode outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    expected_accepted: bool
    predicted_accepted: bool
    question_type: Literal["bridge", "comparison"]
    action_count: int = Field(ge=1)


class M7AcceptanceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_ACCEPTANCE_METRICS_SCHEMA_VERSION] = (
        M7_ACCEPTANCE_METRICS_SCHEMA_VERSION
    )
    count: int = Field(ge=1)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_accept_numerator: int = Field(ge=0)
    false_accept_denominator: int = Field(ge=0)
    false_accept_rate: Optional[float]
    false_reject_numerator: int = Field(ge=0)
    false_reject_denominator: int = Field(ge=0)
    false_reject_rate: Optional[float]

    @model_validator(mode="after")
    def validate_counts(self) -> "M7AcceptanceMetrics":
        if self.count != (
            self.true_positive
            + self.true_negative
            + self.false_accept_numerator
            + self.false_reject_numerator
        ):
            raise ValueError("acceptance confusion counts do not sum to count")
        if self.false_accept_denominator != (
            self.true_negative + self.false_accept_numerator
        ):
            raise ValueError("false_accept_denominator is inconsistent")
        if self.false_reject_denominator != (
            self.true_positive + self.false_reject_numerator
        ):
            raise ValueError("false_reject_denominator is inconsistent")
        expected_fa = (
            None
            if self.false_accept_denominator == 0
            else self.false_accept_numerator / self.false_accept_denominator
        )
        expected_fr = (
            None
            if self.false_reject_denominator == 0
            else self.false_reject_numerator / self.false_reject_denominator
        )
        if (
            self.false_accept_rate != expected_fa
            or self.false_reject_rate != expected_fr
        ):
            raise ValueError("acceptance rates do not match confusion counts")
        return self


class M7RewardErrorMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_REWARD_ERROR_METRICS_SCHEMA_VERSION] = (
        M7_REWARD_ERROR_METRICS_SCHEMA_VERSION
    )
    count: int = Field(ge=1)
    mean_absolute_error: float = Field(ge=0.0)
    root_mean_squared_error: float = Field(ge=0.0)
    bias: float
    max_absolute_error: float = Field(ge=0.0)
    signed_error_total: float
    absolute_error_total: float = Field(ge=0.0)

    @field_validator(
        "mean_absolute_error",
        "root_mean_squared_error",
        "bias",
        "max_absolute_error",
        "signed_error_total",
        "absolute_error_total",
    )
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward error metrics must be finite")
        return value


class M7StratumMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_STRATUM_METRICS_SCHEMA_VERSION] = (
        M7_STRATUM_METRICS_SCHEMA_VERSION
    )
    dimension: Literal["question_type", "action_count"]
    value: str = Field(min_length=1)
    acceptance: M7AcceptanceMetrics


class M7StabilityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_STABILITY_METRICS_SCHEMA_VERSION] = (
        M7_STABILITY_METRICS_SCHEMA_VERSION
    )
    repeat_count: Literal[5] = 5
    permutation_count: Literal[6] = 6
    group_count: int = Field(ge=1)
    repeat_checks: int = Field(ge=5)
    permutation_checks: int = Field(ge=6)
    repeat_digest_agreement_rate: float = Field(ge=0.0, le=1.0)
    permutation_digest_agreement_rate: float = Field(ge=0.0, le=1.0)
    stable: bool

    @model_validator(mode="after")
    def validate_stability(self) -> "M7StabilityMetrics":
        expected = (
            self.repeat_digest_agreement_rate == 1.0
            and self.permutation_digest_agreement_rate == 1.0
        )
        if self.stable != expected:
            raise ValueError("stable flag does not match agreement rates")
        return self


class M7CriticUsageMetrics(BaseModel):
    """Provider-free usage; heuristic tokens are explicitly not billed tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M7_USAGE_METRICS_SCHEMA_VERSION] = (
        M7_USAGE_METRICS_SCHEMA_VERSION
    )
    heuristic: Literal["ceil_utf8_bytes_div_4"] = "ceil_utf8_bytes_div_4"
    cold_calls: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    input_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    input_utf8_bytes: int = Field(ge=0)
    output_utf8_bytes: int = Field(ge=0)
    heuristic_input_tokens: int = Field(ge=0)
    heuristic_output_tokens: int = Field(ge=0)
    provider_input_tokens: None = None
    provider_output_tokens: None = None
    provider_cost: None = None
    real_llm_call_count: Literal[0] = 0

    @model_validator(mode="after")
    def validate_estimates(self) -> "M7CriticUsageMetrics":
        if self.heuristic_input_tokens != (self.input_utf8_bytes + 3) // 4:
            raise ValueError("heuristic input tokens do not match declared method")
        if self.heuristic_output_tokens != (self.output_utf8_bytes + 3) // 4:
            raise ValueError("heuristic output tokens do not match declared method")
        return self


def score_acceptance(rows: Iterable[AcceptanceObservation]) -> M7AcceptanceMetrics:
    items = tuple(rows)
    if not items:
        raise ValueError("at least one acceptance observation is required")
    if len({item.rollout_id for item in items}) != len(items):
        raise ValueError("acceptance observations must have unique rollout_id")
    tp = sum(item.expected_accepted and item.predicted_accepted for item in items)
    tn = sum(
        not item.expected_accepted and not item.predicted_accepted for item in items
    )
    fa = sum(not item.expected_accepted and item.predicted_accepted for item in items)
    fr = sum(item.expected_accepted and not item.predicted_accepted for item in items)
    fa_denom, fr_denom = tn + fa, tp + fr
    return M7AcceptanceMetrics(
        count=len(items),
        true_positive=tp,
        true_negative=tn,
        false_accept_numerator=fa,
        false_accept_denominator=fa_denom,
        false_accept_rate=None if fa_denom == 0 else fa / fa_denom,
        false_reject_numerator=fr,
        false_reject_denominator=fr_denom,
        false_reject_rate=None if fr_denom == 0 else fr / fr_denom,
    )


def score_reward_error(
    expected_by_action: Mapping[str, float], predicted_by_action: Mapping[str, float]
) -> M7RewardErrorMetrics:
    if not expected_by_action or set(expected_by_action) != set(predicted_by_action):
        raise ValueError("reward maps must have the same non-empty action_id set")
    errors = tuple(
        predicted_by_action[key] - expected_by_action[key]
        for key in sorted(expected_by_action)
    )
    count = len(errors)
    absolute = tuple(abs(value) for value in errors)
    return M7RewardErrorMetrics(
        count=count,
        mean_absolute_error=sum(absolute) / count,
        root_mean_squared_error=math.sqrt(
            sum(value * value for value in errors) / count
        ),
        bias=sum(errors) / count,
        max_absolute_error=max(absolute),
        signed_error_total=sum(errors),
        absolute_error_total=sum(absolute),
    )


def score_strata(rows: Iterable[AcceptanceObservation]) -> Tuple[M7StratumMetrics, ...]:
    items = tuple(rows)
    if not items:
        raise ValueError("at least one acceptance observation is required")
    grouped: Dict[Tuple[str, str], list[AcceptanceObservation]] = defaultdict(list)
    for item in items:
        grouped[("question_type", item.question_type)].append(item)
        grouped[("action_count", str(item.action_count))].append(item)
    order = {"question_type": 0, "action_count": 1}
    return tuple(
        M7StratumMetrics(
            dimension=dimension,  # type: ignore[arg-type]
            value=value,
            acceptance=score_acceptance(group),
        )
        for (dimension, value), group in sorted(
            grouped.items(), key=lambda item: (order[item[0][0]], item[0][1])
        )
    )


def usage_from_texts(
    *,
    cold_calls: int,
    cache_hits: int,
    cache_misses: int,
    inputs: Iterable[str],
    outputs: Iterable[str],
) -> M7CriticUsageMetrics:
    input_text = "".join(inputs)
    output_text = "".join(outputs)
    input_bytes = len(input_text.encode("utf-8"))
    output_bytes = len(output_text.encode("utf-8"))
    return M7CriticUsageMetrics(
        cold_calls=cold_calls,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        input_characters=len(input_text),
        output_characters=len(output_text),
        input_utf8_bytes=input_bytes,
        output_utf8_bytes=output_bytes,
        heuristic_input_tokens=(input_bytes + 3) // 4,
        heuristic_output_tokens=(output_bytes + 3) // 4,
    )


def stability_digest(value: object) -> str:
    return canonical_digest(value)


__all__ = [
    "AcceptanceObservation",
    "M7AcceptanceMetrics",
    "M7CriticUsageMetrics",
    "M7RewardErrorMetrics",
    "M7StabilityMetrics",
    "M7StratumMetrics",
    "score_acceptance",
    "score_reward_error",
    "score_strata",
    "stability_digest",
    "usage_from_texts",
]
