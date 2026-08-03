"""Deterministic, exact-set metrics for the M6 extraction benchmark."""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize_exact_text(value: str) -> str:
    """Apply the benchmark's NFKC/casefold/collapsed-space normalization."""

    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("evaluation text must not normalize to an empty string")
    return normalized


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    # Empty prediction and empty gold are conventionally perfect for exact-set
    # scoring.  A non-empty opposite side still drives F1 to zero.
    return 1.0 if denominator == 0 else numerator / denominator


class TripleEvaluationRecord(BaseModel):
    """One predicted or Oracle triple tied to an evidence sentence ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.triple_evaluation_record.v1"] = (
        "agemem.triple_evaluation_record.v1"
    )
    evidence_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    category: str = Field(min_length=1)
    value: str = Field(min_length=1)

    @field_validator("evidence_id", "subject", "category", "value")
    @classmethod
    def fields_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation fields must not be blank")
        return value

    @property
    def key(self) -> Tuple[str, str, str]:
        return (
            normalize_exact_text(self.subject),
            normalize_exact_text(self.category),
            normalize_exact_text(self.value),
        )


class APEvaluationRecord(BaseModel):
    """One atomic proposition tied to the original action identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.ap_evaluation_record.v1"] = (
        "agemem.ap_evaluation_record.v1"
    )
    action_id: str = Field(min_length=1)
    proposition: str = Field(min_length=1)

    @field_validator("action_id", "proposition")
    @classmethod
    def fields_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("AP evaluation fields must not be blank")
        return value

    @property
    def key(self) -> Tuple[str, str]:
        return self.action_id, normalize_exact_text(self.proposition)


class PRFScore(BaseModel):
    """Exact-set true/false counts and their derived P/R/F1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.prf_score.v1"] = "agemem.prf_score.v1"
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    @field_validator("precision", "recall", "f1")
    @classmethod
    def metrics_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_derived_values(self) -> "PRFScore":
        expected = _make_prf_values(
            self.true_positive, self.false_positive, self.false_negative
        )
        actual = (self.precision, self.recall, self.f1)
        if any(not math.isclose(a, b, abs_tol=1e-15) for a, b in zip(actual, expected)):
            raise ValueError("P/R/F1 do not match their confusion counts")
        return self


class MacroPRF(BaseModel):
    """Unweighted mean across evidence sentences or actions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.macro_prf.v1"] = "agemem.macro_prf.v1"
    group_count: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    @field_validator("precision", "recall", "f1")
    @classmethod
    def metrics_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)


class TripleMetrics(BaseModel):
    """Triple exact-match metrics, macro-averaged by evidence sentence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.triple_metrics.v1"] = "agemem.triple_metrics.v1"
    gold_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    micro: PRFScore
    macro: MacroPRF


class APMetrics(BaseModel):
    """AP exact-match metrics, macro-averaged by action ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.ap_metrics.v1"] = "agemem.ap_metrics.v1"
    gold_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    micro: PRFScore
    macro: MacroPRF


def _make_prf_values(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def _make_prf(tp: int, fp: int, fn: int) -> PRFScore:
    precision, recall, f1 = _make_prf_values(tp, fp, fn)
    return PRFScore(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _macro_score(
    gold: Dict[str, set[Tuple[str, ...]]],
    predicted: Dict[str, set[Tuple[str, ...]]],
) -> MacroPRF:
    groups = sorted(set(gold) | set(predicted))
    if not groups:
        return MacroPRF(group_count=0, precision=1.0, recall=1.0, f1=1.0)
    scores = []
    for group in groups:
        gold_set = gold.get(group, set())
        predicted_set = predicted.get(group, set())
        scores.append(
            _make_prf(
                len(gold_set & predicted_set),
                len(predicted_set - gold_set),
                len(gold_set - predicted_set),
            )
        )
    count = len(scores)
    return MacroPRF(
        group_count=count,
        precision=sum(score.precision for score in scores) / count,
        recall=sum(score.recall for score in scores) / count,
        f1=sum(score.f1 for score in scores) / count,
    )


def _group_triples(
    records: Iterable[TripleEvaluationRecord], *, side: str
) -> Dict[str, set[Tuple[str, ...]]]:
    grouped: Dict[str, set[Tuple[str, ...]]] = defaultdict(set)
    for record in records:
        key = record.key
        if key in grouped[record.evidence_id]:
            raise ValueError(
                f"duplicate normalized triple on {side} side for evidence_id "
                f"{record.evidence_id!r}"
            )
        grouped[record.evidence_id].add(key)
    return dict(grouped)


def score_triples(
    gold: Sequence[TripleEvaluationRecord],
    predicted: Sequence[TripleEvaluationRecord],
) -> TripleMetrics:
    """Score normalized exact triples, grouped by evidence sentence."""

    gold_groups = _group_triples(gold, side="gold")
    predicted_groups = _group_triples(predicted, side="predicted")
    gold_pairs = {
        (evidence_id, key) for evidence_id, keys in gold_groups.items() for key in keys
    }
    predicted_pairs = {
        (evidence_id, key)
        for evidence_id, keys in predicted_groups.items()
        for key in keys
    }
    micro = _make_prf(
        len(gold_pairs & predicted_pairs),
        len(predicted_pairs - gold_pairs),
        len(gold_pairs - predicted_pairs),
    )
    return TripleMetrics(
        gold_count=len(gold_pairs),
        predicted_count=len(predicted_pairs),
        micro=micro,
        macro=_macro_score(gold_groups, predicted_groups),
    )


def _group_aps(
    records: Iterable[APEvaluationRecord], *, side: str
) -> Dict[str, set[Tuple[str, ...]]]:
    grouped: Dict[str, set[Tuple[str, ...]]] = defaultdict(set)
    for record in records:
        proposition = (record.key[1],)
        if proposition in grouped[record.action_id]:
            raise ValueError(
                f"duplicate normalized AP on {side} side for action_id "
                f"{record.action_id!r}"
            )
        grouped[record.action_id].add(proposition)
    return dict(grouped)


def score_aps(
    gold: Sequence[APEvaluationRecord],
    predicted: Sequence[APEvaluationRecord],
) -> APMetrics:
    """Score normalized exact APs keyed by ``(action_id, proposition)``."""

    gold_groups = _group_aps(gold, side="gold")
    predicted_groups = _group_aps(predicted, side="predicted")
    gold_pairs = {
        (action_id, key) for action_id, keys in gold_groups.items() for key in keys
    }
    predicted_pairs = {
        (action_id, key) for action_id, keys in predicted_groups.items() for key in keys
    }
    return APMetrics(
        gold_count=len(gold_pairs),
        predicted_count=len(predicted_pairs),
        micro=_make_prf(
            len(gold_pairs & predicted_pairs),
            len(predicted_pairs - gold_pairs),
            len(gold_pairs - predicted_pairs),
        ),
        macro=_macro_score(gold_groups, predicted_groups),
    )


class AcceptanceDecision(BaseModel):
    """Oracle or extracted terminal acceptance for one rollout/action key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.acceptance_decision.v1"] = (
        "agemem.acceptance_decision.v1"
    )
    action_id: str = Field(min_length=1)
    accepted: bool

    @field_validator("action_id")
    @classmethod
    def action_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("action_id must not be blank")
        return value


class AcceptanceMetrics(BaseModel):
    """False-accept/reject rates with explicit zero-denominator semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.acceptance_metrics.v1"] = (
        "agemem.acceptance_metrics.v1"
    )
    action_count: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    false_accept_numerator: int = Field(ge=0)
    false_accept_denominator: int = Field(ge=0)
    false_accept_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    false_reject_numerator: int = Field(ge=0)
    false_reject_denominator: int = Field(ge=0)
    false_reject_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("false_accept_rate", "false_reject_rate")
    @classmethod
    def optional_rates_must_be_finite(
        cls, value: Optional[float], info
    ) -> Optional[float]:
        return None if value is None else _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_confusion(self) -> "AcceptanceMetrics":
        if self.action_count != (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        ):
            raise ValueError("acceptance confusion counts must sum to action_count")
        expected = (
            self.false_positive,
            self.true_negative + self.false_positive,
            self.false_negative,
            self.true_positive + self.false_negative,
        )
        actual = (
            self.false_accept_numerator,
            self.false_accept_denominator,
            self.false_reject_numerator,
            self.false_reject_denominator,
        )
        if actual != expected:
            raise ValueError("false accept/reject counts do not match confusion matrix")
        expected_rates = (
            None
            if self.false_accept_denominator == 0
            else self.false_accept_numerator / self.false_accept_denominator,
            None
            if self.false_reject_denominator == 0
            else self.false_reject_numerator / self.false_reject_denominator,
        )
        for actual_rate, expected_rate in zip(
            (self.false_accept_rate, self.false_reject_rate), expected_rates
        ):
            if actual_rate is None or expected_rate is None:
                if actual_rate is not expected_rate:
                    raise ValueError("zero denominators require a None rate")
            elif not math.isclose(actual_rate, expected_rate, abs_tol=1e-15):
                raise ValueError("false accept/reject rate does not match counts")
        return self


def _decision_map(
    records: Sequence[AcceptanceDecision], *, side: str
) -> Dict[str, bool]:
    result: Dict[str, bool] = {}
    for record in records:
        if record.action_id in result:
            raise ValueError(
                f"duplicate {side} acceptance action_id {record.action_id!r}"
            )
        result[record.action_id] = record.accepted
    return result


def score_acceptance(
    oracle: Sequence[AcceptanceDecision],
    extracted: Sequence[AcceptanceDecision],
) -> AcceptanceMetrics:
    """Compare extracted acceptance against Oracle on an exact action-ID join."""

    oracle_by_id = _decision_map(oracle, side="Oracle")
    extracted_by_id = _decision_map(extracted, side="extracted")
    if set(oracle_by_id) != set(extracted_by_id):
        raise ValueError("Oracle/extracted acceptance action_id sets do not match")
    tp = tn = fp = fn = 0
    for action_id in sorted(oracle_by_id):
        expected = oracle_by_id[action_id]
        actual = extracted_by_id[action_id]
        tp += int(expected and actual)
        tn += int(not expected and not actual)
        fp += int(not expected and actual)
        fn += int(expected and not actual)
    negative = tn + fp
    positive = tp + fn
    return AcceptanceMetrics(
        action_count=len(oracle_by_id),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        false_accept_numerator=fp,
        false_accept_denominator=negative,
        false_accept_rate=None if negative == 0 else fp / negative,
        false_reject_numerator=fn,
        false_reject_denominator=positive,
        false_reject_rate=None if positive == 0 else fn / positive,
    )


class RewardActionValue(BaseModel):
    """The three M6 reward components compared for one original action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.reward_action_value.v1"] = (
        "agemem.reward_action_value.v1"
    )
    action_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    timestep: int = Field(ge=0)
    action_index_in_turn: int = Field(default=0, ge=0)
    total: float
    milestone: float
    violation: float

    @field_validator("action_id", "task_id", "rollout_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reward identifiers must not be blank")
        return value

    @field_validator("total", "milestone", "violation")
    @classmethod
    def rewards_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @classmethod
    def from_action_credit(cls, credit: object) -> "RewardActionValue":
        """Build a comparison row without coupling metrics to schema modules."""

        breakdown = getattr(credit, "reward_breakdown")
        return cls(
            action_id=getattr(credit, "action_id"),
            task_id=getattr(credit, "task_id"),
            rollout_id=getattr(credit, "rollout_id"),
            timestep=getattr(credit, "timestep"),
            total=breakdown.total,
            milestone=breakdown.milestone,
            violation=breakdown.violation,
        )


class ComponentErrorMetrics(BaseModel):
    """Action-level error distribution for one reward component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.component_error_metrics.v1"] = (
        "agemem.component_error_metrics.v1"
    )
    count: int = Field(ge=0)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    bias: float
    max_abs: float = Field(ge=0.0)

    @field_validator("mae", "rmse", "bias", "max_abs")
    @classmethod
    def values_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)


class TrajectoryRewardError(BaseModel):
    """Cumulative total-reward error for one task/rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.trajectory_reward_error.v1"] = (
        "agemem.trajectory_reward_error.v1"
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    action_count: int = Field(ge=1)
    oracle_total: float
    extracted_total: float
    signed_error: float
    absolute_error: float = Field(ge=0.0)

    @field_validator(
        "oracle_total", "extracted_total", "signed_error", "absolute_error"
    )
    @classmethod
    def values_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_error(self) -> "TrajectoryRewardError":
        expected = self.extracted_total - self.oracle_total
        if not math.isclose(self.signed_error, expected, abs_tol=1e-15):
            raise ValueError("trajectory signed_error is inconsistent")
        if not math.isclose(self.absolute_error, abs(expected), abs_tol=1e-15):
            raise ValueError("trajectory absolute_error is inconsistent")
        return self


class RewardDivergence(BaseModel):
    """The first deterministic action where any compared reward differs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.reward_divergence.v1"] = (
        "agemem.reward_divergence.v1"
    )
    action_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    timestep: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)
    total_error: float
    milestone_error: float
    violation_error: float

    @field_validator("total_error", "milestone_error", "violation_error")
    @classmethod
    def errors_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)


class RewardPropagationMetrics(BaseModel):
    """Action and trajectory effects of extracted-AP reward errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.reward_propagation_metrics.v1"] = (
        "agemem.reward_propagation_metrics.v1"
    )
    action_count: int = Field(ge=0)
    trajectory_count: int = Field(ge=0)
    action_total: ComponentErrorMetrics
    action_milestone: ComponentErrorMetrics
    action_violation: ComponentErrorMetrics
    trajectories: Tuple[TrajectoryRewardError, ...]
    trajectory_signed_error_total: float
    trajectory_absolute_error_total: float = Field(ge=0.0)
    first_divergence: Optional[RewardDivergence] = None

    @field_validator("trajectory_signed_error_total", "trajectory_absolute_error_total")
    @classmethod
    def totals_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_aggregates(self) -> "RewardPropagationMetrics":
        if self.trajectory_count != len(self.trajectories):
            raise ValueError("trajectory_count must equal trajectories length")
        if any(
            metric.count != self.action_count
            for metric in (
                self.action_total,
                self.action_milestone,
                self.action_violation,
            )
        ):
            raise ValueError("component counts must equal action_count")
        signed = sum(item.signed_error for item in self.trajectories)
        absolute = sum(item.absolute_error for item in self.trajectories)
        if not math.isclose(
            self.trajectory_signed_error_total, signed, abs_tol=1e-15
        ) or not math.isclose(
            self.trajectory_absolute_error_total, absolute, abs_tol=1e-15
        ):
            raise ValueError("trajectory aggregate errors are inconsistent")
        return self


def _reward_map(
    records: Sequence[RewardActionValue], *, side: str
) -> Dict[str, RewardActionValue]:
    result: Dict[str, RewardActionValue] = {}
    for record in records:
        if record.action_id in result:
            raise ValueError(f"duplicate {side} reward action_id {record.action_id!r}")
        result[record.action_id] = record
    return result


def _component_metrics(errors: Sequence[float]) -> ComponentErrorMetrics:
    if not errors:
        return ComponentErrorMetrics(count=0, mae=0.0, rmse=0.0, bias=0.0, max_abs=0.0)
    count = len(errors)
    return ComponentErrorMetrics(
        count=count,
        mae=sum(abs(value) for value in errors) / count,
        rmse=math.sqrt(sum(value * value for value in errors) / count),
        bias=sum(errors) / count,
        max_abs=max(abs(value) for value in errors),
    )


def score_reward_propagation(
    oracle: Sequence[RewardActionValue],
    extracted: Sequence[RewardActionValue],
    *,
    divergence_tolerance: float = 0.0,
) -> RewardPropagationMetrics:
    """Compare rewards with an exact one-to-one ``action_id`` join."""

    _finite(divergence_tolerance, "divergence_tolerance")
    if divergence_tolerance < 0.0:
        raise ValueError("divergence_tolerance must be non-negative")
    oracle_by_id = _reward_map(oracle, side="Oracle")
    extracted_by_id = _reward_map(extracted, side="extracted")
    if set(oracle_by_id) != set(extracted_by_id):
        raise ValueError("Oracle/extracted reward action_id sets do not match")

    ordered_ids = sorted(
        oracle_by_id,
        key=lambda action_id: (
            oracle_by_id[action_id].task_id,
            oracle_by_id[action_id].rollout_id,
            oracle_by_id[action_id].timestep,
            oracle_by_id[action_id].action_index_in_turn,
            action_id,
        ),
    )
    errors: Dict[str, list[float]] = {
        "total": [],
        "milestone": [],
        "violation": [],
    }
    trajectory_rows: Dict[
        Tuple[str, str], list[Tuple[RewardActionValue, RewardActionValue]]
    ] = defaultdict(list)
    first: Optional[RewardDivergence] = None

    for action_id in ordered_ids:
        expected = oracle_by_id[action_id]
        actual = extracted_by_id[action_id]
        expected_coordinates = (
            expected.task_id,
            expected.rollout_id,
            expected.timestep,
            expected.action_index_in_turn,
        )
        actual_coordinates = (
            actual.task_id,
            actual.rollout_id,
            actual.timestep,
            actual.action_index_in_turn,
        )
        if expected_coordinates != actual_coordinates:
            raise ValueError(f"reward coordinates differ for action_id {action_id!r}")
        deltas = {
            "total": actual.total - expected.total,
            "milestone": actual.milestone - expected.milestone,
            "violation": actual.violation - expected.violation,
        }
        for name, value in deltas.items():
            _finite(value, f"{name}_error")
            errors[name].append(value)
        trajectory_rows[(expected.task_id, expected.rollout_id)].append(
            (expected, actual)
        )
        if first is None and any(
            not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=divergence_tolerance)
            for value in deltas.values()
        ):
            first = RewardDivergence(
                action_id=action_id,
                task_id=expected.task_id,
                rollout_id=expected.rollout_id,
                timestep=expected.timestep,
                action_index_in_turn=expected.action_index_in_turn,
                total_error=deltas["total"],
                milestone_error=deltas["milestone"],
                violation_error=deltas["violation"],
            )

    trajectories = []
    for (task_id, rollout_id), rows in sorted(trajectory_rows.items()):
        oracle_total = sum(expected.total for expected, _ in rows)
        extracted_total = sum(actual.total for _, actual in rows)
        signed_error = extracted_total - oracle_total
        trajectories.append(
            TrajectoryRewardError(
                task_id=task_id,
                rollout_id=rollout_id,
                action_count=len(rows),
                oracle_total=oracle_total,
                extracted_total=extracted_total,
                signed_error=signed_error,
                absolute_error=abs(signed_error),
            )
        )

    trajectory_tuple = tuple(trajectories)
    return RewardPropagationMetrics(
        action_count=len(ordered_ids),
        trajectory_count=len(trajectory_tuple),
        action_total=_component_metrics(errors["total"]),
        action_milestone=_component_metrics(errors["milestone"]),
        action_violation=_component_metrics(errors["violation"]),
        trajectories=trajectory_tuple,
        trajectory_signed_error_total=sum(
            item.signed_error for item in trajectory_tuple
        ),
        trajectory_absolute_error_total=sum(
            item.absolute_error for item in trajectory_tuple
        ),
        first_divergence=first,
    )


__all__ = [
    "APEvaluationRecord",
    "APMetrics",
    "AcceptanceDecision",
    "AcceptanceMetrics",
    "ComponentErrorMetrics",
    "MacroPRF",
    "PRFScore",
    "RewardActionValue",
    "RewardDivergence",
    "RewardPropagationMetrics",
    "TrajectoryRewardError",
    "TripleEvaluationRecord",
    "TripleMetrics",
    "normalize_exact_text",
    "score_acceptance",
    "score_aps",
    "score_reward_propagation",
    "score_triples",
]
