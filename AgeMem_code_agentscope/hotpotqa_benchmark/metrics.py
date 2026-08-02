"""Deterministic, model-free metrics and report schemas for M5."""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..memory_oracle import RewardProfile, RewardReplayResult
from ..trajectory import ReplayResult
from .models import (
    BenchmarkSplit,
    HotpotLevel,
    HotpotQAMemoryTask,
    HotpotQASmokeConfig,
    HotpotType,
    PolicyName,
    SourceFactPointer,
)


def normalize_answer(text: str) -> str:
    """HotpotQA/SQuAD-style lowercase, punctuation, article normalization."""

    lowered = text.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def answer_exact_match(predicted: str, expected: str) -> float:
    return float(normalize_answer(predicted) == normalize_answer(expected))


def answer_f1(predicted: str, expected: str) -> float:
    predicted_normalized = normalize_answer(predicted)
    expected_normalized = normalize_answer(expected)
    special_answers = {"yes", "no", "noanswer"}
    if (
        predicted_normalized in special_answers
        or expected_normalized in special_answers
    ) and predicted_normalized != expected_normalized:
        return 0.0
    predicted_tokens = predicted_normalized.split()
    expected_tokens = expected_normalized.split()
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)
    common = Counter(predicted_tokens) & Counter(expected_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def count_context_tokens(text: str) -> int:
    """Offline tokenizer-independent token estimate used only for M5 diagnostics."""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class MemoryAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(min_length=1)
    fact_id: str = Field(min_length=1)
    status: Literal["active", "superseded", "discarded"]
    version: int = Field(ge=1)


class AutomatonAuditStep(BaseModel):
    """One grounded AP/DFA transition retained for failure explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestep: int = Field(ge=0)
    propositions: Tuple[str, ...]
    state_before: str = Field(min_length=1)
    state_after: str = Field(min_length=1)
    status: Literal["running", "accepted", "rejected", "timed_out"]
    fired_edges: Tuple[str, ...]
    newly_rewarded_edges: Tuple[str, ...]


class FailureAudit(BaseModel):
    """Compact failure evidence without copying full copyrighted context text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    hotpot_id: str
    split: BenchmarkSplit
    policy: PolicyName
    supporting_fact_pointers: Tuple[SourceFactPointer, ...]
    expected_supporting_fact_ids: Tuple[str, ...]
    retrieved_supporting_fact_ids: Tuple[str, ...]
    active_memory_fact_ids: Tuple[str, ...]
    memory_history: Tuple[MemoryAuditItem, ...]
    automaton_trace: Tuple[AutomatonAuditStep, ...]
    automaton_state: str = Field(min_length=1)
    automaton_status: Literal["running", "accepted", "rejected", "timed_out"]
    answer_exact_match: float = Field(ge=0.0, le=1.0)


class BenchmarkRecord(BaseModel):
    """Per-rollout M5 Oracle benchmark metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    hotpot_id: str
    split: BenchmarkSplit
    source_split: Literal["train", "validation"]
    hotpot_type: HotpotType
    level: HotpotLevel
    policy: PolicyName
    seed: int = Field(ge=0)
    trajectory_path: str
    reward_path: str
    trajectory_digest: str = Field(min_length=64, max_length=64)
    reward_digest: str = Field(min_length=64, max_length=64)
    episode_success: bool
    dfa_accepted: bool
    dfa_status: Literal["running", "accepted", "rejected", "timed_out"]
    answer_em: float = Field(ge=0.0, le=1.0)
    answer_f1: float = Field(ge=0.0, le=1.0)
    supporting_fact_exact: float = Field(ge=0.0, le=1.0)
    supporting_fact_coverage: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    retrieval_recall_at_k: float = Field(ge=0.0, le=1.0)
    retrieval_k: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    retrieved_context_tokens: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    memory_tool_calls: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    active_memory_count: int = Field(ge=0)
    supporting_memory_count: int = Field(ge=0)
    env_reward: float
    milestone_reward: float
    violation_reward: float
    format_reward: float
    total_reward: float

    @field_validator(
        "env_reward",
        "milestone_reward",
        "violation_reward",
        "format_reward",
        "total_reward",
    )
    @classmethod
    def rewards_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("benchmark rewards must be finite")
        return value


class BenchmarkAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: BenchmarkSplit
    policy: PolicyName
    rollout_count: int = Field(ge=1)
    episode_success_rate: float = Field(ge=0.0, le=1.0)
    dfa_acceptance_rate: float = Field(ge=0.0, le=1.0)
    answer_em: float = Field(ge=0.0, le=1.0)
    answer_f1: float = Field(ge=0.0, le=1.0)
    supporting_fact_coverage: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    retrieval_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_retrieval_k: float = Field(ge=0.0)
    mean_context_tokens: float = Field(ge=0.0)
    mean_tool_calls: float = Field(ge=0.0)
    mean_total_reward: float

    @field_validator(
        "mean_context_tokens",
        "mean_tool_calls",
        "mean_total_reward",
        "mean_retrieval_k",
    )
    @classmethod
    def means_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("benchmark aggregate values must be finite")
        return value


class OracleBenchmarkReport(BaseModel):
    """Deterministic report joining source, trajectory, AP, DFA, and metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    benchmark_name: Literal["m5-hotpotqa-fullwiki-oracle-smoke"] = (
        "m5-hotpotqa-fullwiki-oracle-smoke"
    )
    dataset_name: Literal["hotpot_qa"] = "hotpot_qa"
    dataset_config: Literal["fullwiki"] = "fullwiki"
    source_split_sizes: Dict[str, int]
    source_fingerprints: Dict[str, str]
    official_test_label_blind_count: int = Field(ge=1)
    manifest_digest: str = Field(min_length=64, max_length=64)
    smoke_config: HotpotQASmokeConfig
    smoke_config_digest: str = Field(min_length=64, max_length=64)
    reward_profile: RewardProfile
    reward_profile_digest: str = Field(min_length=64, max_length=64)
    seed: int = Field(ge=0)
    records: Tuple[BenchmarkRecord, ...]
    aggregates: Tuple[BenchmarkAggregate, ...]
    failures: Tuple[FailureAudit, ...]
    digest: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_records(self) -> "OracleBenchmarkReport":
        if not self.records or not self.aggregates:
            raise ValueError("benchmark report requires records and aggregates")
        if set(self.source_split_sizes) != {"train", "validation", "test"}:
            raise ValueError("report requires all three source split sizes")
        if set(self.source_fingerprints) != {"train", "validation", "test"}:
            raise ValueError("report requires all three source fingerprints")
        if self.official_test_label_blind_count != self.source_split_sizes["test"]:
            raise ValueError("official test label-blind count must cover the split")
        if self.smoke_config_digest != report_digest(
            self.smoke_config.model_dump(mode="json")
        ):
            raise ValueError("smoke config digest does not match its payload")
        if self.reward_profile_digest != report_digest(
            self.reward_profile.model_dump(mode="json")
        ):
            raise ValueError("reward profile digest does not match its payload")
        if self.seed != self.smoke_config.seed:
            raise ValueError("report seed must match the smoke config")
        record_keys = [(item.task_id, item.policy) for item in self.records]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError("benchmark task/policy records must be unique")
        trajectory_paths = [item.trajectory_path for item in self.records]
        reward_paths = [item.reward_path for item in self.records]
        if len(trajectory_paths) != len(set(trajectory_paths)):
            raise ValueError("trajectory paths must be unique")
        if len(reward_paths) != len(set(reward_paths)):
            raise ValueError("reward paths must be unique")
        expected_groups = {
            (split, policy): self.smoke_config.model_dump(mode="python")[
                f"{split}_size"
            ]
            for split in ("train", "dev", "test")
            for policy in self.smoke_config.policies
        }
        if len(self.records) != sum(expected_groups.values()):
            raise ValueError("record count does not match the smoke config")
        actual_groups = {
            key: sum(
                (item.split, item.policy) == key for item in self.records
            )
            for key in expected_groups
        }
        if actual_groups != expected_groups:
            raise ValueError(
                "record groups do not match configured splits and policies"
            )
        if any(item.seed != self.seed for item in self.records):
            raise ValueError("record seeds must match the report seed")
        if tuple(self.aggregates) != aggregate_records(self.records):
            raise ValueError("aggregates do not match per-rollout records")
        failure_keys = {
            (item.task_id, item.policy) for item in self.failures
        }
        if len(failure_keys) != len(self.failures):
            raise ValueError("failure audit task/policy records must be unique")
        expected_failures = {
            (item.task_id, item.policy)
            for item in self.records
            if not item.episode_success or not item.dfa_accepted
        }
        if failure_keys != expected_failures:
            raise ValueError("failure audit must cover every unsuccessful record")
        expected_digest = report_digest(
            self.canonical_dict(include_digest=False)
        )
        if self.digest != expected_digest:
            raise ValueError("benchmark report digest does not match its payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data


def _mean(records: Sequence[BenchmarkRecord], field_name: str) -> float:
    return sum(float(getattr(record, field_name)) for record in records) / len(records)


def aggregate_records(
    records: Iterable[BenchmarkRecord],
) -> Tuple[BenchmarkAggregate, ...]:
    grouped: Dict[Tuple[str, str], List[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.split, record.policy), []).append(record)
    aggregates = []
    split_order = {"train": 0, "dev": 1, "test": 2}
    policy_order = {"gold": 0, "wrong_answer": 1, "missing_support": 2}
    for (split, policy), group in sorted(
        grouped.items(),
        key=lambda item: (split_order[item[0][0]], policy_order[item[0][1]]),
    ):
        aggregates.append(
            BenchmarkAggregate(
                split=split,
                policy=policy,
                rollout_count=len(group),
                episode_success_rate=_mean(group, "episode_success"),
                dfa_acceptance_rate=_mean(group, "dfa_accepted"),
                answer_em=_mean(group, "answer_em"),
                answer_f1=_mean(group, "answer_f1"),
                supporting_fact_coverage=_mean(group, "supporting_fact_coverage"),
                memory_precision=_mean(group, "memory_precision"),
                retrieval_recall_at_k=_mean(group, "retrieval_recall_at_k"),
                mean_retrieval_k=_mean(group, "retrieval_k"),
                mean_context_tokens=_mean(group, "context_tokens"),
                mean_tool_calls=_mean(group, "total_tool_calls"),
                mean_total_reward=_mean(group, "total_reward"),
            )
        )
    return tuple(aggregates)


def _predicted_answer(replay: ReplayResult) -> str:
    for step in reversed(replay.steps):
        for call in step.tool_calls:
            if call.name == "Answer":
                return str(call.input.get("answer", ""))
    return ""


def build_benchmark_record(
    *,
    task: HotpotQAMemoryTask,
    policy: PolicyName,
    seed: int,
    replay: ReplayResult,
    reward: RewardReplayResult,
    trajectory_path: str,
    reward_path: str,
) -> Tuple[BenchmarkRecord, Optional[FailureAudit]]:
    if replay.task_id != task.task_id or reward.task_id != task.task_id:
        raise ValueError("task, trajectory replay, and reward replay identities differ")
    if replay.rollout_id != reward.rollout_id:
        raise ValueError("trajectory and reward rollout IDs differ")
    if not replay.done:
        raise ValueError("M5 benchmark requires a complete trajectory replay")
    if reward.seed != seed:
        raise ValueError("reward seed does not match the benchmark seed")
    if reward.source_trajectory_digest != replay.digest:
        raise ValueError("reward does not derive from the supplied trajectory replay")
    for step in replay.steps:
        metadata = step.tool_results[0].metadata or {}
        identity = (
            metadata.get("task_id"),
            metadata.get("rollout_id"),
            metadata.get("seed"),
        )
        expected_identity = (task.task_id, replay.rollout_id, seed)
        if identity != expected_identity:
            raise ValueError(
                "trajectory metadata identity differs from benchmark identity"
            )
    supporting = set(task.supporting_fact_ids)
    retrieved = {
        fact_id
        for step in reward.steps
        for fact_id in step.event.evidence_fact_ids.get(
            "retrieved_supporting_fact", ()
        )
    }
    active_items = [item for item in replay.final_memory if item.status == "active"]
    active_fact_ids = [str(item.metadata.get("fact_id", "")) for item in active_items]
    if any(not fact_id for fact_id in active_fact_ids):
        raise ValueError("active M5 memory is missing required fact_id metadata")
    supporting_active = [
        fact_id for fact_id in active_fact_ids if fact_id in supporting
    ]
    coverage = len(retrieved & supporting) / len(supporting)
    memory_precision = (
        len(supporting_active) / len(active_items) if active_items else 0.0
    )
    predicted = _predicted_answer(replay)
    total_tool_calls = sum(len(step.tool_calls) for step in replay.steps)
    memory_names = {"Add_memory", "Update_memory", "Delete_memory", "Retrieve_memory"}
    memory_calls = sum(
        call.name in memory_names
        for step in replay.steps
        for call in step.tool_calls
    )
    retrieval_calls = sum(
        call.name == "Retrieve_memory"
        for step in replay.steps
        for call in step.tool_calls
    )
    retrieved_context_tokens = sum(
        count_context_tokens(str(block.get("text", "")))
        for step in replay.steps
        for result in step.tool_results
        if result.name == "Retrieve_memory"
        for block in result.content
    )
    terminal_metadata = replay.steps[-1].tool_results[0].metadata or {}
    episode_success = terminal_metadata.get("episode_success")
    if not isinstance(episode_success, bool):
        raise ValueError("terminal episode_success metadata must be boolean")
    record = BenchmarkRecord(
        task_id=task.task_id,
        hotpot_id=task.hotpot_id,
        split=task.split,
        source_split=task.source_split,
        hotpot_type=task.hotpot_type,
        level=task.level,
        policy=policy,
        seed=seed,
        trajectory_path=trajectory_path,
        reward_path=reward_path,
        trajectory_digest=replay.digest,
        reward_digest=reward.digest,
        episode_success=episode_success,
        dfa_accepted=reward.accepted,
        dfa_status=reward.final_status,
        answer_em=answer_exact_match(predicted, task.answer),
        answer_f1=answer_f1(predicted, task.answer),
        supporting_fact_exact=float(retrieved == supporting),
        supporting_fact_coverage=coverage,
        memory_precision=memory_precision,
        retrieval_recall_at_k=coverage,
        retrieval_k=len(retrieved),
        context_tokens=sum(count_context_tokens(step.observation) for step in replay.steps),
        retrieved_context_tokens=retrieved_context_tokens,
        total_tool_calls=total_tool_calls,
        memory_tool_calls=memory_calls,
        retrieval_calls=retrieval_calls,
        active_memory_count=len(active_items),
        supporting_memory_count=len(supporting_active),
        env_reward=reward.env_total,
        milestone_reward=reward.milestone_total,
        violation_reward=reward.violation_total,
        format_reward=reward.format_total,
        total_reward=reward.total_reward,
    )
    if episode_success and reward.accepted:
        return record, None
    failure = FailureAudit(
        task_id=task.task_id,
        hotpot_id=task.hotpot_id,
        split=task.split,
        policy=policy,
        supporting_fact_pointers=task.supporting_fact_pointers,
        expected_supporting_fact_ids=tuple(sorted(supporting)),
        retrieved_supporting_fact_ids=tuple(sorted(retrieved)),
        active_memory_fact_ids=tuple(sorted(active_fact_ids)),
        memory_history=tuple(
            MemoryAuditItem(
                memory_id=item.memory_id,
                fact_id=str(item.metadata.get("fact_id", "")),
                status=item.status,
                version=item.version,
            )
            for item in replay.final_memory
        ),
        automaton_trace=tuple(
            AutomatonAuditStep(
                timestep=item.event.timestep,
                propositions=item.event.propositions,
                state_before=item.reward.automaton_state_before,
                state_after=item.reward.automaton_state_after,
                status=item.reward.automaton_status,
                fired_edges=item.reward.fired_edges,
                newly_rewarded_edges=item.reward.newly_rewarded_edges,
            )
            for item in reward.steps
        ),
        automaton_state=reward.final_state,
        automaton_status=reward.final_status,
        answer_exact_match=record.answer_em,
    )
    return record, failure


def report_digest(payload: Dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_report_json(report: OracleBenchmarkReport, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def write_failures_jsonl(report: OracleBenchmarkReport, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            failure.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for failure in report.failures
    )
    output.write_text(payload, encoding="utf-8", newline="\n")
    return output


__all__ = [
    "AutomatonAuditStep",
    "BenchmarkAggregate",
    "BenchmarkRecord",
    "FailureAudit",
    "MemoryAuditItem",
    "OracleBenchmarkReport",
    "aggregate_records",
    "answer_exact_match",
    "answer_f1",
    "build_benchmark_record",
    "count_context_tokens",
    "normalize_answer",
    "report_digest",
    "write_failures_jsonl",
    "write_report_json",
]
