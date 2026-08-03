"""Strict M6 action, credit, and migration data contracts.

The schema versions in this module are namespaced strings on purpose.  They
must not be confused with the integer ``schema_version`` used by the immutable
M1/M4 source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..trajectory import MemorySnapshotItem


ACTION_EVENT_SCHEMA_VERSION = "agemem.action_event.v2"
TRAJECTORY_STEP_V2_SCHEMA_VERSION = "agemem.trajectory_step.v2"
REWARD_BREAKDOWN_V2_SCHEMA_VERSION = "agemem.reward_breakdown.v2"
ACTION_CREDIT_SCHEMA_VERSION = "agemem.action_credit.v2"
MIGRATION_FILE_SCHEMA_VERSION = "agemem.migration_file.v1"
MIGRATION_MANIFEST_SCHEMA_VERSION = "agemem.migration_manifest.v1"
M5_TO_M6_MIGRATION_VERSION = "agemem.migration.m5_v1_to_m6_v2.v1"
M5_ORACLE_REWARD_VERSION = "agemem.reward.m4_oracle.v1"

ActionSource = Literal["rule", "oracle", "random", "error_injector", "llm"]
M5Policy = Literal["gold", "wrong_answer", "missing_support"]
DFAStatus = Literal["running", "accepted", "rejected", "timed_out"]


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class ActionEvent(BaseModel):
    """One independently addressable action produced in an assistant turn."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[ACTION_EVENT_SCHEMA_VERSION] = ACTION_EVENT_SCHEMA_VERSION
    action_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    assistant_turn_id: int = Field(ge=0)
    action_index_in_turn: int = Field(ge=0)
    source: ActionSource
    action_type: str = Field(min_length=1)
    action_text: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    response_token_ids: Optional[Tuple[int, ...]] = None
    token_start: Optional[int] = Field(default=None, ge=0)
    token_end: Optional[int] = Field(default=None, ge=0)
    old_logprobs: Optional[Tuple[float, ...]] = None
    policy_version: Optional[str] = Field(default=None, min_length=1)

    @field_validator("response_token_ids")
    @classmethod
    def token_ids_must_be_non_negative(
        cls, value: Optional[Tuple[int, ...]]
    ) -> Optional[Tuple[int, ...]]:
        if value is not None and (not value or any(token < 0 for token in value)):
            raise ValueError("response_token_ids must be non-empty and non-negative")
        return value

    @field_validator("old_logprobs")
    @classmethod
    def logprobs_must_be_finite(
        cls, value: Optional[Tuple[float, ...]]
    ) -> Optional[Tuple[float, ...]]:
        if value is not None and (
            not value or any(not math.isfinite(item) for item in value)
        ):
            raise ValueError("old_logprobs must be non-empty and finite")
        return value

    @model_validator(mode="after")
    def validate_token_contract(self) -> "ActionEvent":
        token_fields = (
            self.response_token_ids,
            self.token_start,
            self.token_end,
            self.old_logprobs,
            self.policy_version,
        )
        populated = tuple(item is not None for item in token_fields)
        if any(populated) and not all(populated):
            raise ValueError("token metadata must be provided together or all be None")
        if self.source == "llm" and not all(populated):
            raise ValueError("LLM actions require token metadata and policy_version")
        if all(populated):
            assert self.response_token_ids is not None
            assert self.token_start is not None
            assert self.token_end is not None
            assert self.old_logprobs is not None
            if len(self.old_logprobs) != len(self.response_token_ids):
                raise ValueError(
                    "old_logprobs length must equal response_token_ids length"
                )
            if not self.token_start < self.token_end <= len(self.response_token_ids):
                raise ValueError(
                    "token span must satisfy 0 <= start < end <= token count"
                )
        return self


class TrajectoryStepV2(BaseModel):
    """M6 view of one trajectory step with explicit action identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[TRAJECTORY_STEP_V2_SCHEMA_VERSION] = (
        TRAJECTORY_STEP_V2_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    observation: str
    actions: Tuple[ActionEvent, ...] = Field(min_length=1)
    memory_before: Tuple[MemorySnapshotItem, ...]
    memory_after: Tuple[MemorySnapshotItem, ...]
    env_reward: float = 0.0
    done: bool = False

    @field_validator("env_reward")
    @classmethod
    def env_reward_must_be_finite(cls, value: float) -> float:
        return _finite(value, "env_reward")

    @model_validator(mode="after")
    def validate_actions(self) -> "TrajectoryStepV2":
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be unique within a step")
        expected_indices = tuple(range(len(self.actions)))
        actual_indices = tuple(action.action_index_in_turn for action in self.actions)
        if actual_indices != expected_indices:
            raise ValueError(
                "actions must follow contiguous action_index_in_turn order"
            )
        assistant_turn_ids = {action.assistant_turn_id for action in self.actions}
        if len(assistant_turn_ids) != 1:
            raise ValueError("all actions in a step must belong to one assistant turn")
        for action in self.actions:
            if (
                action.task_id != self.task_id
                or action.rollout_id != self.rollout_id
                or action.stage_id != self.stage_id
                or action.timestep != self.timestep
            ):
                raise ValueError("action identity must match its containing step")
        tokenized = [action for action in self.actions if action.token_start is not None]
        if tokenized and len(tokenized) != len(self.actions):
            raise ValueError(
                "all actions in one assistant turn must share token metadata presence"
            )
        if tokenized:
            first = tokenized[0]
            shared = (
                first.response_token_ids,
                first.old_logprobs,
                first.policy_version,
            )
            previous_end = -1
            for action in tokenized:
                if (
                    action.response_token_ids,
                    action.old_logprobs,
                    action.policy_version,
                ) != shared:
                    raise ValueError(
                        "actions in one assistant turn must share response tokens, "
                        "old_logprobs, and policy_version"
                    )
                assert action.token_start is not None
                assert action.token_end is not None
                if action.token_start < previous_end:
                    raise ValueError("action token spans must be ordered and non-overlapping")
                previous_end = action.token_end
        return self

    def canonical_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_line(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class RewardBreakdownV2(BaseModel):
    """Non-destructive v2 view of the M4 per-action reward components."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[REWARD_BREAKDOWN_V2_SCHEMA_VERSION] = (
        REWARD_BREAKDOWN_V2_SCHEMA_VERSION
    )
    env: float
    milestone: float
    violation: float
    trend: float
    format: float
    cost: float
    total: float
    automaton_state_before: str = Field(min_length=1)
    automaton_state_after: str = Field(min_length=1)
    automaton_status: DFAStatus
    propositions: Tuple[str, ...] = ()
    fired_edges: Tuple[str, ...] = ()
    newly_rewarded_edges: Tuple[str, ...] = ()
    violation_edges: Tuple[str, ...] = ()

    @field_validator(
        "env", "milestone", "violation", "trend", "format", "cost", "total"
    )
    @classmethod
    def components_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_edges(self) -> "RewardBreakdownV2":
        if len(self.fired_edges) != len(set(self.fired_edges)):
            raise ValueError("fired_edges must be unique and ordered")
        if set(self.newly_rewarded_edges) - set(self.fired_edges):
            raise ValueError("newly_rewarded_edges must be a subset of fired_edges")
        if set(self.violation_edges) - set(self.fired_edges):
            raise ValueError("violation_edges must be a subset of fired_edges")
        return self


class ActionCreditRecord(BaseModel):
    """AP, DFA, and reward derivation joined to exactly one ``action_id``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[ACTION_CREDIT_SCHEMA_VERSION] = ACTION_CREDIT_SCHEMA_VERSION
    action_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    atomic_propositions: Tuple[str, ...] = ()
    atomic_proposition_evidence: Dict[str, Tuple[str, ...]] = Field(
        default_factory=dict
    )
    dfa_spec_id: str = Field(min_length=1)
    transition_ids: Tuple[str, ...] = ()
    transition_id: Optional[str] = None
    dfa_state_before: str = Field(min_length=1)
    dfa_state_after: str = Field(min_length=1)
    reward_breakdown: RewardBreakdownV2
    return_to_go: Optional[float] = None
    advantage: Optional[float] = None
    reward_version: str = Field(min_length=1)

    @field_validator("return_to_go", "advantage")
    @classmethod
    def training_values_must_be_finite(
        cls, value: Optional[float], info
    ) -> Optional[float]:
        return None if value is None else _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_credit(self) -> "ActionCreditRecord":
        if len(self.atomic_propositions) != len(set(self.atomic_propositions)):
            raise ValueError("atomic_propositions must be unique and ordered")
        if set(self.atomic_proposition_evidence) - set(self.atomic_propositions):
            raise ValueError("AP evidence keys must occur in atomic_propositions")
        if len(self.transition_ids) != len(set(self.transition_ids)):
            raise ValueError("transition_ids must be unique and ordered")
        expected_singular = (
            self.transition_ids[0] if len(self.transition_ids) == 1 else None
        )
        if self.transition_id != expected_singular:
            raise ValueError(
                "transition_id is populated only when transition_ids has one item"
            )
        if self.transition_ids != self.reward_breakdown.fired_edges:
            raise ValueError("transition_ids must preserve reward fired_edges order")
        if self.atomic_propositions != self.reward_breakdown.propositions:
            raise ValueError("credit propositions must match reward propositions")
        if (
            self.dfa_state_before != self.reward_breakdown.automaton_state_before
            or self.dfa_state_after != self.reward_breakdown.automaton_state_after
        ):
            raise ValueError("credit DFA states must match reward_breakdown")
        return self

    def canonical_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_line(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class MigrationFileRecord(BaseModel):
    """Hashes and row counts for one canonical M5 rollout migration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[MIGRATION_FILE_SCHEMA_VERSION] = (
        MIGRATION_FILE_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    policy: M5Policy
    source: ActionSource
    source_trajectory_path: str = Field(min_length=1)
    source_reward_path: str = Field(min_length=1)
    target_trajectory_path: str = Field(min_length=1)
    target_credit_path: str = Field(min_length=1)
    source_trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reward_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_credit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_count: int = Field(ge=1)
    credit_count: int = Field(ge=1)
    joined_action_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_join_count(self) -> "MigrationFileRecord":
        if not self.action_count == self.credit_count == self.joined_action_count:
            raise ValueError("each migrated action must have exactly one credit record")
        return self


class MigrationManifest(BaseModel):
    """Deterministic audit manifest for a non-destructive M5 migration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[MIGRATION_MANIFEST_SCHEMA_VERSION] = (
        MIGRATION_MANIFEST_SCHEMA_VERSION
    )
    migration_version: Literal[M5_TO_M6_MIGRATION_VERSION] = M5_TO_M6_MIGRATION_VERSION
    source_benchmark_name: Literal["m5-hotpotqa-fullwiki-oracle-smoke"] = (
        "m5-hotpotqa-fullwiki-oracle-smoke"
    )
    source_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_rollout_count: int = Field(ge=1)
    action_count: int = Field(ge=1)
    credit_count: int = Field(ge=1)
    joined_action_count: int = Field(ge=1)
    source_hashes_verified_unchanged: Literal[True] = True
    files: Tuple[MigrationFileRecord, ...] = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self) -> "MigrationManifest":
        if self.canonical_rollout_count != len(self.files):
            raise ValueError("canonical_rollout_count must equal files length")
        action_count = sum(item.action_count for item in self.files)
        credit_count = sum(item.credit_count for item in self.files)
        join_count = sum(item.joined_action_count for item in self.files)
        if (
            self.action_count != action_count
            or self.credit_count != credit_count
            or self.joined_action_count != join_count
            or not self.action_count == self.credit_count == self.joined_action_count
        ):
            raise ValueError("manifest aggregate join counts are inconsistent")
        source_paths = [item.source_trajectory_path for item in self.files]
        reward_paths = [item.source_reward_path for item in self.files]
        target_paths = [item.target_trajectory_path for item in self.files]
        credit_paths = [item.target_credit_path for item in self.files]
        for paths in (source_paths, reward_paths, target_paths, credit_paths):
            if len(paths) != len(set(paths)):
                raise ValueError("migration file paths must be unique")
        expected_digest = self.digest_payload(self.canonical_dict(include_digest=False))
        if self.digest != expected_digest:
            raise ValueError("migration manifest digest does not match its payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def digest_payload(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "ACTION_CREDIT_SCHEMA_VERSION",
    "ACTION_EVENT_SCHEMA_VERSION",
    "M5_ORACLE_REWARD_VERSION",
    "M5_TO_M6_MIGRATION_VERSION",
    "MIGRATION_FILE_SCHEMA_VERSION",
    "MIGRATION_MANIFEST_SCHEMA_VERSION",
    "REWARD_BREAKDOWN_V2_SCHEMA_VERSION",
    "TRAJECTORY_STEP_V2_SCHEMA_VERSION",
    "ActionCreditRecord",
    "ActionEvent",
    "ActionSource",
    "MigrationFileRecord",
    "MigrationManifest",
    "RewardBreakdownV2",
    "TrajectoryStepV2",
]
