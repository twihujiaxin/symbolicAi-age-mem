"""Strict data contracts for M4 Oracle AP grounding and offline rewards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


APName = Literal[
    "observed_supporting_fact",
    "stored_supporting_fact",
    "stored_irrelevant_fact",
    "updated_stale_fact",
    "deleted_supporting_fact",
    "retrieved_supporting_fact",
    "retrieved_irrelevant_fact",
    "supporting_coverage_complete",
    "answered_correctly",
]
DFAStatus = Literal["running", "accepted", "rejected", "timed_out"]

AP_ORDER: Tuple[APName, ...] = (
    "observed_supporting_fact",
    "stored_supporting_fact",
    "stored_irrelevant_fact",
    "updated_stale_fact",
    "deleted_supporting_fact",
    "retrieved_supporting_fact",
    "retrieved_irrelevant_fact",
    "supporting_coverage_complete",
    "answered_correctly",
)


class OracleAPEvent(BaseModel):
    """Semantic propositions grounded from one validated M3 Oracle label set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    timestep: int = Field(ge=0)
    stage: int = Field(ge=1, le=3)
    propositions: Tuple[APName, ...] = ()
    evidence_fact_ids: Dict[APName, Tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_propositions(self) -> "OracleAPEvent":
        if len(self.propositions) != len(set(self.propositions)):
            raise ValueError("propositions must be unique")
        expected = tuple(ap for ap in AP_ORDER if ap in self.propositions)
        if self.propositions != expected:
            raise ValueError("propositions must follow canonical AP_ORDER")
        if set(self.evidence_fact_ids) - set(self.propositions):
            raise ValueError("evidence keys must also appear in propositions")
        return self


class AutomatonTransition(BaseModel):
    """One deterministic transition in the hand-authored M4 DFA."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(min_length=1)
    proposition: APName
    source_states: Tuple[str, ...] = Field(min_length=1)
    target_state: Optional[str] = None
    priority: int = Field(ge=0)
    progressive: bool = False
    violation: bool = False

    @model_validator(mode="after")
    def validate_kind(self) -> "AutomatonTransition":
        if self.progressive and self.violation:
            raise ValueError("a transition cannot be progressive and a violation")
        if len(self.source_states) != len(set(self.source_states)):
            raise ValueError("source_states must be unique")
        return self


class AutomatonSpec(BaseModel):
    """Validated finite-state specification used by the offline runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    states: Tuple[str, ...] = Field(min_length=2)
    initial_state: str
    accepting_states: Tuple[str, ...] = Field(min_length=1)
    rejecting_states: Tuple[str, ...] = Field(min_length=1)
    timeout_state: str
    transitions: Tuple[AutomatonTransition, ...] = Field(min_length=1)
    source_milestones: Tuple[APName, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "AutomatonSpec":
        states = set(self.states)
        if len(states) != len(self.states):
            raise ValueError("states must be unique")
        required_states = {
            self.initial_state,
            self.timeout_state,
            *self.accepting_states,
            *self.rejecting_states,
        }
        if required_states - states:
            raise ValueError("initial, terminal, and timeout states must be declared")
        if set(self.accepting_states) & set(self.rejecting_states):
            raise ValueError("accepting and rejecting states must be disjoint")
        if self.initial_state in set(self.accepting_states) | set(self.rejecting_states):
            raise ValueError("initial state cannot be terminal")
        if self.timeout_state in set(self.accepting_states) | set(self.rejecting_states):
            raise ValueError("timeout state must be distinct from accept/reject states")

        edge_ids = [transition.edge_id for transition in self.transitions]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge_id values must be unique")
        priorities = [transition.priority for transition in self.transitions]
        if len(priorities) != len(set(priorities)):
            raise ValueError("transition priorities must be globally unique")

        deterministic_keys = set()
        for transition in self.transitions:
            if set(transition.source_states) - states:
                raise ValueError("transition references an unknown source state")
            if transition.target_state is not None and transition.target_state not in states:
                raise ValueError("transition references an unknown target state")
            for source in transition.source_states:
                key = (source, transition.proposition)
                if key in deterministic_keys:
                    raise ValueError(
                        "DFA is nondeterministic for source state and proposition"
                    )
                deterministic_keys.add(key)
        if len(self.source_milestones) != len(set(self.source_milestones)):
            raise ValueError("source_milestones must be unique")
        return self


class RewardProfile(BaseModel):
    """Externalized weights for terminal-only or terminal-plus-DFA replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    env_weight: float = Field(ge=0.0)
    logic_beta: float = Field(ge=0.0)
    milestone_weight: float = Field(ge=0.0)
    violation_weight: float = Field(le=0.0)
    trend_weight: Literal[0.0] = 0.0
    format_weight: float = Field(ge=0.0)
    max_steps: int = Field(ge=1)

    @field_validator(
        "env_weight",
        "logic_beta",
        "milestone_weight",
        "violation_weight",
        "trend_weight",
        "format_weight",
    )
    @classmethod
    def weights_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward weights must be finite")
        return value


class RewardConfig(BaseModel):
    """Named experiment profiles loaded from JSON rather than hard-coded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profiles: Dict[str, RewardProfile]

    @model_validator(mode="after")
    def validate_profiles(self) -> "RewardConfig":
        if not self.profiles:
            raise ValueError("at least one reward profile is required")
        for key, profile in self.profiles.items():
            if key != profile.name:
                raise ValueError("reward profile key must equal profile.name")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "RewardConfig":
        config_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(config_path.read_text(encoding="utf-8"))

    def profile(self, name: str) -> RewardProfile:
        try:
            return self.profiles[name].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"unknown reward profile {name!r}") from exc


class DFAStepResult(BaseModel):
    """Auditable state change produced by one AP event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_before: str
    state_after: str
    status: DFAStatus
    fired_edges: Tuple[str, ...] = ()
    new_progress_edges: Tuple[str, ...] = ()
    repeated_progress_edges: Tuple[str, ...] = ()
    violations: Tuple[str, ...] = ()


class RewardBreakdown(BaseModel):
    """Per-step task, milestone, violation, format, and composite rewards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    rollout_id: str
    seed: int = Field(ge=0)
    timestep: int = Field(ge=0)
    env: float
    milestone: float
    violation: float
    trend: Literal[0.0] = 0.0
    format: float
    total: float
    automaton_state_before: str
    automaton_state_after: str
    automaton_status: DFAStatus
    propositions: Tuple[APName, ...] = ()
    fired_edges: Tuple[str, ...] = ()
    newly_rewarded_edges: Tuple[str, ...] = ()
    violation_edges: Tuple[str, ...] = ()

    @field_validator("env", "milestone", "violation", "format", "total")
    @classmethod
    def reward_values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward values must be finite")
        return value


class RewardedTrajectoryStep(BaseModel):
    """One replayed trajectory step with its grounded APs and reward."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: OracleAPEvent
    reward: RewardBreakdown


class RewardReplayResult(BaseModel):
    """Deterministic offline result for one task rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_id: str
    rollout_id: str
    seed: int = Field(ge=0)
    profile: str
    source_trajectory_digest: str = Field(min_length=64, max_length=64)
    steps: Tuple[RewardedTrajectoryStep, ...]
    final_state: str
    final_status: DFAStatus
    accepted: bool
    env_total: float
    milestone_total: float
    violation_total: float
    format_total: float
    total_reward: float
    digest: str = Field(min_length=64, max_length=64)

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
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

    def write_jsonl(self, path: str | Path) -> Path:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                step.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for step in self.steps
        )
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        return output_path

    @staticmethod
    def digest_payload(payload: Dict[str, object]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "APName",
    "AP_ORDER",
    "AutomatonSpec",
    "AutomatonTransition",
    "DFAStatus",
    "DFAStepResult",
    "OracleAPEvent",
    "RewardBreakdown",
    "RewardConfig",
    "RewardProfile",
    "RewardReplayResult",
    "RewardedTrajectoryStep",
]
