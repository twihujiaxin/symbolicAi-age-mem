"""Deterministic M7 replay of already-grounded M6 action propositions.

This module is deliberately downstream of the M6 semantic pipeline.  It joins
``TrajectoryStepV2`` rows to existing ``ActionCreditRecord`` rows, rebuilds
``OracleAPEvent`` values from the recorded AP/evidence fields, and runs an
arbitrary validated ``AutomatonSpec``.  It never invokes an extractor,
``StateTracker``, or AP grounder.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..action_schema import ActionCreditRecord, RewardBreakdownV2, TrajectoryStepV2
from ..memory_oracle.automaton import DFARunner
from ..memory_oracle.models import (
    AP_ORDER,
    AutomatonSpec,
    DFAStatus,
    OracleAPEvent,
    RewardProfile,
)


GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION = "agemem.group_automaton_replay.v1"
REWARD_FARMING_AUDIT_SCHEMA_VERSION = "agemem.reward_farming_audit.v1"


class GroupAutomatonReplayError(ValueError):
    """Raised when an M6 trajectory/AP-credit join is not trustworthy."""


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class GroupAutomatonReplayResult(BaseModel):
    """Strict action-level result of replaying recorded APs through one DFA."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION] = (
        GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    profile: str = Field(min_length=1)
    dfa_spec_id: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    source_steps_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_credits_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    credits: Tuple[ActionCreditRecord, ...] = Field(min_length=1)
    final_state: str = Field(min_length=1)
    final_status: DFAStatus
    accepted: bool
    env_total: float
    milestone_total: float
    logic_total: float
    violation_total: float
    trend_total: float
    format_total: float
    cost_total: float
    total_reward: float
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "env_total",
        "milestone_total",
        "logic_total",
        "violation_total",
        "trend_total",
        "format_total",
        "cost_total",
        "total_reward",
    )
    @classmethod
    def totals_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> "GroupAutomatonReplayResult":
        action_ids = [credit.action_id for credit in self.credits]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("replayed action_id values must be unique")
        if any(
            credit.task_id != self.task_id
            or credit.rollout_id != self.rollout_id
            or credit.dfa_spec_id != self.dfa_spec_id
            or credit.reward_version != self.reward_version
            or credit.return_to_go is not None
            or credit.advantage is not None
            for credit in self.credits
        ):
            raise ValueError("credit provenance/version fields do not match replay")

        component_totals = {
            "env_total": sum(item.reward_breakdown.env for item in self.credits),
            "milestone_total": sum(
                item.reward_breakdown.milestone for item in self.credits
            ),
            "violation_total": sum(
                item.reward_breakdown.violation for item in self.credits
            ),
            "trend_total": sum(item.reward_breakdown.trend for item in self.credits),
            "format_total": sum(item.reward_breakdown.format for item in self.credits),
            "cost_total": sum(item.reward_breakdown.cost for item in self.credits),
            "total_reward": sum(item.reward_breakdown.total for item in self.credits),
        }
        for field_name, expected in component_totals.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} does not equal the per-action sum")
        if self.accepted != (self.final_status == "accepted"):
            raise ValueError("accepted must agree with final_status")
        expected_logic = sum(
            credit.reward_breakdown.total
            - credit.reward_breakdown.env
            - credit.reward_breakdown.violation
            - credit.reward_breakdown.format
            - credit.reward_breakdown.cost
            for credit in self.credits
        )
        if self.logic_total != expected_logic:
            raise ValueError("logic_total does not match per-action composite rewards")
        last = self.credits[-1]
        if (
            last.dfa_state_after != self.final_state
            or last.reward_breakdown.automaton_status != self.final_status
        ):
            raise ValueError("final DFA state/status must agree with final credit")
        if self.digest != self.expected_digest():
            raise ValueError("replay digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return _canonical_digest(self.canonical_dict(include_digest=False))

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(
                credit.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for credit in self.credits
        )

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_jsonl(), encoding="utf-8", newline="\n")
        return output


class GroupAutomatonReplay:
    """Replay existing action-bound AP evidence without semantic recomputation."""

    def __init__(
        self,
        profile: RewardProfile,
        *,
        spec: AutomatonSpec,
        reward_version: str,
    ) -> None:
        if not reward_version.strip():
            raise ValueError("reward_version must be non-blank")
        self.profile = profile.model_copy(deep=True)
        self.spec = spec.model_copy(deep=True)
        self.reward_version = reward_version

    @staticmethod
    def _validate_sources(
        steps: Sequence[TrajectoryStepV2],
        source_credits: Sequence[ActionCreditRecord],
    ) -> Tuple[str, str]:
        if not steps:
            raise GroupAutomatonReplayError("at least one trajectory step is required")
        if len(steps) != len(source_credits):
            raise GroupAutomatonReplayError(
                "trajectory steps and source credits must have equal length"
            )

        task_id: Optional[str] = None
        rollout_id: Optional[str] = None
        previous_position: Optional[Tuple[int, int, int]] = None
        action_ids: set[str] = set()
        seen_done = False
        previous_memory_after = None

        for index, (step, credit) in enumerate(zip(steps, source_credits)):
            if len(step.actions) != 1:
                raise GroupAutomatonReplayError(
                    "replay requires exactly one action per TrajectoryStepV2"
                )
            action = step.actions[0]
            action_identity = (
                action.task_id,
                action.rollout_id,
                action.stage_id,
                action.timestep,
                action.action_id,
            )
            step_identity = (
                step.task_id,
                step.rollout_id,
                step.stage_id,
                step.timestep,
                action.action_id,
            )
            credit_identity = (
                credit.task_id,
                credit.rollout_id,
                credit.stage_id,
                credit.timestep,
                credit.action_id,
            )
            if action_identity != step_identity or action_identity != credit_identity:
                raise GroupAutomatonReplayError(
                    "trajectory action and source credit identity mismatch"
                )
            if task_id is None:
                task_id, rollout_id = action.task_id, action.rollout_id
            elif (action.task_id, action.rollout_id) != (task_id, rollout_id):
                raise GroupAutomatonReplayError(
                    "all rows must belong to one task rollout"
                )
            if action.action_id in action_ids:
                raise GroupAutomatonReplayError(
                    f"duplicate action_id {action.action_id!r}"
                )
            action_ids.add(action.action_id)

            if (
                previous_memory_after is not None
                and step.memory_before != previous_memory_after
            ):
                raise GroupAutomatonReplayError(
                    "trajectory memory snapshots are not continuous"
                )
            previous_memory_after = step.memory_after

            result_tool_call_id = action.result.get("tool_call_id")
            if (
                not isinstance(result_tool_call_id, str)
                or not result_tool_call_id.strip()
                or result_tool_call_id != action.action_id
            ):
                raise GroupAutomatonReplayError(
                    "action result tool_call_id must match action_id"
                )

            position = (
                action.timestep,
                action.assistant_turn_id,
                action.action_index_in_turn,
            )
            if previous_position is not None and position <= previous_position:
                raise GroupAutomatonReplayError(
                    "trajectory/source credits are not in strict action order"
                )
            previous_position = position
            if seen_done:
                raise GroupAutomatonReplayError("actions cannot occur after done")
            seen_done = step.done
            if step.done and index != len(steps) - 1:
                raise GroupAutomatonReplayError("done step must be the final action")

            propositions = credit.atomic_propositions
            evidence = credit.atomic_proposition_evidence
            if set(evidence) != set(propositions):
                raise GroupAutomatonReplayError(
                    "each source AP must have evidence and no orphan evidence is allowed"
                )
            # Some semantic APs (notably ``answered_correctly``) are grounded by
            # the action coordinate itself and legitimately carry an empty
            # evidence-ID tuple in the canonical M6 artifacts.
            if any(
                len(evidence_ids) != len(set(evidence_ids))
                or any(not item.strip() for item in evidence_ids)
                for evidence_ids in evidence.values()
            ):
                raise GroupAutomatonReplayError(
                    "source AP evidence IDs, when present, must be non-blank and unique"
                )
            expected_order = tuple(ap for ap in AP_ORDER if ap in propositions)
            if propositions != expected_order:
                raise GroupAutomatonReplayError(
                    "source APs are unknown, duplicated, or not in canonical AP_ORDER"
                )

        assert task_id is not None and rollout_id is not None
        return task_id, rollout_id

    def replay(
        self,
        steps: Iterable[TrajectoryStepV2],
        source_credits: Iterable[ActionCreditRecord],
        *,
        seed: int,
    ) -> GroupAutomatonReplayResult:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise GroupAutomatonReplayError("seed must be a non-negative integer")
        step_rows = tuple(steps)
        credit_rows = tuple(source_credits)
        task_id, rollout_id = self._validate_sources(step_rows, credit_rows)
        runner = DFARunner(self.spec, max_steps=self.profile.max_steps)
        replayed = []

        for step, source_credit in zip(step_rows, credit_rows):
            action = step.actions[0]
            try:
                event = OracleAPEvent(
                    task_id=action.task_id,
                    rollout_id=action.rollout_id,
                    seed=seed,
                    timestep=action.timestep,
                    stage=action.stage_id,
                    propositions=source_credit.atomic_propositions,
                    evidence_fact_ids={
                        key: tuple(value)
                        for key, value in source_credit.atomic_proposition_evidence.items()
                    },
                )
            except ValueError as exc:
                raise GroupAutomatonReplayError(
                    f"cannot rebuild OracleAPEvent for {action.action_id!r}: {exc}"
                ) from exc

            transition = runner.step(event, done=step.done)
            env_reward = self.profile.env_weight * step.env_reward
            milestone_reward = self.profile.milestone_weight * len(
                transition.new_progress_edges
            )
            violation_reward = self.profile.violation_weight * len(
                transition.violations
            )
            trend_reward = 0.0
            format_reward = 0.0 * self.profile.format_weight
            cost_reward = 0.0
            total = (
                env_reward
                + self.profile.logic_beta * milestone_reward
                + violation_reward
                + format_reward
                + cost_reward
            )
            breakdown = RewardBreakdownV2(
                env=env_reward,
                milestone=milestone_reward,
                violation=violation_reward,
                trend=trend_reward,
                format=format_reward,
                cost=cost_reward,
                total=total,
                automaton_state_before=transition.state_before,
                automaton_state_after=transition.state_after,
                automaton_status=transition.status,
                propositions=event.propositions,
                fired_edges=transition.fired_edges,
                newly_rewarded_edges=transition.new_progress_edges,
                violation_edges=transition.violations,
            )
            transition_ids = transition.fired_edges
            replayed.append(
                ActionCreditRecord(
                    action_id=action.action_id,
                    task_id=action.task_id,
                    rollout_id=action.rollout_id,
                    stage_id=action.stage_id,
                    timestep=action.timestep,
                    atomic_propositions=event.propositions,
                    atomic_proposition_evidence={
                        key: tuple(value)
                        for key, value in source_credit.atomic_proposition_evidence.items()
                    },
                    dfa_spec_id=self.spec.name,
                    transition_ids=transition_ids,
                    transition_id=(
                        transition_ids[0] if len(transition_ids) == 1 else None
                    ),
                    dfa_state_before=transition.state_before,
                    dfa_state_after=transition.state_after,
                    reward_breakdown=breakdown,
                    return_to_go=None,
                    advantage=None,
                    reward_version=self.reward_version,
                )
            )

        source_steps_digest = _canonical_digest(
            [step.canonical_dict() for step in step_rows]
        )
        source_credits_digest = _canonical_digest(
            [credit.model_dump(mode="json") for credit in credit_rows]
        )
        milestone_total = sum(item.reward_breakdown.milestone for item in replayed)
        base_payload: Dict[str, object] = {
            "schema_version": GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION,
            "task_id": task_id,
            "rollout_id": rollout_id,
            "seed": seed,
            "profile": self.profile.name,
            "dfa_spec_id": self.spec.name,
            "reward_version": self.reward_version,
            "source_steps_digest": source_steps_digest,
            "source_credits_digest": source_credits_digest,
            "credits": [item.model_dump(mode="json") for item in replayed],
            "final_state": runner.state,
            "final_status": runner.status,
            "accepted": runner.status == "accepted",
            "env_total": sum(item.reward_breakdown.env for item in replayed),
            "milestone_total": milestone_total,
            "logic_total": self.profile.logic_beta * milestone_total,
            "violation_total": sum(
                item.reward_breakdown.violation for item in replayed
            ),
            "trend_total": sum(item.reward_breakdown.trend for item in replayed),
            "format_total": sum(item.reward_breakdown.format for item in replayed),
            "cost_total": sum(item.reward_breakdown.cost for item in replayed),
            "total_reward": sum(item.reward_breakdown.total for item in replayed),
        }
        model_payload = dict(base_payload)
        model_payload["credits"] = tuple(replayed)
        return GroupAutomatonReplayResult(
            **model_payload,
            digest=_canonical_digest(base_payload),
        )


class RewardFarmingAudit(BaseModel):
    """Deterministic once-only and reward-cap audit for an adversarial replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[REWARD_FARMING_AUDIT_SCHEMA_VERSION] = (
        REWARD_FARMING_AUDIT_SCHEMA_VERSION
    )
    dfa_spec_id: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    baseline_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    injected_action_ids: Tuple[str, ...] = ()
    progressive_edge_count: int = Field(ge=0)
    maximum_milestone_total: float = Field(ge=0.0)
    maximum_logic_total: float = Field(ge=0.0)
    baseline_milestone_total: float = Field(ge=0.0)
    baseline_logic_total: float = Field(ge=0.0)
    candidate_milestone_total: float = Field(ge=0.0)
    candidate_logic_total: float = Field(ge=0.0)
    injected_milestone_total: float = Field(ge=0.0)
    injected_logic_total: float = Field(ge=0.0)
    newly_rewarded_edges: Tuple[str, ...] = ()
    duplicated_newly_rewarded_edges: Tuple[str, ...] = ()
    once_only: bool
    within_progress_cap: bool
    injected_actions_zero_milestone: bool
    no_reward_gain: bool
    passed: bool
    violations: Tuple[str, ...] = ()
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "maximum_milestone_total",
        "maximum_logic_total",
        "baseline_milestone_total",
        "baseline_logic_total",
        "candidate_milestone_total",
        "candidate_logic_total",
        "injected_milestone_total",
        "injected_logic_total",
    )
    @classmethod
    def rewards_must_be_finite(cls, value: float, info) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_audit(self) -> "RewardFarmingAudit":
        if len(self.injected_action_ids) != len(set(self.injected_action_ids)):
            raise ValueError("injected_action_ids must be unique")
        expected_passed = (
            self.once_only
            and self.within_progress_cap
            and self.injected_actions_zero_milestone
            and self.no_reward_gain
            and not self.violations
        )
        if self.passed != expected_passed:
            raise ValueError("passed does not match farming audit checks")
        if self.digest != self.expected_digest():
            raise ValueError("farming audit digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return _canonical_digest(self.canonical_dict(include_digest=False))


def audit_reward_farming(
    *,
    baseline: GroupAutomatonReplayResult,
    candidate: GroupAutomatonReplayResult,
    spec: AutomatonSpec,
    profile: RewardProfile,
    injected_action_ids: Iterable[str] = (),
) -> RewardFarmingAudit:
    """Audit duplicate/loop actions against once-only and theoretical caps.

    ``injected_action_ids`` identifies adversarial duplicate ADD/RETRIEVE or
    loop actions.  Those actions must receive zero newly-earned milestone
    reward.  The caller constructs these action-level perturbations; replay
    itself never edits canonical trajectories.
    """

    if baseline.dfa_spec_id != spec.name or candidate.dfa_spec_id != spec.name:
        raise GroupAutomatonReplayError("audit results must use the supplied DFA")
    if baseline.profile != profile.name or candidate.profile != profile.name:
        raise GroupAutomatonReplayError("audit results must use the supplied profile")
    if (baseline.task_id, baseline.rollout_id) != (
        candidate.task_id,
        candidate.rollout_id,
    ):
        raise GroupAutomatonReplayError(
            "baseline and candidate must identify the same task and rollout"
        )
    baseline_core_ids = [credit.action_id for credit in baseline.credits]

    injected = tuple(injected_action_ids)
    if len(injected) != len(set(injected)):
        raise GroupAutomatonReplayError("injected_action_ids must be unique")
    candidate_by_id = {credit.action_id: credit for credit in candidate.credits}
    missing = set(injected) - set(candidate_by_id)
    if missing:
        raise GroupAutomatonReplayError(
            f"injected action IDs are absent from candidate: {sorted(missing)}"
        )
    if set(injected) & set(baseline_core_ids):
        raise GroupAutomatonReplayError(
            "injected action IDs must not occur in the baseline replay"
        )
    injected_set = set(injected)
    candidate_core_ids = [
        credit.action_id
        for credit in candidate.credits
        if credit.action_id not in injected_set
    ]
    if candidate_core_ids != baseline_core_ids:
        raise GroupAutomatonReplayError(
            "candidate non-injected actions must exactly preserve the baseline "
            "action_id sequence"
        )
    if len(candidate.credits) != len(baseline.credits) + len(injected):
        raise GroupAutomatonReplayError(
            "candidate must contain only baseline actions and declared injections"
        )

    rewarded_edges = tuple(
        edge
        for credit in candidate.credits
        for edge in credit.reward_breakdown.newly_rewarded_edges
    )
    seen: set[str] = set()
    duplicated = []
    for edge in rewarded_edges:
        if edge in seen and edge not in duplicated:
            duplicated.append(edge)
        seen.add(edge)

    progressive_edge_count = sum(
        1 for transition in spec.transitions if transition.progressive
    )
    maximum_milestone_total = profile.milestone_weight * progressive_edge_count
    maximum_logic_total = profile.logic_beta * maximum_milestone_total
    injected_milestone_total = sum(
        candidate_by_id[action_id].reward_breakdown.milestone for action_id in injected
    )
    injected_logic_total = profile.logic_beta * injected_milestone_total
    tolerance = 1e-12
    once_only = not duplicated and len(rewarded_edges) <= progressive_edge_count
    within_progress_cap = (
        candidate.milestone_total <= maximum_milestone_total + tolerance
        and candidate.logic_total <= maximum_logic_total + tolerance
    )
    injected_zero = abs(injected_milestone_total) <= tolerance
    no_reward_gain = candidate.logic_total <= baseline.logic_total + tolerance

    violations = []
    if not once_only:
        violations.append("progressive_edge_rewarded_more_than_once")
    if not within_progress_cap:
        violations.append("logic_reward_exceeds_progressive_edge_cap")
    if not injected_zero:
        violations.append("injected_duplicate_or_loop_earned_milestone")
    if not no_reward_gain:
        violations.append("adversarial_actions_increased_logic_reward")

    base_payload: Dict[str, object] = {
        "schema_version": REWARD_FARMING_AUDIT_SCHEMA_VERSION,
        "dfa_spec_id": spec.name,
        "profile": profile.name,
        "baseline_digest": baseline.digest,
        "candidate_digest": candidate.digest,
        "injected_action_ids": injected,
        "progressive_edge_count": progressive_edge_count,
        "maximum_milestone_total": maximum_milestone_total,
        "maximum_logic_total": maximum_logic_total,
        "baseline_milestone_total": baseline.milestone_total,
        "baseline_logic_total": baseline.logic_total,
        "candidate_milestone_total": candidate.milestone_total,
        "candidate_logic_total": candidate.logic_total,
        "injected_milestone_total": injected_milestone_total,
        "injected_logic_total": injected_logic_total,
        "newly_rewarded_edges": rewarded_edges,
        "duplicated_newly_rewarded_edges": tuple(duplicated),
        "once_only": once_only,
        "within_progress_cap": within_progress_cap,
        "injected_actions_zero_milestone": injected_zero,
        "no_reward_gain": no_reward_gain,
        "passed": not violations,
        "violations": tuple(violations),
    }
    return RewardFarmingAudit(
        **base_payload,
        digest=_canonical_digest(base_payload),
    )


__all__ = [
    "GROUP_AUTOMATON_REPLAY_SCHEMA_VERSION",
    "REWARD_FARMING_AUDIT_SCHEMA_VERSION",
    "GroupAutomatonReplay",
    "GroupAutomatonReplayError",
    "GroupAutomatonReplayResult",
    "RewardFarmingAudit",
    "audit_reward_farming",
]
