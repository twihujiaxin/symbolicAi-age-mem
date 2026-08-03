"""Deterministic offline DFA rewards for action-grounded M6 propositions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..action_schema import (
    ActionCreditRecord,
    RewardBreakdownV2,
    TrajectoryStepV2,
)
from ..memory_oracle.automaton import DFARunner, hand_authored_memory_dfa
from ..memory_oracle.models import AutomatonSpec, DFAStatus, RewardConfig, RewardProfile
from ..memory_oracle.replay import default_reward_config_path
from .grounding import GroundedAction
from .models import APRecord, canonical_digest


EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION = "agemem.extracted_rewarded_action.v1"
EXTRACTED_REPLAY_SCHEMA_VERSION = "agemem.extracted_reward_replay.v1"
DEFAULT_EXTRACTED_REWARD_VERSION = "agemem.reward.extracted_dfa.v1"


class ExtractedRewardReplayError(ValueError):
    """Raised when trajectory and grounded-action provenance cannot be joined."""


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


class ExtractedRewardedAction(BaseModel):
    """One action's extracted AP records and its derived credit record."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION] = (
        EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    stage_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    atomic_propositions: Tuple[APRecord, ...] = ()
    credit: ActionCreditRecord

    @model_validator(mode="after")
    def validate_join(self) -> "ExtractedRewardedAction":
        identity = (
            self.task_id,
            self.rollout_id,
            self.stage_id,
            self.timestep,
            self.action_id,
        )
        if any(
            (
                item.task_id,
                item.rollout_id,
                item.stage_id,
                item.timestep,
                item.action_id,
            )
            != identity
            for item in self.atomic_propositions
        ):
            raise ValueError("AP identity must match rewarded action")
        credit_identity = (
            self.credit.task_id,
            self.credit.rollout_id,
            self.credit.stage_id,
            self.credit.timestep,
            self.credit.action_id,
        )
        if credit_identity != identity:
            raise ValueError("credit identity must match rewarded action")
        propositions = tuple(item.proposition for item in self.atomic_propositions)
        if propositions != self.credit.atomic_propositions:
            raise ValueError("AP records must match credit propositions in order")
        expected_evidence = {
            item.proposition: (item.ap_id,) for item in self.atomic_propositions
        }
        if self.credit.atomic_proposition_evidence != expected_evidence:
            raise ValueError("credit AP evidence must contain the APRecord IDs")
        return self

    def to_json_line(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ExtractedRewardReplayResult(BaseModel):
    """Complete deterministic replay result for one extracted-AP rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[EXTRACTED_REPLAY_SCHEMA_VERSION] = (
        EXTRACTED_REPLAY_SCHEMA_VERSION
    )
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    profile: str = Field(min_length=1)
    dfa_spec_id: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    source_trajectory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounded_actions_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    actions: Tuple[ExtractedRewardedAction, ...] = Field(min_length=1)
    final_state: str = Field(min_length=1)
    final_status: DFAStatus
    accepted: bool
    env_total: float
    milestone_total: float
    violation_total: float
    trend_total: float
    format_total: float
    cost_total: float
    total_reward: float
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "env_total",
        "milestone_total",
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
    def validate_result(self) -> "ExtractedRewardReplayResult":
        identities = [item.action_id for item in self.actions]
        if len(identities) != len(set(identities)):
            raise ValueError("rewarded action IDs must be unique")
        if any(
            item.task_id != self.task_id or item.rollout_id != self.rollout_id
            for item in self.actions
        ):
            raise ValueError("rewarded actions must belong to the result rollout")
        totals = {
            "env_total": sum(item.credit.reward_breakdown.env for item in self.actions),
            "milestone_total": sum(
                item.credit.reward_breakdown.milestone for item in self.actions
            ),
            "violation_total": sum(
                item.credit.reward_breakdown.violation for item in self.actions
            ),
            "trend_total": sum(
                item.credit.reward_breakdown.trend for item in self.actions
            ),
            "format_total": sum(
                item.credit.reward_breakdown.format for item in self.actions
            ),
            "cost_total": sum(
                item.credit.reward_breakdown.cost for item in self.actions
            ),
            "total_reward": sum(
                item.credit.reward_breakdown.total for item in self.actions
            ),
        }
        for field_name, expected in totals.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} does not equal per-action sum")
        if self.accepted != (self.final_status == "accepted"):
            raise ValueError("accepted must agree with final_status")
        if self.digest != self.expected_digest():
            raise ValueError("replay digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return canonical_digest(self.canonical_dict(include_digest=False))

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def to_jsonl(self) -> str:
        return "".join(item.to_json_line() + "\n" for item in self.actions)

    def write_jsonl(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_jsonl(), encoding="utf-8", newline="\n")
        return output


class ExtractedRewardReplay:
    """Replay action-grounded APs through the unchanged M4 positive DFA."""

    def __init__(
        self,
        profile: RewardProfile,
        *,
        extractor_version: str,
        reward_version: str = DEFAULT_EXTRACTED_REWARD_VERSION,
        spec: Optional[AutomatonSpec] = None,
    ) -> None:
        if not extractor_version.strip():
            raise ValueError("extractor_version must be non-blank")
        if not reward_version.strip():
            raise ValueError("reward_version must be non-blank")
        self.profile = profile.model_copy(deep=True)
        self.spec = (spec or hand_authored_memory_dfa()).model_copy(deep=True)
        self.extractor_version = extractor_version
        self.reward_version = reward_version

    @classmethod
    def from_config(
        cls,
        profile_name: str = "terminal_dfa",
        *,
        extractor_version: str,
        reward_version: str = DEFAULT_EXTRACTED_REWARD_VERSION,
        path: Optional[str | Path] = None,
        spec: Optional[AutomatonSpec] = None,
    ) -> "ExtractedRewardReplay":
        config = RewardConfig.from_json(path or default_reward_config_path())
        return cls(
            config.profile(profile_name),
            extractor_version=extractor_version,
            reward_version=reward_version,
            spec=spec,
        )

    @staticmethod
    def _validate_pairs(
        pairs: Sequence[Tuple[TrajectoryStepV2, GroundedAction]],
    ) -> Tuple[str, str]:
        if not pairs:
            raise ExtractedRewardReplayError("at least one action pair is required")
        task_id: Optional[str] = None
        rollout_id: Optional[str] = None
        action_ids: set[str] = set()
        previous_position: Optional[Tuple[int, int, int]] = None
        seen_done = False
        for index, (step, grounded) in enumerate(pairs):
            if len(step.actions) != 1:
                raise ExtractedRewardReplayError(
                    "reward replay requires one action per memory-snapshot step"
                )
            action = step.actions[0]
            identity = (
                action.task_id,
                action.rollout_id,
                action.stage_id,
                action.timestep,
                action.action_id,
            )
            grounded_identity = (
                grounded.task_id,
                grounded.rollout_id,
                grounded.stage_id,
                grounded.timestep,
                grounded.action_id,
            )
            step_identity = (
                step.task_id,
                step.rollout_id,
                step.stage_id,
                step.timestep,
            )
            if grounded_identity != identity or step_identity != identity[:4]:
                raise ExtractedRewardReplayError(
                    "trajectory action and GroundedAction identity mismatch"
                )
            if task_id is None:
                task_id, rollout_id = action.task_id, action.rollout_id
            elif (action.task_id, action.rollout_id) != (task_id, rollout_id):
                raise ExtractedRewardReplayError(
                    "all action pairs must belong to one task rollout"
                )
            if action.action_id in action_ids:
                raise ExtractedRewardReplayError(
                    f"duplicate action_id {action.action_id!r}"
                )
            action_ids.add(action.action_id)
            position = (
                action.timestep,
                action.assistant_turn_id,
                action.action_index_in_turn,
            )
            if previous_position is not None and position <= previous_position:
                raise ExtractedRewardReplayError(
                    "action pairs are not strictly ordered"
                )
            previous_position = position
            if seen_done:
                raise ExtractedRewardReplayError(
                    "actions cannot occur after a done step"
                )
            seen_done = step.done
            if step.done and index != len(pairs) - 1:
                raise ExtractedRewardReplayError("done step must be the final action")
        assert task_id is not None and rollout_id is not None
        return task_id, rollout_id

    def replay(
        self,
        pairs: Iterable[Tuple[TrajectoryStepV2, GroundedAction]],
        *,
        seed: int,
    ) -> ExtractedRewardReplayResult:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ExtractedRewardReplayError("seed must be a non-negative integer")
        items = tuple(pairs)
        task_id, rollout_id = self._validate_pairs(items)
        runner = DFARunner(self.spec, max_steps=self.profile.max_steps)
        rewarded = []

        for step, grounded in items:
            action = step.actions[0]
            event = grounded.to_oracle_event(seed=seed)
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
            )
            propositions = tuple(
                item.proposition for item in grounded.atomic_propositions
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
                propositions=propositions,
                fired_edges=transition.fired_edges,
                newly_rewarded_edges=transition.new_progress_edges,
                violation_edges=transition.violations,
            )
            evidence = {
                item.proposition: (item.ap_id,) for item in grounded.atomic_propositions
            }
            transition_ids = transition.fired_edges
            credit = ActionCreditRecord(
                action_id=action.action_id,
                task_id=action.task_id,
                rollout_id=action.rollout_id,
                stage_id=action.stage_id,
                timestep=action.timestep,
                atomic_propositions=propositions,
                atomic_proposition_evidence=evidence,
                dfa_spec_id=self.spec.name,
                transition_ids=transition_ids,
                transition_id=(transition_ids[0] if len(transition_ids) == 1 else None),
                dfa_state_before=transition.state_before,
                dfa_state_after=transition.state_after,
                reward_breakdown=breakdown,
                return_to_go=None,
                advantage=None,
                reward_version=self.reward_version,
            )
            rewarded.append(
                ExtractedRewardedAction(
                    task_id=action.task_id,
                    rollout_id=action.rollout_id,
                    stage_id=action.stage_id,
                    timestep=action.timestep,
                    action_id=action.action_id,
                    atomic_propositions=grounded.atomic_propositions,
                    credit=credit,
                )
            )

        source_digest = canonical_digest([step.canonical_dict() for step, _ in items])
        grounded_digest = canonical_digest(
            [grounded.model_dump(mode="json") for _, grounded in items]
        )
        base_payload: Dict[str, object] = {
            "schema_version": EXTRACTED_REPLAY_SCHEMA_VERSION,
            "task_id": task_id,
            "rollout_id": rollout_id,
            "seed": seed,
            "profile": self.profile.name,
            "dfa_spec_id": self.spec.name,
            "reward_version": self.reward_version,
            "extractor_version": self.extractor_version,
            "source_trajectory_digest": source_digest,
            "grounded_actions_digest": grounded_digest,
            "actions": [item.model_dump(mode="json") for item in rewarded],
            "final_state": runner.state,
            "final_status": runner.status,
            "accepted": runner.status == "accepted",
            "env_total": sum(item.credit.reward_breakdown.env for item in rewarded),
            "milestone_total": sum(
                item.credit.reward_breakdown.milestone for item in rewarded
            ),
            "violation_total": sum(
                item.credit.reward_breakdown.violation for item in rewarded
            ),
            "trend_total": sum(item.credit.reward_breakdown.trend for item in rewarded),
            "format_total": sum(
                item.credit.reward_breakdown.format for item in rewarded
            ),
            "cost_total": sum(item.credit.reward_breakdown.cost for item in rewarded),
            "total_reward": sum(
                item.credit.reward_breakdown.total for item in rewarded
            ),
        }
        model_payload = dict(base_payload)
        model_payload["actions"] = tuple(rewarded)
        return ExtractedRewardReplayResult(
            **model_payload,
            digest=canonical_digest(base_payload),
        )


__all__ = [
    "DEFAULT_EXTRACTED_REWARD_VERSION",
    "EXTRACTED_REPLAY_SCHEMA_VERSION",
    "EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION",
    "ExtractedRewardReplay",
    "ExtractedRewardReplayError",
    "ExtractedRewardReplayResult",
    "ExtractedRewardedAction",
]
