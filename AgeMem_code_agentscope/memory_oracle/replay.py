"""Offline M4 trajectory replay with deterministic DFA reward attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..toy_hotpotqa.models import ToyMemoryTask
from ..trajectory import TrajectoryReplay, TrajectoryStep
from .automaton import DFARunner, hand_authored_memory_dfa
from .grounder import MemoryOracleGrounder
from .models import (
    AutomatonSpec,
    RewardBreakdown,
    RewardConfig,
    RewardProfile,
    RewardReplayResult,
    RewardedTrajectoryStep,
)


def default_reward_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "m4_reward.json"


class OfflineRewardReplay:
    """Replay strict M1 JSONL using only M3 metadata and deterministic code."""

    def __init__(
        self,
        profile: RewardProfile,
        *,
        spec: Optional[AutomatonSpec] = None,
    ) -> None:
        self.profile = profile.model_copy(deep=True)
        self.spec = (spec or hand_authored_memory_dfa()).model_copy(deep=True)

    @classmethod
    def from_config(
        cls,
        profile_name: str = "terminal_dfa",
        *,
        path: Optional[str | Path] = None,
        spec: Optional[AutomatonSpec] = None,
    ) -> "OfflineRewardReplay":
        config = RewardConfig.from_json(path or default_reward_config_path())
        return cls(config.profile(profile_name), spec=spec)

    def replay_steps(
        self,
        steps: Iterable[TrajectoryStep],
        *,
        task: ToyMemoryTask,
        rollout_id: str,
    ) -> RewardReplayResult:
        source = TrajectoryReplay(steps).replay(
            task_id=task.task_id,
            rollout_id=rollout_id,
            require_complete=False,
        )
        grounder = MemoryOracleGrounder(task)
        runner = DFARunner(self.spec, max_steps=self.profile.max_steps)
        rewarded_steps = []
        seed: Optional[int] = None

        for step in source.steps:
            event = grounder.from_step(step)
            if seed is None:
                seed = event.seed
            elif event.seed != seed:
                raise ValueError("all trajectory steps must use the same seed")
            transition = runner.step(event, done=step.done)
            env_reward = self.profile.env_weight * step.env_reward
            milestone_reward = (
                self.profile.milestone_weight
                * len(transition.new_progress_edges)
            )
            violation_reward = (
                self.profile.violation_weight * len(transition.violations)
            )
            format_reward = 0.0 * self.profile.format_weight
            total = (
                env_reward
                + self.profile.logic_beta * milestone_reward
                + violation_reward
                + format_reward
            )
            breakdown = RewardBreakdown(
                task_id=step.task_id,
                rollout_id=step.rollout_id,
                seed=event.seed,
                timestep=step.timestep,
                env=env_reward,
                milestone=milestone_reward,
                violation=violation_reward,
                trend=0.0,
                format=format_reward,
                total=total,
                automaton_state_before=transition.state_before,
                automaton_state_after=transition.state_after,
                automaton_status=transition.status,
                propositions=event.propositions,
                fired_edges=transition.fired_edges,
                newly_rewarded_edges=transition.new_progress_edges,
                violation_edges=transition.violations,
            )
            rewarded_steps.append(
                RewardedTrajectoryStep(event=event, reward=breakdown)
            )

        if seed is None:  # TrajectoryReplay already rejects an empty rollout.
            raise ValueError("trajectory contains no rewardable steps")
        env_total = sum(item.reward.env for item in rewarded_steps)
        milestone_total = sum(item.reward.milestone for item in rewarded_steps)
        violation_total = sum(item.reward.violation for item in rewarded_steps)
        format_total = sum(item.reward.format for item in rewarded_steps)
        total_reward = sum(item.reward.total for item in rewarded_steps)
        base_payload = {
            "schema_version": 1,
            "task_id": task.task_id,
            "rollout_id": rollout_id,
            "seed": seed,
            "profile": self.profile.name,
            "source_trajectory_digest": source.digest,
            "steps": [item.model_dump(mode="json") for item in rewarded_steps],
            "final_state": runner.state,
            "final_status": runner.status,
            "accepted": runner.status == "accepted",
            "env_total": env_total,
            "milestone_total": milestone_total,
            "violation_total": violation_total,
            "format_total": format_total,
            "total_reward": total_reward,
        }
        digest = RewardReplayResult.digest_payload(base_payload)
        return RewardReplayResult(**base_payload, digest=digest)

    def replay_jsonl(
        self,
        path: str | Path,
        *,
        task: ToyMemoryTask,
        rollout_id: str,
        output_path: Optional[str | Path] = None,
    ) -> RewardReplayResult:
        trajectory = TrajectoryReplay.from_jsonl(path)
        result = self.replay_steps(
            trajectory.query(task_id=task.task_id, rollout_id=rollout_id),
            task=task,
            rollout_id=rollout_id,
        )
        if output_path is not None:
            result.write_jsonl(output_path)
        return result


__all__ = ["OfflineRewardReplay", "default_reward_config_path"]
