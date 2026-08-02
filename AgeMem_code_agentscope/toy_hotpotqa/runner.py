"""Model-free M3 runner that records actions with the existing M1 recorder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from ..memory_store import MemoryStoreSnapshot
from ..trajectory import (
    MemorySnapshotItem,
    ToolCallSnapshot,
    ToolResultSnapshot,
    TrajectoryRecorder,
    TrajectoryStep,
    snapshot_memory,
)
from .environment import HotpotQAToyEnvironment, ToyEnvironmentPool
from .models import EpisodeStepResult, MemoryEpisode, ToyAction, ToyMemoryTask
from .policies import ToyPolicy


_TOOL_NAMES = {
    "add": "Add_memory",
    "update": "Update_memory",
    "retrieve": "Retrieve_memory",
    "delete": "Delete_memory",
    "advance": "Advance_stage",
    "answer": "Answer",
}


@dataclass(frozen=True)
class EpisodeRunResult:
    """Completed model-free rollout and its final M2 memory snapshot."""

    episode: MemoryEpisode
    steps: tuple[EpisodeStepResult, ...]
    final_memory: tuple[MemorySnapshotItem, ...]
    store_snapshot: MemoryStoreSnapshot


class ToyEpisodeRunner:
    """Execute a deterministic policy without importing or calling an LLM."""

    def __init__(self, pool: Optional[ToyEnvironmentPool] = None) -> None:
        self.pool = pool or ToyEnvironmentPool()

    @staticmethod
    def _trajectory_step(
        *,
        task: ToyMemoryTask,
        stage_observation: str,
        action: ToyAction,
        result: EpisodeStepResult,
        memory_before: List[MemorySnapshotItem],
        memory_after: List[MemorySnapshotItem],
    ) -> TrajectoryStep:
        call_id = f"{result.rollout_id}:call:{result.timestep}"
        action_payload = action.model_dump(mode="json", exclude_none=True)
        action_payload["seed"] = result.seed
        tool_name = _TOOL_NAMES[action.kind]
        tool_call = ToolCallSnapshot(
            id=call_id,
            name=tool_name,
            input=action_payload,
        )
        metadata = {
            "success": result.success,
            "task_id": result.task_id,
            "rollout_id": result.rollout_id,
            "seed": result.seed,
            "stage_before": result.stage_before,
            "stage_after": result.stage_after,
            "oracle_labels": result.labels.model_dump(mode="json"),
            "episode_success": result.episode_success,
            "task_reward": result.task_reward,
        }
        tool_result = ToolResultSnapshot(
            tool_call_id=call_id,
            name=tool_name,
            content=[{"type": "text", "text": result.message}],
            metadata=metadata,
        )
        return TrajectoryStep(
            task_id=task.task_id,
            rollout_id=result.rollout_id,
            stage=result.stage_before,
            timestep=result.timestep,
            observation=stage_observation,
            action_text=json.dumps(
                action_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            tool_calls=[tool_call],
            tool_results=[tool_result],
            memory_before=memory_before,
            memory_after=memory_after,
            env_reward=result.task_reward,
            done=result.done,
        )

    async def run(
        self,
        task: ToyMemoryTask,
        policy: ToyPolicy,
        *,
        rollout_id: str,
        seed: int,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> EpisodeRunResult:
        environment = HotpotQAToyEnvironment(
            task,
            rollout_id=rollout_id,
            seed=seed,
            pool=self.pool,
        )
        results: List[EpisodeStepResult] = []
        for action in policy.actions(task, seed):
            if environment.done:
                break
            public_input = environment.stage_input()
            memory_before = snapshot_memory(environment.memory)
            result = await environment.step(action)
            memory_after = snapshot_memory(environment.memory)
            results.append(result)
            if recorder is not None:
                recorder.record(
                    self._trajectory_step(
                        task=task,
                        stage_observation=public_input.observation,
                        action=action,
                        result=result,
                        memory_before=memory_before,
                        memory_after=memory_after,
                    )
                )

        final_memory = tuple(snapshot_memory(environment.memory))
        return EpisodeRunResult(
            episode=environment.episode(),
            steps=tuple(results),
            final_memory=final_memory,
            store_snapshot=environment.memory.snapshot(),
        )


__all__ = ["EpisodeRunResult", "ToyEpisodeRunner"]
