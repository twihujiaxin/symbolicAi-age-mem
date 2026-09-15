"""P5 publication contract for ``streaming_dynamic_multiquery_v2``.

It reuses v1 ingest samples but keeps dynamic reward identities separate.  No
GPU runtime is initialized here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from trinity.common.streaming_multiquery_contract import ReadActorSample


DYNAMIC_CONTRACT_VERSION = "agemem.dynamic.bundle.v2"


class DynamicBundleError(ValueError):
    pass


@dataclass(frozen=True)
class DynamicQueryBranchScore:
    query_branch_id: str
    query_id: str
    task_score: float
    reader_version: str
    reader_token_count: int
    actor_loss_token_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.task_score <= 1.0:
            raise DynamicBundleError("task_score must be in [0,1]")
        if self.actor_loss_token_count:
            raise DynamicBundleError("fixed reader tokens are forbidden in actor loss")


@dataclass(frozen=True)
class DynamicReadRollout:
    read_rollout_id: str
    snapshot_id: str
    shared_prefix_id: str
    policy_version: str
    samples: tuple[ReadActorSample, ...]
    branches: tuple[DynamicQueryBranchScore, ...]
    task_reward: float
    semantic_component: float
    total_reward: float
    reward_profile: str
    failed: bool = False

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.task_reward, self.semantic_component, self.total_reward)
        ):
            raise DynamicBundleError("reward components must be finite")
        if any(item.shared_prefix_id != self.shared_prefix_id for item in self.samples):
            raise DynamicBundleError("shared-prefix identity mismatch")
        if any(item.policy_version != self.policy_version for item in self.samples):
            raise DynamicBundleError("policy version mismatch")


@dataclass(frozen=True)
class DynamicMemoryRolloutGroupBundle:
    group_id: str
    history_family_id: str
    expected_rollouts: int
    expected_queries_per_rollout: int
    rollouts: tuple[DynamicReadRollout, ...]
    reward_version: str
    reward_profile: str
    lambda_semantic: float
    std_ddof: int = 0
    contract_version: str = DYNAMIC_CONTRACT_VERSION

    def validate_complete(self) -> None:
        if self.contract_version != DYNAMIC_CONTRACT_VERSION:
            raise DynamicBundleError("unsupported dynamic bundle contract")
        if self.std_ddof != 0:
            raise DynamicBundleError("dynamic v2 inherits population std (ddof=0)")
        if len(self.rollouts) != self.expected_rollouts:
            raise DynamicBundleError("incomplete K rollout group")
        if self.reward_profile not in {"V2_T", "V2_END", "V2_LIFE"}:
            raise DynamicBundleError(
                "monitor alias is offline-only and cannot publish to GPU"
            )
        if any(item.reward_profile != self.reward_profile for item in self.rollouts):
            raise DynamicBundleError("mixed reward profiles")
        if len({item.read_rollout_id for item in self.rollouts}) != len(self.rollouts):
            raise DynamicBundleError("duplicate read_rollout_id")
        if len({item.snapshot_id for item in self.rollouts}) != len(self.rollouts):
            raise DynamicBundleError("snapshot IDs must be unique per read rollout")
        if len({item.policy_version for item in self.rollouts}) != 1:
            raise DynamicBundleError("one group cannot mix policy versions")
        action_ids = [
            item.action_id for rollout in self.rollouts for item in rollout.samples
        ]
        if len(action_ids) != len(set(action_ids)):
            raise DynamicBundleError("shared ingest actions were duplicated")
        query_ids = None
        for rollout in self.rollouts:
            if rollout.failed:
                raise DynamicBundleError("failed rollout cannot enter the buffer")
            if len(rollout.branches) != self.expected_queries_per_rollout:
                raise DynamicBundleError("incomplete m query branches")
            current = {item.query_id for item in rollout.branches}
            if len(current) != len(rollout.branches):
                raise DynamicBundleError("duplicate queries cannot fill m")
            if query_ids is None:
                query_ids = current
            elif current != query_ids:
                raise DynamicBundleError("K snapshots must answer the same query set")
            task_mean = sum(item.task_score for item in rollout.branches) / len(
                rollout.branches
            )
            if abs(task_mean - rollout.task_reward) > 1e-9:
                raise DynamicBundleError(
                    "m branch scores must average before K normalization"
                )
            expected = (
                rollout.task_reward + self.lambda_semantic * rollout.semantic_component
            )
            if abs(expected - rollout.total_reward) > 1e-9:
                raise DynamicBundleError(
                    "dynamic reward components do not join exactly"
                )

    def advantages(self, epsilon: float = 1e-8) -> tuple[float, ...]:
        self.validate_complete()
        values = [item.total_reward for item in self.rollouts]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))
        if std == 0.0:
            return tuple(0.0 for _ in values)
        return tuple((item - mean) / (std + epsilon) for item in values)

    def actor_batch(self) -> tuple[dict[str, Any], ...]:
        advantages = self.advantages()
        rows = []
        for rollout, advantage in zip(self.rollouts, advantages):
            for sample in rollout.samples:
                rows.append(
                    {
                        "group_id": self.group_id,
                        "history_family_id": self.history_family_id,
                        "read_rollout_id": rollout.read_rollout_id,
                        "shared_prefix_id": sample.shared_prefix_id,
                        "action_id": sample.action_id,
                        "token_ids": list(sample.token_ids),
                        "old_logprobs": list(sample.old_logprobs),
                        "action_mask": list(sample.action_mask),
                        "advantage": advantage,
                        "policy_version": sample.policy_version,
                        "phase": "ingest",
                        "reward_profile": self.reward_profile,
                        "reward_version": self.reward_version,
                    }
                )
        return tuple(rows)


__all__ = [
    "DYNAMIC_CONTRACT_VERSION",
    "DynamicBundleError",
    "DynamicMemoryRolloutGroupBundle",
    "DynamicQueryBranchScore",
    "DynamicReadRollout",
]
