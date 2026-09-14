"""Atomic S4 buffer contract for ``streaming_multiquery_v1``.

This module contains no Ray/vLLM initialization. A producer publishes exactly
one complete history group; the adapter emits only unique ingest-policy samples.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence


CONTRACT_VERSION = "agemem.stream_mq.bundle.v1"


class StreamingBundleError(ValueError):
    pass


@dataclass(frozen=True)
class QueryBranchScore:
    query_branch_id: str
    query_id: str
    terminal_f1: float
    reader_version: str
    reader_token_count: int
    actor_loss_token_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.terminal_f1 <= 1.0:
            raise StreamingBundleError("terminal_f1 must be in [0,1]")
        if self.actor_loss_token_count != 0:
            raise StreamingBundleError("reader tokens are forbidden in actor loss")


@dataclass(frozen=True)
class ReadActorSample:
    action_id: str
    shared_prefix_id: str
    token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    action_mask: tuple[bool, ...]
    policy_version: str
    phase: str = "ingest"

    def __post_init__(self) -> None:
        if self.phase != "ingest":
            raise StreamingBundleError("only ingest samples are trainable in v1")
        if not self.token_ids or len(self.token_ids) != len(self.old_logprobs):
            raise StreamingBundleError("token_ids/old_logprobs must be non-empty and aligned")
        if len(self.action_mask) != len(self.token_ids) or not any(self.action_mask):
            raise StreamingBundleError("action mask must align and select generated reader actions")
        if not all(math.isfinite(value) for value in self.old_logprobs):
            raise StreamingBundleError("old logprobs must be finite")


@dataclass(frozen=True)
class ReadRollout:
    read_rollout_id: str
    snapshot_id: str
    shared_prefix_id: str
    policy_version: str
    samples: tuple[ReadActorSample, ...]
    branches: tuple[QueryBranchScore, ...]
    reward: float
    failed: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise StreamingBundleError("rollout reward must be finite")
        if any(item.shared_prefix_id != self.shared_prefix_id for item in self.samples):
            raise StreamingBundleError("read sample shared-prefix identity mismatch")
        if any(item.policy_version != self.policy_version for item in self.samples):
            raise StreamingBundleError("read sample policy version mismatch")


@dataclass(frozen=True)
class MemoryRolloutGroupBundle:
    group_id: str
    history_family_id: str
    expected_rollouts: int
    expected_queries_per_rollout: int
    rollouts: tuple[ReadRollout, ...]
    reward_version: str
    std_ddof: int = 0
    contract_version: str = CONTRACT_VERSION

    def validate_complete(self) -> None:
        if self.contract_version != CONTRACT_VERSION:
            raise StreamingBundleError("unsupported bundle contract")
        if self.std_ddof != 0:
            raise StreamingBundleError("streaming v1 requires population std (ddof=0)")
        if len(self.rollouts) != self.expected_rollouts:
            raise StreamingBundleError("incomplete K rollout group")
        if len({item.read_rollout_id for item in self.rollouts}) != len(self.rollouts):
            raise StreamingBundleError("duplicate read_rollout_id")
        if len({item.snapshot_id for item in self.rollouts}) != len(self.rollouts):
            raise StreamingBundleError("snapshot IDs must be unique per read rollout")
        policy_versions = {item.policy_version for item in self.rollouts}
        if len(policy_versions) != 1:
            raise StreamingBundleError("one group cannot mix policy versions")
        action_ids = [sample.action_id for rollout in self.rollouts for sample in rollout.samples]
        if len(action_ids) != len(set(action_ids)):
            raise StreamingBundleError("shared ingest actions were duplicated")
        query_ids: set[str] | None = None
        for rollout in self.rollouts:
            if rollout.failed:
                raise StreamingBundleError("failed rollout requires explicit settlement before publish")
            if len(rollout.branches) != self.expected_queries_per_rollout:
                raise StreamingBundleError("incomplete m query branches")
            current = {branch.query_id for branch in rollout.branches}
            if len(current) != len(rollout.branches):
                raise StreamingBundleError("duplicate queries cannot be used to fill m")
            if query_ids is None:
                query_ids = current
            elif current != query_ids:
                raise StreamingBundleError("all K snapshots must answer the same query set")
            mean_f1 = sum(item.terminal_f1 for item in rollout.branches) / len(rollout.branches)
            if abs(mean_f1 - rollout.reward) > 1e-9:
                raise StreamingBundleError("branch scores must aggregate before the K-group reward")

    def advantages(self, epsilon: float = 1e-8) -> tuple[float, ...]:
        self.validate_complete()
        rewards = [item.reward for item in self.rollouts]
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        std = math.sqrt(variance)
        if std == 0.0:
            return tuple(0.0 for _ in rewards)
        return tuple((value - mean) / (std + epsilon) for value in rewards)

    def actor_batch(self) -> tuple[dict[str, Any], ...]:
        """Flatten unique read actions once; query branches are deliberately absent."""

        advantages = self.advantages()
        output = []
        for rollout, advantage in zip(self.rollouts, advantages):
            for sample in rollout.samples:
                output.append({
                    "group_id": self.group_id,
                    "read_rollout_id": rollout.read_rollout_id,
                    "shared_prefix_id": sample.shared_prefix_id,
                    "action_id": sample.action_id,
                    "token_ids": list(sample.token_ids),
                    "old_logprobs": list(sample.old_logprobs),
                    "action_mask": list(sample.action_mask),
                    "advantage": advantage,
                    "policy_version": sample.policy_version,
                    "phase": "ingest",
                })
        return tuple(output)

    def receipt(self) -> dict[str, Any]:
        self.validate_complete()
        rewards = [item.reward for item in self.rollouts]
        advantages = self.advantages()
        payload = {
            "schema_version": "agemem.stream_mq.receipt.v1",
            "protocol": "streaming_multiquery_v1",
            "contract_version": self.contract_version,
            "group_id": self.group_id,
            "history_family_id": self.history_family_id,
            "complete_bundle_count": 1,
            "read_rollout_count": len(self.rollouts),
            "query_branch_count": sum(len(item.branches) for item in self.rollouts),
            "unique_shared_prefix_count": len({item.shared_prefix_id for item in self.rollouts}),
            "actor_sample_count": sum(len(item.samples) for item in self.rollouts),
            "reader_actor_loss_tokens": 0,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std_ddof0": math.sqrt(
                sum((value - sum(rewards) / len(rewards)) ** 2 for value in rewards)
                / len(rewards)
            ),
            "nonzero_advantage_rollouts": sum(abs(value) > 0 for value in advantages),
            "reward_version": self.reward_version,
            "policy_version": self.rollouts[0].policy_version,
            "reader_versions": sorted({b.reader_version for r in self.rollouts for b in r.branches}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "CONTRACT_VERSION", "MemoryRolloutGroupBundle", "QueryBranchScore",
    "ReadActorSample", "ReadRollout", "StreamingBundleError",
]
