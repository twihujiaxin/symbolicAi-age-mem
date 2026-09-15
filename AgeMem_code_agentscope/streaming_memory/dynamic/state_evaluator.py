"""Direct semantic state evaluator for dynamic checkpoint traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .environment import DynamicCheckpoint
from .schema import DynamicQueryPrivate, PrivateEvent, SourceRegistryRecord
from .temporal_grounder import SemanticQueryState, validate_memory_semantics


@dataclass(frozen=True)
class CheckpointSemanticFrame:
    checkpoint_id: str
    checkpoint_index: int
    by_query: tuple[SemanticQueryState, ...]


@dataclass(frozen=True)
class DirectEvaluation:
    backend: str
    checkpoint_states: tuple[tuple[str, int, str, str, float, bool], ...]
    checkpoint_utilities: tuple[tuple[str, int, float], ...]
    final_utility_by_query: tuple[tuple[str, float], ...]


def build_semantic_frames(
    checkpoints: Sequence[DynamicCheckpoint],
    *,
    queries: Sequence[DynamicQueryPrivate],
    source_registry: Mapping[str, SourceRegistryRecord],
    events_by_id: Mapping[str, PrivateEvent],
) -> tuple[CheckpointSemanticFrame, ...]:
    return tuple(
        CheckpointSemanticFrame(
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_index=checkpoint.checkpoint_index,
            by_query=tuple(
                validate_memory_semantics(
                    checkpoint.active_memories,
                    observed_source_refs=checkpoint.observed_source_refs,
                    source_registry=source_registry,
                    events_by_id=events_by_id,
                    query=query,
                )
                for query in queries
            ),
        )
        for checkpoint in checkpoints
    )


def _state_name(item: SemanticQueryState) -> str:
    if item.conflict:
        return "conflicting"
    if item.coverage == 1.0:
        return "supported"
    if item.coverage > 0.0:
        return "partial"
    return "absent"


class DirectStateEvaluator:
    version = "agemem.dynamic.direct_state.v1"

    def evaluate(self, frames: Sequence[CheckpointSemanticFrame]) -> DirectEvaluation:
        states = []
        utilities = []
        latest: dict[str, float] = {}
        for frame in frames:
            for item in frame.by_query:
                state = _state_name(item)
                states.append(
                    (
                        frame.checkpoint_id,
                        frame.checkpoint_index,
                        item.query_id,
                        state,
                        item.coverage,
                        item.conflict,
                    )
                )
                utilities.append((item.query_id, frame.checkpoint_index, item.utility))
                latest[item.query_id] = item.utility
        return DirectEvaluation(
            backend=self.version,
            checkpoint_states=tuple(states),
            checkpoint_utilities=tuple(utilities),
            final_utility_by_query=tuple(sorted(latest.items())),
        )


__all__ = [
    "CheckpointSemanticFrame",
    "DirectEvaluation",
    "DirectStateEvaluator",
    "build_semantic_frames",
]
