"""Read-only intermediate checkpoint probes for evaluation only."""

from __future__ import annotations

from typing import Mapping, Sequence

from .environment import DynamicCheckpoint
from .schema import DynamicQueryPrivate, PrivateEvent, SourceRegistryRecord
from .temporal_grounder import SemanticQueryState, validate_memory_semantics


def checkpoint_probe(
    checkpoint: DynamicCheckpoint,
    *,
    queries: Sequence[DynamicQueryPrivate],
    source_registry: Mapping[str, SourceRegistryRecord],
    events_by_id: Mapping[str, PrivateEvent],
) -> tuple[SemanticQueryState, ...]:
    """Fork evaluation from immutable bytes without mutating the ingest trajectory."""

    before = (checkpoint.memory_revision, checkpoint.memory_sha256)
    output = tuple(
        validate_memory_semantics(
            checkpoint.active_memories,
            observed_source_refs=checkpoint.observed_source_refs,
            source_registry=source_registry,
            events_by_id=events_by_id,
            query=query,
        )
        for query in queries
        if query.answer_available_at <= checkpoint.checkpoint_index
    )
    after = (checkpoint.memory_revision, checkpoint.memory_sha256)
    if before != after:
        raise RuntimeError("checkpoint probe mutated its source snapshot")
    return output


__all__ = ["checkpoint_probe"]
