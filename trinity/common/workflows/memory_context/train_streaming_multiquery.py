"""S4 adapter boundary for the new streaming multi-query workflow.

The actual GPU producer is intentionally deferred to S5. This boundary rejects
partial groups and returns the exact ingest-only records a Trinity operator may
convert to ``Experience`` objects; no legacy three-stage behavior is changed.
"""

from __future__ import annotations

from trinity.common.streaming_multiquery_contract import MemoryRolloutGroupBundle


def publish_complete_streaming_group(bundle: MemoryRolloutGroupBundle) -> tuple[dict, ...]:
    bundle.validate_complete()
    return bundle.actor_batch()


__all__ = ["publish_complete_streaming_group"]
