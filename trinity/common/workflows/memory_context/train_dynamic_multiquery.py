"""Validated publish boundary for dynamic v2 complete K-by-m groups."""

from __future__ import annotations

from trinity.common.dynamic_multiquery_contract import DynamicMemoryRolloutGroupBundle


def publish_complete_dynamic_group(
    bundle: DynamicMemoryRolloutGroupBundle,
) -> tuple[dict, ...]:
    bundle.validate_complete()
    return bundle.actor_batch()


__all__ = ["publish_complete_dynamic_group"]
