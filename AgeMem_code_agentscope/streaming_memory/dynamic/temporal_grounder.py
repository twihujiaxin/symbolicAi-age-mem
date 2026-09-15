"""Source/content/time grounding over the actual retained or exposed payload set."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import DynamicQueryPrivate, PrivateEvent, SourceRegistryRecord


VALIDATOR_VERSION = "agemem.dynamic.controlled_grounder.v1"


@dataclass(frozen=True)
class GroundingDecision:
    label: str
    source_ok: bool
    content_ok: bool | None
    temporal_ok: bool | None
    evidence_revision_ids: tuple[str, ...]
    reason_code: str
    validator_version: str = VALIDATOR_VERSION


@dataclass(frozen=True)
class SemanticQueryState:
    query_id: str
    coverage: float
    conflict: bool
    utility: float
    obligation_count: int
    supported_obligation_count: int
    decision: GroundingDecision


def _memory_claims(memory: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims = memory.get("claims") or []
    if isinstance(claims, list):
        return [item for item in claims if isinstance(item, dict)]
    return []


def _source_valid(
    memory: Mapping[str, Any],
    observed: set[str],
    registry: Mapping[str, SourceRegistryRecord],
) -> bool:
    refs = memory.get("source_refs") or []
    if not refs or any(ref not in observed or ref not in registry for ref in refs):
        return False
    return all(
        hashlib.sha256(registry[ref].text.encode("utf-8")).hexdigest()
        == registry[ref].text_sha256
        for ref in refs
    )


def _claim_matches_event(
    claim: Mapping[str, Any], event: PrivateEvent, content: str
) -> tuple[bool, bool]:
    semantic = (
        str(claim.get("entity", "")) == event.entity
        and str(claim.get("relation", "")) == event.relation
        and str(claim.get("value", "")) == event.value
    )
    # The actual body must carry the relation and value; a correct hidden claim
    # beside an unrelated body is not credited.
    body = event.entity in content and event.value in content
    relation_words = {
        "project_owner": ("负责人", "project_owner"),
        "member_team": ("所属团队", "member_team"),
        "site_region": ("所在区域", "site_region"),
    }.get(event.relation, (event.relation,))
    body = body and any(word in content for word in relation_words)
    content_ok = semantic and body
    start = claim.get("effective_at", claim.get("valid_from"))
    temporal_ok = start == event.effective_at
    if temporal_ok and claim.get("valid_to") is not None:
        temporal_ok = int(claim["valid_to"]) > int(start)
    return content_ok, temporal_ok


def _event_supported_by_memory(
    event: PrivateEvent,
    memory: Mapping[str, Any],
    observed: set[str],
    registry: Mapping[str, SourceRegistryRecord],
) -> tuple[bool, bool, bool]:
    source_ok = _source_valid(memory, observed, registry)
    if event.source_ref in registry:
        source_ok = (
            source_ok
            and registry[event.source_ref].text_sha256 == event.source_text_hash
        )
    if event.source_ref not in (memory.get("source_refs") or []):
        return source_ok, False, False
    content = str(memory.get("content", ""))
    results = [
        _claim_matches_event(claim, event, content) for claim in _memory_claims(memory)
    ]
    if results:
        return (
            source_ok,
            any(item[0] for item in results),
            any(a and b for a, b in results),
        )
    # Extractive control supports exact visible source text, but source pointer
    # alone never supplies a missing or contradicted body.
    source_text = (
        registry[event.source_ref].text if event.source_ref in registry else ""
    )
    content_ok = bool(source_text and source_text in content)
    return source_ok, content_ok, content_ok


def _has_temporal_conflict(
    memories: Sequence[Mapping[str, Any]], query: DynamicQueryPrivate
) -> bool:
    if query.query_kind == "start_time":
        return False
    time = query.query_time
    if query.query_kind == "current_state":
        time = 10**12
    values: set[str] = set()
    for memory in memories:
        for claim in _memory_claims(memory):
            if (
                claim.get("entity") != query.entity
                or claim.get("relation") != query.relation
            ):
                continue
            start = claim.get("effective_at", claim.get("valid_from"))
            end = claim.get("valid_to")
            if start is None:
                continue
            if int(start) <= int(time) and (end is None or int(time) < int(end)):
                values.add(str(claim.get("value")))
    return len(values) > 1


def validate_memory_semantics(
    memories: Sequence[Mapping[str, Any]],
    *,
    observed_source_refs: Sequence[str],
    source_registry: Mapping[str, SourceRegistryRecord],
    events_by_id: Mapping[str, PrivateEvent],
    query: DynamicQueryPrivate,
) -> SemanticQueryState:
    """Score best legal support alternative over the complete representation set."""

    observed = set(observed_source_refs)
    best_supported: set[str] = set()
    best_total = 1
    best_revisions: set[str] = set()
    any_source = False
    any_content = False
    any_temporal = False
    for alternative in query.support_alternatives:
        supported: set[str] = set()
        revisions: set[str] = set()
        for event_id in alternative:
            event = events_by_id[event_id]
            for memory in memories:
                source_ok, content_ok, temporal_ok = _event_supported_by_memory(
                    event, memory, observed, source_registry
                )
                any_source = any_source or source_ok
                any_content = any_content or content_ok
                any_temporal = any_temporal or temporal_ok
                if source_ok and content_ok and temporal_ok:
                    supported.add(event_id)
                    revisions.add(
                        str(
                            memory.get(
                                "revision_id", memory.get("memory_id", "unknown")
                            )
                        )
                    )
                    break
        if len(supported) / len(alternative) > len(best_supported) / best_total:
            best_supported = supported
            best_total = len(alternative)
            best_revisions = revisions
    coverage = len(best_supported) / best_total
    conflict = _has_temporal_conflict(memories, query)
    utility = coverage * (0.0 if conflict else 1.0)
    label = "supported" if utility == 1.0 else "unsupported"
    reason = (
        "complete_support"
        if label == "supported"
        else ("temporal_conflict" if conflict else "incomplete_or_invalid_support")
    )
    return SemanticQueryState(
        query_id=query.query_id,
        coverage=coverage,
        conflict=conflict,
        utility=utility,
        obligation_count=best_total,
        supported_obligation_count=len(best_supported),
        decision=GroundingDecision(
            label=label,
            source_ok=any_source,
            content_ok=any_content,
            temporal_ok=any_temporal,
            evidence_revision_ids=tuple(sorted(best_revisions)),
            reason_code=reason,
        ),
    )


def exposed_payloads_as_memories(payloads: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Parse only bytes actually placed in the reader prompt; truncation fails closed."""

    output = []
    for index, payload in enumerate(payloads):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = {"content": payload, "source_refs": [], "claims": []}
        if not isinstance(value, dict):
            continue
        value = dict(value)
        value.setdefault("revision_id", f"exposed-{index}")
        output.append(value)
    return tuple(output)


__all__ = [
    "GroundingDecision",
    "SemanticQueryState",
    "VALIDATOR_VERSION",
    "exposed_payloads_as_memories",
    "validate_memory_semantics",
]
