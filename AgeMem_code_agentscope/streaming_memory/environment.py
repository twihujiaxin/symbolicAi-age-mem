"""CPU streaming environment with hard C/B budgets and immutable snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .schema import PublicChunk, StreamEpisodePublic
from .token_budget import BudgetError, TokenAccounting, memory_payload_tokens


INGEST_SYSTEM = (
    "Read the stream and manage long-term memory. The future questions are hidden. "
    "Return exactly one action: ADD, UPDATE, DELETE, SUMMARY, CLEAR, or NEXT."
)


class StreamingEnvironmentError(ValueError):
    pass


@dataclass(frozen=True)
class EnvironmentEvent:
    event_id: str
    kind: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ActionResult:
    admitted: bool
    code: str
    message: str
    memory_tokens: int
    context_tokens: int


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_id: str
    episode_id: str
    read_rollout_id: str
    active_memories: tuple[dict[str, Any], ...]
    context_tail: tuple[dict[str, str], ...]
    observed_sources: tuple[tuple[str, int], ...]
    memory_sha256: str
    tail_sha256: str
    policy_version: str
    protocol: str = "streaming_multiquery_v1"


@dataclass
class _ContextGroup:
    group_id: str
    messages: list[dict[str, str]]
    source_keys: set[tuple[str, int]] = field(default_factory=set)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class StreamingMemoryEnvironment:
    """One ingest rollout. Private questions/gold are intentionally absent."""

    def __init__(
        self,
        episode: StreamEpisodePublic,
        *,
        accounting: TokenAccounting,
        read_rollout_id: str,
        policy_version: str,
        ingest_max_new_tokens: int,
        decisions_per_chunk: int = 2,
        answer_tail_tokens: int = 512,
    ) -> None:
        if not read_rollout_id or not policy_version:
            raise StreamingEnvironmentError("rollout and policy version are required")
        self.episode = episode
        self.accounting = accounting
        self.read_rollout_id = read_rollout_id
        self.policy_version = policy_version
        self.ingest_max_new_tokens = ingest_max_new_tokens
        self.decisions_per_chunk = decisions_per_chunk
        self.answer_tail_tokens = answer_tail_tokens
        self.memories: dict[str, dict[str, Any]] = {}
        self.memory_versions: dict[str, list[dict[str, Any]]] = {}
        self.context: list[_ContextGroup] = []
        self.observed_sources: set[tuple[str, int]] = set()
        self.events: list[EnvironmentEvent] = []
        self._next_chunk = 0
        self._decisions = 0
        self._current_chunk: PublicChunk | None = None

    def _handles(self) -> str:
        handles = [
            {"memory_id": key, "title": value.get("title"), "tags": value.get("tags", [])}
            for key, value in sorted(self.memories.items())
        ]
        return json.dumps(handles, ensure_ascii=False, separators=(",", ":"))

    def _messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": INGEST_SYSTEM}]
        for group in self.context:
            messages.extend(copy.deepcopy(group.messages))
        messages.append({
            "role": "user",
            "content": (
                f"Memory handles: {self._handles()}\n"
                f"Memory budget: {self.memory_tokens()}/{self.episode.memory_budget_tokens}."
            ),
        })
        return messages

    def _context_tokens(self) -> int:
        return self.accounting.count_chat(self._messages())

    def _fit_context(self) -> None:
        while True:
            try:
                self.accounting.enforce_context(
                    self._messages(), max_new_tokens=self.ingest_max_new_tokens,
                    context_total_tokens=self.episode.context_budget_tokens,
                )
                return
            except BudgetError:
                removable = next((g for g in self.context if g is not self.context[-1]), None)
                if removable is None:
                    raise StreamingEnvironmentError(
                        "current chunk plus fixed instructions cannot satisfy C"
                    )
                before = self._context_tokens()
                self.context.remove(removable)
                after = self._context_tokens()
                self.events.append(EnvironmentEvent(
                    event_id=f"env-{len(self.events):06d}", kind="fifo_context_eviction",
                    details={"group_id": removable.group_id, "tokens_removed": before - after,
                             "reason": "reserve_current_chunk_and_response"},
                ))

    def admit_next_chunk(self) -> list[dict[str, str]]:
        if self._next_chunk >= len(self.episode.chunks):
            raise StopIteration("all chunks were already shown")
        chunk = self.episode.chunks[self._next_chunk]
        group = _ContextGroup(
            group_id=chunk.chunk_id,
            messages=[{"role": "user", "content": f"STREAM CHUNK\n{chunk.text}"}],
        )
        for source in chunk.sources:
            for sentence_index in range(source.sentence_start, source.sentence_end + 1):
                group.source_keys.add((source.document_key, sentence_index))
        self.context.append(group)
        self.observed_sources.update(group.source_keys)
        self._current_chunk = chunk
        self._next_chunk += 1
        self._decisions = 0
        self._fit_context()
        # This render occurs before any action, proving successful admission was visible.
        messages = self._messages()
        self.accounting.enforce_context(
            messages, max_new_tokens=self.ingest_max_new_tokens,
            context_total_tokens=self.episode.context_budget_tokens,
        )
        return messages

    def memory_tokens(self, memories: Sequence[Mapping[str, Any]] | None = None) -> int:
        return memory_payload_tokens(
            self.memories.values() if memories is None else memories, self.accounting
        )

    def _validate_sources(self, source_refs: Sequence[Mapping[str, Any]]) -> None:
        for ref in source_refs:
            key = (str(ref.get("document_key", "")), int(ref.get("sentence_index", -1)))
            if key not in self.observed_sources:
                raise StreamingEnvironmentError("source_not_observed")

    def _admit_memories(self, proposed: Mapping[str, Mapping[str, Any]]) -> int:
        value = self.memory_tokens(list(proposed.values()))
        if value > self.episode.memory_budget_tokens:
            raise StreamingEnvironmentError("memory_budget_exceeded")
        return value

    def execute(self, action: Mapping[str, Any]) -> ActionResult:
        if self._current_chunk is None:
            raise StreamingEnvironmentError("a chunk must be admitted before acting")
        if self._decisions >= self.decisions_per_chunk:
            raise StreamingEnvironmentError("decision_limit_reached")
        self._decisions += 1
        action_type = str(action.get("type", "")).upper()
        if action_type not in {"ADD", "UPDATE", "DELETE", "SUMMARY", "CLEAR", "NEXT"}:
            return self._result(False, "invalid_action", "ERROR invalid_action")
        try:
            if action_type == "ADD":
                self._add(action)
            elif action_type == "UPDATE":
                self._update(action)
            elif action_type == "DELETE":
                self._delete(action)
            elif action_type == "SUMMARY":
                self._summary(action)
            elif action_type == "CLEAR":
                self._clear(action)
            elif action_type == "NEXT":
                self._decisions = self.decisions_per_chunk
            self._fit_context()
        except (StreamingEnvironmentError, TypeError, ValueError) as exc:
            return self._result(False, str(exc), f"ERROR {exc}")
        return self._result(True, "ok", "OK")

    def _result(self, admitted: bool, code: str, message: str) -> ActionResult:
        self.context.append(_ContextGroup(
            group_id=f"receipt-{len(self.events):06d}",
            messages=[{"role": "tool", "content": message}],
        ))
        self._fit_context()
        result = ActionResult(
            admitted=admitted, code=code, message=message,
            memory_tokens=self.memory_tokens(), context_tokens=self._context_tokens(),
        )
        self.events.append(EnvironmentEvent(
            event_id=f"env-{len(self.events):06d}", kind="policy_action_result",
            details={"admitted": admitted, "code": code},
        ))
        return result

    def _payload(self, action: Mapping[str, Any]) -> dict[str, Any]:
        content = action.get("content")
        if not isinstance(content, str) or not content.strip():
            raise StreamingEnvironmentError("content_required")
        refs = action.get("source_refs") or []
        if not isinstance(refs, list) or not refs:
            raise StreamingEnvironmentError("source_refs_required")
        self._validate_sources(refs)
        return {
            "content": content.strip(), "title": action.get("title"),
            "tags": list(action.get("tags") or []), "source_refs": copy.deepcopy(refs),
            "custom": copy.deepcopy(action.get("custom") or {}),
        }

    def _add(self, action: Mapping[str, Any]) -> None:
        memory_id = str(action.get("memory_id", ""))
        if not memory_id or memory_id in self.memories:
            raise StreamingEnvironmentError("invalid_or_duplicate_memory_id")
        payload = self._payload(action)
        proposed = dict(self.memories)
        proposed[memory_id] = payload
        self._admit_memories(proposed)
        self.memories = proposed
        self.memory_versions[memory_id] = [copy.deepcopy(payload)]

    def _update(self, action: Mapping[str, Any]) -> None:
        memory_id = str(action.get("memory_id", ""))
        if memory_id not in self.memories:
            raise StreamingEnvironmentError("memory_not_found")
        payload = self._payload(action)
        proposed = dict(self.memories)
        proposed[memory_id] = payload
        self._admit_memories(proposed)
        self.memory_versions[memory_id].append(copy.deepcopy(payload))
        self.memories = proposed

    def _delete(self, action: Mapping[str, Any]) -> None:
        memory_id = str(action.get("memory_id", ""))
        if memory_id not in self.memories:
            raise StreamingEnvironmentError("memory_not_found")
        proposed = dict(self.memories)
        removed = proposed.pop(memory_id)
        self._admit_memories(proposed)
        self.memory_versions[memory_id].append({"deleted": True, **copy.deepcopy(removed)})
        self.memories = proposed

    def _context_target(self, action: Mapping[str, Any]) -> _ContextGroup:
        group_id = str(action.get("context_group_id", ""))
        target = next((group for group in self.context if group.group_id == group_id), None)
        if target is None:
            raise StreamingEnvironmentError("context_group_not_visible")
        return target

    def _summary(self, action: Mapping[str, Any]) -> None:
        target = self._context_target(action)
        replacement = action.get("replacement_text")
        if not isinstance(replacement, str) or not replacement.strip():
            raise StreamingEnvironmentError("replacement_text_required")
        target.messages = [{"role": "user", "content": f"POLICY SUMMARY\n{replacement.strip()}"}]

    def _clear(self, action: Mapping[str, Any]) -> None:
        target = self._context_target(action)
        self.context.remove(target)

    def finalize(self) -> MemorySnapshot:
        if self._next_chunk != len(self.episode.chunks):
            raise StreamingEnvironmentError("cannot snapshot before every chunk was shown")
        memories = tuple(
            {"memory_id": key, **copy.deepcopy(value)}
            for key, value in sorted(self.memories.items())
        )
        tail_groups: list[_ContextGroup] = []
        tail_tokens = 0
        for group in reversed(self.context):
            cost = sum(self.accounting.count_text(m["content"]) for m in group.messages)
            if tail_tokens + cost > self.answer_tail_tokens:
                continue
            tail_groups.append(group)
            tail_tokens += cost
        tail = tuple(
            copy.deepcopy(message)
            for group in reversed(tail_groups)
            for message in group.messages
        )
        memory_sha, tail_sha = _digest(memories), _digest(tail)
        snapshot_id = f"snapshot-{_digest([self.read_rollout_id, memory_sha, tail_sha])[:20]}"
        return MemorySnapshot(
            snapshot_id=snapshot_id, episode_id=self.episode.episode_id,
            read_rollout_id=self.read_rollout_id, active_memories=memories,
            context_tail=tail, observed_sources=tuple(sorted(self.observed_sources)),
            memory_sha256=memory_sha, tail_sha256=tail_sha,
            policy_version=self.policy_version,
        )


__all__ = [
    "ActionResult", "EnvironmentEvent", "MemorySnapshot",
    "StreamingEnvironmentError", "StreamingMemoryEnvironment",
]
