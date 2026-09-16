"""Versioned dynamic memory environment with strict C/B and private audit state."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..token_budget import BudgetError, TokenAccounting
from .schema import DynamicHistoryPublic, PROTOCOL_VERSION, project_public_observation


DYNAMIC_INGEST_SYSTEM = (
    "Read a changing information stream and maintain useful long-term memory. "
    "Future questions are hidden. Times and historical values can both matter. "
    "Use exactly one public action per response: ADD, UPDATE, DELETE, RETRIEVE, "
    "or NEXT. Return only "
    "<tool_call>[{\"name\":\"ACTION\",\"arguments\":{...}}]</tool_call>. "
    "ADD and UPDATE arguments may contain memory_id, content, title, tags, "
    "source_refs, claims, and custom. DELETE and RETRIEVE require memory_id; "
    "NEXT uses an empty arguments object."
)


class DynamicEnvironmentError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def canonical_dynamic_payload(payload: Mapping[str, Any]) -> str:
    """Serialize every policy-recoverable field; opaque control IDs are free."""

    return _canonical(
        {
            "content": payload.get("content", ""),
            "title": payload.get("title"),
            "tags": payload.get("tags") or [],
            "source_refs": payload.get("source_refs") or [],
            "claims": payload.get("claims") or [],
            "custom": payload.get("custom") or {},
        }
    )


@dataclass(frozen=True)
class DynamicActionResult:
    admitted: bool
    code: str
    message: str
    memory_tokens: int
    context_tokens: int
    memory_revision_before: int
    memory_revision_after: int
    displayed_payload: str = ""
    displayed_revision_id: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class DynamicCheckpoint:
    checkpoint_id: str
    checkpoint_index: int
    memory_revision: int
    active_memories: tuple[dict[str, Any], ...]
    observed_source_refs: tuple[str, ...]
    memory_sha256: str


@dataclass(frozen=True)
class DynamicMemorySnapshot:
    snapshot_id: str
    history_id: str
    read_rollout_id: str
    active_memories: tuple[dict[str, Any], ...]
    context_tail: tuple[dict[str, str], ...]
    observed_source_refs: tuple[str, ...]
    memory_sha256: str
    tail_sha256: str
    memory_revision: int
    policy_version: str
    protocol_version: str = PROTOCOL_VERSION


@dataclass(frozen=True)
class AuditRevision:
    audit_event_id: str
    mutation_type: str
    mutation_success: bool
    memory_id: str | None
    revision_id_before: str | None
    revision_id_after: str | None
    memory_revision_before: int
    memory_revision_after: int
    content_hash_before: str | None
    content_hash_after: str | None
    replaced_revision_ids: tuple[str, ...]
    source_validation: str
    old_payload: dict[str, Any] | None


@dataclass
class _ContextGroup:
    group_id: str
    messages: list[dict[str, str]]
    source_refs: set[str] = field(default_factory=set)


class DynamicMemoryEnvironment:
    """Policy runtime; private queries, world events, and source bodies are absent."""

    def __init__(
        self,
        history: DynamicHistoryPublic,
        *,
        accounting: TokenAccounting,
        read_rollout_id: str,
        policy_version: str,
        ingest_max_new_tokens: int,
        max_decisions_per_chunk: int,
        answer_tail_tokens: int,
        retrieval_payload_tokens: int,
    ) -> None:
        self.history = history
        self.accounting = accounting
        self.read_rollout_id = read_rollout_id
        self.policy_version = policy_version
        self.ingest_max_new_tokens = ingest_max_new_tokens
        self.max_decisions_per_chunk = max_decisions_per_chunk
        self.answer_tail_tokens = answer_tail_tokens
        self.retrieval_payload_tokens = retrieval_payload_tokens
        self._active: dict[str, dict[str, Any]] = {}
        self._ledger: list[AuditRevision] = []
        self._context: list[_ContextGroup] = []
        self._observed: set[str] = set()
        self._checkpoints: list[DynamicCheckpoint] = []
        self._memory_revision = 0
        self._next_chunk = 0
        self._current_chunk = None
        self._decisions = 0
        self._chunk_committed = True

    def policy_memory(self) -> tuple[dict[str, Any], ...]:
        """Only active versions are exposed; old bodies and audit judgments are absent."""

        return tuple(copy.deepcopy(self._active[key]) for key in sorted(self._active))

    def policy_observation(self) -> dict[str, Any]:
        if self._current_chunk is None:
            raise DynamicEnvironmentError("no current chunk")
        return project_public_observation(
            self.history, self._current_chunk.observed_at, self.policy_memory()
        )

    def current_messages(self) -> tuple[dict[str, str], ...]:
        """Return a defensive copy of the current policy-visible prompt only."""

        if self._current_chunk is None:
            raise DynamicEnvironmentError("no current chunk")
        messages = self._messages()
        self.accounting.enforce_context(
            messages,
            max_new_tokens=self.ingest_max_new_tokens,
            context_total_tokens=self.history.context_budget_tokens,
        )
        return tuple(copy.deepcopy(messages))

    def _handles(self) -> str:
        return _canonical(
            [
                {
                    "memory_id": item["memory_id"],
                    "revision_id": item["revision_id"],
                    "title": item.get("title"),
                    "tags": item.get("tags", []),
                }
                for item in self.policy_memory()
            ]
        )

    def _messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": DYNAMIC_INGEST_SYSTEM}]
        for group in self._context:
            messages.extend(copy.deepcopy(group.messages))
        messages.append(
            {
                "role": "user",
                "content": f"Active memory handles: {self._handles()}\nMemory B: {self.memory_tokens()}/{self.history.memory_budget_tokens}.",
            }
        )
        return messages

    def _context_tokens(self) -> int:
        return self.accounting.count_chat(self._messages())

    def _fit_context(self) -> None:
        while True:
            try:
                self.accounting.enforce_context(
                    self._messages(),
                    max_new_tokens=self.ingest_max_new_tokens,
                    context_total_tokens=self.history.context_budget_tokens,
                )
                return
            except BudgetError:
                current_id = (
                    self._current_chunk.chunk_id if self._current_chunk else None
                )
                removable = next(
                    (item for item in self._context if item.group_id != current_id),
                    None,
                )
                if removable is None:
                    raise DynamicEnvironmentError("current prompt cannot satisfy C")
                self._context.remove(removable)

    def memory_tokens(self, memories: Sequence[Mapping[str, Any]] | None = None) -> int:
        values = self._active.values() if memories is None else memories
        return sum(
            self.accounting.count_text(canonical_dynamic_payload(item))
            for item in values
        )

    @property
    def memory_revision(self) -> int:
        return self._memory_revision

    @property
    def checkpoints(self) -> tuple[DynamicCheckpoint, ...]:
        return tuple(self._checkpoints)

    def private_audit_ledger(self) -> tuple[AuditRevision, ...]:
        """Reward-side export. This method is never included in policy tools."""

        return tuple(copy.deepcopy(self._ledger))

    def admit_next_chunk(self) -> list[dict[str, str]]:
        if not self._chunk_committed:
            raise DynamicEnvironmentError("commit current chunk before advancing")
        if self._next_chunk >= len(self.history.chunks):
            raise StopIteration("all chunks were shown")
        chunk = self.history.chunks[self._next_chunk]
        self._context.append(
            _ContextGroup(
                group_id=chunk.chunk_id,
                messages=[{"role": "user", "content": f"STREAM CHUNK\n{chunk.text}"}],
                source_refs=set(chunk.source_refs),
            )
        )
        self._observed.update(chunk.source_refs)
        self._current_chunk = chunk
        self._next_chunk += 1
        self._decisions = 0
        self._chunk_committed = False
        self._fit_context()
        messages = self._messages()
        self.accounting.enforce_context(
            messages,
            max_new_tokens=self.ingest_max_new_tokens,
            context_total_tokens=self.history.context_budget_tokens,
        )
        return messages

    def _payload(self, action: Mapping[str, Any]) -> dict[str, Any]:
        content = action.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DynamicEnvironmentError("content_required")
        refs = action.get("source_refs") or []
        if not isinstance(refs, list) or not refs:
            raise DynamicEnvironmentError("source_refs_required")
        if any(not isinstance(ref, str) or ref not in self._observed for ref in refs):
            raise DynamicEnvironmentError("source_not_observed")
        claims = action.get("claims") or []
        if not isinstance(claims, list):
            raise DynamicEnvironmentError("claims_must_be_list")
        return {
            "content": content.strip(),
            "title": action.get("title"),
            "tags": copy.deepcopy(action.get("tags") or []),
            "source_refs": copy.deepcopy(refs),
            "claims": copy.deepcopy(claims),
            "custom": copy.deepcopy(action.get("custom") or {}),
        }

    def _revision_entry(
        self, memory_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        next_revision = self._memory_revision + 1
        return {
            "memory_id": memory_id,
            "revision_id": f"rev-{_digest([self.read_rollout_id, memory_id, next_revision, payload])[:20]}",
            **payload,
        }

    def _admit_proposed(self, proposed: Mapping[str, Mapping[str, Any]]) -> None:
        if (
            self.memory_tokens(list(proposed.values()))
            > self.history.memory_budget_tokens
        ):
            raise DynamicEnvironmentError("memory_budget_exceeded")

    def _record(
        self,
        *,
        action_type: str,
        success: bool,
        memory_id: str | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        memory_revision_before: int,
        source_validation: str,
    ) -> None:
        self._ledger.append(
            AuditRevision(
                audit_event_id=f"audit-{len(self._ledger):06d}",
                mutation_type=action_type,
                mutation_success=success,
                memory_id=memory_id,
                revision_id_before=before.get("revision_id") if before else None,
                revision_id_after=after.get("revision_id") if after else None,
                memory_revision_before=memory_revision_before,
                memory_revision_after=self._memory_revision,
                content_hash_before=_digest(before.get("content")) if before else None,
                content_hash_after=_digest(after.get("content")) if after else None,
                replaced_revision_ids=(before["revision_id"],) if before else (),
                source_validation=source_validation,
                old_payload=copy.deepcopy(before),
            )
        )

    def _retrieve_payload(self, memory: Mapping[str, Any]) -> tuple[str, bool]:
        raw = canonical_dynamic_payload(memory)
        if self.accounting.count_text(raw) <= self.retrieval_payload_tokens:
            return raw, False
        # Deterministic prefix truncation. Exposure is scored from exactly this string.
        lo, hi = 0, len(raw)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.accounting.count_text(raw[:mid]) <= self.retrieval_payload_tokens:
                lo = mid
            else:
                hi = mid - 1
        return raw[:lo], True

    def execute(
        self,
        action: Mapping[str, Any],
        *,
        response_text: str | None = None,
    ) -> DynamicActionResult:
        if self._current_chunk is None or self._chunk_committed:
            raise DynamicEnvironmentError("admit an uncommitted chunk before acting")
        if self._decisions >= self.max_decisions_per_chunk:
            raise DynamicEnvironmentError("decision_limit_reached")
        self._decisions += 1
        action_type = str(action.get("type", "")).upper()
        revision_before = self._memory_revision
        memory_id = str(action.get("memory_id", "")) or None
        old = copy.deepcopy(self._active.get(memory_id)) if memory_id else None
        displayed = ""
        displayed_revision = None
        truncated = False
        success = False
        code = "invalid_action"
        try:
            if action_type == "ADD":
                if not memory_id or memory_id in self._active:
                    raise DynamicEnvironmentError("invalid_or_duplicate_memory_id")
                new = self._revision_entry(memory_id, self._payload(action))
                proposed = dict(self._active)
                proposed[memory_id] = new
                self._admit_proposed(proposed)
                self._active = proposed
                self._memory_revision += 1
                success = True
                code = "ok"
            elif action_type == "UPDATE":
                if not memory_id or memory_id not in self._active:
                    raise DynamicEnvironmentError("memory_not_found")
                new = self._revision_entry(memory_id, self._payload(action))
                proposed = dict(self._active)
                proposed[memory_id] = new
                self._admit_proposed(proposed)
                self._active = proposed
                self._memory_revision += 1
                success = True
                code = "ok"
            elif action_type == "DELETE":
                if not memory_id or memory_id not in self._active:
                    raise DynamicEnvironmentError("memory_not_found")
                proposed = dict(self._active)
                proposed.pop(memory_id)
                self._admit_proposed(proposed)
                self._active = proposed
                self._memory_revision += 1
                success = True
                code = "ok"
                new = None
            elif action_type == "RETRIEVE":
                if not memory_id or memory_id not in self._active:
                    raise DynamicEnvironmentError("memory_not_found")
                memory = self._active[memory_id]
                displayed, truncated = self._retrieve_payload(memory)
                displayed_revision = memory["revision_id"]
                new = memory
                success = True
                code = "ok"
            elif action_type == "NEXT":
                new = None
                success = True
                code = "ok"
                self._decisions = self.max_decisions_per_chunk
            else:
                new = None
                raise DynamicEnvironmentError("invalid_action")
        except (DynamicEnvironmentError, TypeError, ValueError) as exc:
            code = str(exc)
            new = old

        if action_type in {"ADD", "UPDATE", "DELETE"}:
            self._record(
                action_type=action_type,
                success=success,
                memory_id=memory_id,
                before=old,
                after=copy.deepcopy(self._active.get(memory_id)) if success else old,
                memory_revision_before=revision_before,
                source_validation="observed" if success else code,
            )
        public_message = "OK"
        if action_type == "RETRIEVE" and success:
            public_message = f"MEMORY {displayed_revision}\n{displayed}"
        elif not success:
            public_message = f"ERROR {code}"
        receipt_messages = []
        if response_text is not None:
            receipt_messages.append({"role": "assistant", "content": response_text})
        receipt_messages.append({"role": "tool", "content": public_message})
        self._context.append(
            _ContextGroup(
                group_id=f"receipt-{len(self._ledger):06d}-{self._decisions}",
                messages=receipt_messages,
            )
        )
        self._fit_context()
        result = DynamicActionResult(
            admitted=success,
            code=code,
            message=public_message,
            memory_tokens=self.memory_tokens(),
            context_tokens=self._context_tokens(),
            memory_revision_before=revision_before,
            memory_revision_after=self._memory_revision,
            displayed_payload=displayed,
            displayed_revision_id=displayed_revision,
            truncated=truncated,
        )
        if action_type == "NEXT" and success:
            self.commit_chunk()
        return result

    def commit_chunk(self) -> DynamicCheckpoint:
        if self._current_chunk is None or self._chunk_committed:
            raise DynamicEnvironmentError("no uncommitted chunk")
        active = self.policy_memory()
        index = self._current_chunk.observed_at
        checkpoint = DynamicCheckpoint(
            checkpoint_id=f"checkpoint-{_digest([self.read_rollout_id, index, active])[:20]}",
            checkpoint_index=index,
            memory_revision=self._memory_revision,
            active_memories=active,
            observed_source_refs=tuple(sorted(self._observed)),
            memory_sha256=_digest(active),
        )
        self._checkpoints.append(checkpoint)
        self._chunk_committed = True
        return checkpoint

    def finalize(self) -> DynamicMemorySnapshot:
        if self._next_chunk != len(self.history.chunks) or not self._chunk_committed:
            raise DynamicEnvironmentError(
                "cannot snapshot before every chunk is committed"
            )
        active = self.policy_memory()
        tail_groups: list[_ContextGroup] = []
        tail_tokens = 0
        for group in reversed(self._context):
            cost = sum(
                self.accounting.count_text(item["content"]) for item in group.messages
            )
            if tail_tokens + cost <= self.answer_tail_tokens:
                tail_groups.append(group)
                tail_tokens += cost
        tail = tuple(
            copy.deepcopy(message)
            for group in reversed(tail_groups)
            for message in group.messages
        )
        memory_sha, tail_sha = _digest(active), _digest(tail)
        return DynamicMemorySnapshot(
            snapshot_id=f"snapshot-{_digest([self.read_rollout_id, memory_sha, tail_sha])[:20]}",
            history_id=self.history.history_id,
            read_rollout_id=self.read_rollout_id,
            active_memories=active,
            context_tail=tail,
            observed_source_refs=tuple(sorted(self._observed)),
            memory_sha256=memory_sha,
            tail_sha256=tail_sha,
            memory_revision=self._memory_revision,
            policy_version=self.policy_version,
        )


__all__ = [
    "AuditRevision",
    "DynamicActionResult",
    "DynamicCheckpoint",
    "DynamicEnvironmentError",
    "DynamicMemoryEnvironment",
    "DynamicMemorySnapshot",
    "canonical_dynamic_payload",
]
