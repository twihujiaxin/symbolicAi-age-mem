"""Deterministic Stage 1 storage baselines with a hard LTM token budget.

This module is intentionally separate from the M3 environment. It provides an
offline anti-shortcut benchmark without changing the frozen E1/M3 rollout
protocol. Oracle labels are available only to ``OracleSafeStorePolicy`` and
are never exposed through an environment observation.
"""

from __future__ import annotations

import hashlib
import random
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..memory_store import (
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    MemoryStoreSnapshot,
)
from .environment import deterministic_embedding
from .models import ToyAction, ToyFact, ToyMemoryTask


STAGE1_STORAGE_SCHEMA_VERSION = 1
LEXICAL_TOKEN_COUNTER_NAME = "unicode-lexical-v1"
StoragePolicyName = Literal["store-all", "store-none", "oracle-safe-store"]
BudgetOperation = Literal["add", "update", "delete", "restore", "reset"]
BudgetDecision = Literal[
    "admitted",
    "budget_exceeded",
    "not_found",
    "backend_rejected",
]


def count_ltm_tokens(text: str) -> int:
    """Count exact tokens under the offline ``unicode-lexical-v1`` contract.

    Words and individual punctuation characters each cost one token. The
    counter is deliberately local and deterministic; a production tokenizer
    can be injected into :class:`TokenBudgetMemoryStore` instead.
    """

    if not isinstance(text, str):
        raise TypeError("memory content must be a string")
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def ordered_stage1_facts(task: ToyMemoryTask, seed: int) -> Tuple[ToyFact, ...]:
    """Match the M3 environment's deterministic Stage-1 observation order."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    facts = [fact for fact in task.facts if fact.stage == 1]
    seed_bytes = f"{task.task_id}:{seed}:1".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    random.Random(local_seed).shuffle(facts)
    return tuple(facts)


class PublicStage1Fact(BaseModel):
    """One observed fact with an opaque action handle and public text only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_handle: str = Field(pattern=r"^stage1-fact-\d{3,}$")
    title: str = Field(min_length=1)
    sentence: str = Field(min_length=1)


class PublicStage1Input(BaseModel):
    """The complete information boundary presented to non-Oracle policies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    observed_facts: Tuple[PublicStage1Fact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_handles(self) -> "PublicStage1Input":
        handles = [fact.fact_handle for fact in self.observed_facts]
        if len(handles) != len(set(handles)):
            raise ValueError("public Stage 1 fact handles must be unique")
        return self


def _public_stage1_input(
    task: ToyMemoryTask,
    seed: int,
) -> Tuple[PublicStage1Input, Dict[str, str]]:
    """Create a public view and its private handle-to-fact lookup table."""

    observed = ordered_stage1_facts(task, seed)
    public_facts = tuple(
        PublicStage1Fact(
            fact_handle=f"stage1-fact-{index:03d}",
            title=fact.title,
            sentence=fact.sentence,
        )
        for index, fact in enumerate(observed)
    )
    handle_to_fact_id = {
        public.fact_handle: private.fact_id
        for public, private in zip(public_facts, observed)
    }
    return (
        PublicStage1Input(
            task_id=task.task_id,
            seed=seed,
            observed_facts=public_facts,
        ),
        handle_to_fact_id,
    )


def _resolve_public_action(
    action: ToyAction,
    handle_to_fact_id: Mapping[str, str],
) -> ToyAction:
    """Resolve opaque policy handles only after crossing the trusted boundary."""

    if action.kind not in {"add", "update"}:
        raise ValueError("Stage 1 storage policies may only ADD or UPDATE")
    try:
        fact_id = handle_to_fact_id[action.fact_id or ""]
    except KeyError as exc:
        raise ValueError(
            "policy referenced an unknown or unobserved Stage 1 fact handle"
        ) from exc
    target_fact_id = None
    if action.kind == "update":
        try:
            target_fact_id = handle_to_fact_id[action.target_fact_id or ""]
        except KeyError as exc:
            raise ValueError(
                "policy referenced an unknown or unobserved Stage 1 target handle"
            ) from exc
    return ToyAction(
        kind=action.kind,
        fact_id=fact_id,
        target_fact_id=target_fact_id,
    )


class MemoryBudgetAuditEvent(BaseModel):
    """One immutable admission decision made by a bounded store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = STAGE1_STORAGE_SCHEMA_VERSION
    event_index: int = Field(ge=0)
    rollout_id: str = Field(min_length=1)
    operation: BudgetOperation
    memory_id: Optional[str] = None
    source_step: Optional[int] = Field(default=None, ge=0)
    admitted: bool
    reason: BudgetDecision
    token_counter: str = Field(min_length=1)
    budget_tokens: int = Field(ge=0)
    active_tokens_before: int = Field(ge=0)
    previous_content_tokens: Optional[int] = Field(default=None, ge=0)
    attempted_content_tokens: Optional[int] = Field(default=None, ge=0)
    projected_active_tokens: int = Field(ge=0)
    active_tokens_after: int = Field(ge=0)
    token_delta: int
    version_before: Optional[int] = Field(default=None, ge=1)
    version_after: Optional[int] = Field(default=None, ge=1)
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_accounting(self) -> "MemoryBudgetAuditEvent":
        if self.token_delta != self.projected_active_tokens - self.active_tokens_before:
            raise ValueError("token_delta does not match the projected token change")
        if self.admitted:
            if self.reason != "admitted":
                raise ValueError("an admitted event must use reason='admitted'")
            if self.active_tokens_after != self.projected_active_tokens:
                raise ValueError(
                    "an admitted event must reach the projected token count"
                )
            if self.active_tokens_after > self.budget_tokens:
                raise ValueError("an admitted event cannot exceed the token budget")
        else:
            if self.reason == "admitted":
                raise ValueError("a rejected event cannot use reason='admitted'")
            if self.active_tokens_after != self.active_tokens_before:
                raise ValueError(
                    "a rejected event must leave active token use unchanged"
                )
        if (
            self.reason == "budget_exceeded"
            and self.projected_active_tokens <= self.budget_tokens
        ):
            raise ValueError("budget_exceeded requires a projection above budget")
        return self


class MemoryBudgetExceeded(ValueError):
    """Raised after an over-budget mutation is rejected without store changes."""

    def __init__(self, event: MemoryBudgetAuditEvent) -> None:
        self.event = event
        super().__init__(
            "LTM token budget exceeded: "
            f"projected={event.projected_active_tokens}, "
            f"budget={event.budget_tokens}"
        )


class MemoryBudgetReport(BaseModel):
    """Current budget state plus the complete mutation audit log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = STAGE1_STORAGE_SCHEMA_VERSION
    rollout_id: str = Field(min_length=1)
    token_counter: str = Field(min_length=1)
    budget_tokens: int = Field(ge=0)
    active_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    active_memory_count: int = Field(ge=0)
    events: Tuple[MemoryBudgetAuditEvent, ...]

    @model_validator(mode="after")
    def validate_totals(self) -> "MemoryBudgetReport":
        if self.remaining_tokens != self.budget_tokens - self.active_tokens:
            raise ValueError("remaining_tokens does not match budget usage")
        return self


class TokenBudgetMemoryStore:
    """A ``MemoryStore`` wrapper enforcing a fixed active-content token budget.

    Admission and the underlying mutation share a lock. ADD and UPDATE first
    compute their exact projected active-content cost. If the projection is
    above the fixed budget, an audit event is appended and the backend is not
    called. Historical superseded/tombstone contents remain auditable but do
    not consume the active LTM capacity.
    """

    def __init__(
        self,
        backend: MemoryStore,
        *,
        token_budget: int,
        token_counter: Callable[[str], int] = count_ltm_tokens,
        token_counter_name: str = LEXICAL_TOKEN_COUNTER_NAME,
    ) -> None:
        if not isinstance(backend, MemoryStore):
            raise TypeError("backend must implement the MemoryStore protocol")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int):
            raise TypeError("token_budget must be an integer")
        if token_budget < 0:
            raise ValueError("token_budget must be non-negative")
        if not callable(token_counter):
            raise TypeError("token_counter must be callable")
        if not isinstance(token_counter_name, str) or not token_counter_name.strip():
            raise ValueError("token_counter_name must be non-empty")

        self._backend = backend
        self._token_budget = token_budget
        self._token_counter = token_counter
        self._token_counter_name = token_counter_name.strip()
        self._lock = threading.RLock()
        self._events: List[MemoryBudgetAuditEvent] = []

        initial_tokens = self._active_tokens_unlocked()
        if initial_tokens > self._token_budget:
            raise ValueError(
                "backend already exceeds the requested LTM token budget: "
                f"active={initial_tokens}, budget={self._token_budget}"
            )

    @property
    def rollout_id(self) -> str:
        return self._backend.rollout_id

    @property
    def research_mode(self) -> bool:
        return self._backend.research_mode

    @property
    def token_budget(self) -> int:
        return self._token_budget

    @property
    def token_counter_name(self) -> str:
        return self._token_counter_name

    def _count(self, content: str) -> int:
        value = self._token_counter(content)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token_counter must return a non-negative integer")
        return value

    def _active_tokens_unlocked(self) -> int:
        return sum(self._count(record.content) for record in self._backend.get_all())

    def active_tokens(self) -> int:
        with self._lock:
            return self._active_tokens_unlocked()

    def remaining_tokens(self) -> int:
        with self._lock:
            return self._token_budget - self._active_tokens_unlocked()

    def audit_log(self) -> Tuple[MemoryBudgetAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def report(self) -> MemoryBudgetReport:
        with self._lock:
            active_tokens = self._active_tokens_unlocked()
            return MemoryBudgetReport(
                rollout_id=self.rollout_id,
                token_counter=self.token_counter_name,
                budget_tokens=self.token_budget,
                active_tokens=active_tokens,
                remaining_tokens=self.token_budget - active_tokens,
                active_memory_count=self._backend.size(),
                events=tuple(self._events),
            )

    def _append_event(
        self,
        *,
        operation: BudgetOperation,
        memory_id: Optional[str],
        source_step: Optional[int],
        admitted: bool,
        reason: BudgetDecision,
        before: int,
        previous: Optional[int],
        attempted: Optional[int],
        projected: int,
        after: int,
        version_before: Optional[int] = None,
        version_after: Optional[int] = None,
        error: Optional[str] = None,
    ) -> MemoryBudgetAuditEvent:
        event = MemoryBudgetAuditEvent(
            event_index=len(self._events),
            rollout_id=self.rollout_id,
            operation=operation,
            memory_id=memory_id,
            source_step=source_step,
            admitted=admitted,
            reason=reason,
            token_counter=self.token_counter_name,
            budget_tokens=self.token_budget,
            active_tokens_before=before,
            previous_content_tokens=previous,
            attempted_content_tokens=attempted,
            projected_active_tokens=projected,
            active_tokens_after=after,
            token_delta=projected - before,
            version_before=version_before,
            version_after=version_after,
            error=error,
        )
        self._events.append(event)
        return event

    def _reject_budget(
        self,
        *,
        operation: Literal["add", "update", "restore"],
        memory_id: Optional[str],
        source_step: Optional[int],
        before: int,
        previous: Optional[int],
        attempted: Optional[int],
        projected: int,
        version_before: Optional[int] = None,
    ) -> None:
        event = self._append_event(
            operation=operation,
            memory_id=memory_id,
            source_step=source_step,
            admitted=False,
            reason="budget_exceeded",
            before=before,
            previous=previous,
            attempted=attempted,
            projected=projected,
            after=before,
            version_before=version_before,
        )
        raise MemoryBudgetExceeded(event)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            before = self._active_tokens_unlocked()
            attempted = self._count(record.content)
            projected = before + attempted
            if projected > self.token_budget:
                self._reject_budget(
                    operation="add",
                    memory_id=record.memory_id,
                    source_step=record.source_step,
                    before=before,
                    previous=None,
                    attempted=attempted,
                    projected=projected,
                )
            try:
                added = self._backend.add(record)
            except ValueError as exc:
                after = self._active_tokens_unlocked()
                self._append_event(
                    operation="add",
                    memory_id=record.memory_id,
                    source_step=record.source_step,
                    admitted=False,
                    reason="backend_rejected",
                    before=before,
                    previous=None,
                    attempted=attempted,
                    projected=projected,
                    after=after,
                    error=str(exc),
                )
                raise
            after = self._active_tokens_unlocked()
            self._append_event(
                operation="add",
                memory_id=added.memory_id,
                source_step=added.source_step,
                admitted=True,
                reason="admitted",
                before=before,
                previous=None,
                attempted=attempted,
                projected=projected,
                after=after,
                version_after=added.version,
            )
            return added

    def retrieve(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        with self._lock:
            return self._backend.retrieve(query_embedding, top_k, metadata_filter)

    def update(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        source_step: Optional[int] = None,
    ) -> Optional[MemoryRecord]:
        with self._lock:
            before = self._active_tokens_unlocked()
            current = self._backend.get(memory_id)
            if current is None:
                self._append_event(
                    operation="update",
                    memory_id=memory_id,
                    source_step=source_step,
                    admitted=False,
                    reason="not_found",
                    before=before,
                    previous=None,
                    attempted=None,
                    projected=before,
                    after=before,
                )
                return None

            previous = self._count(current.content)
            attempted = self._count(current.content if content is None else content)
            projected = before - previous + attempted
            if projected > self.token_budget:
                self._reject_budget(
                    operation="update",
                    memory_id=memory_id,
                    source_step=source_step,
                    before=before,
                    previous=previous,
                    attempted=attempted,
                    projected=projected,
                    version_before=current.version,
                )
            try:
                updated = self._backend.update(
                    memory_id,
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                    source_step=source_step,
                )
            except ValueError as exc:
                after = self._active_tokens_unlocked()
                self._append_event(
                    operation="update",
                    memory_id=memory_id,
                    source_step=source_step,
                    admitted=False,
                    reason="backend_rejected",
                    before=before,
                    previous=previous,
                    attempted=attempted,
                    projected=projected,
                    after=after,
                    version_before=current.version,
                    error=str(exc),
                )
                raise
            if updated is None:
                after = self._active_tokens_unlocked()
                self._append_event(
                    operation="update",
                    memory_id=memory_id,
                    source_step=source_step,
                    admitted=False,
                    reason="backend_rejected",
                    before=before,
                    previous=previous,
                    attempted=attempted,
                    projected=projected,
                    after=after,
                    version_before=current.version,
                    error="backend returned no updated record",
                )
                return None
            after = self._active_tokens_unlocked()
            self._append_event(
                operation="update",
                memory_id=memory_id,
                source_step=source_step,
                admitted=True,
                reason="admitted",
                before=before,
                previous=previous,
                attempted=attempted,
                projected=projected,
                after=after,
                version_before=current.version,
                version_after=updated.version,
            )
            return updated

    def delete(
        self,
        memory_id: str,
        *,
        source_step: Optional[int] = None,
    ) -> Optional[MemoryRecord]:
        with self._lock:
            before = self._active_tokens_unlocked()
            current = self._backend.get(memory_id)
            if current is None:
                self._append_event(
                    operation="delete",
                    memory_id=memory_id,
                    source_step=source_step,
                    admitted=False,
                    reason="not_found",
                    before=before,
                    previous=None,
                    attempted=None,
                    projected=before,
                    after=before,
                )
                return None
            previous = self._count(current.content)
            projected = before - previous
            deleted = self._backend.delete(memory_id, source_step=source_step)
            if deleted is None:
                after = self._active_tokens_unlocked()
                self._append_event(
                    operation="delete",
                    memory_id=memory_id,
                    source_step=source_step,
                    admitted=False,
                    reason="backend_rejected",
                    before=before,
                    previous=previous,
                    attempted=0,
                    projected=projected,
                    after=after,
                    version_before=current.version,
                    error="backend returned no deleted record",
                )
                return None
            after = self._active_tokens_unlocked()
            self._append_event(
                operation="delete",
                memory_id=memory_id,
                source_step=source_step,
                admitted=True,
                reason="admitted",
                before=before,
                previous=previous,
                attempted=0,
                projected=projected,
                after=after,
                version_before=current.version,
                version_after=deleted.version,
            )
            return deleted

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            return self._backend.get(memory_id)

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return self._backend.get_all()

    def history(self, memory_id: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            return self._backend.history(memory_id)

    def snapshot(self) -> MemoryStoreSnapshot:
        with self._lock:
            return self._backend.snapshot()

    def restore(self, snapshot: MemoryStoreSnapshot) -> None:
        with self._lock:
            before = self._active_tokens_unlocked()
            projected = sum(
                self._count(record.content)
                for record in snapshot.records
                if record.status == "active"
            )
            if projected > self.token_budget:
                self._reject_budget(
                    operation="restore",
                    memory_id=None,
                    source_step=None,
                    before=before,
                    previous=before,
                    attempted=projected,
                    projected=projected,
                )
            try:
                self._backend.restore(snapshot)
            except ValueError as exc:
                after = self._active_tokens_unlocked()
                self._append_event(
                    operation="restore",
                    memory_id=None,
                    source_step=None,
                    admitted=False,
                    reason="backend_rejected",
                    before=before,
                    previous=before,
                    attempted=projected,
                    projected=projected,
                    after=after,
                    error=str(exc),
                )
                raise
            after = self._active_tokens_unlocked()
            self._append_event(
                operation="restore",
                memory_id=None,
                source_step=None,
                admitted=True,
                reason="admitted",
                before=before,
                previous=before,
                attempted=projected,
                projected=projected,
                after=after,
            )

    def reset(self) -> None:
        with self._lock:
            before = self._active_tokens_unlocked()
            self._backend.reset()
            self._append_event(
                operation="reset",
                memory_id=None,
                source_step=None,
                admitted=True,
                reason="admitted",
                before=before,
                previous=before,
                attempted=0,
                projected=0,
                after=0,
            )

    def size(self) -> int:
        with self._lock:
            return self._backend.size()


@runtime_checkable
class Stage1StoragePolicy(Protocol):
    """A non-Oracle policy confined to the public Stage 1 view."""

    name: str

    def actions(self, public_input: PublicStage1Input) -> Tuple[ToyAction, ...]: ...


class StoreAllPolicy:
    """Attempt to ADD every fact visible in Stage 1, including noise/copies."""

    name: Literal["store-all"] = "store-all"
    uses_oracle_labels = False

    def actions(self, public_input: PublicStage1Input) -> Tuple[ToyAction, ...]:
        return tuple(
            ToyAction(kind="add", fact_id=fact.fact_handle)
            for fact in public_input.observed_facts
        )


class StoreNonePolicy:
    """Never write Stage 1 observations to LTM."""

    name: Literal["store-none"] = "store-none"
    uses_oracle_labels = False

    def actions(self, public_input: PublicStage1Input) -> Tuple[ToyAction, ...]:
        del public_input
        return ()


class OracleSafeStorePolicy:
    """Store only current supporting facts using private offline annotations."""

    name: Literal["oracle-safe-store"] = "oracle-safe-store"
    uses_oracle_labels = True

    def oracle_actions(
        self,
        task: ToyMemoryTask,
        seed: int,
    ) -> Tuple[ToyAction, ...]:
        """Use private labels through the benchmark's trusted Oracle path only."""

        supporting = set(task.supporting_fact_ids)
        unavailable = {
            fact_id for fact_id in supporting if task.fact(fact_id).stage != 1
        }
        if unavailable:
            raise ValueError(
                "oracle-safe-store cannot write supporting facts hidden from Stage 1: "
                f"{sorted(unavailable)}"
            )
        return tuple(
            ToyAction(kind="add", fact_id=fact.fact_id)
            for fact in ordered_stage1_facts(task, seed)
            if fact.fact_id in supporting
        )


class Stage1StorageDecision(BaseModel):
    """Policy intent joined to its bounded-store admission event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_index: int = Field(ge=0)
    action: ToyAction
    memory_id: str = Field(min_length=1)
    admitted: bool
    reason: BudgetDecision
    audit_event_index: int = Field(ge=0)


class Stage1ActiveMemory(BaseModel):
    """Compact active-memory view retained in a benchmark report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(min_length=1)
    fact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    content: str
    content_tokens: int = Field(ge=0)
    version: int = Field(ge=1)


class Stage1StorageRunResult(BaseModel):
    """Serializable result for one policy/task/budget combination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = STAGE1_STORAGE_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    policy: str = Field(min_length=1)
    uses_oracle_labels: bool
    token_counter: str = Field(min_length=1)
    budget_tokens: int = Field(ge=0)
    active_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    selected_fact_ids: Tuple[str, ...]
    stored_supporting_fact_ids: Tuple[str, ...]
    stored_non_supporting_fact_ids: Tuple[str, ...]
    decisions: Tuple[Stage1StorageDecision, ...]
    active_memories: Tuple[Stage1ActiveMemory, ...]
    audit_events: Tuple[MemoryBudgetAuditEvent, ...]
    memory_snapshot: Dict[str, Any]

    @model_validator(mode="after")
    def validate_result(self) -> "Stage1StorageRunResult":
        if self.remaining_tokens != self.budget_tokens - self.active_tokens:
            raise ValueError("result remaining_tokens does not match token use")
        if self.active_tokens != sum(
            item.content_tokens for item in self.active_memories
        ):
            raise ValueError("result active_tokens does not match active memories")
        if tuple(event.event_index for event in self.audit_events) != tuple(
            range(len(self.audit_events))
        ):
            raise ValueError("audit event indices must be contiguous")
        if any(
            decision.audit_event_index >= len(self.audit_events)
            for decision in self.decisions
        ):
            raise ValueError("a decision references a missing audit event")
        return self


class _DeterministicClock:
    """Stable report timestamps without depending on wall-clock execution."""

    def __init__(self) -> None:
        self._tick = 0

    def __call__(self) -> str:
        instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=self._tick
        )
        self._tick += 1
        return instant.isoformat()


class Stage1StorageBenchmark:
    """Execute the three Stage 1 baselines without an Agent or LLM."""

    def __init__(
        self,
        *,
        token_budget: int,
        token_counter: Callable[[str], int] = count_ltm_tokens,
        token_counter_name: str = LEXICAL_TOKEN_COUNTER_NAME,
        store_factory: Optional[Callable[[str], MemoryStore]] = None,
    ) -> None:
        self.token_budget = token_budget
        self.token_counter = token_counter
        self.token_counter_name = token_counter_name
        self.store_factory = store_factory

    @staticmethod
    def _role(task: ToyMemoryTask, fact: ToyFact) -> str:
        if fact.fact_id in task.supporting_fact_ids:
            return "supporting"
        if fact.fact_id in task.distractor_fact_ids:
            return "distractor"
        if fact.fact_id in task.stale_fact_ids:
            return "stale"
        if fact.fact_id in task.duplicate_fact_ids:
            return "duplicate"
        return "other"

    def _new_backend(self, rollout_id: str) -> MemoryStore:
        if self.store_factory is not None:
            backend = self.store_factory(rollout_id)
        else:
            backend = InMemoryStore(
                rollout_id,
                research_mode=True,
                clock=_DeterministicClock(),
            )
        if not isinstance(backend, MemoryStore):
            raise TypeError("store_factory must return a MemoryStore")
        if backend.rollout_id != rollout_id:
            raise ValueError("store_factory returned a mismatched rollout_id")
        return backend

    def run(
        self,
        task: ToyMemoryTask,
        policy: Stage1StoragePolicy | OracleSafeStorePolicy,
        *,
        rollout_id: str,
        seed: int,
    ) -> Stage1StorageRunResult:
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        # Oracle access is a capability granted to one exact built-in type. A
        # custom policy cannot opt in by claiming ``uses_oracle_labels=True``
        # or by copying the reserved policy name.
        trusted_oracle = type(policy) is OracleSafeStorePolicy
        if trusted_oracle:
            policy_name = OracleSafeStorePolicy.name
            actions = policy.oracle_actions(task, seed)
        else:
            if not isinstance(policy, Stage1StoragePolicy):
                raise TypeError("policy must implement the public Stage1StoragePolicy")
            policy_name = policy.name
            if not isinstance(policy_name, str) or not policy_name.strip():
                raise ValueError("policy name must be a non-empty string")
            if policy_name == OracleSafeStorePolicy.name:
                raise ValueError(
                    "the oracle-safe-store name is reserved for the trusted Oracle policy"
                )
            public_input, handle_to_fact_id = _public_stage1_input(task, seed)
            public_actions = policy.actions(public_input)
            actions = tuple(
                _resolve_public_action(action, handle_to_fact_id)
                for action in public_actions
            )

        store = TokenBudgetMemoryStore(
            self._new_backend(rollout_id),
            token_budget=self.token_budget,
            token_counter=self.token_counter,
            token_counter_name=self.token_counter_name,
        )
        decisions: List[Stage1StorageDecision] = []
        selected_fact_ids: List[str] = []
        memory_ids_by_fact_id: Dict[str, str] = {}

        for action_index, action in enumerate(actions):
            if action.kind not in {"add", "update"}:
                raise ValueError("Stage 1 storage policies may only ADD or UPDATE")
            fact = task.fact(action.fact_id or "")
            if fact.stage != 1:
                raise ValueError(
                    f"policy attempted to store unobserved Stage 2 fact {fact.fact_id!r}"
                )
            selected_fact_ids.append(fact.fact_id)
            metadata = {
                "task_id": task.task_id,
                "fact_id": fact.fact_id,
                "role": self._role(task, fact),
                "title": fact.title,
                "stage": "1",
                "storage_policy": policy_name,
            }

            if action.kind == "add":
                memory_id = f"{rollout_id}:memory:{fact.fact_id}"
                try:
                    store.add(
                        MemoryRecord(
                            memory_id=memory_id,
                            content=fact.sentence,
                            metadata=metadata,
                            embedding=deterministic_embedding(fact.sentence),
                            source_rollout_id=rollout_id,
                            source_step=action_index,
                        )
                    )
                except MemoryBudgetExceeded:
                    pass
                else:
                    memory_ids_by_fact_id[fact.fact_id] = memory_id
            else:
                target_fact_id = action.target_fact_id or ""
                if fact.replaces_fact_id != target_fact_id:
                    raise ValueError("UPDATE fact is not a correction for its target")
                memory_id = memory_ids_by_fact_id.get(
                    target_fact_id,
                    f"{rollout_id}:memory:{target_fact_id}",
                )
                try:
                    updated = store.update(
                        memory_id,
                        content=fact.sentence,
                        metadata=metadata,
                        embedding=deterministic_embedding(fact.sentence),
                        source_step=action_index,
                    )
                except MemoryBudgetExceeded:
                    updated = None
                if updated is not None:
                    memory_ids_by_fact_id[fact.fact_id] = memory_id

            event = store.audit_log()[-1]
            decisions.append(
                Stage1StorageDecision(
                    action_index=action_index,
                    action=action,
                    memory_id=memory_id,
                    admitted=event.admitted,
                    reason=event.reason,
                    audit_event_index=event.event_index,
                )
            )

        active_records = store.get_all()
        active_memories = tuple(
            Stage1ActiveMemory(
                memory_id=record.memory_id,
                fact_id=str(record.metadata.get("fact_id", "")),
                role=str(record.metadata.get("role", "other")),
                content=record.content,
                content_tokens=self.token_counter(record.content),
                version=record.version,
            )
            for record in active_records
        )
        active_fact_ids = {item.fact_id for item in active_memories}
        supporting = set(task.supporting_fact_ids)
        report = store.report()
        return Stage1StorageRunResult(
            task_id=task.task_id,
            rollout_id=rollout_id,
            seed=seed,
            policy=policy_name,
            uses_oracle_labels=trusted_oracle,
            token_counter=report.token_counter,
            budget_tokens=report.budget_tokens,
            active_tokens=report.active_tokens,
            remaining_tokens=report.remaining_tokens,
            selected_fact_ids=tuple(selected_fact_ids),
            stored_supporting_fact_ids=tuple(sorted(active_fact_ids & supporting)),
            stored_non_supporting_fact_ids=tuple(sorted(active_fact_ids - supporting)),
            decisions=tuple(decisions),
            active_memories=active_memories,
            audit_events=report.events,
            memory_snapshot=store.snapshot().to_dict(),
        )


__all__ = [
    "LEXICAL_TOKEN_COUNTER_NAME",
    "MemoryBudgetAuditEvent",
    "MemoryBudgetExceeded",
    "MemoryBudgetReport",
    "OracleSafeStorePolicy",
    "PublicStage1Fact",
    "PublicStage1Input",
    "STAGE1_STORAGE_SCHEMA_VERSION",
    "Stage1ActiveMemory",
    "Stage1StorageBenchmark",
    "Stage1StorageDecision",
    "Stage1StoragePolicy",
    "Stage1StorageRunResult",
    "StoragePolicyName",
    "StoreAllPolicy",
    "StoreNonePolicy",
    "TokenBudgetMemoryStore",
    "count_ltm_tokens",
    "ordered_stage1_facts",
]
