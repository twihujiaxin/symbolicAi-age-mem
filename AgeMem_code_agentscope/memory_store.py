# -*- coding: utf-8 -*-
"""Backend-independent, rollout-scoped memory storage contracts.

The store is deliberately synchronous: it owns state and vector lookup only.
AgentScope's asynchronous API and embedding calls live in ``memory.py``.
"""

from __future__ import annotations

import math
import threading
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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


MEMORY_STORE_SCHEMA_VERSION = 1
MemoryStatus = Literal["active", "superseded", "discarded"]
_VALID_STATUSES = {"active", "superseded", "discarded"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class MemoryRecord:
    """One immutable, auditable version of a logical memory."""

    memory_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    version: int = 1
    status: MemoryStatus = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source_rollout_id: Optional[str] = None
    source_step: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id:
            raise ValueError("memory_id must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.version < 1:
            raise ValueError("version must be at least 1")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unsupported memory status: {self.status!r}")
        if self.source_step is not None and self.source_step < 0:
            raise ValueError("source_step must be non-negative")
        metadata = deepcopy(dict(self.metadata))
        embedding = None if self.embedding is None else list(self.embedding)
        if embedding is not None and any(not math.isfinite(x) for x in embedding):
            raise ValueError("embedding values must be finite")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "embedding", embedding)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": deepcopy(self.metadata),
            "embedding": None if self.embedding is None else list(self.embedding),
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_rollout_id": self.source_rollout_id,
            "source_step": self.source_step,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(data["memory_id"]),
            content=data["content"],
            metadata=dict(data.get("metadata") or {}),
            embedding=data.get("embedding"),
            version=int(data.get("version", 1)),
            status=data.get("status", "active"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            source_rollout_id=data.get("source_rollout_id"),
            source_step=data.get("source_step"),
        )

    def copy(self) -> "MemoryRecord":
        return MemoryRecord.from_dict(self.to_dict())


@dataclass(frozen=True)
class MemoryStoreSnapshot:
    """Complete store state, including superseded and discarded versions."""

    rollout_id: str
    records: Tuple[MemoryRecord, ...]
    research_mode: bool = True
    schema_version: int = MEMORY_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.rollout_id:
            raise ValueError("snapshot rollout_id must be non-empty")
        if self.schema_version != MEMORY_STORE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported memory snapshot schema version {self.schema_version}"
            )
        object.__setattr__(
            self,
            "records",
            tuple(record.copy() for record in self.records),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rollout_id": self.rollout_id,
            "research_mode": self.research_mode,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryStoreSnapshot":
        records = data.get("records")
        if not isinstance(records, list):
            raise ValueError("snapshot records must be a list")
        return cls(
            schema_version=int(
                data.get("schema_version", MEMORY_STORE_SCHEMA_VERSION)
            ),
            rollout_id=str(data["rollout_id"]),
            research_mode=bool(data.get("research_mode", True)),
            records=tuple(MemoryRecord.from_dict(record) for record in records),
        )


@runtime_checkable
class MemoryStore(Protocol):
    """Storage protocol consumed by the AgentScope memory-manager adapter."""

    @property
    def rollout_id(self) -> str: ...

    @property
    def research_mode(self) -> bool: ...

    def add(self, record: MemoryRecord) -> MemoryRecord: ...

    def retrieve(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]: ...

    def update(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        source_step: Optional[int] = None,
    ) -> Optional[MemoryRecord]: ...

    def delete(
        self,
        memory_id: str,
        *,
        source_step: Optional[int] = None,
    ) -> Optional[MemoryRecord]: ...

    def get(self, memory_id: str) -> Optional[MemoryRecord]: ...

    def get_all(self) -> List[MemoryRecord]: ...

    def history(self, memory_id: Optional[str] = None) -> List[MemoryRecord]: ...

    def snapshot(self) -> MemoryStoreSnapshot: ...

    def restore(self, snapshot: MemoryStoreSnapshot) -> None: ...

    def reset(self) -> None: ...

    def size(self) -> int: ...


class InMemoryStore:
    """Thread-safe versioned store bound to exactly one rollout."""

    def __init__(
        self,
        rollout_id: str,
        *,
        research_mode: bool = True,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        self._rollout_id = rollout_id
        self._research_mode = research_mode
        self._clock = clock
        self._lock = threading.RLock()
        self._versions: Dict[str, List[MemoryRecord]] = {}

    @property
    def rollout_id(self) -> str:
        return self._rollout_id

    @property
    def research_mode(self) -> bool:
        return self._research_mode

    def _normalize_record(self, record: MemoryRecord) -> MemoryRecord:
        if record.source_rollout_id not in (None, self.rollout_id):
            raise ValueError(
                "memory source_rollout_id does not match store rollout_id: "
                f"{record.source_rollout_id!r} != {self.rollout_id!r}"
            )
        now = self._clock()
        return replace(
            record,
            metadata=deepcopy(record.metadata),
            embedding=None if record.embedding is None else list(record.embedding),
            version=1,
            status="active",
            created_at=record.created_at or now,
            updated_at=record.updated_at or now,
            source_rollout_id=self.rollout_id,
        )

    def _active_unlocked(self, memory_id: str) -> Optional[MemoryRecord]:
        for record in reversed(self._versions.get(memory_id, [])):
            if record.status == "active":
                return record
        return None

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            if record.memory_id in self._versions:
                raise ValueError(f"memory_id already exists: {record.memory_id}")
            normalized = self._normalize_record(record)
            self._versions[record.memory_id] = [normalized]
            return normalized.copy()

    def retrieve(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        if top_k <= 0:
            return []
        with self._lock:
            scored: List[Tuple[MemoryRecord, float]] = []
            for memory_id in self._versions:
                record = self._active_unlocked(memory_id)
                if record is None or record.embedding is None:
                    continue
                if metadata_filter and not all(
                    record.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                ):
                    continue
                score = _cosine_similarity(query_embedding, record.embedding)
                if score > 0.0:
                    scored.append((record.copy(), score))
            scored.sort(key=lambda item: (-item[1], item[0].memory_id))
            return scored[:top_k]

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
            current = self._active_unlocked(memory_id)
            if current is None:
                return None
            now = self._clock()
            history = self._versions[memory_id]
            current_index = history.index(current)
            history[current_index] = replace(
                current,
                metadata=deepcopy(current.metadata),
                embedding=None if current.embedding is None else list(current.embedding),
                status="superseded",
                updated_at=now,
            )
            merged_metadata = deepcopy(current.metadata)
            if metadata is not None:
                merged_metadata.update(deepcopy(dict(metadata)))
            if embedding is not None:
                next_embedding = list(embedding)
            elif content is None:
                next_embedding = (
                    None if current.embedding is None else list(current.embedding)
                )
            else:
                # A store cannot infer an embedding for changed text. Leaving it
                # unset is safer than retaining a stale vector.
                next_embedding = None
            next_record = MemoryRecord(
                memory_id=memory_id,
                content=current.content if content is None else content,
                metadata=merged_metadata,
                embedding=next_embedding,
                version=current.version + 1,
                status="active",
                created_at=current.created_at,
                updated_at=now,
                source_rollout_id=self.rollout_id,
                source_step=(current.source_step if source_step is None else source_step),
            )
            history.append(next_record)
            return next_record.copy()

    def delete(
        self,
        memory_id: str,
        *,
        source_step: Optional[int] = None,
    ) -> Optional[MemoryRecord]:
        with self._lock:
            current = self._active_unlocked(memory_id)
            if current is None:
                return None
            if not self.research_mode:
                deleted = current.copy()
                del self._versions[memory_id]
                return deleted

            now = self._clock()
            history = self._versions[memory_id]
            current_index = history.index(current)
            history[current_index] = replace(
                current,
                metadata=deepcopy(current.metadata),
                embedding=None if current.embedding is None else list(current.embedding),
                status="superseded",
                updated_at=now,
            )
            tombstone = MemoryRecord(
                memory_id=memory_id,
                content=current.content,
                metadata=deepcopy(current.metadata),
                embedding=None if current.embedding is None else list(current.embedding),
                version=current.version + 1,
                status="discarded",
                created_at=current.created_at,
                updated_at=now,
                source_rollout_id=self.rollout_id,
                source_step=(current.source_step if source_step is None else source_step),
            )
            history.append(tombstone)
            return tombstone.copy()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            record = self._active_unlocked(memory_id)
            return None if record is None else record.copy()

    def get_all(self) -> List[MemoryRecord]:
        with self._lock:
            return [
                record.copy()
                for memory_id in self._versions
                if (record := self._active_unlocked(memory_id)) is not None
            ]

    def history(self, memory_id: Optional[str] = None) -> List[MemoryRecord]:
        with self._lock:
            ids = [memory_id] if memory_id is not None else list(self._versions)
            return [
                record.copy()
                for item_id in ids
                for record in self._versions.get(item_id, [])
            ]

    def snapshot(self) -> MemoryStoreSnapshot:
        with self._lock:
            return MemoryStoreSnapshot(
                rollout_id=self.rollout_id,
                research_mode=self.research_mode,
                records=tuple(self.history()),
            )

    def restore(self, snapshot: MemoryStoreSnapshot) -> None:
        if snapshot.rollout_id != self.rollout_id:
            raise ValueError(
                "cannot restore snapshot from another rollout: "
                f"{snapshot.rollout_id!r} != {self.rollout_id!r}"
            )
        if snapshot.research_mode != self.research_mode:
            raise ValueError(
                "snapshot research_mode does not match target store: "
                f"{snapshot.research_mode!r} != {self.research_mode!r}"
            )
        restored: Dict[str, List[MemoryRecord]] = {}
        for record in snapshot.records:
            if record.source_rollout_id not in (None, self.rollout_id):
                raise ValueError(
                    "snapshot contains a record from another rollout: "
                    f"{record.source_rollout_id!r}"
                )
            normalized = replace(
                record,
                metadata=deepcopy(record.metadata),
                embedding=None if record.embedding is None else list(record.embedding),
                source_rollout_id=self.rollout_id,
            )
            restored.setdefault(record.memory_id, []).append(normalized)

        for memory_id, records in restored.items():
            versions = [record.version for record in records]
            if versions != list(range(1, len(records) + 1)):
                raise ValueError(
                    f"memory {memory_id!r} has non-contiguous versions: {versions}"
                )
            if sum(record.status == "active" for record in records) > 1:
                raise ValueError(f"memory {memory_id!r} has multiple active versions")
            if any(record.status != "superseded" for record in records[:-1]):
                raise ValueError(
                    f"memory {memory_id!r} has a non-superseded historical version"
                )
            if records[-1].status not in ("active", "discarded"):
                raise ValueError(
                    f"memory {memory_id!r} has no active or discarded terminal version"
                )

        with self._lock:
            self._versions = restored

    def reset(self) -> None:
        with self._lock:
            self._versions.clear()

    def size(self) -> int:
        with self._lock:
            return sum(
                self._active_unlocked(memory_id) is not None
                for memory_id in self._versions
            )


class RolloutMemoryStoreRegistry:
    """Own one independent store instance per rollout identifier."""

    def __init__(
        self,
        factory: Optional[Callable[[str], MemoryStore]] = None,
    ) -> None:
        self._factory = factory or (lambda rollout_id: InMemoryStore(rollout_id))
        self._lock = threading.RLock()
        self._stores: Dict[str, MemoryStore] = {}

    def get_or_create(self, rollout_id: str) -> MemoryStore:
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        with self._lock:
            store = self._stores.get(rollout_id)
            if store is None:
                store = self._factory(rollout_id)
                if store.rollout_id != rollout_id:
                    raise ValueError("store factory returned a mismatched rollout_id")
                self._stores[rollout_id] = store
            return store

    def reset(self, rollout_id: str) -> None:
        self.get_or_create(rollout_id).reset()

    def discard(self, rollout_id: str) -> bool:
        with self._lock:
            return self._stores.pop(rollout_id, None) is not None

    def rollout_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._stores)


__all__ = [
    "MEMORY_STORE_SCHEMA_VERSION",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStatus",
    "MemoryStore",
    "MemoryStoreSnapshot",
    "RolloutMemoryStoreRegistry",
]
