# -*- coding: utf-8 -*-
"""AgentScope adapter for backend-independent, rollout-scoped memory stores."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agentscope.memory import MemoryBase
from openai import OpenAI

from .memory_store import (
    MEMORY_STORE_SCHEMA_VERSION,
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    MemoryStoreSnapshot,
)


# Backward-compatible public name used by the standalone agent and earlier code.
MemoryItem = MemoryRecord


class AgentScopeLongtermMemory(MemoryBase):
    """Wrap a ``MemoryStore`` with AgentScope's asynchronous memory API.

    Embedding remains a manager concern. The injected store only owns state,
    versioning, rollout isolation, and vector lookup, so tools never depend on a
    concrete database implementation.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-v4",
        embedding_dim: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        rollout_id: Optional[str] = None,
        store: Optional[MemoryStore] = None,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        research_mode: bool = True,
    ) -> None:
        super().__init__()
        if store is not None and not isinstance(store, MemoryStore):
            raise TypeError("store must implement the MemoryStore protocol")
        if store is not None and rollout_id is not None and store.rollout_id != rollout_id:
            raise ValueError(
                "store rollout_id does not match requested rollout_id: "
                f"{store.rollout_id!r} != {rollout_id!r}"
            )

        resolved_rollout_id = (
            store.rollout_id if store is not None else rollout_id or str(uuid.uuid4())
        )
        self._store: MemoryStore = store or InMemoryStore(
            resolved_rollout_id,
            research_mode=research_mode,
        )
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._embedding_function = embedding_function
        self.client: Optional[OpenAI] = None
        if embedding_function is None:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url=(
                    base_url
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
            )

    @property
    def rollout_id(self) -> str:
        return self._store.rollout_id

    @property
    def store(self) -> MemoryStore:
        """Expose the abstract store for orchestration and audit code."""

        return self._store

    def embed(self, content: str) -> List[float]:
        if self._embedding_function is not None:
            return list(self._embedding_function(content))
        if self.client is None:
            raise RuntimeError("no embedding client or embedding function is configured")
        completion = self.client.embeddings.create(
            model=self.embedding_model,
            input=content,
            dimensions=self.embedding_dim,
            encoding_format="float",
        )
        data = json.loads(completion.model_dump_json())
        return data["data"][0]["embedding"]

    def snapshot(self) -> MemoryStoreSnapshot:
        return self._store.snapshot()

    def restore(self, snapshot: MemoryStoreSnapshot) -> None:
        self._store.restore(snapshot)

    def reset(self) -> None:
        self._store.reset()

    def state_dict(self) -> dict:
        """Return complete version history in the M1-compatible shape."""

        snapshot = self.snapshot()
        return {
            "memory_store_schema_version": snapshot.schema_version,
            "rollout_id": snapshot.rollout_id,
            "research_mode": snapshot.research_mode,
            "content": [record.to_dict() for record in snapshot.records],
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("memory state_dict must be a dictionary")
        content = state_dict.get("content", [])
        if not isinstance(content, list):
            raise ValueError("memory state_dict content must be a list")
        snapshot_rollout_id = state_dict.get("rollout_id", self.rollout_id)
        if strict and snapshot_rollout_id != self.rollout_id:
            raise ValueError(
                "cannot load memory state from another rollout: "
                f"{snapshot_rollout_id!r} != {self.rollout_id!r}"
            )

        records = []
        for raw_record in content:
            if not isinstance(raw_record, Mapping):
                raise ValueError("each memory state entry must be a dictionary")
            data = deepcopy(dict(raw_record))
            data.setdefault("memory_id", str(uuid.uuid4()))
            data.setdefault("metadata", {})
            data.setdefault("source_rollout_id", self.rollout_id)
            if data.get("embedding") is None:
                data["embedding"] = self.embed(data["content"])
            records.append(MemoryRecord.from_dict(data))

        snapshot = MemoryStoreSnapshot(
            schema_version=int(
                state_dict.get(
                    "memory_store_schema_version",
                    MEMORY_STORE_SCHEMA_VERSION,
                )
            ),
            rollout_id=self.rollout_id,
            research_mode=bool(
                state_dict.get("research_mode", self._store.research_mode)
            ),
            records=tuple(records),
        )
        self.restore(snapshot)

    async def size(self) -> int:
        return self._store.size()

    async def retrieve(
        self,
        query: str | None = None,
        top_k: int | None = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> List[MemoryRecord]:
        if not query:
            return []
        query_embedding = self.embed(query)
        return [
            record
            for record, _score in self._store.retrieve(
                query_embedding,
                top_k=top_k or 5,
                metadata_filter=metadata_filter,
            )
        ]

    async def delete(
        self,
        memory_id: str,
        *,
        source_step: Optional[int] = None,
    ) -> bool:
        return self._store.delete(memory_id, source_step=source_step) is not None

    async def add(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source_step: Optional[int] = None,
    ) -> None:
        embedding = self.embed(content)
        self._store.add(
            MemoryRecord(
                memory_id=memory_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding,
                source_rollout_id=self.rollout_id,
                source_step=source_step,
            )
        )

    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source_step: Optional[int] = None,
    ) -> bool:
        if self._store.get(memory_id) is None:
            return False
        embedding = self.embed(content) if content is not None else None
        return (
            self._store.update(
                memory_id,
                content=content,
                metadata=metadata,
                embedding=embedding,
                source_step=source_step,
            )
            is not None
        )

    async def get_memory(self) -> List[MemoryRecord]:
        """Return only current active records."""

        return self._store.get_all()

    async def get_memory_history(
        self,
        memory_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Return all versions, including discarded tombstones."""

        return self._store.history(memory_id)

    async def clear(self) -> None:
        self.reset()


__all__ = ["AgentScopeLongtermMemory", "MemoryItem"]
