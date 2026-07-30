# -*- coding: utf-8 -*-
"""
AgentScope-compatible long-term memory with embedding-based retrieval.
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agentscope.memory import MemoryBase
from openai import OpenAI


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryItem:
    memory_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryItem":
        if "memory_id" not in data:
            data["memory_id"] = str(uuid.uuid4())
        return MemoryItem(
            memory_id=data["memory_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )


class InMemoryVectorStore:
    """Thread-safe in-memory vector store for agent memories."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: Dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.memory_id] = item

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._items.get(memory_id)

    def update(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        with self._lock:
            item = self._items.get(memory_id)
            if item is None:
                return False
            if new_content is not None:
                item.content = new_content
            if new_metadata is not None:
                item.metadata.update(new_metadata)
            return True

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        with self._lock:
            scored: List[Tuple[MemoryItem, float]] = []
            for item in self._items.values():
                if metadata_filter and not all(
                    item.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue
                if item.embedding is None:
                    continue
                score = _cosine_similarity(query_embedding, item.embedding)
                if score > 0.0:
                    scored.append((item, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[: max(1, top_k)]

    def get_all_memory(self) -> List[MemoryItem]:
        with self._lock:
            return list(self._items.values())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def get_size(self) -> int:
        with self._lock:
            return len(self._items)


class AgentScopeLongtermMemory(MemoryBase):
    """Long-term memory with embedding-based retrieval, compatible with AgentScope."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-v4",
        embedding_dim: int = 256,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._store = InMemoryVectorStore()
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def embed(self, content: str) -> List[float]:
        completion = self.client.embeddings.create(
            model=self.embedding_model,
            input=content,
            dimensions=self.embedding_dim,
            encoding_format="float",
        )
        data = json.loads(completion.model_dump_json())
        return data["data"][0]["embedding"]

    def state_dict(self) -> dict:
        return {"content": [_.to_dict() for _ in self._store.get_all_memory()]}

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self._store.clear()
        for data in state_dict.get("content", []):
            if "embedding" not in data:
                data["embedding"] = self.embed(data["content"])
            if "metadata" not in data:
                data["metadata"] = {}
            self._store.add(MemoryItem.from_dict(data))

    async def size(self) -> int:
        return self._store.get_size()

    async def retrieve(
        self,
        query: str | None = None,
        top_k: int | None = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> List[MemoryItem]:
        if not query:
            return []
        q_emb = self.embed(query)
        return [
            it
            for it, _ in self._store.search(
                q_emb, top_k=top_k or 5, metadata_filter=metadata_filter
            )
        ]

    async def delete(self, memory_id: str) -> bool:
        return self._store.delete(memory_id)

    async def add(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        embedding = self.embed(content)
        self._store.add(
            MemoryItem(
                memory_id=memory_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding,
            )
        )

    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        if content is not None:
            embedding = self.embed(content)
            item = self._store.get(memory_id)
            if item is None:
                return False
            item.embedding = embedding
        return self._store.update(memory_id, content, metadata)

    async def get_memory(self) -> List[MemoryItem]:
        return self._store.get_all_memory()

    async def clear(self) -> None:
        self._store.clear()
