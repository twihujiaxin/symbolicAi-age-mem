"""AgeMem 训练工作流使用的长期记忆（LTM）实现。

阅读提示：
1. InMemoryVectorStore 只负责保存 MemoryItem 和按向量相似度排序；
2. MemoryManager 负责把自然语言转成 embedding，并提供增删改查接口；
3. 这是“每条 rollout 临时存在内存中”的教学/实验实现，不是持久化数据库。
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from AgeMem_code_agentscope.memory_store import (
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    MemoryStoreSnapshot,
    RolloutMemoryStoreRegistry,
)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个 embedding 的余弦相似度；无效或零向量按 0 分处理。"""
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
    """一条长期记忆：正文、可筛选元数据，以及用于语义检索的向量。"""

    memory_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    version: int = 1
    status: str = "active"
    source_rollout_id: Optional[str] = None
    source_step: Optional[int] = None


class InMemoryVectorStore:
    """A minimal thread-safe in-memory vector store for agent memories.

    This is intentionally simple and dependency-free. If you need persistence or
    ANN, swap this with a proper store and keep the same interface.
    """

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
        new_metadata: Optional[Dict[str, Any]] = None,
        new_embedding: Optional[Sequence[float]] = None,
        source_step: Optional[int] = None,
    ) -> bool:
        del source_step  # Legacy backend has no version metadata.
        with self._lock:
            item = self._items.get(memory_id)
            if item is None:
                return False
            if new_content is not None:
                item.content = new_content
            if new_embedding is not None:
                item.embedding = list(new_embedding)
            if new_metadata is not None:
                item.metadata.update(new_metadata)
            return True

    def delete(
        self,
        memory_id: str,
        source_step: Optional[int] = None,
    ) -> bool:
        del source_step  # Legacy backend has no version metadata.
        with self._lock:
            return self._items.pop(memory_id, None) is not None

    def clear(self) -> None:
        """Remove all stored memories."""
        with self._lock:
            self._items.clear()

    def count(self) -> int:
        """Return the number of memories currently stored."""
        with self._lock:
            return len(self._items)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        """先按 metadata 精确过滤，再按 embedding 余弦相似度取 Top-K。"""
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


class VersionedRolloutVectorStore:
    """Compatibility adapter from the M2 ``MemoryStore`` to AgeMem's API.

    The workflow and tools continue to consume ``MemoryItem`` objects while
    the backend keeps version history, research-mode tombstones and strict
    rollout ownership.  No embedding client is owned by this adapter.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._backend = store

    @property
    def rollout_id(self) -> str:
        return self._backend.rollout_id

    @property
    def backend(self) -> MemoryStore:
        return self._backend

    @staticmethod
    def _to_item(record: MemoryRecord) -> MemoryItem:
        return MemoryItem(
            memory_id=record.memory_id,
            content=record.content,
            metadata=dict(record.metadata),
            embedding=(None if record.embedding is None else list(record.embedding)),
            version=record.version,
            status=record.status,
            source_rollout_id=record.source_rollout_id,
            source_step=record.source_step,
        )

    def add(self, item: MemoryItem) -> None:
        self._backend.add(
            MemoryRecord(
                memory_id=item.memory_id,
                content=item.content,
                metadata=dict(item.metadata),
                embedding=(None if item.embedding is None else list(item.embedding)),
                source_rollout_id=self.rollout_id,
                source_step=item.source_step,
            )
        )

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        record = self._backend.get(memory_id)
        return None if record is None else self._to_item(record)

    def update(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Mapping[str, Any]] = None,
        new_embedding: Optional[Sequence[float]] = None,
        source_step: Optional[int] = None,
    ) -> bool:
        return (
            self._backend.update(
                memory_id,
                content=new_content,
                metadata=new_metadata,
                embedding=new_embedding,
                source_step=source_step,
            )
            is not None
        )

    def delete(self, memory_id: str, source_step: Optional[int] = None) -> bool:
        return (
            self._backend.delete(memory_id, source_step=source_step) is not None
        )

    def clear(self) -> None:
        self._backend.reset()

    def count(self) -> int:
        return self._backend.size()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        return [
            (self._to_item(record), score)
            for record, score in self._backend.retrieve(
                query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        ]

    def snapshot(self) -> MemoryStoreSnapshot:
        return self._backend.snapshot()

    def restore(self, snapshot: MemoryStoreSnapshot) -> None:
        self._backend.restore(snapshot)

    def history(self, memory_id: Optional[str] = None) -> List[MemoryItem]:
        return [self._to_item(record) for record in self._backend.history(memory_id)]


class MemoryManager:
    """LTM facade backed by one M2 store per rollout.

    Embedding remains a manager concern.  Tests and offline dry-runs can inject
    a deterministic function; production may retain the original DashScope
    provider without exposing it to the storage backend.
    """

    def __init__(
        self,
        embedding_model: str,
        embedding_dim: int,
        *,
        rollout_id: Optional[str] = None,
        registry: Optional[RolloutMemoryStoreRegistry] = None,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._registry = registry or RolloutMemoryStoreRegistry()
        self._embedding_function = embedding_function
        self.client: Optional[OpenAI] = None
        if embedding_function is None:
            resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            if not resolved_api_key:
                raise ValueError(
                    "DASHSCOPE_API_KEY environment variable is not set. "
                    "Please set it before running the workflow, e.g., export DASHSCOPE_API_KEY='your_key'"
                )
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=(
                    base_url
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
            )
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._rollout_id = ""
        self._store: VersionedRolloutVectorStore
        if rollout_id is None:
            # Construction happens before a concrete task/run is selected.
            # Keep this temporary store outside the registry so it cannot be
            # mistaken for a sampled rollout during isolation audits.
            temporary_id = f"unbound-{uuid.uuid4()}"
            self._rollout_id = temporary_id
            self._store = VersionedRolloutVectorStore(
                InMemoryStore(temporary_id)
            )
        else:
            self.bind_rollout(rollout_id, reset=True)

    @property
    def rollout_id(self) -> str:
        return self._rollout_id

    @property
    def store_registry(self) -> RolloutMemoryStoreRegistry:
        return self._registry

    @property
    def store(self) -> MemoryStore:
        return self._store.backend

    def bind_rollout(self, rollout_id: str, *, reset: bool = True) -> None:
        """Select the isolated store for ``rollout_id``.

        Retried rollout IDs start empty by default.  Existing snapshots can be
        restored explicitly after binding with ``reset=False``.
        """
        if not isinstance(rollout_id, str) or not rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        backend = self._registry.get_or_create(rollout_id)
        if reset:
            backend.reset()
        self._rollout_id = rollout_id
        self._store = VersionedRolloutVectorStore(backend)

    def embed(self, content: str):
        """把记忆文本编码成定长向量，供 add/update/retrieve 共用。"""
        if self._embedding_function is not None:
            embedding = list(self._embedding_function(content))
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    "embedding_function returned an unexpected dimension: "
                    f"{len(embedding)} != {self.embedding_dim}"
                )
            return embedding
        if self.client is None:
            raise RuntimeError("no embedding provider is configured")
        completion = self.client.embeddings.create(
            model=self.embedding_model,
            input=content,
            dimensions=self.embedding_dim,
            encoding_format="float",
        )
        json_response = completion.model_dump_json()
        response = json.loads(json_response)
        return response["data"][0]["embedding"]

    def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source_step: Optional[int] = None,
    ) -> bool:
        """写入记忆时立即计算 embedding，之后检索无需重复编码正文。"""
        if not content:
            return False
        embedding = self.embed(content)
        self._store.add(
            MemoryItem(
                memory_id=memory_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding,
                source_rollout_id=self.rollout_id,
                source_step=source_step,
            )
        )
        return True

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        source_step: Optional[int] = None,
    ) -> bool:
        item = self._store.get(memory_id)
        if item is None:
            return False

        # 正文变化时必须同步刷新 embedding，否则检索仍会使用旧语义。
        embedding = self.embed(content) if content is not None else None
        return self._store.update(
            memory_id,
            content,
            metadata,
            embedding,
            source_step,
        )

    def delete_memory(
        self,
        memory_id: str,
        *,
        source_step: Optional[int] = None,
    ) -> bool:
        return self._store.delete(memory_id, source_step)

    def clear(self) -> None:
        """Remove all memories from the store."""
        self._store.clear()

    def count(self) -> int:
        """Return the current number of long-term memories."""
        return self._store.count()

    def snapshot(self) -> MemoryStoreSnapshot:
        return self._store.snapshot()

    def restore(self, snapshot: MemoryStoreSnapshot) -> None:
        if snapshot.rollout_id != self.rollout_id:
            raise ValueError(
                "cannot restore snapshot from another rollout: "
                f"{snapshot.rollout_id!r} != {self.rollout_id!r}"
            )
        self._store.restore(snapshot)

    def history(self, memory_id: Optional[str] = None) -> List[MemoryItem]:
        return self._store.history(memory_id)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[MemoryItem]:
        """将查询编码后做语义检索；返回 MemoryItem，不把内容自动写入 STM。"""
        if not query:
            return []
        q_emb = self.embed(query)
        return [
            it
            for it, _ in self._store.search(
                q_emb, top_k=top_k, metadata_filter=metadata_filter
            )
        ]


class chat_client:
    """给摘要、相似度判断和 LLM-as-a-Judge 使用的辅助模型客户端。"""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable is not set. "
                "Please set it before running the workflow, e.g., export DASHSCOPE_API_KEY='your_key'"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def chat(self, messages: List[Dict], model_name: str = "qwen-max") -> str:
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content
