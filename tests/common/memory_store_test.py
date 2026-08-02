import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from AgeMem_code_agentscope.agent import AgeMem
from AgeMem_code_agentscope.memory import AgentScopeLongtermMemory
from AgeMem_code_agentscope.memory_store import (
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    MemoryStoreSnapshot,
    RolloutMemoryStoreRegistry,
)


def deterministic_embed(content):
    total = sum(ord(character) for character in content)
    return [float((total % 17) + 1), float((len(content) % 11) + 1)]


class TickClock:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.value += 1
            return f"2026-08-02T00:00:{self.value:02d}+00:00"


class DelegatingStore:
    """A non-InMemoryStore backend used to verify protocol injection."""

    def __init__(self, rollout_id):
        self.backend = InMemoryStore(rollout_id, clock=TickClock())
        self.calls = []

    @property
    def rollout_id(self):
        return self.backend.rollout_id

    @property
    def research_mode(self):
        return self.backend.research_mode

    def add(self, record):
        self.calls.append("add")
        return self.backend.add(record)

    def retrieve(self, query_embedding, top_k=5, metadata_filter=None):
        self.calls.append("retrieve")
        return self.backend.retrieve(query_embedding, top_k, metadata_filter)

    def update(
        self,
        memory_id,
        *,
        content=None,
        metadata=None,
        embedding=None,
        source_step=None,
    ):
        self.calls.append("update")
        return self.backend.update(
            memory_id,
            content=content,
            metadata=metadata,
            embedding=embedding,
            source_step=source_step,
        )

    def delete(self, memory_id, *, source_step=None):
        self.calls.append("delete")
        return self.backend.delete(memory_id, source_step=source_step)

    def get(self, memory_id):
        return self.backend.get(memory_id)

    def get_all(self):
        return self.backend.get_all()

    def history(self, memory_id=None):
        return self.backend.history(memory_id)

    def snapshot(self):
        self.calls.append("snapshot")
        return self.backend.snapshot()

    def restore(self, snapshot):
        self.calls.append("restore")
        return self.backend.restore(snapshot)

    def reset(self):
        self.calls.append("reset")
        return self.backend.reset()

    def size(self):
        return self.backend.size()


class InMemoryStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore("rollout-a", clock=TickClock())

    def add_record(self, memory_id="memory-1", content="fact-v1", **kwargs):
        return self.store.add(
            MemoryRecord(
                memory_id=memory_id,
                content=content,
                metadata=kwargs.pop("metadata", {"kind": "fact"}),
                embedding=kwargs.pop("embedding", [1.0, 0.0]),
                source_step=kwargs.pop("source_step", 0),
                **kwargs,
            )
        )

    def test_implements_protocol_and_retrieves_only_active_records(self):
        self.assertIsInstance(self.store, MemoryStore)
        added = self.add_record()

        self.assertEqual(added.version, 1)
        self.assertEqual(added.status, "active")
        self.assertEqual(added.source_rollout_id, "rollout-a")
        results = self.store.retrieve([1.0, 0.0], metadata_filter={"kind": "fact"})
        self.assertEqual([record.memory_id for record, _ in results], ["memory-1"])
        self.assertEqual(self.store.size(), 1)

    def test_update_appends_version_and_preserves_old_evidence(self):
        self.add_record(metadata={"kind": "fact", "owner": "alice"})
        updated = self.store.update(
            "memory-1",
            content="fact-v2",
            metadata={"owner": "bob"},
            embedding=[0.0, 1.0],
            source_step=3,
        )

        self.assertIsNotNone(updated)
        history = self.store.history("memory-1")
        self.assertEqual([record.version for record in history], [1, 2])
        self.assertEqual([record.status for record in history], ["superseded", "active"])
        self.assertEqual(history[0].content, "fact-v1")
        self.assertEqual(history[0].metadata["owner"], "alice")
        self.assertEqual(history[1].content, "fact-v2")
        self.assertEqual(history[1].metadata["owner"], "bob")
        self.assertEqual(history[1].source_step, 3)
        self.assertEqual(self.store.get("memory-1"), history[1])

    def test_research_delete_is_soft_and_auditable(self):
        self.add_record()
        tombstone = self.store.delete("memory-1", source_step=4)

        self.assertIsNotNone(tombstone)
        self.assertEqual(tombstone.status, "discarded")
        self.assertEqual(tombstone.version, 2)
        self.assertEqual(tombstone.source_step, 4)
        self.assertIsNone(self.store.get("memory-1"))
        self.assertEqual(self.store.retrieve([1.0, 0.0]), [])
        self.assertEqual(self.store.get_all(), [])
        self.assertEqual(self.store.size(), 0)
        self.assertEqual(
            [record.status for record in self.store.history("memory-1")],
            ["superseded", "discarded"],
        )

    def test_snapshot_restore_and_reset_are_exact_and_deep(self):
        self.add_record()
        self.store.update(
            "memory-1",
            content="fact-v2",
            metadata={"revision": 2},
            embedding=[0.0, 1.0],
        )
        snapshot = self.store.snapshot()
        expected = snapshot.to_dict()
        serialized_snapshot = MemoryStoreSnapshot.from_dict(expected)

        self.store.reset()
        self.assertEqual(self.store.history(), [])
        self.store.restore(serialized_snapshot)
        self.assertEqual(self.store.snapshot().to_dict(), expected)

        restored = self.store.get("memory-1")
        restored.metadata["external_mutation"] = True
        self.assertNotIn("external_mutation", self.store.get("memory-1").metadata)

    def test_restore_rejects_cross_rollout_state(self):
        self.add_record()
        other = InMemoryStore("rollout-b")
        with self.assertRaisesRegex(ValueError, "another rollout"):
            other.restore(self.store.snapshot())

    def test_duplicate_id_and_missing_mutations_are_safe(self):
        self.add_record()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.add_record()
        self.assertIsNone(self.store.update("missing", content="none"))
        self.assertIsNone(self.store.delete("missing"))


class RolloutIsolationTest(unittest.TestCase):
    def test_registry_returns_one_independent_store_per_rollout(self):
        registry = RolloutMemoryStoreRegistry()
        first = registry.get_or_create("rollout-a")
        repeated = registry.get_or_create("rollout-a")
        second = registry.get_or_create("rollout-b")

        self.assertIs(first, repeated)
        self.assertIsNot(first, second)
        self.assertEqual(registry.rollout_ids(), ["rollout-a", "rollout-b"])

    def test_parallel_rollouts_cannot_observe_each_other(self):
        registry = RolloutMemoryStoreRegistry()
        barrier = threading.Barrier(2)

        def execute(rollout_id, content, embedding):
            store = registry.get_or_create(rollout_id)
            store.add(
                MemoryRecord(
                    memory_id="shared-logical-id",
                    content=content,
                    metadata={"owner": rollout_id},
                    embedding=embedding,
                )
            )
            barrier.wait()
            own = store.retrieve(embedding)
            other_embedding = [embedding[1], embedding[0]]
            cross = store.retrieve(other_embedding, metadata_filter={"owner": "other"})
            return store.snapshot(), own, cross

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(execute, "rollout-a", "alpha", [1.0, 0.0])
            future_b = executor.submit(execute, "rollout-b", "beta", [0.0, 1.0])
            result_a = future_a.result()
            result_b = future_b.result()

        self.assertEqual(result_a[0].rollout_id, "rollout-a")
        self.assertEqual(result_b[0].rollout_id, "rollout-b")
        self.assertEqual(result_a[1][0][0].content, "alpha")
        self.assertEqual(result_b[1][0][0].content, "beta")
        self.assertEqual(result_a[2], [])
        self.assertEqual(result_b[2], [])


class AgentScopeMemoryAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_tools_delegate_through_protocol_backend(self):
        backend = DelegatingStore("adapter-rollout")
        memory = AgentScopeLongtermMemory(
            store=backend,
            embedding_function=deterministic_embed,
        )
        agent = AgeMem(
            name="Protocol-Agent",
            sys_prompt="test",
            model=object(),
            formatter=object(),
            memory=memory,
            chat_client=object(),
        )

        added = await agent.add_memory("fact-v1", {"kind": "fact"})
        memory_id = added.metadata["memory_id"]
        retrieved = await agent.retrieve_memory("fact-v1", 1)
        self.assertTrue(retrieved.metadata["success"])
        updated = await agent.update_memory(
            memory_id,
            "fact-v2",
            {"revision": 2},
        )
        deleted = await agent.delete_memory(memory_id, confirmation=True)
        self.assertTrue(updated.metadata["success"])
        self.assertTrue(deleted.metadata["success"])

        self.assertEqual(
            backend.calls[:4],
            ["add", "retrieve", "update", "delete"],
        )
        self.assertEqual(await memory.get_memory(), [])
        self.assertEqual(
            [record.status for record in await memory.get_memory_history(memory_id)],
            ["superseded", "superseded", "discarded"],
        )

    async def test_manager_snapshot_restore_preserves_version_history(self):
        memory = AgentScopeLongtermMemory(
            rollout_id="manager-rollout",
            embedding_function=deterministic_embed,
        )
        await memory.add("memory-1", "fact-v1", {"kind": "fact"})
        await memory.update("memory-1", "fact-v2", {"revision": 2})
        expected = memory.state_dict()

        memory.reset()
        self.assertEqual(await memory.size(), 0)
        memory.load_state_dict(expected)
        self.assertEqual(memory.state_dict(), expected)

    async def test_agent_and_memory_rollout_must_match(self):
        memory = AgentScopeLongtermMemory(
            rollout_id="rollout-a",
            embedding_function=deterministic_embed,
        )
        agent = AgeMem(
            name="M2-Agent",
            sys_prompt="test",
            model=object(),
            formatter=object(),
            memory=memory,
            chat_client=object(),
        )
        self.assertEqual(agent.rollout_id, "rollout-a")
        self.assertEqual(agent.memory_manager.store.rollout_id, "rollout-a")

        with self.assertRaisesRegex(ValueError, "another rollout"):
            AgeMem(
                name="M2-Mismatch",
                sys_prompt="test",
                model=object(),
                formatter=object(),
                memory=memory,
                chat_client=object(),
                rollout_id="rollout-b",
            )


if __name__ == "__main__":
    unittest.main()
