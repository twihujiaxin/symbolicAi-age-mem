import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from AgeMem_code_agentscope.memory_store import RolloutMemoryStoreRegistry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_memory_module():
    path = REPOSITORY_ROOT / "trinity/common/workflows/memory_context/memory_store.py"
    module_name = "_agemem_m8a_memory_store_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


memory_store = load_memory_module()


def deterministic_embedding(text):
    return [float(len(text) + 1), float((sum(map(ord, text)) % 13) + 1)]


class M8ARolloutMemoryTest(unittest.TestCase):
    def make_manager(self, registry):
        return memory_store.MemoryManager(
            embedding_model="deterministic-test",
            embedding_dim=2,
            registry=registry,
            embedding_function=deterministic_embedding,
        )

    def test_rollouts_are_isolated_and_rebind_is_auditable(self):
        registry = RolloutMemoryStoreRegistry()
        manager = self.make_manager(registry)

        manager.bind_rollout("rollout-a")
        self.assertTrue(manager.add_memory("shared-id", "alpha", source_step=0))
        self.assertEqual(manager.count(), 1)
        snapshot_a = manager.snapshot()

        manager.bind_rollout("rollout-b")
        self.assertEqual(manager.count(), 0)
        self.assertTrue(manager.add_memory("shared-id", "beta", source_step=0))
        self.assertEqual(manager.retrieve("beta", top_k=1)[0].content, "beta")

        manager.bind_rollout("rollout-a", reset=False)
        self.assertEqual(manager.retrieve("alpha", top_k=1)[0].content, "alpha")
        self.assertEqual(manager.snapshot().to_dict(), snapshot_a.to_dict())
        self.assertEqual(registry.rollout_ids(), ["rollout-a", "rollout-b"])

    def test_update_and_delete_keep_version_history(self):
        manager = self.make_manager(RolloutMemoryStoreRegistry())
        manager.bind_rollout("rollout-versioned")
        manager.add_memory("m1", "old fact", source_step=1)
        self.assertTrue(manager.update_memory("m1", "new fact", source_step=2))
        self.assertTrue(manager.delete_memory("m1", source_step=3))

        history = manager.history("m1")
        self.assertEqual([item.version for item in history], [1, 2, 3])
        self.assertEqual(
            [item.status for item in history],
            ["superseded", "superseded", "discarded"],
        )
        self.assertEqual([item.source_step for item in history], [1, 2, 3])
        self.assertEqual(manager.count(), 0)

    def test_parallel_managers_share_registry_without_state_leakage(self):
        registry = RolloutMemoryStoreRegistry()

        def run(rollout_id, content):
            manager = self.make_manager(registry)
            manager.bind_rollout(rollout_id)
            manager.add_memory("m", content)
            return manager.snapshot().to_dict()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run, "rollout-1", "first").result()
            second = executor.submit(run, "rollout-2", "second").result()

        self.assertEqual(first["rollout_id"], "rollout-1")
        self.assertEqual(second["rollout_id"], "rollout-2")
        self.assertEqual(first["records"][0]["content"], "first")
        self.assertEqual(second["records"][0]["content"], "second")


if __name__ == "__main__":
    unittest.main()
