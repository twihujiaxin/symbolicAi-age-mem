import asyncio
import json
import shutil
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from AgeMem_code_agentscope.toy_hotpotqa import (
    ErrorMemoryPolicy,
    GoldMemoryPolicy,
    HotpotQAToyEnvironment,
    ToyAction,
    ToyEnvironmentPool,
    ToyEpisodeRunner,
    ToyTaskDataset,
)
from AgeMem_code_agentscope.trajectory import TrajectoryRecorder, TrajectoryReplay


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"toy-hotpot-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ToyDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ToyTaskDataset.from_json()

    def test_fixture_has_strict_20_5_5_splits_and_all_difficulties(self):
        self.assertEqual(len(self.dataset), 30)
        self.assertEqual(len(self.dataset.split("train")), 20)
        self.assertEqual(len(self.dataset.split("dev")), 5)
        self.assertEqual(len(self.dataset.split("test")), 5)
        difficulties = {
            difficulty
            for task in self.dataset.all()
            for difficulty in task.difficulty
        }
        self.assertTrue(
            {
                "clean",
                "distractor",
                "duplicate",
                "fact_update",
                "stale_fact",
                "critical_delete",
            }.issubset(difficulties)
        )
        for task in self.dataset.all():
            self.assertEqual(len(task.supporting_fact_ids), 2)
            self.assertIsInstance(task.distractor_fact_ids, tuple)
            self.assertIsInstance(task.stale_fact_ids, tuple)
            self.assertTrue(task.answer)

    def test_test_entity_combinations_are_unseen_in_train(self):
        train = {task.entity_signature() for task in self.dataset.split("train")}
        test = {task.entity_signature() for task in self.dataset.split("test")}
        self.assertFalse(train & test)

    def test_public_stage_input_does_not_expose_private_labels(self):
        task = self.dataset.get("toy-train-001")
        environment = HotpotQAToyEnvironment(
            task,
            rollout_id="public-view",
            seed=4,
        )
        public = environment.stage_input().model_dump(mode="json")
        serialized = json.dumps(public, sort_keys=True)

        self.assertNotIn("answer", public)
        self.assertNotIn("supporting_fact_ids", public)
        self.assertNotIn("oracle_labels", public)
        self.assertNotIn("fact_id", serialized)
        self.assertNotIn("delete", public["allowed_actions"])

    def test_fixed_seed_reproduces_order_and_different_seed_can_vary_it(self):
        task = self.dataset.get("toy-train-005")
        first = HotpotQAToyEnvironment(task, rollout_id="seed-a", seed=1)
        repeated = HotpotQAToyEnvironment(task, rollout_id="seed-b", seed=1)
        varied = HotpotQAToyEnvironment(task, rollout_id="seed-c", seed=2)

        self.assertEqual(
            first.stage_input().observation,
            repeated.stage_input().observation,
        )
        self.assertNotEqual(
            first.stage_input().observation,
            varied.stage_input().observation,
        )


class ToyEnvironmentTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ToyTaskDataset.from_json()

    async def test_stage_protocol_resets_stm_and_retains_ltm(self):
        task = self.dataset.get("toy-train-005")
        environment = HotpotQAToyEnvironment(
            task,
            rollout_id="stage-protocol",
            seed=3,
        )
        stage1_observation = environment.stage_input().observation
        await environment.step(ToyAction(kind="add", fact_id="t005-a"))
        self.assertEqual(await environment.memory.size(), 1)

        stage1_to_2 = await environment.step(ToyAction(kind="advance"))
        self.assertEqual(stage1_to_2.stage_after, 2)
        self.assertNotIn("Memory construction", environment.stage_input().observation)
        self.assertIn("Context interference", environment.stage_input().observation)
        self.assertNotEqual(stage1_observation, environment.stage_input().observation)
        self.assertEqual(await environment.memory.size(), 1)

        stage2_text = environment.stage_input().observation
        await environment.step(ToyAction(kind="advance"))
        stage3_text = environment.stage_input().observation
        self.assertIn(stage2_text, stage3_text)
        self.assertIn(task.question, stage3_text)
        self.assertEqual(await environment.memory.size(), 1)

    async def test_gold_policy_completes_all_tasks(self):
        failures = []
        for task in self.dataset.all():
            result = await ToyEpisodeRunner().run(
                task,
                GoldMemoryPolicy(),
                rollout_id=f"gold-{task.task_id}",
                seed=11,
            )
            if not result.episode.success:
                failures.append(task.task_id)
            self.assertTrue(result.episode.done)
            self.assertEqual(result.steps[-1].task_reward, 1.0)
        self.assertEqual(failures, [])

    async def test_obvious_error_policies_fail_expected_conditions(self):
        cases = [
            ("wrong_answer", "toy-train-001"),
            ("missing_support", "toy-train-001"),
            ("stale_retrieval", "toy-train-013"),
            ("delete_support", "toy-train-017"),
        ]
        for mode, task_id in cases:
            with self.subTest(mode=mode):
                result = await ToyEpisodeRunner().run(
                    self.dataset.get(task_id),
                    ErrorMemoryPolicy(mode),
                    rollout_id=f"error-{mode}",
                    seed=7,
                )
                self.assertTrue(result.episode.done)
                self.assertFalse(result.episode.success)
                self.assertEqual(result.steps[-1].task_reward, 0.0)

    async def test_duplicate_fact_is_ignored_and_repeated_add_is_rejected(self):
        task = self.dataset.get("toy-train-009")
        gold = await ToyEpisodeRunner().run(
            task,
            GoldMemoryPolicy(),
            rollout_id="duplicate-gold",
            seed=1,
        )
        active = [item for item in gold.final_memory if item.status == "active"]
        ignored = {
            fact_id
            for step in gold.steps
            for fact_id in step.labels.ignored_duplicate_fact_ids
        }
        self.assertEqual(len(active), 2)
        self.assertEqual(ignored, {"t009-dup"})

        repeated = await ToyEpisodeRunner().run(
            task,
            ErrorMemoryPolicy("duplicate_add"),
            rollout_id="duplicate-repeat",
            seed=1,
        )
        failed_adds = [
            step
            for step in repeated.steps
            if step.action.kind == "add" and not step.success
        ]
        self.assertEqual(len(failed_adds), 1)
        self.assertTrue(repeated.episode.success)
        self.assertEqual(
            len([item for item in repeated.final_memory if item.status == "active"]),
            2,
        )

    async def test_fact_update_is_versioned_and_stale_retrieval_is_labeled(self):
        task = self.dataset.get("toy-train-013")
        gold = await ToyEpisodeRunner().run(
            task,
            GoldMemoryPolicy(),
            rollout_id="update-gold",
            seed=5,
        )
        histories = {}
        for record in gold.store_snapshot.records:
            histories.setdefault(record.memory_id, []).append(record)
        version_chain = next(records for records in histories.values() if len(records) == 2)
        self.assertEqual([record.version for record in version_chain], [1, 2])
        self.assertEqual(
            [record.status for record in version_chain],
            ["superseded", "active"],
        )
        self.assertEqual(version_chain[0].metadata["fact_id"], "t013-old")
        self.assertEqual(version_chain[1].metadata["fact_id"], "t013-b")

        stale = await ToyEpisodeRunner().run(
            task,
            ErrorMemoryPolicy("stale_retrieval"),
            rollout_id="update-stale",
            seed=5,
        )
        stale_labels = {
            fact_id
            for step in stale.steps
            for fact_id in step.labels.retrieved_stale_fact_ids
        }
        self.assertEqual(stale_labels, {"t013-old"})
        self.assertFalse(stale.episode.success)

    async def test_critical_misdelete_is_soft_and_prevents_success(self):
        task = self.dataset.get("toy-train-017")
        result = await ToyEpisodeRunner().run(
            task,
            ErrorMemoryPolicy("delete_support"),
            rollout_id="critical-delete",
            seed=6,
        )
        deleted = {
            fact_id
            for step in result.steps
            for fact_id in step.labels.deleted_supporting_fact_ids
        }
        self.assertEqual(deleted, {"t017-a"})
        self.assertFalse(result.episode.success)
        deleted_history = [
            record
            for record in result.store_snapshot.records
            if record.memory_id.endswith(":t017-a")
        ]
        self.assertEqual(
            [record.status for record in deleted_history],
            ["superseded", "discarded"],
        )

    async def test_snapshot_restore_and_reset_resume_exactly(self):
        task = self.dataset.get("toy-train-013")
        environment = HotpotQAToyEnvironment(
            task,
            rollout_id="snapshot-resume",
            seed=9,
        )
        actions = GoldMemoryPolicy().actions(task, 9)
        await environment.step(actions[0])
        await environment.step(actions[1])
        checkpoint = environment.snapshot()

        first_result = await environment.step(actions[2])
        first_state = environment.snapshot()
        environment.reset()
        self.assertEqual(await environment.memory.size(), 0)
        environment.restore(checkpoint)
        second_result = await environment.step(actions[2])
        second_state = environment.snapshot()

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_state, second_state)

    async def test_shared_registry_keeps_rollouts_isolated(self):
        task = self.dataset.get("toy-train-001")
        pool = ToyEnvironmentPool()
        first = HotpotQAToyEnvironment(
            task,
            rollout_id="isolated-a",
            seed=1,
            pool=pool,
        )
        second = HotpotQAToyEnvironment(
            task,
            rollout_id="isolated-b",
            seed=1,
            pool=pool,
        )
        await first.step(ToyAction(kind="add", fact_id="t001-a"))

        self.assertEqual(await first.memory.size(), 1)
        self.assertEqual(await second.memory.size(), 0)
        with self.assertRaisesRegex(ValueError, "another rollout"):
            second.memory.restore(first.memory.snapshot())

    async def test_parallel_rollouts_remain_isolated(self):
        dataset = self.dataset
        shared_runner = ToyEpisodeRunner(ToyEnvironmentPool())

        def execute(task_id, rollout_id):
            async def run():
                return await shared_runner.run(
                    dataset.get(task_id),
                    GoldMemoryPolicy(),
                    rollout_id=rollout_id,
                    seed=13,
                )

            return asyncio.run(run())

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(execute, "toy-train-001", "parallel-a")
            second_future = executor.submit(execute, "toy-train-002", "parallel-b")
            first = first_future.result()
            second = second_future.result()

        self.assertTrue(first.episode.success)
        self.assertTrue(second.episode.success)
        self.assertTrue(
            all(record.source_rollout_id == "parallel-a" for record in first.store_snapshot.records)
        )
        self.assertTrue(
            all(record.source_rollout_id == "parallel-b" for record in second.store_snapshot.records)
        )


class ToyTrajectoryIntegrationTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ToyTaskDataset.from_json()

    async def test_gold_jsonl_is_complete_and_byte_deterministic(self):
        task = self.dataset.get("toy-test-004")
        with workspace_temp_directory() as directory:
            first_path = directory / "first.jsonl"
            second_path = directory / "second.jsonl"
            first = await ToyEpisodeRunner().run(
                task,
                GoldMemoryPolicy(),
                rollout_id="deterministic-rollout",
                seed=17,
                recorder=TrajectoryRecorder(first_path),
            )
            second = await ToyEpisodeRunner().run(
                task,
                GoldMemoryPolicy(),
                rollout_id="deterministic-rollout",
                seed=17,
                recorder=TrajectoryRecorder(second_path),
            )

            self.assertTrue(first.episode.success)
            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

            first_replay = TrajectoryReplay.from_jsonl(first_path).replay(
                task_id=task.task_id,
                rollout_id="deterministic-rollout",
                require_complete=True,
            )
            second_replay = TrajectoryReplay.from_jsonl(second_path).replay(
                task_id=task.task_id,
                rollout_id="deterministic-rollout",
                require_complete=True,
            )
            self.assertEqual(first_replay, second_replay)
            self.assertEqual(
                [step.timestep for step in first_replay.steps],
                list(range(len(first_replay.steps))),
            )
            for step in first_replay.steps:
                metadata = step.tool_results[0].metadata
                self.assertEqual(metadata["seed"], 17)
                self.assertEqual(metadata["task_id"], task.task_id)
                self.assertEqual(metadata["rollout_id"], "deterministic-rollout")
                self.assertIn("oracle_labels", metadata)
            self.assertTrue(
                first_replay.steps[0]
                .tool_results[0]
                .metadata["oracle_labels"]["observed_fact_ids"]
            )

    async def test_pipeline_does_not_construct_or_call_real_llm(self):
        task = self.dataset.get("toy-train-001")
        with mock.patch(
            "AgeMem_code_agentscope.memory.OpenAI",
            side_effect=AssertionError("OpenAI client must not be constructed"),
        ), mock.patch(
            "AgeMem_code_agentscope.src.llm_client.chat_client.chat",
            side_effect=AssertionError("LLM must not be called"),
        ):
            result = await ToyEpisodeRunner().run(
                task,
                GoldMemoryPolicy(),
                rollout_id="offline-only",
                seed=21,
            )
        self.assertTrue(result.episode.success)


if __name__ == "__main__":
    unittest.main()
