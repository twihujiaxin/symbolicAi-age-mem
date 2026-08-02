import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from AgeMem_code_agentscope.memory_oracle import (
    AutomatonSpec,
    AutomatonTransition,
    MemoryOracleGrounder,
    OfflineRewardReplay,
    OracleGroundingError,
    RewardConfig,
    hand_authored_memory_dfa,
)
from AgeMem_code_agentscope.toy_hotpotqa import (
    ErrorMemoryPolicy,
    GoldMemoryPolicy,
    ToyAction,
    ToyEpisodeRunner,
    ToyTaskDataset,
)
from AgeMem_code_agentscope.trajectory import TrajectoryRecorder, TrajectoryReplay


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"m4-reward-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class StaticPolicy:
    def __init__(self, actions):
        self._actions = list(actions)

    def actions(self, task, seed):
        del task, seed
        return [action.model_copy(deep=True) for action in self._actions]


class MemoryOracleRewardTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ToyTaskDataset.from_json()
        cls.terminal_dfa = OfflineRewardReplay.from_config("terminal_dfa")
        cls.terminal_only = OfflineRewardReplay.from_config("terminal_only")

    async def record(self, directory, task, policy, rollout_id, seed=7):
        path = directory / f"{rollout_id}.jsonl"
        episode = await ToyEpisodeRunner().run(
            task,
            policy,
            rollout_id=rollout_id,
            seed=seed,
            recorder=TrajectoryRecorder(path),
        )
        return path, episode

    def test_reward_config_and_hand_authored_dfa_are_strict(self):
        config = RewardConfig.from_json(REPOSITORY_ROOT / "configs/m4_reward.json")
        self.assertEqual(set(config.profiles), {"terminal_only", "terminal_dfa"})
        self.assertEqual(config.profile("terminal_only").logic_beta, 0.0)
        self.assertEqual(config.profile("terminal_dfa").trend_weight, 0.0)

        spec = hand_authored_memory_dfa()
        self.assertEqual(spec.initial_state, "q0")
        self.assertEqual(spec.accepting_states, ("q4",))
        self.assertEqual(spec.rejecting_states, ("q_reject",))
        with self.assertRaises(ValidationError):
            AutomatonSpec(
                name="nondeterministic",
                states=("a", "b", "reject", "timeout"),
                initial_state="a",
                accepting_states=("b",),
                rejecting_states=("reject",),
                timeout_state="timeout",
                transitions=(
                    AutomatonTransition(
                        edge_id="one",
                        proposition="stored_supporting_fact",
                        source_states=("a",),
                        target_state="b",
                        priority=1,
                    ),
                    AutomatonTransition(
                        edge_id="two",
                        proposition="stored_supporting_fact",
                        source_states=("a",),
                        target_state="b",
                        priority=2,
                    ),
                ),
                source_milestones=("stored_supporting_fact",),
            )

    async def test_grounder_uses_semantic_labels_not_raw_tool_calls(self):
        task = self.dataset.get("toy-train-001")
        with workspace_temp_directory() as directory:
            path, _ = await self.record(
                directory, task, GoldMemoryPolicy(), "semantic-grounding"
            )
            first = TrajectoryReplay.from_jsonl(path).query(
                task_id=task.task_id,
                rollout_id="semantic-grounding",
                timestep=0,
            )[0]
        grounder = MemoryOracleGrounder(task)
        event = grounder.from_step(first)
        self.assertIn("stored_supporting_fact", event.propositions)
        self.assertIn("observed_supporting_fact", event.propositions)

        metadata = dict(first.tool_results[0].metadata)
        metadata["oracle_labels"] = {
            key: value
            for key, value in metadata["oracle_labels"].items()
            if key in {"supporting_coverage_complete", "answer_correct"}
        }
        empty_result = first.tool_results[0].model_copy(
            update={"metadata": metadata}, deep=True
        )
        raw_add_only = first.model_copy(
            update={"tool_results": [empty_result]}, deep=True
        )
        raw_event = grounder.from_step(raw_add_only)
        self.assertEqual(raw_add_only.tool_calls[0].name, "Add_memory")
        self.assertNotIn("stored_supporting_fact", raw_event.propositions)

    async def test_all_gold_trajectories_are_accepted(self):
        failures = []
        with workspace_temp_directory() as directory:
            for task in self.dataset.all():
                rollout_id = f"m4-gold-{task.task_id}"
                path, _ = await self.record(
                    directory, task, GoldMemoryPolicy(), rollout_id, seed=11
                )
                result = self.terminal_dfa.replay_jsonl(
                    path, task=task, rollout_id=rollout_id
                )
                if not result.accepted:
                    failures.append((task.task_id, result.final_status))
                self.assertEqual(result.env_total, 1.0)
                self.assertTrue(
                    all(step.reward.trend == 0.0 for step in result.steps)
                )
                self.assertTrue(
                    all(step.reward.format == 0.0 for step in result.steps)
                )
        self.assertEqual(failures, [])

    async def test_predefined_failure_trajectories_are_rejected(self):
        cases = (
            ("wrong_answer", "toy-train-001"),
            ("missing_support", "toy-train-001"),
            ("stale_retrieval", "toy-train-013"),
            ("delete_support", "toy-train-017"),
        )
        with workspace_temp_directory() as directory:
            for mode, task_id in cases:
                with self.subTest(mode=mode):
                    task = self.dataset.get(task_id)
                    rollout_id = f"m4-failure-{mode}"
                    path, _ = await self.record(
                        directory,
                        task,
                        ErrorMemoryPolicy(mode),
                        rollout_id,
                    )
                    result = self.terminal_dfa.replay_jsonl(
                        path, task=task, rollout_id=rollout_id
                    )
                    self.assertFalse(result.accepted)
                    self.assertEqual(result.final_status, "rejected")
                    self.assertEqual(result.env_total, 0.0)

    async def test_repeated_add_and_retrieve_cannot_farm_reward(self):
        task = self.dataset.get("toy-train-009")
        gold_actions = GoldMemoryPolicy().actions(task, 3)
        repeated_actions = list(gold_actions)
        first_retrieve = next(
            index
            for index, action in enumerate(repeated_actions)
            if action.kind == "retrieve"
        )
        repeated_actions.insert(first_retrieve + 1, repeated_actions[first_retrieve])
        first_advance = next(
            index for index, action in enumerate(repeated_actions) if action.kind == "advance"
        )
        repeated_actions.insert(
            first_advance,
            ToyAction(kind="add", fact_id=task.supporting_fact_ids[0]),
        )

        with workspace_temp_directory() as directory:
            gold_path, _ = await self.record(
                directory, task, GoldMemoryPolicy(), "farming-gold", seed=3
            )
            repeated_path, _ = await self.record(
                directory,
                task,
                StaticPolicy(repeated_actions),
                "farming-repeated",
                seed=3,
            )
            gold = self.terminal_dfa.replay_jsonl(
                gold_path, task=task, rollout_id="farming-gold"
            )
            repeated = self.terminal_dfa.replay_jsonl(
                repeated_path, task=task, rollout_id="farming-repeated"
            )

        self.assertTrue(gold.accepted)
        self.assertTrue(repeated.accepted)
        self.assertEqual(repeated.milestone_total, gold.milestone_total)
        zero_reward_calls = [
            step
            for step in repeated.steps
            if not step.event.propositions and step.reward.total == 0.0
        ]
        self.assertGreaterEqual(len(zero_reward_calls), 2)

    async def test_update_progress_edge_is_rewarded_once_only(self):
        task = self.dataset.get("toy-train-013")
        gold_actions = GoldMemoryPolicy().actions(task, 5)
        update_index = next(
            index for index, action in enumerate(gold_actions) if action.kind == "update"
        )
        repeated_actions = list(gold_actions)
        repeated_actions.insert(update_index + 1, gold_actions[update_index])

        with workspace_temp_directory() as directory:
            path, episode = await self.record(
                directory,
                task,
                StaticPolicy(repeated_actions),
                "repeated-update",
                seed=5,
            )
            result = self.terminal_dfa.replay_jsonl(
                path, task=task, rollout_id="repeated-update"
            )

        update_steps = [
            step
            for step in result.steps
            if "updated_stale_fact" in step.event.propositions
        ]
        self.assertTrue(episode.episode.success)
        self.assertTrue(result.accepted)
        self.assertEqual(len(update_steps), 2)
        self.assertEqual(update_steps[0].reward.milestone, 0.25)
        self.assertEqual(update_steps[1].reward.milestone, 0.0)
        self.assertIn("progress_update_stale", update_steps[1].reward.fired_edges)
        self.assertNotIn(
            "progress_update_stale", update_steps[1].reward.newly_rewarded_edges
        )

    async def test_loop_trajectory_times_out_without_accumulating_reward(self):
        task = self.dataset.get("toy-train-001")
        loop_actions = [
            ToyAction(kind="add", fact_id=task.supporting_fact_ids[0]),
            ToyAction(kind="advance"),
            ToyAction(kind="advance"),
        ]
        loop_actions.extend(
            ToyAction(kind="retrieve", fact_id=task.supporting_fact_ids[0])
            for _ in range(15)
        )
        with workspace_temp_directory() as directory:
            path, episode = await self.record(
                directory,
                task,
                StaticPolicy(loop_actions),
                "loop-timeout",
                seed=2,
            )
            first = self.terminal_dfa.replay_jsonl(
                path, task=task, rollout_id="loop-timeout"
            )
            second = self.terminal_dfa.replay_jsonl(
                path, task=task, rollout_id="loop-timeout"
            )

        self.assertFalse(episode.episode.done)
        self.assertEqual(first, second)
        self.assertEqual(first.final_status, "timed_out")
        self.assertFalse(first.accepted)
        self.assertEqual(first.milestone_total, 0.25)
        self.assertEqual(first.total_reward, 0.25)

    async def test_irrelevant_memory_events_are_violations_not_progress(self):
        task = self.dataset.get("toy-train-005")
        actions = GoldMemoryPolicy().actions(task, 4)
        first_advance = next(
            index for index, action in enumerate(actions) if action.kind == "advance"
        )
        actions.insert(first_advance, ToyAction(kind="add", fact_id="t005-d1"))
        first_retrieve = next(
            index for index, action in enumerate(actions) if action.kind == "retrieve"
        )
        actions.insert(first_retrieve, ToyAction(kind="retrieve", fact_id="t005-d1"))

        with workspace_temp_directory() as directory:
            path, _ = await self.record(
                directory,
                task,
                StaticPolicy(actions),
                "irrelevant-violations",
                seed=4,
            )
            result = self.terminal_dfa.replay_jsonl(
                path, task=task, rollout_id="irrelevant-violations"
            )

        violation_edges = {
            edge for step in result.steps for edge in step.reward.violation_edges
        }
        self.assertTrue(result.accepted)
        self.assertEqual(
            violation_edges,
            {"violation_store_irrelevant", "violation_retrieve_irrelevant"},
        )
        self.assertEqual(result.violation_total, 0.0)

    async def test_terminal_only_and_terminal_dfa_save_separate_components(self):
        task = self.dataset.get("toy-test-001")
        with workspace_temp_directory() as directory:
            source_path, _ = await self.record(
                directory, task, GoldMemoryPolicy(), "reward-profiles", seed=19
            )
            dfa = self.terminal_dfa.replay_jsonl(
                source_path, task=task, rollout_id="reward-profiles"
            )
            terminal = self.terminal_only.replay_jsonl(
                source_path, task=task, rollout_id="reward-profiles"
            )
            first_output = directory / "reward-first.jsonl"
            second_output = directory / "reward-second.jsonl"
            dfa.write_jsonl(first_output)
            repeated = self.terminal_dfa.replay_jsonl(
                source_path,
                task=task,
                rollout_id="reward-profiles",
                output_path=second_output,
            )
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(dfa.digest, repeated.digest)

        self.assertTrue(dfa.accepted)
        self.assertEqual(dfa.env_total, 1.0)
        self.assertEqual(dfa.milestone_total, 1.0)
        self.assertEqual(dfa.format_total, 0.0)
        self.assertEqual(dfa.total_reward, 2.0)
        self.assertEqual(terminal.total_reward, 1.0)

    async def test_invalid_oracle_ids_fail_closed_and_no_llm_is_called(self):
        task = self.dataset.get("toy-train-001")
        with workspace_temp_directory() as directory:
            path, _ = await self.record(
                directory, task, GoldMemoryPolicy(), "invalid-label", seed=23
            )
            first = TrajectoryReplay.from_jsonl(path).query(
                task_id=task.task_id, rollout_id="invalid-label", timestep=0
            )[0]
            metadata = dict(first.tool_results[0].metadata)
            labels = dict(metadata["oracle_labels"])
            labels["stored_supporting_fact_ids"] = ["not-a-task-fact"]
            metadata["oracle_labels"] = labels
            invalid_result = first.tool_results[0].model_copy(
                update={"metadata": metadata}, deep=True
            )
            invalid_step = first.model_copy(
                update={"tool_results": [invalid_result]}, deep=True
            )
            with self.assertRaisesRegex(
                OracleGroundingError, "semantically invalid fact IDs"
            ):
                MemoryOracleGrounder(task).from_step(invalid_step)

            with mock.patch(
                "AgeMem_code_agentscope.memory.OpenAI",
                side_effect=AssertionError("OpenAI client must not be constructed"),
            ), mock.patch(
                "AgeMem_code_agentscope.src.llm_client.chat_client.chat",
                side_effect=AssertionError("LLM must not be called"),
            ):
                result = self.terminal_dfa.replay_jsonl(
                    path, task=task, rollout_id="invalid-label"
                )
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
