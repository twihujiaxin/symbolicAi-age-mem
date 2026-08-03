import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from AgeMem_code_agentscope.action_schema import (
    ACTION_EVENT_SCHEMA_VERSION,
    ActionCreditRecord,
    ActionEvent,
    RewardBreakdownV2,
    SchemaMigrationError,
    TrajectoryStepV2,
    load_migration_manifest,
    migrate_m5_canonical_report,
    migrate_m5_step,
)
from AgeMem_code_agentscope.hotpotqa_benchmark import (
    HotpotQADataAdapter,
    HotpotQAOracleBenchmark,
)
from AgeMem_code_agentscope.memory_oracle.models import (
    OracleAPEvent,
    RewardBreakdown,
    RewardedTrajectoryStep,
)
from AgeMem_code_agentscope.trajectory import TrajectoryStep
from tests.common.hotpotqa_oracle_benchmark_test import (
    _fixture_config,
    _fixture_dataset,
    workspace_temp_directory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
M5_REPORT = (
    REPOSITORY_ROOT / "artifacts" / "m5_hotpotqa_smoke" / "oracle_benchmark.json"
)
M5_RUNTIME = REPOSITORY_ROOT / "runs" / "m5_hotpotqa_smoke"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_source_files(report_path, runtime_root):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = [report_path]
    for record in report["records"]:
        paths.append(runtime_root / Path(record["trajectory_path"]))
        paths.append(runtime_root / Path(record["reward_path"]))
    return paths


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_models(root, relative_paths, model):
    rows = []
    for relative in relative_paths:
        with (root / relative).open("r", encoding="utf-8") as handle:
            rows.extend(model.model_validate_json(line) for line in handle)
    return rows


def _legacy_pair(*, old_logprob=None, call_count=1):
    rollout_id = "m5-train-fixture-gold"
    calls = [
        {
            "id": f"{rollout_id}:call:{index}",
            "name": "Add_memory",
            "input": {"fact_id": f"fact-{index}"},
        }
        for index in range(call_count)
    ]
    results = [
        {
            "tool_call_id": call["id"],
            "name": call["name"],
            "content": [],
            "metadata": {},
        }
        for call in calls
    ]
    step = TrajectoryStep(
        task_id="fixture-task",
        rollout_id=rollout_id,
        stage=1,
        timestep=0,
        observation="fixture observation",
        action_text="fixture action",
        tool_calls=calls,
        tool_results=results,
        memory_before=[],
        memory_after=[],
        env_reward=0.0,
        done=False,
        old_logprob=old_logprob,
    )
    event = OracleAPEvent(
        task_id=step.task_id,
        rollout_id=step.rollout_id,
        seed=7,
        timestep=0,
        stage=1,
        propositions=("stored_supporting_fact",),
        evidence_fact_ids={"stored_supporting_fact": ("fact-0",)},
    )
    reward = RewardBreakdown(
        task_id=step.task_id,
        rollout_id=step.rollout_id,
        seed=7,
        timestep=0,
        env=0.0,
        milestone=0.25,
        violation=0.0,
        trend=0.0,
        format=0.0,
        total=0.25,
        automaton_state_before="q0",
        automaton_state_after="q1",
        automaton_status="running",
        propositions=("stored_supporting_fact",),
        fired_edges=("progress_store_support",),
        newly_rewarded_edges=("progress_store_support",),
        violation_edges=(),
    )
    return step, RewardedTrajectoryStep(event=event, reward=reward)


class M6ActionSchemaTest(unittest.TestCase):
    def test_schema_versions_and_rule_token_metadata_are_strict(self):
        event = ActionEvent(
            action_id="action-1",
            task_id="task-1",
            rollout_id="rollout-1",
            stage_id=1,
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=0,
            source="oracle",
            action_type="Add_memory",
            action_text="{}",
            arguments={},
            result={},
        )
        self.assertEqual(event.schema_version, ACTION_EVENT_SCHEMA_VERSION)
        self.assertIsNone(event.response_token_ids)
        self.assertIsNone(event.old_logprobs)
        self.assertIsNone(event.policy_version)

        invalid_version = event.model_dump(mode="python")
        invalid_version["schema_version"] = "2"
        with self.assertRaises(ValidationError):
            ActionEvent.model_validate(invalid_version)

        missing_llm_tokens = event.model_dump(mode="python")
        missing_llm_tokens["source"] = "llm"
        with self.assertRaisesRegex(ValidationError, "LLM actions require"):
            ActionEvent.model_validate(missing_llm_tokens)

        partial_tokens = event.model_dump(mode="python")
        partial_tokens["response_token_ids"] = (1, 2)
        with self.assertRaisesRegex(ValidationError, "provided together"):
            ActionEvent.model_validate(partial_tokens)

    def test_transition_ids_preserve_order_without_inventing_singular_edge(self):
        breakdown = RewardBreakdownV2(
            env=0.0,
            milestone=0.5,
            violation=0.0,
            trend=0.0,
            format=0.0,
            cost=0.0,
            total=0.5,
            automaton_state_before="q1",
            automaton_state_after="q3",
            automaton_status="running",
            propositions=("supporting_coverage_complete", "retrieved_supporting_fact"),
            fired_edges=("progress_support_coverage", "progress_retrieve_support"),
            newly_rewarded_edges=(
                "progress_support_coverage",
                "progress_retrieve_support",
            ),
        )
        credit = ActionCreditRecord(
            action_id="action-1",
            task_id="task-1",
            rollout_id="rollout-1",
            stage_id=3,
            timestep=4,
            atomic_propositions=breakdown.propositions,
            dfa_spec_id="m4-memory-oracle-positive-v1",
            transition_ids=breakdown.fired_edges,
            transition_id=None,
            dfa_state_before="q1",
            dfa_state_after="q3",
            reward_breakdown=breakdown,
            reward_version="agemem.reward.test.v1",
        )
        self.assertEqual(credit.transition_ids, breakdown.fired_edges)
        self.assertIsNone(credit.transition_id)

        invalid = credit.model_dump(mode="python")
        invalid["transition_id"] = "progress_support_coverage"
        with self.assertRaisesRegex(ValidationError, "only when"):
            ActionCreditRecord.model_validate(invalid)

    def test_multi_action_llm_turn_requires_shared_non_overlapping_token_spans(self):
        common = {
            "task_id": "task-1",
            "rollout_id": "rollout-1",
            "stage_id": 1,
            "timestep": 0,
            "assistant_turn_id": 4,
            "source": "llm",
            "action_type": "tool",
            "action_text": "call",
            "arguments": {},
            "result": {},
            "response_token_ids": (10, 11, 12, 13),
            "old_logprobs": (-0.1, -0.2, -0.3, -0.4),
            "policy_version": "policy-v1",
        }
        first = ActionEvent(
            action_id="a-1",
            action_index_in_turn=0,
            token_start=0,
            token_end=2,
            **common,
        )
        second = ActionEvent(
            action_id="a-2",
            action_index_in_turn=1,
            token_start=2,
            token_end=4,
            **common,
        )
        step = TrajectoryStepV2(
            task_id="task-1",
            rollout_id="rollout-1",
            stage_id=1,
            timestep=0,
            observation="observation",
            actions=(first, second),
            memory_before=(),
            memory_after=(),
        )
        self.assertEqual(len(step.actions), 2)

        overlap = second.model_copy(update={"token_start": 1})
        with self.assertRaisesRegex(ValidationError, "non-overlapping"):
            TrajectoryStepV2(
                task_id="task-1",
                rollout_id="rollout-1",
                stage_id=1,
                timestep=0,
                observation="observation",
                actions=(first, overlap),
                memory_before=(),
                memory_after=(),
            )

    def test_legacy_scalar_and_unreliable_multi_action_step_fail_closed(self):
        scalar_step, reward = _legacy_pair(old_logprob=-0.4)
        with self.assertRaisesRegex(SchemaMigrationError, "scalar old_logprob"):
            migrate_m5_step(
                scalar_step,
                reward,
                source="oracle",
                dfa_spec_id="m4-memory-oracle-positive-v1",
            )

        multi_step, reward = _legacy_pair(call_count=2)
        with self.assertRaisesRegex(SchemaMigrationError, "exactly one tool call"):
            migrate_m5_step(
                multi_step,
                reward,
                source="oracle",
                dfa_spec_id="m4-memory-oracle-positive-v1",
            )


class PortableM5MigrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_report_driven_migration_is_read_only_joined_and_byte_stable(self):
        adapter = HotpotQADataAdapter(dataset_dict=_fixture_dataset())
        config = _fixture_config()
        manifest = adapter.build_smoke_manifest(config)
        with (
            workspace_temp_directory() as directory,
            mock.patch(
                "AgeMem_code_agentscope.memory.OpenAI",
                side_effect=AssertionError("OpenAI client must not be constructed"),
            ),
            mock.patch(
                "AgeMem_code_agentscope.src.llm_client.chat_client.chat",
                side_effect=AssertionError("LLM must not be called"),
            ),
        ):
            benchmark = await HotpotQAOracleBenchmark(adapter).run(
                config=config,
                manifest=manifest,
                runtime_root=directory / "runtime",
                report_root=directory / "report",
            )
            source_files = _canonical_source_files(
                benchmark.report_path, benchmark.runtime_root
            )
            source_hashes = {path: _sha256(path) for path in source_files}
            first = migrate_m5_canonical_report(
                benchmark.report_path,
                runtime_root=benchmark.runtime_root,
                output_root=directory / "first-migration",
            )
            second = migrate_m5_canonical_report(
                benchmark.report_path,
                runtime_root=benchmark.runtime_root,
                output_root=directory / "second-migration",
            )

            self.assertEqual(
                source_hashes, {path: _sha256(path) for path in source_files}
            )
            self.assertEqual(
                _tree_bytes(first.output_root), _tree_bytes(second.output_root)
            )
            self.assertEqual(first.manifest, second.manifest)
            self.assertEqual(
                load_migration_manifest(first.manifest_path), first.manifest
            )
            self.assertEqual(first.manifest.canonical_rollout_count, 18)
            self.assertEqual(
                first.manifest.action_count,
                first.manifest.credit_count,
            )
            self.assertEqual(
                first.manifest.action_count,
                first.manifest.joined_action_count,
            )

            trajectory_rows = _read_models(
                first.output_root,
                [item.target_trajectory_path for item in first.manifest.files],
                TrajectoryStepV2,
            )
            credit_rows = _read_models(
                first.output_root,
                [item.target_credit_path for item in first.manifest.files],
                ActionCreditRecord,
            )
            action_ids = [row.actions[0].action_id for row in trajectory_rows]
            credit_ids = [row.action_id for row in credit_rows]
            self.assertEqual(action_ids, credit_ids)
            self.assertEqual(len(action_ids), len(set(action_ids)))
            self.assertTrue(
                all(
                    row.actions[0].response_token_ids is None
                    and row.actions[0].old_logprobs is None
                    and row.actions[0].policy_version is None
                    for row in trajectory_rows
                )
            )
            self.assertTrue(
                all(row.reward_breakdown.cost == 0.0 for row in credit_rows)
            )
            self.assertTrue(any(len(row.transition_ids) == 2 for row in credit_rows))


def _canonical_m5_runtime_available():
    if not M5_REPORT.is_file() or not M5_RUNTIME.is_dir():
        return False
    try:
        return all(
            path.is_file() for path in _canonical_source_files(M5_REPORT, M5_RUNTIME)
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


@unittest.skipUnless(
    _canonical_m5_runtime_available(),
    "canonical M5 report-referenced runtime artifacts are required",
)
class CanonicalM5MigrationIntegrationTest(unittest.TestCase):
    def test_all_224_actions_join_and_twenty_double_edges_are_preserved(self):
        source_files = _canonical_source_files(M5_REPORT, M5_RUNTIME)
        before = {path: _sha256(path) for path in source_files}
        with workspace_temp_directory() as directory:
            first = migrate_m5_canonical_report(
                M5_REPORT,
                runtime_root=M5_RUNTIME,
                output_root=directory / "first",
            )
            second = migrate_m5_canonical_report(
                M5_REPORT,
                runtime_root=M5_RUNTIME,
                output_root=directory / "second",
            )
            self.assertEqual(
                _tree_bytes(first.output_root), _tree_bytes(second.output_root)
            )
            self.assertEqual(first.manifest.canonical_rollout_count, 30)
            self.assertEqual(first.manifest.action_count, 224)
            self.assertEqual(first.manifest.credit_count, 224)
            self.assertEqual(first.manifest.joined_action_count, 224)

            credits = _read_models(
                first.output_root,
                [item.target_credit_path for item in first.manifest.files],
                ActionCreditRecord,
            )
            self.assertEqual(sum(len(item.transition_ids) == 2 for item in credits), 20)
            self.assertTrue(
                all(
                    item.transition_id
                    == (
                        item.transition_ids[0]
                        if len(item.transition_ids) == 1
                        else None
                    )
                    for item in credits
                )
            )
        self.assertEqual(before, {path: _sha256(path) for path in source_files})


if __name__ == "__main__":
    unittest.main()
