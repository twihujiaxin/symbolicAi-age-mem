import asyncio
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from AgeMem_code_agentscope.hotpotqa_benchmark import (
    HotpotQADataAdapter,
    HotpotQADataError,
    HotpotQAOracleBenchmark,
    HotpotQARow,
    HotpotQASmokeConfig,
    HotpotQASmokeManifest,
    OracleBenchmarkError,
    OracleBenchmarkReport,
    answer_f1,
    load_manifest,
    stable_fact_id,
)
from AgeMem_code_agentscope.hotpotqa_benchmark.metrics import report_digest
from AgeMem_code_agentscope.toy_hotpotqa import HotpotQAToyEnvironment, ToyAction
from AgeMem_code_agentscope.trajectory import TrajectoryReplay


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOTPOTQA = REPOSITORY_ROOT.parent / "data" / "hotpot_qa" / "fullwiki"
SMOKE_MANIFEST = (
    REPOSITORY_ROOT / "data" / "splits" / "hotpotqa_smoke_manifest.json"
)


@contextmanager
def workspace_temp_directory():
    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"m5-hotpotqa-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _labeled_row(index, hotpot_type, support_count):
    hotpot_id = f"fixture-{index:02d}"
    titles = [f"Alpha {index}", f"Beta {index}", f"Noise {index}"]
    sentences = [
        [f"Alpha fact {index}.", f"Alpha extra {index}."],
        [f"Beta fact {index}.", f"Beta second fact {index}."],
        [f"Noise one {index}.", f"Noise two {index}.", f"Noise three {index}."],
    ]
    pointers = [(titles[0], 0), (titles[1], 0)]
    if support_count == 3:
        pointers.append((titles[1], 1))
    return {
        "id": hotpot_id,
        "question": f"What is the hidden answer for fixture {index}?",
        "answer": f"HiddenAnswer{index}",
        "type": hotpot_type,
        "level": "hard",
        "supporting_facts": {
            "title": [title for title, _ in pointers],
            "sent_id": [sent_id for _, sent_id in pointers],
        },
        "context": {"title": titles, "sentences": sentences},
    }


def _blind_row(index):
    row = _labeled_row(100 + index, "bridge", 3)
    row["answer"] = None
    row["type"] = None
    row["level"] = None
    row["supporting_facts"] = {"title": [], "sent_id": []}
    return row


def _fixture_dataset():
    train = [
        _labeled_row(0, "bridge", 3),
        _labeled_row(1, "comparison", 2),
        _labeled_row(2, "bridge", 2),
        _labeled_row(3, "comparison", 2),
    ]
    validation = [
        _labeled_row(10, "bridge", 3),
        _labeled_row(11, "comparison", 2),
        _labeled_row(12, "bridge", 3),
        _labeled_row(13, "comparison", 2),
        _labeled_row(14, "bridge", 2),
        _labeled_row(15, "comparison", 2),
    ]
    return {"train": train, "validation": validation, "test": [_blind_row(0)]}


def _fixture_config():
    return HotpotQASmokeConfig(
        seed=42,
        train_size=2,
        dev_size=2,
        test_size=2,
        min_supporting_facts=2,
        max_supporting_facts=3,
        stage1_distractors=1,
        stage2_distractors=1,
        policies=("gold", "wrong_answer", "missing_support"),
    )


class HotpotQAAdapterUnitTest(unittest.TestCase):
    def setUp(self):
        self.adapter = HotpotQADataAdapter(dataset_dict=_fixture_dataset())
        self.config = _fixture_config()

    def test_smoke_manifest_is_deterministic_disjoint_and_split_safe(self):
        first = self.adapter.build_smoke_manifest(self.config)
        second = self.adapter.build_smoke_manifest(self.config)
        self.assertEqual(first, second)
        self.adapter.verify_manifest(first, self.config)

        ids = [item.hotpot_id for item in first.selections]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(first.split_sizes, {"train": 2, "dev": 2, "test": 2})
        self.assertTrue(
            all(
                item.source_split == "train"
                for item in first.selections
                if item.benchmark_split == "train"
            )
        )
        self.assertTrue(
            all(
                item.source_split == "validation"
                for item in first.selections
                if item.benchmark_split in {"dev", "test"}
            )
        )
        for split in ("train", "dev", "test"):
            self.assertTrue(
                any(
                    item.benchmark_split == split
                    and item.supporting_fact_count >= 3
                    for item in first.selections
                )
            )

    def test_supporting_facts_use_exact_pointer_and_stable_ids(self):
        manifest = self.adapter.build_smoke_manifest(self.config)
        selection = manifest.selections[0]
        task = self.adapter.adapt(selection, self.config)
        row = self.adapter.row(selection.source_split, selection.source_index)

        self.assertEqual(
            [(item.title, item.sent_id) for item in task.supporting_fact_pointers],
            list(row.supporting_facts.pairs()),
        )
        for pointer in task.supporting_fact_pointers:
            fact = task.fact(pointer.fact_id)
            self.assertEqual(fact.role, "supporting")
            self.assertEqual(fact.stage, 1)
            self.assertEqual(
                pointer.fact_id,
                stable_fact_id(row.id, fact.title, fact.sent_id, fact.sentence),
            )

        bad = _labeled_row(90, "bridge", 3)
        bad["context"]["title"][0] = f"{bad['context']['title'][0]} Extended"
        with self.assertRaisesRegex(HotpotQADataError, "absent"):
            self.adapter._resolve_supporting(HotpotQARow.model_validate(bad))

    def test_public_stage_input_has_no_private_answer_or_oracle_fields(self):
        selection = self.adapter.build_smoke_manifest(self.config).selections[0]
        task = self.adapter.adapt(selection, self.config)
        changed_answer = task.model_copy(update={"answer": "PRIVATE-ANSWER-CHANGED"})

        async def collect_public_inputs(private_task):
            environment = HotpotQAToyEnvironment(
                private_task,
                rollout_id="m5-answer-invisibility",
                seed=self.config.seed,
            )
            inputs = [environment.stage_input()]
            await environment.step(ToyAction(kind="advance"))
            inputs.append(environment.stage_input())
            await environment.step(ToyAction(kind="advance"))
            inputs.append(environment.stage_input())
            return inputs

        first = asyncio.run(collect_public_inputs(task))
        second = asyncio.run(collect_public_inputs(changed_answer))
        self.assertEqual(first, second)
        for stage_input in first:
            public = stage_input.model_dump(mode="json")
            self.assertEqual(
                set(public),
                {
                    "task_id",
                    "rollout_id",
                    "seed",
                    "stage",
                    "observation",
                    "allowed_actions",
                },
            )
            self.assertNotIn("Expected Answer:", public["observation"])
            self.assertNotIn("supporting_fact_ids", public)
            self.assertNotIn("oracle_labels", public)

    def test_official_test_labels_fail_closed(self):
        self.assertEqual(self.adapter.validate_official_test_is_label_blind(), 1)
        leaking = _fixture_dataset()
        leaking["test"][0]["answer"] = "leaked"
        with self.assertRaisesRegex(HotpotQADataError, "hidden labels"):
            HotpotQADataAdapter(
                dataset_dict=leaking
            ).validate_official_test_is_label_blind()

    def test_manifest_schema_rejects_cross_source_split(self):
        manifest = self.adapter.build_smoke_manifest(self.config)
        payload = manifest.model_dump(mode="python")
        payload["selections"][0]["source_split"] = "validation"
        with self.assertRaises(ValidationError):
            HotpotQASmokeManifest.model_validate(payload)

    def test_official_yes_no_f1_rule_and_step_budget_are_explicit(self):
        self.assertEqual(answer_f1("no", "yes"), 0.0)
        self.assertEqual(answer_f1("yes indeed", "yes"), 0.0)
        self.assertEqual(answer_f1("yes", "yes"), 1.0)
        with self.assertRaisesRegex(ValidationError, "balancing requires"):
            HotpotQASmokeConfig(
                seed=1,
                train_size=2,
                dev_size=2,
                test_size=2,
                min_supporting_facts=2,
                max_supporting_facts=2,
                stage1_distractors=1,
                stage2_distractors=1,
            )


class HotpotQAOracleBenchmarkTest(unittest.IsolatedAsyncioTestCase):
    async def test_collect_replay_reward_report_is_offline_and_deterministic(self):
        adapter = HotpotQADataAdapter(dataset_dict=_fixture_dataset())
        config = _fixture_config()
        manifest = adapter.build_smoke_manifest(config)

        with workspace_temp_directory() as directory, mock.patch(
            "AgeMem_code_agentscope.memory.OpenAI",
            side_effect=AssertionError("OpenAI client must not be constructed"),
        ), mock.patch(
            "AgeMem_code_agentscope.src.llm_client.chat_client.chat",
            side_effect=AssertionError("LLM must not be called"),
        ):
            first = await HotpotQAOracleBenchmark(adapter).run(
                config=config,
                manifest=manifest,
                runtime_root=directory / "first-runtime",
                report_root=directory / "first-report",
            )
            second = await HotpotQAOracleBenchmark(adapter).run(
                config=config,
                manifest=manifest,
                runtime_root=directory / "second-runtime",
                report_root=directory / "second-report",
            )

            self.assertEqual(first.report, second.report)
            self.assertEqual(first.report_path.read_bytes(), second.report_path.read_bytes())
            self.assertEqual(first.failures_path.read_bytes(), second.failures_path.read_bytes())

            first_record = first.report.records[0]
            trajectory_path = first.runtime_root / first_record.trajectory_path
            replay = TrajectoryReplay.from_jsonl(trajectory_path)
            one = replay.replay(
                task_id=first_record.task_id,
                rollout_id=replay.steps[0].rollout_id,
                require_complete=True,
            )
            two = replay.replay(
                task_id=first_record.task_id,
                rollout_id=replay.steps[0].rollout_id,
                require_complete=True,
            )
            self.assertEqual(one, two)

        self.assertEqual(len(first.report.records), 18)
        gold = [item for item in first.report.records if item.policy == "gold"]
        wrong = [
            item for item in first.report.records if item.policy == "wrong_answer"
        ]
        missing = [
            item for item in first.report.records if item.policy == "missing_support"
        ]
        self.assertEqual(len(gold), 6)
        self.assertTrue(all(item.episode_success and item.dfa_accepted for item in gold))
        self.assertTrue(
            all(
                item.answer_em == 1.0
                and item.supporting_fact_coverage == 1.0
                and item.memory_precision == 1.0
                for item in gold
            )
        )
        self.assertTrue(
            all(
                item.answer_em == 0.0
                and item.supporting_fact_coverage == 1.0
                and not item.dfa_accepted
                for item in wrong
            )
        )
        self.assertTrue(
            all(
                item.answer_em == 1.0
                and item.supporting_fact_coverage < 1.0
                and not item.dfa_accepted
                for item in missing
            )
        )
        self.assertEqual(len(first.report.failures), 12)
        self.assertTrue(all(item.automaton_trace for item in first.report.failures))

        tampered = first.report.canonical_dict()
        tampered["aggregates"][0]["answer_em"] = 0.5
        without_digest = dict(tampered)
        without_digest.pop("digest")
        tampered["digest"] = report_digest(without_digest)
        with self.assertRaisesRegex(ValidationError, "aggregates do not match"):
            OracleBenchmarkReport.model_validate(tampered)

    async def test_reward_step_budget_fails_before_collecting_trajectories(self):
        adapter = HotpotQADataAdapter(dataset_dict=_fixture_dataset())
        config = _fixture_config()
        manifest = adapter.build_smoke_manifest(config)
        over_budget = config.model_copy(update={"max_supporting_facts": 5})
        with workspace_temp_directory() as directory, self.assertRaisesRegex(
            OracleBenchmarkError, "DFA replay budget"
        ):
            await HotpotQAOracleBenchmark(adapter).run(
                config=over_budget,
                manifest=manifest,
                runtime_root=directory / "runtime",
            )


@unittest.skipUnless(
    LOCAL_HOTPOTQA.is_dir() and SMOKE_MANIFEST.is_file(),
    "local HotpotQA fullwiki data and committed smoke manifest are required",
)
class LocalHotpotQAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = HotpotQADataAdapter(LOCAL_HOTPOTQA)
        cls.config = HotpotQASmokeConfig.model_validate_json(
            (REPOSITORY_ROOT / "configs" / "m5_hotpotqa_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        cls.manifest = load_manifest(SMOKE_MANIFEST)

    def test_local_source_schema_split_and_label_blind_contracts(self):
        self.assertEqual(
            {
                split: self.adapter.split_size(split)
                for split in ("train", "validation", "test")
            },
            {"train": 90447, "validation": 7405, "test": 7405},
        )
        self.assertEqual(self.adapter.validate_official_test_is_label_blind(), 7405)
        self.adapter.verify_manifest(self.manifest, self.config)

    def test_local_smoke_tasks_resolve_every_official_support_pointer(self):
        for selection in self.manifest.selections:
            task = self.adapter.adapt(selection, self.config)
            self.assertEqual(
                len(task.supporting_fact_ids), selection.supporting_fact_count
            )
            self.assertEqual(
                set(task.supporting_fact_ids),
                {pointer.fact_id for pointer in task.supporting_fact_pointers},
            )
            self.assertTrue(
                all(task.fact(fact_id).stage == 1 for fact_id in task.supporting_fact_ids)
            )


if __name__ == "__main__":
    unittest.main()
