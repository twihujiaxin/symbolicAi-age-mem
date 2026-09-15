import json
import shutil
import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.config import load_dynamic_config
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import (
    DynamicBuildConfig,
    DynamicHistoryPublic,
    DynamicQueryPrivate,
    PrivateEvent,
    project_public_observation,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.world_generator import (
    DynamicWorldBuilder,
    validate_dynamic_manifest,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.world_oracle import WorldOracle
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    DebugLexicalTokenizer,
    TokenAccounting,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[2]


def event(index, value, effective, relation="owner", entity="project"):
    text = f"{entity} {relation} {value} {effective}"
    return PrivateEvent(
        event_id=f"e{index}",
        history_id="h",
        entity=entity,
        relation=relation,
        value=value,
        observed_at=index,
        effective_at=effective,
        source_ref=f"s{index}",
        source_text_hash=sha256_text(text),
    )


def query(kind, time=None, answers=("placeholder",), join=None, target=None):
    return DynamicQueryPrivate(
        query_id="q",
        history_id="h",
        task_family="historical_state",
        query_kind=kind,
        entity="project",
        relation="owner",
        query_time=time,
        join_relation=join,
        answers=answers,
        answer_available_at=0,
        target_value=target,
        support_alternatives=(("e0",),),
        answer_type="entity",
        answerable=True,
    )


class DynamicOracleTest(unittest.TestCase):
    def test_current_historical_half_open_and_a_b_a(self):
        oracle = WorldOracle((event(0, "A", 1), event(1, "B", 4), event(2, "A", 7)))
        self.assertEqual(oracle.solve(query("state_at", 3)).answers, ("A",))
        self.assertEqual(oracle.solve(query("state_at", 4)).answers, ("B",))
        self.assertEqual(oracle.solve(query("state_at", 5)).answers, ("B",))
        self.assertEqual(oracle.solve(query("current_state")).answers, ("A",))

    def test_temporal_join_uses_one_query_time(self):
        events = (
            event(0, "Alice", 1),
            event(1, "Bob", 5),
            event(2, "Red", 1, "team", "Alice"),
            event(3, "Blue", 4, "team", "Alice"),
            event(4, "Green", 1, "team", "Bob"),
        )
        oracle = WorldOracle(events)
        q = query("temporal_join", 4, join="team")
        self.assertEqual(oracle.solve(q).answers, ("Blue",))

    def test_unknown_does_not_become_none(self):
        oracle = WorldOracle((event(0, "A", 3),))
        solved = oracle.solve(query("state_at", 1))
        self.assertFalse(solved.answerable)
        self.assertEqual(solved.answers, ("unknown",))


class DynamicBuildTest(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "runs" / "dynamic_v2_unit_scratch"
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        config = load_dynamic_config(ROOT / "configs/stream_dynamic_v2/data_debug.yaml")
        raw = config.model_dump(mode="json")
        raw["data"]["d1_family_counts"] = {"train": 3, "dev": 2, "test": 1}
        raw["data"]["d0_fixture_count"] = 42
        raw["budget"]["context_total_tokens"] = 512
        raw["budget"]["chunk_target_tokens"] = 64
        raw["budget"]["chunk_max_tokens"] = 80
        self.config = DynamicBuildConfig.model_validate(raw)
        self.accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())

    def tearDown(self):
        if self.scratch.exists():
            shutil.rmtree(self.scratch)

    def test_split_first_private_public_and_independent_labels(self):
        manifest = DynamicWorldBuilder(
            config=self.config,
            accounting=self.accounting,
            repository_root=ROOT,
        ).build(self.scratch)
        report = validate_dynamic_manifest(self.scratch / "manifest.json")
        self.assertEqual(report["histories"], 6)
        self.assertEqual(report["queries"], 60)
        self.assertEqual(report["independent_oracle_issues"], 0)
        self.assertEqual(report["cross_split_family_overlap"], 0)
        self.assertEqual(manifest.statistics["d0_fixture_count"], 42)
        self.assertGreater(manifest.statistics["duplicate_source_text_records"], 0)
        self.assertGreaterEqual(manifest.statistics["alpha_min"], 5.0)
        self.assertTrue(manifest.statistics["feasible_reference_within_B"])
        self.assertTrue(manifest.statistics["feasible_reference_within_retrieval_cap"])
        self.assertTrue(manifest.statistics["feasible_reference_within_C"])
        public = (self.scratch / "histories.public.jsonl").read_text(encoding="utf-8")
        for forbidden in (
            "answers",
            "support_alternatives",
            "effective_at",
            "event_id",
        ):
            self.assertNotIn(f'"{forbidden}"', public)

    def test_same_public_prefix_cannot_encode_hidden_future(self):
        DynamicWorldBuilder(
            config=self.config,
            accounting=self.accounting,
            repository_root=ROOT,
        ).build(self.scratch)
        history = DynamicHistoryPublic.model_validate_json(
            (self.scratch / "histories.public.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        changed_future = history.model_copy(
            update={
                "chunks": (history.chunks[0],)
                + tuple(
                    item.model_copy(update={"text": "different hidden future"})
                    for item in history.chunks[1:]
                )
            }
        )
        first = project_public_observation(history, 0, ())
        second = project_public_observation(changed_future, 0, ())
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertNotIn("question", json.dumps(first))

    def test_infeasible_reference_is_marked_not_silently_promoted(self):
        raw = self.config.model_dump(mode="json")
        raw["budget"]["persistent_memory_tokens"] = 1
        config = DynamicBuildConfig.model_validate(raw)
        manifest = DynamicWorldBuilder(
            config=config,
            accounting=self.accounting,
            repository_root=ROOT,
        ).build(self.scratch / "small_b")
        self.assertFalse(manifest.statistics["feasible_reference_within_B"])


if __name__ == "__main__":
    unittest.main()
