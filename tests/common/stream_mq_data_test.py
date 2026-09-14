import json
import shutil
import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.data_builder import (
    StreamingDataError,
    StreamingHotpotBuilder,
    validate_manifest,
)
from AgeMem_code_agentscope.streaming_memory.schema import StreamingBuildConfig
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    DebugLexicalTokenizer,
    TokenAccounting,
)


ROOT = Path(__file__).resolve().parents[2]


def row(index, title_prefix):
    titles = [f"{title_prefix}-{index}-{i}" for i in range(3)]
    sentences = [[f"Entity {index} fact {i}a. ", f"Entity {index} fact {i}b."] for i in range(3)]
    return {
        "id": f"q-{title_prefix}-{index}", "question": f"What is entity {index}?",
        "answer": f"Entity {index}", "type": "bridge", "level": "medium",
        "supporting_facts": {"title": titles[:2], "sent_id": [0, 1]},
        "context": {"title": titles, "sentences": sentences},
    }


class FakeSplit:
    def __init__(self, rows, fingerprint): self.rows, self._fingerprint = rows, fingerprint
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


class StreamingDataTest(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT / "runs" / "stream_mq_unit_scratch"
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        self.scratch.mkdir(parents=True)

    def tearDown(self):
        if self.scratch.exists():
            shutil.rmtree(self.scratch)

    def config(self):
        payload = json.loads((ROOT / "configs/stream_mq/debug.yaml").read_text(encoding="utf-8"))
        payload["data"]["debug_episode_counts"] = {"train": 2, "dev": 1, "test": 1}
        payload["data"]["candidate_query_target"] = 6
        payload["data"]["alpha_targets"] = [1.0]
        payload["environment"]["chunk_target_tokens"] = 30
        payload["environment"]["chunk_max_tokens"] = 50
        payload["environment"]["context_total_tokens"] = 220
        return StreamingBuildConfig.model_validate(payload)

    def test_config_freezes_visibility_and_training_invariants(self):
        config = self.config()
        self.assertEqual(config.protocol, "streaming_multiquery_v1")
        self.assertEqual(config.training["rollouts_per_history"], 4)
        self.assertFalse(config.training["reader_tokens_in_actor_loss"])

    def test_builder_splits_before_episode_and_keeps_gold_private(self):
        dataset = {
            "train": FakeSplit([row(i, "train") for i in range(20)], "train-fp"),
            "validation": FakeSplit([row(i, "valid") for i in range(20)], "valid-fp"),
            "test": FakeSplit([], "test-fp"),
        }
        accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())
        builder = StreamingHotpotBuilder(
            dataset=dataset, config=self.config(), accounting=accounting,
            source_path=ROOT, source_fingerprints={k: v._fingerprint for k, v in dataset.items()},
            repository_root=ROOT,
        )
        output = self.scratch / "output"
        manifest = builder.build(output)
        self.assertEqual(manifest.split_counts, {"train": 2, "dev": 1, "test": 1})
        result = validate_manifest(output / "manifest.json")
        self.assertEqual(result["queries"], 16)
        public = (output / "episodes.public.jsonl").read_text(encoding="utf-8")
        self.assertNotIn('"question":', public)
        self.assertNotIn('"answer":', public)
        self.assertNotIn('"split":', public)
        self.assertNotIn("source_question_id", public)
        audit = json.loads((output / "split_audit.jsonl").read_text())
        self.assertTrue(audit["split_first"])
        self.assertEqual(audit["exact_paragraph_overlap"], 0)

    def test_mutating_build_refuses_identity_conflict(self):
        dataset = {
            "train": FakeSplit([row(i, "train") for i in range(20)], "train-fp"),
            "validation": FakeSplit([row(i, "valid") for i in range(20)], "valid-fp"),
            "test": FakeSplit([], "test-fp"),
        }
        builder = StreamingHotpotBuilder(
            dataset=dataset, config=self.config(),
            accounting=TokenAccounting.from_tokenizer(DebugLexicalTokenizer()),
            source_path=ROOT, source_fingerprints={k: v._fingerprint for k, v in dataset.items()},
            repository_root=ROOT,
        )
        output = self.scratch / "output"; output.mkdir(); (output / "foreign").write_text("x")
        with self.assertRaises(StreamingDataError): builder.build(output)


if __name__ == "__main__": unittest.main()
