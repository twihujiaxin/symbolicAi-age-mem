"""Contract tests for the format-conditioned 4B question-retrieve bench.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_fc_question_retrieve import (
    ALL_JOBS,
    CHECKPOINT_ROOT,
    EXPECTED_REVISION,
    FORBIDDEN_FOREIGN_JOBS,
    JOB,
    QUESTION_RETRIEVE_TOP_K,
    WORKFLOW_EXTRA,
    load_lock,
    render_bench,
    write_runtime_yaml,
)
from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS as DIAGNOSIS_JOBS,
    DIAGNOSIS_FLAG_STRINGS,
    FROZEN_CLEAN_YAMLS,
    load_lock as load_fc_lock,
    selection_is_frozen,
)


def _load_module(name: str, path: Path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_question_retrieve.sh"
WORKFLOW = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "train_hotpotQA.py"
)
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
GROUP_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
DIAG_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag.sh"
PILOT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_pilot.sh"
DRY_RUN_4B = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
METRICS_PATH = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "workflow_metrics.py"
)
STORE_PATH = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "memory_store.py"
)


def _embed(text: str) -> list[float]:
    return [float(len(text)), float(sum(ord(ch) for ch in text) % 13)]


class QuestionRetrieveContractTest(unittest.TestCase):
    def test_lock_and_flags_stay_off_gold(self):
        lock = load_lock()
        self.assertEqual(lock["schema_version"], "agemem.e1_4b_fc_question_retrieve.lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_question_retrieve")
        self.assertEqual(lock["checkpoint_root"], CHECKPOINT_ROOT)
        self.assertEqual(lock["protocol_lock"], "configs/e1_4b_format_conditioned.json")
        self.assertTrue(lock["stage3_question_retrieve"])
        self.assertTrue(lock["stage3_index_observed_context"])
        self.assertFalse(lock["stage3_inject_gold_supporting"])
        self.assertEqual(lock["stage3_question_retrieve_top_k"], QUESTION_RETRIEVE_TOP_K)
        self.assertEqual(lock["jobs"]["bench"], JOB)
        self.assertEqual(lock["model"]["expected_revision"], EXPECTED_REVISION)
        self.assertIn("stage3_question_retrieve: true", WORKFLOW_EXTRA)
        self.assertIn("stage3_index_observed_context: true", WORKFLOW_EXTRA)
        self.assertNotIn("stage3_inject_gold_supporting: true", WORKFLOW_EXTRA)

    def test_runtime_yaml_requires_frozen_dev(self):
        fc_lock = load_fc_lock()
        if not selection_is_frozen(fc_lock):
            with self.assertRaises(ValueError):
                render_bench(fc_lock)
            return
        text = render_bench(fc_lock)
        self.assertIn(f'name: "{JOB}"', text)
        self.assertIn("mode: bench", text)
        self.assertIn("repeat_times: 1", text)
        self.assertIn("temperature: 0.0", text)
        self.assertIn("stage3_question_retrieve: true", text)
        self.assertIn("stage3_index_observed_context: true", text)
        self.assertNotIn("stage3_inject_gold_supporting", text)
        self.assertNotIn("stage3_disable_ltm_retrieve", text)
        self.assertNotIn("consume_put_batch", text)
        for row in fc_lock["fixed_dev_rows"]:
            self.assertIn(row["hotpot_id"], text)
        with tempfile.TemporaryDirectory() as raw:
            path = write_runtime_yaml(Path(raw), fc_lock)
            self.assertEqual(path.read_text(encoding="utf-8"), text)

    def test_observed_context_ignores_gold_labels(self):
        metrics = _load_module("qr_workflow_metrics", METRICS_PATH)
        context = {
            "title": ["Alpha", "Beta"],
            "sentences": [
                ["Alpha is a fruit.", "Alpha grows on trees.", "Ignore me later."],
                ["Beta is a city."],
            ],
        }
        supporting = {"title": ["Alpha"], "sent_id": [1]}
        observed = metrics.observed_context_sentences(context)
        gold = metrics.extract_sentences_from_supporting_facts(supporting, context)
        self.assertEqual(
            observed,
            ["Alpha is a fruit.", "Alpha grows on trees.", "Ignore me later.", "Beta is a city."],
        )
        self.assertEqual(gold, ["Alpha grows on trees."])
        self.assertNotEqual(observed, gold)

    def test_hybrid_retrieve_prefers_question_overlap(self):
        store = _load_module("qr_memory_store", STORE_PATH)
        manager = store.MemoryManager(
            embedding_model="unused",
            embedding_dim=2,
            rollout_id="qr-test",
            embedding_function=_embed,
        )
        manager.add_memory("1", "The river Thames flows through London.")
        manager.add_memory("2", "Bananas are yellow fruit.")
        manager.add_memory("3", "London is the capital of England.")
        self.assertGreater(
            store.lexical_overlap("Where is London?", "London is the capital of England."),
            0.0,
        )
        self.assertGreater(
            store.hybrid_retrieve_score(0.1, 0.5),
            store.hybrid_retrieve_score(0.4, 0.0),
        )
        items = manager.retrieve_hybrid("Where is London the capital?", top_k=2)
        contents = [item.content for item in items]
        self.assertIn("London is the capital of England.", contents)
        self.assertNotIn("Bananas are yellow fruit.", contents)

    def test_workflow_flags_default_false(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('workflow_args.get("stage3_question_retrieve", False)', workflow)
        self.assertIn('workflow_args.get("stage3_index_observed_context", False)', workflow)
        self.assertIn("_prepend_stage3_question_retrieve", workflow)
        self.assertIn("elif self.stage3_question_retrieve", workflow)
        self.assertIn("retrieve_hybrid", workflow)
        for path in FROZEN_CLEAN_YAMLS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("stage3_question_retrieve", text, msg=path.name)
            self.assertNotIn("stage3_index_observed_context", text, msg=path.name)
        self.assertTrue(yaml_forbids_nudge(DRY_RUN_4B.read_text(encoding="utf-8")))
        for flag in DIAGNOSIS_FLAG_STRINGS:
            self.assertNotIn(flag, DRY_RUN_4B.read_text(encoding="utf-8"))

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_fc_question_retrieve_contract_test", gate)
        self.assertNotIn("agemem_e1_4b_fc_question_retrieve", gate)
        self.assertIn("configs/e1_4b_fc_question_retrieve.json", launcher)
        self.assertIn("checkpoints-e1-4b-fc-question-retrieve", launcher)
        self.assertIn("stage3_question_retrieve", launcher)
        self.assertIn("stage3_index_observed_context", launcher)
        self.assertIn("flash_attn", launcher)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        self.assertNotIn("agemem_e1_4b_fc_pilot.yaml", launcher)
        self.assertNotIn("stage3_inject_gold_supporting: true", launcher)
        for job in ALL_JOBS:
            self.assertIn(job, launcher)
        for job in DIAGNOSIS_JOBS:
            self.assertIn(job, launcher)
        for other in (
            VANILLA_LAUNCHER,
            FORMAT_LAUNCHER,
            VAR_LAUNCHER,
            GROUP_LAUNCHER,
            PROBE_LAUNCHER,
            DIAG_LAUNCHER,
            PILOT_LAUNCHER,
        ):
            text = other.read_text(encoding="utf-8")
            self.assertIn(JOB, text, msg=other.name)
        self.assertTrue(set(DIAGNOSIS_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))


if __name__ == "__main__":
    unittest.main()
