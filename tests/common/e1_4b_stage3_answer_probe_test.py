"""Contract tests for the independent Qwen3-4B Stage-3 answer-format probe.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import E0_YAML, E1_YAML, EVAL_YAML, yaml_forbids_nudge
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b_stage3_answer_probe.json"
E1_4B_LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_4b.json"
ONE_POINT_FIVE_B_LOCK = REPOSITORY_ROOT / "configs" / "e1_stage3_answer_probe.json"
PROBE_YAML = (
    REPOSITORY_ROOT
    / "examples"
    / "agemem_hotpotqa"
    / "agemem_e1_4b_stage3_answer_probe.yaml"
)
ONE_POINT_FIVE_B_YAML = (
    REPOSITORY_ROOT
    / "examples"
    / "agemem_hotpotqa"
    / "agemem_e1_stage3_answer_probe.yaml"
)
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
PROBE_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
E1_4B_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"

M5_TRAIN_IDS = [
    "5a85aaee5542991dd0999e84",
    "5a74b19355429916b01641dd",
    "5abecbed5542997719eab5c5",
    "5a8ac7d055429950cd6afb8f",
    "5a83df2655429933447460a1",
    "5a76f83e55429972597f1405",
]


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


class E14BStage3AnswerProbeTest(unittest.TestCase):
    def test_lock_is_independent_of_1p5b_probe_and_4b_e1(self):
        lock = _lock()
        e1_4b = json.loads(E1_4B_LOCK_PATH.read_text(encoding="utf-8"))
        one_five = json.loads(ONE_POINT_FIVE_B_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], "agemem.e1_4b_stage3_answer_probe.lock.v1")
        self.assertEqual(lock["job_name"], "agemem-e1-4b-stage3-answer-probe")
        self.assertNotEqual(lock["job_name"], one_five["job_name"])
        self.assertNotEqual(lock["job_name"], e1_4b["paths"]["clean_job_relative_paths"][1].rsplit("/", 1)[-1])
        self.assertEqual(lock["one_point_five_b_probe_job_name"], "agemem-e1-stage3-answer-probe")
        self.assertEqual(lock["e1_4b_job_name"], "agemem-e1-terminal-only-4b-dry-run")
        self.assertEqual(lock["model"]["repository_id"], "Qwen/Qwen3-4B")
        self.assertEqual(
            lock["model"]["expected_revision"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertEqual(lock["stage3_max_rounds"], 2)
        self.assertEqual(lock["trainer_total_steps"], 0)
        self.assertEqual(lock["reward_profile"], "terminal_only")
        self.assertEqual(lock["source_train_row_ids"], M5_TRAIN_IDS)
        self.assertFalse(e1_4b.get("stage3_require_final_answer"))
        self.assertFalse(e1_4b.get("stage3_repair_untagged_answer"))

    def test_yaml_matches_lock_digest_and_enables_nudge(self):
        lock = _lock()
        yaml_text = PROBE_YAML.read_text(encoding="utf-8")
        self.assertEqual(_source_digest(PROBE_YAML), lock["config_sha256"])
        self.assertIn('name: "agemem-e1-4b-stage3-answer-probe"', yaml_text)
        self.assertNotIn(lock["smoke_job_name"], yaml_text)
        self.assertNotIn(lock["one_point_five_b_probe_job_name"], yaml_text)
        self.assertNotIn(lock["e1_4b_job_name"], yaml_text)
        self.assertNotIn("Qwen2.5-1.5B-Instruct", yaml_text)
        self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", yaml_text)
        self.assertIn("stage3_require_final_answer: true", yaml_text)
        self.assertIn("stage3_repair_untagged_answer: true", yaml_text)
        self.assertIn("stage3_max_rounds: 2", yaml_text)
        self.assertIn("mode: bench", yaml_text)
        self.assertIn("temperature: 0.0", yaml_text)
        self.assertIn("enable_thinking: false", yaml_text)
        self.assertIn("gpu_memory_utilization: 0.6", yaml_text)
        self.assertIn("ppo_max_token_len_per_gpu: 2304", yaml_text)
        self.assertIn("max_model_len: 5120", yaml_text)
        self.assertIn("max_response_tokens: 1024", yaml_text)
        self.assertIn("reward_profile: terminal_only", yaml_text)
        for row_id in M5_TRAIN_IDS:
            self.assertIn(row_id, yaml_text)
        self.assertNotEqual(_source_digest(PROBE_YAML), _source_digest(ONE_POINT_FIVE_B_YAML))

    def test_4b_e1_and_1p5b_dry_run_still_forbid_nudge(self):
        for path in (E0_YAML, E1_YAML, EVAL_YAML):
            self.assertTrue(yaml_forbids_nudge(path.read_text(encoding="utf-8")), msg=path.name)
        dry_run = DRY_RUN.read_text(encoding="utf-8")
        self.assertNotIn("stage3_require_final_answer: true", dry_run)
        self.assertNotIn("stage3_repair_untagged_answer: true", dry_run)

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        script = PROBE_SCRIPT.read_text(encoding="utf-8")
        e1_4b = E1_4B_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_stage3_answer_probe_test", gate)
        self.assertNotIn("e1_stage3_answer_probe_test", gate)
        self.assertNotIn("autodl_m8b_smoke.sh", script)
        self.assertNotIn("agemem_e1_dry_run.yaml", script)
        self.assertNotIn("agemem_e1_4b_dry_run.yaml", script)
        self.assertNotIn("agemem_e1_stage3_answer_probe.yaml", script)
        self.assertIn("agemem_e1_4b_stage3_answer_probe.yaml", script)
        self.assertIn("configs/e1_4b_stage3_answer_probe.json", script)
        self.assertIn("stage3_final_turn.jsonl", script)
        self.assertIn("Qwen3-4B", script)
        self.assertIn("agemem-e1-4b-stage3-answer-probe", e1_4b)
        self.assertNotIn("agemem_e1_4b_stage3_answer_probe.yaml", e1_4b)


if __name__ == "__main__":
    unittest.main()
