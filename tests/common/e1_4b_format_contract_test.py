"""Contract tests for the independent format-conditioned 4B GRPO protocol.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_format import (
    E0_YAML,
    E1_YAML,
    EVAL_YAML,
    FORBIDDEN_FOREIGN_JOBS,
    FORMAT_E0_JOB,
    FORMAT_E1_JOB,
    LOCK_PATH,
    VANILLA_E0_JOB,
    VANILLA_E1_JOB,
    VANILLA_LOCK_PATH,
    job_names,
    load_lock,
    yaml_requires_nudge,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "m8b_autodl_preflight.json"
VANILLA_E0_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e0_4b_frozen_eval.yaml"
)
VANILLA_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
)
VANILLA_EVAL_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_checkpoint_eval.yaml"
)
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
HELPER = REPOSITORY_ROOT / "trinity" / "common" / "e1_4b_format.py"

M5_TRAIN_IDS = [
    "5a85aaee5542991dd0999e84",
    "5a74b19355429916b01641dd",
    "5abecbed5542997719eab5c5",
    "5a8ac7d055429950cd6afb8f",
    "5a83df2655429933447460a1",
    "5a76f83e55429972597f1405",
]
M5_EVAL_IDS = [
    "5ab7c6995542993667794005",
    "5adc8c545542994734353734",
]


class E14BFormatContractTest(unittest.TestCase):
    def test_lock_is_independent_of_vanilla_4b_and_1p5b(self):
        lock = load_lock()
        vanilla = json.loads(VANILLA_LOCK_PATH.read_text(encoding="utf-8"))
        smoke = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        names = job_names(lock)
        self.assertEqual(lock["schema_version"], "agemem.m8b_preflight_lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_single_update")
        self.assertNotEqual(lock["experiment_id"], vanilla["experiment_id"])
        self.assertEqual(lock["model"]["repository_id"], "Qwen/Qwen3-4B")
        self.assertEqual(
            lock["model"]["expected_revision"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(lock["model"]["config_assertions"]["model_type"], "qwen3")
        self.assertNotEqual(lock["model"]["repository_id"], smoke["model"]["repository_id"])
        self.assertEqual(names["e0"], FORMAT_E0_JOB)
        self.assertEqual(names["e1"], FORMAT_E1_JOB)
        self.assertNotEqual(names["e0"], VANILLA_E0_JOB)
        self.assertNotEqual(names["e1"], VANILLA_E1_JOB)
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertFalse(vanilla.get("stage3_require_final_answer"))
        self.assertFalse(vanilla.get("stage3_repair_untagged_answer"))
        assertions = {
            entry["path"]: entry["equals"] for entry in lock["config_assertions"]
        }
        self.assertEqual(assertions["name"], names["e1"])
        self.assertEqual(assertions["explorer.rollout_model.seed"], 7)
        self.assertEqual(assertions["explorer.rollout_model.gpu_memory_utilization"], 0.6)
        self.assertEqual(
            assertions["trainer.trainer_config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
            2304,
        )
        self.assertEqual(assertions["model.max_model_len"], 5120)
        self.assertEqual(assertions["model.max_response_tokens"], 1024)
        self.assertEqual(assertions["trainer.total_steps"], 1)
        self.assertIs(assertions["buffer.explorer_input.taskset.workflow_args.milestone_reward_enabled"], False)
        self.assertIs(
            assertions["buffer.explorer_input.taskset.workflow_args.stage3_require_final_answer"],
            True,
        )
        self.assertIs(
            assertions["buffer.explorer_input.taskset.workflow_args.stage3_repair_untagged_answer"],
            True,
        )
        self.assertEqual(
            [row["hotpot_id"] for row in lock["dataset"]["fixed_train_rows"]],
            M5_TRAIN_IDS,
        )
        self.assertEqual(
            [row["hotpot_id"] for row in lock["dataset"]["fixed_eval_rows"]],
            M5_EVAL_IDS,
        )

    def test_yaml_matches_lock_digests_and_requires_nudge(self):
        lock = load_lock()
        sources = lock["source_files"]
        self.assertEqual(sources["config"]["path"], "examples/agemem_hotpotqa/agemem_e1_4b_format.yaml")
        self.assertEqual(
            sources["e0_config"]["path"],
            "examples/agemem_hotpotqa/agemem_e0_4b_format_eval.yaml",
        )
        self.assertEqual(
            sources["checkpoint_eval_config"]["path"],
            "examples/agemem_hotpotqa/agemem_e1_4b_format_eval.yaml",
        )
        self.assertEqual(_source_digest(E1_YAML), sources["config"]["sha256"])
        self.assertEqual(_source_digest(E0_YAML), sources["e0_config"]["sha256"])
        self.assertEqual(_source_digest(EVAL_YAML), sources["checkpoint_eval_config"]["sha256"])
        for path in (E0_YAML, E1_YAML, EVAL_YAML):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(yaml_requires_nudge(text), msg=path.name)
            self.assertIn("reward_profile: terminal_only", text)
            self.assertIn("enable_thinking: false", text)
            self.assertIn("gpu_memory_utilization: 0.6", text)
            self.assertIn("ppo_max_token_len_per_gpu: 2304", text)
            self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", text)
            self.assertNotIn(VANILLA_E0_JOB, text)
            self.assertNotIn(VANILLA_E1_JOB, text)
            for row_id in M5_TRAIN_IDS:
                self.assertIn(row_id, text)
        eval_text = EVAL_YAML.read_text(encoding="utf-8")
        for row_id in M5_EVAL_IDS:
            self.assertIn(row_id, eval_text)
        self.assertIn(
            "Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-4b-format/dummy_lora",
            eval_text,
        )
        e1_text = E1_YAML.read_text(encoding="utf-8")
        self.assertIn(f'name: "{FORMAT_E1_JOB}"', e1_text)
        self.assertIn("temperature: 0.6", e1_text)
        self.assertIn("total_steps: 1", e1_text)
        self.assertIn("max_model_len: 5120", e1_text)
        self.assertIn("max_response_tokens: 1024", e1_text)
        e0_text = E0_YAML.read_text(encoding="utf-8")
        self.assertIn(f'name: "{FORMAT_E0_JOB}"', e0_text)
        self.assertIn("mode: bench", e0_text)
        self.assertIn("temperature: 0.0", e0_text)

    def test_vanilla_4b_and_1p5b_locks_stay_untouched(self):
        vanilla = json.loads(VANILLA_LOCK_PATH.read_text(encoding="utf-8"))
        smoke = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            _source_digest(VANILLA_E1_YAML),
            vanilla["source_files"]["config"]["sha256"],
        )
        self.assertEqual(
            _source_digest(VANILLA_E0_YAML),
            vanilla["source_files"]["e0_config"]["sha256"],
        )
        self.assertEqual(
            _source_digest(VANILLA_EVAL_YAML),
            vanilla["source_files"]["checkpoint_eval_config"]["sha256"],
        )
        self.assertEqual(_source_digest(DRY_RUN), smoke["source_files"]["config"]["sha256"])
        for path in (VANILLA_E0_YAML, VANILLA_E1_YAML, VANILLA_EVAL_YAML):
            self.assertTrue(yaml_forbids_nudge(path.read_text(encoding="utf-8")), msg=path.name)
        dry_run = DRY_RUN.read_text(encoding="utf-8")
        self.assertNotIn("stage3_require_final_answer: true", dry_run)
        self.assertNotIn("stage3_repair_untagged_answer: true", dry_run)
        self.assertFalse(vanilla.get("stage3_require_final_answer"))
        self.assertEqual(vanilla["experiment_id"], "e1_terminal_only_4b_single_update")

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        vanilla_launcher = VANILLA_LAUNCHER.read_text(encoding="utf-8")
        probe_launcher = PROBE_LAUNCHER.read_text(encoding="utf-8")
        helper = HELPER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_format_contract_test", gate)
        self.assertNotIn("e1_4b_contract_test", gate)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        self.assertNotIn("agemem_e1_dry_run.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_dry_run.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_stage3_answer_probe.yaml", launcher)
        self.assertIn("agemem_e1_4b_format.yaml", launcher)
        self.assertIn("configs/e1_4b_format.json", launcher)
        self.assertIn("stage3_require_final_answer", launcher)
        self.assertIn("Qwen3-4B", helper)
        self.assertIn(FORMAT_E0_JOB, launcher)
        self.assertIn(FORMAT_E1_JOB, launcher)
        self.assertIn(FORMAT_E0_JOB, vanilla_launcher)
        self.assertIn(FORMAT_E1_JOB, vanilla_launcher)
        self.assertNotIn("agemem_e1_4b_format.yaml", vanilla_launcher)
        self.assertIn(FORMAT_E0_JOB, probe_launcher)
        self.assertIn(FORMAT_E1_JOB, probe_launcher)
        self.assertNotIn("agemem_e1_4b_format.yaml", probe_launcher)
        for job in FORBIDDEN_FOREIGN_JOBS:
            self.assertIn(job, launcher)


if __name__ == "__main__":
    unittest.main()
