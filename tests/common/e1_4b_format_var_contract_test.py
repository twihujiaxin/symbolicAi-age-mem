"""Contract tests for the format-variance 4B GRPO protocol.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_format import yaml_requires_nudge as format_yaml_requires_nudge
from trinity.common.e1_4b_format_var import (
    E0_YAML,
    E1_YAML,
    EVAL_YAML,
    FORBIDDEN_FOREIGN_JOBS,
    FORMAT_E0_JOB,
    FORMAT_E1_JOB,
    FORMAT_LOCK_PATH,
    LOCK_PATH,
    VANILLA_E0_JOB,
    VANILLA_E1_JOB,
    VAR_E0_JOB,
    VAR_E1_JOB,
    job_names,
    load_lock,
    yaml_requires_nudge,
)
from trinity.common.m8b_preflight import _source_digest
from trinity.common.runtime_receipt import experience_reward_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORMAT_E0_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e0_4b_format_eval.yaml"
)
FORMAT_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format.yaml"
)
FORMAT_EVAL_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_eval.yaml"
)
VANILLA_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
)
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"

M5_TRAIN_IDS = [
    "5a85aaee5542991dd0999e84",
    "5a74b19355429916b01641dd",
    "5abecbed5542997719eab5c5",
    "5a8ac7d055429950cd6afb8f",
    "5a83df2655429933447460a1",
    "5a76f83e55429972597f1405",
]


class _Batch:
    def __init__(self, rewards, eids):
        self.rewards = rewards
        self.eids = eids


def _eid(task, run, step):
    return SimpleNamespace(batch=0, task=task, run=run, step=step)


class E14BFormatVarContractTest(unittest.TestCase):
    def test_lock_uses_k4_and_three_trainer_steps(self):
        lock = load_lock()
        format_lock = json.loads(FORMAT_LOCK_PATH.read_text(encoding="utf-8"))
        names = job_names(lock)
        self.assertEqual(lock["schema_version"], "agemem.m8b_preflight_lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_group_variance")
        self.assertNotEqual(lock["experiment_id"], format_lock["experiment_id"])
        self.assertEqual(names["e0"], VAR_E0_JOB)
        self.assertEqual(names["e1"], VAR_E1_JOB)
        self.assertNotEqual(names["e1"], FORMAT_E1_JOB)
        self.assertTrue(lock["stage3_require_final_answer"])
        assertions = {
            entry["path"]: entry["equals"] for entry in lock["config_assertions"]
        }
        self.assertEqual(assertions["name"], VAR_E1_JOB)
        self.assertEqual(assertions["algorithm.repeat_times"], 4)
        self.assertEqual(assertions["buffer.train_batch_size"], 8)
        self.assertEqual(assertions["explorer.max_repeat_times_per_runner"], 4)
        self.assertEqual(assertions["trainer.total_steps"], 3)
        self.assertEqual(assertions["buffer.total_steps"], 3)
        self.assertIs(
            assertions["buffer.explorer_input.taskset.workflow_args.stage3_require_final_answer"],
            True,
        )

    def test_yaml_matches_lock_and_keeps_nudge(self):
        lock = load_lock()
        sources = lock["source_files"]
        self.assertEqual(
            sources["config"]["path"],
            "examples/agemem_hotpotqa/agemem_e1_4b_format_var.yaml",
        )
        self.assertEqual(_source_digest(E1_YAML), sources["config"]["sha256"])
        self.assertEqual(_source_digest(E0_YAML), sources["e0_config"]["sha256"])
        self.assertEqual(_source_digest(EVAL_YAML), sources["checkpoint_eval_config"]["sha256"])
        for path in (E0_YAML, E1_YAML, EVAL_YAML):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(yaml_requires_nudge(text), msg=path.name)
            self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", text)
            self.assertIn("reward_profile: terminal_only", text)
            for row_id in M5_TRAIN_IDS:
                self.assertIn(row_id, text)
        e1_text = E1_YAML.read_text(encoding="utf-8")
        self.assertIn(f'name: "{VAR_E1_JOB}"', e1_text)
        self.assertIn("repeat_times: 4", e1_text)
        self.assertIn("train_batch_size: 8", e1_text)
        self.assertIn("max_repeat_times_per_runner: 4", e1_text)
        self.assertIn("total_steps: 3", e1_text)
        self.assertIn("temperature: 0.6", e1_text)
        eval_text = EVAL_YAML.read_text(encoding="utf-8")
        self.assertIn(
            "Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-4b-format-var/dummy_lora",
            eval_text,
        )
        self.assertIn("temperature: 0.0", eval_text)

    def test_k2_format_and_vanilla_locks_stay_untouched(self):
        format_lock = json.loads(FORMAT_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            _source_digest(FORMAT_E1_YAML),
            format_lock["source_files"]["config"]["sha256"],
        )
        self.assertEqual(
            format_lock["experiment_id"],
            "e1_format_conditioned_4b_single_update",
        )
        self.assertTrue(
            format_yaml_requires_nudge(FORMAT_E0_YAML.read_text(encoding="utf-8"))
        )
        self.assertTrue(yaml_forbids_nudge(VANILLA_E1_YAML.read_text(encoding="utf-8")))
        self.assertEqual(_source_digest(FORMAT_EVAL_YAML), format_lock["source_files"]["checkpoint_eval_config"]["sha256"])

    def test_launcher_skips_e0_and_stays_out_of_318(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        format_launcher = FORMAT_LAUNCHER.read_text(encoding="utf-8")
        vanilla_launcher = VANILLA_LAUNCHER.read_text(encoding="utf-8")
        probe_launcher = PROBE_LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_format_var_contract_test", gate)
        self.assertIn("agemem_e1_4b_format_var.yaml", launcher)
        self.assertIn("configs/e1_4b_format_var.json", launcher)
        self.assertIn("trainer_step_3.json", launcher)
        self.assertIn("global_step_3", launcher)
        self.assertNotIn("agemem_e0_4b_format_var_eval.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_format.yaml", launcher)
        self.assertIn(VAR_E0_JOB, format_launcher)
        self.assertIn(VAR_E1_JOB, format_launcher)
        self.assertIn(VAR_E1_JOB, vanilla_launcher)
        self.assertIn(VAR_E1_JOB, probe_launcher)
        for job in FORBIDDEN_FOREIGN_JOBS:
            self.assertIn(job, launcher)

    def test_experience_reward_metrics_report_zero_group_std(self):
        tied = _Batch(
            [0.4, 0.4, 0.4, 0.4],
            [_eid(0, 0, 0), _eid(0, 0, 1), _eid(0, 1, 0), _eid(0, 1, 1)],
        )
        tied_metrics = experience_reward_metrics(tied)
        self.assertEqual(tied_metrics["training/reward_mean"], 0.4)
        self.assertEqual(tied_metrics["training/last_step_unique_count"], 1.0)
        self.assertEqual(tied_metrics["training/group_reward_std_mean"], 0.0)

        mixed = _Batch(
            [0.0, 1.0, 0.4, 0.4],
            [_eid(0, 0, 1), _eid(0, 1, 1), _eid(1, 0, 1), _eid(1, 1, 1)],
        )
        mixed_metrics = experience_reward_metrics(mixed)
        self.assertEqual(mixed_metrics["training/last_step_unique_count"], 3.0)
        self.assertGreater(mixed_metrics["training/group_reward_std_mean"], 0.0)
        self.assertEqual(mixed_metrics["training/group_reward_std_min"], 0.0)
        self.assertGreater(mixed_metrics["training/group_reward_std_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
