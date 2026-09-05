"""Contract tests for the format-group 4B GRPO protocol.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_format import yaml_requires_nudge as format_yaml_requires_nudge
from trinity.common.e1_4b_format_group import (
    E0_YAML,
    E1_YAML,
    EVAL_YAML,
    FORBIDDEN_FOREIGN_JOBS,
    FORMAT_E1_JOB,
    FORMAT_LOCK_PATH,
    GROUP_E0_JOB,
    GROUP_E1_JOB,
    VAR_E1_JOB,
    VAR_LOCK_PATH,
    job_names,
    load_lock,
    yaml_requires_nudge,
)
from trinity.common.e1_4b_format_var import yaml_requires_nudge as var_yaml_requires_nudge
from trinity.common.m8b_preflight import _source_digest
from trinity.common.runtime_receipt import experience_reward_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VAR_E0_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e0_4b_format_var_eval.yaml"
)
VAR_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_var.yaml"
)
VAR_EVAL_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_var_eval.yaml"
)
FORMAT_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format.yaml"
)
VANILLA_E1_YAML = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
)
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
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


class E14BFormatGroupContractTest(unittest.TestCase):
    def test_lock_uses_complete_put_batch_and_k4(self):
        lock = load_lock()
        var_lock = json.loads(VAR_LOCK_PATH.read_text(encoding="utf-8"))
        format_lock = json.loads(FORMAT_LOCK_PATH.read_text(encoding="utf-8"))
        names = job_names(lock)
        self.assertEqual(lock["schema_version"], "agemem.m8b_preflight_lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_complete_group")
        self.assertNotEqual(lock["experiment_id"], var_lock["experiment_id"])
        self.assertNotEqual(lock["experiment_id"], format_lock["experiment_id"])
        self.assertEqual(names["e0"], GROUP_E0_JOB)
        self.assertEqual(names["e1"], GROUP_E1_JOB)
        self.assertNotEqual(names["e1"], VAR_E1_JOB)
        self.assertNotEqual(names["e1"], FORMAT_E1_JOB)
        self.assertTrue(lock["stage3_require_final_answer"])
        assertions = {
            entry["path"]: entry["equals"] for entry in lock["config_assertions"]
        }
        self.assertEqual(assertions["name"], GROUP_E1_JOB)
        self.assertEqual(assertions["algorithm.repeat_times"], 4)
        self.assertEqual(assertions["buffer.train_batch_size"], 8)
        self.assertIs(
            assertions["buffer.trainer_input.experience_buffer.consume_put_batch"],
            True,
        )
        self.assertEqual(assertions["explorer.max_repeat_times_per_runner"], 4)
        self.assertEqual(assertions["trainer.total_steps"], 3)
        self.assertEqual(assertions["buffer.total_steps"], 3)
        self.assertIs(
            assertions["buffer.explorer_input.taskset.workflow_args.stage3_require_final_answer"],
            True,
        )
        config_py = (REPOSITORY_ROOT / "trinity" / "common" / "config.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("consume_put_batch: bool = False", config_py)

    def test_yaml_matches_lock_and_keeps_nudge(self):
        lock = load_lock()
        sources = lock["source_files"]
        self.assertEqual(
            sources["config"]["path"],
            "examples/agemem_hotpotqa/agemem_e1_4b_format_group.yaml",
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
        self.assertIn(f'name: "{GROUP_E1_JOB}"', e1_text)
        self.assertIn("repeat_times: 4", e1_text)
        self.assertIn("train_batch_size: 8", e1_text)
        self.assertIn("consume_put_batch: true", e1_text)
        self.assertIn("max_repeat_times_per_runner: 4", e1_text)
        self.assertIn("total_steps: 3", e1_text)
        self.assertIn("temperature: 0.6", e1_text)
        eval_text = EVAL_YAML.read_text(encoding="utf-8")
        self.assertIn(
            "Trinity-RFT-AgeMem-M8/agemem-e1-terminal-only-4b-format-group/dummy_lora",
            eval_text,
        )
        self.assertIn("temperature: 0.0", eval_text)

    def test_format_var_and_vanilla_locks_stay_untouched(self):
        var_lock = json.loads(VAR_LOCK_PATH.read_text(encoding="utf-8"))
        format_lock = json.loads(FORMAT_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            _source_digest(VAR_E1_YAML),
            var_lock["source_files"]["config"]["sha256"],
        )
        self.assertEqual(
            _source_digest(VAR_E0_YAML),
            var_lock["source_files"]["e0_config"]["sha256"],
        )
        self.assertEqual(
            _source_digest(VAR_EVAL_YAML),
            var_lock["source_files"]["checkpoint_eval_config"]["sha256"],
        )
        self.assertEqual(
            var_lock["experiment_id"],
            "e1_format_conditioned_4b_group_variance",
        )
        self.assertTrue(var_yaml_requires_nudge(VAR_E1_YAML.read_text(encoding="utf-8")))
        self.assertEqual(
            _source_digest(FORMAT_E1_YAML),
            format_lock["source_files"]["config"]["sha256"],
        )
        self.assertTrue(
            format_yaml_requires_nudge(FORMAT_E1_YAML.read_text(encoding="utf-8"))
        )
        self.assertTrue(yaml_forbids_nudge(VANILLA_E1_YAML.read_text(encoding="utf-8")))
        self.assertNotIn("consume_put_batch", VAR_E1_YAML.read_text(encoding="utf-8"))

    def test_launcher_skips_e0_and_stays_out_of_318(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        var_launcher = VAR_LAUNCHER.read_text(encoding="utf-8")
        format_launcher = FORMAT_LAUNCHER.read_text(encoding="utf-8")
        vanilla_launcher = VANILLA_LAUNCHER.read_text(encoding="utf-8")
        probe_launcher = PROBE_LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_format_group_contract_test", gate)
        self.assertIn("agemem_e1_4b_format_group.yaml", launcher)
        self.assertIn("configs/e1_4b_format_group.json", launcher)
        self.assertIn("trainer_step_3.json", launcher)
        self.assertIn("bench_step_3_model_3.json", launcher)
        self.assertNotIn("bench_step_1_model_1.json", launcher)
        self.assertIn("global_step_3", launcher)
        self.assertNotIn("agemem_e0_4b_format_group_eval.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_format_var.yaml", launcher)
        self.assertIn("consume_put_batch", launcher)
        self.assertIn(GROUP_E0_JOB, var_launcher)
        self.assertIn(GROUP_E1_JOB, var_launcher)
        self.assertIn(GROUP_E1_JOB, format_launcher)
        self.assertIn(GROUP_E1_JOB, vanilla_launcher)
        self.assertIn(GROUP_E1_JOB, probe_launcher)
        for job in FORBIDDEN_FOREIGN_JOBS:
            self.assertIn(job, launcher)

    def test_complete_group_receipt_metrics_count_all_runs(self):
        rewards = []
        eids = []
        for task in (0, 1):
            for run in range(4):
                for step in range(3):
                    rewards.append(0.4 if task == 0 else 0.0)
                    eids.append(_eid(task, run, step))
        metrics = experience_reward_metrics(_Batch(rewards, eids))
        self.assertEqual(metrics["training/experience_count"], 24.0)
        self.assertEqual(metrics["training/last_step_run_count"], 8.0)
        self.assertEqual(metrics["training/last_step_unique_count"], 2.0)
        self.assertEqual(metrics["training/group_reward_std_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
