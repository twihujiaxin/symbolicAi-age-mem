"""Contract tests for the format-conditioned 4B 36-step GRPO pilot.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_fc_pilot import (
    ALL_JOBS,
    CHECKPOINT_ROOT,
    E0_JOB,
    EVAL_JOB_BY_STEP,
    EXPECTED_REVISION,
    FORBIDDEN_FOREIGN_JOBS,
    LOCK_PATH,
    SAVE_INTERVAL,
    TRAIN_JOB,
    TRAIN_YAML,
    TRAINER_TOTAL_STEPS,
    load_lock,
    render_checkpoint_eval_yaml,
    render_e0_yaml,
    render_train_yaml,
)
from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS as DIAGNOSIS_JOBS,
)
from trinity.common.e1_4b_format_conditioned import (
    DIAGNOSIS_FLAG_STRINGS,
    FROZEN_CLEAN_YAMLS,
    SCALE_LOCK_PATH,
    load_lock as load_fc_lock,
    load_scale_lock,
    selection_is_frozen,
    train_rows_match_scale,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_pilot.sh"
DIAG_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
GROUP_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
QR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_question_retrieve.sh"
DRY_RUN_4B = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
GROUP_YAML = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_format_group.yaml"


class E14BFcPilotContractTest(unittest.TestCase):
    def test_lock_and_train_yaml_match_format_conditioned_protocol(self):
        lock = load_lock()
        fc_lock = load_fc_lock()
        scale = load_scale_lock()
        self.assertEqual(lock["schema_version"], "agemem.e1_4b_fc_pilot.lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_36step_pilot")
        self.assertEqual(lock["checkpoint_root"], CHECKPOINT_ROOT)
        self.assertNotEqual(lock["checkpoint_root"], fc_lock["checkpoint_root"])
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertEqual(lock["reward_profile"], "terminal_only")
        self.assertEqual(lock["terminal_reward_metric"], "hotpotqa_official")
        self.assertEqual(lock["seed"], 7)
        self.assertEqual(lock["later_seeds"], [17, 27])
        self.assertEqual(lock["trainer_total_steps"], TRAINER_TOTAL_STEPS)
        self.assertEqual(lock["save_interval"], SAVE_INTERVAL)
        self.assertEqual(lock["eval_steps"], [0, 12, 24, 36])
        self.assertEqual(lock["repeat_times"], 4)
        self.assertTrue(lock["consume_put_batch"])
        self.assertEqual(lock["model"]["expected_revision"], EXPECTED_REVISION)
        self.assertEqual(lock["jobs"]["train"], TRAIN_JOB)
        self.assertEqual(lock["jobs"]["e0"], E0_JOB)
        self.assertEqual(lock["jobs"]["eval_s36"], EVAL_JOB_BY_STEP[36])
        self.assertTrue(train_rows_match_scale(fc_lock, scale))
        self.assertEqual(_source_digest(TRAIN_YAML), lock["source_files"]["train_config"]["sha256"])
        train = TRAIN_YAML.read_text(encoding="utf-8")
        self.assertEqual(train, render_train_yaml(fc_lock))
        self.assertIn(f'name: "{TRAIN_JOB}"', train)
        self.assertIn("mode: both", train)
        self.assertIn("repeat_times: 4", train)
        self.assertIn("total_steps: 36", train)
        self.assertIn("save_interval: 12", train)
        self.assertIn("consume_put_batch: true", train)
        self.assertIn("train_batch_size: 8", train)
        self.assertIn("gpu_memory_utilization: 0.5", train)
        self.assertIn("enable_prefix_caching: false", train)
        self.assertNotIn("gpu_memory_utilization: 0.6", train)
        self.assertIn("eval_tasksets: []", train)
        self.assertIn("stage3_require_final_answer: true", train)
        self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", train)
        self.assertNotIn("stage3_disable_ltm_retrieve", train)
        self.assertNotIn("stage3_inject_gold_supporting", train)
        for row in fc_lock["fixed_train_rows"]:
            self.assertIn(row["hotpot_id"], train)
        self.assertEqual(
            json.loads(SCALE_LOCK_PATH.read_text(encoding="utf-8"))["fixed_train_rows"],
            fc_lock["fixed_train_rows"],
        )

    def test_eval_renderers_require_frozen_dev_and_skip_when_pending(self):
        fc_lock = load_fc_lock()
        if selection_is_frozen(fc_lock):
            e0 = render_e0_yaml(fc_lock)
            self.assertIn(f'name: "{E0_JOB}"', e0)
            self.assertIn("repeat_times: 1", e0)
            self.assertIn("temperature: 0.0", e0)
            self.assertNotIn("consume_put_batch", e0)
            for row in fc_lock["fixed_dev_rows"]:
                self.assertIn(row["hotpot_id"], e0)
            eval36 = render_checkpoint_eval_yaml(fc_lock, 36)
            self.assertIn(f'name: "{EVAL_JOB_BY_STEP[36]}"', eval36)
            self.assertIn(f"{TRAIN_JOB}/global_step_36/actor/lora_adapter", eval36)
            self.assertNotIn("consume_put_batch", eval36)
        else:
            with self.assertRaises(ValueError):
                render_e0_yaml(fc_lock)
            with self.assertRaises(ValueError):
                render_checkpoint_eval_yaml(fc_lock, 12)

    def test_diagnosis_flags_stay_out_of_pilot_and_frozen_yamls(self):
        train = TRAIN_YAML.read_text(encoding="utf-8")
        for flag in DIAGNOSIS_FLAG_STRINGS:
            self.assertNotIn(flag, train)
        self.assertTrue(yaml_forbids_nudge(DRY_RUN_4B.read_text(encoding="utf-8")))
        group = GROUP_YAML.read_text(encoding="utf-8")
        self.assertNotIn(TRAIN_JOB, group)
        self.assertNotIn("total_steps: 36", group)
        for path in FROZEN_CLEAN_YAMLS:
            if path.resolve() == TRAIN_YAML.resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for job in ALL_JOBS:
                self.assertNotIn(job, text, msg=path.name)

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_fc_pilot_contract_test", gate)
        self.assertNotIn("agemem_e1_4b_fc_pilot", gate)
        self.assertIn("configs/e1_4b_fc_pilot.json", launcher)
        self.assertIn("checkpoints-e1-4b-fc-pilot", launcher)
        self.assertIn("consume_put_batch", launcher)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", launcher)
        self.assertIn("expandable_segments:True", launcher)
        self.assertIn("trainer_step_36.json", launcher)
        self.assertIn("global_step_12", launcher)
        self.assertIn("global_step_24", launcher)
        self.assertIn("global_step_36", launcher)
        self.assertIn("flash_attn", launcher)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        self.assertNotIn("agemem_e1_4b_format_group.yaml", launcher)
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
            QR_LAUNCHER,
        ):
            text = other.read_text(encoding="utf-8")
            for job in ALL_JOBS:
                self.assertIn(job, text, msg=other.name)
        self.assertTrue(set(DIAGNOSIS_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))


if __name__ == "__main__":
    unittest.main()
