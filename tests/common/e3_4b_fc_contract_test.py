"""Contract tests for format-conditioned 4B E3 Oracle DFA.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_fc_pilot import ALL_JOBS as PILOT_JOBS
from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS as DIAGNOSIS_JOBS,
)
from trinity.common.e1_4b_format_conditioned import (
    DIAGNOSIS_FLAG_STRINGS,
    FROZEN_CLEAN_YAMLS,
    SCALE_LOCK_PATH,
    load_lock as load_fc_lock,
    selection_is_frozen,
    train_rows_match_scale,
)
from trinity.common.e3_4b_fc import (
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
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e3_4b_fc.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
GROUP_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
DIAG_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag.sh"
PILOT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_pilot.sh"
QR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_question_retrieve.sh"
DRY_RUN_4B = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"
WORKFLOW = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "train_hotpotQA.py"
)


class E34BFcContractTest(unittest.TestCase):
    def test_lock_and_train_yaml_match_e3_protocol(self):
        lock = load_lock()
        fc_lock = load_fc_lock()
        scale = json.loads(SCALE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], "agemem.e3_4b_fc.lock.v1")
        self.assertEqual(lock["experiment_id"], "e3_format_conditioned_4b_oracle_dfa")
        self.assertEqual(lock["checkpoint_root"], CHECKPOINT_ROOT)
        self.assertNotEqual(lock["checkpoint_root"], fc_lock["checkpoint_root"])
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertTrue(lock["stage3_question_retrieve"])
        self.assertTrue(lock["stage3_index_observed_context"])
        self.assertFalse(lock["stage3_inject_gold_supporting"])
        self.assertEqual(lock["reward_profile"], "terminal_dfa")
        self.assertEqual(lock["terminal_reward_metric"], "hotpotqa_official")
        self.assertTrue(lock["e3_dfa_shadow_on_eval"])
        self.assertEqual(lock["seed"], 7)
        self.assertEqual(lock["trainer_total_steps"], TRAINER_TOTAL_STEPS)
        self.assertEqual(lock["save_interval"], SAVE_INTERVAL)
        self.assertEqual(lock["eval_steps"], [0, 12])
        self.assertEqual(lock["repeat_times"], 4)
        self.assertTrue(lock["consume_put_batch"])
        self.assertEqual(lock["model"]["expected_revision"], EXPECTED_REVISION)
        self.assertEqual(lock["jobs"]["train"], TRAIN_JOB)
        self.assertEqual(lock["jobs"]["e0"], E0_JOB)
        self.assertEqual(lock["jobs"]["eval_s12"], EVAL_JOB_BY_STEP[12])
        self.assertTrue(train_rows_match_scale(fc_lock, scale))
        self.assertEqual(
            _source_digest(TRAIN_YAML), lock["source_files"]["train_config"]["sha256"]
        )
        train = TRAIN_YAML.read_text(encoding="utf-8")
        self.assertEqual(train, render_train_yaml(fc_lock))
        self.assertIn(f'name: "{TRAIN_JOB}"', train)
        self.assertIn("reward_profile: terminal_dfa", train)
        self.assertIn("e3_dfa_shadow: false", train)
        self.assertIn("stage3_question_retrieve: true", train)
        self.assertIn("stage3_index_observed_context: true", train)
        self.assertIn("consume_put_batch: true", train)
        self.assertIn("gpu_memory_utilization: 0.5", train)
        self.assertIn("enable_prefix_caching: false", train)
        self.assertIn("repeat_times: 4", train)
        self.assertIn("total_steps: 12", train)
        self.assertNotIn("stage3_inject_gold_supporting: true", train)
        self.assertNotIn("agemem_heuristic", train)
        for row in fc_lock["fixed_train_rows"]:
            self.assertIn(row["hotpot_id"], train)

    def test_eval_renderers_require_frozen_dev_and_shadow_dfa(self):
        fc_lock = load_fc_lock()
        if not selection_is_frozen(fc_lock):
            with self.assertRaises(ValueError):
                render_e0_yaml(fc_lock)
            return
        e0 = render_e0_yaml(fc_lock)
        self.assertIn(f'name: "{E0_JOB}"', e0)
        self.assertIn("repeat_times: 1", e0)
        self.assertIn("temperature: 0.0", e0)
        self.assertIn("e3_dfa_shadow: true", e0)
        self.assertIn("reward_profile: terminal_dfa", e0)
        self.assertIn("stage3_question_retrieve: true", e0)
        self.assertNotIn("consume_put_batch", e0)
        for row in fc_lock["fixed_dev_rows"]:
            self.assertIn(row["hotpot_id"], e0)
        eval12 = render_checkpoint_eval_yaml(fc_lock, 12)
        self.assertIn(f'name: "{EVAL_JOB_BY_STEP[12]}"', eval12)
        self.assertIn("e3_dfa_shadow: true", eval12)
        self.assertIn("global_step_12/actor/lora_adapter", eval12)

    def test_frozen_dry_run_and_clean_yamls_stay_terminal_only(self):
        self.assertTrue(yaml_forbids_nudge(DRY_RUN_4B.read_text(encoding="utf-8")))
        dry = DRY_RUN_4B.read_text(encoding="utf-8")
        self.assertNotIn("terminal_dfa", dry)
        self.assertNotIn("e3_dfa_shadow", dry)
        self.assertNotIn("agemem-e3-4b-fc", dry)
        for path in FROZEN_CLEAN_YAMLS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("reward_profile: terminal_dfa", text, msg=path.name)
            self.assertNotIn("e3_dfa_shadow", text, msg=path.name)

    def test_workflow_keeps_e1_terminal_branch_and_adds_e3(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("if self.reward_profile.is_terminal_only", workflow)
        self.assertIn("elif self.reward_profile.is_oracle_dfa", workflow)
        self.assertIn("replay_hotpotqa_oracle_dfa", workflow)
        self.assertIn("e3_dfa_shadow", workflow)
        self.assertIn('workflow_args.get("stage3_question_retrieve", False)', workflow)

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e3_4b_fc_contract_test", gate)
        self.assertNotIn("e3_oracle_dfa_test", gate)
        self.assertNotIn("agemem_e3_4b_fc", gate)
        self.assertIn("configs/e3_4b_fc.json", launcher)
        self.assertIn("checkpoints-e3-4b-fc", launcher)
        self.assertIn("terminal_dfa", launcher)
        self.assertIn("flash_attn", launcher)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
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
            QR_LAUNCHER,
        ):
            text = other.read_text(encoding="utf-8")
            self.assertIn(TRAIN_JOB, text, msg=other.name)
        self.assertTrue(set(DIAGNOSIS_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))
        self.assertTrue(set(PILOT_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))


if __name__ == "__main__":
    unittest.main()
