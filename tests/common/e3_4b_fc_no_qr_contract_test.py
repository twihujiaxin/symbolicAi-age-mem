"""Contract tests for format-conditioned 4B E3 without question-retrieve.

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
    FROZEN_CLEAN_YAMLS,
    SCALE_LOCK_PATH,
    load_lock as load_fc_lock,
    selection_is_frozen,
    train_rows_match_scale,
)
from trinity.common.e3_4b_fc import ALL_JOBS as QR_E3_JOBS
from trinity.common.e3_4b_fc_no_qr import (
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
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e3_4b_fc_no_qr.sh"
OFFLINE_LAUNCHER = (
    REPOSITORY_ROOT / "scripts" / "agemem_e3_oracle_offline_compare.sh"
)
OFFLINE_REPORT = (
    REPOSITORY_ROOT / "scripts" / "agemem_e3_oracle_offline_compare.py"
)
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
GROUP_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
DIAG_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag.sh"
PILOT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_pilot.sh"
QR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_fc_question_retrieve.sh"
E3_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e3_4b_fc.sh"
DRY_RUN_4B = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"


class E34BFcNoQrContractTest(unittest.TestCase):
    def test_lock_and_train_yaml_disable_question_retrieve(self):
        lock = load_lock()
        fc_lock = load_fc_lock()
        scale = json.loads(SCALE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(lock["schema_version"], "agemem.e3_4b_fc_no_qr.lock.v1")
        self.assertEqual(
            lock["experiment_id"], "e3_format_conditioned_4b_oracle_dfa_no_qr"
        )
        self.assertEqual(lock["checkpoint_root"], CHECKPOINT_ROOT)
        self.assertNotEqual(lock["checkpoint_root"], fc_lock["checkpoint_root"])
        self.assertTrue(lock["checkpoint_root"].endswith("checkpoints-e3-4b-fc-no-qr"))
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertFalse(lock["stage3_question_retrieve"])
        self.assertFalse(lock["stage3_index_observed_context"])
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
        self.assertIn("stage3_question_retrieve: false", train)
        self.assertIn("stage3_index_observed_context: false", train)
        self.assertNotIn("stage3_question_retrieve: true", train)
        self.assertNotIn("stage3_index_observed_context: true", train)
        self.assertIn("consume_put_batch: true", train)
        self.assertIn("gpu_memory_utilization: 0.5", train)
        self.assertIn("enable_prefix_caching: false", train)
        self.assertIn("repeat_times: 4", train)
        self.assertIn("total_steps: 12", train)
        self.assertNotIn("stage3_inject_gold_supporting: true", train)
        self.assertNotIn("agemem_heuristic", train)
        for row in fc_lock["fixed_train_rows"]:
            self.assertIn(row["hotpot_id"], train)

    def test_eval_renderers_require_frozen_dev_and_disable_qr(self):
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
        self.assertIn("stage3_question_retrieve: false", e0)
        self.assertIn("stage3_index_observed_context: false", e0)
        self.assertNotIn("stage3_question_retrieve: true", e0)
        self.assertNotIn("consume_put_batch", e0)
        for row in fc_lock["fixed_dev_rows"]:
            self.assertIn(row["hotpot_id"], e0)
        eval12 = render_checkpoint_eval_yaml(fc_lock, 12)
        self.assertIn(f'name: "{EVAL_JOB_BY_STEP[12]}"', eval12)
        self.assertIn("e3_dfa_shadow: true", eval12)
        self.assertIn("stage3_question_retrieve: false", eval12)
        self.assertIn("global_step_12/actor/lora_adapter", eval12)
        self.assertIn(TRAIN_JOB, eval12)

    def test_frozen_dry_run_stays_terminal_only_and_excludes_no_qr_jobs(self):
        self.assertTrue(yaml_forbids_nudge(DRY_RUN_4B.read_text(encoding="utf-8")))
        dry = DRY_RUN_4B.read_text(encoding="utf-8")
        self.assertNotIn("terminal_dfa", dry)
        self.assertNotIn("e3_dfa_shadow", dry)
        self.assertNotIn(TRAIN_JOB, dry)
        for path in FROZEN_CLEAN_YAMLS:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("reward_profile: terminal_dfa", text, msg=path.name)
            self.assertNotIn(TRAIN_JOB, text, msg=path.name)

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("e3_4b_fc_no_qr_contract_test", gate)
        self.assertNotIn("agemem_e3_4b_fc_no_qr", gate)
        self.assertIn("configs/e3_4b_fc_no_qr.json", launcher)
        self.assertIn("checkpoints-e3-4b-fc-no-qr", launcher)
        self.assertIn("terminal_dfa", launcher)
        self.assertIn("flash_attn", launcher)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        for job in ALL_JOBS:
            self.assertIn(job, launcher)
        for job in QR_E3_JOBS:
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
            E3_LAUNCHER,
        ):
            text = other.read_text(encoding="utf-8")
            self.assertIn(TRAIN_JOB, text, msg=other.name)
        self.assertTrue(set(DIAGNOSIS_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))
        self.assertTrue(set(PILOT_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))
        self.assertTrue(set(QR_E3_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))
        self.assertFalse(set(ALL_JOBS).issubset(set(FORBIDDEN_FOREIGN_JOBS)))

    def test_cpu_offline_comparison_uses_frozen_real_action_artifacts(self):
        launcher = OFFLINE_LAUNCHER.read_text(encoding="utf-8")
        report = OFFLINE_REPORT.read_text(encoding="utf-8")
        self.assertIn("AGEMEM_DIAGNOSIS_ROOT", launcher)
        self.assertIn("agemem-e1-4b-fc-signal-diag", launcher)
        self.assertIn("buffer/explorer_output.jsonl", launcher)
        self.assertIn("trajectories/tool_calls.jsonl", launcher)
        self.assertIn("trajectories/stage3_final_turn.jsonl", launcher)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", launcher)
        self.assertNotIn("ray start", launcher)
        self.assertNotIn("trinity run", launcher)
        self.assertIn("replay_hotpotqa_oracle_comparison", report)
        self.assertIn("_validate_trace_join", report)
        self.assertIn("fixed_train_rows", report)
        self.assertIn("ActionEvent", report)
        self.assertIn("flat_oracle_credits.jsonl", report)
        self.assertIn("oracle_dfa_credits.jsonl", report)
        self.assertIn("semantic_audit.csv", report)
        self.assertIn("positive_controls.jsonl", report)
        self.assertIn("contains_privileged_gold", report)
        self.assertIn("blocked_no_natural_reward_signal", report)


if __name__ == "__main__":
    unittest.main()
