"""Contract tests for the 1.5B terminal-only E1 scale protocol.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_scale import (
    load_lock,
    render_eval_yaml,
    render_train_yaml,
    selection_is_complete,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_scale.json"
SMOKE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "m8b_autodl_preflight.json"
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
SELECT_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_scale_select.py"
SCALE_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_scale.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"


def _complete_lock() -> dict:
    lock = load_lock()
    extra = []
    for index in range(int(lock["extra_train_size"])):
        extra.append(
            {
                "content_sha256": f"{index:064x}",
                "hotpot_id": f"extra-{index:02d}",
                "source_index": 100000 + index,
            }
        )
    lock["fixed_train_rows"] = list(lock["prefix_train_rows"]) + extra
    lock["selection_status"] = "frozen"
    return lock


class E1ScaleContractTest(unittest.TestCase):
    def test_lock_keeps_vanilla_grpo_and_expands_beyond_smoke(self):
        lock = load_lock()
        self.assertEqual(lock["schema_version"], "agemem.e1_scale.lock.v1")
        self.assertEqual(lock["selection_status"], "pending")
        self.assertEqual(lock["job_name"], "agemem-e1-terminal-only-scale")
        self.assertNotEqual(lock["job_name"], lock["smoke_job_name"])
        self.assertEqual(lock["reward_profile"], "terminal_only")
        self.assertEqual(lock["trainer_total_steps"], 8)
        self.assertEqual(lock["train_size"], 24)
        self.assertEqual(lock["extra_train_size"], 18)
        self.assertEqual(lock["seed"], 7)
        self.assertNotEqual(lock["extra_selection_seed"], lock["smoke_seed"])
        self.assertFalse(lock["stage3_require_final_answer"])
        self.assertFalse(lock["stage3_repair_untagged_answer"])
        self.assertEqual(len(lock["source_train_prefix_ids"]), 6)
        self.assertEqual(lock["fixed_train_rows"], [])
        self.assertFalse(selection_is_complete(lock))

    def test_generated_yaml_has_no_answer_nudge(self):
        lock = _complete_lock()
        self.assertTrue(selection_is_complete(lock))
        train = render_train_yaml(lock)
        eval_text = render_eval_yaml(lock)
        for text in (train, eval_text):
            self.assertIn('name: "agemem-e1-terminal-only-scale"', text)
            self.assertNotIn("agemem-e1-terminal-only-dry-run", text)
            self.assertIn("reward_profile: terminal_only", text)
            self.assertIn("milestone_reward_enabled: false", text)
            self.assertNotIn("stage3_require_final_answer", text)
            self.assertNotIn("stage3_repair_untagged_answer", text)
            self.assertIn("stage3_max_rounds: 2", text)
            for row_id in lock["source_train_prefix_ids"]:
                self.assertIn(row_id, text)
            for row in lock["fixed_train_rows"][6:]:
                self.assertIn(row["hotpot_id"], text)
        self.assertIn("total_steps: 8", train)
        self.assertIn("total_steps: 96", train)
        self.assertIn("mode: bench", eval_text)
        self.assertIn("continue_from_checkpoint: true", eval_text)
        for row_id in lock["held_out_row_ids"]:
            self.assertIn(row_id, eval_text)

    def test_frozen_dry_run_digest_is_unchanged(self):
        smoke_lock = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(_source_digest(DRY_RUN), smoke_lock["source_files"]["config"]["sha256"])
        dry_run = DRY_RUN.read_text(encoding="utf-8")
        self.assertNotIn("stage3_require_final_answer: true", dry_run)
        self.assertNotIn("hotpotqa_e1_train_scale", dry_run)
        script = SCALE_SCRIPT.read_text(encoding="utf-8")
        select = SELECT_SCRIPT.read_text(encoding="utf-8")
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        self.assertNotIn("e1_scale_contract_test", gate)
        self.assertNotIn("autodl_m8b_smoke.sh", script)
        self.assertNotIn("agemem_e1_dry_run.yaml", script)
        self.assertIn("agemem_e1_scale.yaml", script)
        self.assertIn("agemem_e1_scale_eval.yaml", script)
        self.assertIn("stage3_require_final_answer", script)
        self.assertIn("HOTPOTQA_PATH", select)
        self.assertIn("write-yaml", select)


if __name__ == "__main__":
    unittest.main()
