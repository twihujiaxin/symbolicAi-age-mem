"""Contract tests for E1 terminal-only multi-seed repeats.

These tests are not part of the frozen M8b 318-count runtime gate. The repeat
launcher runs them before starting Ray.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPOSITORY_ROOT / "configs" / "e1_repeat.json"
SMOKE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "m8b_autodl_preflight.json"
REPEAT_TRAIN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_repeat.yaml"
REPEAT_EVAL = (
    REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_repeat_eval.yaml"
)
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
REPEAT_SCRIPT = REPOSITORY_ROOT / "scripts" / "agemem_e1_repeat.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
_OC_ENV = re.compile(r"\$\{oc\.env:([A-Z0-9_]+)(?:,([^}]*))?\}")


def _lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _materialize(path: Path, env: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        if key in env and env[key] != "":
            return env[key]
        if default is not None:
            return default
        raise KeyError(key)

    return _OC_ENV.sub(replace, path.read_text(encoding="utf-8"))


class E1RepeatContractTest(unittest.TestCase):
    def test_lock_lists_three_seeds_and_the_m5_six_train_ids(self):
        lock = _lock()
        self.assertEqual(lock["schema_version"], "agemem.e1_repeat.lock.v1")
        self.assertEqual(lock["seeds"], [7, 17, 27])
        self.assertNotIn(lock["smoke_seed"], lock["seeds"])
        self.assertEqual(lock["reward_profile"], "terminal_only")
        self.assertEqual(lock["repeat_times"], 2)
        self.assertEqual(lock["trainer_total_steps"], 1)
        self.assertEqual(lock["smoke_job_name"], "agemem-e1-terminal-only-dry-run")
        self.assertEqual(
            lock["job_name_template"], "agemem-e1-terminal-only-repeat-s{seed}"
        )
        self.assertEqual(len(lock["source_train_row_ids"]), 6)
        self.assertEqual(len(lock["held_out_row_ids"]), 2)
        for seed in lock["seeds"]:
            self.assertNotEqual(
                lock["job_name_template"].format(seed=seed),
                lock["smoke_job_name"],
            )

    def test_repeat_yaml_keeps_terminal_only_and_does_not_reuse_smoke_job(self):
        lock = _lock()
        seed = lock["seeds"][0]
        job = lock["job_name_template"].format(seed=seed)
        env = {
            "AGEMEM_E1_SEED": str(seed),
            "AGEMEM_E1_JOB_NAME": job,
            "TRINITY_MODEL_PATH": "/data/hjx/Age_mem/models/Qwen2.5-1.5B-Instruct",
            "HOTPOTQA_PATH": "/data/hjx/Age_mem/data/hotpot_qa/fullwiki",
            "TRINITY_CHECKPOINT_ROOT_DIR": "/data/hjx/Age_mem/checkpoints-e1-repeat",
        }
        train = _materialize(REPEAT_TRAIN, env)
        eval_text = _materialize(REPEAT_EVAL, env)

        self.assertIn(f"name: {job}", train)
        self.assertNotIn(lock["smoke_job_name"], train)
        self.assertIn(f"seed: {seed}", train)
        self.assertIn("total_steps: 1", train)
        self.assertIn("reward_profile: terminal_only", train)
        self.assertIn("milestone_reward_enabled: false", train)
        for row_id in lock["source_train_row_ids"]:
            self.assertIn(row_id, train)
            self.assertIn(row_id, eval_text)
        self.assertIn("mode: bench", eval_text)
        self.assertIn("continue_from_checkpoint: true", eval_text)
        self.assertIn(f"name: {job}", eval_text)
        for row_id in lock["held_out_row_ids"]:
            self.assertIn(row_id, eval_text)

    def test_smoke_dry_run_digest_is_unchanged(self):
        smoke_lock = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        expected = smoke_lock["source_files"]["config"]["sha256"]
        self.assertEqual(_source_digest(DRY_RUN), expected)
        self.assertNotEqual(DRY_RUN.read_bytes(), REPEAT_TRAIN.read_bytes())

    def test_repeat_script_stays_outside_the_frozen_m8b_gate(self):
        script = REPEAT_SCRIPT.read_text(encoding="utf-8")
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        self.assertNotIn("autodl_m8b_smoke.sh", script)
        self.assertNotIn("autodl_m8b_preflight.sh", script)
        self.assertNotIn("agemem_e1_dry_run.yaml", script)
        self.assertIn("agemem_e1_repeat.yaml", script)
        self.assertIn("agemem_e1_repeat_eval.yaml", script)
        self.assertIn("agemem-e1-terminal-only-dry-run", script)
        self.assertIn("flash_attn", script)
        self.assertIn("2.8.1", script)
        self.assertNotIn("e1_repeat_contract_test", gate)


if __name__ == "__main__":
    unittest.main()
