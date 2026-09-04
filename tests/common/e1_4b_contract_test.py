"""Contract tests for the independent 4B terminal-only E1 protocol.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import (
    E0_YAML,
    E1_YAML,
    EVAL_YAML,
    FORBIDDEN_LEGACY_JOBS,
    LOCK_PATH,
    SMOKE_E0_JOB,
    SMOKE_E1_JOB,
    job_names,
    load_lock,
    yaml_forbids_nudge,
)
from trinity.common.m8b_preflight import _source_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_LOCK_PATH = REPOSITORY_ROOT / "configs" / "m8b_autodl_preflight.json"
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
HELPER = REPOSITORY_ROOT / "trinity" / "common" / "e1_4b.py"

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


class E14BContractTest(unittest.TestCase):
    def test_lock_is_independent_of_1p5b_smoke(self):
        lock = load_lock()
        smoke = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        names = job_names(lock)
        self.assertEqual(lock["schema_version"], "agemem.m8b_preflight_lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_terminal_only_4b_single_update")
        self.assertEqual(lock["model"]["repository_id"], "Qwen/Qwen3-4B")
        self.assertEqual(
            lock["model"]["expected_revision"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(lock["model"]["config_assertions"]["model_type"], "qwen3")
        self.assertEqual(
            lock["model"]["config_assertions"]["architectures.0"],
            "Qwen3ForCausalLM",
        )
        self.assertGreaterEqual(lock["model"]["minimum_weight_bytes"], 7000000000)
        self.assertNotEqual(lock["model"]["repository_id"], smoke["model"]["repository_id"])
        self.assertEqual(names["e0"], "agemem-e0-terminal-only-4b-frozen-eval")
        self.assertEqual(names["e1"], "agemem-e1-terminal-only-4b-dry-run")
        self.assertNotEqual(names["e0"], SMOKE_E0_JOB)
        self.assertNotEqual(names["e1"], SMOKE_E1_JOB)
        self.assertFalse(lock.get("stage3_require_final_answer"))
        self.assertFalse(lock.get("stage3_repair_untagged_answer"))
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
        self.assertIs(assertions["buffer.explorer_input.taskset.workflow_args.milestone_reward_enabled"], False)
        self.assertEqual(
            [row["hotpot_id"] for row in lock["dataset"]["fixed_train_rows"]],
            M5_TRAIN_IDS,
        )
        self.assertEqual(
            [row["hotpot_id"] for row in lock["dataset"]["fixed_eval_rows"]],
            M5_EVAL_IDS,
        )

    def test_yaml_matches_lock_digests_and_forbids_nudge(self):
        lock = load_lock()
        sources = lock["source_files"]
        self.assertEqual(sources["config"]["path"], "examples/agemem_hotpotqa/agemem_e1_4b_dry_run.yaml")
        self.assertEqual(
            sources["e0_config"]["path"],
            "examples/agemem_hotpotqa/agemem_e0_4b_frozen_eval.yaml",
        )
        self.assertEqual(
            sources["checkpoint_eval_config"]["path"],
            "examples/agemem_hotpotqa/agemem_e1_4b_checkpoint_eval.yaml",
        )
        self.assertEqual(_source_digest(E1_YAML), sources["config"]["sha256"])
        self.assertEqual(_source_digest(E0_YAML), sources["e0_config"]["sha256"])
        self.assertEqual(_source_digest(EVAL_YAML), sources["checkpoint_eval_config"]["sha256"])
        for path in (E0_YAML, E1_YAML, EVAL_YAML):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(yaml_forbids_nudge(text), msg=path.name)
            self.assertIn("reward_profile: terminal_only", text)
            self.assertIn("enable_thinking: false", text)
            self.assertIn("gpu_memory_utilization: 0.6", text)
            self.assertIn("ppo_max_token_len_per_gpu: 2304", text)
            self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", text)
            for row_id in M5_TRAIN_IDS:
                self.assertIn(row_id, text)
        eval_text = EVAL_YAML.read_text(encoding="utf-8")
        for row_id in M5_EVAL_IDS:
            self.assertIn(row_id, eval_text)
        e1_text = E1_YAML.read_text(encoding="utf-8")
        self.assertIn('name: "agemem-e1-terminal-only-4b-dry-run"', e1_text)
        self.assertIn("temperature: 0.6", e1_text)
        self.assertIn("total_steps: 1", e1_text)
        self.assertIn("max_model_len: 5120", e1_text)
        self.assertIn("max_response_tokens: 1024", e1_text)
        self.assertNotIn("max_response_tokens: 512\n", e1_text)

    def test_skipped_special_tokens_keep_exact_zero_width_offsets(self):
        from trinity.common.action_event_contract import (
            ActionContractError,
            derive_response_token_char_offsets,
        )

        class _Tokenizer:
            all_special_ids = [99]

            def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
                del add_special_tokens
                result = {"input_ids": [ord(character) for character in text]}
                if return_offsets_mapping:
                    result["offset_mapping"] = [
                        (index, index + 1) for index in range(len(text))
                    ]
                return result

            def decode(self, token_ids, *, skip_special_tokens=False, **kwargs):
                del kwargs
                chars = []
                for token_id in token_ids:
                    if skip_special_tokens and token_id in self.all_special_ids:
                        continue
                    chars.append(chr(token_id) if token_id != 99 else "")
                return "".join(chars)

        tokenizer = _Tokenizer()
        offsets = derive_response_token_char_offsets(tokenizer, [99, 65, 66], "AB")
        self.assertEqual(offsets, ((0, 0), (0, 1), (1, 2)))
        with self.assertRaisesRegex(ActionContractError, "cannot derive exact"):
            derive_response_token_char_offsets(tokenizer, [1, 2], "ab")

    def test_frozen_1p5b_dry_run_and_runtime_gate_are_untouched(self):
        smoke = json.loads(SMOKE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(_source_digest(DRY_RUN), smoke["source_files"]["config"]["sha256"])
        self.assertEqual(smoke["model"]["repository_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        helper = HELPER.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_contract_test", gate)
        self.assertNotIn("e1_scale_contract_test", gate)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        self.assertNotIn("agemem_e1_dry_run.yaml", launcher)
        self.assertIn("agemem_e1_4b_dry_run.yaml", launcher)
        self.assertIn("configs/e1_4b.json", launcher)
        self.assertIn("stage3_require_final_answer", launcher)
        self.assertIn("Qwen3-4B", helper)
        for job in FORBIDDEN_LEGACY_JOBS[:2]:
            self.assertIn(job, launcher)


if __name__ == "__main__":
    unittest.main()
