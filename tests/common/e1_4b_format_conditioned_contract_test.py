"""Contract tests for the format-conditioned 4B protocol and frozen diagnosis.

These tests are not part of the frozen M8b 318-count runtime gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from trinity.common.e1_4b import yaml_forbids_nudge
from trinity.common.e1_4b_format_conditioned import (
    ALL_JOBS,
    DIAGNOSIS_FLAG_STRINGS,
    FROZEN_CLEAN_YAMLS,
    HELDOUT_JOB,
    HELDOUT_YAML,
    LOCK_PATH,
    MEM_GOLD_JOB,
    MEM_GOLD_YAML,
    MEM_NO_RETRIEVE_JOB,
    MEM_NO_RETRIEVE_YAML,
    MEM_NORMAL_JOB,
    MEM_NORMAL_YAML,
    SCALE_LOCK_PATH,
    SELECTION_SEED,
    SIGNAL_JOB,
    SIGNAL_YAML,
    job_requires_frozen_selection,
    load_lock,
    load_scale_lock,
    selection_is_frozen,
    train_rows_match_scale,
    write_mem_yamls,
)
from trinity.common.m8b_preflight import _source_digest


def _load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag.sh"
SELECTOR = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_select.py"
REPORT = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_conditioned_diag_report.py"
WORKFLOW = (
    REPOSITORY_ROOT
    / "trinity"
    / "common"
    / "workflows"
    / "memory_context"
    / "train_hotpotQA.py"
)
RUNTIME_GATE = REPOSITORY_ROOT / "scripts" / "agemem_m8b_runtime_gate.py"
VANILLA_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b.sh"
FORMAT_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format.sh"
VAR_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_var.sh"
GROUP_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_format_group.sh"
PROBE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "agemem_e1_4b_stage3_answer_probe.sh"
SCALE_YAML = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_scale.yaml"
DRY_RUN = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
DRY_RUN_4B = REPOSITORY_ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_4b_dry_run.yaml"

VIEWED_VALIDATION_IDS = [
    "5ab3d2b7554299233954ffb8",
    "5ab299d6554299449642c926",
    "5ab7c6995542993667794005",
    "5adc8c545542994734353734",
]
HELD_OUT_IDS = [
    "5ab7c6995542993667794005",
    "5adc8c545542994734353734",
]


class E14BFormatConditionedContractTest(unittest.TestCase):
    def test_lock_copies_scale_train_and_tracks_selection_status(self):
        lock = load_lock()
        scale = load_scale_lock()
        status = str(lock["selection_status"])
        self.assertEqual(lock["schema_version"], "agemem.e1_4b_format_conditioned.lock.v1")
        self.assertEqual(lock["experiment_id"], "e1_format_conditioned_4b_protocol")
        self.assertIn(status, {"pending", "frozen"})
        self.assertTrue(lock["stage3_require_final_answer"])
        self.assertTrue(lock["stage3_repair_untagged_answer"])
        self.assertEqual(lock["reward_profile"], "terminal_only")
        self.assertEqual(lock["terminal_reward_metric"], "hotpotqa_official")
        self.assertEqual(lock["trainer_total_steps"], 0)
        self.assertEqual(lock["selection_seed"], SELECTION_SEED)
        self.assertNotEqual(lock["selection_seed"], lock["smoke_seed"])
        self.assertNotEqual(lock["selection_seed"], lock["scale_selection_seed"])
        self.assertEqual(lock["smoke_seed"], 20260802)
        self.assertEqual(lock["scale_selection_seed"], 20260904)
        self.assertEqual(lock["model"]["repository_id"], "Qwen/Qwen3-4B")
        self.assertEqual(
            lock["model"]["expected_revision"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(lock["expected_dataset_fingerprint"], "c369f1b07b350d37")
        self.assertEqual(lock["eval_dataset_fingerprint"], "fbe86cb2d14cb199")
        self.assertEqual(lock["train_size"], 24)
        self.assertEqual(lock["dev_size"], 32)
        self.assertEqual(lock["test_size"], 128)
        self.assertEqual(lock["jobs"]["signal"], SIGNAL_JOB)
        self.assertEqual(lock["jobs"]["heldout"], HELDOUT_JOB)
        self.assertEqual(lock["jobs"]["mem_normal"], MEM_NORMAL_JOB)
        self.assertEqual(lock["jobs"]["mem_no_retrieve"], MEM_NO_RETRIEVE_JOB)
        self.assertEqual(lock["jobs"]["mem_gold_support"], MEM_GOLD_JOB)
        self.assertEqual(lock["fixed_train_rows"], scale["fixed_train_rows"])
        self.assertTrue(train_rows_match_scale(lock, scale))
        self.assertEqual(lock["held_out_row_ids"], HELD_OUT_IDS)
        self.assertEqual(lock["excluded_validation_ids"], VIEWED_VALIDATION_IDS)
        self.assertEqual(
            json.loads(SCALE_LOCK_PATH.read_text(encoding="utf-8"))["fixed_train_rows"],
            lock["fixed_train_rows"],
        )
        if status == "pending":
            self.assertFalse(selection_is_frozen(lock))
            self.assertEqual(lock["fixed_dev_rows"], [])
            self.assertEqual(lock["fixed_test_rows"], [])
        else:
            self.assertTrue(selection_is_frozen(lock))
            excluded = set(VIEWED_VALIDATION_IDS)
            dev_ids = [str(row["hotpot_id"]) for row in lock["fixed_dev_rows"]]
            test_ids = [str(row["hotpot_id"]) for row in lock["fixed_test_rows"]]
            self.assertEqual(len(dev_ids), 32)
            self.assertEqual(len(test_ids), 128)
            self.assertTrue(excluded.isdisjoint(dev_ids))
            self.assertTrue(excluded.isdisjoint(test_ids))
            self.assertTrue(set(dev_ids).isdisjoint(test_ids))

    def test_signal_and_heldout_yamls_match_lock(self):
        lock = load_lock()
        sources = lock["source_files"]
        self.assertEqual(_source_digest(SIGNAL_YAML), sources["signal_config"]["sha256"])
        self.assertEqual(_source_digest(HELDOUT_YAML), sources["heldout_config"]["sha256"])
        signal = SIGNAL_YAML.read_text(encoding="utf-8")
        heldout = HELDOUT_YAML.read_text(encoding="utf-8")
        self.assertIn(f'name: "{SIGNAL_JOB}"', signal)
        self.assertIn("mode: bench", signal)
        self.assertIn("repeat_times: 4", signal)
        self.assertIn("max_repeat_times_per_runner: 4", signal)
        self.assertIn("temperature: 0.6", signal)
        self.assertIn("total_steps: 12", signal)
        self.assertIn("stage3_require_final_answer: true", signal)
        self.assertIn("stage3_repair_untagged_answer: true", signal)
        self.assertIn("reward_profile: terminal_only", signal)
        self.assertIn("hotpotqa_official", signal)
        self.assertIn("/data/hjx/Age_mem/models/Qwen3-4B", signal)
        self.assertIn("max_model_len: 5120", signal)
        self.assertIn("gpu_memory_utilization: 0.6", signal)
        self.assertNotIn("consume_put_batch", signal)
        self.assertNotIn("stage3_disable_ltm_retrieve", signal)
        self.assertNotIn("stage3_inject_gold_supporting", signal)
        for row in lock["fixed_train_rows"]:
            self.assertIn(row["hotpot_id"], signal)
        self.assertIn(f'name: "{HELDOUT_JOB}"', heldout)
        self.assertIn("repeat_times: 1", heldout)
        self.assertIn("temperature: 0.0", heldout)
        self.assertIn("total_steps: 1", heldout)
        self.assertIn("split: validation", heldout)
        for row_id in HELD_OUT_IDS:
            self.assertIn(row_id, heldout)
        self.assertNotIn("consume_put_batch", heldout)

    def test_memory_yamls_match_selection_status(self):
        lock = load_lock()
        self.assertTrue(job_requires_frozen_selection(MEM_NORMAL_JOB))
        self.assertTrue(job_requires_frozen_selection("mem-gold-support"))
        self.assertFalse(job_requires_frozen_selection(SIGNAL_JOB))
        self.assertFalse(job_requires_frozen_selection("heldout"))
        if lock["selection_status"] == "pending":
            for path in (MEM_NORMAL_YAML, MEM_NO_RETRIEVE_YAML, MEM_GOLD_YAML):
                self.assertFalse(path.exists(), msg=path.name)
            with self.assertRaises(ValueError):
                write_mem_yamls(lock)
            for key in (
                "mem_normal_config",
                "mem_no_retrieve_config",
                "mem_gold_support_config",
            ):
                self.assertIsNone(lock["source_files"][key]["sha256"])
            return
        self.assertTrue(selection_is_frozen(lock))
        sources = lock["source_files"]
        self.assertTrue(MEM_NORMAL_YAML.is_file())
        self.assertTrue(MEM_NO_RETRIEVE_YAML.is_file())
        self.assertTrue(MEM_GOLD_YAML.is_file())
        self.assertEqual(_source_digest(MEM_NORMAL_YAML), sources["mem_normal_config"]["sha256"])
        self.assertEqual(
            _source_digest(MEM_NO_RETRIEVE_YAML),
            sources["mem_no_retrieve_config"]["sha256"],
        )
        self.assertEqual(_source_digest(MEM_GOLD_YAML), sources["mem_gold_support_config"]["sha256"])
        normal = MEM_NORMAL_YAML.read_text(encoding="utf-8")
        no_retrieve = MEM_NO_RETRIEVE_YAML.read_text(encoding="utf-8")
        gold = MEM_GOLD_YAML.read_text(encoding="utf-8")
        self.assertNotIn("stage3_disable_ltm_retrieve", normal)
        self.assertNotIn("stage3_inject_gold_supporting", normal)
        self.assertIn("stage3_disable_ltm_retrieve: true", no_retrieve)
        self.assertIn("stage3_inject_gold_supporting: true", gold)
        self.assertNotIn("consume_put_batch", normal)
        for row in lock["fixed_dev_rows"]:
            self.assertIn(row["hotpot_id"], normal)
            self.assertIn(row["hotpot_id"], no_retrieve)
            self.assertIn(row["hotpot_id"], gold)

    def test_diagnosis_flags_default_false_and_stay_out_of_frozen_yamls(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('workflow_args.get("stage3_disable_ltm_retrieve", False)', workflow)
        self.assertIn('workflow_args.get("stage3_inject_gold_supporting", False)', workflow)
        self.assertIn("extract_sentences_from_supporting_facts", workflow)
        self.assertIn("privileged_gold_supporting", workflow)
        self.assertIn("ltm_retrieve_disabled:stage3_diagnosis", workflow)
        for path in FROZEN_CLEAN_YAMLS:
            text = path.read_text(encoding="utf-8")
            for flag in DIAGNOSIS_FLAG_STRINGS:
                self.assertNotIn(flag, text, msg=path.name)
        self.assertTrue(yaml_forbids_nudge(DRY_RUN_4B.read_text(encoding="utf-8")))
        self.assertNotIn("stage3_require_final_answer: true", DRY_RUN.read_text(encoding="utf-8"))
        scale_yaml = SCALE_YAML.read_text(encoding="utf-8")
        for flag in DIAGNOSIS_FLAG_STRINGS:
            self.assertNotIn(flag, scale_yaml)

    def test_launcher_and_runtime_gate_stay_independent(self):
        gate = RUNTIME_GATE.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        selector = SELECTOR.read_text(encoding="utf-8")
        report = REPORT.read_text(encoding="utf-8")
        self.assertNotIn("e1_4b_format_conditioned_contract_test", gate)
        self.assertNotIn("agemem_e1_4b_format_conditioned", gate)
        self.assertIn("configs/e1_4b_format_conditioned.json", launcher)
        self.assertIn("agemem_e1_4b_format_conditioned_diag_report.py", launcher)
        self.assertIn("frozen 32-dev selection", launcher)
        self.assertIn('selection_status"] = "frozen"', selector)
        self.assertIn("HOTPOTQA_PATH is missing", selector)
        self.assertIn("source_split=\"validation\"", selector.replace(" ", ""))
        self.assertIn("benchmark_split=\"dev\"", selector.replace(" ", ""))
        self.assertIn("group_std", report)
        self.assertIn("used_by_following_response", report)
        self.assertNotIn("autodl_m8b_smoke.sh", launcher)
        self.assertNotIn("agemem_e1_dry_run.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_dry_run.yaml", launcher)
        self.assertNotIn("agemem_e1_4b_format_group.yaml", launcher)
        self.assertIn("Qwen3-4B", launcher)
        for job in ALL_JOBS:
            self.assertIn(job, launcher)
        for other in (VANILLA_LAUNCHER, FORMAT_LAUNCHER, VAR_LAUNCHER, GROUP_LAUNCHER, PROBE_LAUNCHER):
            text = other.read_text(encoding="utf-8")
            for job in ALL_JOBS:
                self.assertIn(job, text, msg=other.name)

    def test_diag_report_reads_stage3_and_receipts(self):
        import tempfile

        summarize_job = _load_module("fc_report", REPORT).summarize_job

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            job_dir = root / "Trinity-RFT-AgeMem-M8" / SIGNAL_JOB
            traj = job_dir / "trajectories"
            receipts = job_dir / "receipts"
            traj.mkdir(parents=True)
            receipts.mkdir(parents=True)
            (traj / "stage3_final_turn.jsonl").write_text(
                json.dumps(
                    {
                        "execution_id": "a",
                        "found_answer": True,
                        "has_answer_tag": True,
                        "has_tool_call": False,
                        "nudged": True,
                        "parsed_answer": "x",
                        "repaired": False,
                        "round": 1,
                        "stage": 3,
                        "task_id": "task-1",
                        "task_score": 0.4,
                    },
                    sort_keys=True,
                )
                + "\n"
                + json.dumps(
                    {
                        "execution_id": "b",
                        "found_answer": True,
                        "has_answer_tag": True,
                        "has_tool_call": False,
                        "nudged": True,
                        "parsed_answer": "y",
                        "repaired": False,
                        "round": 1,
                        "stage": 3,
                        "task_id": "task-1",
                        "task_score": 1.0,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (traj / "tool_calls.jsonl").write_text(
                json.dumps(
                    {
                        "tool_name": "Retrieve_memory",
                        "status": "success",
                        "result": {
                            "outcome": "retrieved",
                            "used_by_following_response": True,
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (receipts / "bench_step_0_model_0.json").write_text(
                json.dumps(
                    {
                        "metrics": {"eval/hotpotqa_fc_train_24/task_score/mean": 0.7},
                        "task_summaries": [{"taskset": "hotpotqa_fc_train_24", "failed_count": 0}],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            summary = summarize_job(root, SIGNAL_JOB)
            self.assertEqual(summary["tasks_with_group_std_gt_0"], 1)
            self.assertEqual(summary["action_contract_join_failures"], 0)
            self.assertEqual(summary["retrieve_used_by_following_response"], 1)

    def test_scale_lock_file_was_not_edited(self):
        scale = json.loads(SCALE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(scale["schema_version"], "agemem.e1_scale.lock.v1")
        self.assertEqual(scale["selection_status"], "frozen")
        self.assertFalse(scale["stage3_require_final_answer"])
        self.assertEqual(len(scale["fixed_train_rows"]), 24)


if __name__ == "__main__":
    unittest.main()
