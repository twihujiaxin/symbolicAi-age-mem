from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from trinity.common import runtime_receipt

from trinity.common.m8b_postflight import (
    E0_JOB_RELATIVE_PATH,
    E1_JOB_RELATIVE_PATH,
    POSTFLIGHT_SCHEMA_VERSION,
    build_postflight_report,
    main,
)
from trinity.common.runtime_receipt import (
    write_benchmark_receipt,
    write_training_receipt,
)


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"m8b-postflight-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _benchmark_metrics() -> dict[str, float]:
    return {
        "bench/hotpotqa_m5_heldout_smoke/task_score/mean": 0.5,
        "bench/total_time": 1.25,
    }


def _training_metrics() -> dict[str, float]:
    return {
        "actor/pg_loss": 0.125,
        "actor/ppo_kl": 0.01,
        "critic/rewards/mean": 0.5,
        "training/actor_update_completed": 1.0,
    }


def _task_summaries() -> list[dict[str, object]]:
    return [
        {
            "taskset": "hotpotqa_m5_heldout_smoke",
            "task_count": 2,
            "failed_count": 0,
        }
    ]


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _build_valid_evidence(checkpoint_root: Path) -> tuple[Path, Path]:
    e0_job = checkpoint_root / E0_JOB_RELATIVE_PATH
    e1_job = checkpoint_root / E1_JOB_RELATIVE_PATH
    with patch.object(runtime_receipt, "_PROCESS_EXECUTION_ID", "a" * 32):
        write_benchmark_receipt(
            str(e0_job),
            prefix="bench",
            step=0,
            model_version=0,
            task_summaries=_task_summaries(),
            metrics=_benchmark_metrics(),
        )
    with patch.object(runtime_receipt, "_PROCESS_EXECUTION_ID", "b" * 32):
        write_training_receipt(
            str(e1_job),
            completed_step=1,
            configured_total_steps=1,
            metrics=_training_metrics(),
        )
    with patch.object(runtime_receipt, "_PROCESS_EXECUTION_ID", "c" * 32):
        write_benchmark_receipt(
            str(e1_job),
            prefix="bench",
            step=1,
            model_version=1,
            task_summaries=_task_summaries(),
            metrics=_benchmark_metrics(),
        )
    _write_json(
        e1_job / "trainer_meta.json",
        {"latest_exp_index": 4, "latest_iteration": 1},
    )
    (e1_job / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")

    actor_dir = e1_job / "global_step_1" / "actor"
    actor_dir.mkdir(parents=True)
    for prefix in ("model", "optim", "extra_state"):
        (actor_dir / f"{prefix}_world_size_1_rank_0.pt").write_bytes(
            f"{prefix}-state".encode("ascii")
        )
    trained_lora = actor_dir / "lora_adapter"
    dummy_lora = e1_job / "dummy_lora"
    trained_lora.mkdir()
    dummy_lora.mkdir()
    _write_json(trained_lora / "adapter_config.json", {"r": 16})
    _write_json(dummy_lora / "adapter_config.json", {"r": 16})
    (trained_lora / "adapter_model.safetensors").write_bytes(b"trained")
    (dummy_lora / "adapter_model.safetensors").write_bytes(b"dummy")
    return e0_job, e1_job


def _check(report: dict, name: str) -> dict:
    return next(item for item in report["checks"] if item["name"] == name)


class M8bPostflightTest(unittest.TestCase):
    def test_complete_evidence_passes(self):
        with workspace_temp_directory() as checkpoint_root:
            _build_valid_evidence(checkpoint_root)

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(report["schema_version"], POSTFLIGHT_SCHEMA_VERSION)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["checks"]), 10)
            self.assertTrue(all(item["status"] == "pass" for item in report["checks"]))
            checkpoint_eval = report["evidence"]["checkpoint_evaluation"]
            self.assertEqual(checkpoint_eval["model_version"], 1)
            self.assertEqual(
                checkpoint_eval["new_process_proof"],
                "distinct_process_execution_id",
            )

    def test_checkpoint_eval_must_use_a_distinct_process_execution_id(self):
        with workspace_temp_directory() as checkpoint_root:
            _e0_job, e1_job = _build_valid_evidence(checkpoint_root)
            training = json.loads(
                (e1_job / "receipts" / "trainer_step_1.json").read_text(
                    encoding="utf-8"
                )
            )
            eval_receipt = (
                e1_job / "receipts" / "bench_step_1_model_1.json"
            )
            evaluation = json.loads(eval_receipt.read_text(encoding="utf-8"))
            evaluation["process_execution_id"] = training[
                "process_execution_id"
            ]
            _write_json(eval_receipt, evaluation)

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                _check(report, "e1.checkpoint_eval_new_process")["status"],
                "fail",
            )

    def test_e0_receipt_must_prove_base_model_evaluation(self):
        with workspace_temp_directory() as checkpoint_root:
            e0_job, _e1_job = _build_valid_evidence(checkpoint_root)
            receipt = e0_job / "receipts" / "bench_step_0_model_0.json"
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            # bool is an int subclass in Python; strict schema validation must
            # still reject it instead of accepting False as model version zero.
            payload["model_version"] = False
            _write_json(receipt, payload)

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(report["status"], "fail")
            self.assertEqual(_check(report, "e0.benchmark_receipt")["status"], "fail")

    def test_benchmark_receipt_requires_complete_fixed_taskset_and_score(self):
        with workspace_temp_directory() as temp_root:
            for scenario in ("partial_taskset", "missing_score"):
                with self.subTest(scenario=scenario):
                    checkpoint_root = temp_root / scenario
                    e0_job, _e1_job = _build_valid_evidence(checkpoint_root)
                    receipt = e0_job / "receipts" / "bench_step_0_model_0.json"
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                    if scenario == "partial_taskset":
                        payload["task_summaries"][0]["task_count"] = 1
                    else:
                        payload["metrics"] = {"bench/total_time": 1.0}
                    _write_json(receipt, payload)

                    report = build_postflight_report(
                        checkpoint_root=checkpoint_root
                    )

                    self.assertEqual(report["status"], "fail")
                    self.assertEqual(
                        _check(report, "e0.benchmark_receipt")["status"],
                        "fail",
                    )

    def test_training_receipt_requires_semantic_metrics_and_update_sentinel(self):
        scenarios = {
            "missing_reward": {
                "actor/pg_loss": 0.1,
                "actor/ppo_kl": 0.01,
                "training/actor_update_completed": 1.0,
            },
            "missing_sentinel": {
                "actor/pg_loss": 0.1,
                "actor/ppo_kl": 0.01,
                "critic/rewards/mean": 0.5,
            },
            "nonfinite": {
                "actor/pg_loss": float("nan"),
                "actor/ppo_kl": 0.01,
                "critic/rewards/mean": 0.5,
                "training/actor_update_completed": 1.0,
            },
        }
        with workspace_temp_directory() as temp_root:
            for name, metrics in scenarios.items():
                with self.subTest(name=name):
                    checkpoint_root = temp_root / name
                    _e0_job, e1_job = _build_valid_evidence(checkpoint_root)
                    receipt = e1_job / "receipts" / "trainer_step_1.json"
                    payload = json.loads(receipt.read_text(encoding="utf-8"))
                    payload["metrics"] = metrics
                    _write_json(receipt, payload, allow_nan=True)

                    report = build_postflight_report(checkpoint_root=checkpoint_root)

                    self.assertEqual(report["status"], "fail")
                    failed_names = {
                        item["name"]
                        for item in report["checks"]
                        if item["status"] == "fail"
                    }
                    self.assertTrue(
                        failed_names
                        & {
                            "e1.training_receipt",
                            "e1.training_metrics",
                            "e1.actor_update",
                        }
                    )

    def test_trainer_state_and_checkpoint_marker_must_equal_step_one(self):
        with workspace_temp_directory() as checkpoint_root:
            _e0_job, e1_job = _build_valid_evidence(checkpoint_root)
            _write_json(
                e1_job / "trainer_meta.json",
                {"latest_exp_index": 0, "latest_iteration": 0},
            )
            (e1_job / "latest_checkpointed_iteration.txt").write_text(
                "2", encoding="utf-8"
            )

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(_check(report, "e1.trainer_meta")["status"], "fail")
            self.assertEqual(_check(report, "e1.latest_checkpoint")["status"], "fail")

    def test_checkpoint_shards_must_be_complete_and_nonempty(self):
        with workspace_temp_directory() as temp_root:
            missing_root = temp_root / "missing"
            _e0_job, e1_job = _build_valid_evidence(missing_root)
            actor_dir = e1_job / "global_step_1" / "actor"
            (actor_dir / "extra_state_world_size_1_rank_0.pt").unlink()
            missing_report = build_postflight_report(checkpoint_root=missing_root)

            empty_root = temp_root / "empty"
            _e0_job, e1_job = _build_valid_evidence(empty_root)
            actor_dir = e1_job / "global_step_1" / "actor"
            (actor_dir / "optim_world_size_1_rank_0.pt").write_bytes(b"")
            empty_report = build_postflight_report(checkpoint_root=empty_root)

            self.assertEqual(
                _check(missing_report, "e1.checkpoint_shards")["status"],
                "fail",
            )
            self.assertEqual(
                _check(empty_report, "e1.checkpoint_shards")["status"],
                "fail",
            )

    def test_trained_lora_must_differ_from_dummy_lora(self):
        with workspace_temp_directory() as checkpoint_root:
            _e0_job, e1_job = _build_valid_evidence(checkpoint_root)
            trained_weights = (
                e1_job
                / "global_step_1"
                / "actor"
                / "lora_adapter"
                / "adapter_model.safetensors"
            )
            dummy_weights = e1_job / "dummy_lora" / "adapter_model.safetensors"
            trained_weights.write_bytes(dummy_weights.read_bytes())

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(_check(report, "e1.lora_adapter")["status"], "fail")

    def test_checkpoint_eval_must_prove_model_version_one(self):
        with workspace_temp_directory() as checkpoint_root:
            _e0_job, e1_job = _build_valid_evidence(checkpoint_root)
            receipt = e1_job / "receipts" / "bench_step_1_model_1.json"
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["model_version"] = 0
            _write_json(receipt, payload)

            report = build_postflight_report(checkpoint_root=checkpoint_root)

            self.assertEqual(
                _check(report, "e1.checkpoint_eval_receipt")["status"],
                "fail",
            )

    def test_cli_writes_report_and_returns_nonzero_for_missing_evidence(self):
        with workspace_temp_directory() as checkpoint_root:
            e0_job, _e1_job = _build_valid_evidence(checkpoint_root)
            output = checkpoint_root / "postflight.json"
            with redirect_stdout(StringIO()):
                success = main(
                    [
                        "--checkpoint-root",
                        str(checkpoint_root),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(success, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "pass",
            )

            (e0_job / "receipts" / "bench_step_0_model_0.json").unlink()
            with redirect_stdout(StringIO()):
                failure = main(
                    ["--checkpoint-root", str(checkpoint_root), "--no-write"]
                )
            self.assertEqual(failure, 1)


if __name__ == "__main__":
    unittest.main()
