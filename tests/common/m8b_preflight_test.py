from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from trinity.common import m8b_preflight
from trinity.common.hf_task_dataset import load_task_dataset, select_task_rows
from trinity.common.m8b_model_manifest import (
    build_model_manifest,
    write_model_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/agemem_hotpotqa/agemem_e1_dry_run.yaml"
LOCK = ROOT / "configs/m8b_autodl_preflight.json"
E0_CONFIG = ROOT / "examples/agemem_hotpotqa/agemem_e0_frozen_eval.yaml"
CHECKPOINT_EVAL_CONFIG = (
    ROOT / "examples/agemem_hotpotqa/agemem_e1_checkpoint_eval.yaml"
)


@contextmanager
def workspace_temp_directory():
    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"m8b-preflight-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class PreflightLockTest(unittest.TestCase):
    def test_real_lock_config_and_manifest_are_aligned(self):
        gates = m8b_preflight.GateBook()
        lock = m8b_preflight._check_lock(LOCK, CONFIG, ROOT, gates)
        self.assertIsNotNone(lock)
        self.assertEqual(
            lock["model"]["repository_id"],
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(
            lock["model"]["config_assertions"],
            {
                "architectures.0": "Qwen2ForCausalLM",
                "hidden_size": 1536,
                "intermediate_size": 8960,
                "model_type": "qwen2",
                "num_attention_heads": 12,
                "num_hidden_layers": 28,
                "num_key_value_heads": 2,
                "vocab_size": 151936,
            },
        )
        self.assertEqual(lock["model"]["minimum_weight_bytes"], 3_000_000_000)
        self.assertEqual(
            lock["model"]["required_files"],
            [
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
        config = m8b_preflight._check_config(CONFIG, lock, gates)
        self.assertIsNotNone(config)
        self.assertIn(
            "Qwen2.5-1.5B-Instruct",
            config["model"]["model_path"],
        )
        manifest = m8b_preflight._check_manifest_consistency(
            ROOT, lock, config, gates
        )
        self.assertIsNotNone(manifest)
        self.assertTrue(gates.passed, [result.to_dict() for result in gates.results])

    def test_terminal_reward_or_provider_mutation_fails_contract(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        config = m8b_preflight._load_yaml_object(CONFIG)
        config["buffer"]["explorer_input"]["taskset"]["workflow_args"][
            "milestone_reward_enabled"
        ] = True
        config["buffer"]["explorer_input"]["taskset"]["workflow_args"][
            "auxiliary_provider"
        ]["embedding_model"] = "unfrozen-model"

        with patch.object(
            m8b_preflight, "_load_yaml_object", return_value=config
        ):
            gates = m8b_preflight.GateBook()
            m8b_preflight._check_config(CONFIG, lock, gates)

        result = gates.results[-1]
        self.assertEqual(result.status, m8b_preflight.FAIL)
        error_paths = {error["path"] for error in result.details["errors"]}
        self.assertIn(
            "buffer.explorer_input.taskset.workflow_args.milestone_reward_enabled",
            error_paths,
        )
        self.assertIn(
            "buffer.explorer_input.taskset.workflow_args.auxiliary_provider",
            error_paths,
        )

    def test_e0_and_checkpoint_eval_use_the_fixed_heldout_rows(self):
        manifest = json.loads(
            (ROOT / "data/splits/hotpotqa_smoke_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_rows = [
            row
            for row in manifest["selections"]
            if row["benchmark_split"] == "test"
        ]
        expected_indices = [row["source_index"] for row in expected_rows]
        expected_ids = [row["hotpot_id"] for row in expected_rows]

        e0 = m8b_preflight._load_yaml_object(E0_CONFIG)
        checkpoint_eval = m8b_preflight._load_yaml_object(
            CHECKPOINT_EVAL_CONFIG
        )
        train = m8b_preflight._load_yaml_object(CONFIG)
        for config in (e0, checkpoint_eval):
            taskset = config["buffer"]["explorer_input"]["eval_tasksets"][0]
            workflow_args = taskset["workflow_args"]
            self.assertEqual(config["mode"], "bench")
            self.assertEqual(config["algorithm"]["repeat_times"], 2)
            self.assertEqual(taskset["split"], "validation")
            self.assertEqual(taskset["row_indices"], expected_indices)
            self.assertEqual(taskset["expected_row_ids"], expected_ids)
            self.assertEqual(
                taskset["expected_dataset_fingerprint"],
                manifest["source_fingerprints"]["validation"],
            )
            self.assertEqual(workflow_args["reward_profile"], "terminal_only")
            self.assertIs(workflow_args["milestone_reward_enabled"], False)
            self.assertEqual(
                workflow_args["auxiliary_provider"],
                train["buffer"]["explorer_input"]["taskset"][
                    "workflow_args"
                ]["auxiliary_provider"],
            )

        self.assertNotIn("lora_configs", e0["model"])
        self.assertIs(e0["explorer"]["eval_on_startup"], True)
        self.assertIs(e0["explorer"]["bench_on_latest_checkpoint"], False)
        self.assertEqual(checkpoint_eval["name"], train["name"])
        self.assertIs(checkpoint_eval["continue_from_checkpoint"], True)
        self.assertIs(
            checkpoint_eval["explorer"]["bench_on_latest_checkpoint"], True
        )
        self.assertIs(
            checkpoint_eval["explorer"]["eval_on_startup"], False
        )

    def test_source_digest_mutation_is_rejected(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        lock["source_files"]["config"]["sha256"] = "0" * 64
        with patch.object(
            m8b_preflight, "_load_json_object", return_value=lock
        ):
            gates = m8b_preflight.GateBook()
            m8b_preflight._check_lock(
                LOCK, CONFIG, ROOT.resolve(), gates
            )

        config_gate = next(
            result for result in gates.results if result.name == "lock.source.config"
        )
        self.assertEqual(config_gate.status, m8b_preflight.FAIL)


class CredentialIsolationTest(unittest.TestCase):
    def test_report_records_presence_only(self):
        secret = "this-value-must-never-appear"
        lock = {
            "credentials": {
                "required_env": ["DASHSCOPE_API_KEY"],
                "forbidden_repo_paths": [],
            }
        }
        gates = m8b_preflight.GateBook()
        presence = m8b_preflight._check_credential_isolation(
            ROOT,
            lock,
            {"DASHSCOPE_API_KEY": secret},
            mode="autodl",
            gates=gates,
        )
        serialized = json.dumps(
            {
                "presence": presence,
                "gates": [result.to_dict() for result in gates.results],
            },
            sort_keys=True,
        )
        self.assertNotIn(secret, serialized)
        self.assertIn('"DASHSCOPE_API_KEY": true', serialized)

    def test_empty_key_and_nested_dotenv_fail_without_reading_file(self):
        with workspace_temp_directory() as root:
            nested = root / "nested"
            nested.mkdir()
            (nested / ".env").write_text(
                "DASHSCOPE_API_KEY=must-not-be-read\n", encoding="utf-8"
            )
            lock = {
                "credentials": {
                    "required_env": ["DASHSCOPE_API_KEY"],
                    "forbidden_repo_paths": [".env"],
                }
            }
            gates = m8b_preflight.GateBook()
            presence = m8b_preflight._check_credential_isolation(
                root,
                lock,
                {"DASHSCOPE_API_KEY": "   "},
                mode="autodl",
                gates=gates,
            )
        self.assertEqual(presence, {"DASHSCOPE_API_KEY": False})
        self.assertFalse(gates.passed)
        serialized = json.dumps(
            [result.to_dict() for result in gates.results], sort_keys=True
        )
        self.assertNotIn("must-not-be-read", serialized)
        self.assertIn("nested/.env", serialized)


class PersistentPathGateTest(unittest.TestCase):
    def test_nonempty_smoke_job_fails_autodl_gate(self):
        lock = {
            "paths": {
                "autodl_persistent_prefix": str(ROOT),
                "minimum_checkpoint_free_gib": 0,
                "clean_job_relative_paths": ["docs"],
            }
        }
        gates = m8b_preflight.GateBook()
        m8b_preflight._check_paths(
            mode="autodl",
            model_path=None,
            model_revision=None,
            dataset_path=ROOT,
            checkpoint_root=ROOT,
            lock=lock,
            gates=gates,
        )
        job_gate = next(
            result
            for result in gates.results
            if result.name == "checkpoint.jobs"
        )
        self.assertEqual(job_gate.status, m8b_preflight.FAIL)
        self.assertEqual(job_gate.details["nonempty_jobs"], ["docs"])

    def test_file_at_smoke_job_path_is_not_treated_as_clean(self):
        with workspace_temp_directory() as root:
            (root / "occupied").write_text("not a directory", encoding="utf-8")
            lock = {
                "paths": {
                    "autodl_persistent_prefix": str(root),
                    "minimum_checkpoint_free_gib": 0,
                    "clean_job_relative_paths": ["occupied"],
                }
            }
            gates = m8b_preflight.GateBook()
            m8b_preflight._check_paths(
                mode="autodl",
                model_path=None,
                model_revision=None,
                dataset_path=root,
                checkpoint_root=root,
                lock=lock,
                gates=gates,
            )
        job_gate = next(
            result
            for result in gates.results
            if result.name == "checkpoint.jobs"
        )
        self.assertEqual(job_gate.status, m8b_preflight.FAIL)


class FakeSelection:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [record[key] for record in self.records]
        return self.records[key]


class FakeSplit:
    def __init__(self, size, fingerprint, selected_records=None):
        self._size = size
        self._fingerprint = fingerprint
        self._selected_records = selected_records or []

    def __len__(self):
        return self._size

    def select(self, _indices):
        return FakeSelection(list(self._selected_records))


class FakeDatasetDict(dict):
    pass


class DatasetIdentityTest(unittest.TestCase):
    def make_datasets_module(self, dataset):
        module = ModuleType("datasets")
        module.DatasetDict = FakeDatasetDict
        module.load_from_disk = lambda _path: dataset
        return module

    def test_exact_split_fingerprint_and_ids_pass(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        dataset_lock = lock["dataset"]
        records = {
            "train": [
                {"id": row["hotpot_id"], "question": f"train-{index}"}
                for index, row in enumerate(dataset_lock["fixed_train_rows"])
            ],
            "validation": [
                {"id": row["hotpot_id"], "question": f"eval-{index}"}
                for index, row in enumerate(dataset_lock["fixed_eval_rows"])
            ],
        }
        for label, row_lock in (
            ("train", dataset_lock["fixed_train_rows"]),
            ("validation", dataset_lock["fixed_eval_rows"]),
        ):
            for row, record in zip(row_lock, records[label]):
                row["content_sha256"] = m8b_preflight._canonical_json_sha256(
                    record
                )
        dataset = FakeDatasetDict(
            {
                split: FakeSplit(
                    dataset_lock["source_split_sizes"][split],
                    dataset_lock["source_fingerprints"][split],
                    records.get(split),
                )
                for split in ("train", "validation", "test")
            }
        )
        gates = m8b_preflight.GateBook()
        with patch.object(Path, "is_file", return_value=True), patch.dict(
            sys.modules,
            {"datasets": self.make_datasets_module(dataset)},
        ):
            inventory = m8b_preflight._check_dataset(ROOT, lock, gates)
        self.assertTrue(gates.passed)
        self.assertEqual(
            inventory["selected_train_ids"],
            [record["id"] for record in records["train"]],
        )
        self.assertEqual(
            inventory["selected_eval_ids"],
            [record["id"] for record in records["validation"]],
        )

    def test_wrong_selected_id_fails(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        dataset_lock = lock["dataset"]
        train_records = [
            {"id": "wrong", "question": str(index)}
            for index, _row in enumerate(dataset_lock["fixed_train_rows"])
        ]
        eval_records = [
            {"id": row["hotpot_id"], "question": str(index)}
            for index, row in enumerate(dataset_lock["fixed_eval_rows"])
        ]
        dataset = FakeDatasetDict(
            {
                split: FakeSplit(
                    dataset_lock["source_split_sizes"][split],
                    dataset_lock["source_fingerprints"][split],
                    train_records
                    if split == "train"
                    else (eval_records if split == "validation" else None),
                )
                for split in ("train", "validation", "test")
            }
        )
        gates = m8b_preflight.GateBook()
        with patch.object(Path, "is_file", return_value=True), patch.dict(
            sys.modules,
            {"datasets": self.make_datasets_module(dataset)},
        ):
            m8b_preflight._check_dataset(ROOT, lock, gates)
        self.assertFalse(gates.passed)
        self.assertIn(
            "fixed_train_ids", gates.results[-1].details["mismatches"]
        )

    def test_selected_row_content_mutation_fails(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        dataset_lock = lock["dataset"]
        train_records = [
            {"id": row["hotpot_id"], "question": "mutated"}
            for row in dataset_lock["fixed_train_rows"]
        ]
        eval_records = [
            {"id": row["hotpot_id"], "question": "mutated"}
            for row in dataset_lock["fixed_eval_rows"]
        ]
        dataset = FakeDatasetDict(
            {
                split: FakeSplit(
                    dataset_lock["source_split_sizes"][split],
                    dataset_lock["source_fingerprints"][split],
                    train_records
                    if split == "train"
                    else (eval_records if split == "validation" else None),
                )
                for split in ("train", "validation", "test")
            }
        )
        gates = m8b_preflight.GateBook()
        with patch.object(Path, "is_file", return_value=True), patch.dict(
            sys.modules,
            {"datasets": self.make_datasets_module(dataset)},
        ):
            m8b_preflight._check_dataset(ROOT, lock, gates)
        self.assertIn(
            "fixed_train_content",
            gates.results[-1].details["mismatches"],
        )


class RealDatasetAdapterTest(unittest.TestCase):
    def test_real_saved_dataset_routes_through_m8a_reader(self):
        path = Path(
            os.environ.get(
                "HOTPOTQA_PATH",
                str(ROOT.parent / "data" / "hotpot_qa" / "fullwiki"),
            )
        )
        self.assertTrue(
            (path / "dataset_dict.json").is_file(),
            f"M8b requires a saved fullwiki DatasetDict at {path}",
        )
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        rows = lock["dataset"]["fixed_train_rows"]
        dataset = load_task_dataset(str(path), None, "train")
        selected = select_task_rows(
            dataset,
            [row["source_index"] for row in rows],
            expected_row_ids=[row["hotpot_id"] for row in rows],
            row_id_key="id",
            expected_dataset_fingerprint=lock["dataset"][
                "source_fingerprints"
            ]["train"],
        )
        self.assertEqual(
            selected["id"], [row["hotpot_id"] for row in rows]
        )


class ModelIdentityTest(unittest.TestCase):
    def _make_model(self, root: Path, revision: str):
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 8,
            "model_type": "qwen2",
        }
        (root / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
        )
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
        output = root / ".agemem_model_manifest.json"
        manifest = build_model_manifest(
            root,
            repository_id="Qwen/Qwen2.5-1.5B-Instruct",
            revision=revision,
            output_path=output,
        )
        write_model_manifest(manifest, output)

    @staticmethod
    def _lock():
        return {
            "model": {
                "repository_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "manifest_filename": ".agemem_model_manifest.json",
                "minimum_weight_bytes": 1,
                "required_files": [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                ],
                "config_assertions": {
                    "architectures.0": "Qwen2ForCausalLM",
                    "hidden_size": 8,
                    "model_type": "qwen2",
                },
            }
        }

    def test_manifest_and_structure_pass_then_mutation_fails(self):
        with workspace_temp_directory() as root:
            revision = "c" * 40
            self._make_model(root, revision)
            gates = m8b_preflight.GateBook()
            inventory = m8b_preflight._check_model(
                root,
                revision,
                self._lock(),
                mode="autodl",
                gates=gates,
            )
            self.assertTrue(gates.passed)
            self.assertEqual(inventory["revision"], revision)

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            config["hidden_size"] = 16
            (root / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            mutated_gates = m8b_preflight.GateBook()
            m8b_preflight._check_model(
                root,
                revision,
                self._lock(),
                mode="autodl",
                gates=mutated_gates,
            )
        self.assertFalse(mutated_gates.passed)
        self.assertIn(
            "config:hidden_size:mismatch",
            mutated_gates.results[-1].details["mismatches"],
        )

    def test_manifest_rejects_unlisted_files_and_incorrect_totals(self):
        with workspace_temp_directory() as root:
            revision = "d" * 40
            self._make_model(root, revision)
            (root / "unlisted-generation-config.json").write_text(
                "{}", encoding="utf-8"
            )
            manifest_path = root / ".agemem_model_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["file_count"] += 1
            manifest["total_size_bytes"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            gates = m8b_preflight.GateBook()
            m8b_preflight._check_model(
                root,
                revision,
                self._lock(),
                mode="autodl",
                gates=gates,
            )

        self.assertFalse(gates.passed)
        mismatches = gates.results[-1].details["mismatches"]
        self.assertIn("manifest:file_inventory", mismatches)
        self.assertIn("manifest:file_count", mismatches)
        self.assertIn("manifest:total_size_bytes", mismatches)


class GpuGateTest(unittest.TestCase):
    def test_busy_gpu_fails_even_when_total_memory_is_large(self):
        lock = {
            "gpu": {
                "minimum_count": 2,
                "minimum_memory_mib": 76000,
                "minimum_free_memory_mib": 74000,
                "require_exact_count": True,
            }
        }
        nvidia = [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "A100",
                "memory_mib": 81920,
                "free_memory_mib": 1000 if index == 0 else 80000,
            }
            for index in range(2)
        ]
        torch_cuda = {
            "torch_version": "2.6.0",
            "torch_cuda_version": "12.4",
            "cuda_available": True,
            "device_count": 2,
            "devices": [
                {
                    "index": index,
                    "uuid": f"GPU-{index}",
                    "name": "A100",
                    "memory_mib": 81920,
                }
                for index in range(2)
            ],
        }
        gates = m8b_preflight.GateBook()
        with patch.object(
            m8b_preflight, "_query_nvidia_smi", return_value=nvidia
        ), patch.object(
            m8b_preflight, "_query_torch_cuda", return_value=torch_cuda
        ):
            m8b_preflight._check_gpu(
                ROOT,
                lock,
                mode="autodl",
                gates=gates,
            )
        self.assertFalse(gates.passed)


class RuntimeGateReportTest(unittest.TestCase):
    def test_skipped_test_is_a_hard_failure(self):
        from scripts.agemem_m8b_runtime_gate import result_report

        class FakeTest:
            @staticmethod
            def id():
                return "tests.fake.RuntimeTest.test_requires_gpu"

        result = unittest.TestResult()
        result.testsRun = 1
        result.skipped = [(FakeTest(), "runtime unavailable")]
        report = result_report(result, scope="m8a")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["skip_count"], 1)


if __name__ == "__main__":
    unittest.main()
