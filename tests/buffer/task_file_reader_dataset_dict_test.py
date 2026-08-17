"""M8a contracts for local save-to-disk task data and the E1 smoke config."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from datasets import Dataset, DatasetDict

from trinity.common.hf_task_dataset import (
    load_task_dataset,
    select_task_rows,
)


ROOT = Path(__file__).resolve().parents[2]
E1_CONFIG = ROOT / "examples" / "agemem_hotpotqa" / "agemem_e1_dry_run.yaml"
M5_MANIFEST = ROOT / "data" / "splits" / "hotpotqa_smoke_manifest.json"


class TestSavedDatasetDictTaskReader(unittest.TestCase):
    def setUp(self):
        self.dataset_path = ROOT / "tests" / "fixtures" / "m8a_saved_dataset_dict"
        self.saved_dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "id": ["t0", "t1", "t2", "t3"],
                        "question": ["q0", "q1", "q2", "q3"],
                        "answer": ["a0", "a1", "a2", "a3"],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "id": ["v0"],
                        "question": ["vq0"],
                        "answer": ["va0"],
                    }
                ),
            }
        )
        self.load_patch = patch(
            "trinity.common.hf_task_dataset.load_from_disk",
            return_value=self.saved_dataset,
        )
        self.load_patch.start()

    def tearDown(self):
        self.load_patch.stop()

    def test_loads_requested_split_from_saved_dataset_dict(self):
        dataset = load_task_dataset(str(self.dataset_path), None, "validation")

        self.assertEqual(dataset[0]["id"], "v0")
        self.assertEqual(len(dataset), 1)

    def test_missing_split_and_subset_name_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "available splits: train, validation"):
            load_task_dataset(str(self.dataset_path), None, "test")
        with self.assertRaisesRegex(ValueError, "subset_name"):
            load_task_dataset(str(self.dataset_path), "fullwiki", "train")

    def test_non_saved_path_keeps_legacy_load_dataset_call(self):
        legacy_dataset = Dataset.from_dict({"id": ["legacy"]})
        ordinary_path = ROOT / "tests"

        with patch(
            "trinity.common.hf_task_dataset.load_dataset",
            return_value=legacy_dataset,
        ) as mocked_load:
            result = load_task_dataset(str(ordinary_path), "subset", "train")

        self.assertIs(result, legacy_dataset)
        mocked_load.assert_called_once_with(
            str(ordinary_path), name="subset", split="train"
        )

    def test_ordered_subset_preserves_manifest_order(self):
        dataset = load_task_dataset(str(self.dataset_path), None, "train")
        selected = select_task_rows(dataset, [2, 0])

        self.assertEqual(selected["id"], ["t2", "t0"])

    def test_subset_rejects_empty_duplicate_typed_and_out_of_range_indices(self):
        dataset = Dataset.from_dict({"id": ["a", "b"]})

        for indices, exception in (
            ([], ValueError),
            ([0, 0], ValueError),
            ([True], TypeError),
            ([2], IndexError),
            ([-1], IndexError),
        ):
            with self.subTest(indices=indices):
                with self.assertRaises(exception):
                    select_task_rows(dataset, indices)


class TestTaskSubsetIdentityContract(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_dict(
            {"id": ["t0", "t1", "t2"], "question": ["q0", "q1", "q2"]}
        )

    def test_subset_verifies_source_fingerprint_and_ordered_ids(self):
        selected = select_task_rows(
            self.dataset,
            [2, 0],
            expected_row_ids=["t2", "t0"],
            row_id_key="id",
            expected_dataset_fingerprint=self.dataset._fingerprint,
        )
        self.assertEqual(selected["id"], ["t2", "t0"])

    def test_identity_mismatch_and_partial_contracts_fail_closed(self):
        cases = (
            (
                {"expected_dataset_fingerprint": "wrong"},
                ValueError,
                "fingerprint mismatch",
            ),
            (
                {"expected_row_ids": ["t0", "t2"], "row_id_key": "id"},
                ValueError,
                "do not match",
            ),
            (
                {"expected_row_ids": ["t2", "t0"]},
                ValueError,
                "row_id_key",
            ),
            (
                {"expected_row_ids": ["t2", "t0"], "row_id_key": "missing"},
                ValueError,
                "no row ID column",
            ),
        )
        for kwargs, exception, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(exception, message):
                    select_task_rows(self.dataset, [2, 0], **kwargs)


class TestE1DryRunConfig(unittest.TestCase):
    def test_config_is_fixed_to_m5_smoke_and_two_gpu_dry_run_contract(self):
        config = yaml.safe_load(E1_CONFIG.read_text(encoding="utf-8"))
        manifest = json.loads(M5_MANIFEST.read_text(encoding="utf-8"))
        expected_indices = [
            selection["source_index"]
            for selection in manifest["selections"]
            if selection["benchmark_split"] == "train"
        ]
        expected_ids = [
            selection["hotpot_id"]
            for selection in manifest["selections"]
            if selection["benchmark_split"] == "train"
        ]
        taskset = config["buffer"]["explorer_input"]["taskset"]
        workflow_args = taskset["workflow_args"]
        trainer_config = config["trainer"]["trainer_config"]

        self.assertEqual(config["algorithm"]["algorithm_type"], "multi_step_grpo")
        self.assertEqual(config["algorithm"]["advantage_fn"], "step_wise_grpo")
        self.assertEqual(config["algorithm"]["repeat_times"], 2)
        self.assertGreaterEqual(
            config["explorer"]["max_repeat_times_per_runner"],
            config["algorithm"]["repeat_times"],
        )
        self.assertIsNone(config["model"]["lora_configs"][0]["path"])
        self.assertEqual(taskset["split"], "train")
        self.assertEqual(taskset["row_indices"], expected_indices)
        self.assertEqual(taskset["expected_row_ids"], expected_ids)
        self.assertEqual(taskset["row_id_key"], "id")
        self.assertEqual(
            taskset["expected_dataset_fingerprint"],
            manifest["source_fingerprints"]["train"],
        )
        self.assertEqual(len(taskset["row_indices"]), 6)
        self.assertEqual(len(set(taskset["row_indices"])), 6)
        self.assertEqual(
            config["buffer"]["batch_size"] * config["buffer"]["total_steps"],
            len(taskset["row_indices"]),
        )
        self.assertEqual(config["buffer"]["train_batch_size"], 4)
        self.assertEqual(config["trainer"]["total_steps"], 1)

        self.assertEqual(config["cluster"]["node_num"], 1)
        self.assertEqual(config["cluster"]["gpu_per_node"], 2)
        self.assertEqual(config["explorer"]["rollout_model"]["engine_num"], 1)
        self.assertEqual(config["explorer"]["rollout_model"]["tensor_parallel_size"], 1)
        self.assertEqual(
            trainer_config["actor_rollout_ref"]["actor"][
                "ulysses_sequence_parallel_size"
            ],
            1,
        )
        self.assertEqual(
            trainer_config["actor_rollout_ref"]["ref"][
                "ulysses_sequence_parallel_size"
            ],
            1,
        )

        self.assertEqual(workflow_args["reward_profile"], "terminal_only")
        self.assertEqual(workflow_args["terminal_reward_metric"], "hotpotqa_official")
        self.assertIs(workflow_args["milestone_reward_enabled"], False)
        self.assertEqual(workflow_args["stage2_distractor_source"], "fixed")
        self.assertEqual(
            workflow_args["auxiliary_provider"],
            {
                "schema_version": "agemem.auxiliary_provider.v1",
                "provider": "dashscope",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "embedding_model": "text-embedding-v4",
                "embedding_dimensions": 256,
                "chat_model": "qwen-max",
                "usage_tracking": True,
            },
        )
        self.assertIs(config["continue_from_checkpoint"], False)


if __name__ == "__main__":
    unittest.main()
