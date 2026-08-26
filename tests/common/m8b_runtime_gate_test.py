from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import agemem_m8b_runtime_gate as runtime_gate


class RuntimeGateCountTest(unittest.TestCase):
    @staticmethod
    def _result(tests_run: int) -> unittest.TestResult:
        result = unittest.TestResult()
        result.testsRun = tests_run
        return result

    def test_exact_locked_count_can_pass(self):
        report = runtime_gate.result_report(
            self._result(7),
            scope="m8a",
            expected_tests=7,
            suite_tests=7,
        )
        self.assertEqual(report["status"], "pass")
        self.assertIs(report["count_match"], True)

    def test_suite_count_mismatch_fails(self):
        report = runtime_gate.result_report(
            self._result(6),
            scope="m8a",
            expected_tests=7,
            suite_tests=6,
        )
        self.assertEqual(report["status"], "fail")
        self.assertIs(report["count_match"], False)

    def test_executed_count_mismatch_fails(self):
        report = runtime_gate.result_report(
            self._result(6),
            scope="m8a",
            expected_tests=7,
            suite_tests=7,
        )
        self.assertEqual(report["status"], "fail")
        self.assertIs(report["count_match"], False)

    def test_lock_requires_exact_positive_scope_counts(self):
        invalid_locks = (
            {"runtime": {}},
            {"runtime": {"expected_test_counts": {"m8a": 1}}},
            {
                "runtime": {
                    "expected_test_counts": {"m8a": True, "all": 2}
                }
            },
            {
                "runtime": {
                    "expected_test_counts": {"m8a": 3, "all": 2}
                }
            },
        )
        for lock in invalid_locks:
            with self.subTest(lock=lock), patch.object(
                Path, "read_text", return_value=json.dumps(lock)
            ):
                with self.assertRaises(ValueError):
                    runtime_gate.load_expected_test_counts(Path("lock.json"))

    def test_m8b_gate_covers_manifest_runtime_and_postflight_modules(self):
        self.assertIn(
            "tests.common.m8b_model_manifest_test", runtime_gate.M8A_MODULES
        )
        self.assertIn(
            "tests.common.m8b_runtime_gate_test", runtime_gate.M8A_MODULES
        )
        self.assertIn(
            "tests.common.m8b_runtime_fail_closed_test",
            runtime_gate.M8A_MODULES,
        )
        self.assertIn(
            "tests.common.m8b_postflight_test", runtime_gate.M8A_MODULES
        )
        self.assertIn(
            "tests.common.stage1_storage_baseline_test",
            runtime_gate.M8A_MODULES,
        )
        self.assertIn(
            "tests.common.stage2_context_challenge_test",
            runtime_gate.M8A_MODULES,
        )
        self.assertIn(
            "tests.common.anti_shortcut_benchmark_test",
            runtime_gate.M8A_MODULES,
        )
        self.assertIn(
            "tests.common.anti_shortcut_stress_test",
            runtime_gate.M8A_MODULES,
        )
        self.assertEqual(
            runtime_gate.load_expected_test_counts(
                runtime_gate.DEFAULT_LOCK_PATH
            ),
            {"m8a": 141, "all": 316},
        )
        lock = json.loads(
            runtime_gate.DEFAULT_LOCK_PATH.read_text(encoding="utf-8")
        )
        assertions = {
            entry["path"]: entry["equals"]
            for entry in lock["config_assertions"]
        }
        self.assertIn(
            "buffer.trainer_input.experience_buffer.path", assertions
        )
        self.assertIs(
            assertions["buffer.trainer_input.experience_buffer.path"], None
        )


if __name__ == "__main__":
    unittest.main()
