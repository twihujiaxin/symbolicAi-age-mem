import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from AgeMem_code_agentscope.toy_hotpotqa import (
    AntiShortcutBenchmarkReport,
    run_anti_shortcut_benchmark,
    shortcut_report_digest,
    write_anti_shortcut_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    """Use the workspace because Windows system-temp ACLs can be restrictive."""

    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"anti-shortcut-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class AntiShortcutBenchmarkTest(unittest.TestCase):
    def test_report_exposes_fixed_shortcuts_and_preserves_oracle_feasibility(self):
        report = run_anti_shortcut_benchmark(seed=7)

        self.assertTrue(report.passed)
        self.assertEqual(report.real_llm_call_count, 0)
        self.assertEqual(report.schema_version, "agemem.anti_shortcut_benchmark.v2")
        self.assertEqual(len(report.stage1_task_digest), 64)
        self.assertEqual(len(report.stage2.dataset_digest), 64)
        self.assertEqual(report.stage1_token_counter, "unicode-lexical-v1")
        self.assertEqual(report.stage2.token_counter, "unicode-lexical-v1")
        self.assertTrue(all(gate.passed for gate in report.gates))
        self.assertEqual(len(report.gates), 7)
        self.assertEqual(report.stage1["store-all"].supporting_recall, 0.5)
        self.assertEqual(report.stage1["store-all"].memory_precision, 0.5)
        self.assertFalse(report.stage1["store-all"].supporting_complete)
        self.assertEqual(report.stage1["oracle-safe-store"].supporting_recall, 1.0)
        self.assertEqual(
            report.stage2.aggregates["always_clear"].safe_success_rate,
            0.0,
        )
        self.assertLess(
            report.stage2.aggregates["opaque_id_control"].safe_success_rate,
            1.0,
        )
        self.assertEqual(
            report.stage2.aggregates["oracle_safe_compress"].safe_success_rate,
            1.0,
        )

    def test_report_and_digest_are_repeatable(self):
        first = run_anti_shortcut_benchmark(seed=7)
        second = run_anti_shortcut_benchmark(seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first.digest, shortcut_report_digest(first))

    def test_tampered_report_fails_digest_validation(self):
        report = run_anti_shortcut_benchmark(seed=7)
        payload = report.model_dump(mode="python")
        payload["stage1_task_id"] = "tampered-task-id"

        with self.assertRaisesRegex(ValidationError, "digest"):
            AntiShortcutBenchmarkReport.model_validate(payload)

        missing_counter = report.model_dump(mode="python")
        missing_counter.pop("stage1_token_counter")
        with self.assertRaises(ValidationError):
            AntiShortcutBenchmarkReport.model_validate(missing_counter)

        wrong_policy = report.model_dump(mode="python")
        wrong_policy["stage1"]["store-all"]["policy"] = "store-none"
        wrong_policy["digest"] = shortcut_report_digest(wrong_policy)
        with self.assertRaisesRegex(ValidationError, "policy"):
            AntiShortcutBenchmarkReport.model_validate(wrong_policy)

        fabricated_gate = report.model_dump(mode="python")
        fabricated_gate["gates"] = [
            {
                "name": "fabricated_pass",
                "passed": True,
                "evidence": "not derived from metrics",
            }
        ]
        fabricated_gate["passed"] = True
        fabricated_gate["digest"] = shortcut_report_digest(fabricated_gate)
        with self.assertRaisesRegex(ValidationError, "gate"):
            AntiShortcutBenchmarkReport.model_validate(fabricated_gate)

        wrong_aggregate = report.model_dump(mode="python")
        wrong_aggregate["stage2"]["aggregates"]["always_keep"]["safe_success_rate"] = (
            1.0
        )
        wrong_aggregate["digest"] = shortcut_report_digest(wrong_aggregate)
        with self.assertRaisesRegex(ValidationError, "aggregate"):
            AntiShortcutBenchmarkReport.model_validate(wrong_aggregate)

    def test_writer_is_byte_deterministic_and_contains_no_private_stage2_text(self):
        report = run_anti_shortcut_benchmark(seed=7)
        with (
            workspace_temp_directory() as first_dir,
            workspace_temp_directory() as second_dir,
        ):
            first_json, first_md = write_anti_shortcut_report(
                report,
                output_dir=first_dir,
            )
            second_json, second_md = write_anti_shortcut_report(
                report,
                output_dir=second_dir,
            )

            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            payload = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["digest"], report.digest)
            self.assertEqual(
                AntiShortcutBenchmarkReport.model_validate(payload),
                report,
            )
            markdown = first_md.read_text(encoding="utf-8")
            self.assertNotIn("Northstar Institute", markdown)
            self.assertIn("Store-All", markdown)


if __name__ == "__main__":
    unittest.main()
