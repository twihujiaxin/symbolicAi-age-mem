from __future__ import annotations

import io
import shutil
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pydantic import ValidationError

from AgeMem_code_agentscope.toy_hotpotqa import (
    AntiShortcutStressReport,
    CounterfactualDataset,
    CounterfactualPair,
    run_anti_shortcut_benchmark,
    run_anti_shortcut_stress,
    run_stage2_counterfactual_stress,
    stress_report_digest,
    write_anti_shortcut_stress_report,
)
from AgeMem_code_agentscope.toy_hotpotqa.shortcut_stress import (
    STAGE2_STRESS_POLICIES,
    _blind_stage2_input,
    lexical_token_counter,
)
from scripts import agemem_anti_shortcut_stress as stress_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class AntiShortcutStressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_anti_shortcut_stress()

    def test_default_protocol_covers_tasks_seeds_budgets_and_strong_baselines(self):
        report = self.report

        self.assertTrue(report.passed)
        self.assertEqual(report.schema_version, "agemem.anti_shortcut_stress.v1")
        self.assertEqual(report.real_llm_call_count, 0)
        self.assertEqual(report.stage1.task_count, 16)
        self.assertEqual(report.stage1.seeds, tuple(range(50)))
        self.assertEqual(report.stage1.budgets, (12, 20, 28))
        self.assertEqual(report.stage1.arm_count_per_policy, 16 * 50 * 3)
        self.assertEqual(
            set(report.stage1.by_split),
            {"train", "dev", "test"},
        )
        self.assertEqual(
            set(report.stage1.by_scenario),
            {"distractor", "duplicate", "stale_fact"},
        )
        self.assertEqual(
            min(item.coverage_rate for item in report.stage1.permutation_coverage),
            1.0,
        )
        self.assertIn("entity_chain", report.stage1.aggregates)
        self.assertLess(
            report.stage1.aggregates["store_all"].support_recall,
            report.stage1.aggregates["oracle_support"].support_recall,
        )

    def test_counterfactual_protocol_has_exact_ceiling_and_one_shared_decision(self):
        report = self.report

        self.assertEqual(report.stage2.pair_count, 6)
        self.assertEqual(report.stage2.variant_count, 12)
        self.assertEqual(report.stage2.budgets, (19,))
        self.assertEqual(report.stage2.arm_count_per_policy, 12 * 50)
        self.assertEqual(report.stage2.public_input_identity_rate, 1.0)
        self.assertEqual(report.stage2.public_safe_success_ceiling, 0.5)
        self.assertEqual(report.stage2.max_target_token_gap, 0)
        self.assertEqual(report.stage2.max_target_capitalized_gap, 0)
        self.assertEqual(
            report.stage2.query_blind_decision_count,
            (len(STAGE2_STRESS_POLICIES) - 1) * 6 * 50,
        )
        self.assertEqual(report.stage2.hindsight_decision_count, 12 * 50)
        self.assertEqual(
            report.stage2.aggregates["pair_blind_oracle"].safe_success_rate,
            0.5,
        )
        self.assertEqual(
            report.stage2.aggregates["oracle_future"].safe_success_rate,
            1.0,
        )
        self.assertAlmostEqual(
            report.stage2.aggregates["random_hash"].safe_success_rate,
            1 / 3,
            delta=0.05,
        )
        public_names = set(STAGE2_STRESS_POLICIES) - {
            "pair_blind_oracle",
            "oracle_future",
        }
        self.assertLessEqual(
            max(
                report.stage2.aggregates[name].safe_success_rate
                for name in public_names
            ),
            0.5,
        )

    def test_public_counterfactual_view_omits_seed_query_labels_and_private_keys(self):
        dataset = CounterfactualDataset.from_json()
        pair = dataset.all()[0]
        counter, _ = lexical_token_counter()
        public_input, _private = _blind_stage2_input(pair, 17, counter)
        payload = public_input.model_dump(mode="json")
        serialized = public_input.model_dump_json()

        self.assertEqual(set(payload), {"max_context_tokens", "segments"})
        for forbidden in (
            "seed",
            "pair_id",
            "split",
            "scenario",
            "variant_id",
            "future_query",
            "future_answer",
            "support_segment_keys",
        ):
            self.assertNotIn(forbidden, serialized)
        for future in pair.futures:
            self.assertNotIn(future.future_query, serialized)
        public_handles = {segment["segment_handle"] for segment in payload["segments"]}
        self.assertTrue(
            public_handles.isdisjoint(
                {segment.segment_key for segment in pair.segments}
            )
        )

    def test_counterfactual_fixture_is_paired_and_package_copy_is_byte_identical(self):
        source = REPOSITORY_ROOT / "data" / "toy" / "stage2_counterfactual_pairs.json"
        packaged = (
            REPOSITORY_ROOT
            / "AgeMem_code_agentscope"
            / "toy_hotpotqa"
            / "data"
            / "stage2_counterfactual_pairs.json"
        )
        self.assertEqual(source.read_bytes(), packaged.read_bytes())
        dataset = CounterfactualDataset.from_json(source)
        for pair in dataset.all():
            first, second = pair.futures
            self.assertNotEqual(first.future_query, second.future_query)
            self.assertTrue(
                set(first.support_segment_keys).isdisjoint(second.support_segment_keys)
            )
            self.assertEqual(pair.max_context_tokens, 19)

        first_payload = dataset.all()[0].model_dump(mode="python")
        first_payload["futures"][1]["support_segment_keys"] = first_payload["futures"][
            0
        ]["support_segment_keys"]
        first_payload["futures"][1]["future_answer"] = first_payload["futures"][0][
            "future_answer"
        ]
        with self.assertRaisesRegex(ValidationError, "disjoint"):
            CounterfactualPair.model_validate(first_payload)

        mutated = [pair.model_dump(mode="python") for pair in dataset.all()]
        mutated[0]["max_context_tokens"] = 64
        union_fits = CounterfactualDataset(
            [CounterfactualPair.model_validate(payload) for payload in mutated]
        )
        with self.assertRaisesRegex(ValueError, "jointly retainable"):
            run_stage2_counterfactual_stress(dataset=union_fits)

    def test_counterfactual_budget_fits_frozen_qwen_token_count_snapshot(self):
        dataset = CounterfactualDataset.from_json()
        qwen_token_counts = {
            "cf-dev-entity-001": (17, 19, 17),
            "cf-dev-length-001": (17, 17, 17),
            "cf-dev-style-001": (11, 11, 11),
            "cf-test-entity-001": (16, 15, 15),
            "cf-test-length-001": (14, 14, 14),
            "cf-test-style-001": (11, 11, 11),
        }
        counts_by_text = {}
        for pair in dataset.all():
            observed = qwen_token_counts[pair.pair_id]
            self.assertEqual(len(observed), len(pair.segments))
            counts_by_text.update(
                {
                    segment.text: token_count
                    for segment, token_count in zip(pair.segments, observed)
                }
            )

        report = run_stage2_counterfactual_stress(
            dataset=dataset,
            seeds=(0,),
            token_counter=counts_by_text.__getitem__,
        )

        self.assertEqual(report.budgets, (19,))
        self.assertEqual(report.max_target_token_gap, 2)
        self.assertEqual(report.public_safe_success_ceiling, 0.5)
        self.assertEqual(
            report.aggregates["oracle_future"].safe_success_rate,
            1.0,
        )

    def test_report_digest_rejects_tampering_and_is_repeatable(self):
        report = self.report
        self.assertEqual(report.digest, stress_report_digest(report))
        repeated = run_anti_shortcut_stress()
        self.assertEqual(report, repeated)

        payload = report.model_dump(mode="python")
        payload["stage2"]["aggregates"]["random_hash"]["safe_success_rate"] = 1.0
        with self.assertRaisesRegex(ValidationError, "gates|digest"):
            AntiShortcutStressReport.model_validate(payload)

        extra = report.model_dump(mode="python")
        extra["untracked"] = True
        with self.assertRaises(ValidationError):
            AntiShortcutStressReport.model_validate(extra)

    def test_writer_is_deterministic_and_does_not_copy_private_fixture_text(self):
        temp_root = REPOSITORY_ROOT / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        first = temp_root / f"stress-first-{uuid.uuid4().hex}"
        second = temp_root / f"stress-second-{uuid.uuid4().hex}"
        try:
            first_json, first_md = write_anti_shortcut_stress_report(
                self.report,
                output_dir=first,
            )
            second_json, second_md = write_anti_shortcut_stress_report(
                self.report,
                output_dir=second,
            )
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            markdown = first_md.read_text(encoding="utf-8")
            self.assertNotIn("Northstar Lab assigned archive code", markdown)
            self.assertIn("paired counterfactual", markdown.casefold())
        finally:
            shutil.rmtree(first, ignore_errors=True)
            shutil.rmtree(second, ignore_errors=True)

    def test_cli_no_write_and_verify_existing(self):
        temp_root = REPOSITORY_ROOT / "tmp" / f"stress-cli-{uuid.uuid4().hex}"
        try:
            with redirect_stdout(io.StringIO()):
                code = stress_cli.main(
                    [
                        "--no-write",
                        "--output-dir",
                        str(temp_root),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(temp_root.exists())

            with (
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                stress_cli.main(
                    [
                        "--no-write",
                        "--tokenizer-path",
                        str(REPOSITORY_ROOT / "missing-tokenizer"),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(temp_root.exists())

            write_anti_shortcut_stress_report(self.report, output_dir=temp_root)
            with redirect_stdout(io.StringIO()):
                code = stress_cli.main(
                    [
                        "--verify-existing",
                        "--output-dir",
                        str(temp_root),
                    ]
                )
            self.assertEqual(code, 0)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_original_v2_canary_digest_is_unchanged(self):
        report = run_anti_shortcut_benchmark(seed=7)
        self.assertTrue(report.passed)
        self.assertEqual(
            report.digest,
            "b5ced8e688194d3d9e7cb3a6b4bd8d256d7cc38610fcb56a1d8c37987a7b952c",
        )


if __name__ == "__main__":
    unittest.main()
