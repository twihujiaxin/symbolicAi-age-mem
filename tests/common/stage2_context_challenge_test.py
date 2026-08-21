import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from AgeMem_code_agentscope.toy_hotpotqa import (
    AlwaysClearPolicy,
    AlwaysKeepPolicy,
    OpaqueIdControlPolicy,
    OracleSafeCompressPolicy,
    Stage2ChallengeCase,
    Stage2ChallengeDataset,
    Stage2CompressionDecision,
    evaluate_stage2_decision,
    run_stage2_challenge_benchmark,
    stage2_report_digest,
)
from AgeMem_code_agentscope.toy_hotpotqa.stage2_challenge import (
    MAX_CHALLENGE_CASES,
    MAX_CONTEXT_TOKENS,
    MAX_MESSAGES_PER_CASE,
    deterministic_token_count,
)


class Stage2ChallengeDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Stage2ChallengeDataset.from_json()

    def test_fixture_is_bounded_and_covers_each_scenario_per_split(self):
        package_fixture = (
            Path(__file__).resolve().parents[2]
            / "AgeMem_code_agentscope"
            / "toy_hotpotqa"
            / "data"
            / "stage2_context_challenges.json"
        )
        source_fixture = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "toy"
            / "stage2_context_challenges.json"
        )
        self.assertEqual(package_fixture.read_bytes(), source_fixture.read_bytes())
        self.assertEqual(len(self.dataset), 6)
        self.assertLessEqual(len(self.dataset), MAX_CHALLENGE_CASES)
        expected = {"hard_negative", "partial_relevance", "delayed_relevance"}
        for split in ("dev", "test"):
            cases = self.dataset.split(split)
            self.assertEqual({case.scenario for case in cases}, expected)
            for case in cases:
                self.assertLessEqual(len(case.messages), MAX_MESSAGES_PER_CASE)
                self.assertLessEqual(case.max_context_tokens, MAX_CONTEXT_TOKENS)
                total_tokens = sum(
                    deterministic_token_count(segment.text)
                    for segment in case.segments()
                )
                support_tokens = sum(
                    deterministic_token_count(segment.text)
                    for segment in case.segments()
                    if segment.oracle_role == "future_support"
                )
                self.assertGreater(total_tokens, case.max_context_tokens)
                self.assertLessEqual(support_tokens, case.max_context_tokens)

    def test_public_view_is_deterministic_and_hides_query_and_oracle_fields(self):
        case = self.dataset.get("stage2-dev-001")
        first = case.public_input(seed=73)
        repeated = case.public_input(seed=73)
        self.assertEqual(first, repeated)

        public = first.model_dump(mode="json")
        serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("future_query", serialized)
        self.assertNotIn("future_answer", serialized)
        self.assertNotIn("oracle_role", serialized)
        self.assertNotIn("shared_terms", serialized)
        self.assertNotIn("hard_negative", serialized)
        self.assertNotIn(case.task_id, serialized)
        self.assertNotIn(case.split, serialized)
        for private_id in (
            *(message.message_id for message in case.messages),
            *(segment.segment_id for segment in case.segments()),
        ):
            self.assertNotIn(private_id, serialized)
        self.assertNotIn(case.future_query, first.observation)

        changed_seed = case.public_input(seed=74)
        self.assertNotEqual(first.segment_ids(), changed_seed.segment_ids())
        self.assertTrue(
            all(item.startswith("segment-") for item in first.segment_ids())
        )

        orderings = {
            tuple(message.message_id for message in case.public_input(seed).messages)
            for seed in range(12)
        }
        self.assertGreater(len(orderings), 1)

    def test_hard_negative_requires_auditable_query_overlap(self):
        case = self.dataset.get("stage2-dev-001")
        raw = case.model_dump(mode="python")
        raw["future_query"] = "Which tree grows beside the southern lake?"
        with self.assertRaisesRegex(ValidationError, "shared term"):
            Stage2ChallengeCase.model_validate(raw)

        missing_answer = case.model_dump(mode="python")
        missing_answer["future_answer"] = "an answer absent from every segment"
        with self.assertRaisesRegex(ValidationError, "grounded"):
            Stage2ChallengeCase.model_validate(missing_answer)

        leaked_answer = case.model_dump(mode="python")
        leaked_answer["messages"][0]["segments"][0]["text"] += " Northstar Institute."
        with self.assertRaisesRegex(ValidationError, "distractor"):
            Stage2ChallengeCase.model_validate(leaked_answer)

    def test_partial_relevance_requires_support_and_noise_in_one_message(self):
        case = self.dataset.get("stage2-dev-002")
        raw = case.model_dump(mode="python")
        mixed = raw["messages"][0]
        for segment in mixed["segments"]:
            segment["oracle_role"] = "distractor"
        with self.assertRaisesRegex(ValidationError, "partial_relevance"):
            Stage2ChallengeCase.model_validate(raw)


class Stage2ChallengePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Stage2ChallengeDataset.from_json()

    def test_fixed_baselines_expose_the_shortcut_tradeoff(self):
        report = run_stage2_challenge_benchmark(self.dataset, seed=2026)
        self.assertEqual(report.token_counter, "unicode-lexical-v1")
        keep = report.aggregates["always_keep"]
        clear = report.aggregates["always_clear"]
        id_only = report.aggregates["opaque_id_control"]
        oracle = report.aggregates["oracle_safe_compress"]

        self.assertEqual(keep.future_support_recall, 1.0)
        self.assertEqual(keep.distractor_removal_recall, 0.0)
        self.assertEqual(keep.budget_compliance_rate, 0.0)
        self.assertEqual(keep.safe_success_rate, 0.0)

        self.assertEqual(clear.future_support_recall, 0.0)
        self.assertEqual(clear.distractor_removal_recall, 1.0)
        self.assertEqual(clear.budget_compliance_rate, 1.0)
        self.assertEqual(clear.safe_success_rate, 0.0)

        self.assertLess(id_only.future_support_recall, 1.0)
        self.assertLess(id_only.safe_success_rate, 1.0)

        self.assertEqual(oracle.future_support_recall, 1.0)
        self.assertEqual(oracle.distractor_removal_recall, 1.0)
        self.assertEqual(oracle.removal_precision, 1.0)
        self.assertEqual(oracle.budget_compliance_rate, 1.0)
        self.assertEqual(oracle.safe_success_rate, 1.0)

    def test_partial_message_is_compressed_at_segment_granularity(self):
        case = self.dataset.get("stage2-test-002")
        public_input = case.public_input(seed=9)
        decision = OracleSafeCompressPolicy().decide(
            public_input,
            oracle_case=case,
        )
        kept_text = [
            segment.text
            for message in public_input.messages
            for segment in message.segments
            if segment.segment_id in decision.kept_segment_ids
        ]
        self.assertEqual(
            kept_text,
            ["The Helios station is powered by the Nera River."],
        )
        metrics = evaluate_stage2_decision(
            case,
            decision,
            public_input=public_input,
        )
        self.assertEqual(metrics.retained_support_segments, 1)
        self.assertEqual(metrics.removed_distractor_segments, 2)
        self.assertTrue(metrics.safe_success)

    def test_non_oracle_baselines_depend_only_on_public_input(self):
        case = self.dataset.get("stage2-test-003")
        public_input = case.public_input(seed=4)
        keep = AlwaysKeepPolicy().decide(public_input)
        clear = AlwaysClearPolicy().decide(public_input)
        id_only = OpaqueIdControlPolicy().decide(public_input)

        relabelled_messages = tuple(
            message.model_copy(
                update={
                    "segments": tuple(
                        segment.model_copy(
                            update={
                                "oracle_role": (
                                    "distractor"
                                    if segment.oracle_role == "future_support"
                                    else "future_support"
                                )
                            }
                        )
                        for segment in message.segments
                    )
                }
            )
            for message in case.messages
        )
        # model_copy deliberately bypasses schema revalidation here: the point
        # is to prove that private role changes cannot alter the public view.
        relabelled = case.model_copy(update={"messages": relabelled_messages})
        relabelled_public = relabelled.public_input(seed=4)
        self.assertEqual(public_input, relabelled_public)
        self.assertEqual(keep, AlwaysKeepPolicy().decide(relabelled_public))
        self.assertEqual(clear, AlwaysClearPolicy().decide(relabelled_public))
        self.assertEqual(
            id_only,
            OpaqueIdControlPolicy().decide(relabelled_public),
        )

        class ClaimedOraclePolicy:
            name = "oracle_safe_compress"

            def decide(self, visible, *, oracle_case=None):
                self.received_oracle_case = oracle_case
                return Stage2CompressionDecision(
                    policy=self.name,
                    kept_segment_ids=visible.segment_ids(),
                )

        claimed_oracle = ClaimedOraclePolicy()
        run_stage2_challenge_benchmark(
            self.dataset,
            seed=4,
            policies=(claimed_oracle,),
        )
        self.assertIsNone(claimed_oracle.received_oracle_case)

    def test_unknown_segment_id_fails_closed(self):
        case = self.dataset.get("stage2-test-003")
        public_input = case.public_input(seed=4)
        decision = Stage2CompressionDecision(
            policy="always_keep",
            kept_segment_ids=("not-a-real-segment",),
        )
        with self.assertRaisesRegex(ValueError, "unknown segment IDs"):
            evaluate_stage2_decision(
                case,
                decision,
                public_input=public_input,
            )

    def test_report_is_exactly_repeatable(self):
        first = run_stage2_challenge_benchmark(self.dataset, seed=17)
        second = run_stage2_challenge_benchmark(self.dataset, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.dataset_digest, self.dataset.digest())
        self.assertEqual(len(first.rows), len(self.dataset) * 4)
        self.assertEqual(stage2_report_digest(first), stage2_report_digest(second))
        self.assertEqual(len(stage2_report_digest(first)), 64)

        mutated_cases = self.dataset.all()
        raw = mutated_cases[0].model_dump(mode="python")
        raw["future_answer"] = "Polar Institute"
        for message in raw["messages"]:
            for segment in message["segments"]:
                if segment["oracle_role"] == "future_support":
                    segment["text"] = segment["text"].replace(
                        "Northstar Institute",
                        "Polar Institute",
                    )
        mutated_cases[0] = Stage2ChallengeCase.model_validate(raw)
        mutated_dataset = Stage2ChallengeDataset(mutated_cases)
        mutated_report = run_stage2_challenge_benchmark(mutated_dataset, seed=17)
        self.assertNotEqual(first.dataset_digest, mutated_report.dataset_digest)
        self.assertNotEqual(
            stage2_report_digest(first),
            stage2_report_digest(mutated_report),
        )


if __name__ == "__main__":
    unittest.main()
