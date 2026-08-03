"""Tests for M6 manual-annotation integrity and offline evaluation metrics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from AgeMem_code_agentscope.hotpotqa_benchmark.adapter import stable_fact_id
from AgeMem_code_agentscope.memory_extraction.annotations import (
    MANUAL_TRIPLES_SCHEMA_VERSION,
    SEMANTIC_TARGETS_SCHEMA_VERSION,
    AnnotationCorpus,
    ManualTriple,
    ManualTripleCorpus,
    SemanticTargetCorpus,
    default_manual_triples_path,
    default_semantic_targets_path,
    load_annotation_corpus,
)
from AgeMem_code_agentscope.memory_extraction.metrics import (
    APEvaluationRecord,
    AcceptanceDecision,
    RewardActionValue,
    TripleEvaluationRecord,
    normalize_exact_text,
    score_acceptance,
    score_aps,
    score_reward_propagation,
    score_triples,
)


class _FakeAdapter:
    def __init__(self, rows):
        self.rows = rows

    def row(self, split, index):
        return self.rows[(split, index)]


def _tiny_corpus(sentence="Alpha is in Beta.", *, fact_id=None):
    hotpot_id = "abc123"
    title = "Alpha"
    expected_fact_id = stable_fact_id(hotpot_id, title, 0, sentence)
    actual_fact_id = fact_id or expected_fact_id
    manual = ManualTripleCorpus.model_validate(
        {
            "schema_version": MANUAL_TRIPLES_SCHEMA_VERSION,
            "annotation_method": "human_manual",
            "guidelines_version": "agemem.m6.triple_guidelines.v1",
            "normalization": (
                "NFKC + casefold + collapsed whitespace for exact-set scoring"
            ),
            "records": [
                {
                    "annotation_id": f"m6-{actual_fact_id}",
                    "hotpot_id": hotpot_id,
                    "benchmark_split": "dev",
                    "source_split": "validation",
                    "source_index": 0,
                    "fact_id": actual_fact_id,
                    "title": title,
                    "sent_id": 0,
                    "text_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
                    "triples": [
                        {"subject": "Alpha", "category": "located_in", "value": "Beta"}
                    ],
                }
            ],
        }
    )
    # Build a valid two-record corpus so full-cover validation remains active.
    second_sentence = "Noise is elsewhere."
    second_id = stable_fact_id(hotpot_id, "Noise", 0, second_sentence)
    records = list(manual.model_dump(mode="python")["records"])
    records.append(
        {
            "annotation_id": f"m6-{second_id}",
            "hotpot_id": hotpot_id,
            "benchmark_split": "dev",
            "source_split": "validation",
            "source_index": 0,
            "fact_id": second_id,
            "title": "Noise",
            "sent_id": 0,
            "text_sha256": hashlib.sha256(second_sentence.encode()).hexdigest(),
            "triples": [
                {"subject": "Noise", "category": "located_in", "value": "elsewhere"}
            ],
        }
    )
    manual = ManualTripleCorpus(
        schema_version=MANUAL_TRIPLES_SCHEMA_VERSION,
        annotation_method="human_manual",
        guidelines_version="agemem.m6.triple_guidelines.v1",
        normalization="NFKC + casefold + collapsed whitespace for exact-set scoring",
        records=tuple(records),
    )
    targets = SemanticTargetCorpus.model_validate(
        {
            "schema_version": SEMANTIC_TARGETS_SCHEMA_VERSION,
            "annotation_method": "human_oracle_target",
            "usage": "evaluation_only_not_extractor_input",
            "tasks": [
                {
                    "hotpot_id": hotpot_id,
                    "relevant_fact_ids": [actual_fact_id],
                    "irrelevant_fact_ids": [second_id],
                }
            ],
        }
    )
    row = SimpleNamespace(
        id=hotpot_id,
        supporting_facts=SimpleNamespace(pairs=lambda: ((title, 0),)),
        context=SimpleNamespace(
            title=(title, "Noise"),
            sentences=((sentence,), (second_sentence,)),
        ),
    )
    return AnnotationCorpus(manual=manual, targets=targets), _FakeAdapter(
        {("validation", 0): row}
    )


def _triple(evidence_id, subject, category, value):
    return TripleEvaluationRecord(
        evidence_id=evidence_id,
        subject=subject,
        category=category,
        value=value,
    )


def _ap(action_id, proposition):
    return APEvaluationRecord(action_id=action_id, proposition=proposition)


def _reward(
    action_id,
    timestep,
    total,
    milestone,
    violation,
    *,
    rollout_id="rollout-1",
):
    return RewardActionValue(
        action_id=action_id,
        task_id="task-1",
        rollout_id=rollout_id,
        timestep=timestep,
        total=total,
        milestone=milestone,
        violation=violation,
    )


class AnnotationCorpusTest(unittest.TestCase):
    def test_checked_in_release_has_fixed_counts_and_no_source_text(self):
        self.assertTrue(default_manual_triples_path().is_file())
        self.assertTrue(default_semantic_targets_path().is_file())
        corpus = load_annotation_corpus()
        self.assertEqual(len(corpus.manual.records), 34)
        self.assertEqual(sum(len(row.triples) for row in corpus.manual.records), 37)
        self.assertEqual(len(corpus.relevant_fact_ids), 24)
        self.assertEqual(len(corpus.irrelevant_fact_ids), 10)
        dumped = corpus.model_dump(mode="json")
        self.assertFalse(
            any("sentence" in record for record in dumped["manual"]["records"])
        )

    def test_explicit_fake_adapter_validates_pointer_hash_and_stable_id(self):
        corpus, adapter = _tiny_corpus()
        summary = corpus.validate_against_adapter(adapter)
        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.source_rows_checked, 1)
        self.assertNotIn("sentence", summary.model_dump(mode="json"))

    def test_annotation_poison_and_full_cover_are_rejected(self):
        with self.assertRaises(ValidationError):
            ManualTriple.model_validate(
                {
                    "subject": "Alpha",
                    "category": "located_in",
                    "value": "Beta",
                    "poison": True,
                }
            )

        manual_data = json.loads(
            Path(default_manual_triples_path()).read_text(encoding="utf-8")
        )
        target_data = json.loads(
            Path(default_semantic_targets_path()).read_text(encoding="utf-8")
        )
        target_data = copy.deepcopy(target_data)
        target_data["tasks"][0]["relevant_fact_ids"].pop()
        manual = ManualTripleCorpus.model_validate(manual_data)
        targets = SemanticTargetCorpus.model_validate(target_data)
        with self.assertRaises(ValidationError):
            AnnotationCorpus(manual=manual, targets=targets)

    def test_source_digest_and_stable_fact_id_mismatch_are_rejected(self):
        corpus, adapter = _tiny_corpus()
        poisoned_row = SimpleNamespace(
            id="abc123",
            supporting_facts=SimpleNamespace(pairs=lambda: (("Alpha", 0),)),
            context=SimpleNamespace(
                title=("Alpha", "Noise"),
                sentences=(("Alpha is NOT in Beta.",), ("Noise is elsewhere.",)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            corpus.validate_against_adapter(
                _FakeAdapter({("validation", 0): poisoned_row})
            )

        wrong_id = "hp-abc123-1111111111111111"
        wrong_corpus, correct_adapter = _tiny_corpus(fact_id=wrong_id)
        with self.assertRaisesRegex(ValueError, "stable_fact_id mismatch"):
            wrong_corpus.validate_against_adapter(correct_adapter)


class ExactSetMetricTest(unittest.TestCase):
    def test_normalized_perfect_triple_and_ap_scores(self):
        gold = [_triple("f1", "Café", "Born In", "New   York")]
        predicted = [_triple("f1", "ＣＡＦÉ", "born in", "new york")]
        triple_metrics = score_triples(gold, predicted)
        self.assertEqual(triple_metrics.micro.f1, 1.0)
        self.assertEqual(triple_metrics.macro.f1, 1.0)
        self.assertEqual(normalize_exact_text("  A\t B  "), "a b")

        ap_metrics = score_aps(
            [_ap("action-1", "Relevant Fact Added")],
            [_ap("action-1", " relevant   fact added ")],
        )
        self.assertEqual(ap_metrics.micro.f1, 1.0)

    def test_triple_and_ap_false_positive_false_negative(self):
        gold = [
            _triple("f1", "A", "kind", "one"),
            _triple("f2", "B", "kind", "two"),
        ]
        predicted = [
            _triple("f1", "A", "kind", "one"),
            _triple("f1", "A", "kind", "wrong"),
        ]
        metrics = score_triples(gold, predicted)
        self.assertEqual(
            (
                metrics.micro.true_positive,
                metrics.micro.false_positive,
                metrics.micro.false_negative,
            ),
            (1, 1, 1),
        )
        self.assertEqual(metrics.micro.f1, 0.5)
        self.assertAlmostEqual(metrics.macro.precision, 0.75)
        self.assertAlmostEqual(metrics.macro.recall, 0.5)
        self.assertAlmostEqual(metrics.macro.f1, 1.0 / 3.0)

        ap_metrics = score_aps(
            [_ap("a1", "p"), _ap("a2", "q")],
            [_ap("a1", "p"), _ap("a1", "wrong")],
        )
        self.assertEqual(ap_metrics.micro.f1, 0.5)

    def test_duplicate_normalized_keys_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate normalized triple"):
            score_triples(
                [
                    _triple("f1", "A", "kind", "one"),
                    _triple("f1", " a ", "KIND", "ONE"),
                ],
                [],
            )
        with self.assertRaisesRegex(ValueError, "duplicate normalized AP"):
            score_aps([_ap("a1", "P"), _ap("a1", " p ")], [])


class AcceptanceAndRewardMetricTest(unittest.TestCase):
    def test_acceptance_confusion_and_zero_denominators(self):
        metrics = score_acceptance(
            [
                AcceptanceDecision(action_id="a", accepted=True),
                AcceptanceDecision(action_id="b", accepted=False),
            ],
            [
                AcceptanceDecision(action_id="a", accepted=False),
                AcceptanceDecision(action_id="b", accepted=True),
            ],
        )
        self.assertEqual(metrics.false_accept_numerator, 1)
        self.assertEqual(metrics.false_accept_denominator, 1)
        self.assertEqual(metrics.false_reject_numerator, 1)
        self.assertEqual(metrics.false_reject_denominator, 1)
        self.assertEqual(metrics.false_accept_rate, 1.0)
        self.assertEqual(metrics.false_reject_rate, 1.0)

        no_negative = score_acceptance(
            [AcceptanceDecision(action_id="a", accepted=True)],
            [AcceptanceDecision(action_id="a", accepted=True)],
        )
        self.assertIsNone(no_negative.false_accept_rate)
        self.assertEqual(no_negative.false_reject_rate, 0.0)
        empty = score_acceptance([], [])
        self.assertIsNone(empty.false_accept_rate)
        self.assertIsNone(empty.false_reject_rate)

    def test_acceptance_duplicate_and_join_mismatch_fail(self):
        decision = AcceptanceDecision(action_id="a", accepted=True)
        with self.assertRaisesRegex(ValueError, "duplicate Oracle"):
            score_acceptance([decision, decision], [decision])
        with self.assertRaisesRegex(ValueError, "sets do not match"):
            score_acceptance(
                [decision], [AcceptanceDecision(action_id="b", accepted=True)]
            )

    def test_reward_error_propagation_and_first_divergence(self):
        oracle = [
            _reward("a0", 0, 1.0, 0.5, 0.0),
            _reward("a1", 1, 2.0, 1.0, -0.1),
        ]
        extracted = [
            _reward("a0", 0, 1.0, 0.5, 0.0),
            _reward("a1", 1, 3.0, 0.5, -0.2),
        ]
        metrics = score_reward_propagation(oracle, extracted)
        self.assertEqual(metrics.action_count, 2)
        self.assertAlmostEqual(metrics.action_total.mae, 0.5)
        self.assertAlmostEqual(metrics.action_total.rmse, math.sqrt(0.5))
        self.assertAlmostEqual(metrics.action_total.bias, 0.5)
        self.assertEqual(metrics.action_total.max_abs, 1.0)
        self.assertAlmostEqual(metrics.action_milestone.mae, 0.25)
        self.assertAlmostEqual(metrics.action_violation.mae, 0.05)
        self.assertEqual(metrics.trajectory_signed_error_total, 1.0)
        self.assertEqual(metrics.trajectory_absolute_error_total, 1.0)
        self.assertIsNotNone(metrics.first_divergence)
        self.assertEqual(metrics.first_divergence.action_id, "a1")

        perfect = score_reward_propagation(oracle, oracle)
        self.assertEqual(perfect.action_total.mae, 0.0)
        self.assertIsNone(perfect.first_divergence)

    def test_reward_duplicate_join_coordinate_and_nan_fail_closed(self):
        row = _reward("a", 0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "duplicate Oracle"):
            score_reward_propagation([row, row], [row])
        with self.assertRaisesRegex(ValueError, "sets do not match"):
            score_reward_propagation([row], [_reward("b", 0, 0.0, 0.0, 0.0)])
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            score_reward_propagation([row], [_reward("a", 1, 0.0, 0.0, 0.0)])
        with self.assertRaises(ValidationError):
            _reward("nan", 0, float("nan"), 0.0, 0.0)
        with self.assertRaises(ValueError):
            score_reward_propagation([row], [row], divergence_tolerance=float("nan"))


if __name__ == "__main__":
    unittest.main()
