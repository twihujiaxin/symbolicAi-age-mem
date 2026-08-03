"""Tests for strict M6 extraction schemas, adapters, and group cache."""

from __future__ import annotations

import json
import math
import unittest
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from AgeMem_code_agentscope.memory_extraction import (
    APRecord,
    ActionBinding,
    EvidenceSpan,
    ExtractionCacheError,
    ExtractionRequest,
    GroupExtractionCache,
    LLMTripleExtractor,
    MockTripleExtractor,
    RelevanceDecision,
    TripleCandidate,
    TripleExtractor,
)
from AgeMem_code_agentscope.memory_extraction.models import (
    EXTRACTOR_OUTPUT_SCHEMA_VERSION,
    text_digest,
)


OBSERVATION = "Alice lives in Paris. Bob lives in Rome."


def valid_payload():
    return {
        "schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION,
        "triples": [
            {
                "subject": "Alice",
                "category": "location",
                "value": "Paris",
                "confidence": 0.9,
                "evidence": [
                    {
                        "source": "observation",
                        "text": "Alice lives in Paris.",
                        "start": 0,
                        "end": 21,
                    }
                ],
            }
        ],
    }


def request(*, rollout_id="rollout-gold", task_id="task-1", split_id="dev"):
    return ExtractionRequest(
        task_id=task_id,
        split_id=split_id,
        rollout_id=rollout_id,
        group_id="task-1-policy-group",
        stage_id=1,
        observation=OBSERVATION,
        question="Where does Alice live?",
        known_subjects=("Alice", "Bob"),
        allowed_categories=("location", "status"),
    )


def binding(*, rollout_id="rollout-gold", action_id="action-1", timestep=0):
    return ActionBinding(
        task_id="task-1",
        rollout_id=rollout_id,
        stage_id=1,
        timestep=timestep,
        action_id=action_id,
        assistant_turn_id=timestep,
        action_index_in_turn=0,
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.prompts = []

    def complete(self, *, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


class M6ExtractionSchemaTest(unittest.TestCase):
    def test_strict_frozen_models_and_finite_confidence(self):
        source = "Alice lives in Paris."
        evidence = EvidenceSpan.from_source(
            source="observation",
            source_text=source,
            start=0,
            end=len(source),
        )
        evidence.validate_against(source)
        candidate = TripleCandidate.create(
            subject="Alice",
            category="location",
            value="Paris",
            confidence=1.0,
            evidence=(evidence,),
            extractor_version="mock-v1",
            extractor_kind="mock",
            model_version="fixture-v1",
        )
        self.assertNotIn("action_id", type(candidate).model_fields)
        self.assertNotIn("role", type(candidate).model_fields)
        with self.assertRaises(ValidationError):
            candidate.subject = "Bob"
        with self.assertRaises(ValidationError):
            ExtractionRequest.model_validate(
                {**request().model_dump(mode="json"), "unexpected": True}
            )
        for invalid in (math.nan, math.inf, -0.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TripleCandidate.create(
                    subject="Alice",
                    category="location",
                    value="Paris",
                    confidence=invalid,
                    evidence=(evidence,),
                    extractor_version="mock-v1",
                    extractor_kind="mock",
                    model_version="fixture-v1",
                )

    def test_unknown_subject_category_and_bad_span_are_quarantined(self):
        payload = valid_payload()
        payload["triples"].extend(
            [
                {
                    **payload["triples"][0],
                    "subject": "Mallory",
                },
                {
                    **payload["triples"][0],
                    "category": "unsupported-category",
                },
                {
                    **payload["triples"][0],
                    "evidence": [
                        {
                            "source": "observation",
                            "text": "Paris",
                            "start": 0,
                            "end": 5,
                        }
                    ],
                },
            ]
        )
        extractor = MockTripleExtractor({OBSERVATION: payload})
        result = extractor.extract(request())
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            tuple(item.reason for item in result.diagnostics.quarantine),
            ("unknown_subject", "unknown_category", "invalid_evidence"),
        )
        self.assertEqual(result.diagnostics.accepted_count, 1)
        self.assertEqual(result.diagnostics.quarantined_count, 3)

    def test_materialization_rechecks_exact_source_digest_and_slice(self):
        source = "Alice lives in Paris."
        forged = EvidenceSpan(
            source="observation",
            source_digest=text_digest(source),
            text="XXXXX",
            start=0,
            end=5,
        )
        candidate = TripleCandidate.create(
            subject="Alice",
            category="location",
            value="Paris",
            confidence=0.8,
            evidence=(forged,),
            extractor_version="mock-v1",
            extractor_kind="mock",
            model_version="fixture-v1",
        )
        cache = GroupExtractionCache()
        with self.assertRaises(ExtractionCacheError):
            cache.materialize(request(), (candidate,), binding())

    def test_action_bound_relevance_and_ap_have_explicit_provenance(self):
        extractor = MockTripleExtractor({OBSERVATION: valid_payload()})
        materialized = GroupExtractionCache().get_or_extract_and_materialize(
            request(), extractor, binding()
        )
        record = materialized.records[0]
        self.assertEqual(record.action_id, "action-1")
        decision = RelevanceDecision.create(
            record,
            role="relevant",
            confidence=0.75,
            decision_version="relevance-v1",
        )
        self.assertEqual(decision.action_id, record.action_id)
        self.assertEqual(decision.triple_id, record.triple_id)
        ap = APRecord.create(
            task_id=record.task_id,
            rollout_id=record.rollout_id,
            stage_id=record.stage_id,
            timestep=record.timestep,
            action_id=record.action_id,
            proposition="observed_supporting_fact",
            confidence=record.confidence,
            evidence_triple_ids=(record.triple_id,),
            grounder_version="grounder-v1",
        )
        self.assertIn(record.action_id, ap.evidence_action_ids)
        with self.assertRaises(ValidationError):
            APRecord.create(
                task_id=record.task_id,
                rollout_id=record.rollout_id,
                stage_id=record.stage_id,
                timestep=record.timestep,
                action_id=record.action_id,
                proposition="observed_supporting_fact",
                confidence=1.0,
                evidence_action_ids=("different-action",),
                grounder_version="grounder-v1",
            )


class M6ExtractorAdapterTest(unittest.TestCase):
    def make_llm(self, client, *, model_version="fake-model-v1"):
        return LLMTripleExtractor(
            client,
            extractor_version="llm-adapter-v1",
            model_version=model_version,
            prompt_version="prompt-v1",
        )

    def test_mock_satisfies_protocol_and_is_deterministic(self):
        extractor = MockTripleExtractor({OBSERVATION: valid_payload()})
        self.assertIsInstance(extractor, TripleExtractor)
        first = extractor.extract(request())
        second = extractor.extract(request())
        self.assertEqual(first, second)
        self.assertEqual(extractor.call_count, 2)

    def test_injected_llm_client_accepts_only_strict_single_object_json(self):
        client = FakeClient(json.dumps(valid_payload()))
        extractor = self.make_llm(client)
        result = extractor.extract(request())
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(client.calls, 1)
        self.assertIn("Return exactly one JSON object", client.prompts[0])

        invalid_outputs = (
            json.dumps(valid_payload()) + " trailing prose",
            "[]",
            '{"schema_version":"agemem.extractor_output.v1","triples":NaN}',
        )
        for raw in invalid_outputs:
            with self.subTest(raw=raw):
                failed = self.make_llm(FakeClient(raw)).extract(request())
                self.assertEqual(failed.candidates, ())
                self.assertEqual(
                    failed.diagnostics.quarantine[0].reason, "invalid_json"
                )

    def test_llm_schema_and_client_errors_fail_closed(self):
        payload = valid_payload()
        payload["unexpected"] = True
        invalid_schema = self.make_llm(FakeClient(json.dumps(payload))).extract(
            request()
        )
        self.assertEqual(invalid_schema.candidates, ())
        self.assertEqual(
            invalid_schema.diagnostics.quarantine[0].reason, "invalid_schema"
        )

        client_error = self.make_llm(
            FakeClient(error=RuntimeError("secret service detail"))
        ).extract(request())
        self.assertEqual(client_error.candidates, ())
        diagnostic = client_error.diagnostics.quarantine[0]
        self.assertEqual(diagnostic.reason, "extractor_error")
        self.assertNotIn("secret service detail", diagnostic.message)
        with self.assertRaises(ValueError):
            LLMTripleExtractor(
                None,
                extractor_version="v1",
                model_version="m1",
                prompt_version="p1",
            )


class M6GroupCacheTest(unittest.TestCase):
    def test_candidate_reuse_across_policy_rollouts_rebinds_provenance(self):
        extractor = MockTripleExtractor({OBSERVATION: valid_payload()})
        cache = GroupExtractionCache()
        first = cache.get_or_extract_and_materialize(
            request(rollout_id="rollout-gold"),
            extractor,
            binding(rollout_id="rollout-gold", action_id="action-gold"),
        )
        second = cache.get_or_extract_and_materialize(
            request(rollout_id="rollout-error"),
            extractor,
            binding(rollout_id="rollout-error", action_id="action-error"),
        )
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.cache_key_digest, second.cache_key_digest)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(
            first.candidates[0].candidate_id, second.candidates[0].candidate_id
        )
        self.assertNotEqual(first.records[0].triple_id, second.records[0].triple_id)
        self.assertEqual(second.records[0].rollout_id, "rollout-error")
        self.assertEqual(second.records[0].action_id, "action-error")

    def test_cache_isolated_by_task_split_group_stage_and_versions(self):
        cache = GroupExtractionCache()
        extractor = MockTripleExtractor({OBSERVATION: valid_payload()})
        self.assertFalse(cache.get_or_extract(request(), extractor).cache_hit)
        variants = (
            request(task_id="task-2").model_copy(
                update={"group_id": "task-2-policy-group"}
            ),
            request(split_id="test"),
            request().model_copy(update={"group_id": "different-group"}),
            request().model_copy(update={"stage_id": 2}),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(cache.get_or_extract(variant, extractor).cache_hit)
        versioned = MockTripleExtractor(
            {OBSERVATION: valid_payload()},
            extractor_version="mock-v2",
            model_version="fixture-v2",
            prompt_version="prompt-v2",
        )
        self.assertFalse(cache.get_or_extract(request(), versioned).cache_hit)
        self.assertEqual(cache.size, 6)
        self.assertEqual(cache.invalidate(model_version="fixture-v2"), 1)
        self.assertEqual(cache.size, 5)

    def test_concurrent_identical_lookups_extract_once(self):
        extractor = MockTripleExtractor({OBSERVATION: valid_payload()})
        cache = GroupExtractionCache()

        def lookup(index):
            return cache.get_or_extract_and_materialize(
                request(),
                extractor,
                binding(action_id=f"action-{index}", timestep=index),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(lookup, range(16)))
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 15)
        self.assertEqual(len({output.records[0].triple_id for output in outputs}), 16)
        self.assertEqual(
            {output.records[0].action_id for output in outputs},
            {f"action-{index}" for index in range(16)},
        )

    def test_quarantine_diagnostics_are_deterministic_on_cache_hit(self):
        payload = valid_payload()
        payload["triples"].append({**payload["triples"][0], "subject": "Unknown"})
        extractor = MockTripleExtractor({OBSERVATION: payload})
        cache = GroupExtractionCache()
        first = cache.get_or_extract(request(), extractor)
        second = cache.get_or_extract(request(), extractor)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.result, second.result)
        self.assertEqual(
            second.result.diagnostics.quarantine[0].reason, "unknown_subject"
        )


if __name__ == "__main__":
    unittest.main()
