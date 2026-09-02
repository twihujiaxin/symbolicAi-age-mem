"""Focused schema, cache, and injected-client tests for M7 group critic."""

from __future__ import annotations

import json
import hashlib
import unittest
from dataclasses import dataclass

from pydantic import ValidationError

from AgeMem_code_agentscope.group_critic import (
    ActionAPTrace,
    CounterfactualSuggestion,
    CriticCallUsage,
    CriticGroupInput,
    CriticHotpotQAPrivateReference,
    CriticInvocationResult,
    CriticOutput,
    CriticRolloutTrace,
    EvidenceStepRef,
    GroupCriticCache,
    LLMGroupCritic,
    MockGroupCritic,
    select_critic_automaton,
    validate_critic_output,
)
from AgeMem_code_agentscope.hotpotqa_benchmark.models import (
    HotpotContext,
    HotpotSupportingFacts,
)


TASK_ID = "hotpot-task-1"
TASK_QUESTION = "Find and retain two-hop evidence, then answer."


def _action(
    rollout_id: str,
    timestep: int,
    proposition: str,
    *,
    evidence_ids: tuple[str, ...] | None = None,
) -> ActionAPTrace:
    ids = evidence_ids if evidence_ids is not None else (f"ap-{rollout_id}-{timestep}",)
    return ActionAPTrace(
        evidence=EvidenceStepRef(
            task_id=TASK_ID,
            rollout_id=rollout_id,
            stage_id=min(timestep + 1, 3),
            timestep=timestep,
            action_id=f"action-{rollout_id}-{timestep}",
            assistant_turn_id=timestep,
            action_index_in_turn=0,
            ap_evidence_ids=tuple(sorted(ids)),
        ),
        action_type=("answer" if proposition == "answered_correctly" else "memory"),
        propositions=(proposition,),
        atomic_proposition_evidence={proposition: ids},
    )


def _rollout(
    rollout_id: str,
    outcome: str,
    propositions: tuple[str, ...],
) -> CriticRolloutTrace:
    return CriticRolloutTrace(
        task_id=TASK_ID,
        rollout_id=rollout_id,
        terminal_outcome=outcome,
        actions=tuple(
            _action(
                rollout_id,
                index,
                proposition,
                evidence_ids=(() if proposition == "answered_correctly" else None),
            )
            for index, proposition in enumerate(propositions)
        ),
        source_trajectory_digest=("a" if outcome == "success" else "b") * 64,
        ap_trace_digest=("c" if outcome == "success" else "d") * 64,
    )


def make_group(
    *, all_failed: bool = False, profile: str = "oracle"
) -> CriticGroupInput:
    chain = (
        "stored_supporting_fact",
        "supporting_coverage_complete",
        "retrieved_supporting_fact",
        "answered_correctly",
    )
    rollouts = (
        _rollout("failure-a", "failure", chain[:-1]),
        _rollout("failure-b", "failure", chain[:2]),
    )
    if not all_failed:
        rollouts = (_rollout("success", "success", chain), *rollouts)
    return CriticGroupInput(
        task_id=TASK_ID,
        group_id="group-1",
        split_id="train",
        task_description=TASK_QUESTION,
        critic_only_reference=CriticHotpotQAPrivateReference(
            hotpot_id="task-1",
            source_split="train",
            source_index=7,
            question=TASK_QUESTION,
            answer="Example answer",
            hotpot_type="bridge",
            level="medium",
            context=HotpotContext(
                title=("Document A", "Document B"),
                sentences=(("First supporting sentence.",), ("Second support.",)),
            ),
            supporting_facts=HotpotSupportingFacts(
                title=("Document A", "Document B"),
                sent_id=(0, 0),
            ),
        ),
        ap_profile=profile,
        rollouts=rollouts,
        source_report_digests=("e" * 64, "f" * 64),
    )


@dataclass
class _FakeClient:
    response: str
    calls: int = 0

    def complete(self, *, prompt: str) -> str:
        if "GROUP_INPUT" not in prompt:
            raise AssertionError("critic prompt omitted its complete group input")
        self.calls += 1
        return self.response


class _MeteredFakeCritic:
    """Protocol fake proving cache hits never repeat provider-billed usage."""

    critic_kind = "mock"
    critic_version = "agemem.group_critic.metered_fake.v1"
    model_version = "fake-provider-model"
    prompt_version = "agemem.group_critic.prompt.v2"

    def critique(self, group_input: CriticGroupInput) -> CriticInvocationResult:
        source = MockGroupCritic(
            critic_version=self.critic_version,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        ).critique(group_input)
        payload = source.model_dump(mode="python")
        payload["usage"] = CriticCallUsage(
            call_count=1,
            input_chars=source.usage.input_chars,
            output_chars=source.usage.output_chars,
            estimated_input_tokens=source.usage.estimated_input_tokens,
            estimated_output_tokens=source.usage.estimated_output_tokens,
            provider_input_tokens=17,
            provider_output_tokens=11,
            provider_cost=0.25,
            latency_ms=12.5,
        )
        return CriticInvocationResult.model_validate(payload)


class M7GroupCriticSchemaTest(unittest.TestCase):
    def test_schemas_are_frozen_extra_forbid_and_namespaced(self) -> None:
        reference = EvidenceStepRef(
            task_id="task-1",
            rollout_id="rollout-1",
            stage_id=1,
            timestep=0,
            action_id="action-1",
            assistant_turn_id=0,
            action_index_in_turn=0,
        )
        self.assertEqual(reference.schema_version, "agemem.critic_evidence_step_ref.v1")
        with self.assertRaises(ValidationError):
            EvidenceStepRef.model_validate(
                {**reference.model_dump(mode="python"), "unexpected": True}
            )
        with self.assertRaises(ValidationError):
            reference.action_id = "mutated"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            CounterfactualSuggestion(
                suggestion_id="cf-1",
                description="Never executable.",
                confidence=0.5,
                reward_eligible=True,
            )

    def test_private_hotpotqa_reference_is_complete_and_task_bound(self) -> None:
        group = make_group()
        payload = group.model_dump(mode="python")
        payload["critic_only_reference"]["answer"] = " "
        with self.assertRaisesRegex(ValidationError, "question and answer"):
            CriticGroupInput.model_validate(payload)

        payload = group.model_dump(mode="python")
        payload["critic_only_reference"]["supporting_facts"] = {
            "title": (),
            "sent_id": (),
        }
        with self.assertRaisesRegex(ValidationError, "requires supporting facts"):
            CriticGroupInput.model_validate(payload)

        payload = group.model_dump(mode="python")
        payload["critic_only_reference"]["hotpot_id"] = "another-task"
        with self.assertRaisesRegex(ValidationError, "must match group task_id"):
            CriticGroupInput.model_validate(payload)

        payload = group.model_dump(mode="python")
        payload["critic_only_reference"]["supporting_facts"]["sent_id"] = (9, 0)
        with self.assertRaisesRegex(ValidationError, "outside the context"):
            CriticGroupInput.model_validate(payload)

        payload = group.model_dump(mode="python")
        payload["critic_only_reference"]["source_split"] = "validation"
        with self.assertRaisesRegex(ValidationError, "source split"):
            CriticGroupInput.model_validate(payload)

    def test_answer_ap_allows_empty_record_ids_without_fabricating_provenance(
        self,
    ) -> None:
        answer = _action(
            "success",
            3,
            "answered_correctly",
            evidence_ids=(),
        )
        self.assertEqual(answer.atomic_proposition_evidence["answered_correctly"], ())
        self.assertEqual(answer.evidence.action_id, "action-success-3")

    def test_mock_critic_is_evidence_grounded_and_permutation_stable(self) -> None:
        group = make_group()
        critic = MockGroupCritic()
        first = critic.critique(group)
        self.assertIsNone(first.error)
        self.assertIsNotNone(first.output)
        assert first.output is not None
        self.assertEqual(
            tuple(item.proposition for item in first.output.milestones),
            (
                "stored_supporting_fact",
                "supporting_coverage_complete",
                "retrieved_supporting_fact",
                "answered_correctly",
            ),
        )
        report = validate_critic_output(group, first.output)
        self.assertTrue(report.valid and report.automaton_compilable)
        self.assertTrue(all(item.evidence_steps for item in first.output.milestones))

        repeated = critic.critique(group)
        self.assertEqual(repeated, first)
        permuted = group.model_copy(
            update={"rollouts": tuple(reversed(group.rollouts))}
        )
        second = critic.critique(permuted)
        self.assertEqual(permuted.digest, group.digest)
        self.assertIsNotNone(second.output)
        assert second.output is not None
        self.assertEqual(second.output.digest, first.output.digest)

    def test_group_cache_covers_profile_model_and_all_source_digests(self) -> None:
        group = make_group()
        cache = GroupCriticCache()
        critic = MockGroupCritic()
        cold = cache.get_or_critique(group, critic)
        warm = cache.get_or_critique(group, critic)
        self.assertFalse(cold.cache_hit)
        self.assertEqual(cold.result.usage.call_count, 1)
        self.assertTrue(warm.cache_hit)
        self.assertEqual(warm.result.usage.call_count, 0)
        self.assertEqual((cache.hits, cache.misses, cache.size), (1, 1, 1))

        changed_profile = group.model_copy(update={"ap_profile": "controlled_error"})
        changed_model = MockGroupCritic(model_version="deterministic-no-llm-v2")
        changed_source = group.model_copy(
            update={"source_report_digests": ("e" * 64, "1" * 64)}
        )
        changed_reference = group.model_copy(
            update={
                "critic_only_reference": group.critic_only_reference.model_copy(
                    update={"answer": "Different gold answer"}
                )
            }
        )
        self.assertNotEqual(
            cache.key_for(changed_profile, critic).digest,
            cold.cache_key_digest,
        )
        self.assertNotEqual(
            cache.key_for(group, changed_model).digest,
            cold.cache_key_digest,
        )
        self.assertNotEqual(
            cache.key_for(changed_source, critic).digest,
            cold.cache_key_digest,
        )
        self.assertNotEqual(
            cache.key_for(changed_reference, critic).digest,
            cold.cache_key_digest,
        )

    def test_cache_hit_zeros_provider_billed_usage(self) -> None:
        group = make_group()
        cache = GroupCriticCache()
        cold = cache.get_or_critique(group, _MeteredFakeCritic())
        warm = cache.get_or_critique(group, _MeteredFakeCritic())

        self.assertEqual(cold.result.usage.provider_input_tokens, 17)
        self.assertEqual(cold.result.usage.provider_output_tokens, 11)
        self.assertEqual(cold.result.usage.provider_cost, 0.25)
        self.assertEqual(warm.result.usage.call_count, 0)
        self.assertTrue(warm.result.usage.cache_hit)
        self.assertEqual(warm.result.usage.provider_input_tokens, 0)
        self.assertEqual(warm.result.usage.provider_output_tokens, 0)
        self.assertEqual(warm.result.usage.provider_cost, 0.0)
        self.assertIsNone(warm.result.usage.latency_ms)

    def test_llm_wrapper_uses_fake_and_never_silently_adopts_bad_json(self) -> None:
        group = make_group()
        expected = MockGroupCritic().critique(group).output
        self.assertIsNotNone(expected)
        assert expected is not None
        client = _FakeClient(expected.model_dump_json())
        critic = LLMGroupCritic(
            client,
            critic_version="agemem.group_critic.llm_adapter.v1",
            model_version="fake-model",
        )
        prompt = critic._prompt(group)
        self.assertIn('"visibility":"critic_only_privileged"', prompt)
        self.assertIn('"answer":"Example answer"', prompt)
        self.assertIn('"supporting_facts"', prompt)
        self.assertIn("First supporting sentence.", prompt)
        invocation = critic.critique(group)
        self.assertEqual(client.calls, 1)
        self.assertEqual(invocation.output, expected)
        self.assertEqual(
            invocation.raw_output_digest,
            hashlib.sha256(client.response.encode("utf-8")).hexdigest(),
        )
        decision = select_critic_automaton(group, invocation)
        self.assertEqual(decision.selected_source, "critic")

        invalid = LLMGroupCritic(
            _FakeClient(json.dumps({"schema_version": "wrong", "milestones": []})),
            critic_version="agemem.group_critic.llm_adapter.v1",
            model_version="fake-model",
        ).critique(group)
        self.assertIsNone(invalid.output)
        self.assertIsNotNone(invalid.error)
        fallback = select_critic_automaton(group, invalid)
        self.assertEqual(fallback.selected_source, "hand_authored")
        self.assertIsNotNone(fallback.automaton_spec)
        assert fallback.automaton_spec is not None
        self.assertEqual(fallback.automaton_spec.name, "m4-memory-oracle-positive-v1")
        assert fallback.fallback_reason is not None
        self.assertTrue(fallback.fallback_reason.startswith("llm_critic_error:"))

    def test_all_failure_group_is_counterfactual_only_and_terminal_only(self) -> None:
        group = make_group(all_failed=True)
        invocation = MockGroupCritic().critique(group)
        self.assertIsNotNone(invocation.output)
        assert invocation.output is not None
        output = invocation.output
        self.assertEqual(output.milestones, ())
        self.assertTrue(output.counterfactual_suggestions)
        self.assertTrue(
            all(
                suggestion.reward_eligible is False
                for suggestion in output.counterfactual_suggestions
            )
        )
        report = validate_critic_output(group, output)
        self.assertTrue(report.valid and not report.automaton_compilable)
        decision = select_critic_automaton(group, invocation)
        self.assertEqual(decision.selected_source, "terminal_only")
        self.assertIsNone(decision.automaton_spec)

    def test_all_failure_without_counterfactual_fails_closed(self) -> None:
        group = make_group(all_failed=True)
        empty = CriticOutput(task_id=group.task_id, group_id=group.group_id)
        report = validate_critic_output(group, empty)
        self.assertFalse(report.valid)
        self.assertIn(
            "all_failure_missing_counterfactual",
            {item.code for item in report.issues},
        )


if __name__ == "__main__":
    unittest.main()
