"""CPU-only contract tests for M8a online ActionEvents."""

from __future__ import annotations

import asyncio
import json
import types
import unittest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

from pydantic import ValidationError

from AgeMem_code_agentscope.action_schema import (
    ActionCreditRecord,
    ActionEvent,
    RewardBreakdownV2,
)
from trinity.common.action_event_contract import (
    ACTION_CHARACTER_SPANS_KEY,
    ACTION_CREDITS_KEY,
    ACTION_DRAFTS_KEY,
    ACTION_EVENTS_KEY,
    RESPONSE_TOKEN_OFFSETS_KEY,
    ActionContractError,
    derive_response_token_char_offsets,
    finalize_experience_action_contract,
    freeze_rollout_policy_version,
    join_action_events_to_credits,
    mark_experience_off_policy,
    parse_tool_calls_with_char_spans,
    prepare_experience_action_drafts,
    record_experience_action_result,
    response_metadata_for_generation,
    validate_on_policy_experiences,
)

try:
    from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
    from trinity.common.experience import Experience as RuntimeExperience
    from trinity.common.workflows.workflow import Task
    from trinity.explorer.workflow_runner import WorkflowRunner
except ModuleNotFoundError as exc:  # Windows/local schema environment.
    ExperiencePipeline = None
    RuntimeExperience = None
    Task = None
    WorkflowRunner = None
    RUNTIME_IMPORT_ERROR: Optional[BaseException] = exc
else:
    RUNTIME_IMPORT_ERROR = None


@dataclass
class FakeEID:
    batch: str = "batch-1"
    task: str = "task-2"
    run: int = 3
    step: int = 0

    @property
    def tid(self):
        return f"{self.batch}/{self.task}"

    @property
    def rid(self):
        return f"{self.batch}/{self.task}/{self.run}"


@dataclass
class FakeExperience:
    tokens: Any
    prompt_length: int
    logprobs: Any
    response_text: str
    info: dict[str, Any]
    eid: FakeEID = field(default_factory=FakeEID)
    action_mask: Any = None


class CharacterTokenizer:
    """One Unicode code point per token; sufficient for deterministic CPU tests."""

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


def make_response(*, repeated=False):
    first = {"name": "Add_memory", "arguments": {"content": "Paris"}}
    second = (
        first
        if repeated
        else {
            "name": "Retrieve_memory",
            "arguments": {"query": "Paris", "top_k": 1},
        }
    )
    return (
        "plan\n<tool_call>"
        + json.dumps([first, second], separators=(",", ":"))
        + "</tool_call>\n"
    )


def make_experience(response_text, *, logprob_delta=0, step=0):
    tokenizer = CharacterTokenizer()
    response_ids = [ord(character) for character in response_text]
    return FakeExperience(
        eid=FakeEID(step=step),
        tokens=[999, *response_ids],
        prompt_length=1,
        logprobs=[
            -0.01 * (index + 1) for index in range(len(response_ids) + logprob_delta)
        ],
        response_text=response_text,
        info=response_metadata_for_generation(tokenizer, response_ids, response_text),
    )


def complete_drafts(experience):
    drafts = experience.info[ACTION_DRAFTS_KEY]
    for index, draft in enumerate(drafts):
        trace_call_id = f"trace-{index}"
        info = dict(experience.info)
        info.setdefault("tool_call_ids", []).append(trace_call_id)
        experience.info = info
        record_experience_action_result(
            [experience],
            action_index_in_turn=index,
            trace_call_id=trace_call_id,
            action_type=draft["action_type"],
            status="success",
            result={"effect_applied": True, "index": index},
        )


def credit_for(event):
    breakdown = RewardBreakdownV2(
        env=0.0,
        milestone=0.0,
        violation=0.0,
        trend=0.0,
        format=0.0,
        cost=0.0,
        total=0.0,
        automaton_state_before="q0",
        automaton_state_after="q0",
        automaton_status="running",
    )
    return ActionCreditRecord(
        action_id=event.action_id,
        task_id=event.task_id,
        rollout_id=event.rollout_id,
        stage_id=event.stage_id,
        timestep=event.timestep,
        dfa_spec_id="test-dfa",
        dfa_state_before="q0",
        dfa_state_after="q0",
        reward_breakdown=breakdown,
        reward_version="test-reward-v1",
    )


class TokenCharacterAlignmentTest(unittest.TestCase):
    def test_exact_offsets_use_generated_token_ids(self):
        tokenizer = CharacterTokenizer()
        text = "A工具B"
        token_ids = [ord(character) for character in text]
        self.assertEqual(
            derive_response_token_char_offsets(tokenizer, token_ids, text),
            ((0, 1), (1, 2), (2, 3), (3, 4)),
        )

    def test_mismatched_generation_tokens_fail_closed(self):
        with self.assertRaisesRegex(ActionContractError, "cannot derive exact"):
            derive_response_token_char_offsets(CharacterTokenizer(), [1, 2], "ab")


class PolicyVersionContractTest(unittest.TestCase):
    def test_k_rollout_policy_version_is_frozen_or_rejected(self):
        self.assertEqual(freeze_rollout_policy_version(5, 5), "model_version:5")
        with self.assertRaisesRegex(ActionContractError, "policy version changed"):
            freeze_rollout_policy_version(5, 6)
        with self.assertRaisesRegex(ActionContractError, "must be integers"):
            freeze_rollout_policy_version("5", "5")


class ToolCallSpanTest(unittest.TestCase):
    def test_repeated_calls_keep_distinct_exact_character_spans(self):
        response = make_response(repeated=True)
        parsed = parse_tool_calls_with_char_spans(response)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].call, parsed[1].call)
        self.assertLessEqual(parsed[0].char_end, parsed[1].char_start)
        for item in parsed:
            self.assertEqual(
                json.loads(response[item.char_start : item.char_end]), item.call
            )

    def test_truncated_tolerant_call_is_not_given_a_fake_span(self):
        with self.assertRaisesRegex(ActionContractError, "truncated"):
            parse_tool_calls_with_char_spans(
                '<tool_call>[{"name":"Add_memory","arguments":{}}'
            )


class OnlineActionEventContractTest(unittest.TestCase):
    def finalized(self, *, repeated=False, run=3, policy_version="model_version:11"):
        experience = make_experience(make_response(repeated=repeated), step=4)
        experience.eid.run = run
        prepare_experience_action_drafts(
            experience,
            stage_id=1,
            timestep=4,
            assistant_turn_id=7,
        )
        complete_drafts(experience)
        events = finalize_experience_action_contract(
            experience, policy_version=policy_version
        )
        return experience, events

    def test_online_contract_preserves_tokens_logprobs_spans_and_stable_ids(self):
        experience, events = self.finalized(repeated=True)
        self.assertEqual(len(events), 2)
        self.assertEqual([event.action_index_in_turn for event in events], [0, 1])
        self.assertEqual(len({event.action_id for event in events}), 2)
        response_ids = list(experience.tokens[experience.prompt_length :])
        for event in events:
            self.assertEqual(list(event.response_token_ids), response_ids)
            self.assertEqual(len(event.old_logprobs), len(response_ids))
            self.assertEqual(event.policy_version, "model_version:11")
            self.assertEqual(event.source, "llm")
        self.assertLessEqual(events[0].token_end, events[1].token_start)
        validate_on_policy_experiences([experience])

        same_experience, same_events = self.finalized(repeated=True)
        self.assertEqual(
            [event.action_id for event in events],
            [event.action_id for event in same_events],
        )
        self.assertEqual(
            experience.info[ACTION_CHARACTER_SPANS_KEY],
            same_experience.info[ACTION_CHARACTER_SPANS_KEY],
        )

    def test_length_mismatch_missing_result_and_bad_offsets_fail_closed(self):
        with self.assertRaisesRegex(ActionContractError, "identical lengths"):
            prepare_experience_action_drafts(
                make_experience(make_response(), logprob_delta=-1),
                stage_id=1,
                timestep=0,
                assistant_turn_id=0,
            )

        missing_result = make_experience(make_response())
        prepare_experience_action_drafts(
            missing_result, stage_id=1, timestep=0, assistant_turn_id=0
        )
        with self.assertRaisesRegex(ActionContractError, "missing its tool result"):
            finalize_experience_action_contract(
                missing_result, policy_version="model_version:0"
            )

        bad_offsets = make_experience(make_response())
        bad_offsets.info[RESPONSE_TOKEN_OFFSETS_KEY][1][0] = 0
        with self.assertRaisesRegex(ActionContractError, "contiguous"):
            prepare_experience_action_drafts(
                bad_offsets, stage_id=1, timestep=0, assistant_turn_id=0
            )

    def test_tampered_overlap_and_duplicate_action_ids_are_rejected(self):
        experience, events = self.finalized()
        second = events[1].model_copy(update={"token_start": events[0].token_end - 1})
        experience.info[ACTION_EVENTS_KEY][1] = second.model_dump(mode="json")
        with self.assertRaisesRegex(ActionContractError, "overlap"):
            validate_on_policy_experiences([experience])

        first, _ = self.finalized()
        duplicate, _ = self.finalized()
        with self.assertRaisesRegex(
            ActionContractError, "duplicate on-policy action_id"
        ):
            validate_on_policy_experiences([first, duplicate])

    def test_tampered_span_mapping_and_tool_trace_join_are_rejected(self):
        experience, events = self.finalized()
        shifted = events[0].model_copy(
            update={"token_start": events[0].token_start + 1}
        )
        experience.info[ACTION_EVENTS_KEY][0] = shifted.model_dump(mode="json")
        with self.assertRaisesRegex(ActionContractError, "character span"):
            validate_on_policy_experiences([experience])

        experience, _ = self.finalized()
        experience.info["tool_call_ids"][0] = "wrong-trace-call"
        with self.assertRaisesRegex(ActionContractError, "tool trace"):
            validate_on_policy_experiences([experience])

    def test_agemem_admission_requires_a_complete_contract(self):
        uncontracted = make_experience("answer")
        validate_on_policy_experiences([uncontracted])
        with self.assertRaisesRegex(ActionContractError, "missing its action"):
            validate_on_policy_experiences(
                [uncontracted], require_contract=True
            )

        experience, _ = self.finalized()
        experience.info.pop(ACTION_EVENTS_KEY)
        with self.assertRaisesRegex(ActionContractError, "incomplete"):
            validate_on_policy_experiences([experience])

    def test_trace_and_experience_coordinates_must_match(self):
        experience, _ = self.finalized()
        experience.info["trace_step"] = 5
        with self.assertRaisesRegex(ActionContractError, "trace stage/step"):
            validate_on_policy_experiences([experience])

        wrong_source = make_experience(make_response(), step=2)
        with self.assertRaisesRegex(ActionContractError, "EID step"):
            prepare_experience_action_drafts(
                wrong_source,
                stage_id=1,
                timestep=3,
                assistant_turn_id=0,
            )

    def test_one_task_cannot_mix_policy_versions_across_rollouts(self):
        first, _ = self.finalized(run=3, policy_version="model_version:11")
        second, _ = self.finalized(run=4, policy_version="model_version:12")
        with self.assertRaisesRegex(ActionContractError, "multiple rollout policy"):
            validate_on_policy_experiences([first, second])

    def test_action_credit_join_is_exact_and_validated_at_buffer_boundary(self):
        experience, events = self.finalized()
        credits = [credit_for(event) for event in events]
        joined = join_action_events_to_credits(events, credits)
        self.assertEqual(
            [event.action_id for event, _ in joined], [c.action_id for c in credits]
        )
        experience.info[ACTION_CREDITS_KEY] = [
            credit.model_dump(mode="json") for credit in credits
        ]
        validate_on_policy_experiences([experience])

        experience.info[ACTION_CREDITS_KEY].pop()
        with self.assertRaisesRegex(ActionContractError, "sets must match exactly"):
            validate_on_policy_experiences([experience])

    def test_offline_sources_have_no_training_metadata_and_never_enter_buffer(self):
        event = ActionEvent(
            action_id="offline-action",
            task_id="task",
            rollout_id="rollout",
            stage_id=1,
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=0,
            source="oracle",
            action_type="Add_memory",
            action_text="{}",
            arguments={},
            result={},
        )
        self.assertIsNone(event.response_token_ids)
        with self.assertRaisesRegex(ValidationError, "non-LLM actions"):
            ActionEvent(
                **event.model_dump(
                    exclude={
                        "response_token_ids",
                        "token_start",
                        "token_end",
                        "old_logprobs",
                        "policy_version",
                    }
                ),
                response_token_ids=(1,),
                token_start=0,
                token_end=1,
                old_logprobs=(-0.1,),
                policy_version="fabricated",
            )

        experience = make_experience("answer")
        mark_experience_off_policy(experience, source="oracle")
        with self.assertRaisesRegex(ActionContractError, "forbidden"):
            validate_on_policy_experiences([experience])

    def test_unfinished_draft_is_never_buffer_eligible(self):
        experience = make_experience(make_response())
        prepare_experience_action_drafts(
            experience, stage_id=1, timestep=0, assistant_turn_id=0
        )
        with self.assertRaisesRegex(ActionContractError, "unfinished"):
            validate_on_policy_experiences([experience])


class _VersionSequence:
    def __init__(self, versions):
        self.versions = iter(versions)

    @property
    def model_version_async(self):
        async def read_version():
            return next(self.versions)

        return read_version()


@unittest.skipIf(
    WorkflowRunner is None,
    f"Trinity runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}",
)
class RolloutPolicyFreezeTest(unittest.TestCase):
    @staticmethod
    def runner(versions):
        runner = WorkflowRunner.__new__(WorkflowRunner)
        runner.model_wrapper = _VersionSequence(versions)
        runner.logger = MagicMock()

        async def fake_run_task(self, task, repeat_times, run_id_base):
            del self, task, repeat_times, run_id_base
            return [
                RuntimeExperience(
                    tokens=[1, 2],
                    prompt_length=1,
                    logprobs=[-0.1],
                    response_text="x",
                )
            ]

        runner._run_task = types.MethodType(fake_run_task, runner)
        return runner

    def test_same_policy_version_is_written_to_every_rollout_experience(self):
        task = Task(batch_id="batch", task_id="task", is_eval=False)
        status, experiences = asyncio.run(self.runner([5, 5]).run_task(task))
        self.assertTrue(status.ok)
        self.assertEqual(len(experiences), 1)
        self.assertEqual(experiences[0].info["model_version"], 5)

    def test_policy_change_during_k_rollouts_discards_entire_group(self):
        task = Task(batch_id="batch", task_id="task", is_eval=False)
        status, experiences = asyncio.run(self.runner([5, 6]).run_task(task))
        self.assertFalse(status.ok)
        self.assertEqual(experiences, [])
        self.assertIn("policy version changed", status.message)


class _CapturingWriter:
    def __init__(self):
        self.writes = []

    async def write_async(self, experiences):
        self.writes.append(experiences)


class _DroppingContractOperator:
    def process(self, experiences):
        for experience in experiences:
            experience.info = {}
        return experiences, {}


@unittest.skipIf(
    ExperiencePipeline is None,
    f"Trinity runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}",
)
class BufferAdmissionGuardTest(unittest.TestCase):
    def test_offline_experience_is_rejected_before_any_buffer_write(self):
        pipeline = ExperiencePipeline.__new__(ExperiencePipeline)
        pipeline.require_agemem_action_contract = True
        pipeline.input_store = _CapturingWriter()
        pipeline.output = _CapturingWriter()
        pipeline.operators = []
        experience = make_experience("answer")
        mark_experience_off_policy(experience, source="error_injector")

        with self.assertRaisesRegex(ActionContractError, "forbidden"):
            asyncio.run(pipeline.process([experience]))
        self.assertEqual(pipeline.input_store.writes, [])
        self.assertEqual(pipeline.output.writes, [])

        online = make_experience(make_response(), step=4)
        prepare_experience_action_drafts(
            online,
            stage_id=1,
            timestep=4,
            assistant_turn_id=7,
        )
        complete_drafts(online)
        finalize_experience_action_contract(
            online, policy_version="model_version:11"
        )
        pipeline.input_store = None
        pipeline.output = _CapturingWriter()
        pipeline.operators = [_DroppingContractOperator()]
        with self.assertRaisesRegex(ActionContractError, "missing its action"):
            asyncio.run(pipeline.process([online]))
        self.assertEqual(pipeline.output.writes, [])


if __name__ == "__main__":
    unittest.main()
