import asyncio
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.schema import (
    DynamicHistoryPublic,
    DynamicPublicChunk,
    DynamicQueryPrivate,
    DynamicQueryPublic,
    PrivateEvent,
    SourceRegistryRecord,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    DebugLexicalTokenizer,
    TokenAccounting,
)

if importlib.util.find_spec("datasets") is None:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = type("Dataset", (), {})
    sys.modules["datasets"] = datasets_stub

from trinity.common.action_event_contract import (
    ACTION_EVENTS_KEY,
    RESPONSE_TOKEN_OFFSETS_KEY,
    finalize_experience_action_contract,
    validate_on_policy_experiences,
)
from trinity.common.experience import Experience


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "agemem_dynamic_runtime_producer_test_module",
    ROOT
    / "trinity/common/workflows/memory_context/dynamic_runtime_producer.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNTIME_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME_MODULE
SPEC.loader.exec_module(RUNTIME_MODULE)
DynamicRuntimeProducer = RUNTIME_MODULE.DynamicRuntimeProducer
FrozenReaderOutput = RUNTIME_MODULE.FrozenReaderOutput
RUNTIME_PRODUCER_VERSION = RUNTIME_MODULE.RUNTIME_PRODUCER_VERSION


def tool_call(name, arguments=None):
    return (
        "<tool_call>"
        + json.dumps(
            [{"name": name, "arguments": arguments or {}}],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


def experience(text):
    response_ids = list(range(10, 10 + len(text)))
    return Experience(
        tokens=[1, 2, *response_ids],
        prompt_length=2,
        logprobs=[-0.1] * len(response_ids),
        action_mask=[True] * len(response_ids),
        response_text=text,
        info={
            RESPONSE_TOKEN_OFFSETS_KEY: [
                [index, index + 1]
                for index in range(len(text))
            ]
        },
    )


class FakePolicy:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    async def chat_async(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        return [experience(self.responses.pop(0))]


class DynamicRuntimeProducerTest(unittest.TestCase):
    def fixture(self):
        text0 = "projectA project_owner Alice effective 0"
        text1 = "projectA site_region Blue effective 0"
        events = (
            PrivateEvent(
                event_id="e0",
                history_id="h0",
                entity="projectA",
                relation="project_owner",
                value="Alice",
                observed_at=0,
                effective_at=0,
                source_ref="s0",
                source_text_hash=hashlib.sha256(text0.encode()).hexdigest(),
            ),
            PrivateEvent(
                event_id="e1",
                history_id="h0",
                entity="projectA",
                relation="site_region",
                value="Blue",
                observed_at=0,
                effective_at=0,
                source_ref="s1",
                source_text_hash=hashlib.sha256(text1.encode()).hexdigest(),
            ),
        )
        registry = (
            SourceRegistryRecord(
                source_ref="s0",
                history_id="h0",
                observed_at=0,
                text=text0,
                text_sha256=events[0].source_text_hash,
            ),
            SourceRegistryRecord(
                source_ref="s1",
                history_id="h0",
                observed_at=0,
                text=text1,
                text_sha256=events[1].source_text_hash,
            ),
        )
        history = DynamicHistoryPublic(
            history_id="h0",
            history_family_id="family0",
            chunks=(
                DynamicPublicChunk(
                    chunk_id="c0",
                    observed_at=0,
                    text=text0 + "\n" + text1,
                    source_refs=("s0", "s1"),
                    content_token_count=8,
                ),
            ),
            context_budget_tokens=4096,
            memory_budget_tokens=2048,
        )
        public = (
            DynamicQueryPublic(
                query_id="q0",
                history_id="h0",
                question="projectA project_owner value?",
            ),
            DynamicQueryPublic(
                query_id="q1",
                history_id="h0",
                question="projectA site_region value?",
            ),
        )
        private = (
            DynamicQueryPrivate(
                query_id="q0",
                history_id="h0",
                task_family="current_state",
                query_kind="current_state",
                entity="projectA",
                relation="project_owner",
                answers=("Alice",),
                answer_available_at=0,
                support_alternatives=(("e0",),),
            ),
            DynamicQueryPrivate(
                query_id="q1",
                history_id="h0",
                task_family="current_state",
                query_kind="current_state",
                entity="projectA",
                relation="site_region",
                answers=("Blue",),
                answer_available_at=0,
                support_alternatives=(("e1",),),
            ),
        )
        return history, public, private, events, registry

    def test_model_group_reaches_experience_and_ddof0_advantage(self):
        memory = {
            "memory_id": "m0",
            "content": (
                "projectA project_owner Alice; projectA site_region Blue"
            ),
            "source_refs": ["s0", "s1"],
            "claims": [
                {
                    "entity": "projectA",
                    "relation": "project_owner",
                    "value": "Alice",
                    "effective_at": 0,
                },
                {
                    "entity": "projectA",
                    "relation": "site_region",
                    "value": "Blue",
                    "effective_at": 0,
                },
            ],
        }
        policy = FakePolicy(
            [tool_call("ADD", memory), tool_call("NEXT"), tool_call("NEXT")]
        )
        reader_messages = []

        async def reader(messages):
            reader_messages.append(messages)
            rendered = str(messages)
            if "project_owner" in rendered and "Alice" in rendered:
                answer = "<answer>Alice</answer>"
            elif "site_region" in rendered and "Blue" in rendered:
                answer = "<answer>Blue</answer>"
            else:
                answer = "<answer>wrong</answer>"
            return FrozenReaderOutput(answer, 999, "frozen-reader:test")

        accounting = TokenAccounting.from_tokenizer(
            DebugLexicalTokenizer(), revision="fixture-v1"
        )
        config = {
            "budget": {
                "context_total_tokens": 4096,
                "persistent_memory_tokens": 2048,
                "ingest_max_new_tokens": 512,
                "answer_max_new_tokens": 64,
                "answer_tail_tokens": 64,
                "retrieved_payload_tokens": 1024,
                "max_decisions_per_chunk": 2,
            },
            "data": {
                "memory_rollouts_per_group": 2,
                "questions_per_snapshot": 2,
            },
            "reward": {"profile": "V2_LIFE", "lambda_semantic": 0.25},
            "training": {"std_ddof": 0},
        }
        history, public, private, events, registry = self.fixture()
        producer = DynamicRuntimeProducer(
            policy_model=policy,
            frozen_reader=reader,
            accounting=accounting,
            config=config,
            policy_version="model_version:0",
            generation_args={"temperature": 0.6},
        )
        produced = asyncio.run(
            producer.produce_group(
                history=history,
                public_queries=public,
                private_queries=private,
                events=events,
                source_registry=registry,
                batch_id="batch0",
                task_id="task0",
                run_id_base=0,
                repeat_times=2,
            )
        )
        self.assertEqual(produced.receipt["runtime_producer_version"], RUNTIME_PRODUCER_VERSION)
        self.assertEqual(produced.receipt["read_rollout_count"], 2)
        self.assertEqual(produced.receipt["query_branch_count"], 4)
        self.assertEqual(len(produced.experiences), 3)
        self.assertEqual(len(reader_messages), 4)
        self.assertTrue(all("QUESTION" not in str(item[0]) for item in policy.messages))
        self.assertTrue(all(exp.info["phase"] == "ingest" for exp in produced.experiences))
        self.assertTrue(all(exp.reward is not None for exp in produced.experiences))
        for exp in produced.experiences:
            public_info = json.dumps(exp.info, ensure_ascii=False)
            self.assertNotIn("<answer>Alice</answer>", public_info)
            self.assertNotIn("<answer>Blue</answer>", public_info)
            self.assertNotIn('"retrieved_payloads"', public_info)
        for exp in produced.experiences:
            finalize_experience_action_contract(
                exp, policy_version="model_version:0"
            )
            self.assertEqual(len(exp.info[ACTION_EVENTS_KEY]), 1)
        validate_on_policy_experiences(list(produced.experiences))

        self.assertEqual(
            {
                round(float(exp.info["dynamic_precomputed_advantage_ddof0"]), 6)
                for exp in produced.experiences
            },
            {-1.0, 1.0},
        )


if __name__ == "__main__":
    unittest.main()
