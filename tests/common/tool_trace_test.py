import importlib.util
import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from trinity.common.constants import LOG_DIR_ENV_VAR
from trinity.common.tool_trace import ToolTraceRecorder, resolve_tool_trace_path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    """Use a normal workspace directory; Windows tempfile ACLs can be restrictive."""
    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"tool-trace-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def load_source_module(module_name: str, relative_path: str):
    """Load a lightweight source file without importing workflows/__init__.py."""
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


memory_utils = load_source_module(
    "_agemem_memory_utils_for_test",
    "trinity/common/workflows/memory_context/utils.py",
)
memory_reward = load_source_module(
    "_agemem_memory_reward_for_test",
    "trinity/common/workflows/memory_reward/my_reward.py",
)
memory_store = load_source_module(
    "_agemem_memory_store_for_test",
    "trinity/common/workflows/memory_context/memory_store.py",
)


def load_training_module_with_stubs():
    """Load the workflow class without torch, datasets, Ray, or network clients."""

    class FakeRegistry:
        @staticmethod
        def register_module(_name):
            return lambda cls: cls

    class FakeWorkflow:
        pass

    experience_module = ModuleType("trinity.common.experience")
    experience_module.Experience = type("Experience", (), {})

    models_package = ModuleType("trinity.common.models")
    models_package.__path__ = []
    model_module = ModuleType("trinity.common.models.model")
    model_module.ModelWrapper = type("ModelWrapper", (), {})

    workflows_package = ModuleType("trinity.common.workflows")
    workflows_package.__path__ = [str(REPOSITORY_ROOT / "trinity/common/workflows")]
    workflow_module = ModuleType("trinity.common.workflows.workflow")
    workflow_module.WORKFLOWS = FakeRegistry()
    workflow_module.MultiTurnWorkflow = FakeWorkflow
    workflow_module.Task = type("Task", (), {})

    memory_context_package = ModuleType("trinity.common.workflows.memory_context")
    memory_context_package.__path__ = [
        str(REPOSITORY_ROOT / "trinity/common/workflows/memory_context")
    ]
    memory_store_module = ModuleType(
        "trinity.common.workflows.memory_context.memory_store"
    )
    memory_store_module.MemoryManager = type("MemoryManager", (), {})
    memory_store_module.chat_client = type("chat_client", (), {})

    prompt_module = ModuleType(
        "trinity.common.workflows.memory_context.workflow_prompt"
    )
    prompt_module.TOOL_CALL_SYS_PROMPT = "{tools}"
    prompt_module.SUMMARY_CONTEXT_SYS_PROMPT = "{conversation_text}"
    prompt_module.TEXT_SIMILARITY_SYS_PROMPT = "{text1}\n{text2}"

    metrics_module = ModuleType(
        "trinity.common.workflows.memory_context.workflow_metrics"
    )

    async def fake_answer_score(*_args, **_kwargs):
        return 0.0

    metrics_module.get_answer_llm_judge_score = fake_answer_score

    memory_reward_package = ModuleType("trinity.common.workflows.memory_reward")
    memory_reward_package.__path__ = [
        str(REPOSITORY_ROOT / "trinity/common/workflows/memory_reward")
    ]

    utils_package = ModuleType("trinity.utils")
    utils_package.__path__ = []
    log_module = ModuleType("trinity.utils.log")
    log_module.get_logger = lambda **_kwargs: mock.MagicMock()

    stubs = {
        "trinity.common.experience": experience_module,
        "trinity.common.models": models_package,
        "trinity.common.models.model": model_module,
        "trinity.common.workflows": workflows_package,
        "trinity.common.workflows.workflow": workflow_module,
        "trinity.common.workflows.memory_context": memory_context_package,
        "trinity.common.workflows.memory_context.memory_store": memory_store_module,
        "trinity.common.workflows.memory_context.workflow_prompt": prompt_module,
        "trinity.common.workflows.memory_context.utils": memory_utils,
        "trinity.common.workflows.memory_context.workflow_metrics": metrics_module,
        "trinity.common.workflows.memory_reward": memory_reward_package,
        "trinity.common.workflows.memory_reward.my_reward": memory_reward,
        "trinity.utils": utils_package,
        "trinity.utils.log": log_module,
    }

    module_name = "trinity.common.workflows.memory_context._train_hotpotqa_for_test"
    source_path = (
        REPOSITORY_ROOT / "trinity/common/workflows/memory_context/train_hotpotQA.py"
    )
    with mock.patch.dict(sys.modules, stubs, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


training_module = load_training_module_with_stubs()


class ToolTraceRecorderTest(unittest.TestCase):
    def test_resolves_default_and_explicit_paths(self):
        with workspace_temp_directory() as temp_dir:
            job_dir = Path(temp_dir) / "job"
            log_dir = job_dir / "log"
            with mock.patch.dict(
                os.environ,
                {LOG_DIR_ENV_VAR: str(log_dir)},
                clear=True,
            ):
                self.assertEqual(
                    Path(resolve_tool_trace_path()),
                    job_dir.resolve() / "trajectories" / "tool_calls.jsonl",
                )

            explicit_dir = Path(temp_dir) / "custom"
            explicit_path = resolve_tool_trace_path(
                {"tool_trace_path": str(explicit_dir)}
            )
            self.assertEqual(
                Path(explicit_path),
                explicit_dir.resolve() / "tool_calls.jsonl",
            )

    def test_writes_linked_start_and_finish_records(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "trace" / "calls.jsonl"
            recorder = ToolTraceRecorder(str(trace_path))
            context = {
                "batch_id": 3,
                "task_id": 4,
                "run_id": 5,
                "execution_id": "execution-1",
                "stage": 2,
                "round": 1,
                "step": 6,
                "turn": 1,
            }

            call_id, started_at = recorder.record_start(
                context=context,
                tool_name="Add_memory",
                tool_index=0,
                arguments={"content": "需要记住的事实"},
                state_before={"context_message_count": 2, "memory_count": 0},
            )
            finish = recorder.record_finish(
                call_id=call_id,
                started_at=started_at,
                context=context,
                tool_name="Add_memory",
                tool_index=0,
                arguments={"content": "需要记住的事实"},
                status="success",
                result={"memory_id": "m1", "effect_applied": True},
                state_after={"context_message_count": 3, "memory_count": 1},
            )

            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["phase"] for record in records], ["start", "finish"]
            )
            self.assertEqual({record["call_id"] for record in records}, {call_id})
            self.assertEqual(len({record["record_id"] for record in records}), 2)
            self.assertEqual(records[1]["execution_id"], "execution-1")
            self.assertEqual(records[1]["step"], 6)
            self.assertEqual(finish["result"]["memory_id"], "m1")

    def test_redacts_secrets_and_bounds_long_strings(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "calls.jsonl"
            recorder = ToolTraceRecorder(
                str(trace_path),
                max_string_chars=256,
            )
            arguments = {
                "api_key": "real-secret-key",
                "dashscope_api_key": "prefixed-real-secret-key",
                "dashscopeApiKey": "camel-case-key-secret",
                "clientSecret": "camel-case-client-secret",
                "accessToken": "camel-case-access-token",
                "note": "password=hunter2",
                "headers": "Authorization: Bearer another-secret",
                "json_text": 'request headers: {"api_key": "quoted-secret"}',
                "env_text": "DASHSCOPE_API_KEY=environment-secret",
                "token_text": "request failed token='token-secret'",
                "bearer_text": "Bearer bearer-secret-value",
                "bare_api_token": "sk-live-1234567890",
                "github_text": "ghp_abcdefghijklmnopqrstuvwxyz123456",
                "jwt_text": (
                    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123"
                ),
                "secret_keys_as_data": {
                    "ghp_abcdefghijklmnopqrstuvwxyz123456": "first",
                    "ghp_zyxwvutsrqponmlkjihgfedcba654321": "second",
                },
                "long_text": "甲" * 300,
                "not_a_number": float("nan"),
                "unordered": {"b", "a"},
            }

            call_id, started_at = recorder.record_start(
                context={},
                tool_name="Add_memory",
                tool_index=0,
                arguments=arguments,
                state_before={},
            )
            recorder.record_finish(
                call_id=call_id,
                started_at=started_at,
                context={},
                tool_name="Add_memory",
                tool_index=0,
                arguments=arguments,
                status="success",
                result={},
                state_after={},
            )

            raw_text = trace_path.read_text(encoding="utf-8")
            self.assertNotIn("real-secret-key", raw_text)
            self.assertNotIn("prefixed-real-secret-key", raw_text)
            self.assertNotIn("camel-case-key-secret", raw_text)
            self.assertNotIn("camel-case-client-secret", raw_text)
            self.assertNotIn("camel-case-access-token", raw_text)
            self.assertNotIn("another-secret", raw_text)
            self.assertNotIn("hunter2", raw_text)
            self.assertNotIn("quoted-secret", raw_text)
            self.assertNotIn("environment-secret", raw_text)
            self.assertNotIn("token-secret", raw_text)
            self.assertNotIn("bearer-secret-value", raw_text)
            self.assertNotIn("sk-live-1234567890", raw_text)
            self.assertNotIn(
                "ghp_abcdefghijklmnopqrstuvwxyz123456",
                raw_text,
            )
            self.assertNotIn("signature123", raw_text)
            self.assertNotIn(
                "ghp_zyxwvutsrqponmlkjihgfedcba654321",
                raw_text,
            )
            records = [json.loads(line) for line in raw_text.splitlines()]
            safe_arguments = records[0]["arguments"]
            self.assertEqual(safe_arguments["api_key"], "[REDACTED]")
            self.assertEqual(safe_arguments["dashscope_api_key"], "[REDACTED]")
            self.assertEqual(safe_arguments["note"], "password=[REDACTED]")
            self.assertEqual(
                safe_arguments["headers"],
                "Authorization: [REDACTED]",
            )
            self.assertTrue(safe_arguments["long_text"]["_truncated"])
            self.assertEqual(safe_arguments["not_a_number"], "nan")
            self.assertEqual(safe_arguments["unordered"], ["a", "b"])
            safe_secret_keys = safe_arguments["secret_keys_as_data"]
            self.assertEqual(len(safe_secret_keys), 2)
            self.assertEqual(len(set(safe_secret_keys)), 2)

    def test_disabled_recorder_keeps_in_memory_event_without_file(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "disabled.jsonl"
            recorder = ToolTraceRecorder(str(trace_path), enabled=False)
            call_id, started_at = recorder.record_start(
                context={},
                tool_name="Clear_context",
                tool_index=0,
                arguments={"criteria": "noise"},
                state_before={},
            )
            finish = recorder.record_finish(
                call_id=call_id,
                started_at=started_at,
                context={},
                tool_name="Clear_context",
                tool_index=0,
                arguments={"criteria": "noise"},
                status="success",
                result={"effect_applied": False},
                state_after={},
            )
            self.assertFalse(trace_path.exists())
            self.assertEqual(finish["call_id"], call_id)

    def test_setup_failure_disables_trace_without_breaking_caller(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "unwritable" / "calls.jsonl"
            with mock.patch(
                "trinity.common.tool_trace.Path.mkdir",
                side_effect=PermissionError("read only"),
            ):
                recorder = ToolTraceRecorder(str(trace_path))
            self.assertFalse(recorder.enabled)

            invalid_config = ToolTraceRecorder.from_workflow_args(
                {"tool_trace_max_string_chars": "not-an-integer"}
            )
            self.assertFalse(invalid_config.enabled)

    def test_concurrent_local_appends_remain_valid_jsonl(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "concurrent.jsonl"
            recorders = [ToolTraceRecorder(str(trace_path)) for _ in range(4)]

            def write_call(index: int):
                recorder = recorders[index % len(recorders)]
                call_id, started_at = recorder.record_start(
                    context={"stage": 1, "round": index},
                    tool_name="Retrieve_memory",
                    tool_index=0,
                    arguments={"query": str(index)},
                    state_before={},
                )
                recorder.record_finish(
                    call_id=call_id,
                    started_at=started_at,
                    context={"stage": 1, "round": index},
                    tool_name="Retrieve_memory",
                    tool_index=0,
                    arguments={"query": str(index)},
                    status="success",
                    result={"retrieved_count": 0, "effect_applied": False},
                    state_after={},
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write_call, range(20)))

            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 40)
            self.assertEqual(
                sum(record["phase"] == "start" for record in records),
                20,
            )
            self.assertEqual(
                sum(record["phase"] == "finish" for record in records),
                20,
            )

    def test_ray_timeout_switches_to_fallback_without_blocking(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "ray-timeout.jsonl"
            recorder = ToolTraceRecorder(
                str(trace_path),
                ray_write_timeout_seconds=0.25,
            )
            remote_method = SimpleNamespace(remote=mock.Mock(return_value="object-ref"))
            fake_ray = SimpleNamespace(
                get=mock.Mock(side_effect=TimeoutError("writer stalled")),
                cancel=mock.Mock(),
            )
            recorder._ray = fake_ray
            recorder._ray_writer = SimpleNamespace(append_line=remote_method)

            recorder.record_start(
                context={},
                tool_name="Retrieve_memory",
                tool_index=0,
                arguments={"query": "fact"},
                state_before={},
            )

            fake_ray.get.assert_called_once_with(
                "object-ref",
                timeout=0.25,
            )
            fake_ray.cancel.assert_called_once_with(
                "object-ref",
                force=False,
            )
            self.assertIsNotNone(recorder.fallback_path)
            self.assertTrue(Path(recorder.fallback_path).exists())
            self.assertEqual(recorder.dropped_record_count, 0)

    def test_double_write_failure_is_exposed(self):
        with workspace_temp_directory() as temp_dir:
            recorder = ToolTraceRecorder(str(Path(temp_dir) / "write-failure.jsonl"))
            with mock.patch.object(
                recorder,
                "_append_line_locally",
                side_effect=OSError("disk unavailable"),
            ):
                recorder.record_start(
                    context={},
                    tool_name="Clear_context",
                    tool_index=0,
                    arguments={"criteria": "noise"},
                    state_before={},
                )

            self.assertEqual(recorder.dropped_record_count, 1)
            self.assertIn("disk unavailable", recorder.last_write_error)


class ToolValidationAndSelectionTest(unittest.TestCase):
    def test_stage_intermediate_experience_rules_are_exact(self):
        expected_by_stage = {
            1: {"Add_memory", "Retrieve_memory", "Update_memory"},
            2: {"Summary_context", "Clear_context"},
            3: {"Summary_context", "Clear_context", "Retrieve_memory"},
        }
        all_tools = {
            "Summary_context",
            "Clear_context",
            "Retrieve_memory",
            "Add_memory",
            "Update_memory",
            "Delete_memory",
        }

        for stage, expected_tools in expected_by_stage.items():
            for tool_name in all_tools:
                with self.subTest(stage=stage, tool_name=tool_name):
                    selected = memory_utils.should_collect_intermediate_experience(
                        stage,
                        [{"name": tool_name, "arguments": {}}],
                        is_last_round=False,
                    )
                    self.assertEqual(selected, tool_name in expected_tools)
                    self.assertFalse(
                        memory_utils.should_collect_intermediate_experience(
                            stage,
                            [{"name": tool_name, "arguments": {}}],
                            is_last_round=True,
                        )
                    )

    def test_tool_validation_normalizes_safe_inputs(self):
        retrieve, error = memory_utils.validate_tool_call(
            {
                "name": "Retrieve_memory",
                "arguments": {"query": "fact", "top_k": "4"},
            }
        )
        self.assertIsNone(error)
        self.assertEqual(retrieve["arguments"]["top_k"], 4)

        summary, error = memory_utils.validate_tool_call(
            {
                "name": "Summary_context",
                "arguments": {"span": "3"},
            }
        )
        self.assertIsNone(error)
        self.assertEqual(summary["arguments"]["span"], "3")

        update, error = memory_utils.validate_tool_call(
            {
                "name": "Update_memory",
                "arguments": {"memory_id": "m1", "metadata": {"tag": "new"}},
            }
        )
        self.assertIsNone(error)
        self.assertIsNone(update["arguments"]["content"])

    def test_tool_validation_rejects_dangerous_or_empty_inputs(self):
        invalid_calls = [
            {"name": "Add_memory", "arguments": {"content": ""}},
            {"name": "Retrieve_memory", "arguments": {"query": "x", "top_k": 0}},
            {
                "name": "Retrieve_memory",
                "arguments": {"query": "x", "top_k": float("inf")},
            },
            {
                "name": "Delete_memory",
                "arguments": {"memory_id": "m1", "confirmation": "false"},
            },
            {"name": "Update_memory", "arguments": {"memory_id": "m1"}},
            {"name": "Clear_context", "arguments": None},
            {"name": "Summary_context", "arguments": None},
            {"name": "Summary_context", "arguments": {}},
            {"name": "Summary_context", "arguments": {"span": "0"}},
            {"name": "Summary_context", "arguments": {"span": "-2"}},
            {"name": "Summary_context", "arguments": {"span": "1-3"}},
            {"name": "Summary_context", "arguments": {"span": "4-2"}},
            {"name": "Summary_context", "arguments": {"span": "invalid"}},
            {"name": " Add_memory ", "arguments": {"content": "fact"}},
            "not-an-object",
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                _, error = memory_utils.validate_tool_call(call)
                self.assertIsNotNone(error)

    def test_parser_preserves_identical_repeated_calls(self):
        repeated_call = {
            "name": "Add_memory",
            "arguments": {"content": "same fact"},
        }
        text = (
            "<tool_call>" + json.dumps([repeated_call, repeated_call]) + "</tool_call>"
        )
        self.assertEqual(
            memory_utils.parse_tool_calls(text),
            [repeated_call, repeated_call],
        )

    def test_parser_preserves_multiple_close_only_and_nested_calls(self):
        first_call = {
            "name": "Add_memory",
            "arguments": {
                "content": "fact",
                "metadata": {"tags": ["one", "two]"]},
            },
        }
        second_call = {
            "name": "Retrieve_memory",
            "arguments": {"query": "fact"},
        }
        close_only_text = (
            json.dumps([first_call])
            + "</tool_call>\n"
            + json.dumps([second_call])
            + "</tool_call>"
        )
        standard_text = "<tool_call>" + json.dumps([first_call]) + "</tool_call>"

        self.assertEqual(
            memory_utils.parse_tool_calls(close_only_text),
            [first_call, second_call],
        )
        self.assertEqual(
            memory_utils.parse_tool_calls(standard_text),
            [first_call],
        )


class ToolEventStatsTest(unittest.TestCase):
    @staticmethod
    def finish_event(tool_name, status="success", result=None, arguments=None):
        return {
            "phase": "finish",
            "tool_name": tool_name,
            "status": status,
            "result": result or {"effect_applied": False},
            "arguments": arguments or {},
        }

    def test_event_stats_count_actual_calls_across_stages(self):
        events = [
            self.finish_event(
                "Add_memory",
                result={"effect_applied": True},
                arguments={"content": "important fact to remember " * 3},
            ),
            self.finish_event(
                "Add_memory",
                result={"effect_applied": True},
                arguments={"content": "second fact"},
            ),
            self.finish_event("Add_memory", status="error"),
            self.finish_event(
                "Summary_context",
                result={"effect_applied": True},
            ),
            self.finish_event(
                "Summary_context",
                result={"effect_applied": True},
            ),
            self.finish_event(
                "Retrieve_memory",
                result={
                    "effect_applied": True,
                    "retrieved_count": 1,
                    "items": [{"memory_id": "m1", "content": "important fact"}],
                },
            ),
            self.finish_event("Delete_memory", status="cancelled"),
            self.finish_event(
                "Update_memory",
                result={"effect_applied": False, "outcome": "not_found"},
            ),
            self.finish_event("Made_up_tool", status="unknown_tool"),
        ]

        usage = memory_reward.extract_tool_usage_stats([], events)
        self.assertEqual(
            usage,
            {
                "Summary_context": 2,
                "Clear_context": 0,
                "Retrieve_memory": 1,
                "Add_memory": 2,
                "Update_memory": 0,
                "Delete_memory": 0,
            },
        )

        attempts = memory_reward.extract_tool_attempt_stats(events)
        self.assertEqual(attempts["by_tool"]["Add_memory"]["attempted"], 3)
        self.assertEqual(attempts["by_tool"]["Add_memory"]["succeeded"], 2)
        self.assertEqual(attempts["by_tool"]["Add_memory"]["errored"], 1)
        self.assertEqual(attempts["by_tool"]["Delete_memory"]["cancelled"], 1)
        self.assertEqual(attempts["unknown_tool_calls"], 1)
        self.assertEqual(attempts["total_attempted"], len(events))

        class FakeMemoryManager:
            @staticmethod
            def count():
                return 2

        memory_stats = memory_reward.extract_memory_stats(
            [
                {"role": "tool", "content": "[retrieved memories]\n- important fact"},
                {"role": "assistant", "content": "I used it."},
            ],
            FakeMemoryManager(),
            events,
        )
        self.assertEqual(memory_stats["added_count"], 2)
        self.assertEqual(memory_stats["retrieved_count"], 1)
        self.assertEqual(memory_stats["deleted_count"], 0)
        self.assertTrue(memory_stats["used_retrieved_memory"])
        self.assertEqual(memory_stats["current_memory_count"], 2)

    def test_retrieval_use_survives_later_context_clear(self):
        events = [
            self.finish_event(
                "Retrieve_memory",
                result={
                    "effect_applied": True,
                    "retrieved_count": 1,
                    "used_by_following_response": True,
                    "items": [{"memory_id": "m1", "content": "important fact"}],
                },
            )
        ]

        class FakeMemoryManager:
            @staticmethod
            def count():
                return 1

        stats = memory_reward.extract_memory_stats(
            [],
            FakeMemoryManager(),
            events,
        )
        self.assertTrue(stats["used_retrieved_memory"])

    def test_legacy_text_fallback_remains_available(self):
        context_messages = [
            {
                "role": "assistant",
                "content": "Add_memory Add_memory Retrieve_memory",
            }
        ]
        self.assertEqual(
            memory_reward.extract_tool_usage_stats(context_messages),
            {
                "Summary_context": 0,
                "Clear_context": 0,
                "Retrieve_memory": 1,
                "Add_memory": 1,
                "Update_memory": 0,
                "Delete_memory": 0,
            },
        )


class RewardRoundSemanticsTest(unittest.TestCase):
    def test_stage3_termination_penalty_is_independent_of_tool_rounds(self):
        calculator = memory_reward.ThreeStageRewardCalculator(
            task_completion_weight=0.0,
            tool_efficiency_weight=0.0,
            context_management_weight=0.0,
            memory_management_weight=0.0,
            max_rounds_penalty=-1.0,
            min_reward=-10.0,
            max_reward=10.0,
        )
        common_arguments = {
            "task_score": 0.0,
            "tool_usage_stats": {},
            "context_stats": {},
            "memory_stats": {},
            "finished_at_round": 2,
            "max_rounds": 10,
            "found_answer": False,
        }

        total_without_stage3_cap, breakdown_without_stage3_cap = (
            calculator.calculate_total_reward(**common_arguments)
        )
        total_with_stage3_cap, breakdown_with_stage3_cap = (
            calculator.calculate_total_reward(
                **common_arguments,
                termination_finished_at_round=5,
                termination_max_rounds=5,
            )
        )

        self.assertEqual(total_without_stage3_cap, -1.0)
        self.assertNotIn(
            "max_rounds_penalty",
            breakdown_without_stage3_cap,
        )
        self.assertEqual(total_with_stage3_cap, -2.0)
        self.assertEqual(
            breakdown_with_stage3_cap["max_rounds_penalty"],
            -1.0,
        )

    def test_no_op_attempt_spam_reduces_tool_policy_score(self):
        calculator = memory_reward.ThreeStageRewardCalculator()
        effective_usage = {
            "Summary_context": 0,
            "Clear_context": 0,
            "Retrieve_memory": 0,
            "Add_memory": 1,
            "Update_memory": 0,
            "Delete_memory": 0,
        }

        normal_score = calculator._tool_policy(
            effective_usage,
            finished_at_round=5,
            max_rounds=10,
            attempted_call_count=1,
        )
        spam_score = calculator._tool_policy(
            effective_usage,
            finished_at_round=5,
            max_rounds=10,
            attempted_call_count=101,
        )

        self.assertLess(spam_score, normal_score)

    def test_trace_memory_judge_ignores_ineffective_add_text(self):
        class CapturingChatClient:
            def __init__(self):
                self.prompts = []

            def chat(self, messages, **_kwargs):
                self.prompts.append(messages[0]["content"])
                return "1"

        chat_client = CapturingChatClient()
        calculator = memory_reward.ThreeStageRewardCalculator(chat_client=chat_client)
        memory_stats = {
            "added_count": 1,
            "high_quality_additions": 0,
            "retrieved_count": 0,
            "updated_count": 0,
            "deleted_count": 0,
            "used_retrieved_memory": False,
            "redundant_storage": False,
            "_added_contents": ["valid effective memory"],
            "_retrieved_contents": [],
        }
        invalid_call_text = (
            '<tool_call>[{"name":"Add_memory","arguments":'
            '{"content":"invalid attempted memory"}}]</tool_call>'
        )

        calculator._memory_quality(
            memory_stats,
            question="What matters?",
            supporting_facts=[],
            context_messages=[{"role": "assistant", "content": invalid_call_text}],
        )

        self.assertEqual(len(chat_client.prompts), 1)
        self.assertIn("valid effective memory", chat_client.prompts[0])
        self.assertNotIn("invalid attempted memory", chat_client.prompts[0])

    def test_trace_context_reward_ignores_ineffective_tool_text(self):
        calculator = memory_reward.ThreeStageRewardCalculator()
        messages = [
            {
                "role": "assistant",
                "content": "invalid attempt mentioning Summary_context",
            }
        ]
        base_stats = {
            "token_usage_ratio": 1.0,
            "overflow_occurred": False,
            "preserved_user_query": True,
            "preserved_key_info": True,
        }

        ineffective_score = calculator._context_quality(
            {
                **base_stats,
                "effective_context_management_call": False,
            },
            messages,
        )
        effective_score = calculator._context_quality(
            {
                **base_stats,
                "effective_context_management_call": True,
            },
            messages,
        )

        self.assertLess(ineffective_score, effective_score)


class MemoryManagerTest(unittest.TestCase):
    def test_missing_update_does_not_request_embedding(self):
        manager = object.__new__(memory_store.MemoryManager)
        manager._store = memory_store.InMemoryVectorStore()
        manager.embed = mock.Mock(return_value=[1.0, 0.0])

        self.assertFalse(
            manager.update_memory(
                "missing-memory",
                content="replacement",
            )
        )
        manager.embed.assert_not_called()


class TrainingToolExecutionTest(unittest.TestCase):
    class FakeMemoryManager:
        def __init__(self):
            self.items = {
                "m1": SimpleNamespace(
                    memory_id="m1",
                    content="stored fact",
                    metadata={"kind": "fact"},
                )
            }
            self.delete_calls = 0

        def count(self):
            return len(self.items)

        def retrieve(self, _query, top_k, _metadata_filter):
            return list(self.items.values())[:top_k]

        def add_memory(self, memory_id, content, metadata):
            self.items[memory_id] = SimpleNamespace(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
            )
            return True

        def update_memory(self, memory_id, content, metadata):
            item = self.items.get(memory_id)
            if item is None:
                return False
            if content is not None:
                item.content = content
            item.metadata.update(metadata)
            return True

        def delete_memory(self, memory_id):
            self.delete_calls += 1
            return self.items.pop(memory_id, None) is not None

    class FakeChatClient:
        @staticmethod
        def chat(**_kwargs):
            return "0.9"

    @staticmethod
    def candidate_experience():
        return SimpleNamespace(
            eid=SimpleNamespace(step=0),
            info={"existing": "preserved"},
        )

    def make_workflow(self, memory_manager=None):
        workflow_class = training_module.AgeMemHotpotWorkflowTraining
        workflow = object.__new__(workflow_class)
        workflow.task = SimpleNamespace(batch_id=7, task_id=8)
        workflow.context_messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "please remember this"},
            {"role": "assistant", "content": "tool request"},
        ]
        workflow.memory_manager = memory_manager or self.FakeMemoryManager()
        workflow.chat_client = self.FakeChatClient()
        workflow.current_stage = 1
        workflow.current_round = 2
        workflow.current_step = 2
        workflow.current_turn_index = 0
        workflow.current_run_id = 3
        workflow.current_execution_id = "execution-test"
        workflow.tool_trace_recorder = ToolTraceRecorder(None, enabled=False)
        workflow.tool_trace_console = False
        workflow._tool_trace_events = []
        workflow.logger = mock.MagicMock()
        return workflow

    def test_six_tools_execute_in_order_and_link_to_experience(self):
        workflow = self.make_workflow()
        experience = self.candidate_experience()
        calls = [
            {
                "name": "Summary_context",
                "arguments": {"span": "all"},
            },
            {
                "name": "Clear_context",
                "arguments": {"criteria": "noise"},
            },
            {
                "name": "Retrieve_memory",
                "arguments": {"query": "fact", "top_k": 1},
            },
            {
                "name": "Add_memory",
                "arguments": {"content": "new fact"},
            },
            {
                "name": "Update_memory",
                "arguments": {"memory_id": "m1", "content": "updated fact"},
            },
            {
                "name": "Delete_memory",
                "arguments": {"memory_id": "m1", "confirmation": True},
            },
        ]

        workflow._apply_tools(calls, [experience])

        events = workflow._tool_trace_events
        self.assertEqual(
            [event["tool_name"] for event in events],
            [call["name"] for call in calls],
        )
        self.assertEqual([event["tool_index"] for event in events], list(range(6)))
        self.assertTrue(all(event["status"] == "success" for event in events))
        self.assertTrue(
            all(event["result"]["effect_applied"] is True for event in events)
        )
        self.assertEqual(events[0]["stage"], 1)
        self.assertEqual(events[0]["round"], 2)
        self.assertEqual(events[0]["execution_id"], "execution-test")
        self.assertEqual(experience.info["existing"], "preserved")
        self.assertEqual(len(experience.info["tool_call_ids"]), 6)
        self.assertEqual(
            set(experience.info["tool_call_ids"]),
            {event["call_id"] for event in events},
        )

    def test_six_tools_persist_paired_jsonl_records(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "six-tools.jsonl"
            workflow = self.make_workflow()
            workflow.tool_trace_recorder = ToolTraceRecorder(str(trace_path))
            calls = [
                {"name": "Summary_context", "arguments": {"span": "all"}},
                {"name": "Clear_context", "arguments": {"criteria": "noise"}},
                {
                    "name": "Retrieve_memory",
                    "arguments": {"query": "fact", "top_k": 1},
                },
                {"name": "Add_memory", "arguments": {"content": "new fact"}},
                {
                    "name": "Update_memory",
                    "arguments": {
                        "memory_id": "m1",
                        "content": "updated fact",
                    },
                },
                {
                    "name": "Delete_memory",
                    "arguments": {
                        "memory_id": "m1",
                        "confirmation": True,
                    },
                },
            ]

            workflow._apply_tools(calls, [])

            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 12)
            by_call_id = {}
            for record in records:
                by_call_id.setdefault(record["call_id"], []).append(record)
                self.assertEqual(record["schema_version"], 2)
                self.assertEqual(record["stage"], 1)
                self.assertEqual(record["round"], 2)
            self.assertEqual(len(by_call_id), 6)
            for paired_records in by_call_id.values():
                self.assertEqual(
                    [record["phase"] for record in paired_records],
                    ["start", "finish"],
                )
            finish_records = [
                record for record in records if record["phase"] == "finish"
            ]
            self.assertEqual(
                [record["tool_name"] for record in finish_records],
                [call["name"] for call in calls],
            )
            self.assertTrue(
                all(record["status"] == "success" for record in finish_records)
            )
            self.assertTrue(
                all(
                    record["result"]["effect_applied"] is True
                    for record in finish_records
                )
            )

    def test_identical_repeated_calls_each_execute_and_trace(self):
        workflow = self.make_workflow()
        repeated_call = {
            "name": "Add_memory",
            "arguments": {"content": "same fact"},
        }
        calls = memory_utils.parse_tool_calls(
            "<tool_call>" + json.dumps([repeated_call, repeated_call]) + "</tool_call>"
        )

        workflow._apply_tools(calls, [])

        self.assertEqual(len(workflow._tool_trace_events), 2)
        self.assertEqual(len(workflow.memory_manager.items), 3)
        self.assertEqual(
            len({event["call_id"] for event in workflow._tool_trace_events}),
            2,
        )

    def test_retrieval_is_marked_when_present_in_next_model_input(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "retrieval-usage.jsonl"
            workflow = self.make_workflow()
            workflow.tool_trace_recorder = ToolTraceRecorder(str(trace_path))
            workflow._apply_tools(
                [
                    {
                        "name": "Retrieve_memory",
                        "arguments": {"query": "fact", "top_k": 1},
                    }
                ],
                [],
            )
            event = workflow._tool_trace_events[0]
            self.assertNotIn(
                "used_by_following_response",
                event["result"],
            )

            workflow._mark_retrievals_used_by_next_response(workflow.context_messages)
            self.assertTrue(event["result"]["used_by_following_response"])

            persisted_events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["phase"] for record in persisted_events],
                ["start", "finish", "usage"],
            )
            self.assertEqual(
                {record["call_id"] for record in persisted_events},
                {event["call_id"]},
            )
            self.assertTrue(persisted_events[-1]["usage"]["used_by_following_response"])

            workflow.context_messages.clear()
            raw_stats = memory_reward.extract_memory_stats(
                workflow.context_messages,
                workflow.memory_manager,
                workflow._tool_trace_events,
            )
            persisted_stats = memory_reward.extract_memory_stats(
                workflow.context_messages,
                workflow.memory_manager,
                persisted_events,
            )
            self.assertTrue(raw_stats["used_retrieved_memory"])
            self.assertTrue(persisted_stats["used_retrieved_memory"])

    def test_tool_exception_is_traced_then_original_failure_stops_array(self):
        class FailingMemoryManager(self.FakeMemoryManager):
            def retrieve(self, _query, _top_k, _metadata_filter):
                raise RuntimeError("retrieval failed")

        manager = FailingMemoryManager()
        workflow = self.make_workflow(manager)
        calls = [
            {
                "name": "Retrieve_memory",
                "arguments": {"query": "fact"},
            },
            {
                "name": "Add_memory",
                "arguments": {"content": "must not execute"},
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            workflow._apply_tools(calls, [])

        self.assertEqual(len(workflow._tool_trace_events), 1)
        self.assertEqual(workflow._tool_trace_events[0]["status"], "error")
        self.assertFalse(
            any(item.content == "must not execute" for item in manager.items.values())
        )

    def test_string_false_delete_is_validation_error_not_deletion(self):
        manager = self.FakeMemoryManager()
        workflow = self.make_workflow(manager)
        workflow._apply_tools(
            [
                {
                    "name": "Delete_memory",
                    "arguments": {
                        "memory_id": "m1",
                        "confirmation": "false",
                    },
                }
            ],
            [],
        )
        self.assertIn("m1", manager.items)
        self.assertEqual(manager.delete_calls, 0)
        self.assertEqual(
            workflow._tool_trace_events[0]["status"],
            "validation_error",
        )

    def test_disk_truncation_does_not_change_in_memory_reward_event(self):
        with workspace_temp_directory() as temp_dir:
            trace_path = Path(temp_dir) / "calls.jsonl"
            workflow = self.make_workflow()
            workflow.tool_trace_recorder = ToolTraceRecorder(
                str(trace_path),
                max_string_chars=256,
            )
            content = "important fact to remember " * 20

            workflow._apply_tools(
                [
                    {
                        "name": "Add_memory",
                        "arguments": {"content": content},
                    }
                ],
                [],
            )

            internal_event = workflow._tool_trace_events[0]
            self.assertEqual(internal_event["arguments"]["content"], content)
            memory_stats = memory_reward.extract_memory_stats(
                workflow.context_messages,
                workflow.memory_manager,
                workflow._tool_trace_events,
            )
            self.assertEqual(memory_stats["added_count"], 1)
            self.assertEqual(memory_stats["high_quality_additions"], 1)

            persisted_records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            persisted_finish = next(
                record for record in persisted_records if record["phase"] == "finish"
            )
            self.assertTrue(persisted_finish["arguments"]["content"]["_truncated"])


if __name__ == "__main__":
    unittest.main()
