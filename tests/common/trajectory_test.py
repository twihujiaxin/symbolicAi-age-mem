import asyncio
import hashlib
import io
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentscope.message import Msg, ToolUseBlock
from agentscope.tool import ToolResponse

from AgeMem_code_agentscope.agent import AgeMem
from AgeMem_code_agentscope.memory import AgentScopeLongtermMemory
from AgeMem_code_agentscope.replay import main as replay_main
from AgeMem_code_agentscope.trajectory import (
    MemorySnapshotItem,
    ToolCallSnapshot,
    ToolResultSnapshot,
    TrajectoryRecorder,
    TrajectoryReplay,
    TrajectoryStep,
    TrajectoryValidationError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def workspace_temp_directory():
    """Use the workspace tmp directory to avoid restrictive Windows temp ACLs."""

    temp_root = REPOSITORY_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"trajectory-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def memory_item(memory_id="memory-1", content="fact-v1"):
    return MemorySnapshotItem(
        memory_id=memory_id,
        content=content,
        metadata={"kind": "fact"},
        embedding=[0.1, 0.2, 0.3],
    )


def build_step(
    timestep,
    *,
    before=None,
    after=None,
    task_id="task-1",
    rollout_id="rollout-1",
    done=False,
    env_reward=0.0,
):
    call = ToolCallSnapshot(
        id=f"call-{timestep}",
        name="test_action",
        input={"timestep": timestep},
    )
    result = ToolResultSnapshot(
        tool_call_id=call.id,
        name=call.name,
        content=[{"type": "text", "text": "ok"}],
        metadata={"success": True},
    )
    return TrajectoryStep(
        task_id=task_id,
        rollout_id=rollout_id,
        timestep=timestep,
        observation=f"observation-{timestep}",
        action_text=call.model_dump_json(),
        tool_calls=[call],
        tool_results=[result],
        memory_before=before or [],
        memory_after=after or [],
        env_reward=env_reward,
        done=done,
    )


class OfflineChatClient:
    def chat(self, messages, model_name="qwen-max"):
        return "offline deterministic response"


class OfflineFormatter:
    async def format(self, msgs):
        return msgs


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)

    async def __call__(self, prompt, tools=None):
        if not self.responses:
            raise AssertionError("scripted model received an unexpected call")
        return SimpleNamespace(content=self.responses.pop(0))


def deterministic_embed(text):
    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    return [float(value + 1) for value in digest[:16]]


class TrajectorySchemaTest(unittest.TestCase):
    def test_step_serialization_round_trip(self):
        item = memory_item()
        step = build_step(
            0,
            before=[],
            after=[item],
            done=True,
            env_reward=1.25,
        )

        decoded = TrajectoryStep.model_validate_json(step.to_json_line())

        self.assertEqual(decoded, step)
        self.assertEqual(decoded.env_reward, 1.25)
        self.assertTrue(decoded.done)

    def test_legacy_memory_snapshot_defaults_remain_replayable(self):
        step = build_step(0, after=[memory_item()], done=True).model_dump(mode="json")
        legacy_item = step["memory_after"][0]
        for field_name in (
            "version",
            "status",
            "created_at",
            "updated_at",
            "source_rollout_id",
            "source_step",
        ):
            legacy_item.pop(field_name)

        decoded = TrajectoryStep.model_validate(step)

        self.assertEqual(decoded.memory_after[0].version, 1)
        self.assertEqual(decoded.memory_after[0].status, "active")

    def test_rejects_non_finite_numeric_fields(self):
        with self.assertRaisesRegex(ValueError, "env_reward must be finite"):
            build_step(0, env_reward=float("nan"))
        with self.assertRaisesRegex(ValueError, "embedding values must be finite"):
            MemorySnapshotItem(
                memory_id="bad",
                content="bad",
                embedding=[float("inf")],
            )

    def test_corrupt_json_is_rejected_with_line_number(self):
        with workspace_temp_directory() as directory:
            path = directory / "corrupt.jsonl"
            path.write_text('{"schema_version": 1}\n{broken\n', encoding="utf-8")

            with self.assertRaisesRegex(
                TrajectoryValidationError, "schema validation failed at line 1"
            ):
                TrajectoryReplay.from_jsonl(path)

            path.write_text(build_step(0).to_json_line() + "\n{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(
                TrajectoryValidationError, "invalid JSON at line 2"
            ):
                TrajectoryReplay.from_jsonl(path)

    def test_missing_and_extra_fields_are_rejected(self):
        step = build_step(0).model_dump(mode="json")
        with workspace_temp_directory() as directory:
            missing_path = directory / "missing.jsonl"
            missing = dict(step)
            missing.pop("task_id")
            missing_path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TrajectoryValidationError, "task_id"):
                TrajectoryReplay.from_jsonl(missing_path)

            extra_path = directory / "extra.jsonl"
            extra = dict(step)
            extra["unexpected"] = True
            extra_path.write_text(json.dumps(extra) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TrajectoryValidationError, "unexpected"):
                TrajectoryReplay.from_jsonl(extra_path)

    def test_duplicate_timestep_is_rejected_by_recorder_and_loader(self):
        with workspace_temp_directory() as directory:
            path = directory / "duplicate.jsonl"
            recorder = TrajectoryRecorder(path)
            step = build_step(0)
            recorder.record(step)
            with self.assertRaisesRegex(
                TrajectoryValidationError, "duplicate trajectory timestep"
            ):
                recorder.record(step)

            path.write_text(
                step.to_json_line() + "\n" + step.to_json_line() + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                TrajectoryValidationError, "duplicate trajectory timestep"
            ):
                TrajectoryReplay.from_jsonl(path)


class TrajectoryReplayTest(unittest.TestCase):
    def write_valid_trajectory(self, path):
        item_v1 = memory_item(content="fact-v1")
        item_v2 = memory_item(content="fact-v2")
        recorder = TrajectoryRecorder(path)
        recorder.record(build_step(0, before=[], after=[item_v1]))
        recorder.record(build_step(1, before=[item_v1], after=[item_v2]))
        recorder.record(build_step(2, before=[item_v2], after=[item_v2], done=True))

    def test_replay_is_deterministic_and_queryable(self):
        with workspace_temp_directory() as directory:
            path = directory / "valid.jsonl"
            self.write_valid_trajectory(path)
            trajectory = TrajectoryReplay.from_jsonl(path)

            first = trajectory.replay(
                task_id="task-1",
                rollout_id="rollout-1",
                require_complete=True,
            )
            second = trajectory.replay(
                task_id="task-1",
                rollout_id="rollout-1",
                require_complete=True,
            )

            self.assertEqual(first, second)
            self.assertEqual(first.digest, second.digest)
            self.assertEqual(len(first.memory_states), 4)
            self.assertEqual(first.final_memory[0].content, "fact-v2")
            self.assertEqual(
                trajectory.query(
                    task_id="task-1",
                    rollout_id="rollout-1",
                    timestep=1,
                )[0].timestep,
                1,
            )
            self.assertEqual(len(trajectory.query(task_id="task-1")), 3)
            self.assertEqual(
                trajectory.available_rollouts(), [("task-1", "rollout-1")]
            )

    def test_replay_does_not_call_embedding_or_llm(self):
        with workspace_temp_directory() as directory:
            path = directory / "offline.jsonl"
            self.write_valid_trajectory(path)
            trajectory = TrajectoryReplay.from_jsonl(path)

            with mock.patch(
                "AgeMem_code_agentscope.memory.AgentScopeLongtermMemory.embed",
                side_effect=AssertionError("embedding must not be called"),
            ), mock.patch(
                "AgeMem_code_agentscope.src.llm_client.chat_client.chat",
                side_effect=AssertionError("LLM must not be called"),
            ):
                result = trajectory.replay(
                    task_id="task-1",
                    rollout_id="rollout-1",
                    require_complete=True,
                )

            self.assertEqual(result.final_memory[0].content, "fact-v2")

    def test_memory_discontinuity_and_non_contiguous_steps_are_rejected(self):
        item_v1 = memory_item(content="fact-v1")
        item_v2 = memory_item(content="fact-v2")

        discontinuous = TrajectoryReplay(
            [
                build_step(0, before=[], after=[item_v1]),
                build_step(1, before=[item_v2], after=[item_v2]),
            ]
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "discontinuity"):
            discontinuous.replay(task_id="task-1", rollout_id="rollout-1")

        non_contiguous = TrajectoryReplay(
            [
                build_step(0, before=[], after=[item_v1]),
                build_step(2, before=[item_v1], after=[item_v1]),
            ]
        )
        with self.assertRaisesRegex(TrajectoryValidationError, "contiguous"):
            non_contiguous.replay(task_id="task-1", rollout_id="rollout-1")

    def test_require_complete_rejects_unfinished_rollout(self):
        trajectory = TrajectoryReplay([build_step(0)])
        with self.assertRaisesRegex(TrajectoryValidationError, "incomplete"):
            trajectory.replay(
                task_id="task-1",
                rollout_id="rollout-1",
                require_complete=True,
            )

    def test_replay_cli_queries_and_replays_without_network(self):
        with workspace_temp_directory() as directory:
            path = directory / "cli.jsonl"
            self.write_valid_trajectory(path)

            query_output = io.StringIO()
            with redirect_stdout(query_output):
                query_code = replay_main(
                    [
                        str(path),
                        "--task-id",
                        "task-1",
                        "--rollout-id",
                        "rollout-1",
                        "--timestep",
                        "1",
                    ]
                )
            self.assertEqual(query_code, 0)
            self.assertEqual(json.loads(query_output.getvalue())[0]["timestep"], 1)

            replay_output = io.StringIO()
            with redirect_stdout(replay_output):
                replay_code = replay_main(
                    [
                        str(path),
                        "--task-id",
                        "task-1",
                        "--rollout-id",
                        "rollout-1",
                        "--replay",
                        "--require-complete",
                    ]
                )
            self.assertEqual(replay_code, 0)
            self.assertEqual(
                json.loads(replay_output.getvalue())["final_memory"][0]["content"],
                "fact-v2",
            )


class AgentTrajectoryHookTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_context = workspace_temp_directory()
        self.directory = self.temp_context.__enter__()
        self.path = self.directory / "agent.jsonl"
        self.memory = AgentScopeLongtermMemory(
            api_key="test",
            rollout_id="agent-rollout",
        )
        self.memory.embed = deterministic_embed
        self.chat_client = OfflineChatClient()
        self.recorder = TrajectoryRecorder(self.path)
        self.agent = AgeMem(
            name="AgeMem-M1-Test",
            sys_prompt="M1 trajectory test",
            model=object(),
            formatter=object(),
            memory=self.memory,
            chat_client=self.chat_client,
            trajectory_recorder=self.recorder,
            task_id="agent-task",
            rollout_id="agent-rollout",
        )
        self.agent.context_messages.append(
            Msg("user", "Remember and maintain the project codename.", "user")
        )

    async def asyncTearDown(self):
        self.memory.client.close()
        self.temp_context.__exit__(None, None, None)

    async def apply(self, name, input_data):
        tool_call = ToolUseBlock(
            type="tool_use",
            id=f"call-{uuid.uuid4().hex}",
            name=name,
            input=input_data,
        )
        self.agent.context_messages.append(
            Msg(self.agent.name, [tool_call], "assistant")
        )
        return await self.agent._apply_tool(tool_call)

    async def test_tool_hooks_capture_complete_replayable_trajectory(self):
        await self.apply(
            "add_memory",
            {
                "content": "Project codename is Atlas.",
                "metadata": {"scope": "m1"},
                "memory_type": "fact",
            },
        )
        items = await self.memory.get_memory()
        memory_id = items[0].memory_id

        await self.apply(
            "update_memory",
            {
                "memory_id": memory_id,
                "content": "Project codename is Borealis.",
                "metadata": {"status": "updated"},
            },
        )
        await self.apply(
            "retrieve_memory",
            {"query": "Project codename is Borealis.", "top_k": 1},
        )
        await self.apply(
            "delete_memory",
            {"memory_id": memory_id, "confirmation": True},
        )

        def environment_action(label: str) -> ToolResponse:
            """A deterministic environment action used by the M1 test."""

            return ToolResponse(
                content=[{"type": "text", "text": f"acted:{label}"}],
                metadata={"success": True, "env_reward": 1.5},
            )

        self.agent.toolkit.register_tool_function(environment_action)
        await self.apply("environment_action", {"label": "finish-task"})
        reply = await self.apply("generate_response", {"response": "Done."})
        self.assertEqual(reply.get_text_content(), "Done.")

        trajectory = TrajectoryReplay.from_jsonl(self.path)
        steps = trajectory.query(
            task_id="agent-task",
            rollout_id="agent-rollout",
        )
        self.assertEqual([step.timestep for step in steps], list(range(6)))
        self.assertEqual(
            [step.tool_calls[0].name for step in steps],
            [
                "add_memory",
                "update_memory",
                "retrieve_memory",
                "delete_memory",
                "environment_action",
                "generate_response",
            ],
        )
        self.assertEqual(steps[0].observation, "Remember and maintain the project codename.")
        self.assertEqual(steps[0].memory_before, [])
        self.assertEqual(len(steps[0].memory_after), 1)
        self.assertEqual(steps[1].memory_before[0].content, "Project codename is Atlas.")
        self.assertEqual(len(steps[1].memory_after), 2)
        self.assertEqual(
            [item.status for item in steps[1].memory_after],
            ["superseded", "active"],
        )
        self.assertEqual(steps[1].memory_after[0].content, "Project codename is Atlas.")
        self.assertEqual(steps[1].memory_after[1].content, "Project codename is Borealis.")
        self.assertEqual(steps[2].memory_before, steps[2].memory_after)
        self.assertEqual(len(steps[3].memory_before), 2)
        self.assertEqual(len(steps[3].memory_after), 3)
        self.assertEqual(
            [item.status for item in steps[3].memory_after],
            ["superseded", "superseded", "discarded"],
        )
        self.assertEqual(steps[4].env_reward, 1.5)
        self.assertFalse(steps[4].done)
        self.assertTrue(steps[5].done)
        self.assertEqual(steps[5].memory_before, steps[5].memory_after)
        self.assertIn("Done.", json.dumps(steps[5].tool_results[0].metadata))

        result = trajectory.replay(
            task_id="agent-task",
            rollout_id="agent-rollout",
            require_complete=True,
        )
        repeated = trajectory.replay(
            task_id="agent-task",
            rollout_id="agent-rollout",
            require_complete=True,
        )
        self.assertEqual(result, repeated)
        self.assertEqual(len(result.final_memory), 3)
        self.assertEqual(result.final_memory[-1].status, "discarded")

    async def test_injected_chat_client_is_lazy_and_recorder_is_optional(self):
        memory = AgentScopeLongtermMemory(api_key="test")
        memory.embed = deterministic_embed
        try:
            with mock.patch(
                "AgeMem_code_agentscope.agent.chat_client",
                side_effect=AssertionError("default client must not be constructed"),
            ):
                agent = AgeMem(
                    name="No-Recorder",
                    sys_prompt="test",
                    model=object(),
                    formatter=object(),
                    memory=memory,
                    chat_client=OfflineChatClient(),
                )
            self.assertIsNone(agent.trajectory_recorder)
            added = await agent.add_memory("No recorder side effect.")
            self.assertTrue(added.metadata["success"])
        finally:
            memory.client.close()

    async def test_full_offline_demo_loop_generates_complete_jsonl(self):
        demo_path = self.directory / "demo.jsonl"
        add_call = ToolUseBlock(
            type="tool_use",
            id="demo-add",
            name="add_memory",
            input={"content": "Demo fact.", "memory_type": "fact"},
        )
        finish_call = ToolUseBlock(
            type="tool_use",
            id="demo-finish",
            name="generate_response",
            input={"response": "Demo complete."},
        )
        model = ScriptedModel([[add_call], [finish_call]])
        demo_memory = AgentScopeLongtermMemory(
            api_key="test",
            rollout_id="demo-rollout",
        )
        demo_memory.embed = deterministic_embed
        try:
            agent = AgeMem(
                name="AgeMem-M1-Demo",
                sys_prompt="Offline M1 demo",
                model=model,
                formatter=OfflineFormatter(),
                memory=demo_memory,
                chat_client=OfflineChatClient(),
                trajectory_recorder=TrajectoryRecorder(demo_path),
                task_id="demo-task",
                rollout_id="demo-rollout",
            )
            reply = await agent.reply(Msg("user", "Remember the demo fact.", "user"))
            self.assertEqual(reply.get_text_content(), "Demo complete.")

            replay = TrajectoryReplay.from_jsonl(demo_path)
            result = replay.replay(
                task_id="demo-task",
                rollout_id="demo-rollout",
                require_complete=True,
            )
            self.assertEqual(len(result.steps), 2)
            self.assertEqual(
                [step.tool_calls[0].name for step in result.steps],
                ["add_memory", "generate_response"],
            )
            self.assertEqual(len(result.memory_states), 3)
            self.assertEqual(result.final_memory[0].content, "Demo fact.")
            self.assertTrue(result.done)
            self.assertEqual(model.responses, [])
        finally:
            demo_memory.client.close()


if __name__ == "__main__":
    unittest.main()
