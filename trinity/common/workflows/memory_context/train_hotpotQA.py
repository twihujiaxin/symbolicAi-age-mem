from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from trinity.common.action_event_contract import (
    prepare_experience_action_drafts,
    record_experience_action_result,
)
from trinity.common.auxiliary_provider import (
    AuxiliaryProviderUsageTracker,
    load_auxiliary_provider_config,
    resolve_auxiliary_provider_telemetry_path,
)
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.tool_trace import ToolTraceRecorder
from trinity.common.workflows.workflow import WORKFLOWS, MultiTurnWorkflow, Task
from trinity.utils.log import get_logger

from .memory_store import MemoryManager, chat_client
from .distractors import DISTRACTOR_SOURCES, resolve_stage2_distractors
from .workflow_prompt import (
    TOOL_CALL_SYS_PROMPT,
    SUMMARY_CONTEXT_SYS_PROMPT,
    TEXT_SIMILARITY_SYS_PROMPT,
    STAGE3_FINAL_ANSWER_NUDGE,
)
from .utils import (
    TOOL_SCHEMA as COMMON_TOOL_SCHEMA,
    TOOL_NAMES,
    DistractorGenerator as CommonDistractorGenerator,
    extract_score as common_extract_score,
    parse_answer as common_parse_answer,
    parse_tool_calls as common_parse_tool_calls,
    should_collect_intermediate_experience,
    should_emit_stage3_final_answer_nudge,
    validate_tool_call,
)

from ..memory_reward.my_reward import (
    ThreeStageRewardCalculator,
    extract_context_stats,
    extract_memory_stats,
    extract_tool_attempt_stats,
    extract_tool_usage_stats,
)
from ..memory_reward.reward_profiles import (
    HotpotAnswerScore,
    RewardProfileName,
    calculate_terminal_reward,
    load_workflow_reward_profile,
    score_hotpot_answer,
    terminal_task_score,
)
from .workflow_metrics import get_answer_llm_judge_score

# Use shared utility implementations to avoid train/eval drift.
TOOL_SCHEMA = COMMON_TOOL_SCHEMA
parse_tool_calls = common_parse_tool_calls
parse_answer = common_parse_answer
extract_score = common_extract_score
DistractorGenerator = CommonDistractorGenerator


@WORKFLOWS.register_module("AgeMem_hotpot_workflow_training")
class AgeMemHotpotWorkflowTraining(MultiTurnWorkflow):
    """
    HotpotQA 上的 AgeMem 三阶段训练工作流（论文方法的主入口）。

    Stage 1：把资料包装成闲聊，让模型学习构建长期记忆。
    Stage 2：清空短期上下文并注入干扰信息，让模型学习压缩/过滤上下文。
    Stage 3：提出正式问题，让模型联合使用 STM 与此前保留的 LTM 作答。
    """

    can_repeat: bool = True
    is_async: bool = True

    def __init__(
        self,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )

        self.logger = get_logger(name="AgeMem_workflow")

        # Task configuration
        self.task = task
        self.repeat_times = task.repeat_times
        self.workflow_args = task.workflow_args
        self.verbose: bool = bool(self.workflow_args.get("verbose_logging", False))
        self.reward_profile = load_workflow_reward_profile(self.workflow_args)
        self.auxiliary_provider_config = load_auxiliary_provider_config(
            self.workflow_args,
            required=self.reward_profile.is_terminal_only,
        )
        self.auxiliary_provider_telemetry_path = (
            resolve_auxiliary_provider_telemetry_path(self.workflow_args)
        )
        if (
            self.reward_profile.is_terminal_only
            and self.auxiliary_provider_telemetry_path is None
        ):
            raise ValueError(
                "terminal_only requires TRINITY_LOG_DIR, tool_trace_path, or "
                "auxiliary_provider_telemetry_path for durable provider telemetry"
            )
        self.auxiliary_provider_usage = AuxiliaryProviderUsageTracker(
            self.auxiliary_provider_config,
            telemetry_path=self.auxiliary_provider_telemetry_path,
        )

        # STM（context_messages）的容量与自动摘要阈值。
        self.max_context_tokens = self.workflow_args.get("max_context_tokens", 32768)
        self.auto_summary_token_threshold = self.workflow_args.get(
            "auto_summary_threshold", 0.8
        )
        self.max_tool_rounds_per_turn = self.workflow_args.get(
            "max_tool_rounds_per_turn", 3
        )

        # Multi-stage configuration.
        self.stage2_distractor_messages = self.workflow_args.get(
            "stage2_distractor_messages", 5
        )
        self.stage2_distractor_source = (
            str(self.workflow_args.get("stage2_distractor_source", "provider"))
            .strip()
            .lower()
        )
        if self.stage2_distractor_source not in DISTRACTOR_SOURCES:
            allowed = ", ".join(sorted(DISTRACTOR_SOURCES))
            raise ValueError(f"stage2_distractor_source must be one of: {allowed}")
        if (
            self.reward_profile.is_terminal_only
            and self.stage2_distractor_source == "provider"
        ):
            raise ValueError(
                "terminal_only requires a fixed or task-persisted Stage-2 "
                "distractor source"
            )
        self.stage3_max_rounds = self.workflow_args.get("stage3_max_rounds", 5)
        self.stage3_require_final_answer = bool(
            self.workflow_args.get("stage3_require_final_answer", False)
        )
        self.stage1_max_rounds = self.workflow_args.get("stage1_max_rounds", 5)
        self.stage2_max_rounds = self.workflow_args.get("stage2_max_rounds", 5)
        self.tool_reward_stats_source = (
            str(self.workflow_args.get("tool_reward_stats_source", "legacy"))
            .strip()
            .lower()
        )
        if self.tool_reward_stats_source not in {"legacy", "trace"}:
            self.logger.warning(
                "Unknown tool_reward_stats_source=%r; falling back to legacy",
                self.tool_reward_stats_source,
            )
            self.tool_reward_stats_source = "legacy"

        # memory_manager 是 LTM；chat_client 是摘要、相似度判断、Judge 等辅助调用。
        self.memory_manager = MemoryManager(
            embedding_model=self.auxiliary_provider_config.embedding_model,
            embedding_dim=self.auxiliary_provider_config.embedding_dimensions,
            base_url=self.auxiliary_provider_config.base_url,
            usage_tracker=self.auxiliary_provider_usage,
        )
        self.chat_client = chat_client(
            base_url=self.auxiliary_provider_config.base_url,
            default_model=self.auxiliary_provider_config.chat_model,
            usage_tracker=self.auxiliary_provider_usage,
        )
        self.distractor_generator = DistractorGenerator(self.chat_client)

        # context_messages 是 STM。它与 LTM 分开维护，因此可以只清空当前上下文。
        self.context_messages: List[Dict] = []
        self.current_turn: int = 0
        self.final_reward: float = 0.0
        self.current_stage: int = 0
        self.current_round: int = 0
        self.current_step: int = 0
        self.current_turn_index: Optional[int] = None
        self.current_run_id: int = 0
        self.run_id_base: int = 0
        self.current_execution_id: str = ""
        self._model_round_count: int = 0
        self._stage3_round_count: int = 0
        self._last_answer_score: Optional[HotpotAnswerScore] = None

        # 完整工具轨迹单独写 JSONL；Experience.info 只保存轻量关联 ID。
        self.tool_trace_recorder = ToolTraceRecorder.from_workflow_args(
            self.workflow_args
        )
        self.tool_trace_console = bool(
            self.workflow_args.get("tool_trace_console", False)
        )
        self._tool_trace_events: List[Dict[str, Any]] = []
        self._last_tool_result: Dict[str, Any] = {}

        # Question and expected answer
        self.question: Optional[str] = None
        self.expected_answer: Optional[str] = None

        # Facts and context
        self.facts: Optional[dict] = None
        self.context: Optional[dict] = None

        self.sys_prompt = TOOL_CALL_SYS_PROMPT.format(tools=json.dumps(TOOL_SCHEMA))

    @property
    def asynchronous(self):
        return True

    @property
    def repeatable(self):
        return True

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base

    def _append_context(self, role: str, content: str):
        """Add a message to the context."""
        self.context_messages.append({"role": role, "content": content})

    def _should_autosummarize(self) -> bool:
        """Check if context should be auto-summarized based on token threshold."""
        total_chars = sum(len(m.get("content", "")) for m in self.context_messages)
        approx_tokens = total_chars / 4
        return (
            approx_tokens > self.max_context_tokens * self.auto_summary_token_threshold
        )

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> Optional[str]:
        """执行模型输出的工具调用，并直接修改 STM 或 LTM。

        这是“工具协议”落到真实状态变化的汇合点：
        Summary/Clear/Retrieve 操作当前上下文，Add/Update/Delete 操作长期存储。
        """
        reply_note: Optional[str] = None

        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {})
            self._last_tool_result = {
                "effect_applied": False,
                "outcome": "not_executed",
            }

            # self.logger.info(f"Applying tool: {name} with args: {args}")

            if name == "Summary_context":
                # SUMMARY：用辅助 LLM 把选中的多条 STM 消息压缩成一条 tool 消息。
                span = args.get("span", "all")
                preserve_user_query = args.get("preserve_user_query", False)

                # Find messages to summarize and their indices
                messages_to_summarize = []
                indices_to_replace = []

                # Filter out system messages for summarization
                non_system_messages = []
                for i, m in enumerate(self.context_messages):
                    if m.get("role") != "system":
                        non_system_messages.append((i, m))

                # Determine which messages to summarize based on span
                if span == "all":
                    # Summarize all non-system messages
                    messages_to_summarize = [(i, m) for i, m in non_system_messages]
                    indices_to_replace = [i for i, m in non_system_messages]
                else:
                    # Validation guarantees a positive integer here.
                    n = int(span)
                    messages_to_summarize = non_system_messages[-n:]
                    indices_to_replace = [i for i, m in messages_to_summarize]

                # Preserve user query if requested
                if preserve_user_query:
                    # Find user messages and exclude them from summarization
                    user_indices = [
                        i for i, m in messages_to_summarize if m.get("role") == "user"
                    ]
                    indices_to_replace = [
                        i for i in indices_to_replace if i not in user_indices
                    ]
                    messages_to_summarize = [
                        (i, m)
                        for i, m in messages_to_summarize
                        if i not in user_indices
                    ]

                # Generate summary from selected messages
                if messages_to_summarize:
                    conversation_text = "\n".join(
                        [
                            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                            for i, m in messages_to_summarize
                        ]
                    )
                    summary = self.chat_client.chat(
                        messages=[
                            {
                                "role": "user",
                                "content": SUMMARY_CONTEXT_SYS_PROMPT.format(
                                    conversation_text=conversation_text
                                ),
                            }
                        ],
                        model_name=getattr(
                            getattr(self, "auxiliary_provider_config", None),
                            "chat_model",
                            "qwen-max",
                        ),
                    )

                    # Replace the original messages with summary
                    # Sort indices in descending order to avoid index shifting issues
                    indices_to_replace.sort(reverse=True)

                    for idx in indices_to_replace:
                        # Remove the original message
                        self.context_messages.pop(idx)

                    # Insert summary at the position of the first removed message
                    if indices_to_replace:
                        insert_position = min(indices_to_replace)
                        self.context_messages.insert(
                            insert_position,
                            {
                                "role": "tool",
                                "content": f"[summary of {len(messages_to_summarize)} messages]\n{summary}",
                            },
                        )

                    reply_note = "summary_context_applied:success"
                    self._last_tool_result = {
                        "effect_applied": True,
                        "outcome": "summarized",
                        "summarized_count": len(messages_to_summarize),
                        "summary": summary,
                    }

                    # self.logger.info(f"Summarized {len(messages_to_summarize)} messages and replaced them with summary")
                else:
                    reply_note = "summary_context_applied:no_messages_to_summarize"
                    self._last_tool_result = {
                        "effect_applied": False,
                        "outcome": "no_messages",
                        "summarized_count": 0,
                    }
                    # self.logger.info("No messages to summarize")
                result_text = f"[context tool result]\n{reply_note}"
                self._append_context("tool", result_text)
                self._last_tool_result["result_text"] = result_text

            elif name == "Clear_context":
                # FILTER：逐条判断消息与删除条件的语义相似度，命中阈值则移除。
                criteria = args.get("criteria", "")
                # preserve_user_query = args.get("preserve_user_query", True)
                # preserve_system_messages = args.get("preserve_system_messages", True)

                filtered_messages = []
                removed_count = 0
                for m in self.context_messages:
                    # # Always preserve user query if requested
                    # if preserve_user_query and m.get("role") == "user":
                    #     should_keep = True
                    # # Always preserve system messages if requested
                    # elif preserve_system_messages and m.get("role") == "system":
                    #     should_keep = True
                    # # Check criteria for other messages
                    # elif criteria and criteria in m.get("content", ""):
                    #     should_keep = False
                    if m.get("role") == "system":
                        filtered_messages.append(m)
                        continue

                    if criteria:
                        similarity_text = self.chat_client.chat(
                            messages=[
                                {
                                    "role": "user",
                                    "content": TEXT_SIMILARITY_SYS_PROMPT.format(
                                        text1=criteria, text2=m.get("content", "")
                                    ),
                                }
                            ],
                            model_name=getattr(
                                getattr(self, "auxiliary_provider_config", None),
                                "chat_model",
                                "qwen-max",
                            ),
                        )

                        similarity_score = extract_score(similarity_text, default=0.0)
                        if similarity_score >= 0.6:
                            removed_count += 1
                            continue

                    # if criteria and criteria.lower() in m.get("content", "").lower():
                    #     removed_count += 1
                    #     continue

                    filtered_messages.append(m)

                self.context_messages = filtered_messages
                reply_note = (
                    f"clear_context_applied:success:removed_{removed_count}_messages"
                )
                result_text = f"[context tool result]\n{reply_note}"
                self._append_context("tool", result_text)
                self._last_tool_result = {
                    "effect_applied": removed_count > 0,
                    "outcome": "cleared",
                    "removed_count": removed_count,
                    "result_text": result_text,
                }

            elif name == "Retrieve_memory":
                # RETRIEVE：从 LTM 取回内容后写入 STM，长期存储本身不发生变化。
                query = args.get("query", "")
                top_k = int(args.get("top_k", 3))
                metadata_filter = args.get("metadata_filter", {})

                items = self.memory_manager.retrieve(query, top_k, metadata_filter)
                retrieved_block = "\n".join(
                    f"- {it.content} (Memory ID: {it.memory_id})" for it in items
                )
                if retrieved_block:
                    result_text = f"[retrieved memories]\n{retrieved_block}"
                else:
                    result_text = "[no related memories found]"
                self._append_context("tool", result_text)
                self._last_tool_result = {
                    "effect_applied": bool(items),
                    "outcome": "retrieved" if items else "no_matches",
                    "retrieved_count": len(items),
                    "items": [
                        {
                            "memory_id": item.memory_id,
                            "content": item.content,
                            "metadata": dict(item.metadata or {}),
                        }
                        for item in items
                    ],
                    "result_text": result_text,
                }

            elif name == "Add_memory":
                # ADD：为新记忆生成唯一 ID，并记录产生它的训练阶段。
                content = args.get("content", "")
                metadata = args.get("metadata", {}) or {}
                memory_type = args.get("memory_type", "general")

                # Add memory_type to metadata if provided
                if memory_type:
                    metadata["type"] = memory_type
                metadata["stage"] = str(self.current_stage)

                mem_id = str(uuid.uuid4())
                stored = self.memory_manager.add_memory(
                    mem_id,
                    content,
                    metadata,
                    source_step=self.current_step,
                )
                reply_note = f"memory_added:{mem_id}"
                result_text = f"[memory tool result]\n{reply_note}"
                self._append_context("tool", result_text)
                self._last_tool_result = {
                    "effect_applied": bool(stored),
                    "outcome": "added" if stored else "not_added",
                    "memory_id": mem_id,
                    "result_text": result_text,
                }

            elif name == "Update_memory":
                mem_id = args.get("memory_id", "")
                content = args.get("content")
                metadata = args.get("metadata", {})

                ok = self.memory_manager.update_memory(
                    mem_id,
                    content,
                    metadata,
                    source_step=self.current_step,
                )
                reply_note = f"memory_updated:{ok}"
                result_text = f"[memory tool result]\n{reply_note}"
                self._append_context("tool", result_text)
                self._last_tool_result = {
                    "effect_applied": ok,
                    "outcome": "updated" if ok else "not_found",
                    "memory_id": mem_id,
                    "result_text": result_text,
                }

            elif name == "Delete_memory":
                mem_id = args.get("memory_id", "")
                confirmation = args.get("confirmation", False)

                if confirmation:
                    ok = self.memory_manager.delete_memory(
                        mem_id,
                        source_step=self.current_step,
                    )
                    reply_note = f"memory_deleted:{ok}"
                    outcome = "deleted" if ok else "not_found"
                else:
                    reply_note = "memory_deletion_cancelled:confirmation_required"
                    ok = False
                    outcome = "cancelled"
                result_text = f"[memory tool result]\n{reply_note}"
                self._append_context("tool", result_text)
                self._last_tool_result = {
                    "effect_applied": ok,
                    "outcome": outcome,
                    "memory_id": mem_id,
                    "result_text": result_text,
                }

        return reply_note

    def _tool_trace_context(self) -> Dict[str, Any]:
        """Build the stable identity shared by all records for one call."""
        batch_id = getattr(self.task, "batch_id", "")
        task_id = getattr(self.task, "task_id", "")
        return {
            "batch_id": batch_id,
            "task_id": task_id,
            "run_id": self.current_run_id,
            "rollout_id": f"{batch_id}/{task_id}/{self.current_run_id}",
            "execution_id": self.current_execution_id,
            "stage": self.current_stage,
            "round": self.current_round,
            "step": self.current_step,
            "turn": self.current_turn_index,
        }

    def _tool_state_summary(self) -> Dict[str, Any]:
        """Return a compact state summary without clients, embeddings, or keys."""
        role_counts: Dict[str, int] = {}
        total_chars = 0
        for message in self.context_messages:
            role = str(message.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            total_chars += len(str(message.get("content", "")))

        memory_count = None
        try:
            memory_count = self.memory_manager.count()
        except (AttributeError, TypeError):
            pass

        return {
            "context_message_count": len(self.context_messages),
            "context_chars": total_chars,
            "context_roles": role_counts,
            "memory_count": memory_count,
        }

    def _annotate_experiences(
        self,
        experiences: List[Experience],
        *,
        stage: int,
        round_index: int,
        step_index: int,
    ) -> None:
        """Attach only compact trace linkage to candidate Experiences."""
        for experience in experiences:
            info = dict(experience.info or {})
            info.update(
                {
                    "trace_execution_id": self.current_execution_id,
                    "trace_stage": stage,
                    "trace_round": round_index,
                    "trace_step": step_index,
                }
            )
            experience.info = info
            prepare_experience_action_drafts(
                experience,
                stage_id=stage,
                timestep=step_index,
                assistant_turn_id=max(0, self._model_round_count - 1),
            )

    @staticmethod
    def _attach_tool_call_id(
        experiences: List[Experience],
        call_id: str,
    ) -> None:
        for experience in experiences:
            info = dict(experience.info or {})
            call_ids = list(info.get("tool_call_ids", []))
            call_ids.append(call_id)
            info["tool_call_ids"] = call_ids
            experience.info = info

    def _finish_tool_trace(
        self,
        *,
        call_id: str,
        started_at: float,
        trace_context: Dict[str, Any],
        tool_name: str,
        tool_index: int,
        arguments: Any,
        status: str,
        result: Dict[str, Any],
        experiences: List[Experience],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace_error_message = None
        try:
            self.tool_trace_recorder.record_finish(
                call_id=call_id,
                started_at=started_at,
                context=trace_context,
                tool_name=tool_name,
                tool_index=tool_index,
                arguments=arguments,
                status=status,
                result=result,
                state_after=self._tool_state_summary(),
                error=error,
            )
        except Exception as trace_error:
            # Trace failures must never change tool execution semantics.
            trace_error_message = f"{type(trace_error).__name__}: {trace_error}"
            try:
                self.logger.warning(
                    "Unable to serialize tool trace call %s: %s",
                    call_id,
                    trace_error,
                )
            except Exception:
                pass

        # Reward/statistics use the original execution data. Only the JSONL
        # copy returned by ToolTraceRecorder is redacted and truncated.
        finish_event = {
            "phase": "finish",
            "call_id": call_id,
            "tool_name": tool_name,
            "tool_index": tool_index,
            "status": status,
            "arguments": arguments,
            "result": result,
            "error": error,
            **trace_context,
        }
        if trace_error_message:
            finish_event["trace_error"] = trace_error_message
        self._tool_trace_events.append(finish_event)
        self._attach_tool_call_id(experiences, call_id)
        record_experience_action_result(
            experiences,
            action_index_in_turn=tool_index,
            trace_call_id=call_id,
            action_type=tool_name,
            status=status,
            result=result,
            error=error,
        )

        if self.tool_trace_console:
            try:
                self.logger.info(
                    "Tool trace: execution=%s stage=%s round=%s index=%s "
                    "tool=%s status=%s call_id=%s",
                    self.current_execution_id,
                    self.current_stage,
                    self.current_round,
                    tool_index,
                    tool_name,
                    status,
                    call_id,
                )
            except Exception:
                pass
        return finish_event

    def _apply_tools(
        self,
        tool_calls: List[Dict],
        experiences: Optional[List[Experience]] = None,
    ) -> Optional[str]:
        """Validate, trace, and execute every parsed call in its original order."""
        experiences = experiences or []
        reply_note: Optional[str] = None

        for tool_index, raw_call in enumerate(tool_calls):
            raw_name = (
                raw_call.get("name")
                if isinstance(raw_call, dict)
                else "<invalid_tool_call>"
            )
            tool_name = raw_name if isinstance(raw_name, str) else str(raw_name)
            raw_arguments = (
                raw_call.get("arguments", {})
                if isinstance(raw_call, dict)
                else raw_call
            )
            trace_context = self._tool_trace_context()
            state_before = self._tool_state_summary()
            try:
                call_id, started_at = self.tool_trace_recorder.record_start(
                    context=trace_context,
                    tool_name=tool_name,
                    tool_index=tool_index,
                    arguments=raw_arguments,
                    state_before=state_before,
                )
            except Exception as trace_error:
                call_id = str(uuid.uuid4())
                started_at = time.perf_counter()
                try:
                    self.logger.warning(
                        "Unable to start tool trace call %s: %s",
                        call_id,
                        trace_error,
                    )
                except Exception:
                    pass

            normalized_call, validation_error = validate_tool_call(raw_call)
            if validation_error:
                normalized_name = (
                    normalized_call.get("name")
                    if isinstance(normalized_call, dict)
                    else tool_name
                )
                status = (
                    "unknown_tool"
                    if isinstance(normalized_call, dict)
                    and normalized_name not in TOOL_NAMES
                    else "validation_error"
                )
                self._finish_tool_trace(
                    call_id=call_id,
                    started_at=started_at,
                    trace_context=trace_context,
                    tool_name=tool_name,
                    tool_index=tool_index,
                    arguments=raw_arguments,
                    status=status,
                    result={
                        "effect_applied": False,
                        "validation_error": validation_error,
                    },
                    experiences=experiences,
                )
                continue

            arguments = normalized_call["arguments"]
            try:
                call_reply_note = self._execute_tool_calls([normalized_call])
                if call_reply_note is not None:
                    reply_note = call_reply_note
                result = dict(self._last_tool_result)
                status = (
                    "cancelled"
                    if tool_name == "Delete_memory"
                    and arguments.get("confirmation") is not True
                    else "success"
                )
                finish_event = self._finish_tool_trace(
                    call_id=call_id,
                    started_at=started_at,
                    trace_context=trace_context,
                    tool_name=tool_name,
                    tool_index=tool_index,
                    arguments=raw_arguments,
                    status=status,
                    result=result,
                    experiences=experiences,
                )
                if (
                    tool_name == "Retrieve_memory"
                    and result.get("effect_applied") is True
                    and self.context_messages
                    and self.context_messages[-1].get("role") == "tool"
                ):
                    # Keep only an in-memory object reference. It lets the
                    # reward path tell whether this exact retrieval result was
                    # still present in the next model input, even if a later
                    # stage clears the STM.
                    finish_event["_retrieval_context_message"] = self.context_messages[
                        -1
                    ]
            except Exception as exc:
                self._finish_tool_trace(
                    call_id=call_id,
                    started_at=started_at,
                    trace_context=trace_context,
                    tool_name=tool_name,
                    tool_index=tool_index,
                    arguments=raw_arguments,
                    status="error",
                    result={"effect_applied": False},
                    experiences=experiences,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

        return reply_note

    def reset_per_run(self, *, rollout_id: Optional[str] = None):
        """开始一条独立 rollout：同时清空 STM 和 LTM，防止轨迹之间信息泄漏。"""
        self.context_messages.clear()
        if rollout_id is None:
            self.memory_manager.clear()
        else:
            self.memory_manager.bind_rollout(rollout_id, reset=True)
        self.final_reward = -0.1
        self.current_stage = 0
        self.current_round = 0
        self.current_step = 0
        self.current_turn_index = None
        self.current_execution_id = ""
        self._model_round_count = 0
        self._stage3_round_count = 0
        self._last_answer_score = None
        self._tool_trace_events.clear()
        self._last_tool_result = {}
        provider_usage = getattr(self, "auxiliary_provider_usage", None)
        if provider_usage is not None:
            provider_usage.reset()

    async def run_async(self) -> List[Experience]:
        """Initialize the workflow and start the multi-turn, multi-step process."""
        rollout_n = self.repeat_times
        try:
            # self.logger.info("=== Starting Multi-Turn Multi-Step Context Memory Workflow ===")

            # Extract question and expected answer
            self.question = self.task.raw_task.get(self.task.format_args.prompt_key)
            self.expected_answer = self.task.raw_task.get(
                self.task.format_args.response_key
            )

            self.context_info = self.task.raw_task.get("context")
            self.supporting_facts = self.task.raw_task.get("supporting_facts")

            # Verify required fields exist.
            if not self.question:
                self.logger.error("Question is missing from task data")
                return []

            if not self.context_info:
                self.logger.error("Context info is missing from task data")
                return []

            if not isinstance(self.context_info, dict):
                self.logger.error(
                    f"Context info should be a dict, got {type(self.context_info)}"
                )
                return []

            return await self.inference_samples(rollout_n)

        except Exception as e:
            self.logger.error(
                "Error in run (%s)",
                type(e).__name__[:128],
            )
            raise

    def _mark_retrievals_used_by_next_response(self, messages) -> None:
        """Mark retrieval results that are present in the next model input."""
        for event in self._tool_trace_events:
            if event.get("tool_name") != "Retrieve_memory":
                continue
            result = event.get("result")
            if (
                not isinstance(result, dict)
                or result.get("used_by_following_response") is True
            ):
                continue
            retrieved_message = event.get("_retrieval_context_message")
            if retrieved_message is not None and any(
                message is retrieved_message for message in messages
            ):
                result["used_by_following_response"] = True
                source_context = {
                    key: event.get(key)
                    for key in (
                        "batch_id",
                        "task_id",
                        "run_id",
                        "rollout_id",
                        "execution_id",
                        "stage",
                        "round",
                        "step",
                        "turn",
                    )
                }
                try:
                    usage_event = self.tool_trace_recorder.record_usage(
                        call_id=event["call_id"],
                        context=source_context,
                        tool_name="Retrieve_memory",
                        tool_index=event["tool_index"],
                        usage={
                            "used_by_following_response": True,
                            "following_response_context": (self._tool_trace_context()),
                        },
                    )
                    event["usage_record_id"] = usage_event["record_id"]
                except Exception as trace_error:
                    event["trace_usage_error"] = (
                        f"{type(trace_error).__name__}: {trace_error}"
                    )
                    try:
                        self.logger.warning(
                            "Unable to record retrieval usage for call %s: %s",
                            event["call_id"],
                            trace_error,
                        )
                    except Exception:
                        pass

    async def get_model_response_text(self, messages):
        """Get model response text."""
        self._mark_retrievals_used_by_next_response(messages)
        responses = await self.model.chat_async(
            messages,
            n=1,
            record_action_metadata=True,
        )
        self._model_round_count += 1
        return responses[0].response_text

    async def inference_samples(self, rollout_num: int) -> List[Experience]:
        """对同一道题采样多条三阶段轨迹，并为每条轨迹计算终局奖励。"""

        reward_calculator = None
        if self.reward_profile.name is RewardProfileName.E2_AGEMEM_HEURISTIC:
            reward_calculator = ThreeStageRewardCalculator(
                **self.reward_profile.heuristic_calculator_kwargs(),
                chat_client=self.chat_client,
            )

        experience_list = []

        for i in range(rollout_num):
            self.current_run_id = i + self.run_id_base
            batch_id = getattr(self.task, "batch_id", "")
            task_id = getattr(self.task, "task_id", "")
            memory_rollout_id = f"{batch_id}/{task_id}/{self.current_run_id}"
            self.reset_per_run(rollout_id=memory_rollout_id)
            self.current_execution_id = str(uuid.uuid4())
            self.auxiliary_provider_usage.start_rollout(
                task_id=task_id,
                rollout_id=memory_rollout_id,
                execution_id=self.current_execution_id,
                is_eval=bool(getattr(self.task, "is_eval", False)),
            )
            self._append_context("system", self.sys_prompt)

            all_stage_experiences: List[Experience] = []

            # Stage 1（LTM 构建）：从 HotpotQA context 中识别并保存未来有用的事实。
            self.current_stage = 1
            if self.verbose:
                self.logger.info(f"Rollout {i} - Stage 1: Casual chat based on context")

            stage1_exps = await self._run_stage1_casual_chat()
            self.logger.info(
                f"Rollout {i} - Stage 1 returned {len(stage1_exps)} experiences"
            )
            all_stage_experiences.extend(stage1_exps)

            # Stage 2（STM 控制）：故意清空 STM，但保留 Stage 1 建立的 LTM。
            self.current_stage = 2
            if self.verbose:
                self.logger.info(
                    f"Rollout {i} - Stage 2: Distractor messages injection"
                )

            self.context_messages.clear()
            self._append_context("system", self.sys_prompt)

            stage2_exps = await self._run_stage2_distractor_injection()
            self.logger.info(
                f"Rollout {i} - Stage 2 returned {len(stage2_exps)} experiences"
            )
            all_stage_experiences.extend(stage2_exps)

            # Stage 3（联合使用）：正式提问，模型需要主动 Retrieve 才能拿回 Stage 1 信息。
            self.current_stage = 3
            if self.verbose:
                self.logger.info(f"Rollout {i} - Stage 3: Formal Q&A")

            stage3_exps, task_score, found_answer = await self._run_stage3_formal_qa()
            self.logger.info(
                f"Rollout {i} - Stage 3 returned {len(stage3_exps)} experiences"
            )
            all_stage_experiences.extend(stage3_exps)

            # Ensure at least some experiences were collected.
            if not all_stage_experiences:
                self.logger.error(
                    f"Rollout {i} - No experiences collected from any stage! This will cause timeout."
                )
                # Add a dummy experience to avoid total failure.
                # Better would be to check why no experiences were collected.

            # 从最终上下文中统计工具、STM、LTM 行为，再与答案得分合成轨迹奖励。
            legacy_tool_usage_stats = extract_tool_usage_stats(self.context_messages)
            trace_tool_usage_stats = extract_tool_usage_stats(
                self.context_messages,
                self._tool_trace_events,
            )
            tool_attempt_stats = extract_tool_attempt_stats(self._tool_trace_events)
            tool_attempt_summary = {
                "total_attempted": tool_attempt_stats["total_attempted"],
                "total_errored": sum(
                    stats["errored"] for stats in tool_attempt_stats["by_tool"].values()
                ),
                "total_cancelled": sum(
                    stats["cancelled"]
                    for stats in tool_attempt_stats["by_tool"].values()
                ),
                "total_validation_error": sum(
                    stats["validation_error"]
                    for stats in tool_attempt_stats["by_tool"].values()
                ),
                "unknown_tool_calls": tool_attempt_stats["unknown_tool_calls"],
            }
            context_stats = extract_context_stats(
                self.context_messages,
                self.max_context_tokens,
                target_question=self.question,
            )
            if self.tool_reward_stats_source == "trace":
                context_stats["effective_context_management_call"] = (
                    trace_tool_usage_stats["Summary_context"] > 0
                    or trace_tool_usage_stats["Clear_context"] > 0
                )
            legacy_memory_stats = extract_memory_stats(
                self.context_messages,
                self.memory_manager,
            )
            trace_memory_stats = extract_memory_stats(
                self.context_messages,
                self.memory_manager,
                self._tool_trace_events,
            )
            if self.tool_reward_stats_source == "trace":
                reward_tool_usage_stats = trace_tool_usage_stats
                reward_memory_stats = trace_memory_stats
                reward_finished_at_round = max(1, self._model_round_count)
                reward_max_rounds = max(
                    1,
                    self.stage1_max_rounds
                    + (self.stage2_distractor_messages * self.stage2_max_rounds)
                    + self.stage3_max_rounds,
                )
            else:
                reward_tool_usage_stats = legacy_tool_usage_stats
                reward_memory_stats = legacy_memory_stats
                reward_finished_at_round = len(stage3_exps)
                reward_max_rounds = self.stage3_max_rounds
            termination_finished_at_round = (
                self._stage3_round_count
                if self.tool_reward_stats_source == "trace"
                else len(stage3_exps)
            )
            termination_max_rounds = self.stage3_max_rounds

            public_memory_stats = {
                key: value
                for key, value in reward_memory_stats.items()
                if not key.startswith("_")
            }
            public_trace_memory_stats = {
                key: value
                for key, value in trace_memory_stats.items()
                if not key.startswith("_")
            }

            if self.reward_profile.is_terminal_only:
                reward_outcome = calculate_terminal_reward(
                    self.reward_profile,
                    task_score=task_score,
                    found_answer=found_answer,
                )
                total_reward = reward_outcome.total
                reward_breakdown = reward_outcome.breakdown
            else:
                if reward_calculator is None:
                    raise RuntimeError("E2 reward calculator was not initialized")
                total_reward, reward_breakdown = (
                    reward_calculator.calculate_total_reward(
                        task_score=task_score,
                        tool_usage_stats=reward_tool_usage_stats,
                        context_stats=context_stats,
                        memory_stats=reward_memory_stats,
                        finished_at_round=reward_finished_at_round,
                        max_rounds=reward_max_rounds,
                        found_answer=found_answer,
                        question=self.question,
                        supporting_facts=self.supporting_facts,
                        context_messages=self.context_messages,
                        termination_finished_at_round=termination_finished_at_round,
                        termination_max_rounds=termination_max_rounds,
                        tool_attempt_stats=(
                            tool_attempt_stats
                            if self.tool_reward_stats_source == "trace"
                            else None
                        ),
                    )
                )

            answer_metrics = (
                {
                    "answer_exact_match": self._last_answer_score.exact_match,
                    "answer_f1": self._last_answer_score.f1,
                    "answer_precision": self._last_answer_score.precision,
                    "answer_recall": self._last_answer_score.recall,
                }
                if self._last_answer_score is not None
                else {}
            )
            auxiliary_provider_usage = self.auxiliary_provider_usage.snapshot()

            detailed_info = {
                "task_score": task_score,
                "found_answer": found_answer,
                "reward_breakdown": reward_breakdown,
                "reward_profile_schema_version": self.reward_profile.schema_version,
                "reward_profile": self.reward_profile.name.value,
                "terminal_reward_metric": (
                    self.reward_profile.terminal_metric.value
                    if self.reward_profile.terminal_metric is not None
                    else None
                ),
                "milestone_reward_enabled": (
                    self.reward_profile.milestone_reward_enabled
                ),
                **answer_metrics,
                "tool_usage_stats": reward_tool_usage_stats,
                "legacy_tool_usage_stats": legacy_tool_usage_stats,
                "trace_tool_usage_stats": trace_tool_usage_stats,
                "tool_attempt_summary": tool_attempt_summary,
                "context_stats": context_stats,
                "memory_stats": public_memory_stats,
                "trace_memory_stats": public_trace_memory_stats,
                "tool_reward_stats_source": self.tool_reward_stats_source,
                "reward_finished_at_round": reward_finished_at_round,
                "reward_max_rounds": reward_max_rounds,
                "termination_finished_at_round": termination_finished_at_round,
                "termination_max_rounds": termination_max_rounds,
                "tool_trace_path": (
                    self.tool_trace_recorder.path
                    if self.tool_trace_recorder.enabled
                    else None
                ),
                "tool_trace_fallback_path": (self.tool_trace_recorder.fallback_path),
                "tool_trace_call_count": len(self._tool_trace_events),
                "tool_trace_dropped_record_count": (
                    self.tool_trace_recorder.dropped_record_count
                ),
                "tool_trace_last_write_error": (
                    self.tool_trace_recorder.last_write_error
                ),
                "trace_execution_id": self.current_execution_id,
                "memory_rollout_id": self.memory_manager.rollout_id,
                "auxiliary_provider": (
                    self.auxiliary_provider_config.public_dict()
                ),
                "auxiliary_provider_usage": auxiliary_provider_usage,
                "auxiliary_provider_telemetry_path": (
                    self.auxiliary_provider_telemetry_path
                ),
                "num_stages": 3,
            }

            if self.verbose:
                self.logger.info(f"Rollout {i} - Total Reward: {total_reward:.3f}")
                self.logger.info(
                    "  Task Score (%s): %.3f",
                    (
                        self.reward_profile.terminal_metric.value
                        if self.reward_profile.is_terminal_only
                        else "legacy_llm_judge"
                    ),
                    task_score,
                )
                self.logger.info(f"  Reward Breakdown: {reward_breakdown}")

            # 三个阶段共享同一个终局奖励。之后 Step-wise GRPO 会做组内标准化，
            # 并把 advantage 按 action_mask 广播到此前每个记忆决策 token。
            for exp in all_stage_experiences:
                exp.reward = total_reward
                merged_info = dict(exp.info or {})
                merged_info.update(detailed_info)
                exp.info = merged_info
                exp.eid.run = self.current_run_id
                if exp.metrics is None:
                    exp.metrics = {}
                exp.metrics["task_score"] = task_score
                exp.metrics["auxiliary_provider_calls"] = (
                    auxiliary_provider_usage["total_calls"]
                )
                exp.metrics["auxiliary_provider_failures"] = (
                    auxiliary_provider_usage["failed_calls"]
                )
                exp.metrics["auxiliary_provider_latency_ms"] = (
                    auxiliary_provider_usage["total_latency_ms"]
                )
                exp.metrics["auxiliary_provider_total_tokens"] = (
                    auxiliary_provider_usage["usage"]["total_tokens"]
                )

            experience_list.extend(all_stage_experiences)

        if not experience_list:
            self.logger.error(
                f"No experiences collected after {rollout_num} rollouts! This will cause timeout."
            )

        self.logger.info(f"Total experiences collected: {len(experience_list)}")
        return experience_list

    async def _get_answer_score(self, answer: str) -> float:
        """Score an answer according to the explicitly selected reward arm."""
        if not answer or not self.expected_answer:
            self._last_answer_score = None
            return 0.0
        if self.reward_profile.is_terminal_only:
            self._last_answer_score = score_hotpot_answer(
                str(answer),
                str(self.expected_answer),
            )
            return terminal_task_score(
                self.reward_profile,
                self._last_answer_score,
            )
        self._last_answer_score = None
        return await get_answer_llm_judge_score(
            self.question, answer, self.expected_answer, self.chat_client
        )

    async def _run_stage1_casual_chat(self) -> List[Experience]:
        """
        Stage 1：将多篇资料合成闲聊输入，诱导模型学习 LTM 的增删改。
        """
        stage_experiences = []

        if not self.context_info:
            self.logger.warning("context_info is None, skipping stage 1")
            return stage_experiences

        titles = self.context_info.get("title", [])
        sentences_list = self.context_info.get("sentences", [])

        if not titles or not sentences_list:
            self.logger.warning(
                f"Empty titles or sentences_list. titles: {len(titles) if titles else 0}, sentences: {len(sentences_list) if sentences_list else 0}"
            )
            return stage_experiences

        if len(titles) != len(sentences_list):
            self.logger.warning(
                f"titles and sentences_list length mismatch: {len(titles)} vs {len(sentences_list)}"
            )
            # Use the shorter length.
            min_len = min(len(titles), len(sentences_list))
            titles = titles[:min_len]
            sentences_list = sentences_list[:min_len]

        # Merge all title/sentences into a single casual-chat input.
        merged_context_lines = []
        for title, sents in zip(titles, sentences_list):
            # Truncate to avoid being overly long.
            sents_short = sents[
                : min(10, len(sents))
            ]  # Take up to 10 sentences per entry.
            merged_context_lines.append(f"{title}: {' '.join(sents_short)}")
        merged_context_text = "\n".join(merged_context_lines)

        casual_user_msg = (
            "Just chatting about several topics together. Here are the related contents grouped by title:\n"
            f"{merged_context_text}"
        )

        # Send the casual chat message in one shot.
        self._append_context("user", casual_user_msg)

        found_answer = False
        exps = []  # Initialize; avoid using an undefined variable outside loops.
        context_autosummarized = False

        # Multi-turn interaction until an answer is found or max rounds reached.
        for r in range(self.stage1_max_rounds):
            collected_exp_in_advance = False
            self.current_round = r
            self.current_step = r
            self.current_turn_index = 0

            if self.verbose:
                self.logger.info(
                    f"Stage 1, round {r} - Before Context messages: {self.context_messages}"
                )
            response_text = await self.get_model_response_text(self.context_messages)
            if self.verbose:
                self.logger.info(f"Stage 1, round {r} - Response text: {response_text}")
            exps = self.model.extract_experience_from_history(clear_history=True)
            for exp in exps:
                exp.eid.step = r
            self._annotate_experiences(
                exps,
                stage=1,
                round_index=r,
                step_index=r,
            )

            if not exps:
                self.logger.warning(
                    f"Stage 1, round {r}: extract_experience_from_history returned empty list"
                )
                # Even without experiences, continue to record at least one response.
                self._append_context("assistant", response_text)
                # Handle tool calls (if any) first.
                tool_calls = parse_tool_calls(response_text)
                if tool_calls:
                    self._apply_tools(tool_calls, exps)
                # Then check whether an answer is present.
                final_answer = parse_answer(response_text)
                if final_answer:
                    found_answer = True
                    break
                continue

            self._append_context("assistant", response_text)
            if self.verbose:
                self.logger.info(
                    f"Stage 1, round {r} - After Context messages: {self.context_messages}"
                )

            # Handle tool calls (if any) first.
            tool_calls = parse_tool_calls(response_text)

            # Mark experiences that used memory-management tools.
            memory_related_tool = should_collect_intermediate_experience(
                1,
                tool_calls,
                is_last_round=r >= self.stage1_max_rounds - 1,
            )

            if memory_related_tool:
                collected_exp_in_advance = True
                stage_experiences.extend(exps)

            if tool_calls:
                self._apply_tools(tool_calls, exps)

            # Then check for an answer (apply tools before returning if both exist).
            final_answer = parse_answer(response_text)
            if final_answer:
                found_answer = True
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                break

            # Check whether context overflow is triggered.
            if self._should_autosummarize():
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                context_autosummarized = True
                break

        # If no answer is found, add the last experience when context didn't overflow.
        if not found_answer and not context_autosummarized:
            if exps:
                stage_experiences.extend(exps)
            else:
                self.logger.warning(
                    "Stage 1: No experiences collected and no final answer found"
                )

        return stage_experiences

    async def _run_stage2_distractor_injection(self) -> List[Experience]:
        """
        Stage 2：逐条加入无关对话，训练模型主动 Summary/Clear 以控制 STM。
        """
        stage_experiences = []

        # E1 uses a fixed or task-persisted source so no auxiliary model call
        # can silently change the environment between reward arms.
        distractor_messages = resolve_stage2_distractors(
            source=self.stage2_distractor_source,
            count=self.stage2_distractor_messages,
            task_messages=self.task.raw_task.get("distractor_messages"),
            provider_generate=lambda count: (
                self.distractor_generator.generate_distractor_messages(
                    self.question,
                    num_messages=count,
                )
            ),
        )

        for idx, distractor_msg in enumerate(distractor_messages):
            # Send the distractor message as a user input.
            self._append_context("user", distractor_msg)

            if self.verbose:
                self.logger.info(
                    f"Stage 2, distractor {idx} - User message: {distractor_msg}"
                )

            found_answer = False
            exps = []  # Initialize; avoid using before assignment outside loops.
            context_autosummarized = False

            # Multi-turn interaction until an answer is found or max rounds reached.
            for r in range(self.stage2_max_rounds):
                collected_exp_in_advance = False
                step_index = idx * self.stage2_max_rounds + r
                self.current_round = r
                self.current_step = step_index
                self.current_turn_index = idx
                if self.verbose:
                    self.logger.info(
                        f"Stage 2, distractor {idx}, round {r} - Before Context messages: {self.context_messages}"
                    )
                response_text = await self.get_model_response_text(
                    self.context_messages
                )
                if self.verbose:
                    self.logger.info(
                        f"Stage 2, distractor {idx}, round {r} - Response text: {response_text}"
                    )
                exps = self.model.extract_experience_from_history(clear_history=True)
                for exp in exps:
                    exp.eid.step = step_index
                self._annotate_experiences(
                    exps,
                    stage=2,
                    round_index=r,
                    step_index=step_index,
                )

                if not exps:
                    self.logger.warning(
                        f"Stage 2, distractor {idx}, round {r}: extract_experience_from_history returned empty list"
                    )
                    # Even without experiences, continue to record at least one response.
                    self._append_context("assistant", response_text)
                    # Handle tool calls (if any) first.
                    tool_calls = parse_tool_calls(response_text)
                    if tool_calls:
                        self._apply_tools(tool_calls, exps)
                    # Then check whether an answer is present.
                    final_answer = parse_answer(response_text)
                    if final_answer:
                        found_answer = True
                        break
                    continue

                self._append_context("assistant", response_text)
                if self.verbose:
                    self.logger.info(
                        f"Stage 2, distractor {idx}, round {r} - After Context messages: {self.context_messages}"
                    )

                # Handle tool calls (if any) first.
                tool_calls = parse_tool_calls(response_text)

                # Mark experiences that used context-management tools.
                context_related_tool = should_collect_intermediate_experience(
                    2,
                    tool_calls,
                    is_last_round=r >= self.stage2_max_rounds - 1,
                )

                if context_related_tool:
                    collected_exp_in_advance = True
                    stage_experiences.extend(exps)

                if tool_calls:
                    self._apply_tools(tool_calls, exps)

                # Then check whether an answer is found (apply tools before returning if both exist).
                final_answer = parse_answer(response_text)
                if final_answer:
                    found_answer = True
                    if not collected_exp_in_advance:
                        stage_experiences.extend(exps)
                    break

                # Check context overflow.
                if self._should_autosummarize():
                    if not collected_exp_in_advance:
                        stage_experiences.extend(exps)
                    context_autosummarized = True
                    break

            # If no answer is found, add the last experience.
            if not found_answer and not context_autosummarized:
                if exps:
                    stage_experiences.extend(exps)
                else:
                    self.logger.warning(
                        f"Stage 2, distractor {idx}: No experiences collected and no final answer found"
                    )

        return stage_experiences

    async def _run_stage3_formal_qa(self) -> Tuple[List[Experience], float, bool]:
        """
        Stage 3：正式问答。模型要自行检索 LTM，并在有限 STM 中完成推理。

        Returns:
            (experiences, task_score, found_answer)
        """
        stage_experiences = []

        # User asks the formal question.
        self._append_context("user", self.question)

        # Hint: the model can retrieve previously stored memories.
        # items = self.memory_manager.retrieve(query=self.question, top_k=3)
        # retrieved_block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)
        # if retrieved_block:
        #     self._append_context("user", f"[related memories about the query]\n{retrieved_block}")

        found_final_answer = False
        final_answer = None
        task_score = 0.0

        context_autosummarized = False
        exps = []  # Initialize; avoid using before assignment outside loops.

        # Multi-turn interaction to find an answer.
        for r in range(self.stage3_max_rounds):
            collected_exp_in_advance = False
            self.current_round = r
            self.current_step = r
            self.current_turn_index = 0
            nudge_this_round = should_emit_stage3_final_answer_nudge(
                enabled=self.stage3_require_final_answer,
                round_index=r,
                max_rounds=self.stage3_max_rounds,
                found_answer=found_final_answer,
            )
            if nudge_this_round:
                self._append_context("user", STAGE3_FINAL_ANSWER_NUDGE)
                self.logger.info(
                    "Stage 3 round %s: appended final-answer nudge", r
                )
            if self.verbose:
                self.logger.info(
                    f"Id {r} - Before Context messages: {self.context_messages}"
                )
            response_text = await self.get_model_response_text(self.context_messages)
            self._stage3_round_count = r + 1
            if self.verbose:
                self.logger.info(f"Id {r} - Response text: {response_text}")
            exps = self.model.extract_experience_from_history(clear_history=True)
            for exp in exps:
                exp.eid.step = r
            self._annotate_experiences(
                exps,
                stage=3,
                round_index=r,
                step_index=r,
            )
            if nudge_this_round:
                for exp in exps:
                    info = dict(exp.info or {})
                    info["stage3_final_answer_nudge"] = True
                    exp.info = info

            if not exps:
                self.logger.warning(
                    f"Stage 3, round {r}: extract_experience_from_history returned empty list"
                )
                # Even without experiences, continue to record at least one response.
                self._append_context("assistant", response_text)
                # Handle tool calls (if any) first.
                tool_calls = parse_tool_calls(response_text)
                if tool_calls:
                    self._apply_tools(tool_calls, exps)
                # Then check whether an answer is present.
                final_answer = parse_answer(response_text)
                if final_answer:
                    found_final_answer = True
                    task_score = await self._get_answer_score(final_answer)
                    break
                continue

            self._append_context("assistant", response_text)
            if self.verbose:
                self.logger.info(
                    f"Id {r} - After Context messages: {self.context_messages}"
                )

            # Handle tool calls (if any) first.
            tool_calls = parse_tool_calls(response_text)

            # Mark experiences that used context-management tools.
            context_related_tool = should_collect_intermediate_experience(
                3,
                tool_calls,
                is_last_round=r >= self.stage3_max_rounds - 1,
            )

            if context_related_tool:
                collected_exp_in_advance = True
                stage_experiences.extend(exps)

            if tool_calls:
                self._apply_tools(tool_calls, exps)

            # Then check for the final answer (apply tools before returning if both exist).
            final_answer = parse_answer(response_text)
            if final_answer:
                found_final_answer = True
                task_score = await self._get_answer_score(final_answer)
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                break

            # Check context overflow.
            if self._should_autosummarize():
                context_autosummarized = True
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                break

        # If no answer is found, add the last experience.
        if not found_final_answer and not context_autosummarized:
            # Ensure at least one experience exists.
            if exps:
                stage_experiences.append(exps[-1])
            else:
                self.logger.warning(
                    "Stage 3: No experiences collected and no final answer found"
                )

        return stage_experiences, task_score, found_final_answer
