# -*- coding: utf-8 -*-
"""ReAct-style agent with 6 memory/context management tools (AgeMem)."""
from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

import shortuuid
from pydantic import BaseModel, ValidationError

from agentscope.agent import AgentBase
from agentscope._logging import logger
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase
from agentscope.message import Msg, TextBlock, ToolResultBlock, ToolUseBlock
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit, ToolResponse, execute_python_code

from .memory import AgentScopeLongtermMemory
from .prompts import SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
from .src import (
    GenerateResponseSchema,
    chat_client,
    extract_score,
    finish_function_pre_print_hook,
)


class AgeMem(AgentBase):
    """不依赖 RL 训练流程的 AgeMem 智能体，是理解六类工具的最佳入口。"""

    finish_function_name: str = "generate_response"
    """The function name used to finish replying and return a response to
    the user."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        toolkit: Optional[Toolkit] = None,
        memory: Optional[MemoryBase] = None,
        max_iters: int = 10,
        max_context_tokens: int = 32768,
        auto_summary_token_threshold: float = 0.8,
        show_tool_trace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Static variables in the agent
        self.name = name
        self._sys_prompt = sys_prompt
        self.model = model
        self.formatter = formatter
        self.max_iters = max_iters
        self.max_context_tokens = max_context_tokens
        self.auto_summary_token_threshold = auto_summary_token_threshold
        # 是否把每次工具调用及其返回值实时打印到终端。
        self.show_tool_trace = show_tool_trace

        self.chat_client = kwargs.get("chat_client", chat_client())

        # -------------- Memory management --------------
        # LTM：独立于对话上下文的向量记忆库，可跨多轮检索。
        self.memory_manager = memory or AgentScopeLongtermMemory(
            embedding_model="text-embedding-v4",
            embedding_dim=256,
        )

        # STM：当前 ReAct 循环能直接看到的消息列表。
        self.context_messages: List[Msg] = []

        # Stage tracking for multi-stage training
        self.current_stage: int = 0

        # -------------- Tool management --------------
        # 把“结束回答 + 六个记忆工具”注册成模型可选择的动作空间。
        # If None, a default Toolkit will be created
        self.toolkit = toolkit or Toolkit()
        self.toolkit.register_tool_function(getattr(self, self.finish_function_name))
        self._required_structured_model: Optional[Type[BaseModel]] = None
        self.toolkit.set_extended_model(self.finish_function_name, GenerateResponseSchema)
        self.toolkit.register_tool_function(self.summary_context)
        self.toolkit.register_tool_function(self.filter_context)
        self.toolkit.register_tool_function(self.retrieve_memory)
        self.toolkit.register_tool_function(self.add_memory)
        self.toolkit.register_tool_function(self.update_memory)
        self.toolkit.register_tool_function(self.delete_memory)

        self.toolkit.register_tool_function(execute_python_code)

        self.register_state("name")
        self.register_state("sys_prompt")
        self.register_instance_hook(
            "pre_print",
            "finish_function_pre_print_hook",
            finish_function_pre_print_hook,
        )

    @property
    def sys_prompt(self) -> str:
        return self._sys_prompt

    def _append_context(self, name: str, content: str, role: str) -> None:
        self.context_messages.append(
            Msg(name=name or role, content=content, role=role)
        )

    async def reply(
        self,
        msg: Optional[Msg | List[Msg]] = None,
        structured_model: Optional[Type[BaseModel]] = None,
        clear_context: bool = False,
    ) -> Msg:
        """执行一次 ReAct 循环：模型选工具 -> 执行工具 -> 观察结果 -> 继续决策。"""

        # Clear the context if clear_context is True
        if clear_context:
            self.context_messages.clear()
        if msg is not None:
            if isinstance(msg, Msg):
                msg = [msg]
            for m in msg:
                self.context_messages.append(m)
            # await self._retrieve_from_long_term_memory(msg)

        self._required_structured_model = structured_model
        if structured_model:
            self.toolkit.set_extended_model(
                self.finish_function_name,
                structured_model,
            )

        reply_msg = None
        for _ in range(self.max_iters):
            # 每轮都用最新 STM 构造 prompt；LTM 只有经 retrieve 才进入 prompt。
            prompt = await self.formatter.format(
                msgs=[
                    Msg("system", self.sys_prompt, "system"),
                    *self.context_messages,
                ],
            )
            logger.debug(f"User message {_+1}: {prompt}")
            logger.debug(f"Agent prompt length: {len(prompt)}")

            response = await self.model(
                prompt,
                tools=self.toolkit.get_json_schemas(),
            )
            if not hasattr(response, "content"):
                final_response = None
                async for chunk in response:
                    final_response = chunk
                response = final_response
                if response is None:
                    raise RuntimeError("Model stream yielded no response.")

            msg = Msg(
                name=self.name,
                content=list(response.content),
                role="assistant",
            )
            if msg and not msg.has_content_blocks("tool_use"):
                msg = Msg.from_dict(msg.to_dict())
                msg.content = [
                    ToolUseBlock(
                        id=shortuuid.uuid(),
                        type="tool_use",
                        name=self.finish_function_name,
                        input={"response": msg.get_text_content()},
                    ),
                ]

            self.context_messages.append(msg)

            # print(f"msg: {msg}")

            futures = [
                self._apply_tool(tool_call)
                for tool_call in msg.get_content_blocks("tool_use")
            ]
            acting_responses = [await f for f in futures]
            for acting_msg in acting_responses:
                reply_msg = reply_msg or acting_msg
            if reply_msg:
                # 只有 generate_response 会产生最终 reply_msg，其余工具会进入下一轮。
                break

        if reply_msg is None:
            reply_msg = await self._summarizing()
        self.context_messages.append(reply_msg)
        return reply_msg

    async def _apply_tool(self, tool_call: ToolUseBlock) -> Msg | None:
        # generate_response 是结束 ReAct 循环的内部工具；main.py 会单独打印最终回答，
        # 因此这里只展示真正参与推理/记忆管理的工具，避免最终回答重复出现。
        should_trace = (
            self.show_tool_trace
            and tool_call["name"] != self.finish_function_name
        )
        if should_trace:
            await self.print(
                Msg(self.name, [tool_call], "assistant"),
                True,
            )

        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        try:
            tool_res = await self.toolkit.call_tool_function(tool_call)
            response_msg = None
            async for chunk in tool_res:
                tool_res_msg.content[0]["output"] = chunk.content
                if should_trace:
                    # AgentScope 会把 ToolResultBlock 以 JSON 形式打印；
                    # 流式工具只在最后一个分块完整展示结果。
                    await self.print(tool_res_msg, chunk.is_last)
                if chunk.metadata and chunk.metadata.get("success", True):
                    if tool_call["name"] == self.finish_function_name:
                        response_msg = chunk.metadata.get("response_msg")
            return response_msg
        finally:
            self.context_messages.append(tool_res_msg)

    def generate_response(self, response: str, **kwargs: Any) -> ToolResponse:
        """Final response to the user."""
        response = response or kwargs.get("reply", "")
        response_msg = Msg(self.name, response, "assistant")

        # Structured output support (optional)
        if self._required_structured_model:
            try:
                structured_data = dict(kwargs)
                if hasattr(self._required_structured_model, "model_fields"):
                    fields = self._required_structured_model.model_fields
                    if "result" in fields and "result" not in structured_data:
                        structured_data["result"] = response
                    if "response" in fields and "response" not in structured_data:
                        structured_data["response"] = response
                response_msg.metadata = (
                    self._required_structured_model.model_validate(structured_data).model_dump()
                )
            except ValidationError as e:
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"Arguments Validation Error: {e}")],
                    metadata={"success": False, "response_msg": None},
                )
        return ToolResponse(
            content=[TextBlock(type="text", text="Successfully generated response.")],
            metadata={"success": True, "response_msg": response_msg},
            is_last=True,
        )

    # ---------- AgeMem Tool implementations (aligned return messages) ----------
    async def summary_context(
        self,
        span
    ) -> ToolResponse:
        """压缩 STM 中选定的历史消息，同时保留发起本次工具调用的消息。

        Args:
            span (`str`):
                The range to summarize. "all" | integer string (last N).

        Returns:
            `ToolResponse`:
                The response containing the result of summarizing the context.
        """
        context_messages = self.context_messages[:]
        last_idx = len(context_messages) - 1
        # The message that invoked this tool (assistant with summary_context tool_calls); must stay at end.
        keeper = context_messages[last_idx] if context_messages else None

        filtered: List[Msg] = []
        non_system_messages: List[tuple[int, Msg]] = []
        for i, m in enumerate(context_messages):
            if m.role == "system":
                filtered.append(m)
            else:
                non_system_messages.append((i, m))

        messages_to_summarize: List[tuple] = []
        indices_to_replace: List[int] = []
        if span == "all":
            messages_to_summarize = list(non_system_messages)
            indices_to_replace = [i for i, _ in non_system_messages]
        else:
            try:
                n = int(span)
                messages_to_summarize = non_system_messages[-n:]
                indices_to_replace = [i for i, _ in messages_to_summarize]
            except Exception:
                messages_to_summarize = list(non_system_messages)
                indices_to_replace = [i for i, _ in non_system_messages]

        # Never remove the last message (assistant that called summary_context), so after append we have assistant + tool.
        indices_to_replace = [i for i in indices_to_replace if i != last_idx]

        if not messages_to_summarize:
            return ToolResponse(
                content=[TextBlock(type="text", text="The span is invalid. There are no messages to summarize.")],
                metadata={"success": False, "span": span},
            )

        conversation_text = "\n".join(
            f"{m.role}: {m.get_text_content()}" for _, m in messages_to_summarize
        )
        summary = self.chat_client.chat(
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_CONTEXT_SYS_PROMPT.format(conversation_text=conversation_text),
                }
            ],
            model_name="qwen-max",
        )
        note = f"Successfully summarized the context of {len(messages_to_summarize)} messages. \n Summary: {summary}"

        if span == "all":
            self.context_messages = [keeper] if keeper else []
        else:
            if not indices_to_replace:
                self.context_messages = context_messages
            else:
                lo, hi = min(indices_to_replace), max(indices_to_replace)
                indices_to_remove = set(indices_to_replace)
                for i in range(lo, hi + 1):
                    if context_messages[i].role == "system":
                        indices_to_remove.add(i)
                indices_to_remove.discard(last_idx)
                filtered = [m for i, m in enumerate(context_messages) if i not in indices_to_remove]
                self.context_messages = filtered

        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "summary_msg": summary, "span": span},
        )

    async def filter_context(
        self, 
        criteria: str
        ) -> ToolResponse:
        """Filters out irrelevant or outdated content from the conversation context to improve task-solving efficiency. 

        Args:
            criteria (`str`):
                The criteria for content removal. Can be keywords, phrases, or descriptions of content types to remove (e.g., ’the birthday of John’, ’the age of Mary’).

        Returns:
            `ToolResponse`:
                The response containing the result of clearing the context.
        """
        filtered = []
        removed_count = 0

        for m in self.context_messages:
            if m.role == "system":
                filtered.append(m)
                continue
            if criteria:
                # Use LLM-based similarity matching for better accuracy
                similarity_text = self.chat_client.chat(
                    messages=[{
                        "role": "user",
                        "content": TEXT_SIMILARITY_SYS_PROMPT.format(
                            text1=criteria,
                            text2=m.get_text_content(),
                        ),
                    }],
                    model_name="qwen-max",
                )
                if extract_score(similarity_text, default=0.0) >= 0.1:
                    removed_count += 1
                    continue
            filtered.append(m)

        self.context_messages = filtered
        note = f"Successfully cleared {removed_count} messages from the context."
        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "removed_count": removed_count, "criteria": criteria},
        )

    async def retrieve_memory(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None,
    ) -> ToolResponse:
        """从 LTM 检索相关条目；工具结果随后会作为观察进入下一轮 STM。

        Args:
            query (`str`):
                The search query to find relevant memories. Should describe what kind of information or context is needed.
            top_k (`int`, defaults to `3`):
                The maximum number of memories to retrieve. Defaults to 3.
            metadata_filter (`dict`, defaults to `None`):
                Optional metadata constraints.

        Returns:
            `ToolResponse`:
                The response containing the result of retrieving the memories.
        """
        items = await self.memory_manager.retrieve(
            query, int(top_k), metadata_filter or {}
        )
        if items:
            block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)
            # self._append_context("tool", f"[retrieved memories]\n{block}", "assistant")
            text = f"Successfully retrieved {len(items)} memories. Detailed memories: {block}"
        else:
            # self._append_context("tool", "[no related memories found]", "assistant")
            text = "No related memories found."
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"success": True, "query": query},
        )

    async def add_memory(
        self,
        content: str,
        metadata: dict = None,
        memory_type: str = "general",
    ) -> ToolResponse:
        """Add new information into the long term memory store for future retrieval.

        Args:
            content (`str`):
                The content to store.
            metadata (`dict`, defaults to `None`):
                Optional metadata tags.
            memory_type (`str`, defaults to `"general"`):
                Logical type tag; will be stored as metadata["type"].

        Returns:
            `ToolResponse`:
                The response containing the result of adding the memory.
        """
        md = dict(metadata or {})
        if memory_type:
            md["type"] = memory_type

        # Add stage information to metadata
        md["stage"] = str(self.current_stage)
        mem_id = str(uuid.uuid4())
        await self.memory_manager.add(mem_id, content, md)
        note = f"Successfully added a new memory. Memory ID: {mem_id}, content: {content}"
        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "memory_id": mem_id, "content": content},
        )

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: dict = None,
    ) -> ToolResponse:
        """Update an existing memory's content and/or metadata in the long term memory store.

        Args:
            memory_id (`str`):
                The target memory identifier.
            content (`str`):
                New content to replace existing content.
            metadata (`dict`, defaults to `None`):
                Metadata to merge/update.

        Returns:
            `ToolResponse`:
                The response containing the result of updating the memory.
        """
        ok = await self.memory_manager.update(memory_id, content, metadata or {})
        if ok:
            note = f"Successfully updated the memory. Memory ID: {memory_id}, updated content: {content}"
            #self._append_context("tool", f"[memory tool result]\n{note}", "assistant")
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": True,
                    "memory_id": memory_id,
                    "content": content,
                },
            )
        else:
            note = f"Failed to update the memory. Memory ID: {memory_id} is not found."
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": False,
                    "memory_id": memory_id,
                },
            )

    async def delete_memory(self, memory_id: str, confirmation: bool = False) -> ToolResponse:
        """Delete a memory permanently from the long term memory store  when confirmation is True.

        Args:
            memory_id (`str`):
                The target memory identifier.
            confirmation (`bool`, defaults to `False`):
                Must be True to proceed.

        Returns:
            `ToolResponse`:
                The response containing the result of deleting the memory.
        """
        if not confirmation:
            note = "Deletion is cancelled. The argument 'confirmation' must be True to proceed."
            return ToolResponse(
                content=[TextBlock(type="text", text=note)],
                metadata={"success": False, "memory_id": memory_id},
            )
        ok = await self.memory_manager.delete(memory_id)
        if ok:
            note = f"Successfully deleted the memory. Memory ID: {memory_id}."
            # self._append_context("tool", f"[memory tool result]\n{note}", "assistant")
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": True,
                    "memory_id": memory_id,
                },
            )
        else:
            note = f"Failed to delete the memory. Memory ID: {memory_id} is not found."
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": False,
                    "memory_id": memory_id,
                },
            )

    async def _retrieve_from_long_term_memory(self, msg: Msg | list[Msg] | None) -> None:
        """Retrieve relevant memories by semantic similarity from the long term memory store and attach them to context."""
        if msg:
            query = str()
            if isinstance(msg, Msg):
                msg = [msg]
            for m in msg:
                query += m.get_text_content() + "\n"
            items = await self.memory_manager.retrieve(query, 3)
            if items:
                retrieved_block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)

                memory_content = "<long_term_memory>The content below are retrieved from long-term memory, which maybe " \
                    + f"useful:\n{retrieved_block}</long_term_memory>"

                self._append_context("tool", memory_content, "assistant")

    async def _summarizing(self) -> Msg:
        """Generate a response when the agent fails to solve the problem in
        the maximum iterations."""
        hint_msg = Msg(
            "user",
            "You have failed to generate response within the maximum "
            "iterations. Now respond directly by summarizing the current "
            "situation.",
            role="user",
        )

        # Generate a reply by summarizing the current situation
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *self.context_messages,
                hint_msg,
            ],
        )
        # TODO: handle the structured output here, maybe force calling the
        #  finish_function here
        res = await self.model(prompt)

        res_msg = Msg(self.name, [], "assistant")
        if isinstance(res, AsyncGenerator):
            async for chunk in res:
                res_msg.content = chunk.content
        else:
            res_msg.content = res.content
        
        logger.info(f"res_msg: {res_msg}")
        has_answer = TextBlock(type="text", text="has_answer:No")
        res_msg.content.append(has_answer)

        return res_msg
