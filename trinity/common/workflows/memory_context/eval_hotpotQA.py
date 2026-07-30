from __future__ import annotations

import json
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, MultiTurnWorkflow, Task
from trinity.utils.log import get_logger

from .memory_store import MemoryManager, chat_client
from .workflow_prompt import TOOL_CALL_SYS_PROMPT, SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
from .utils import (
    TOOL_SCHEMA as COMMON_TOOL_SCHEMA,
    DistractorGenerator as CommonDistractorGenerator,
    build_tool_schema as common_build_tool_schema,
    create_tool_counter,
    extract_score as common_extract_score,
    parse_answer as common_parse_answer,
    parse_tool_calls as common_parse_tool_calls,
    record_tool_usage,
)

from ..memory_reward.my_reward import ThreeStageRewardCalculator, extract_tool_usage_stats, extract_context_stats, extract_memory_stats
from .workflow_metrics import (
    get_supporting_facts_llm_judge_score,
    extract_supporting_facts_from_context,
    extract_sentences_from_supporting_facts,
    get_answer_llm_judge_score,
    calculate_joint_llm_judge,
)

# Use shared utility implementations to avoid train/eval drift.
TOOL_SCHEMA = COMMON_TOOL_SCHEMA
build_tool_schema = common_build_tool_schema
parse_tool_calls = common_parse_tool_calls
parse_answer = common_parse_answer
extract_score = common_extract_score
DistractorGenerator = CommonDistractorGenerator


@WORKFLOWS.register_module("AgeMem_hotpot_workflow_evaluation")
class AgeMemHotpotWorkflowEvaluation(MultiTurnWorkflow):
    """
    Three-stage evaluation workflow for HotpotQA.
    Stage 1: context-grounded casual interaction.
    Stage 2: optional distractor injection.
    Stage 3: formal QA with integrated memory usage.
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

        # Context and memory management settings
        self.max_context_tokens = self.workflow_args.get("max_context_tokens", 32768)
        self.auto_summary_token_threshold = self.workflow_args.get("auto_summary_threshold", 0.8)
        self.max_tool_rounds_per_turn = self.workflow_args.get("max_tool_rounds_per_turn", 3)

        # Multi-stage configuration.
        self.stage2_distractor_messages = self.workflow_args.get("stage2_distractor_messages", 5)
        self.stage3_max_rounds = self.workflow_args.get("stage3_max_rounds", 5)
        self.stage1_max_rounds = self.workflow_args.get("stage1_max_rounds", 5)
        self.stage2_max_rounds = self.workflow_args.get("stage2_max_rounds", 5)

        # Initialize memory manager and chat client
        self.memory_manager = MemoryManager(embedding_model="text-embedding-v4", embedding_dim=256)
        self.chat_client = chat_client()
        self.distractor_generator = DistractorGenerator(self.chat_client)

        # State management
        self.context_messages: List[Dict] = []
        self.current_turn: int = 0
        self.final_reward: float = 0.0

        # Question and expected answer
        self.question: Optional[str] = None
        self.expected_answer: Optional[str] = None

        # Facts and context
        self.facts: Optional[dict] = None
        self.context: Optional[dict] = None
        raw_use_context_tools = self.workflow_args.get("use_context_tools", True)
        if isinstance(raw_use_context_tools, str):
            self.use_context_tools = raw_use_context_tools.strip().lower() not in ("false", "0", "no")
        else:
            self.use_context_tools = bool(raw_use_context_tools)
        raw_manual_top_k = self.workflow_args.get("manual_retrieve_top_k", 3)
        try:
            self.manual_retrieve_top_k = max(1, int(raw_manual_top_k))
        except (TypeError, ValueError):
            self.manual_retrieve_top_k = 3

        self.tool_schema = build_tool_schema(self.use_context_tools)
        self.sys_prompt = TOOL_CALL_SYS_PROMPT.format(tools=json.dumps(self.tool_schema))

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
        if role == "user" and not self.use_context_tools:
            self._append_manual_memory_retrieval(content)

    def _append_manual_memory_retrieval(self, query: Optional[str]):
        """Retrieve long-term memory manually when context tools are disabled."""
        if not query:
            return
        query_text = query.strip()
        if not query_text:
            return
        try:
            items = self.memory_manager.retrieve(query_text, self.manual_retrieve_top_k, {})
        except Exception as exc:
            self.logger.warning(f"Manual memory retrieval failed: {exc}")
            return

        if not items:
            return

        retrieved_block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items if getattr(it, "content", None))
        if not retrieved_block:
            return

        self.context_messages.append({
            "role": "tool",
            "content": f"[retrieved memories]\n[manual]\n{retrieved_block}"
        })

    def _should_autosummarize(self) -> bool:
        """Check if context should be auto-summarized based on token threshold."""
        total_chars = sum(len(m.get("content", "")) for m in self.context_messages)
        approx_tokens = total_chars / 4
        return approx_tokens > self.max_context_tokens * self.auto_summary_token_threshold

    def _get_retrieved_memory_text(self) -> str:
        """Extract the most recently retrieved memory content."""
        memory_text = ""
        for msg in self.context_messages:
            if msg.get("role") == "tool" and "[retrieved memories]" in msg.get("content", ""):
                content = msg.get("content", "")
                if "[retrieved memories]" in content:
                    memory_text = content.split("[retrieved memories]")[-1].strip()
                    break
        return memory_text

    def _get_all_memory_sentences(self) -> List[str]:
        """Split memory content (from MemoryManager or context) into a list of sentences."""
        memory_sentences: List[str] = []

        if not self.question:
            return memory_sentences

        top_k = int(self.workflow_args.get("memory_eval_top_k", 20)) # You can set this larger to retrieve all memories.

        try:
            items = self.memory_manager.retrieve(
                self.question,
                top_k,
                {},
            )
            for item in items or []:
                content = getattr(item, "content", "") or ""
                if content:
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        sentences = [s.strip() for s in re.split(r'[.!?]\s+', para) if s.strip()]
                        memory_sentences.extend(sentences)
        except Exception as exc:
            self.logger.warning(f"Failed to retrieve memory sentences: {exc}")

        if not memory_sentences:
            memory_text = self._get_retrieved_memory_text()
            if memory_text:
                memory_sentences = [s.strip() for s in re.split(r'[.!?]\s+', memory_text) if s.strip()]

        return memory_sentences

    def _apply_tools(self, tool_calls: List[Dict]) -> Optional[str]:

        """Apply tool calls and return any reply note."""
        reply_note: Optional[str] = None

        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {})

            # self.logger.info(f"Applying tool: {name} with args: {args}")

            if name == "Summary_context":
                if not self.use_context_tools:
                    reply_note = "summary_context_disabled"
                    self._append_context("tool", f"[context tool result]\n{reply_note}")
                    continue
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
                    # Handle different span formats
                    if "-" in span:
                        # Range format like "3-7"
                        try:
                            start, end = map(int, span.split("-"))
                            messages_to_summarize = non_system_messages[start - 2: end-1]
                            indices_to_replace = [i for i, m in messages_to_summarize]
                        except Exception:
                            # Fallback to all if parsing fails
                            messages_to_summarize = [(i, m) for i, m in non_system_messages]
                            indices_to_replace = [i for i, m in non_system_messages]
                    else:
                        # Number format like "5" for last N messages
                        try:
                            n = int(span)
                            messages_to_summarize = non_system_messages[-n:]
                            indices_to_replace = [i for i, m in messages_to_summarize]
                        except Exception:
                            # Fallback to all if parsing fails
                            messages_to_summarize = [(i, m) for i, m in non_system_messages]
                            indices_to_replace = [i for i, m in non_system_messages]

                # Preserve user query if requested
                if preserve_user_query:
                    # Find user messages and exclude them from summarization
                    user_indices = [i for i, m in messages_to_summarize if m.get("role") == "user"]
                    indices_to_replace = [i for i in indices_to_replace if i not in user_indices]
                    messages_to_summarize = [(i, m) for i, m in messages_to_summarize if i not in user_indices]

                # Generate summary from selected messages
                if messages_to_summarize:
                    conversation_text = "\n".join(
                        [f"{m.get('role', 'unknown')}: {m.get('content', '')}" for i, m in messages_to_summarize])
                    summary = self.chat_client.chat(messages=[{"role": "user",
                                                               "content": SUMMARY_CONTEXT_SYS_PROMPT.format(
                                                                   conversation_text=conversation_text)}],
                                                    model_name="qwen-max")

                    # Replace the original messages with summary
                    # Sort indices in descending order to avoid index shifting issues
                    indices_to_replace.sort(reverse=True)

                    for idx in indices_to_replace:
                        # Remove the original message
                        self.context_messages.pop(idx)

                    # Insert summary at the position of the first removed message
                    if indices_to_replace:
                        insert_position = min(indices_to_replace)
                        self.context_messages.insert(insert_position, {
                            "role": "tool",
                            "content": f"[summary of {len(messages_to_summarize)} messages]\n{summary}"
                        })

                    reply_note = f"summary_context_applied:success"

                    # self.logger.info(f"Summarized {len(messages_to_summarize)} messages and replaced them with summary")
                else:
                    reply_note = f"summary_context_applied:no_messages_to_summarize"
                    # self.logger.info("No messages to summarize")
                self._append_context("tool", f"[context tool result]\n{reply_note}")

            elif name == "Clear_context":
                if not self.use_context_tools:
                    reply_note = "clear_context_disabled"
                    self._append_context("tool", f"[context tool result]\n{reply_note}")
                    continue
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
                            messages=[{"role": "user", "content": TEXT_SIMILARITY_SYS_PROMPT.format(
                                text1=criteria, text2=m.get("content", ""))}],
                            model_name="qwen-max",
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
                reply_note = f"clear_context_applied:success:removed_{removed_count}_messages"
                self._append_context("tool", f"[context tool result]\n{reply_note}")

            elif name == "Retrieve_memory":
                if not self.use_context_tools:
                    reply_note = "retrieve_memory_disabled"
                    self._append_context("tool", f"[context tool result]\n{reply_note}")
                    continue
                query = args.get("query", "")
                top_k = int(args.get("top_k", 3))
                metadata_filter = args.get("metadata_filter", {})

                items = self.memory_manager.retrieve(query, top_k, metadata_filter)
                retrieved_block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)
                if retrieved_block:
                    self._append_context("tool", f"[retrieved memories]\n{retrieved_block}")
                else:
                    self._append_context("tool", f"[no related memories found]")

            elif name == "Add_memory":
                content = args.get("content", "")
                metadata = args.get("metadata", {}) or {}
                memory_type = args.get("memory_type", "general")

                # Add memory_type to metadata if provided
                if memory_type:
                    try:
                        metadata["type"] = memory_type
                    except Exception:
                        metadata = {}
                        metadata = {"type": memory_type}
                metadata["stage"] = str(self.current_stage)

                mem_id = str(uuid.uuid4())
                self.memory_manager.add_memory(mem_id, content, metadata)
                reply_note = f"memory_added:{mem_id}"
                self._append_context("tool", f"[memory tool result]\n{reply_note}")

            elif name == "Update_memory":
                mem_id = args.get("memory_id", "")
                content = args.get("content")
                metadata = args.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                ok = self.memory_manager.update_memory(mem_id, content, metadata)
                reply_note = f"memory_updated:{ok}"
                self._append_context("tool", f"[memory tool result]\n{reply_note}")

            elif name == "Delete_memory":
                mem_id = args.get("memory_id", "")
                confirmation = args.get("confirmation", False)

                if confirmation:
                    ok = self.memory_manager.delete_memory(mem_id)
                    reply_note = f"memory_deleted:{ok}"
                else:
                    reply_note = "memory_deletion_cancelled:confirmation_required"
                self._append_context("tool", f"[memory tool result]\n{reply_note}")

        return reply_note

    def reset_per_run(self):
        """Reset the workflow for each run."""
        self.context_messages.clear()
        self.memory_manager.clear()
        self.final_reward = -0.1
        self.current_stage = 0

    def _extract_answer(self) -> Optional[str]:
        """
        Extract answer from context using chat_client.
        
        Returns:
            Extracted answer string, or None if extraction fails
        """
        if not self.question or not self.context_info:
            return None
        
        # Build context text
        titles = self.context_info.get("title", [])
        sentences_list = self.context_info.get("sentences", [])
        
        if not titles or not sentences_list:
            return None
        
        # Build formatted context text
        context_text = ""
        for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
            context_text += f"\n[{idx}] {title}:\n"
            for sent_idx, sent in enumerate(sents):
                context_text += f"  {sent_idx}: {sent}\n"
        
        # Build prompt to extract answer
        extract_prompt = f"""You are a professional QA system. Please answer the question based on the given context.

Question: {self.question}

Context:
{context_text}

Please provide a concise and direct answer to the question. Only output the answer itself, without any additional explanation or formatting."""
        
        try:
            response = self.chat_client.chat(
                messages=[{"role": "user", "content": extract_prompt}],
                model_name="qwen-max"
            )
            
            # Clean and extract answer
            answer = response.strip() if response else None
            
            if answer:
                # Remove possible prefixes like "Answer:", "The answer is", etc.
                answer = re.sub(r'^(Answer|The answer is|The answer)[:：]\s*', '', answer, flags=re.IGNORECASE)
                # Remove quotes if present
                answer = re.sub(r'^["\']|["\']$', '', answer)
                answer = answer.strip()
            
            #self.logger.info(f"Extracted answer: {answer}")
            
            return answer if answer else None
            
        except Exception as e:
            self.logger.error(f"Error extracting answer: {e}", exc_info=True)
            return None

    async def run_async(self) -> List[Experience]:
        """Initialize the workflow and start the multi-turn, multi-step process."""
        rollout_n = self.repeat_times
        try:
            # self.logger.info("=== Starting Multi-Turn Multi-Step Context Memory Workflow ===")

            # Extract question and expected answer
            self.question = self.task.raw_task.get(self.task.format_args.prompt_key)
            self.expected_answer = self.task.raw_task.get(self.task.format_args.response_key)

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
                self.logger.error(f"Context info should be a dict, got {type(self.context_info)}")
                return []
            
            # If answer is missing in test set, extract it using chat_client
            if not self.expected_answer:
                # self.logger.info("Answer missing, extracting from context using chat_client...")
                extracted_answer = self._extract_answer()
                if extracted_answer:
                    self.expected_answer = extracted_answer
                    # self.logger.info(f"Extracted expected_answer: {self.expected_answer}")
                else:
                    self.logger.warning("Failed to extract answer from context")

            if self.supporting_facts is None:
                self.supporting_facts = {"title": [], "sent_id": []}

            if (
                isinstance(self.supporting_facts, dict)
                and not self.supporting_facts.get("title")
                and not self.supporting_facts.get("sent_id")
                and self.question
                and self.expected_answer
                and self.context_info
            ):
                self.supporting_facts = extract_supporting_facts_from_context(
                    self.question,
                    self.expected_answer,
                    self.context_info,
                    self.chat_client,
                )

            return await self.inference_samples(rollout_n)

        except Exception as e:
            self.logger.error(f"Error in run: {e}", exc_info=True)
            raise

    async def get_model_response_text(self, messages):
        """Get model response text."""
        responses = await self.model.chat_async(messages, n=1)
        return responses[0].response_text

    async def inference_samples(self, rollout_num: int) -> List[Experience]:
        """Run the three-stage evaluation rollout pipeline."""
        
        reward_calculator = ThreeStageRewardCalculator(
            task_completion_weight=0.5,
            tool_efficiency_weight=0.2,
            context_management_weight=0.15,
            memory_management_weight=0.15,
            chat_client=self.chat_client,
        )

        experience_list = []
        
        for i in range(rollout_num):
            self.reset_per_run()
            self._append_context("system", self.sys_prompt)

            all_stage_experiences: List[Experience] = []
            
            # Stage 1: Casual chat based on context
            self.current_stage = 1
            if self.verbose:
                self.logger.info(f"Rollout {i} - Stage 1: Casual chat based on context")
            
            stage1_exps, total_tool_calls_num_stage1 = await self._run_stage1_casual_chat()
            # self.logger.info(f"Rollout {i} - Stage 1 returned {len(stage1_exps)} experiences")
            # all_stage_experiences.extend(stage1_exps)
            
            # Stage 2: Distractor messages injection
            # Note: For consistency with the current repository behavior, Stage 2 is disabled by default.
            enable_stage2_in_eval = bool(self.workflow_args.get("enable_stage2_in_eval", False))
            self.context_messages.clear()
            self._append_context("system", self.sys_prompt)
            if enable_stage2_in_eval:
                self.current_stage = 2
                if self.verbose:
                    self.logger.info(f"Rollout {i} - Stage 2: Distractor messages injection (enabled)")
                stage2_exps = await self._run_stage2_distractor_injection()
                if stage2_exps:
                    all_stage_experiences.extend(stage2_exps)
            
            # Stage 3: Formal Q&A
            self.current_stage = 3
            if self.verbose:
                self.logger.info(f"Rollout {i} - Stage 3: Formal Q&A")
            
            stage3_exps, stage3_metrics = await self._run_stage3_formal_qa()
            self.logger.info(f"Rollout {i} - Stage 3 returned {len(stage3_exps)} experiences")
            if stage3_exps:
                # Use all Stage-3 experiences so the final-answer turn is never discarded.
                all_stage_experiences.extend(stage3_exps)
            
            # Ensure at least some experiences were collected.
            if not all_stage_experiences:
                self.logger.error(f"Rollout {i} - No experiences collected from any stage! This will cause timeout.")
                # Add a dummy experience to avoid total failure.
                # Better would be to check why no experiences were collected.
            
            # Compute total reward.
            tool_usage_stats = extract_tool_usage_stats(self.context_messages)
            context_stats = extract_context_stats(self.context_messages, self.max_context_tokens)
            memory_stats = extract_memory_stats(self.context_messages, self.memory_manager)
            
            task_score = stage3_metrics.get("task_score", stage3_metrics.get("answer_llm_judge", 0.0))
            found_answer = stage3_metrics.get("found_answer", False)
            total_tool_calls_num_stage3 = stage3_metrics.get("total_tool_calls_num", {})

            total_reward, reward_breakdown = reward_calculator.calculate_total_reward(
                task_score=task_score,
                tool_usage_stats=tool_usage_stats,
                context_stats=context_stats,
                memory_stats=memory_stats,
                finished_at_round=len(stage3_exps),
                max_rounds=self.stage3_max_rounds,
                found_answer=found_answer,
                question=self.question,
                supporting_facts=self.supporting_facts,
                context_messages=self.context_messages,
            )
            
            detailed_info = {
                "answer_llm_judge": stage3_metrics.get("answer_llm_judge", 0.0),
                "supporting_facts_em": stage3_metrics.get("supporting_facts_em", 0.0),
                "supporting_facts_recall": stage3_metrics.get("supporting_facts_recall", 0.0),
                "supporting_facts_precision": stage3_metrics.get("supporting_facts_precision", 0.0),
                "supporting_facts_llm_judge": stage3_metrics.get("supporting_facts_llm_judge", 0.0),
                "joint_em": stage3_metrics.get("joint_em", 0.0),
                "joint_llm_judge": stage3_metrics.get("joint_llm_judge", 0.0),
                "task_score": task_score,
                "found_answer": found_answer,
                "reward_breakdown": reward_breakdown,
                "tool_usage_stats": tool_usage_stats,
                "context_stats": context_stats,
                "memory_stats": memory_stats,
                "num_stages": 3,
                "total_tool_calls_num_stage3": total_tool_calls_num_stage3,
            }
            
            if self.verbose:
                self.logger.info(f"Rollout {i} - Total Reward: {total_reward:.3f}")
                self.logger.info(f"  Task Score (LLM-as-a-Judge): {task_score:.3f}")
                self.logger.info(f"  Reward Breakdown: {reward_breakdown}")
            
            # Assign the reward to all experiences.
            total_tool_calls_num = {k: total_tool_calls_num_stage1.get(k, 0) for k in total_tool_calls_num_stage1}
            for key, value in total_tool_calls_num_stage3.items():
                total_tool_calls_num[key] = total_tool_calls_num.get(key, 0) + value
            for exp in all_stage_experiences:
                exp.reward = total_reward
                exp.info = detailed_info
                exp.eid.run = i + self.run_id_base
                if exp.metrics is None:
                    exp.metrics = {}
                exp.metrics.update({
                    "answer_llm_judge": stage3_metrics.get("answer_llm_judge", 0.0),
                    "supporting_facts_em": stage3_metrics.get("supporting_facts_em", 0.0),
                    "supporting_facts_recall": stage3_metrics.get("supporting_facts_recall", 0.0),
                    "supporting_facts_precision": stage3_metrics.get("supporting_facts_precision", 0.0),
                    "supporting_facts_llm_judge": stage3_metrics.get("supporting_facts_llm_judge", 0.0),
                    "task_score": task_score,
                })
                exp.metrics["prompt_length"] = exp.prompt_length
                exp.metrics["prompt_tokens"] = exp.tokens.shape[0] if exp.tokens is not None else 0
                exp.metrics["Summary_context"] = total_tool_calls_num["Summary_context"]
                exp.metrics["Clear_context"] = total_tool_calls_num["Clear_context"]
                exp.metrics["Retrieve_memory"] = total_tool_calls_num["Retrieve_memory"]
                exp.metrics["Add_memory"] = total_tool_calls_num["Add_memory"]
                exp.metrics["Update_memory"] = total_tool_calls_num["Update_memory"]
                exp.metrics["Delete_memory"] = total_tool_calls_num["Delete_memory"]
                exp.metrics["long_term_memory"] = total_tool_calls_num["long_term_memory"]
                exp.metrics["short_term_memory"] = total_tool_calls_num["short_term_memory"]
                exp.metrics["total_tool_calls_num"] = total_tool_calls_num["long_term_memory"] + total_tool_calls_num["short_term_memory"]
                exp.metrics["finished_at_round"] = len(stage3_exps)

            
            experience_list.extend(all_stage_experiences)
        
        if not experience_list:
            self.logger.error(f"No experiences collected after {rollout_num} rollouts! This will cause timeout.")
        
        self.logger.info(f"Total experiences collected: {len(experience_list)}")
        return experience_list

    async def _run_stage1_casual_chat(self) -> List[Experience]:
        """
        Stage 1: Casual chat based on context.
        Goal: train the model's Add_memory capability.
        Supports multi-turn tool calls until the model provides an answer or reaches the maximum rounds.
        """
        stage_experiences = []

        if not self.context_info:
            self.logger.warning("context_info is None, skipping stage 1")
            return stage_experiences

        titles = self.context_info.get("title", [])
        sentences_list = self.context_info.get("sentences", [])
        
        if not titles or not sentences_list:
            self.logger.warning(f"Empty titles or sentences_list. titles: {len(titles) if titles else 0}, sentences: {len(sentences_list) if sentences_list else 0}")
            return stage_experiences
        
        if len(titles) != len(sentences_list):
            self.logger.warning(f"titles and sentences_list length mismatch: {len(titles)} vs {len(sentences_list)}")
            # Use the shorter length.
            min_len = min(len(titles), len(sentences_list))
            titles = titles[:min_len]
            sentences_list = sentences_list[:min_len]

        # Merge all title/sentences into a single casual-chat input.
        merged_context_lines = []
        for title, sents in zip(titles, sentences_list):
            # Truncate to avoid being overly long.
            sents_short = sents[: min(10, len(sents))]  # Take up to 10 sentences per entry.
            merged_context_lines.append(f"{title}: {' '.join(sents_short)}")
        merged_context_text = "\n".join(merged_context_lines)

        casual_user_msg = (
            "Just chatting about several topics together. Here are the related contents grouped by title:\n"
            f"{merged_context_text}"
        )

        # Send the casual chat message in one shot.
        self._append_context("user", casual_user_msg)

        found_answer = False
        exps = []  # Initialize; avoid using before assignment outside loops.
        context_autosummarized = False

        total_tool_calls_num = create_tool_counter()
        
        # Multi-turn interaction until an answer is found or max rounds reached.
        for r in range(self.stage1_max_rounds):
            collected_exp_in_advance = False
            
            if self.verbose:
                self.logger.info(f"Stage 1, round {r} - Before Context messages: {self.context_messages}")
            response_text = await self.get_model_response_text(self.context_messages)
            if self.verbose:
                self.logger.info(f"Stage 1, round {r} - Response text: {response_text}")
            exps = self.model.extract_experience_from_history(clear_history=True)
            # self.logger.info(f"Stage 1, round {r}: exps: {exps[0].prompt_length}, {exps[0].tokens.shape[0]}")
            # raise Exception("Stop here")
            
            if not exps:
                self.logger.warning(f"Stage 1, round {r}: extract_experience_from_history returned empty list")
                # Even without experiences, continue to record at least one response.
                self._append_context("assistant", response_text)
                # Handle tool calls (if any) first.
                tool_calls = parse_tool_calls(response_text)
                if tool_calls:
                    record_tool_usage(total_tool_calls_num, tool_calls)
                    self._apply_tools(tool_calls)
                # Then check whether an answer is present.
                final_answer = parse_answer(response_text)
                if final_answer:
                    found_answer = True
                    break
                continue
            
            for exp in exps:
                exp.eid.step = r
                # exp.eid.stage = 1
            
            self._append_context("assistant", response_text)
            if self.verbose:
                self.logger.info(f"Stage 1, round {r} - After Context messages: {self.context_messages}")
            
            # Handle tool calls (if any) first.
            tool_calls = parse_tool_calls(response_text)
            
            # Mark experiences that used memory-management tools.
            memory_related_tool = any(
                tool_call.get("name") in ["Add_memory", "Retrieve_memory", "Update_memory"]
                for tool_call in tool_calls
            )
            
            if memory_related_tool and r < self.stage1_max_rounds - 1:
                collected_exp_in_advance = True
                stage_experiences.extend(exps)
            
            if tool_calls:
                record_tool_usage(total_tool_calls_num, tool_calls)
                # self.logger.info(f"Id {r} - Memory context before apply tools: {self.context_messages}")
                self._apply_tools(tool_calls)
                # self.logger.info(f"Id {r} - Tool calls: {tool_calls}")
                # self.logger.info(f"Id {r} - Memory context after apply tools: {self.context_messages}")

            
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
                self.logger.warning("Stage 1: No experiences collected and no final answer found")

        return stage_experiences, total_tool_calls_num

    async def _run_stage2_distractor_injection(self) -> List[Experience]:
        """
        Stage 2: Distractor messages injection.
        Goal: train the model's Clear_context capability.
        Supports multi-turn tool calls until the model provides an answer or reaches the maximum rounds.
        """
        stage_experiences = []
        
        # Generate distractor messages.
        distractor_messages = self.distractor_generator.generate_distractor_messages(
            self.question,
            num_messages=self.stage2_distractor_messages
        )
        
        for idx, distractor_msg in enumerate(distractor_messages):       
            # Send the distractor message as a user input.
            self._append_context("user", distractor_msg)
            
            if self.verbose:
                self.logger.info(f"Stage 2, distractor {idx} - User message: {distractor_msg}")
            
            found_answer = False
            exps = []  # Initialize; avoid using before assignment outside loops.
            context_autosummarized = False
            
            # Multi-turn interaction until an answer is found or max rounds reached.
            for r in range(self.stage2_max_rounds):
                collected_exp_in_advance = False
                if self.verbose:
                    self.logger.info(f"Stage 2, distractor {idx}, round {r} - Before Context messages: {self.context_messages}")
                response_text = await self.get_model_response_text(self.context_messages)
                if self.verbose:
                    self.logger.info(f"Stage 2, distractor {idx}, round {r} - Response text: {response_text}")
                exps = self.model.extract_experience_from_history(clear_history=True)
                
                if not exps:
                    self.logger.warning(f"Stage 2, distractor {idx}, round {r}: extract_experience_from_history returned empty list")
                    # Even without experiences, continue to record at least one response.
                    self._append_context("assistant", response_text)
                    # Handle tool calls (if any) first.
                    tool_calls = parse_tool_calls(response_text)
                    if tool_calls:
                        self._apply_tools(tool_calls)
                    # Then check whether an answer is present.
                    final_answer = parse_answer(response_text)
                    if final_answer:
                        found_answer = True
                        break
                    continue
                
                for exp in exps:
                    exp.eid.step = idx * self.stage2_max_rounds + r
                    # exp.eid.stage = 2
                
                self._append_context("assistant", response_text)
                if self.verbose:
                    self.logger.info(f"Stage 2, distractor {idx}, round {r} - After Context messages: {self.context_messages}")
                
                # Handle tool calls (if any) first.
                tool_calls = parse_tool_calls(response_text)
                
                # Mark experiences that used context-management tools.
                context_related_tool = any(
                    tool_call.get("name") in ["Clear_context", "Summary_context"]
                    for tool_call in tool_calls
                )
                
                if context_related_tool and r < self.stage2_max_rounds - 1:
                    collected_exp_in_advance = True
                    stage_experiences.extend(exps)
                
                if tool_calls:
                    self._apply_tools(tool_calls)
                
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
                    self.logger.warning(f"Stage 2, distractor {idx}: No experiences collected and no final answer found")
        
        return stage_experiences

    async def _run_stage3_formal_qa(self) -> Tuple[List[Experience], Dict]:
        """
        Stage 3: Formal Q&A.
        Goal: train the model's Retrieve_memory and Summary_context capabilities, and overall reasoning performance.
        """
        stage_experiences = []

        # User asks the formal question.
        self._append_context("user", self.question)

        found_final_answer = False
        final_answer = None
        context_autosummarized = False
        exps: List[Experience] = []

        total_tool_calls_num = create_tool_counter()

        for r in range(self.stage3_max_rounds):
            collected_exp_in_advance = False
            if self.verbose:
                self.logger.info(f"Id {r} - Before Context messages: {self.context_messages}")

            response_text = await self.get_model_response_text(self.context_messages)
            if self.verbose:
                self.logger.info(f"Id {r} - Response text: {response_text}")

            exps = self.model.extract_experience_from_history(clear_history=True)

            if not exps:
                self.logger.warning(f"Stage 3, round {r}: extract_experience_from_history returned empty list")
                self._append_context("assistant", response_text)

                tool_calls = parse_tool_calls(response_text)
                if tool_calls:
                    record_tool_usage(total_tool_calls_num, tool_calls)
                    self._apply_tools(tool_calls)

                final_answer = parse_answer(response_text)
                if final_answer:
                    found_final_answer = True
                    break
                continue

            for exp in exps:
                exp.eid.step = r

            self._append_context("assistant", response_text)
            if self.verbose:
                self.logger.info(f"Id {r} - After Context messages: {self.context_messages}")

            tool_calls = parse_tool_calls(response_text)

            context_related_tool = any(
                tool_call.get("name") in ["Summary_context", "Clear_context", "Retrieve_memory"]
                for tool_call in tool_calls
            )

            if context_related_tool and r < self.stage3_max_rounds - 1:
                collected_exp_in_advance = True
                stage_experiences.extend(exps)

            if tool_calls:
                record_tool_usage(total_tool_calls_num, tool_calls)
                self._apply_tools(tool_calls)

            final_answer = parse_answer(response_text)
            if final_answer:
                found_final_answer = True
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                break

            if self._should_autosummarize():
                context_autosummarized = True
                if not collected_exp_in_advance:
                    stage_experiences.extend(exps)
                break

        if not found_final_answer and not context_autosummarized:
            if exps:
                stage_experiences.append(exps[-1])
            else:
                self.logger.warning("Stage 3: No experiences collected and no final answer found")

        metrics: Dict[str, float] = {}

        if final_answer and self.expected_answer:
            metrics["answer_llm_judge"] = await get_answer_llm_judge_score(
                self.question,
                final_answer,
                self.expected_answer,
                self.chat_client,
            )
        else:
            metrics["answer_llm_judge"] = 0.0

        expected_supporting_facts = self.supporting_facts or {"title": [], "sent_id": []}
        if not isinstance(expected_supporting_facts, dict):
            expected_supporting_facts = {"title": [], "sent_id": []}
        if "title" not in expected_supporting_facts or "sent_id" not in expected_supporting_facts:
            expected_supporting_facts = {"title": [], "sent_id": []}

        expected_facts_sentences: List[str] = []
        if expected_supporting_facts and self.context_info:
            expected_facts_sentences = extract_sentences_from_supporting_facts(
                expected_supporting_facts,
                self.context_info,
            )

        predicted_facts_sentences = self._get_all_memory_sentences()
        memory_sentences = predicted_facts_sentences[:]

        if expected_facts_sentences:
            metrics["supporting_facts_llm_judge"] = await get_supporting_facts_llm_judge_score(
                self.question,
                final_answer or "",
                predicted_facts_sentences,
                expected_facts_sentences,
                memory_sentences,
                self.chat_client,
            )
        else:
            empty_score = 1.0 if not predicted_facts_sentences else 0.0
            metrics["supporting_facts_llm_judge"] = empty_score

        metrics["joint_llm_judge"] = calculate_joint_llm_judge(
            metrics["answer_llm_judge"],
            metrics["supporting_facts_llm_judge"],
        )

        metrics["found_answer"] = found_final_answer
        metrics["task_score"] = metrics["answer_llm_judge"]
        metrics["total_tool_calls_num"] = total_tool_calls_num

        return stage_experiences, metrics
