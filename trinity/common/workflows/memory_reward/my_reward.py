from typing import Dict, List, Optional, Tuple
import re

_TRACKED_TOOL_NAMES = (
    "Summary_context",
    "Clear_context",
    "Retrieve_memory",
    "Add_memory",
    "Update_memory",
    "Delete_memory",
)


class RewardCalculator:
    """基础多维奖励计算器；论文训练实际使用下方的 ThreeStageRewardCalculator。"""

    def __init__(
        self,
        # Reward weight configuration
        task_completion_weight: float = 1.0,
        tool_efficiency_weight: float = 0.3,
        context_management_weight: float = 0.2,
        memory_management_weight: float = 0.2,
        # Penalty configuration
        max_rounds_penalty: float = -1.0,
        context_overflow_penalty: float = -0.5,
        # Reward clipping range
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ):
        self.task_completion_weight = task_completion_weight
        self.tool_efficiency_weight = tool_efficiency_weight
        self.context_management_weight = context_management_weight
        self.memory_management_weight = memory_management_weight
        self.max_rounds_penalty = max_rounds_penalty
        self.context_overflow_penalty = context_overflow_penalty
        self.min_reward = min_reward
        self.max_reward = max_reward

    def calculate_total_reward(
        self,
        task_score: float,
        tool_usage_stats: Dict,
        context_stats: Dict,
        memory_stats: Dict,
        finished_at_round: int,
        max_rounds: int,
        found_answer: bool,
    ) -> Tuple[float, Dict]:
        """
        Compute total reward and breakdown.
        """
        reward_breakdown = {}

        if found_answer:
            task_reward = task_score * self.task_completion_weight
        else:
            task_reward = self.max_rounds_penalty
        reward_breakdown["task_completion"] = task_reward

        tool_reward = self._calculate_tool_efficiency_reward(
            tool_usage_stats, finished_at_round, max_rounds
        )
        reward_breakdown["tool_efficiency"] = tool_reward * self.tool_efficiency_weight

        context_reward = self._calculate_context_management_reward(context_stats)
        reward_breakdown["context_management"] = (
            context_reward * self.context_management_weight
        )

        memory_reward = self._calculate_memory_management_reward(memory_stats)
        reward_breakdown["memory_management"] = (
            memory_reward * self.memory_management_weight
        )

        total_reward = sum(reward_breakdown.values())

        if finished_at_round >= max_rounds:
            total_reward += self.max_rounds_penalty
            reward_breakdown["max_rounds_penalty"] = self.max_rounds_penalty

        if context_stats.get("overflow_occurred", False):
            total_reward += self.context_overflow_penalty
            reward_breakdown["context_overflow_penalty"] = self.context_overflow_penalty

        total_reward = max(self.min_reward, min(self.max_reward, total_reward))
        reward_breakdown["total"] = total_reward

        return total_reward, reward_breakdown

    def _calculate_tool_efficiency_reward(
        self, tool_usage_stats: Dict, finished_at_round: int, max_rounds: int
    ) -> float:
        """
        Calculate tool efficiency reward
        """
        reward = 0.0

        summary_calls = tool_usage_stats.get("Summary_context", 0)
        retrieve_calls = tool_usage_stats.get("Retrieve_memory", 0)
        add_memory_calls = tool_usage_stats.get("Add_memory", 0)
        clear_calls = tool_usage_stats.get("Clear_context", 0)
        total_calls = sum(tool_usage_stats.values())
        # Presence uses effective calls; density uses all attempts when trace
        # statistics are available so no-op spam still has a cost.

        if summary_calls > 0 or clear_calls > 0:
            reward += 0.3
            if summary_calls + clear_calls > finished_at_round * 0.5:
                reward -= 0.1

        if retrieve_calls > 0:
            reward += 0.2
            if retrieve_calls > finished_at_round * 0.8:
                reward -= 0.1

        if add_memory_calls > 0:
            reward += 0.2
            if add_memory_calls > 5:
                reward -= 0.05 * (add_memory_calls - 5)

        if total_calls > 0:
            tool_call_ratio = total_calls / max(finished_at_round, 1)
            if 0.3 <= tool_call_ratio <= 2.0:
                reward += 0.2
            elif tool_call_ratio > 3.0:
                reward -= 0.2

        efficiency_bonus = (1.0 - finished_at_round / max_rounds) * 0.1
        reward += efficiency_bonus

        return max(-0.5, min(1.0, reward))

    def _calculate_context_management_reward(self, context_stats: Dict) -> float:
        """
        Calculate context management reward
        """
        reward = 0.0

        current_token_ratio = context_stats.get("token_usage_ratio", 0.0)
        if current_token_ratio < 0.6:
            reward += 0.4
        elif current_token_ratio < 0.8:
            reward += 0.2
        else:
            reward -= 0.2

        messages_removed = context_stats.get("messages_removed", 0)
        if messages_removed > 0:
            reward += 0.3
            total_messages = context_stats.get("total_messages", 1)
            if messages_removed > total_messages * 0.7:
                reward -= 0.2

        preserved_user_query = context_stats.get("preserved_user_query", True)
        preserved_key_info = context_stats.get("preserved_key_info", True)
        if preserved_user_query and preserved_key_info:
            reward += 0.3
        elif not preserved_user_query:
            reward -= 0.5

        return max(-0.5, min(1.0, reward))

    def _calculate_memory_management_reward(self, memory_stats: Dict) -> float:
        """
        Calculate the memory-management quality reward.

        Reward components:
        - Retrieve relevant memories and use them: +0.4
        - Store high-quality memories (reusable knowledge): +0.3
        - Update/delete outdated memories: +0.2
        - Avoid redundant storage: +0.1
        """
        reward = 0.0

        retrieved_count = memory_stats.get("retrieved_count", 0)
        used_retrieved_memory = memory_stats.get("used_retrieved_memory", False)
        if retrieved_count > 0 and used_retrieved_memory:
            reward += 0.4
        elif retrieved_count > 0:
            reward += 0.1

        added_count = memory_stats.get("added_count", 0)
        high_quality_additions = memory_stats.get("high_quality_additions", 0)
        if added_count > 0:
            quality_ratio = high_quality_additions / added_count
            reward += 0.3 * quality_ratio

        updated_count = memory_stats.get("updated_count", 0)
        deleted_count = memory_stats.get("deleted_count", 0)
        if updated_count > 0 or deleted_count > 0:
            reward += 0.2

        is_redundant = memory_stats.get("redundant_storage", False)
        if not is_redundant:
            reward += 0.1
        else:
            reward -= 0.2

        return max(-0.5, min(1.0, reward))


def extract_tool_usage_stats(
    context_messages: List[Dict],
    tool_events: Optional[List[Dict]] = None,
) -> Dict:
    """Count effective six-tool calls from trace events or legacy STM text."""
    tool_stats = {tool_name: 0 for tool_name in _TRACKED_TOOL_NAMES}

    if tool_events is not None:
        for event in tool_events:
            if (
                isinstance(event, dict)
                and event.get("phase") == "finish"
                and event.get("status") == "success"
                and event.get("tool_name") in tool_stats
                and isinstance(event.get("result"), dict)
                and event["result"].get("effect_applied") is True
            ):
                tool_stats[event["tool_name"]] += 1
        return tool_stats

    for msg in context_messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            for tool_name in tool_stats.keys():
                if tool_name in content:
                    tool_stats[tool_name] += 1

    return tool_stats


def extract_tool_attempt_stats(tool_events: List[Dict]) -> Dict:
    """Return attempt/effect counters for audit and trace density cost."""
    by_tool = {
        tool_name: {
            "attempted": 0,
            "succeeded": 0,
            "errored": 0,
            "cancelled": 0,
            "validation_error": 0,
            "effect_applied": 0,
        }
        for tool_name in _TRACKED_TOOL_NAMES
    }
    unknown_tool_calls = 0
    total_attempted = 0

    for event in tool_events:
        if not isinstance(event, dict) or event.get("phase") != "finish":
            continue
        total_attempted += 1

        tool_name = event.get("tool_name")
        if tool_name not in by_tool:
            if event.get("status") == "unknown_tool":
                unknown_tool_calls += 1
            continue

        stats = by_tool[tool_name]
        stats["attempted"] += 1
        status = event.get("status")
        if status == "success":
            stats["succeeded"] += 1
        elif status == "error":
            stats["errored"] += 1
        elif status == "cancelled":
            stats["cancelled"] += 1
        elif status == "validation_error":
            stats["validation_error"] += 1

        result = event.get("result")
        if isinstance(result, dict) and result.get("effect_applied") is True:
            stats["effect_applied"] += 1

    return {
        "by_tool": by_tool,
        "unknown_tool_calls": unknown_tool_calls,
        "total_attempted": total_attempted,
    }


def extract_context_stats(context_messages: List[Dict], max_tokens: int) -> Dict:
    """提取 STM 指标。

    注意：这里用“字符数 / 4”近似 token 数，适合轻量反馈但不是精确计数。
    """
    total_chars = sum(len(m.get("content", "")) for m in context_messages)
    approx_tokens = total_chars / 4

    # Find the first real user message (not a tool/bracket-prefixed injection).
    first_user_query = ""
    preserved_user_query = False
    for m in context_messages:
        if m.get("role") == "user" and not m.get("content", "").startswith("["):
            preserved_user_query = True
            if not first_user_query:
                first_user_query = m.get("content", "")

    # R_preservation (Eq. 21): check whether key tokens from the original query
    # are still present somewhere in the current context.
    preserved_key_info = True
    if first_user_query:
        key_tokens = [
            w.lower() for w in re.findall(r"\b\w+\b", first_user_query) if len(w) > 3
        ]
        if key_tokens:
            context_text = " ".join(
                m.get("content", "").lower() for m in context_messages
            )
            hit_count = sum(1 for t in key_tokens if t in context_text)
            preserved_key_info = (hit_count / len(key_tokens)) >= 0.5

    messages_removed = 0
    for msg in context_messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if "summary of" in content:
                match = re.search(r"summary of (\d+) messages", content)
                if match:
                    messages_removed += int(match.group(1)) - 1

    return {
        "token_usage_ratio": approx_tokens / max_tokens,
        "total_messages": len(context_messages),
        "messages_removed": messages_removed,
        "preserved_user_query": preserved_user_query,
        "preserved_key_info": preserved_key_info,
        "overflow_occurred": approx_tokens > max_tokens,
    }


def extract_memory_stats(
    context_messages: List[Dict],
    memory_manager,
    tool_events: Optional[List[Dict]] = None,
) -> Dict:
    """从工具结果中估计 LTM 的写入、维护、检索以及检索后使用情况。"""
    retrieved_count = 0
    added_count = 0
    updated_count = 0
    deleted_count = 0
    used_retrieved_memory = False

    retrieved_memory_content = []
    added_memory_content = []

    if tool_events is not None:
        used_retrieval_call_ids = {
            event.get("call_id")
            for event in tool_events
            if isinstance(event, dict)
            and event.get("phase") == "usage"
            and event.get("tool_name") == "Retrieve_memory"
            and event.get("call_id") is not None
            and isinstance(event.get("usage"), dict)
            and event["usage"].get("used_by_following_response") is True
        }
        for event in tool_events:
            if (
                not isinstance(event, dict)
                or event.get("phase") != "finish"
                or event.get("status") != "success"
            ):
                continue

            tool_name = event.get("tool_name")
            result = event.get("result")
            result = result if isinstance(result, dict) else {}
            effect_applied = result.get("effect_applied") is True

            if tool_name == "Retrieve_memory" and result.get("retrieved_count", 0) > 0:
                retrieved_count += 1
                if (
                    result.get("used_by_following_response") is True
                    or event.get("call_id") in used_retrieval_call_ids
                ):
                    used_retrieved_memory = True
                for item in result.get("items", []):
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        retrieved_memory_content.append(item["content"])
            elif tool_name == "Add_memory" and effect_applied:
                added_count += 1
                arguments = event.get("arguments")
                if isinstance(arguments, dict) and isinstance(
                    arguments.get("content"), str
                ):
                    added_memory_content.append(arguments["content"])
            elif tool_name == "Update_memory" and effect_applied:
                updated_count += 1
            elif tool_name == "Delete_memory" and effect_applied:
                deleted_count += 1

    for i, msg in enumerate(context_messages):
        content = msg.get("content", "")

        if msg.get("role") == "tool" and "[retrieved memories]" in content:
            if tool_events is None:
                retrieved_count += 1
            lines = content.split("\n")
            for line in lines[1:]:
                if tool_events is None and line.strip().startswith("-"):
                    retrieved_memory_content.append(line)

            if i + 1 < len(context_messages):
                next_msg = context_messages[i + 1]
                if next_msg.get("role") == "assistant":
                    used_retrieved_memory = True

        if tool_events is None and msg.get("role") == "tool":
            if "memory_added:" in content:
                added_count += 1

            if "memory_updated:" in content:
                updated_count += 1

            if "memory_deleted:" in content:
                deleted_count += 1

    high_quality_additions = 0
    if tool_events is not None:
        for content in added_memory_content:
            if len(content) > 50 and any(
                keyword in content.lower()
                for keyword in ["important", "key", "remember", "note"]
            ):
                high_quality_additions += 1
    else:
        for msg in context_messages:
            if msg.get("role") == "assistant" and "Add_memory" in msg.get(
                "content", ""
            ):
                content = msg.get("content", "")
                if len(content) > 50 and any(
                    keyword in content.lower()
                    for keyword in ["important", "key", "remember", "note"]
                ):
                    high_quality_additions += 1

    memory_count = None
    try:
        memory_count = memory_manager.count()
    except (AttributeError, TypeError):
        pass

    stats = {
        "retrieved_count": retrieved_count,
        "added_count": added_count,
        "updated_count": updated_count,
        "deleted_count": deleted_count,
        "used_retrieved_memory": used_retrieved_memory,
        "high_quality_additions": high_quality_additions,
        "redundant_storage": False,
        "current_memory_count": memory_count,
    }
    if tool_events is not None:
        stats["_retrieved_contents"] = retrieved_memory_content
        stats["_added_contents"] = added_memory_content
    return stats


def _parse_tool_calls_in_text(text: str) -> List[Dict]:
    """Parse a JSON array inside an assistant message's <tool_call> block."""
    if not text:
        return []
    m = re.search(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", text, re.S)
    if not m:
        return []
    json_str = m.group(1)
    try:
        import json

        return json.loads(json_str)
    except Exception:
        return []


class ThreeStageRewardCalculator(RewardCalculator):
    """
    三阶段 AgeMem 的终局奖励。

    总奖励由任务正确性、上下文质量、记忆质量、工具策略四部分加权组成，
    另加最大轮数与上下文溢出惩罚。该结果会回传给同一 rollout 的所有步骤。
    """

    def __init__(
        self,
        *args,
        chat_client=None,
        preventive_threshold: float = 0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chat_client = chat_client
        self.preventive_threshold = preventive_threshold

    def calculate_total_reward(
        self,
        task_score: float,
        tool_usage_stats: Dict,
        context_stats: Dict,
        memory_stats: Dict,
        finished_at_round: int,
        max_rounds: int,
        found_answer: bool,
        question: Optional[str] = None,
        supporting_facts: Optional[List] = None,
        context_messages: Optional[List[Dict]] = None,
        termination_finished_at_round: Optional[int] = None,
        termination_max_rounds: Optional[int] = None,
        tool_attempt_stats: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        breakdown: Dict[str, float] = {}

        task_reward = (
            task_score if found_answer else 0.0
        ) * self.task_completion_weight
        if not found_answer:
            task_reward += self.max_rounds_penalty
        breakdown["task_completion"] = task_reward
        context_term = self._context_quality(context_stats, context_messages)
        breakdown["context_management"] = context_term * self.context_management_weight
        memory_term, mem_details = self._memory_quality(
            memory_stats, question, supporting_facts, context_messages
        )
        breakdown["memory_management"] = memory_term * self.memory_management_weight
        attempted_call_count = None
        if tool_attempt_stats is not None:
            attempted_call_count = max(
                0,
                int(tool_attempt_stats.get("total_attempted", 0)),
            )
        tool_term = self._tool_policy(
            tool_usage_stats,
            finished_at_round,
            max_rounds,
            attempted_call_count=attempted_call_count,
        )
        breakdown["tool_efficiency"] = tool_term * self.tool_efficiency_weight
        total = sum(breakdown.values())
        termination_finished_at_round = (
            finished_at_round
            if termination_finished_at_round is None
            else termination_finished_at_round
        )
        termination_max_rounds = (
            max_rounds if termination_max_rounds is None else termination_max_rounds
        )
        if termination_finished_at_round >= termination_max_rounds:
            total += self.max_rounds_penalty
            breakdown["max_rounds_penalty"] = self.max_rounds_penalty
        if context_stats.get("overflow_occurred", False):
            total += self.context_overflow_penalty
            breakdown["context_overflow_penalty"] = self.context_overflow_penalty

        total = max(self.min_reward, min(self.max_reward, total))
        breakdown["total"] = total

        breakdown.update(
            {
                "memory_r_storage": mem_details.get("r_storage", 0.0),
                "memory_r_maintenance": mem_details.get("r_maintenance", 0.0),
                "memory_r_relevance": mem_details.get("r_relevance", 0.0),
            }
        )

        return total, breakdown

    # ---------- Internals ----------
    def _context_quality(
        self, context_stats: Dict, context_messages: Optional[List[Dict]]
    ) -> float:
        token_ratio = context_stats.get("token_usage_ratio", 1.0)
        # R_compression：上下文越精简越高，但不能以丢失关键信息为代价。
        r_compression = max(0.0, 1.0 - token_ratio)

        # R_preventive：在真正溢出前主动调用 Summary/Clear。
        r_preventive = 0.0
        if "effective_context_management_call" in context_stats:
            invoked = bool(context_stats.get("effective_context_management_call"))
            if invoked and not context_stats.get("overflow_occurred", False):
                r_preventive = 1.0
        elif context_messages:
            invoked = any(
                msg.get("role") == "assistant"
                and (
                    "Summary_context" in msg.get("content", "")
                    or "Clear_context" in msg.get("content", "")
                )
                for msg in context_messages
            )
            if invoked and not context_stats.get("overflow_occurred", False):
                r_preventive = 1.0

        # R_preservation：压缩/过滤后，用户问题和关键 token 仍然保留。
        r_preservation = (
            1.0
            if (
                context_stats.get("preserved_user_query", True)
                and context_stats.get("preserved_key_info", True)
            )
            else 0.0
        )

        return (r_compression + r_preventive + r_preservation) / 3.0

    def _memory_quality(
        self,
        memory_stats: Dict,
        question: Optional[str],
        supporting_facts: Optional[List],
        context_messages: Optional[List[Dict]],
    ) -> Tuple[float, Dict[str, float]]:
        """长期记忆质量奖励。

        R_memory = (R_storage + R_maintenance + R_relevance) / 3

        Each component starts from a lightweight count-based base signal to
        ensure dense feedback during early training (when the model has not yet
        learned to invoke memory tools regularly).  When the LLM judge is
        available and the relevant operations have occurred, the base is
        overridden with the more accurate LLM score.
        """

        # R_maintenance：能更新或删除过时条目，说明模型不只会无限追加。
        r_maintenance = (
            1.0
            if (
                memory_stats.get("updated_count", 0) > 0
                or memory_stats.get("deleted_count", 0) > 0
            )
            else 0.0
        )

        # ── Base signals ──────────────────────────────────────────────────────
        # R_storage base: any memory was added → a small positive signal.
        added_count = memory_stats.get("added_count", 0)
        high_quality_additions = memory_stats.get("high_quality_additions", 0)
        if added_count > 0:
            quality_ratio = high_quality_additions / added_count
            r_storage = 0.3 * quality_ratio
        else:
            r_storage = 0.0

        is_redundant = memory_stats.get("redundant_storage", False)
        if is_redundant:
            r_storage = max(0.0, r_storage - 0.1)

        # R_relevance base: retrieve was called; reward more if the retrieved
        # content was immediately followed by an assistant response.
        if memory_stats.get("retrieved_count", 0) > 0:
            r_relevance = (
                0.5 if memory_stats.get("used_retrieved_memory", False) else 0.2
            )
        else:
            r_relevance = 0.0

        # ── LLM overrides ─────────────────────────────────────────────────────
        if self.chat_client and question and context_messages:
            trace_retrieval_contents = "_retrieved_contents" in memory_stats
            trace_added_contents = "_added_contents" in memory_stats
            retrieved_blobs: List[str] = [
                value
                for value in memory_stats.get("_retrieved_contents", [])
                if isinstance(value, str)
            ]
            added_contents: List[str] = [
                value
                for value in memory_stats.get("_added_contents", [])
                if isinstance(value, str)
            ]

            for msg in context_messages:
                content = msg.get("content", "")
                if (
                    not trace_retrieval_contents
                    and msg.get("role") == "tool"
                    and content.startswith("[retrieved memories]")
                ):
                    for line in content.split("\n")[1:]:
                        line = line.strip()
                        if line.startswith("-"):
                            blob = re.sub(r"^-\s*", "", line)
                            if blob not in retrieved_blobs:
                                retrieved_blobs.append(blob)

                if (
                    not trace_added_contents
                    and msg.get("role") == "assistant"
                    and "<tool_call>" in content
                ):
                    calls = _parse_tool_calls_in_text(content)
                    for c in calls:
                        if c.get("name") == "Add_memory":
                            args = c.get("arguments", {}) or {}
                            if isinstance(args, dict):
                                text = args.get("content") or ""
                                if text and text not in added_contents:
                                    added_contents.append(str(text))

            if retrieved_blobs:
                prompt = (
                    "You are grading retrieval relevance.\n"
                    f"Question: {question}\n"
                    "Retrieved items (one per line):\n"
                    + "\n".join([f"- {b}" for b in retrieved_blobs[:10]])
                    + "\n\n"
                    "Score on 0.0-1.0 how relevant these are overall to answering "
                    "the question. Respond with a single number."
                )
                try:
                    grade = self.chat_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model_name="qwen-max",
                    )
                    m = re.search(r"(\d+\.?\d*)", (grade or "").strip())
                    if m:
                        r_relevance = max(0.0, min(1.0, float(m.group(1))))
                except Exception:
                    pass  # keep base value on failure

            # Ask the LLM to count how many added entries are high-quality,
            # given the query q and supporting_facts (expected-answer evidence A_q).
            n_total = memory_stats.get("added_count", 0)
            if added_contents and n_total > 0:
                sample = added_contents[:10]
                facts_hint = ""
                if supporting_facts:
                    if isinstance(supporting_facts, list):
                        facts_text = "\n".join(
                            [f"- {f}" for f in supporting_facts[:10]]
                        )
                    else:
                        facts_text = str(supporting_facts)[:500]
                    facts_hint = f"Key supporting facts (expected-answer evidence):\n{facts_text}\n\n"
                prompt = (
                    "You are grading stored memory entries for quality.\n"
                    f"Question: {question}\n"
                    + facts_hint
                    + f"Stored memory entries ({len(sample)} total):\n"
                    + "\n".join([f"{i + 1}. {a[:300]}" for i, a in enumerate(sample)])
                    + "\n\nCount how many entries are HIGH-QUALITY: "
                    "useful for answering the question and not redundant. "
                    "Respond with only a single integer."
                )
                try:
                    grade = self.chat_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model_name="qwen-max",
                    )
                    m = re.search(r"(\d+)", (grade or "").strip())
                    if m:
                        n_high = min(int(m.group(1)), len(sample))
                        r_storage = n_high / max(1, len(sample))
                except Exception:
                    pass  # keep base value on failure

        return (r_storage + r_maintenance + r_relevance) / 3.0, {
            "r_storage": r_storage,
            "r_maintenance": r_maintenance,
            "r_relevance": r_relevance,
        }

    def _tool_policy(
        self,
        tool_usage_stats: Dict,
        finished_at_round: int,
        max_rounds: int,
        *,
        attempted_call_count: Optional[int] = None,
    ) -> float:
        """奖励“必要且适量”的工具调用，并鼓励更少轮次完成任务。"""
        # 调用密度控制：过少学不到记忆管理，过多则增加成本和噪声。
        total_calls = sum(tool_usage_stats.values())
        density_calls = (
            total_calls if attempted_call_count is None else attempted_call_count
        )
        ratio = density_calls / max(1, finished_at_round)
        scale = 0.2 if 0.3 <= ratio <= 2.0 else (-0.1 if ratio > 3.0 else 0.0)

        # Presence of key tools.
        existence = 0.0
        if tool_usage_stats.get("Retrieve_memory", 0) > 0:
            existence += 0.2
        if (
            tool_usage_stats.get("Summary_context", 0)
            + tool_usage_stats.get("Clear_context", 0)
            > 0
        ):
            existence += 0.2
        if tool_usage_stats.get("Add_memory", 0) > 0:
            existence += 0.1

        # Efficiency based on completion rounds.
        efficiency = (1.0 - finished_at_round / max_rounds) * 0.2

        return max(-0.5, min(1.0, scale + existence + efficiency))
