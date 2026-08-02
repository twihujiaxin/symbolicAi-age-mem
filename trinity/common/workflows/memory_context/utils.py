from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

TOOL_SCHEMA = [
    {
        "name": "Summary_context",
        "description": "Summarizes previous conversation rounds in the context to reduce token usage while preserving important information. Should be called when context length approaches token limits or when summarizing specific conversation spans.",
        "parameters": {
            "type": "dict",
            "properties": {
                "span": {
                    "description": "The messages to summarize. Use 'all' for all non-system messages, or a positive number (e.g., '5') for the last N non-system messages.\n\nExamples:\n- \"all\": summarize all non-system messages\n- \"5\": summarize the last 5 non-system messages",
                    "type": "string",
                }
            },
            "required": ["span"],
        },
        "required": None,
    },
    {
        "name": "Clear_context",
        "description": "Removes irrelevant or outdated content from the conversation context to improve task-solving efficiency. Helps maintain focus on current task by filtering out noise.",
        "parameters": {
            "type": "dict",
            "properties": {
                "criteria": {
                    "description": "The criteria for content removal. Can be keywords, phrases, or descriptions of content types to remove (e.g., 'the birthday of John', 'the age of Mary').",
                    "type": "string",
                }
            },
            "required": ["criteria"],
        },
        "required": None,
    },
    {
        "name": "Retrieve_memory",
        "description": "Retrieves relevant memories from the external memory store and adds them to the current context. Uses semantic similarity search to find memories that match the query.",
        "parameters": {
            "type": "dict",
            "properties": {
                "query": {
                    "description": "The search query to find relevant memories. Should describe what kind of information or context is needed.",
                    "type": "string",
                },
                "top_k": {
                    "description": "The maximum number of memories to retrieve. Defaults to 3.",
                    "type": "integer",
                },
                "metadata_filter": {
                    "description": "Optional metadata filters to narrow down memory search (e.g., {'type': 'user_info', 'domain': 'math'}).",
                    "type": "object",
                },
            },
            "required": ["query"],
        },
        "required": None,
    },
    {
        "name": "Add_memory",
        "description": "Adds new information to the external memory store. Can store chat information, user preferences, task-solving pipelines, or any other useful knowledge for future reference.",
        "parameters": {
            "type": "dict",
            "properties": {
                "content": {
                    "description": "The content to store in memory. Should be clear, concise, and self-contained for future retrieval.",
                    "type": "string",
                },
                "metadata": {
                    "description": "Optional metadata tags to categorize and filter the memory (e.g., {'type': 'user_preference', 'domain': 'math', 'importance': 'high'}).",
                    "type": "object",
                },
                "memory_type": {
                    "description": "The type of memory being stored. Common types: 'chat_info', 'user_info', 'task_pipeline', 'knowledge', 'preference'.",
                    "type": "string",
                },
            },
            "required": ["content"],
        },
        "required": None,
    },
    {
        "name": "Update_memory",
        "description": "Updates an existing memory in the external memory store. Requires first retrieving relevant memories to identify which one to update. The memory_id should be obtained from retrieved memories.",
        "parameters": {
            "type": "dict",
            "properties": {
                "memory_id": {
                    "description": "The unique identifier of the memory to update. Must be obtained from a previous memory retrieval operation.",
                    "type": "string",
                },
                "content": {
                    "description": "The new content to replace the existing memory content. If not provided, only metadata will be updated.",
                    "type": "string",
                },
                "metadata": {
                    "description": "Updated metadata for the memory. Will be merged with existing metadata.",
                    "type": "object",
                },
            },
            "required": ["memory_id", "content"],
        },
        "required": None,
    },
    {
        "name": "Delete_memory",
        "description": "Removes a memory from the external memory store. Requires first retrieving relevant memories to identify which one to delete. The memory_id should be obtained from retrieved memories.",
        "parameters": {
            "type": "dict",
            "properties": {
                "memory_id": {
                    "description": "The unique identifier of the memory to delete. Must be obtained from a previous memory retrieval operation.",
                    "type": "string",
                },
                "confirmation": {
                    "description": "Confirmation that this memory should be permanently deleted. Should be set to true to confirm deletion.",
                    "type": "boolean",
                },
            },
            "required": ["memory_id", "confirmation"],
        },
        "required": None,
    },
]

CONTEXT_TOOL_NAMES = {"Summary_context", "Clear_context", "Retrieve_memory"}
DEFAULT_TOOL_COUNTER = {
    "Summary_context": 0,
    "Clear_context": 0,
    "Retrieve_memory": 0,
    "Add_memory": 0,
    "Update_memory": 0,
    "Delete_memory": 0,
    "long_term_memory": 0,
    "short_term_memory": 0,
}
LONG_TERM_TOOL_NAMES = {"Add_memory", "Delete_memory", "Update_memory"}
TOOL_NAMES = {
    "Summary_context",
    "Clear_context",
    "Retrieve_memory",
    "Add_memory",
    "Update_memory",
    "Delete_memory",
}

# 这些集合只决定“中间轮是否单独保留 Experience”，不决定工具是否执行。
# 不在集合中的工具仍会按顺序执行并进入 tool trace。
INTERMEDIATE_EXPERIENCE_TOOL_NAMES = {
    1: frozenset({"Add_memory", "Retrieve_memory", "Update_memory"}),
    2: frozenset({"Summary_context", "Clear_context"}),
    3: frozenset({"Summary_context", "Clear_context", "Retrieve_memory"}),
}

DEFAULT_DISTRACTOR_MESSAGES = [
    "What's the weather like today?",
    "Can you recommend a good recipe for chocolate cake?",
    "I'm thinking about learning a new programming language.",
    "Do you know any interesting facts about ancient civilizations?",
    "What are your thoughts on modern art movements?",
]


def build_tool_schema(use_context_tools: bool) -> List[Dict]:
    """Return the tool schema filtered by the current configuration."""
    if use_context_tools:
        return TOOL_SCHEMA
    return [tool for tool in TOOL_SCHEMA if tool.get("name") not in CONTEXT_TOOL_NAMES]


def should_collect_intermediate_experience(
    stage: int,
    tool_calls: List[Dict],
    *,
    is_last_round: bool,
) -> bool:
    """Return whether this intermediate round belongs in the training sample.

    This helper centralizes the original three-stage filtering rules. It must
    not be used to filter execution: every parsed tool call is still applied.
    """
    if is_last_round:
        return False

    selected_names = INTERMEDIATE_EXPERIENCE_TOOL_NAMES.get(stage, frozenset())
    return any(
        isinstance(call, dict) and call.get("name") in selected_names
        for call in tool_calls
    )


def validate_tool_call(
    call: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize one parsed call without mutating model output."""
    if not isinstance(call, dict):
        return None, "tool call must be an object"

    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "tool name must be a non-empty string"
    if name != name.strip():
        return None, "tool name must not contain leading or trailing whitespace"

    raw_arguments = call.get("arguments", {})
    if "arguments" in call and raw_arguments is None:
        return {"name": name, "arguments": None}, "arguments must be an object"
    if not isinstance(raw_arguments, dict):
        return {"name": name, "arguments": raw_arguments}, "arguments must be an object"

    arguments = dict(raw_arguments)
    normalized = {"name": name, "arguments": arguments}
    if name not in TOOL_NAMES:
        return normalized, f"unknown tool: {name}"

    if name == "Summary_context":
        if "span" not in arguments:
            return normalized, "span is required"
        span = arguments.get("span", "all")
        if not isinstance(span, str) or not span.strip():
            return normalized, "span must be a non-empty string"
        span = span.strip()
        if span != "all":
            if not re.fullmatch(r"\d+", span):
                return normalized, "span must be 'all' or a positive integer"
            try:
                span_count = int(span)
            except (ValueError, OverflowError):
                return normalized, "span must be 'all' or a positive integer"
            if span_count <= 0:
                return normalized, "span must be 'all' or a positive integer"
        preserve_user_query = arguments.get("preserve_user_query", False)
        if not isinstance(preserve_user_query, bool):
            return normalized, "preserve_user_query must be a boolean"
        arguments["span"] = span
        arguments["preserve_user_query"] = preserve_user_query

    elif name == "Clear_context":
        criteria = arguments.get("criteria", "")
        if not isinstance(criteria, str) or not criteria.strip():
            return normalized, "criteria must be a non-empty string"
        arguments["criteria"] = criteria.strip()

    elif name == "Retrieve_memory":
        query = arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return normalized, "query must be a non-empty string"

        raw_top_k = arguments.get("top_k", 3)
        if isinstance(raw_top_k, bool):
            return normalized, "top_k must be a positive integer"
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError, OverflowError):
            return normalized, "top_k must be a positive integer"
        if top_k <= 0 or (isinstance(raw_top_k, float) and not raw_top_k.is_integer()):
            return normalized, "top_k must be a positive integer"

        metadata_filter = arguments.get("metadata_filter", {})
        if metadata_filter is None:
            metadata_filter = {}
        if not isinstance(metadata_filter, dict):
            return normalized, "metadata_filter must be an object"

        arguments["query"] = query.strip()
        arguments["top_k"] = top_k
        arguments["metadata_filter"] = dict(metadata_filter)

    elif name == "Add_memory":
        content = arguments.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return normalized, "content must be a non-empty string"

        metadata = arguments.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            return normalized, "metadata must be an object"

        memory_type = arguments.get("memory_type", "general")
        if not isinstance(memory_type, str):
            return normalized, "memory_type must be a string"

        arguments["content"] = content
        arguments["metadata"] = dict(metadata)
        arguments["memory_type"] = memory_type

    elif name == "Update_memory":
        memory_id = arguments.get("memory_id", "")
        if not isinstance(memory_id, str) or not memory_id.strip():
            return normalized, "memory_id must be a non-empty string"

        content = arguments.get("content")
        if content is not None and (
            not isinstance(content, str) or not content.strip()
        ):
            return normalized, "content must be a non-empty string when provided"

        metadata = arguments.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            return normalized, "metadata must be an object"
        if content is None and not metadata:
            return normalized, "content or metadata must be provided"

        arguments["memory_id"] = memory_id.strip()
        arguments["content"] = content
        arguments["metadata"] = dict(metadata)

    elif name == "Delete_memory":
        memory_id = arguments.get("memory_id", "")
        if not isinstance(memory_id, str) or not memory_id.strip():
            return normalized, "memory_id must be a non-empty string"

        confirmation = arguments.get("confirmation", False)
        if not isinstance(confirmation, bool):
            return normalized, "confirmation must be a boolean"

        arguments["memory_id"] = memory_id.strip()
        arguments["confirmation"] = confirmation

    return normalized, None


def parse_tool_calls(text: str) -> List[Dict]:
    """Parse tool calls from model text with tolerant patterns."""
    if not text or not text.strip():
        return []

    all_tool_calls = []

    result = _parse_all_standard_format(text)
    if result:
        all_tool_calls.extend(result)

    if not all_tool_calls:
        result = _parse_all_open_tag_only(text)
        if result:
            all_tool_calls.extend(result)

    if not all_tool_calls:
        result = _parse_close_tag_only(text)
        if result:
            all_tool_calls.extend(result)

    # Preserve each call exactly as emitted. Repeated calls in one JSON array
    # are separate attempts and must each execute and receive their own trace ID.
    return all_tool_calls


def _parse_all_standard_format(text: str) -> List[Dict]:
    """Parse all standard <tool_call>[...]</tool_call> blocks."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.finditer(pattern, text, re.DOTALL)

    all_calls: List[Dict] = []
    for match in matches:
        try:
            json_text = _extract_complete_json_array(match.group(1))
            if not json_text:
                continue
            calls = json.loads(json_text)
            if isinstance(calls, list):
                all_calls.extend(calls)
            elif isinstance(calls, dict):
                all_calls.append(calls)
        except json.JSONDecodeError:
            continue

    return all_calls


def _parse_all_open_tag_only(text: str) -> List[Dict]:
    """
    Parse cases where only the opening tag exists: <tool_call>[{...}]
    Supports multiple <tool_call> tags, and each may miss the closing tag.
    """
    tag_pattern = r"<tool_call>"
    tag_positions = [m.start() for m in re.finditer(tag_pattern, text)]

    if not tag_positions:
        return []

    all_calls: List[Dict] = []
    for i, start_pos in enumerate(tag_positions):
        end_pos = tag_positions[i + 1] if i + 1 < len(tag_positions) else len(text)
        segment = text[start_pos:end_pos]

        segment = segment.replace("<tool_call>", "", 1).strip()
        if "</tool_call>" in segment:
            segment = segment.split("</tool_call>")[0].strip()

        try:
            json_str = _extract_complete_json_array(segment)
            if json_str:
                calls = json.loads(json_str)
                if isinstance(calls, list):
                    all_calls.extend(calls)
                elif isinstance(calls, dict):
                    all_calls.append(calls)
        except (json.JSONDecodeError, ValueError):
            continue

    return all_calls


def _parse_close_tag_only(text: str) -> List[Dict]:
    """Parse one or more close-tag-only blocks: [{...}]</tool_call>."""
    all_calls: List[Dict] = []
    for segment in text.split("</tool_call>")[:-1]:
        try:
            json_text = _extract_complete_json_array(segment)
            if not json_text:
                continue
            calls = json.loads(json_text)
            if isinstance(calls, list):
                all_calls.extend(calls)
            elif isinstance(calls, dict):
                all_calls.append(calls)
        except json.JSONDecodeError:
            continue
    return all_calls


def _extract_complete_json_array(text: str) -> str:
    """Extract a complete JSON array from possibly truncated model output."""
    text = text.strip()

    start = text.find("[")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape = False
    end = -1

    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end != -1:
        return text[start : end + 1]

    chunk = text[start:]
    close_count = chunk.count("]")
    open_count = chunk.count("[")
    if open_count > close_count:
        chunk += "]" * (open_count - close_count)

    return chunk


def parse_answer(text: str) -> Optional[str]:
    """Extract final answer from <answer>...</answer> tag."""
    if not text:
        return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_score(text: str, default: float = 0.0) -> float:
    """Extract a float score from text (fallback to default)."""
    if not text:
        return default
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def create_tool_counter() -> Dict[str, int]:
    """Create a fresh counter dict for tool usage."""
    return dict(DEFAULT_TOOL_COUNTER)


def record_tool_usage(
    counter: Dict[str, int], tool_calls: List[Dict[str, Any]]
) -> None:
    """Update tool usage counters from parsed tool call list."""
    for tool_call in tool_calls:
        try:
            name = tool_call.get("name")
            if name in counter:
                counter[name] += 1
            if name in LONG_TERM_TOOL_NAMES:
                counter["long_term_memory"] += 1
            else:
                counter["short_term_memory"] += 1
        except Exception:
            continue


class DistractorGenerator:
    """Generate context-related or unrelated distractor messages."""

    def __init__(self, chat_client):
        self.chat_client = chat_client

    def generate_context_related_messages(
        self,
        context_info: Dict,
        num_messages: int = 5,
    ) -> List[str]:
        """
        Generate casual conversation that is related to context but not direct QA.
        """
        titles = context_info.get("title", [])
        sentences = context_info.get("sentences", [])

        if not titles or not sentences:
            return DEFAULT_DISTRACTOR_MESSAGES[:num_messages]

        context_summary = []
        for title, sents in zip(titles[:3], sentences[:3]):
            if sents:
                context_summary.append(f"{title}: {' '.join(sents[:2])}")

        prompt = f"""Based on the following context information:
{chr(10).join(context_summary)}

Generate {num_messages} casual conversation messages that are related to the topics but do NOT directly ask about specific facts from the context.
These should be natural, everyday conversation starters.

Format: One message per line, no numbering."""

        try:
            response = self.chat_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model_name="qwen-max",
            )

            messages: List[str] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    line = re.sub(r"^[-•*\d.]\s*", "", line)
                    if line:
                        messages.append(line)

            return (
                messages[:num_messages]
                if messages
                else DEFAULT_DISTRACTOR_MESSAGES[:num_messages]
            )
        except Exception:
            return DEFAULT_DISTRACTOR_MESSAGES[:num_messages]

    def generate_distractor_messages(
        self, question: str, num_messages: int = 5
    ) -> List[str]:
        """Generate messages unrelated to the target question."""
        prompt = f"""Given the question: "{question}"

Generate {num_messages} completely unrelated casual conversation messages that would distract from answering this question.
These should be about different topics entirely.

Format: One message per line, no numbering."""

        try:
            response = self.chat_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model_name="qwen-max",
            )

            messages: List[str] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    line = re.sub(r"^[-•*\d.]\s*", "", line)
                    if line:
                        messages.append(line)

            return (
                messages[:num_messages]
                if messages
                else DEFAULT_DISTRACTOR_MESSAGES[:num_messages]
            )
        except Exception:
            return DEFAULT_DISTRACTOR_MESSAGES[:num_messages]
