"""Strict public action grammar; no repairs, aliases or private supervision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PUBLIC_ACTIONS = frozenset({"ADD", "UPDATE", "DELETE", "RETRIEVE", "NEXT"})


@dataclass(frozen=True)
class PublicActionParse:
    name: str | None
    arguments: dict[str, Any]
    code: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant")


def parse_public_action(text: str) -> PublicActionParse:
    def failure(code):
        return PublicActionParse(None, {}, code)

    if not isinstance(text, str) or not text.strip():
        return failure("empty_response")
    payload = text.strip()
    if payload.startswith("<tool_call>") and payload.endswith("</tool_call>"):
        payload = payload[len("<tool_call>"):-len("</tool_call>")].strip()
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return failure("invalid_json")
    if not isinstance(value, list):
        return failure("not_array")
    if len(value) != 1:
        return failure("action_count_not_one")
    call = value[0]
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return failure("invalid_envelope")
    name, arguments = call["name"], call["arguments"]
    if not isinstance(name, str) or name not in PUBLIC_ACTIONS:
        return failure("unknown_action")
    if not isinstance(arguments, dict):
        return failure("arguments_not_object")
    if "type" in arguments:
        return failure("reserved_type_argument")
    if name == "NEXT" and arguments:
        return failure("next_arguments_not_empty")
    return PublicActionParse(name, dict(arguments), "ok")


def format_error_feedback(code: str) -> str:
    reasons = {
        "empty_response": "Empty response.",
        "invalid_json": "Complete valid JSON required; no prose or missing brackets.",
        "not_array": "An array is required, not a single object or ACTION-key object.",
        "action_count_not_one": "Exactly one array element required; no batches.",
        "invalid_envelope": "Only name and arguments keys required.",
        "unknown_action": "name must be ADD/UPDATE/DELETE/RETRIEVE/NEXT, never ACTION.",
        "arguments_not_object": "arguments must be an object.",
        "reserved_type_argument": "Remove arguments.type; choose the action in name only.",
        "next_arguments_not_empty": "NEXT requires empty arguments.",
    }
    reason = reasons.get(code, "Invalid public action.")
    return ('ERROR ' + reason + ' Format: [{"name":"NEXT","arguments":{}}]. '
            'For ADD, use name="ADD" and memory_id/content/source_refs in arguments.')
