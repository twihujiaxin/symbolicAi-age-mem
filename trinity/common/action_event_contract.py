"""Fail-closed online action metadata for AgeMem on-policy rollouts.

The model response ``Experience`` is the source of truth for response token
IDs and sampling log-probabilities.  The workflow only records character
spans and tool results; the runner adds the frozen policy version after it has
verified that the version did not change while the rollout group was being
sampled.

Offline rule/oracle/error-injector trajectories deliberately have no training
metadata and are never eligible for the on-policy buffer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

from AgeMem_code_agentscope.action_schema import ActionCreditRecord, ActionEvent

if TYPE_CHECKING:
    from .experience import Experience


ACTION_CONTRACT_VERSION = "agemem.online_action_contract.v1"
ACTION_DRAFT_VERSION = "agemem.online_action_draft.v1"
ACTION_CONTRACT_KEY = "agemem_action_contract"
ACTION_DRAFTS_KEY = "agemem_action_event_drafts"
ACTION_EVENTS_KEY = "agemem_action_events"
ACTION_CHARACTER_SPANS_KEY = "agemem_action_character_spans"
ACTION_CREDITS_KEY = "agemem_action_credit_records"
TRAJECTORY_SOURCE_KEY = "agemem_trajectory_source"
ON_POLICY_ELIGIBLE_KEY = "agemem_on_policy_eligible"
RESPONSE_TOKEN_OFFSETS_KEY = "agemem_response_token_char_offsets"

OFF_POLICY_SOURCES = frozenset({"rule", "oracle", "random", "error_injector"})
TRUNCATED_TOOL_CALL_SPAN_ERROR = "truncated tool-call JSON has no exact character span"


class ActionContractError(ValueError):
    """Raised before corrupt action metadata can reach the on-policy buffer."""


@dataclass(frozen=True)
class ParsedToolCallSpan:
    """One parsed tool call and its exact half-open response character span."""

    call: Any
    char_start: int
    char_end: int


def _as_int_list(value: Any, *, name: str) -> list[int]:
    value = _to_plain_sequence(value)
    if not isinstance(value, (list, tuple)):
        raise ActionContractError(f"{name} must be a one-dimensional sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ActionContractError(f"{name} must contain integers")
        result.append(int(item))
    return result


def _as_float_list(value: Any, *, name: str) -> list[float]:
    value = _to_plain_sequence(value)
    if not isinstance(value, (list, tuple)):
        raise ActionContractError(f"{name} must be a one-dimensional sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ActionContractError(f"{name} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ActionContractError(f"{name} must contain finite numbers")
        result.append(number)
    return result


def _normalize_tokenizer_output(value: Any) -> list[Any]:
    value = _to_plain_sequence(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ActionContractError("tokenizer output must be a list")
    return value


def _to_plain_sequence(value: Any) -> Any:
    """Convert tensor-like values without importing a training framework.

    M1--M7 schema/replay tooling and lightweight workflow tests must remain
    importable without PyTorch.  Real tensors expose the same
    ``detach().cpu().tolist()`` chain, while ordinary lists pass through.
    """

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    return value


def _special_token_id_set(tokenizer: Any) -> set[int]:
    raw_ids = getattr(tokenizer, "all_special_ids", None)
    if raw_ids is None:
        return set()
    try:
        return {
            int(item)
            for item in raw_ids
            if not isinstance(item, bool) and isinstance(item, int)
        }
    except TypeError:
        return set()


def _offsets_with_skipped_specials(
    tokenizer: Any,
    token_ids: Sequence[int],
    response_text: str,
) -> tuple[tuple[int, int], ...] | None:
    """Map special tokens to zero-width spans when they are absent from text.

    vLLM records ``response_text`` with ``skip_special_tokens=True``.  Qwen3
    often inserts thinking/chat specials that survive in ``token_ids`` but not
    in the detokenized string.  Offsets remain exact: specials cover no
    characters, and the remaining IDs must re-encode ``response_text``.
    """

    special_ids = _special_token_id_set(tokenizer)
    if not special_ids or not any(token_id in special_ids for token_id in token_ids):
        return None
    try:
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return None
    if decoded != response_text:
        return None
    content_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    try:
        encoded = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = _normalize_tokenizer_output(encoded["input_ids"])
        raw_offsets = _normalize_tokenizer_output(encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError, ActionContractError):
        return None
    if encoded_ids != content_ids:
        return None
    content_offsets = [(int(item[0]), int(item[1])) for item in raw_offsets]
    if len(content_offsets) != len(content_ids):
        return None
    merged: list[tuple[int, int]] = []
    content_index = 0
    position = 0
    for token_id in token_ids:
        if token_id in special_ids:
            merged.append((position, position))
            continue
        start, end = content_offsets[content_index]
        if start != position:
            return None
        merged.append((start, end))
        position = end
        content_index += 1
    try:
        return _validate_response_offsets(merged, len(token_ids), len(response_text))
    except ActionContractError:
        return None


def _offsets_from_convert_tokens_to_string(
    tokenizer: Any,
    token_ids: Sequence[int],
    response_text: str,
) -> tuple[tuple[int, int], ...] | None:
    """Exact offsets when BPE decode(ids[:k]) is not a character prefix.

    Qwen3 byte-level tokens plus vLLM ``skip_special_tokens=True`` make
    incremental ``tokenizer.decode`` fail ``startswith``, even when the full
    detokenized string matches.  ``convert_tokens_to_string`` on the growing
    visible token list is monotonic and is accepted only when it equals
    ``response_text`` exactly.
    """

    convert_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
    to_string = getattr(tokenizer, "convert_tokens_to_string", None)
    if not callable(convert_ids) or not callable(to_string):
        return None
    special_ids = _special_token_id_set(tokenizer)
    special_tokens = {
        token
        for token in (getattr(tokenizer, "all_special_tokens", None) or ())
        if isinstance(token, str)
    }
    try:
        raw_tokens = convert_ids(list(token_ids))
    except TypeError:
        try:
            raw_tokens = [convert_ids(token_id) for token_id in token_ids]
        except TypeError:
            return None
    if isinstance(raw_tokens, str) or len(raw_tokens) != len(token_ids):
        return None
    prefixes = [""]
    visible: list[str] = []
    for token, token_id in zip(raw_tokens, token_ids):
        if not isinstance(token, str):
            return None
        if token_id in special_ids or token in special_tokens:
            prefixes.append(prefixes[-1])
            continue
        visible.append(token)
        try:
            piece = to_string(visible)
        except TypeError:
            return None
        if not isinstance(piece, str) or not response_text.startswith(piece):
            return None
        prefixes.append(piece)
    if prefixes[-1] != response_text:
        return None
    offsets = tuple(
        (len(prefixes[index]), len(prefixes[index + 1]))
        for index in range(len(token_ids))
    )
    try:
        return _validate_response_offsets(offsets, len(token_ids), len(response_text))
    except ActionContractError:
        return None


_GPT2_UNICODE_TO_BYTE: dict[str, int] | None = None


def _gpt2_unicode_to_byte() -> dict[str, int]:
    """Inverse of GPT-2 / Qwen ``bytes_to_unicode`` (byte-level BPE alphabet)."""

    global _GPT2_UNICODE_TO_BYTE
    if _GPT2_UNICODE_TO_BYTE is None:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        present = set(bs)
        cs = bs[:]
        n = 0
        for byte in range(256):
            if byte not in present:
                bs.append(byte)
                cs.append(256 + n)
                n += 1
        _GPT2_UNICODE_TO_BYTE = {chr(code): byte for byte, code in zip(bs, cs)}
    return _GPT2_UNICODE_TO_BYTE


def _unicode_to_byte_map(tokenizer: Any) -> dict[str, int]:
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    if isinstance(byte_decoder, dict) and byte_decoder:
        mapped: dict[str, int] = {}
        for key, value in byte_decoder.items():
            if not isinstance(key, str) or len(key) != 1:
                return _gpt2_unicode_to_byte()
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                return _gpt2_unicode_to_byte()
            mapped[key] = value
        if mapped:
            return mapped
    return _gpt2_unicode_to_byte()


def _utf8_complete_prefix(data: bytes) -> bytes:
    """Longest prefix of *data* that is valid UTF-8."""

    for extra in range(0, min(3, len(data)) + 1):
        end = len(data) - extra
        try:
            data[:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return data[:end]
    return b""


def _offsets_from_byte_level_tokens(
    tokenizer: Any,
    token_ids: Sequence[int],
    response_text: str,
) -> tuple[tuple[int, int], ...] | None:
    """Exact offsets for GPT-2 / Qwen byte-level BPE.

    Incremental ``decode(ids[:k])`` is not a character prefix when a UTF-8
    code point is split across tokens: incomplete bytes become U+FFFD, then
    disappear when the character completes.  Mapping each visible token back
    to bytes and emitting characters only when a UTF-8 sequence completes
    keeps contiguous exact spans.  Accepted only when the reconstructed
    string equals ``response_text``.
    """

    convert_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_ids):
        return None
    special_ids = _special_token_id_set(tokenizer)
    special_tokens = {
        token
        for token in (getattr(tokenizer, "all_special_tokens", None) or ())
        if isinstance(token, str)
    }
    try:
        raw_tokens = convert_ids(list(token_ids))
    except TypeError:
        try:
            raw_tokens = [convert_ids(token_id) for token_id in token_ids]
        except TypeError:
            return None
    if isinstance(raw_tokens, str) or len(raw_tokens) != len(token_ids):
        return None
    unicode_to_byte = _unicode_to_byte_map(tokenizer)
    byte_chunks: list[bytes] = []
    for token, token_id in zip(raw_tokens, token_ids):
        if not isinstance(token, str):
            return None
        if token_id in special_ids or token in special_tokens:
            byte_chunks.append(b"")
            continue
        try:
            byte_chunks.append(bytes(unicode_to_byte[character] for character in token))
        except KeyError:
            return None
    raw = b"".join(byte_chunks)
    try:
        reconstructed = raw.decode("utf-8")
        replace_tail = False
    except UnicodeDecodeError:
        reconstructed = raw.decode("utf-8", errors="replace")
        replace_tail = True
    if reconstructed != response_text:
        return None
    offsets: list[tuple[int, int]] = []
    pending = b""
    char_pos = 0
    for chunk in byte_chunks:
        pending += chunk
        complete = _utf8_complete_prefix(pending)
        new_text = complete.decode("utf-8")
        start = char_pos
        char_pos += len(new_text)
        offsets.append((start, char_pos))
        pending = pending[len(complete) :]
    if pending:
        if not replace_tail:
            return None
        replacement = pending.decode("utf-8", errors="replace")
        owner = max(
            (index for index, chunk in enumerate(byte_chunks) if chunk),
            default=None,
        )
        if owner is None or not replacement:
            return None
        start, _ = offsets[owner]
        char_pos += len(replacement)
        offsets[owner] = (start, char_pos)
        for index in range(owner + 1, len(offsets)):
            offsets[index] = (char_pos, char_pos)
    try:
        return _validate_response_offsets(offsets, len(token_ids), len(response_text))
    except ActionContractError:
        return None


def derive_response_token_char_offsets(
    tokenizer: Any,
    response_token_ids: Sequence[int],
    response_text: str,
) -> tuple[tuple[int, int], ...]:
    """Derive exact token/character alignment from the generation tokenizer.

    Fast-tokenizer offsets are preferred and accepted only when re-encoding the
    response produces the exact generated token IDs.  Byte-level BPE (Qwen3)
    uses UTF-8 completion so a character split across tokens still gets
    exact contiguous spans.  A strict prefix-decode fallback supports
    simple/slow tokenizers.  Approximate offsets are never fabricated.
    """

    token_ids = _as_int_list(response_token_ids, name="response_token_ids")
    if not isinstance(response_text, str):
        raise ActionContractError("response_text must be a string")
    if not token_ids:
        raise ActionContractError("response_token_ids must not be empty")

    try:
        encoded = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = _normalize_tokenizer_output(encoded["input_ids"])
        raw_offsets = _normalize_tokenizer_output(encoded["offset_mapping"])
        offsets = tuple((int(item[0]), int(item[1])) for item in raw_offsets)
        if encoded_ids == token_ids:
            _validate_response_offsets(offsets, len(token_ids), len(response_text))
            return offsets
    except (ActionContractError, KeyError, TypeError, ValueError):
        pass

    special_offsets = _offsets_with_skipped_specials(tokenizer, token_ids, response_text)
    if special_offsets is not None:
        return special_offsets

    string_offsets = _offsets_from_convert_tokens_to_string(
        tokenizer, token_ids, response_text
    )
    if string_offsets is not None:
        return string_offsets

    byte_offsets = _offsets_from_byte_level_tokens(tokenizer, token_ids, response_text)
    if byte_offsets is not None:
        return byte_offsets

    decoded_prefixes: list[str] = [""]
    for end in range(1, len(token_ids) + 1):
        try:
            decoded = tokenizer.decode(
                token_ids[:end],
                # vLLM records response_text with SamplingParams'
                # skip_special_tokens=True.  A sampled terminal special token
                # therefore receives an honest zero-width offset at the end.
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(token_ids[:end])
        if not isinstance(decoded, str) or not response_text.startswith(decoded):
            raise ActionContractError(
                "cannot derive exact response token character offsets"
            )
        decoded_prefixes.append(decoded)
    if decoded_prefixes[-1] != response_text:
        raise ActionContractError(
            "generated token IDs do not decode to the recorded response_text"
        )
    offsets = tuple(
        (len(decoded_prefixes[index]), len(decoded_prefixes[index + 1]))
        for index in range(len(token_ids))
    )
    _validate_response_offsets(offsets, len(token_ids), len(response_text))
    return offsets


def _validate_response_offsets(
    offsets: Sequence[Sequence[int]],
    token_count: int,
    text_length: int,
) -> tuple[tuple[int, int], ...]:
    if len(offsets) != token_count:
        raise ActionContractError(
            "response token offsets length must equal response token count"
        )
    normalized: list[tuple[int, int]] = []
    previous_end = 0
    for raw in offsets:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ActionContractError("each response token offset must be [start, end]")
        start, end = raw
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ActionContractError("response token offsets must be integers")
        if start != previous_end or not start <= end <= text_length:
            raise ActionContractError(
                "response token offsets must be contiguous and within response_text"
            )
        normalized.append((start, end))
        previous_end = end
    if previous_end != text_length:
        raise ActionContractError(
            "response token offsets must cover response_text exactly"
        )
    return tuple(normalized)


def response_metadata_for_generation(
    tokenizer: Any,
    response_token_ids: Sequence[int],
    response_text: str,
) -> dict[str, Any]:
    """Build JSON/pickle-safe metadata at the model generation boundary."""

    return {
        RESPONSE_TOKEN_OFFSETS_KEY: [
            [start, end]
            for start, end in derive_response_token_char_offsets(
                tokenizer, response_token_ids, response_text
            )
        ]
    }


def _balanced_array_end(text: str, start: int) -> Optional[int]:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _array_element_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return top-level JSON array element ranges, excluding separators."""

    ranges: list[tuple[int, int]] = []
    element_start = start + 1
    object_depth = 0
    array_depth = 1
    in_string = False
    escaped = False
    for index in range(start + 1, end - 1):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            object_depth += 1
        elif character == "}":
            object_depth -= 1
        elif character == "[":
            array_depth += 1
        elif character == "]":
            array_depth -= 1
        elif character == "," and object_depth == 0 and array_depth == 1:
            ranges.append((element_start, index))
            element_start = index + 1
    ranges.append((element_start, end - 1))
    normalized = []
    for raw_start, raw_end in ranges:
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_start < raw_end:
            normalized.append((raw_start, raw_end))
    return normalized


def _calls_from_segment(
    text: str, segment_start: int, segment_end: int
) -> list[ParsedToolCallSpan]:
    array_start = text.find("[", segment_start, segment_end)
    if array_start < 0:
        return []
    array_end = _balanced_array_end(text, array_start)
    if array_end is None or array_end > segment_end:
        # The tolerant execution parser may repair a missing closing bracket.
        # Such a synthetic character has no honest response/token span.
        raise ActionContractError(TRUNCATED_TOOL_CALL_SPAN_ERROR)
    calls: list[ParsedToolCallSpan] = []
    for call_start, call_end in _array_element_ranges(text, array_start, array_end):
        try:
            call = json.loads(text[call_start:call_end])
        except json.JSONDecodeError as exc:
            raise ActionContractError("tool-call element is not valid JSON") from exc
        calls.append(
            ParsedToolCallSpan(
                call=call,
                char_start=call_start,
                char_end=call_end,
            )
        )
    return calls


def parse_tool_calls_with_char_spans(text: str) -> tuple[ParsedToolCallSpan, ...]:
    """Mirror AgeMem's tolerant parser while retaining exact action spans."""

    if not isinstance(text, str) or not text.strip():
        return ()

    standard_matches = tuple(
        re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    )
    standard_calls: list[ParsedToolCallSpan] = []
    for match in standard_matches:
        standard_calls.extend(_calls_from_segment(text, match.start(1), match.end(1)))
    if standard_calls:
        return tuple(standard_calls)

    open_positions = [match.start() for match in re.finditer(r"<tool_call>", text)]
    open_calls: list[ParsedToolCallSpan] = []
    for index, tag_start in enumerate(open_positions):
        content_start = tag_start + len("<tool_call>")
        content_end = (
            open_positions[index + 1] if index + 1 < len(open_positions) else len(text)
        )
        close = text.find("</tool_call>", content_start, content_end)
        if close >= 0:
            content_end = close
        open_calls.extend(_calls_from_segment(text, content_start, content_end))
    if open_calls:
        return tuple(open_calls)

    close_calls: list[ParsedToolCallSpan] = []
    segment_start = 0
    for close_match in re.finditer(r"</tool_call>", text):
        close_calls.extend(
            _calls_from_segment(text, segment_start, close_match.start())
        )
        segment_start = close_match.end()
    return tuple(close_calls)


def _response_arrays(experience: Experience) -> tuple[list[int], list[float]]:
    if experience.tokens is None:
        raise ActionContractError("Experience.tokens is required")
    tokens = _as_int_list(experience.tokens, name="Experience.tokens")
    if not 0 < experience.prompt_length < len(tokens):
        raise ActionContractError("Experience.prompt_length is outside token bounds")
    response_ids = tokens[experience.prompt_length :]
    if experience.logprobs is None:
        raise ActionContractError("Experience.logprobs is required for LLM actions")
    old_logprobs = _as_float_list(experience.logprobs, name="Experience.logprobs")
    if len(response_ids) != len(old_logprobs):
        raise ActionContractError(
            "response_token_ids and old_logprobs must have identical lengths"
        )
    if experience.action_mask is not None and len(experience.action_mask) != len(
        response_ids
    ):
        raise ActionContractError(
            "Experience.action_mask length must equal response token count"
        )
    return response_ids, old_logprobs


def _token_span_for_char_span(
    offsets: Sequence[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    covered = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > char_start and token_start < char_end
    ]
    if not covered:
        raise ActionContractError("action character span maps to no response tokens")
    return covered[0], covered[-1] + 1


def prepare_experience_action_drafts(
    experience: Experience,
    *,
    stage_id: int,
    timestep: int,
    assistant_turn_id: int,
) -> None:
    """Attach validated, policy-version-free action drafts to an Experience."""

    if min(stage_id, timestep, assistant_turn_id) < 0:
        raise ActionContractError("action coordinates must be non-negative")
    if getattr(getattr(experience, "eid", None), "step", None) != timestep:
        raise ActionContractError(
            "action timestep must equal the source Experience EID step"
        )
    response_text = experience.response_text
    if not isinstance(response_text, str):
        raise ActionContractError("Experience.response_text is required")
    try:
        parsed = parse_tool_calls_with_char_spans(response_text)
    except ActionContractError as exc:
        if str(exc) != TRUNCATED_TOOL_CALL_SPAN_ERROR:
            raise
        # Hitting the generation budget mid-tool-call is a real model outcome.
        # It must not receive a fake span, and it must not abort the workflow.
        parsed = ()
    info: MutableMapping[str, Any] = dict(experience.info or {})
    if ACTION_DRAFTS_KEY in info or ACTION_EVENTS_KEY in info:
        raise ActionContractError("Experience already contains action metadata")
    for key, expected in (("trace_stage", stage_id), ("trace_step", timestep)):
        if key in info and info[key] != expected:
            raise ActionContractError(f"{key} differs from the action coordinate")
        info[key] = expected

    info[ACTION_CONTRACT_KEY] = ACTION_CONTRACT_VERSION
    info[TRAJECTORY_SOURCE_KEY] = "llm"
    info[ON_POLICY_ELIGIBLE_KEY] = False
    if not parsed:
        info[ACTION_DRAFTS_KEY] = []
        experience.info = dict(info)
        return

    response_ids, old_logprobs = _response_arrays(experience)
    raw_offsets = info.get(RESPONSE_TOKEN_OFFSETS_KEY)
    if raw_offsets is None:
        raise ActionContractError(
            f"Experience.info[{RESPONSE_TOKEN_OFFSETS_KEY!r}] is required"
        )
    offsets = _validate_response_offsets(
        raw_offsets, len(response_ids), len(response_text)
    )

    drafts: list[dict[str, Any]] = []
    previous_token_end = -1
    for action_index, item in enumerate(parsed):
        token_start, token_end = _token_span_for_char_span(
            offsets, item.char_start, item.char_end
        )
        if token_start < previous_token_end:
            raise ActionContractError("action token spans overlap")
        previous_token_end = token_end
        if experience.action_mask is not None:
            action_mask = _to_plain_sequence(
                experience.action_mask[token_start:token_end]
            )
            if not isinstance(action_mask, (list, tuple)) or not all(
                bool(item) for item in action_mask
            ):
                raise ActionContractError(
                    "action span contains masked non-model tokens"
                )
        if isinstance(item.call, dict):
            raw_action_type = item.call.get("name")
            action_type = (
                raw_action_type
                if isinstance(raw_action_type, str) and raw_action_type
                else str(raw_action_type)
            )
            if not action_type:
                action_type = "<invalid_tool_call>"
            arguments = item.call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"raw_arguments": arguments}
        else:
            action_type = "<invalid_tool_call>"
            arguments = {"raw_call": item.call}
        drafts.append(
            {
                "schema_version": ACTION_DRAFT_VERSION,
                "stage_id": stage_id,
                "timestep": timestep,
                "assistant_turn_id": assistant_turn_id,
                "action_index_in_turn": action_index,
                "action_type": action_type,
                "action_text": response_text[item.char_start : item.char_end],
                "arguments": dict(arguments),
                "result": None,
                "char_start": item.char_start,
                "char_end": item.char_end,
                "response_token_ids": list(response_ids),
                "token_start": token_start,
                "token_end": token_end,
                "old_logprobs": list(old_logprobs),
            }
        )
    info[ACTION_DRAFTS_KEY] = drafts
    experience.info = dict(info)


def record_experience_action_result(
    experiences: Sequence[Experience],
    *,
    action_index_in_turn: int,
    trace_call_id: str,
    action_type: str,
    status: str,
    result: Mapping[str, Any],
    error: Optional[str] = None,
) -> None:
    """Complete exactly one prepared draft with its executed tool result."""

    for experience in experiences:
        info = dict(experience.info or {})
        if ACTION_DRAFTS_KEY not in info:
            continue
        drafts = [dict(item) for item in info[ACTION_DRAFTS_KEY]]
        if not 0 <= action_index_in_turn < len(drafts):
            raise ActionContractError(
                "executed tool call has no one-to-one prepared action draft"
            )
        draft = drafts[action_index_in_turn]
        if draft["action_type"] != action_type and not (
            draft["action_type"] == "<invalid_tool_call>" and not action_type
        ):
            raise ActionContractError("executed tool name differs from action draft")
        if draft.get("result") is not None:
            raise ActionContractError("action draft result was already recorded")
        draft["result"] = {
            "trace_call_id": trace_call_id,
            "status": status,
            "output": dict(result),
            "error": error,
        }
        drafts[action_index_in_turn] = draft
        info[ACTION_DRAFTS_KEY] = drafts
        experience.info = info


def stable_action_id(
    *,
    rollout_id: str,
    stage_id: int,
    timestep: int,
    assistant_turn_id: int,
    action_index_in_turn: int,
) -> str:
    """Return a deterministic identity for one action coordinate."""

    payload = json.dumps(
        {
            "rollout_id": rollout_id,
            "stage_id": stage_id,
            "timestep": timestep,
            "assistant_turn_id": assistant_turn_id,
            "action_index_in_turn": action_index_in_turn,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"agemem-act-{hashlib.sha256(payload).hexdigest()[:24]}"


def freeze_rollout_policy_version(before: Any, after: Any) -> str:
    """Validate one K-rollout group used a single policy checkpoint."""

    if (
        isinstance(before, bool)
        or not isinstance(before, int)
        or isinstance(after, bool)
        or not isinstance(after, int)
    ):
        raise ActionContractError("rollout model versions must be integers")
    if before != after:
        raise ActionContractError(
            "rollout policy version changed while sampling one K-rollout "
            f"group: {before!r} -> {after!r}"
        )
    return f"model_version:{before}"


def finalize_experience_action_contract(
    experience: Experience, *, policy_version: str
) -> tuple[ActionEvent, ...]:
    """Convert completed drafts into strict M6 ``ActionEvent`` records."""

    info = dict(experience.info or {})
    if ACTION_DRAFTS_KEY not in info:
        return ()
    if not isinstance(policy_version, str) or not policy_version:
        raise ActionContractError("policy_version must be a non-empty string")
    if info.get(TRAJECTORY_SOURCE_KEY) != "llm":
        raise ActionContractError("only LLM drafts can become on-policy ActionEvents")
    raw_drafts = info[ACTION_DRAFTS_KEY]
    if not isinstance(raw_drafts, list):
        raise ActionContractError("action drafts must be a list")
    trace_stage = info.get("trace_stage")
    trace_step = info.get("trace_step")
    if (
        isinstance(trace_stage, bool)
        or not isinstance(trace_stage, int)
        or isinstance(trace_step, bool)
        or not isinstance(trace_step, int)
        or experience.eid.step != trace_step
    ):
        raise ActionContractError(
            "trace stage/step must be integers and match the Experience EID"
        )

    task_id = experience.eid.tid
    rollout_id = experience.eid.rid
    events: list[ActionEvent] = []
    character_spans: list[dict[str, Any]] = []
    for expected_index, draft in enumerate(raw_drafts):
        if (
            not isinstance(draft, dict)
            or draft.get("schema_version") != ACTION_DRAFT_VERSION
        ):
            raise ActionContractError("invalid action draft schema")
        if draft.get("action_index_in_turn") != expected_index:
            raise ActionContractError("action draft indices must be contiguous")
        if draft.get("stage_id") != trace_stage or draft.get("timestep") != trace_step:
            raise ActionContractError(
                "action draft coordinates differ from the source Experience trace"
            )
        if draft.get("result") is None:
            raise ActionContractError("action draft is missing its tool result")
        action_id = stable_action_id(
            rollout_id=rollout_id,
            stage_id=draft["stage_id"],
            timestep=draft["timestep"],
            assistant_turn_id=draft["assistant_turn_id"],
            action_index_in_turn=expected_index,
        )
        event = ActionEvent(
            action_id=action_id,
            task_id=task_id,
            rollout_id=rollout_id,
            stage_id=draft["stage_id"],
            timestep=draft["timestep"],
            assistant_turn_id=draft["assistant_turn_id"],
            action_index_in_turn=expected_index,
            source="llm",
            action_type=draft["action_type"],
            action_text=draft["action_text"],
            arguments=draft["arguments"],
            result=draft["result"],
            response_token_ids=tuple(draft["response_token_ids"]),
            token_start=draft["token_start"],
            token_end=draft["token_end"],
            old_logprobs=tuple(draft["old_logprobs"]),
            policy_version=policy_version,
        )
        events.append(event)
        character_spans.append(
            {
                "action_id": action_id,
                "char_start": draft["char_start"],
                "char_end": draft["char_end"],
            }
        )

    _validate_tool_trace_join(events, info.get("tool_call_ids", []))

    info.pop(ACTION_DRAFTS_KEY, None)
    info[ACTION_EVENTS_KEY] = [event.model_dump(mode="json") for event in events]
    info[ACTION_CHARACTER_SPANS_KEY] = character_spans
    info[ON_POLICY_ELIGIBLE_KEY] = True
    info["policy_version"] = policy_version
    experience.info = info
    validate_on_policy_experiences([experience])
    return tuple(events)


def _load_action_events(info: Mapping[str, Any]) -> tuple[ActionEvent, ...]:
    raw_events = info.get(ACTION_EVENTS_KEY, [])
    if not isinstance(raw_events, list):
        raise ActionContractError("ActionEvents must be stored as a list")
    try:
        return tuple(
            ActionEvent.model_validate_json(
                json.dumps(item, ensure_ascii=False, allow_nan=False)
            )
            for item in raw_events
        )
    except Exception as exc:
        raise ActionContractError("invalid stored ActionEvent") from exc


def _validate_tool_trace_join(
    events: Sequence[ActionEvent], raw_call_ids: Any
) -> None:
    if not isinstance(raw_call_ids, list) or len(raw_call_ids) != len(events):
        raise ActionContractError(
            "ActionEvent and executed tool result counts are not one-to-one"
        )
    if any(not isinstance(call_id, str) or not call_id for call_id in raw_call_ids):
        raise ActionContractError("tool trace call IDs must be non-empty strings")
    if len(raw_call_ids) != len(set(raw_call_ids)):
        raise ActionContractError("tool trace call IDs must be unique")
    for event, trace_call_id in zip(events, raw_call_ids):
        if event.result.get("trace_call_id") != trace_call_id:
            raise ActionContractError(
                "ActionEvent result does not join its tool trace call one-to-one"
            )


def join_action_events_to_credits(
    events: Iterable[ActionEvent],
    credits: Iterable[ActionCreditRecord | Mapping[str, Any]],
) -> tuple[tuple[ActionEvent, ActionCreditRecord], ...]:
    """Perform an exact, unique ``action_id`` join or fail closed."""

    event_sequence = tuple(events)
    credit_sequence = tuple(credits)
    event_by_id: dict[str, ActionEvent] = {}
    for event in event_sequence:
        if event.action_id in event_by_id:
            raise ActionContractError(
                f"duplicate ActionEvent action_id {event.action_id!r}"
            )
        event_by_id[event.action_id] = event
    credit_by_id: dict[str, ActionCreditRecord] = {}
    for raw_credit in credit_sequence:
        credit = (
            raw_credit
            if isinstance(raw_credit, ActionCreditRecord)
            else ActionCreditRecord.model_validate_json(
                json.dumps(raw_credit, ensure_ascii=False, allow_nan=False)
            )
        )
        if credit.action_id in credit_by_id:
            raise ActionContractError(
                f"duplicate ActionCreditRecord action_id {credit.action_id!r}"
            )
        credit_by_id[credit.action_id] = credit
    if set(event_by_id) != set(credit_by_id):
        raise ActionContractError(
            "ActionEvent and ActionCreditRecord action_id sets must match exactly"
        )
    joined = []
    for event in event_sequence:
        credit = credit_by_id[event.action_id]
        if (
            event.task_id,
            event.rollout_id,
            event.stage_id,
            event.timestep,
        ) != (
            credit.task_id,
            credit.rollout_id,
            credit.stage_id,
            credit.timestep,
        ):
            raise ActionContractError(
                f"ActionEvent/credit coordinates differ for {event.action_id!r}"
            )
        joined.append((event, credit))
    return tuple(joined)


def validate_on_policy_experiences(
    experiences: Sequence[Experience], *, require_contract: bool = False
) -> None:
    """Validate action contracts immediately before any buffer write."""

    seen_action_ids: set[str] = set()
    policy_versions_by_task: dict[str, set[str]] = {}
    for experience in experiences:
        info = experience.info or {}
        source = info.get(TRAJECTORY_SOURCE_KEY)
        if source in OFF_POLICY_SOURCES:
            raise ActionContractError(
                f"{source} trajectories are forbidden in the on-policy buffer"
            )
        if ACTION_DRAFTS_KEY in info:
            raise ActionContractError(
                "unfinished action drafts are forbidden in the on-policy buffer"
            )
        has_contract_version = ACTION_CONTRACT_KEY in info
        has_action_events = ACTION_EVENTS_KEY in info
        if not has_contract_version and not has_action_events:
            if require_contract:
                raise ActionContractError(
                    "AgeMem on-policy Experience is missing its action contract"
                )
            continue
        if has_contract_version != has_action_events:
            raise ActionContractError("online action contract is incomplete")
        if info.get(ACTION_CONTRACT_KEY) != ACTION_CONTRACT_VERSION:
            raise ActionContractError("unknown online action contract version")
        if info.get(ON_POLICY_ELIGIBLE_KEY) is not True or source != "llm":
            raise ActionContractError("Experience is not explicitly on-policy eligible")
        events = _load_action_events(info)
        trace_stage = info.get("trace_stage")
        trace_step = info.get("trace_step")
        if (
            isinstance(trace_stage, bool)
            or not isinstance(trace_stage, int)
            or isinstance(trace_step, bool)
            or not isinstance(trace_step, int)
            or experience.eid.step != trace_step
        ):
            raise ActionContractError(
                "trace stage/step must be integers and match the Experience EID"
            )
        _validate_tool_trace_join(events, info.get("tool_call_ids", []))
        if tuple(event.action_index_in_turn for event in events) != tuple(
            range(len(events))
        ):
            raise ActionContractError(
                "ActionEvent action_index_in_turn values must be contiguous"
            )
        if len({event.assistant_turn_id for event in events}) > 1:
            raise ActionContractError(
                "one Experience may contain only one assistant turn's actions"
            )
        response_ids, old_logprobs = _response_arrays(experience)
        response_text = experience.response_text
        if not isinstance(response_text, str):
            raise ActionContractError("Experience.response_text is required")
        raw_offsets = info.get(RESPONSE_TOKEN_OFFSETS_KEY)
        if raw_offsets is None:
            raise ActionContractError(
                f"Experience.info[{RESPONSE_TOKEN_OFFSETS_KEY!r}] is required"
            )
        offsets = _validate_response_offsets(
            raw_offsets, len(response_ids), len(response_text)
        )
        policy_version = info.get("policy_version")
        if not isinstance(policy_version, str) or not policy_version:
            raise ActionContractError(
                "on-policy Experience requires a non-empty policy_version"
            )
        policy_versions_by_task.setdefault(experience.eid.tid, set()).add(
            policy_version
        )
        if "model_version" in info and policy_version != (
            f"model_version:{info['model_version']}"
        ):
            raise ActionContractError(
                "Experience policy_version differs from rollout model_version"
            )
        previous_by_turn: dict[int, int] = {}
        for event in events:
            if event.source != "llm":
                raise ActionContractError("only LLM ActionEvents may enter on-policy")
            if event.action_id in seen_action_ids:
                raise ActionContractError(
                    f"duplicate on-policy action_id {event.action_id!r}"
                )
            seen_action_ids.add(event.action_id)
            if (
                event.task_id != experience.eid.tid
                or event.rollout_id != experience.eid.rid
            ):
                raise ActionContractError(
                    "ActionEvent identity differs from Experience EID"
                )
            if event.stage_id != trace_stage or event.timestep != trace_step:
                raise ActionContractError(
                    "ActionEvent coordinates differ from the Experience trace"
                )
            if event.action_id != stable_action_id(
                rollout_id=event.rollout_id,
                stage_id=event.stage_id,
                timestep=event.timestep,
                assistant_turn_id=event.assistant_turn_id,
                action_index_in_turn=event.action_index_in_turn,
            ):
                raise ActionContractError("ActionEvent action_id is not deterministic")
            if list(event.response_token_ids or ()) != response_ids:
                raise ActionContractError(
                    "ActionEvent response_token_ids differ from Experience"
                )
            if len(event.old_logprobs or ()) != len(old_logprobs) or any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(event.old_logprobs or (), old_logprobs)
            ):
                raise ActionContractError(
                    "ActionEvent old_logprobs differ from Experience"
                )
            if event.policy_version != policy_version:
                raise ActionContractError(
                    "ActionEvent policy_version differs from Experience"
                )
            assert event.token_start is not None and event.token_end is not None
            previous_end = previous_by_turn.get(event.assistant_turn_id, -1)
            if event.token_start < previous_end:
                raise ActionContractError("ActionEvent token spans overlap")
            previous_by_turn[event.assistant_turn_id] = event.token_end

        raw_spans = info.get(ACTION_CHARACTER_SPANS_KEY, [])
        if not isinstance(raw_spans, list) or len(raw_spans) != len(events):
            raise ActionContractError(
                "ActionEvent and character span counts must match exactly"
            )
        previous_char_end = -1
        for event, span in zip(events, raw_spans):
            if not isinstance(span, dict) or span.get("action_id") != event.action_id:
                raise ActionContractError("character span does not join ActionEvent")
            start, end = span.get("char_start"), span.get("char_end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(response_text)
                or start < previous_char_end
                or response_text[start:end] != event.action_text
            ):
                raise ActionContractError("ActionEvent character span is invalid")
            expected_token_span = _token_span_for_char_span(offsets, start, end)
            if expected_token_span != (event.token_start, event.token_end):
                raise ActionContractError(
                    "ActionEvent token span does not match its character span"
                )
            previous_char_end = end

        if ACTION_CREDITS_KEY in info:
            raw_credits = info[ACTION_CREDITS_KEY]
            if not isinstance(raw_credits, list):
                raise ActionContractError(
                    "ActionCreditRecords must be stored as a list"
                )
            join_action_events_to_credits(events, raw_credits)

    mixed_tasks = sorted(
        task_id
        for task_id, versions in policy_versions_by_task.items()
        if len(versions) != 1
    )
    if mixed_tasks:
        raise ActionContractError(
            "one grouped task contains multiple rollout policy versions: "
            + ", ".join(mixed_tasks)
        )


def mark_experience_off_policy(experience: Experience, *, source: str) -> None:
    """Explicitly mark a synthetic/offline Experience as buffer-ineligible."""

    if source not in OFF_POLICY_SOURCES:
        raise ActionContractError(f"unsupported offline trajectory source {source!r}")
    info = dict(experience.info or {})
    info[TRAJECTORY_SOURCE_KEY] = source
    info[ON_POLICY_ELIGIBLE_KEY] = False
    experience.info = info


__all__ = [
    "ACTION_CHARACTER_SPANS_KEY",
    "ACTION_CONTRACT_KEY",
    "ACTION_CONTRACT_VERSION",
    "ACTION_CREDITS_KEY",
    "ACTION_DRAFTS_KEY",
    "ACTION_EVENTS_KEY",
    "ON_POLICY_ELIGIBLE_KEY",
    "OFF_POLICY_SOURCES",
    "RESPONSE_TOKEN_OFFSETS_KEY",
    "TRAJECTORY_SOURCE_KEY",
    "TRUNCATED_TOOL_CALL_SPAN_ERROR",
    "ActionContractError",
    "ParsedToolCallSpan",
    "derive_response_token_char_offsets",
    "finalize_experience_action_contract",
    "freeze_rollout_policy_version",
    "join_action_events_to_credits",
    "mark_experience_off_policy",
    "parse_tool_calls_with_char_spans",
    "prepare_experience_action_drafts",
    "record_experience_action_result",
    "response_metadata_for_generation",
    "stable_action_id",
    "validate_on_policy_experiences",
]
