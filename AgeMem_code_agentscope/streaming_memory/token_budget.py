"""Exact tokenizer-bound accounting for context and serialized memory payloads."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class BudgetError(ValueError):
    """Raised before a prompt or persistent payload can exceed its hard budget."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DebugLexicalTokenizer:
    """Deterministic CPU fixture tokenizer; never valid for a model run."""

    name_or_path = "agemem-debug-unicode-lexical-v1"
    revision = "fixture-v1"
    chat_template = "<|{role}|>\n{content}\n"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))))

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        rendered = "".join(
            f"<|{item['role']}|>\n{item['content']}\n" for item in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return self.encode(rendered) if tokenize else rendered


def load_tokenizer(path: str | None, revision: str | None = None) -> Any:
    if path in (None, "", "debug-lexical"):
        return DebugLexicalTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - minimal install only
        raise BudgetError("transformers is required for a frozen model tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        str(Path(path).expanduser()),
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )


@dataclass(frozen=True)
class TokenAccounting:
    tokenizer: Any
    name: str
    revision: str

    @classmethod
    def from_tokenizer(
        cls, tokenizer: Any, *, name: str | None = None, revision: str | None = None
    ) -> "TokenAccounting":
        resolved_name = name or str(getattr(tokenizer, "name_or_path", "unknown"))
        resolved_revision = revision or str(getattr(tokenizer, "revision", "unresolved"))
        return cls(tokenizer=tokenizer, name=resolved_name, revision=resolved_revision)

    @property
    def chat_template_sha256(self) -> str:
        template = str(getattr(self.tokenizer, "chat_template", ""))
        return sha256_text(template)

    def count_text(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(ids)

    def render_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str):
            raise BudgetError("tokenizer chat template did not return text")
        return rendered

    def count_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return len(ids)

    def enforce_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        context_total_tokens: int,
    ) -> int:
        prompt_tokens = self.count_chat(messages)
        if prompt_tokens + max_new_tokens > context_total_tokens:
            raise BudgetError(
                "rendered prompt exceeds C: "
                f"prompt={prompt_tokens}, max_new={max_new_tokens}, "
                f"C={context_total_tokens}"
            )
        return prompt_tokens


def canonical_memory_payload(payload: Mapping[str, Any]) -> str:
    """Serialize every model-recoverable memory field counted by B.

    Environment-owned opaque identifiers are excluded. Content, titles, tags,
    custom metadata and source references are included, so metadata cannot be
    used as a free side channel.
    """

    allowed = {
        "content": payload.get("content", ""),
        "title": payload.get("title"),
        "tags": payload.get("tags") or [],
        "source_refs": payload.get("source_refs") or [],
        "custom": payload.get("custom") or {},
    }
    return json.dumps(
        allowed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def memory_payload_tokens(
    memories: Iterable[Mapping[str, Any]], accounting: TokenAccounting
) -> int:
    return sum(accounting.count_text(canonical_memory_payload(item)) for item in memories)


__all__ = [
    "BudgetError",
    "DebugLexicalTokenizer",
    "TokenAccounting",
    "canonical_memory_payload",
    "load_tokenizer",
    "memory_payload_tokens",
    "sha256_text",
]
