# -*- coding: utf-8 -*-
"""Pre-print hook so finish_function is shown as text reply."""
from __future__ import annotations

from typing import Any

from agentscope.message import TextBlock


def finish_function_pre_print_hook(
    self: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Replace finish_function tool_use block with its response text for display."""
    msg = kwargs["msg"]
    if isinstance(msg.content, str):
        return None
    if isinstance(msg.content, list):
        for i, block in enumerate(msg.content):
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == self.finish_function_name
            ):
                try:
                    msg.content[i] = TextBlock(
                        type="text",
                        text=(block.get("input") or {}).get("response", ""),
                    )
                    return kwargs
                except Exception:
                    pass
    return None
