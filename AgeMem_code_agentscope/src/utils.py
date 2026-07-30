# -*- coding: utf-8 -*-
"""Utility functions for AgeMem agent."""
from __future__ import annotations

import json
import re
from typing import Any, List, Union


def extract_score(text: str, default: float = 0.0) -> float:
    """Extract a score in [0, 1] from LLM output."""
    try:
        s = (text or "").strip()
        if not s:
            return default
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ["score", "similarity", "value", "confidence"]:
                    if key in obj:
                        try:
                            f = float(obj[key])
                            if 0.0 <= f <= 1.0:
                                return f
                            if 1.0 < f <= 100.0:
                                return max(0.0, min(1.0, f / 100.0))
                        except Exception:
                            pass
        except Exception:
            pass
        m = re.search(r"(\d+\.?\d*)\s*%", s)
        if m:
            return max(0.0, min(1.0, float(m.group(1)) / 100.0))
        m = re.search(r"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2)) or 1.0
            if den != 0:
                return max(0.0, min(1.0, num / den))
        for m in re.finditer(r"\d+\.?\d*", s):
            v = float(m.group(0))
            if 0.0 <= v <= 1.0:
                return v
            if 1.0 < v <= 100.0:
                return max(0.0, min(1.0, v / 100.0))
        return default
    except Exception:
        return default


def extract_reply_from_model_output(
    content: Union[str, List[Any]],
    default: str = "",
) -> str:
    """Extract the first real text reply from model content (str or list of blocks)."""
    if isinstance(content, str):
        return (content or "").strip() or default
    if not isinstance(content, list):
        return default
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                t = block.get("text") or block.get("content")
                if t and isinstance(t, str) and t.strip():
                    return t.strip()
            continue
        t = getattr(block, "text", None) or getattr(block, "content", None)
        if t and isinstance(t, str) and t.strip():
            return t.strip()
    return default
