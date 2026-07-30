# -*- coding: utf-8 -*-
"""LLM client for summary/similarity calls (DashScope-compatible API)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class chat_client:
    """Client for chat completions (default: DashScope)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model_name: str = "qwen-max",
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return completion.choices[0].message.content or ""
