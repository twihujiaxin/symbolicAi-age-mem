# -*- coding: utf-8 -*-
"""AgeMem: Agent with long/short-term memory (AgentScope)."""
from .agent import AgeMem
from .memory import AgentScopeLongtermMemory
from .prompts import SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT

__all__ = [
    "AgeMem",
    "AgentScopeLongtermMemory",
    "SUMMARY_CONTEXT_SYS_PROMPT",
    "TEXT_SIMILARITY_SYS_PROMPT",
]
