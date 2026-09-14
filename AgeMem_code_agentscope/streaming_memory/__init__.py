"""Contracts for the isolated ``streaming_multiquery_v1`` protocol.

The package is intentionally independent from the legacy three-stage workflow.
Importing it does not initialize Ray, vLLM, AgentScope, or a model provider.
"""

from .schema import (
    PublicChunk,
    PublicSourceSpan,
    QueryGold,
    QueryRecord,
    SourceSentenceRef,
    StreamEpisodePublic,
    StreamingManifest,
)

PROTOCOL = "streaming_multiquery_v1"

__all__ = [
    "PROTOCOL",
    "PublicChunk",
    "PublicSourceSpan",
    "QueryGold",
    "QueryRecord",
    "SourceSentenceRef",
    "StreamEpisodePublic",
    "StreamingManifest",
]
