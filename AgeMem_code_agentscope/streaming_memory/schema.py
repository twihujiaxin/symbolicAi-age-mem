"""Strict public/private schemas for streaming multi-query episodes."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "agemem.stream_mq.episode.v1"


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSentenceRef(StrictFrozenModel):
    source_dataset: Literal["hotpot_qa"] = "hotpot_qa"
    source_revision: str = Field(min_length=1)
    document_key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sentence_index: int = Field(ge=0)
    sentence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicSourceSpan(StrictFrozenModel):
    """A visible pointer; it contains no QA identity or support label."""

    document_key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    sentence_start: int = Field(ge=0)
    sentence_end: int = Field(ge=0)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_range(self) -> "PublicSourceSpan":
        if self.sentence_end < self.sentence_start:
            raise ValueError("sentence_end must be >= sentence_start")
        return self


class PublicChunk(StrictFrozenModel):
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sources: Tuple[PublicSourceSpan, ...] = Field(min_length=1)
    content_token_count: int = Field(ge=1)


class StreamEpisodePublic(StrictFrozenModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    history_family_id: str = Field(min_length=1)
    chunks: Tuple[PublicChunk, ...] = Field(min_length=1)
    context_budget_tokens: int = Field(ge=1)
    memory_budget_tokens: int = Field(ge=1)


class QueryRecord(StrictFrozenModel):
    query_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class QueryGold(StrictFrozenModel):
    query_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    support_refs: Tuple[SourceSentenceRef, ...] = Field(min_length=1)
    source_question_id: str = Field(min_length=1)


class StreamingBuildConfig(StrictFrozenModel):
    schema_version: Literal["agemem.stream_mq.config.v1"]
    protocol: Literal["streaming_multiquery_v1"]
    experiment_family: str = Field(min_length=1)
    data: Dict[str, Any]
    model: Dict[str, Any]
    environment: Dict[str, Any]
    query: Dict[str, Any]
    reward: Dict[str, Any]
    training: Dict[str, Any]
    runtime: Dict[str, Any]

    @model_validator(mode="after")
    def validate_protocol_invariants(self) -> "StreamingBuildConfig":
        env, query, training, reward = (
            self.environment,
            self.query,
            self.training,
            self.reward,
        )
        required_env = {
            "context_total_tokens",
            "persistent_memory_tokens",
            "chunk_target_tokens",
            "chunk_max_tokens",
            "ingest_max_new_tokens",
            "answer_max_new_tokens",
            "decisions_per_chunk",
            "answer_tail_tokens",
        }
        missing = sorted(required_env - set(env))
        if missing:
            raise ValueError(f"environment is missing required fields: {missing}")
        if env.get("one_action_per_response") is not True:
            raise ValueError("one_action_per_response must be true")
        if env.get("external_summary_model") is not False:
            raise ValueError("default protocol forbids an external summary model")
        if query.get("mode") != "independent_snapshot_branches":
            raise ValueError("query mode must use independent snapshot branches")
        if query.get("branch_writeback") is not False:
            raise ValueError("answer branches must not write back")
        if training.get("train_scope") != "ingest_only":
            raise ValueError("v1 trains the ingest policy only")
        if training.get("reader_tokens_in_actor_loss") is not False:
            raise ValueError("reader tokens must be excluded from actor loss")
        if training.get("std_ddof") != 0:
            raise ValueError("streaming_multiquery_v1 freezes std_ddof=0")
        if reward.get("profile") == "terminal" and reward.get("semantic_lambda") != 0:
            raise ValueError("terminal reward requires semantic_lambda=0")
        if int(self.data.get("queries_per_snapshot", 0)) < 2:
            raise ValueError("queries_per_snapshot must be at least 2")
        return self


class ManifestFile(StrictFrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=0)


class StreamingManifest(StrictFrozenModel):
    schema_version: Literal["agemem.stream_mq.manifest.v1"] = (
        "agemem.stream_mq.manifest.v1"
    )
    protocol: Literal["streaming_multiquery_v1"] = "streaming_multiquery_v1"
    build_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    source_fingerprints: Dict[str, str]
    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_rule: str = Field(min_length=1)
    build_seed: int
    split_counts: Dict[str, int]
    query_counts: Dict[str, int]
    files: Tuple[ManifestFile, ...]
    statistics: Dict[str, Any]
    rejected: Dict[str, int]
    git_commit: Optional[str] = None
    dirty_patch_sha256: Optional[str] = None


def public_episode_observation(episode: StreamEpisodePublic, chunk_index: int) -> dict:
    """Whitelist the only episode fields an ingest policy may observe."""

    chunk = episode.chunks[chunk_index]
    return {
        "protocol": "streaming_multiquery_v1",
        "phase": "ingest",
        "chunk": chunk.model_dump(mode="json"),
        "context_budget_tokens": episode.context_budget_tokens,
        "memory_budget_tokens": episode.memory_budget_tokens,
    }


__all__ = [
    "ManifestFile",
    "PublicChunk",
    "PublicSourceSpan",
    "QueryGold",
    "QueryRecord",
    "SCHEMA_VERSION",
    "SourceSentenceRef",
    "StreamEpisodePublic",
    "StreamingBuildConfig",
    "StreamingManifest",
    "public_episode_observation",
]
