"""Strict public/private schemas for dynamic streaming memory v2."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROTOCOL_VERSION = "streaming_dynamic_multiquery_v2"
SCHEMA_VERSION = "dynamic_stream_v2"
TIME_SEMANTICS_VERSION = "monotonic_single_valued_v1"
WORLD_ORACLE_VERSION = "agemem.dynamic.world_oracle.v1"
REWARD_VERSION = "dynamic_reward_v2"
MONITOR_VERSION = "agemem.dynamic.compiled_monitor.v1"
VALID_PROFILES = frozenset({"V2_T", "V2_END", "V2_LIFE", "V2_LIFE_MONITOR"})


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DynamicPublicChunk(StrictFrozenModel):
    chunk_id: str = Field(min_length=1)
    observed_at: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_refs: Tuple[str, ...] = Field(min_length=1)
    content_token_count: int = Field(ge=1)


class DynamicHistoryPublic(StrictFrozenModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    history_id: str = Field(min_length=1)
    history_family_id: str = Field(min_length=1)
    chunks: Tuple[DynamicPublicChunk, ...] = Field(min_length=1)
    context_budget_tokens: int = Field(ge=1)
    memory_budget_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def monotonic_chunks(self) -> "DynamicHistoryPublic":
        observed = [item.observed_at for item in self.chunks]
        if observed != list(range(len(observed))):
            raise ValueError("chunk observed_at must be contiguous from zero")
        refs = [ref for chunk in self.chunks for ref in chunk.source_refs]
        if len(refs) != len(set(refs)):
            raise ValueError("a public source_ref may occur in only one chunk")
        return self


class DynamicQueryPublic(StrictFrozenModel):
    query_id: str = Field(min_length=1)
    history_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class PrivateEvent(StrictFrozenModel):
    event_id: str = Field(min_length=1)
    history_id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    value: str = Field(min_length=1)
    operation: Literal["set"] = "set"
    observed_at: int = Field(ge=0)
    effective_at: int = Field(ge=0)
    source_ref: str = Field(min_length=1)
    source_text_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DynamicQueryPrivate(StrictFrozenModel):
    query_id: str = Field(min_length=1)
    history_id: str = Field(min_length=1)
    task_family: Literal[
        "current_state",
        "historical_state",
        "multi_update",
        "temporal_join",
        "retention_under_budget",
    ]
    query_kind: Literal["state_at", "current_state", "start_time", "temporal_join"]
    entity: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    query_time: Optional[int] = Field(default=None, ge=0)
    join_relation: Optional[str] = None
    target_value: Optional[str] = None
    answers: Tuple[str, ...] = Field(min_length=1)
    answer_available_at: int = Field(ge=0)
    support_alternatives: Tuple[Tuple[str, ...], ...] = Field(min_length=1)
    answer_type: Literal["entity", "time", "state", "unknown"] = "entity"
    answerable: bool = True

    @model_validator(mode="after")
    def valid_query(self) -> "DynamicQueryPrivate":
        if self.query_kind in {"state_at", "temporal_join"} and self.query_time is None:
            raise ValueError("temporal query requires query_time")
        if self.query_kind == "temporal_join" and not self.join_relation:
            raise ValueError("temporal_join requires join_relation")
        if self.query_kind == "start_time" and not self.target_value:
            raise ValueError("start_time requires target_value")
        if not self.answerable and self.answers != ("unknown",):
            raise ValueError(
                "unanswerable queries must use the explicit unknown answer"
            )
        return self


class SourceRegistryRecord(StrictFrozenModel):
    source_ref: str = Field(min_length=1)
    history_id: str = Field(min_length=1)
    observed_at: int = Field(ge=0)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SplitAssignment(StrictFrozenModel):
    history_id: str = Field(min_length=1)
    history_family_id: str = Field(min_length=1)
    split: Literal["train", "dev", "test", "diagnostic"]
    data_layer: Literal["D0", "D1"]


class DynamicBuildConfig(StrictFrozenModel):
    schema_version: Literal["agemem.dynamic.config.v2"]
    protocol_version: Literal[PROTOCOL_VERSION]
    experiment: Dict[str, Any]
    inherit: Dict[str, Any]
    model: Dict[str, Any]
    budget: Dict[str, Any]
    data: Dict[str, Any]
    memory: Dict[str, Any]
    grounding: Dict[str, Any]
    reward: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    runtime: Dict[str, Any]

    @model_validator(mode="after")
    def protocol_invariants(self) -> "DynamicBuildConfig":
        required_budget = {
            "context_total_tokens",
            "persistent_memory_tokens",
            "ingest_max_new_tokens",
            "answer_max_new_tokens",
            "answer_tail_tokens",
            "retrieved_payload_tokens",
            "max_decisions_per_chunk",
            "chunk_target_tokens",
            "chunk_max_tokens",
        }
        missing = sorted(required_budget - set(self.budget))
        if missing:
            raise ValueError(f"budget is missing required fields: {missing}")
        if self.data.get("split_unit") != "history_family_id":
            raise ValueError("dynamic v2 must split by history_family_id")
        if int(self.data.get("questions_per_snapshot", 0)) < 2:
            raise ValueError("questions_per_snapshot must be at least two")
        if int(self.data.get("memory_rollouts_per_group", 0)) < 2:
            raise ValueError("memory_rollouts_per_group must be at least two")
        if self.memory.get("allow_raw_source_retrieval") is not False:
            raise ValueError("raw source retrieval is forbidden")
        if self.memory.get("allow_audit_history_read") is not False:
            raise ValueError("audit history is private")
        if self.training.get("shared_prefix_weighting") != "once_per_read_rollout":
            raise ValueError("shared read prefix must be trained exactly once")
        if self.training.get("answer_tokens_in_actor_loss") is not False:
            raise ValueError("fixed reader tokens cannot enter actor loss")
        if self.model.get("train_reader") is not False:
            raise ValueError("v2 first-round protocol freezes the reader")
        profile = self.reward.get("profile")
        if profile not in VALID_PROFILES:
            raise ValueError(f"unsupported reward profile: {profile}")
        if self.reward.get("checkpoint_schedule") != "every_chunk_commit":
            raise ValueError("LIFE denominator requires external chunk checkpoints")
        return self


class DynamicManifestFile(StrictFrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=0)
    visibility: Literal["public", "private", "audit"]


class DynamicManifest(StrictFrozenModel):
    schema_version: Literal["agemem.dynamic.manifest.v2"] = "agemem.dynamic.manifest.v2"
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    time_semantics_version: Literal[TIME_SEMANTICS_VERSION] = TIME_SEMANTICS_VERSION
    world_oracle_version: Literal[WORLD_ORACLE_VERSION] = WORLD_ORACLE_VERSION
    build_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_seed: int
    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_rule: str = Field(min_length=1)
    split_counts: Dict[str, int]
    files: Tuple[DynamicManifestFile, ...]
    statistics: Dict[str, Any]
    validation: Dict[str, Any]
    git_commit: Optional[str] = None
    dirty_patch_sha256: Optional[str] = None


def project_public_observation(
    history: DynamicHistoryPublic, chunk_index: int, policy_memory: Tuple[dict, ...]
) -> dict[str, Any]:
    """Whitelist policy-visible fields; private event/query models are unrepresentable."""

    chunk = history.chunks[chunk_index]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "ingest",
        "chunk": chunk.model_dump(mode="json"),
        "active_memory": policy_memory,
        "context_budget_tokens": history.context_budget_tokens,
        "memory_budget_tokens": history.memory_budget_tokens,
    }


__all__ = [
    name
    for name in globals()
    if name.startswith("Dynamic")
    or name
    in {
        "MONITOR_VERSION",
        "PROTOCOL_VERSION",
        "REWARD_VERSION",
        "SCHEMA_VERSION",
        "TIME_SEMANTICS_VERSION",
        "WORLD_ORACLE_VERSION",
        "PrivateEvent",
        "SourceRegistryRecord",
        "SplitAssignment",
        "VALID_PROFILES",
        "project_public_observation",
    }
]
