# -*- coding: utf-8 -*-
"""AgeMem public API with dependency-safe lazy exports.

Importing the backend-independent M2 store must not eagerly import AgentScope,
``shortuuid`` or model clients. Trinity's M8 workflow can therefore reuse the
store contract in a minimal training environment, while the standalone public
names remain backwards compatible through PEP 562 lazy attributes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AgeMem": (".agent", "AgeMem"),
    "AgentScopeLongtermMemory": (".memory", "AgentScopeLongtermMemory"),
    "AutomatonSpec": (".memory_oracle", "AutomatonSpec"),
    "DFARunner": (".memory_oracle", "DFARunner"),
    "ErrorMemoryPolicy": (".toy_hotpotqa", "ErrorMemoryPolicy"),
    "GoldMemoryPolicy": (".toy_hotpotqa", "GoldMemoryPolicy"),
    "HotpotQAToyEnvironment": (".toy_hotpotqa", "HotpotQAToyEnvironment"),
    "HotpotQADataAdapter": (".hotpotqa_benchmark", "HotpotQADataAdapter"),
    "HotpotQAOracleBenchmark": (
        ".hotpotqa_benchmark",
        "HotpotQAOracleBenchmark",
    ),
    "HotpotQASmokeConfig": (".hotpotqa_benchmark", "HotpotQASmokeConfig"),
    "HotpotQASmokeManifest": (
        ".hotpotqa_benchmark",
        "HotpotQASmokeManifest",
    ),
    "InMemoryStore": (".memory_store", "InMemoryStore"),
    "MemoryRecord": (".memory_store", "MemoryRecord"),
    "MemoryOracleGrounder": (".memory_oracle", "MemoryOracleGrounder"),
    "MemoryStore": (".memory_store", "MemoryStore"),
    "MemoryStoreSnapshot": (".memory_store", "MemoryStoreSnapshot"),
    "OfflineRewardReplay": (".memory_oracle", "OfflineRewardReplay"),
    "OracleAPEvent": (".memory_oracle", "OracleAPEvent"),
    "OracleBenchmarkReport": (
        ".hotpotqa_benchmark",
        "OracleBenchmarkReport",
    ),
    "ReplayResult": (".trajectory", "ReplayResult"),
    "RewardBreakdown": (".memory_oracle", "RewardBreakdown"),
    "RewardConfig": (".memory_oracle", "RewardConfig"),
    "RewardReplayResult": (".memory_oracle", "RewardReplayResult"),
    "RolloutMemoryStoreRegistry": (
        ".memory_store",
        "RolloutMemoryStoreRegistry",
    ),
    "SUMMARY_CONTEXT_SYS_PROMPT": (
        ".prompts",
        "SUMMARY_CONTEXT_SYS_PROMPT",
    ),
    "TEXT_SIMILARITY_SYS_PROMPT": (
        ".prompts",
        "TEXT_SIMILARITY_SYS_PROMPT",
    ),
    "TrajectoryRecorder": (".trajectory", "TrajectoryRecorder"),
    "TrajectoryReplay": (".trajectory", "TrajectoryReplay"),
    "TrajectoryStep": (".trajectory", "TrajectoryStep"),
    "TrajectoryValidationError": (
        ".trajectory",
        "TrajectoryValidationError",
    ),
    "ToyEnvironmentPool": (".toy_hotpotqa", "ToyEnvironmentPool"),
    "ToyEpisodeRunner": (".toy_hotpotqa", "ToyEpisodeRunner"),
    "ToyTaskDataset": (".toy_hotpotqa", "ToyTaskDataset"),
    "hand_authored_memory_dfa": (
        ".memory_oracle",
        "hand_authored_memory_dfa",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Resolve one public name on first use and cache it in the module."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
