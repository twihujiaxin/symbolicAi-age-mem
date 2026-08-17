"""M6 action schemas with dependency-safe lazy migration exports.

Online training needs the strict action models but must not eagerly import the
M5 benchmark/migration graph (and, transitively, AgentScope). Migration names
remain API-compatible and are resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module


_MODEL_EXPORTS = {
    "ACTION_CREDIT_SCHEMA_VERSION",
    "ACTION_EVENT_SCHEMA_VERSION",
    "M5_ORACLE_REWARD_VERSION",
    "M5_TO_M6_MIGRATION_VERSION",
    "MIGRATION_FILE_SCHEMA_VERSION",
    "MIGRATION_MANIFEST_SCHEMA_VERSION",
    "REWARD_BREAKDOWN_V2_SCHEMA_VERSION",
    "TRAJECTORY_STEP_V2_SCHEMA_VERSION",
    "ActionCreditRecord",
    "ActionEvent",
    "ActionSource",
    "MigrationFileRecord",
    "MigrationManifest",
    "RewardBreakdownV2",
    "TrajectoryStepV2",
}
_MIGRATION_EXPORTS = {
    "MigrationResult",
    "SchemaMigrationError",
    "load_migration_manifest",
    "migrate_m5_canonical_report",
    "migrate_m5_step",
}

__all__ = sorted(_MODEL_EXPORTS | _MIGRATION_EXPORTS)


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        module = import_module(".models", __name__)
    elif name in _MIGRATION_EXPORTS:
        module = import_module(".migration", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
