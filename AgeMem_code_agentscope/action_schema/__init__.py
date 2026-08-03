"""M6 action-level schemas and non-destructive M5 migration."""

from .migration import (
    MigrationResult,
    SchemaMigrationError,
    load_migration_manifest,
    migrate_m5_canonical_report,
    migrate_m5_step,
)
from .models import (
    ACTION_CREDIT_SCHEMA_VERSION,
    ACTION_EVENT_SCHEMA_VERSION,
    M5_ORACLE_REWARD_VERSION,
    M5_TO_M6_MIGRATION_VERSION,
    MIGRATION_FILE_SCHEMA_VERSION,
    MIGRATION_MANIFEST_SCHEMA_VERSION,
    REWARD_BREAKDOWN_V2_SCHEMA_VERSION,
    TRAJECTORY_STEP_V2_SCHEMA_VERSION,
    ActionCreditRecord,
    ActionEvent,
    ActionSource,
    MigrationFileRecord,
    MigrationManifest,
    RewardBreakdownV2,
    TrajectoryStepV2,
)

__all__ = [
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
    "MigrationResult",
    "RewardBreakdownV2",
    "SchemaMigrationError",
    "TrajectoryStepV2",
    "load_migration_manifest",
    "migrate_m5_canonical_report",
    "migrate_m5_step",
]
