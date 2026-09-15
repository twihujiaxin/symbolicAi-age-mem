"""Dynamic time/version extension for ``streaming_multiquery_v1``.

The package is deliberately namespaced: importing it does not alter the v1
schemas, rewards, environment, or historical experiment identities.
"""

from .schema import (
    PROTOCOL_VERSION,
    REWARD_VERSION,
    SCHEMA_VERSION,
    TIME_SEMANTICS_VERSION,
    WORLD_ORACLE_VERSION,
)

__all__ = [
    "PROTOCOL_VERSION",
    "REWARD_VERSION",
    "SCHEMA_VERSION",
    "TIME_SEMANTICS_VERSION",
    "WORLD_ORACLE_VERSION",
]
