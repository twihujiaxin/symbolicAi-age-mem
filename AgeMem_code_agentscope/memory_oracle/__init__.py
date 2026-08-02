"""M4 Memory Oracle AP, hand-authored DFA, and offline reward replay."""

from .automaton import DFARunner, hand_authored_memory_dfa
from .grounder import MemoryOracleGrounder, OracleGroundingError
from .models import (
    APName,
    AP_ORDER,
    AutomatonSpec,
    AutomatonTransition,
    DFAStepResult,
    OracleAPEvent,
    RewardBreakdown,
    RewardConfig,
    RewardProfile,
    RewardReplayResult,
    RewardedTrajectoryStep,
)
from .replay import OfflineRewardReplay, default_reward_config_path

__all__ = [
    "APName",
    "AP_ORDER",
    "AutomatonSpec",
    "AutomatonTransition",
    "DFARunner",
    "DFAStepResult",
    "MemoryOracleGrounder",
    "OfflineRewardReplay",
    "OracleAPEvent",
    "OracleGroundingError",
    "RewardBreakdown",
    "RewardConfig",
    "RewardProfile",
    "RewardReplayResult",
    "RewardedTrajectoryStep",
    "default_reward_config_path",
    "hand_authored_memory_dfa",
]
