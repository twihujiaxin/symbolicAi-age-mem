"""Compiled finite monitor backend over frozen semantic frames."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import MONITOR_VERSION
from .state_evaluator import CheckpointSemanticFrame, DirectEvaluation


class MonitorCompileError(ValueError):
    pass


DEFAULT_LOGIC_SPEC = {
    "spec_version": "agemem.dynamic.logic_spec.v1",
    "states": ["absent", "partial", "supported", "conflicting"],
    "output": "coverage*(1-conflict)",
    "checkpoint_schedule": "every_chunk_commit",
    "rollback": True,
}


@dataclass
class CompiledMonitor:
    logic_spec_hash: str
    state_by_query: dict[str, str]
    version: str = MONITOR_VERSION

    def transition(self, coverage: float, conflict: bool) -> tuple[str, float]:
        if not 0.0 <= coverage <= 1.0:
            raise MonitorCompileError("coverage outside [0,1]")
        if conflict:
            return "conflicting", 0.0
        if coverage == 1.0:
            return "supported", 1.0
        if coverage > 0.0:
            return "partial", coverage
        return "absent", 0.0

    def evaluate(self, frames: Sequence[CheckpointSemanticFrame]) -> DirectEvaluation:
        states = []
        utilities = []
        latest: dict[str, float] = {}
        for frame in frames:
            for observation in frame.by_query:
                state, utility = self.transition(
                    observation.coverage, observation.conflict
                )
                self.state_by_query[observation.query_id] = state
                states.append(
                    (
                        frame.checkpoint_id,
                        frame.checkpoint_index,
                        observation.query_id,
                        state,
                        observation.coverage,
                        observation.conflict,
                    )
                )
                utilities.append(
                    (observation.query_id, frame.checkpoint_index, utility)
                )
                latest[observation.query_id] = utility
        return DirectEvaluation(
            backend=self.version,
            checkpoint_states=tuple(states),
            checkpoint_utilities=tuple(utilities),
            final_utility_by_query=tuple(sorted(latest.items())),
        )


def compile_monitor(
    logic_spec: Mapping[str, Any] | None = None,
) -> CompiledMonitor:
    spec = dict(logic_spec or DEFAULT_LOGIC_SPEC)
    if spec.get("output") != "coverage*(1-conflict)":
        raise MonitorCompileError("unsupported output expression")
    if spec.get("checkpoint_schedule") != "every_chunk_commit":
        raise MonitorCompileError("unsupported checkpoint schedule")
    if spec.get("rollback") is not True:
        raise MonitorCompileError("dynamic monitor must permit rollback")
    required = {"absent", "partial", "supported", "conflicting"}
    if set(spec.get("states") or ()) != required:
        raise MonitorCompileError("logic spec has invalid states")
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return CompiledMonitor(hashlib.sha256(raw).hexdigest(), {})


def compare_evaluations(
    direct: DirectEvaluation, compiled: DirectEvaluation
) -> dict[str, Any]:
    state_equal = direct.checkpoint_states == compiled.checkpoint_states
    utility_equal = direct.checkpoint_utilities == compiled.checkpoint_utilities
    final_equal = direct.final_utility_by_query == compiled.final_utility_by_query
    return {
        "status": "pass" if state_equal and utility_equal and final_equal else "fail",
        "state_equal": state_equal,
        "checkpoint_utility_equal": utility_equal,
        "final_equal": final_equal,
        "compared_state_rows": len(direct.checkpoint_states),
        "max_abs_utility_diff": max(
            (
                abs(a[2] - b[2])
                for a, b in zip(
                    direct.checkpoint_utilities, compiled.checkpoint_utilities
                )
            ),
            default=0.0,
        ),
        "duplicate_gpu_experiment_required": False
        if state_equal and utility_equal and final_equal
        else None,
    }


__all__ = [
    "CompiledMonitor",
    "DEFAULT_LOGIC_SPEC",
    "MonitorCompileError",
    "compare_evaluations",
    "compile_monitor",
]
