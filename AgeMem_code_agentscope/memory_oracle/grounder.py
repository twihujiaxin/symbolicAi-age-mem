"""Deterministic M3 Oracle-label to M4 atomic-proposition grounding."""

from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

from pydantic import ValidationError

from ..toy_hotpotqa.models import OracleLabels, ToyMemoryTask
from ..trajectory import TrajectoryStep
from .models import APName, AP_ORDER, OracleAPEvent


class OracleGroundingError(ValueError):
    """Raised when trajectory metadata cannot be trusted as M3 Oracle labels."""


def _as_set(values: Iterable[str]) -> Set[str]:
    return set(values)


class MemoryOracleGrounder:
    """Map validated semantic outcomes to APs without inspecting tool names."""

    def __init__(self, task: ToyMemoryTask) -> None:
        self.task = task.model_copy(deep=True)
        self._all = {fact.fact_id for fact in task.facts}
        self._supporting = set(task.supporting_fact_ids)
        self._distractors = set(task.distractor_fact_ids)
        self._stale = set(task.stale_fact_ids)
        self._duplicates = set(task.duplicate_fact_ids)

    @staticmethod
    def _require_subset(name: str, actual: Set[str], expected: Set[str]) -> None:
        unexpected = actual - expected
        if unexpected:
            raise OracleGroundingError(
                f"{name} contains semantically invalid fact IDs: {sorted(unexpected)}"
            )

    def _validate_labels(self, labels: OracleLabels) -> None:
        self._require_subset(
            "observed_fact_ids", _as_set(labels.observed_fact_ids), self._all
        )
        self._require_subset(
            "stored_supporting_fact_ids",
            _as_set(labels.stored_supporting_fact_ids),
            self._supporting,
        )
        self._require_subset(
            "stored_distractor_fact_ids",
            _as_set(labels.stored_distractor_fact_ids),
            self._distractors,
        )
        self._require_subset(
            "ignored_duplicate_fact_ids",
            _as_set(labels.ignored_duplicate_fact_ids),
            self._duplicates,
        )
        self._require_subset(
            "updated_stale_fact_ids",
            _as_set(labels.updated_stale_fact_ids),
            self._stale,
        )
        self._require_subset(
            "retrieved_supporting_fact_ids",
            _as_set(labels.retrieved_supporting_fact_ids),
            self._supporting,
        )
        self._require_subset(
            "retrieved_distractor_fact_ids",
            _as_set(labels.retrieved_distractor_fact_ids),
            self._distractors,
        )
        self._require_subset(
            "retrieved_stale_fact_ids",
            _as_set(labels.retrieved_stale_fact_ids),
            self._stale,
        )
        self._require_subset(
            "deleted_supporting_fact_ids",
            _as_set(labels.deleted_supporting_fact_ids),
            self._supporting,
        )

    def from_step(self, step: TrajectoryStep) -> OracleAPEvent:
        """Ground one M3 trajectory step solely from validated result metadata."""

        if step.task_id != self.task.task_id:
            raise OracleGroundingError(
                f"task mismatch: expected {self.task.task_id!r}, got {step.task_id!r}"
            )
        if len(step.tool_results) != 1:
            raise OracleGroundingError("M3 Oracle replay requires one tool result per step")
        metadata = step.tool_results[0].metadata
        if not isinstance(metadata, dict):
            raise OracleGroundingError("tool result metadata is required")
        required = {"task_id", "rollout_id", "seed", "stage_before", "oracle_labels"}
        missing = required - set(metadata)
        if missing:
            raise OracleGroundingError(
                f"tool result metadata is missing fields: {sorted(missing)}"
            )
        identity = (
            metadata["task_id"],
            metadata["rollout_id"],
            metadata["stage_before"],
        )
        expected = (step.task_id, step.rollout_id, step.stage)
        if identity != expected:
            raise OracleGroundingError(
                f"tool result identity mismatch: expected {expected!r}, got {identity!r}"
            )
        seed = metadata["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise OracleGroundingError("tool result seed must be a non-negative integer")
        try:
            labels = OracleLabels.model_validate(metadata["oracle_labels"])
        except ValidationError as exc:
            raise OracleGroundingError(f"invalid M3 Oracle labels: {exc}") from exc
        self._validate_labels(labels)

        evidence: Dict[APName, Tuple[str, ...]] = {}
        observed_supporting = tuple(
            sorted(set(labels.observed_fact_ids) & self._supporting)
        )
        if observed_supporting:
            evidence["observed_supporting_fact"] = observed_supporting
        if labels.stored_supporting_fact_ids:
            evidence["stored_supporting_fact"] = tuple(
                sorted(labels.stored_supporting_fact_ids)
            )
        if labels.stored_distractor_fact_ids:
            evidence["stored_irrelevant_fact"] = tuple(
                sorted(labels.stored_distractor_fact_ids)
            )
        if labels.updated_stale_fact_ids:
            evidence["updated_stale_fact"] = tuple(
                sorted(labels.updated_stale_fact_ids)
            )
        if labels.deleted_supporting_fact_ids:
            evidence["deleted_supporting_fact"] = tuple(
                sorted(labels.deleted_supporting_fact_ids)
            )
        if labels.retrieved_supporting_fact_ids:
            evidence["retrieved_supporting_fact"] = tuple(
                sorted(labels.retrieved_supporting_fact_ids)
            )
        irrelevant_retrievals = tuple(
            sorted(
                set(labels.retrieved_distractor_fact_ids)
                | set(labels.retrieved_stale_fact_ids)
            )
        )
        if irrelevant_retrievals:
            evidence["retrieved_irrelevant_fact"] = irrelevant_retrievals
        if labels.supporting_coverage_complete:
            evidence["supporting_coverage_complete"] = tuple(
                sorted(self._supporting)
            )
        if labels.answer_correct is True:
            evidence["answered_correctly"] = ()

        propositions = tuple(ap for ap in AP_ORDER if ap in evidence)
        return OracleAPEvent(
            task_id=step.task_id,
            rollout_id=step.rollout_id,
            seed=seed,
            timestep=step.timestep,
            stage=step.stage,
            propositions=propositions,
            evidence_fact_ids=evidence,
        )


__all__ = ["MemoryOracleGrounder", "OracleGroundingError"]
