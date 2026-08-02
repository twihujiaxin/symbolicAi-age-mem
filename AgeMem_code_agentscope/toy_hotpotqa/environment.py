"""Deterministic three-stage HotpotQA-style memory environment."""

from __future__ import annotations

import hashlib
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from ..memory import AgentScopeLongtermMemory
from ..memory_store import InMemoryStore, RolloutMemoryStoreRegistry
from .models import (
    EpisodeSnapshot,
    EpisodeStepResult,
    MemoryEpisode,
    OracleLabels,
    StageInput,
    ToyAction,
    ToyFact,
    ToyMemoryTask,
    sorted_tuple,
)


def deterministic_embedding(text: str) -> List[float]:
    """Produce a stable local vector without a model or network call."""

    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    return [float(value + 1) for value in digest[:16]]


class _StepClock:
    """A timestamp source controlled by episode timestep for exact replay."""

    def __init__(self) -> None:
        self._step = 0
        self._lock = threading.Lock()

    def set_step(self, step: int) -> None:
        with self._lock:
            self._step = step

    def __call__(self) -> str:
        with self._lock:
            instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
                seconds=self._step
            )
            return instant.isoformat()


class ToyEnvironmentPool:
    """Shared rollout registry with one deterministic clock per store."""

    def __init__(self) -> None:
        self._clocks: Dict[str, _StepClock] = {}
        self._lock = threading.RLock()
        self.registry = RolloutMemoryStoreRegistry(self._create_store)

    def _create_store(self, rollout_id: str) -> InMemoryStore:
        with self._lock:
            clock = _StepClock()
            self._clocks[rollout_id] = clock
            return InMemoryStore(rollout_id, research_mode=True, clock=clock)

    def memory(self, rollout_id: str) -> AgentScopeLongtermMemory:
        store = self.registry.get_or_create(rollout_id)
        return AgentScopeLongtermMemory(
            store=store,
            embedding_function=deterministic_embedding,
        )

    def set_step(self, rollout_id: str, step: int) -> None:
        with self._lock:
            clock = self._clocks.get(rollout_id)
        if clock is None:
            # Do not hold the pool lock while entering the registry: its store
            # factory acquires the pool lock, and reversing that order could
            # deadlock concurrent first access.
            self.registry.get_or_create(rollout_id)
            with self._lock:
                clock = self._clocks[rollout_id]
        if clock is not None:
            clock.set_step(step)


class HotpotQAToyEnvironment:
    """Three-stage episode reusing M2 memory and M1 snapshot contracts."""

    _PUBLIC_ACTIONS = {
        1: ("add", "update", "advance"),
        2: ("advance",),
        3: ("retrieve", "answer"),
    }

    def __init__(
        self,
        task: ToyMemoryTask,
        *,
        rollout_id: str,
        seed: int,
        pool: Optional[ToyEnvironmentPool] = None,
    ) -> None:
        if not rollout_id:
            raise ValueError("rollout_id must be non-empty")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.task = task.model_copy(deep=True)
        self.rollout_id = rollout_id
        self.seed = seed
        self.pool = pool or ToyEnvironmentPool()
        self.memory = self.pool.memory(rollout_id)
        self.stage = 1
        self.timestep = 0
        self.stm: List[str] = []
        self._pending_observed_fact_ids: Set[str] = set()
        self.retrieved_supporting_fact_ids: Set[str] = set()
        self._memory_ids_by_fact_id: Dict[str, str] = {}
        self.done = False
        self.success = False
        self.reset()

    def _ordered_facts(self, stage: int) -> List[ToyFact]:
        facts = [fact for fact in self.task.facts if fact.stage == stage]
        seed_bytes = f"{self.task.task_id}:{self.seed}:{stage}".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
        random.Random(local_seed).shuffle(facts)
        return facts

    def _stage1_observation(self) -> str:
        lines = [
            "Stage 1 — Memory construction.",
            "Here are several factual notes grouped by title:",
        ]
        for fact in self._ordered_facts(1):
            lines.append(f"- {fact.title}: {fact.sentence}")
        return "\n".join(lines)

    def _stage2_observation(self) -> str:
        facts = self._ordered_facts(2)
        if not facts:
            return (
                "Stage 2 — Context interference. "
                "No additional factual notes are provided in this clean episode."
            )
        lines = ["Stage 2 — Context interference. Unrelated messages follow:"]
        lines.extend(f"- {fact.sentence}" for fact in facts)
        return "\n".join(lines)

    def _stage3_observation(self) -> str:
        return f"Stage 3 — Retrieval QA. Question: {self.task.question}"

    def stage_input(self) -> StageInput:
        return StageInput(
            task_id=self.task.task_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            stage=self.stage,
            observation="\n".join(self.stm),
            allowed_actions=self._PUBLIC_ACTIONS[self.stage],
        )

    def episode(self) -> MemoryEpisode:
        return MemoryEpisode(
            task_id=self.task.task_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            split=self.task.split,
            stage=self.stage,
            timestep=self.timestep,
            stm=tuple(self.stm),
            retrieved_supporting_fact_ids=sorted_tuple(
                self.retrieved_supporting_fact_ids
            ),
            done=self.done,
            success=self.success,
        )

    def reset(self) -> StageInput:
        self.pool.registry.reset(self.rollout_id)
        self.pool.set_step(self.rollout_id, 0)
        self.stage = 1
        self.timestep = 0
        self.stm = [self._stage1_observation()]
        self._pending_observed_fact_ids = {
            fact.fact_id for fact in self._ordered_facts(1)
        }
        self.retrieved_supporting_fact_ids = set()
        self._memory_ids_by_fact_id = {}
        self.done = False
        self.success = False
        return self.stage_input()

    def snapshot(self) -> EpisodeSnapshot:
        return EpisodeSnapshot(
            task_id=self.task.task_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            stage=self.stage,
            timestep=self.timestep,
            stm=tuple(self.stm),
            pending_observed_fact_ids=sorted_tuple(
                self._pending_observed_fact_ids
            ),
            retrieved_supporting_fact_ids=sorted_tuple(
                self.retrieved_supporting_fact_ids
            ),
            memory_ids_by_fact_id=tuple(sorted(self._memory_ids_by_fact_id.items())),
            done=self.done,
            success=self.success,
            memory=self.memory.snapshot(),
        )

    def restore(self, snapshot: EpisodeSnapshot) -> StageInput:
        expected = (self.task.task_id, self.rollout_id, self.seed)
        actual = (snapshot.task_id, snapshot.rollout_id, snapshot.seed)
        if actual != expected:
            raise ValueError(
                "episode snapshot identity mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
        if snapshot.stage not in (1, 2, 3) or snapshot.timestep < 0:
            raise ValueError("episode snapshot contains invalid stage or timestep")
        self.memory.restore(snapshot.memory)
        self.stage = snapshot.stage
        self.timestep = snapshot.timestep
        self.stm = list(snapshot.stm)
        self._pending_observed_fact_ids = set(snapshot.pending_observed_fact_ids)
        self.retrieved_supporting_fact_ids = set(
            snapshot.retrieved_supporting_fact_ids
        )
        self._memory_ids_by_fact_id = dict(snapshot.memory_ids_by_fact_id)
        self.done = snapshot.done
        self.success = snapshot.success
        self.pool.set_step(self.rollout_id, self.timestep)
        return self.stage_input()

    def _fact_role(self, fact_id: str) -> str:
        if fact_id in self.task.supporting_fact_ids:
            return "supporting"
        if fact_id in self.task.distractor_fact_ids:
            return "distractor"
        if fact_id in self.task.stale_fact_ids:
            return "stale"
        if fact_id in self.task.duplicate_fact_ids:
            return "duplicate"
        return "other"

    def _memory_metadata(self, fact: ToyFact) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "task_id": self.task.task_id,
            "fact_id": fact.fact_id,
            "role": self._fact_role(fact.fact_id),
            "stale": fact.fact_id in self.task.stale_fact_ids,
            "duplicate_of_fact_id": fact.duplicate_of_fact_id,
            "stage": str(self.stage),
            "title": fact.title,
        }
        # M3 synthetic facts carry explicit subject/relation/object fields. Real
        # M5 HotpotQA sentences deliberately do not invent M6 semantic triples;
        # their exact (title, sent_id) source pointer is retained instead.
        for field_name in ("subject", "relation", "object", "sent_id"):
            value = getattr(fact, field_name, None)
            if value is not None:
                metadata[field_name] = value
        return metadata

    def _memory_id(self, fact_id: str) -> str:
        return f"{self.rollout_id}:memory:{fact_id}"

    async def _add(self, action: ToyAction) -> tuple[bool, str, OracleLabels]:
        if self.stage != 1:
            return False, "ADD is only allowed during Stage 1.", OracleLabels()
        fact = self.task.fact(action.fact_id or "")
        if fact.stage != 1:
            return False, "The requested fact has not been observed.", OracleLabels()
        memory_id = self._memory_id(fact.fact_id)
        try:
            await self.memory.add(
                memory_id,
                fact.sentence,
                self._memory_metadata(fact),
                source_step=self.timestep,
            )
        except ValueError as exc:
            return False, f"Memory was not added: {exc}", OracleLabels()
        self._memory_ids_by_fact_id[fact.fact_id] = memory_id
        role = self._fact_role(fact.fact_id)
        labels = OracleLabels(
            stored_supporting_fact_ids=(fact.fact_id,) if role == "supporting" else (),
            stored_distractor_fact_ids=(fact.fact_id,) if role == "distractor" else (),
        )
        return True, f"Stored factual note: {fact.sentence}", labels

    async def _update(self, action: ToyAction) -> tuple[bool, str, OracleLabels]:
        if self.stage != 1:
            return False, "UPDATE is only allowed during Stage 1.", OracleLabels()
        replacement = self.task.fact(action.fact_id or "")
        target_id = action.target_fact_id or ""
        if replacement.replaces_fact_id != target_id:
            return False, "The fact is not a valid correction for the target.", OracleLabels()
        memory_id = self._memory_ids_by_fact_id.get(target_id)
        if memory_id is None:
            return False, "The stale memory was not found.", OracleLabels()
        ok = await self.memory.update(
            memory_id,
            replacement.sentence,
            self._memory_metadata(replacement),
            source_step=self.timestep,
        )
        if not ok:
            return False, "The stale memory could not be updated.", OracleLabels()
        self._memory_ids_by_fact_id[replacement.fact_id] = memory_id
        return (
            True,
            f"Updated stale note to: {replacement.sentence}",
            OracleLabels(updated_stale_fact_ids=(target_id,)),
        )

    async def _retrieve(self, action: ToyAction) -> tuple[bool, str, OracleLabels]:
        if self.stage != 3:
            return False, "RETRIEVE is only allowed during Stage 3.", OracleLabels()
        fact = self.task.fact(action.fact_id or "")
        items = await self.memory.retrieve(
            fact.sentence,
            1,
            {"fact_id": fact.fact_id},
        )
        if not items:
            return False, "No active memory matched the requested fact.", OracleLabels()
        actual_fact_ids = {
            str(item.metadata.get("fact_id", "")) for item in items
        }
        supporting = actual_fact_ids & set(self.task.supporting_fact_ids)
        distractors = actual_fact_ids & set(self.task.distractor_fact_ids)
        stale = actual_fact_ids & set(self.task.stale_fact_ids)
        self.retrieved_supporting_fact_ids.update(supporting)
        coverage = set(self.task.supporting_fact_ids).issubset(
            self.retrieved_supporting_fact_ids
        )
        labels = OracleLabels(
            retrieved_supporting_fact_ids=sorted_tuple(supporting),
            retrieved_distractor_fact_ids=sorted_tuple(distractors),
            retrieved_stale_fact_ids=sorted_tuple(stale),
            supporting_coverage_complete=coverage,
        )
        rendered = "\n".join(
            f"- {item.content} (Memory ID: {item.memory_id})" for item in items
        )
        return True, f"Retrieved memories:\n{rendered}", labels

    async def _delete(self, action: ToyAction) -> tuple[bool, str, OracleLabels]:
        # DELETE is intentionally not advertised to the normal M3 policy. It is
        # available only to deterministic error policies for the misdelete case.
        if self.stage != 2:
            return False, "Fault-injection DELETE is only valid in Stage 2.", OracleLabels()
        fact_id = action.fact_id or ""
        memory_id = self._memory_ids_by_fact_id.get(fact_id)
        if memory_id is None:
            return False, "The requested memory was not found.", OracleLabels()
        ok = await self.memory.delete(memory_id, source_step=self.timestep)
        if not ok:
            return False, "The requested memory was not deleted.", OracleLabels()
        labels = OracleLabels(
            deleted_supporting_fact_ids=(fact_id,)
            if fact_id in self.task.supporting_fact_ids
            else (),
        )
        return True, f"Soft-deleted memory for fact {fact_id}.", labels

    async def _advance(self) -> tuple[bool, str, OracleLabels]:
        if self.stage == 1:
            ignored_duplicates = {
                fact_id
                for fact_id in self.task.duplicate_fact_ids
                if fact_id not in self._memory_ids_by_fact_id
            }
            self.stage = 2
            # Match the existing AgeMem workflow: Stage 1 STM is cleared while
            # LTM is retained, then Stage 2 distractors are inserted.
            stage2_observation = self._stage2_observation()
            self.stm = [stage2_observation]
            return (
                True,
                stage2_observation,
                OracleLabels(
                    observed_fact_ids=tuple(
                        fact.fact_id for fact in self._ordered_facts(2)
                    ),
                    ignored_duplicate_fact_ids=sorted_tuple(ignored_duplicates),
                ),
            )
        if self.stage == 2:
            self.stage = 3
            # Stage 3 appends the question to the noisy Stage 2 context, matching
            # the current training workflow rather than clearing STM again.
            stage3_observation = self._stage3_observation()
            self.stm.append(stage3_observation)
            return True, stage3_observation, OracleLabels()
        return False, "The episode is already in Stage 3.", OracleLabels()

    async def _answer(self, action: ToyAction) -> tuple[bool, str, OracleLabels, float]:
        if self.stage != 3:
            return False, "ANSWER is only allowed during Stage 3.", OracleLabels(), 0.0
        answer_correct = _normalize_answer(action.answer or "") == _normalize_answer(
            self.task.answer
        )
        active_fact_ids = {
            str(item.metadata.get("fact_id", ""))
            for item in await self.memory.get_memory()
        }
        supporting_active = set(self.task.supporting_fact_ids).issubset(active_fact_ids)
        coverage = set(self.task.supporting_fact_ids).issubset(
            self.retrieved_supporting_fact_ids
        )
        self.success = answer_correct and supporting_active and coverage
        self.done = True
        labels = OracleLabels(
            supporting_coverage_complete=coverage,
            answer_correct=answer_correct,
        )
        message = (
            "Episode succeeded."
            if self.success
            else "Episode failed the answer, active-memory, or retrieval requirement."
        )
        return True, message, labels, 1.0 if self.success else 0.0

    async def step(self, action: ToyAction) -> EpisodeStepResult:
        if self.done:
            raise RuntimeError("cannot act after the episode is done")
        stage_before = self.stage
        step_index = self.timestep
        self.pool.set_step(self.rollout_id, step_index)
        task_reward = 0.0

        if action.kind == "add":
            success, message, labels = await self._add(action)
        elif action.kind == "update":
            success, message, labels = await self._update(action)
        elif action.kind == "retrieve":
            success, message, labels = await self._retrieve(action)
        elif action.kind == "delete":
            success, message, labels = await self._delete(action)
        elif action.kind == "advance":
            success, message, labels = await self._advance()
        elif action.kind == "answer":
            success, message, labels, task_reward = await self._answer(action)
        else:  # pragma: no cover - ToyAction schema makes this unreachable.
            raise ValueError(f"unsupported action kind {action.kind!r}")

        if action.kind not in {"advance"}:
            self.stm.append(message)
        if self._pending_observed_fact_ids:
            label_data = labels.model_dump(mode="python")
            observed = set(label_data["observed_fact_ids"])
            observed.update(self._pending_observed_fact_ids)
            label_data["observed_fact_ids"] = sorted_tuple(observed)
            labels = OracleLabels.model_validate(label_data)
            self._pending_observed_fact_ids.clear()
        self.timestep += 1
        return EpisodeStepResult(
            task_id=self.task.task_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            timestep=step_index,
            stage_before=stage_before,
            stage_after=self.stage,
            action=action,
            success=success,
            message=message,
            labels=labels,
            task_reward=task_reward,
            done=self.done,
            episode_success=self.success,
        )


def _normalize_answer(answer: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in answer).split()
    )


__all__ = [
    "HotpotQAToyEnvironment",
    "ToyEnvironmentPool",
    "deterministic_embedding",
]
