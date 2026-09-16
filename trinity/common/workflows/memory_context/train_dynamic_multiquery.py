"""Trinity workflow boundary for dynamic streaming multi-query V2.

The task row is a public history only. Private events, answers and source
registry records are loaded from the frozen manifest inside the reward side of
the workflow and are never appended to policy or reader observations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from AgeMem_code_agentscope.streaming_memory.dynamic.config import load_dynamic_config
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import (
    DynamicHistoryPublic,
    DynamicManifest,
    DynamicQueryPrivate,
    DynamicQueryPublic,
    PrivateEvent,
    SourceRegistryRecord,
    SplitAssignment,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.world_generator import (
    validate_dynamic_manifest,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    TokenAccounting,
    load_tokenizer,
)
from trinity.common.dynamic_multiquery_contract import DynamicMemoryRolloutGroupBundle
from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, MultiTurnWorkflow, Task

from .dynamic_runtime_producer import (
    DynamicRuntimeError,
    DynamicRuntimeProducer,
    FrozenReaderOutput,
)


WORKFLOW_NAME = "AgeMem_dynamic_multiquery_v2_training"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class _RuntimeIndex:
    manifest: DynamicManifest
    histories: Mapping[str, DynamicHistoryPublic]
    public_queries: Mapping[str, tuple[DynamicQueryPublic, ...]]
    private_queries: Mapping[str, Mapping[str, DynamicQueryPrivate]]
    events: Mapping[str, tuple[PrivateEvent, ...]]
    sources: Mapping[str, tuple[SourceRegistryRecord, ...]]
    splits: Mapping[str, SplitAssignment]


def _group_by_history(items: Sequence[Any]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for item in items:
        grouped.setdefault(item.history_id, []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


@lru_cache(maxsize=4)
def _load_runtime_index(manifest_text: str) -> _RuntimeIndex:
    manifest_path = Path(manifest_text).expanduser().resolve()
    validate_dynamic_manifest(manifest_path)
    root = manifest_path.parent
    manifest = DynamicManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    histories = tuple(
        DynamicHistoryPublic.model_validate(item)
        for item in _read_jsonl(root / "histories.public.jsonl")
    )
    public_queries = tuple(
        DynamicQueryPublic.model_validate(item)
        for item in _read_jsonl(root / "queries.public.jsonl")
    )
    private_queries = tuple(
        DynamicQueryPrivate.model_validate(item)
        for item in _read_jsonl(root / "queries.private.jsonl")
    )
    events = tuple(
        PrivateEvent.model_validate(item)
        for item in _read_jsonl(root / "events.private.jsonl")
    )
    sources = tuple(
        SourceRegistryRecord.model_validate(item)
        for item in _read_jsonl(root / "source_registry.private.jsonl")
    )
    assignments = tuple(
        SplitAssignment.model_validate(item)
        for item in _read_jsonl(root / "splits.private.jsonl")
    )
    private_by_history: dict[str, dict[str, DynamicQueryPrivate]] = {}
    for item in private_queries:
        private_by_history.setdefault(item.history_id, {})[item.query_id] = item
    return _RuntimeIndex(
        manifest=manifest,
        histories={item.history_id: item for item in histories},
        public_queries=_group_by_history(public_queries),
        private_queries=private_by_history,
        events=_group_by_history(events),
        sources=_group_by_history(sources),
        splits={item.history_id: item for item in assignments},
    )


def _select_queries(
    queries: Sequence[DynamicQueryPublic],
    *,
    count: int,
    seed: int,
    history_id: str,
) -> tuple[DynamicQueryPublic, ...]:
    if len(queries) < count:
        raise DynamicRuntimeError("history has fewer than frozen m public queries")
    ranked = sorted(
        queries,
        key=lambda item: hashlib.sha256(
            f"{seed}:{history_id}:{item.query_id}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ranked[:count])


def _same_model_identity(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    actual_path = Path(actual).expanduser()
    expected_path = Path(expected).expanduser()
    return (
        actual_path.exists()
        and expected_path.exists()
        and actual_path.resolve() == expected_path.resolve()
    )


def publish_complete_dynamic_group(
    bundle: DynamicMemoryRolloutGroupBundle,
) -> tuple[dict, ...]:
    """Compatibility boundary retained for CPU/offline callers."""

    bundle.validate_complete()
    return bundle.actor_batch()


@WORKFLOWS.register_module(WORKFLOW_NAME)
class AgeMemDynamicMultiQueryV2Training(MultiTurnWorkflow):
    """Model-backed K-memory/m-query workflow with a frozen reader."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[list] = None,
    ) -> None:
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.workflow_args = dict(task.workflow_args or {})
        config_value = self.workflow_args.get("dynamic_config_path")
        manifest_value = self.workflow_args.get("dynamic_manifest_path")
        if not isinstance(config_value, str) or not config_value:
            raise DynamicRuntimeError("dynamic_config_path is required")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise DynamicRuntimeError("dynamic_manifest_path is required")
        self.dynamic_config_path = Path(config_value).expanduser().resolve()
        self.manifest_path = Path(manifest_value).expanduser().resolve()
        self.dynamic_config = load_dynamic_config(self.dynamic_config_path)
        self.runtime_index = _load_runtime_index(str(self.manifest_path))
        configured_manifest = self.dynamic_config.data.get("manifest")
        if configured_manifest:
            frozen = Path(str(configured_manifest)).expanduser().resolve()
            if frozen != self.manifest_path:
                raise DynamicRuntimeError("workflow manifest differs from frozen config")
        raw_task = task.raw_task
        if not isinstance(raw_task, dict):
            raise DynamicRuntimeError("dynamic task row must be a public history object")
        self.history = DynamicHistoryPublic.model_validate(raw_task)
        indexed = self.runtime_index.histories.get(self.history.history_id)
        if indexed is None or indexed != self.history:
            raise DynamicRuntimeError("task history is absent or differs from manifest")
        expected_split = self.workflow_args.get("expected_split")
        assignment = self.runtime_index.splits.get(self.history.history_id)
        if assignment is None:
            raise DynamicRuntimeError("history has no frozen split assignment")
        if expected_split and assignment.split != expected_split:
            raise DynamicRuntimeError("task history is outside expected split")
        if not auxiliary_models or len(auxiliary_models) != 1:
            raise DynamicRuntimeError("exactly one frozen reader model is required")
        self.reader_client = auxiliary_models[0]
        reader_path = self.dynamic_config.model.get("reader_path")
        reader_revision = self.dynamic_config.model.get("reader_revision")
        if not reader_path or not reader_revision:
            raise DynamicRuntimeError("frozen reader path/revision are required")
        actual_reader_path = getattr(self.reader_client, "model_path", None)
        if not isinstance(actual_reader_path, str) or not actual_reader_path:
            raise DynamicRuntimeError("reader server did not expose its model identity")
        if not _same_model_identity(actual_reader_path, str(reader_path)):
            raise DynamicRuntimeError("reader server differs from frozen reader path")
        self.reader_model = actual_reader_path
        self.reader_version = f"{reader_path}@{reader_revision}"
        tokenizer_path = self.dynamic_config.model.get("tokenizer_path")
        tokenizer_revision = self.dynamic_config.model.get("tokenizer_revision")
        if not tokenizer_path or not tokenizer_revision:
            raise DynamicRuntimeError("frozen tokenizer path/revision are required")
        tokenizer = load_tokenizer(str(tokenizer_path), str(tokenizer_revision))
        self.accounting = TokenAccounting.from_tokenizer(
            tokenizer,
            name=str(tokenizer_path),
            revision=str(tokenizer_revision),
        )
        manifest = self.runtime_index.manifest
        if (
            not _same_model_identity(self.accounting.name, manifest.tokenizer_name)
            or self.accounting.revision != manifest.tokenizer_revision
            or self.accounting.chat_template_sha256 != manifest.chat_template_sha256
        ):
            raise DynamicRuntimeError("runtime tokenizer differs from data manifest")
        self.repeat_times = int(task.repeat_times or 1)

    @property
    def asynchronous(self) -> bool:
        return True

    @property
    def repeatable(self) -> bool:
        return True

    async def _frozen_reader(
        self, messages: tuple[dict[str, str], ...]
    ) -> FrozenReaderOutput:
        answer_max = int(self.dynamic_config.budget["answer_max_new_tokens"])

        def invoke() -> Any:
            return self.reader_client.chat.completions.create(
                model=self.reader_model,
                messages=list(messages),
                temperature=0.0,
                max_tokens=answer_max,
            )

        completion = await asyncio.to_thread(invoke)
        choices = getattr(completion, "choices", None)
        if not choices or not isinstance(choices[0].message.content, str):
            raise DynamicRuntimeError("frozen reader returned no text")
        usage = getattr(completion, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if isinstance(total_tokens, bool) or not isinstance(total_tokens, int):
            raise DynamicRuntimeError("frozen reader returned no token receipt")
        return FrozenReaderOutput(
            answer_text=choices[0].message.content,
            total_token_count=total_tokens,
            reader_version=self.reader_version,
        )

    async def run_async(self) -> list[Experience]:
        expected_k = int(self.dynamic_config.data["memory_rollouts_per_group"])
        if self.repeat_times != expected_k:
            raise DynamicRuntimeError("scheduler repeat_times differs from frozen K")
        m = int(self.dynamic_config.data["questions_per_snapshot"])
        public_queries = _select_queries(
            self.runtime_index.public_queries.get(self.history.history_id, ()),
            count=m,
            seed=int(self.dynamic_config.experiment["seed"]),
            history_id=self.history.history_id,
        )
        private_index = self.runtime_index.private_queries.get(
            self.history.history_id, {}
        )
        try:
            private_queries = tuple(
                private_index[item.query_id] for item in public_queries
            )
        except KeyError as exc:
            raise DynamicRuntimeError("public/private query join is incomplete") from exc
        actual_policy_path = await self.model.model_path_async
        expected_policy_path = self.dynamic_config.model.get("policy_path")
        if (
            not isinstance(actual_policy_path, str)
            or not expected_policy_path
            or not _same_model_identity(actual_policy_path, str(expected_policy_path))
        ):
            raise DynamicRuntimeError("rollout model differs from frozen policy path")
        model_version = await self.model.model_version_async
        producer = DynamicRuntimeProducer(
            policy_model=self.model,
            frozen_reader=self._frozen_reader,
            accounting=self.accounting,
            config=self.dynamic_config.model_dump(mode="python"),
            policy_version=f"model_version:{model_version}",
            generation_args=self.rollout_args,
        )
        produced = await producer.produce_group(
            history=self.history,
            public_queries=public_queries,
            private_queries=private_queries,
            events=self.runtime_index.events.get(self.history.history_id, ()),
            source_registry=self.runtime_index.sources.get(self.history.history_id, ()),
            batch_id=self.task.batch_id,
            task_id=self.task.task_id,
            run_id_base=self.run_id_base,
            repeat_times=self.repeat_times,
        )
        experiences = list(produced.experiences)
        if not experiences:
            raise DynamicRuntimeError("runtime producer emitted no ingest Experience")
        info = dict(experiences[0].info or {})
        info["dynamic_group_receipt"] = produced.receipt
        experiences[0].info = info
        return experiences


__all__ = [
    "AgeMemDynamicMultiQueryV2Training",
    "WORKFLOW_NAME",
    "publish_complete_dynamic_group",
]
