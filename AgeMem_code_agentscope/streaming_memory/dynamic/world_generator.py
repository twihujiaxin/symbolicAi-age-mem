"""Deterministic D0/D1 dynamic world generation and split-first manifests."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..token_budget import TokenAccounting, sha256_text
from .environment import canonical_dynamic_payload
from .independent_validator import validate_private_labels
from .schema import (
    DynamicBuildConfig,
    DynamicHistoryPublic,
    DynamicManifest,
    DynamicManifestFile,
    DynamicPublicChunk,
    DynamicQueryPrivate,
    DynamicQueryPublic,
    PrivateEvent,
    SourceRegistryRecord,
    SplitAssignment,
)
from .world_oracle import WorldOracle


class DynamicDataError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _opaque(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{sha256_text(_canonical(parts))[:20]}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    path.write_text(
        "".join(_canonical(row) + "\n" for row in values),
        encoding="utf-8",
        newline="\n",
    )
    return len(values)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> tuple[str | None, str | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        untracked = bytearray()
        for relative in untracked_output.decode(
            "utf-8", errors="surrogateescape"
        ).splitlines():
            target = (root / relative).resolve()
            if target.is_file() and root in target.parents:
                untracked.extend(relative.encode("utf-8"))
                untracked.extend(b"\0")
                untracked.extend(hashlib.sha256(target.read_bytes()).digest())
        dirty_payload = diff + bytes(untracked)
        dirty = hashlib.sha256(dirty_payload).hexdigest() if dirty_payload else None
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _render_event(entity: str, relation: str, value: str, effective_at: int) -> str:
    relation_text = {
        "project_owner": "负责人",
        "member_team": "所属团队",
        "site_region": "所在区域",
    }.get(relation, relation)
    return f"自第 {effective_at} 日起，{entity}的{relation_text}为{value}。"


def _make_source(*, history_id: str, index: int, text: str) -> SourceRegistryRecord:
    return SourceRegistryRecord(
        source_ref=_opaque("src", history_id, index),
        history_id=history_id,
        observed_at=0,
        text=text,
        text_sha256=sha256_text(text),
    )


def _make_event(
    *,
    history_id: str,
    index: int,
    entity: str,
    relation: str,
    value: str,
    effective_at: int,
) -> tuple[PrivateEvent, SourceRegistryRecord]:
    text = _render_event(entity, relation, value, effective_at)
    source = _make_source(history_id=history_id, index=index, text=text)
    event = PrivateEvent(
        event_id=_opaque(
            "evt", history_id, index, entity, relation, value, effective_at
        ),
        history_id=history_id,
        entity=entity,
        relation=relation,
        value=value,
        observed_at=0,
        effective_at=effective_at,
        source_ref=source.source_ref,
        source_text_hash=source.text_sha256,
    )
    return event, source


def _query_shell(
    *,
    history_id: str,
    number: int,
    family: str,
    kind: str,
    entity: str,
    relation: str,
    question: str,
    query_time: int | None = None,
    join_relation: str | None = None,
    answer_type: str = "entity",
    answer_hint: str | None = None,
) -> tuple[DynamicQueryPublic, dict[str, Any]]:
    query_id = _opaque("qry", history_id, number, family, kind)
    public = DynamicQueryPublic(
        query_id=query_id, history_id=history_id, question=question
    )
    shell = {
        "query_id": query_id,
        "history_id": history_id,
        "task_family": family,
        "query_kind": kind,
        "entity": entity,
        "relation": relation,
        "query_time": query_time,
        "join_relation": join_relation,
        "target_value": answer_hint,
        "answers": (answer_hint or "placeholder",),
        "answer_available_at": 0,
        "support_alternatives": (("placeholder",),),
        "answer_type": answer_type,
        "answerable": True,
    }
    return public, shell


class DynamicWorldBuilder:
    """Build controlled D1 language from split-assigned synthetic families."""

    def __init__(
        self,
        *,
        config: DynamicBuildConfig,
        accounting: TokenAccounting,
        repository_root: Path,
    ) -> None:
        self.config = config
        self.accounting = accounting
        self.repository_root = repository_root.resolve()

    def _base_world(
        self, family_index: int, target_tokens: int
    ) -> tuple[list[PrivateEvent], list[SourceRegistryRecord], dict[str, str]]:
        seed = int(self.config.experiment["seed"])
        history_id = _opaque("hist", seed, family_index)
        project = f"星河项目{family_index:04d}"
        person_a = f"成员甲{family_index:04d}"
        person_b = f"成员乙{family_index:04d}"
        team_a0 = f"晨曦组{family_index:04d}"
        team_a1 = f"远航组{family_index:04d}"
        team_b0 = f"青峰组{family_index:04d}"
        team_b1 = f"云杉组{family_index:04d}"
        events: list[PrivateEvent] = []
        sources: list[SourceRegistryRecord] = []

        def add(entity: str, relation: str, value: str, effective: int) -> None:
            event, source = _make_event(
                history_id=history_id,
                index=len(sources),
                entity=entity,
                relation=relation,
                value=value,
                effective_at=effective,
            )
            events.append(event)
            sources.append(source)

        def duplicate(source_index: int) -> None:
            sources.append(
                _make_source(
                    history_id=history_id,
                    index=len(sources),
                    text=sources[source_index].text,
                )
            )

        # The critical A->B->A timeline is separated by meaningful unrelated updates.
        add(project, "project_owner", person_a, 1)
        add(person_a, "member_team", team_a0, 1)
        add(person_b, "member_team", team_b0, 1)

        rng = random.Random(seed * 1_000_003 + family_index)

        def filler_until(limit: int, phase: int) -> None:
            while (
                sum(self.accounting.count_text(item.text) for item in sources) < limit
            ):
                number = len(sources)
                entity = f"背景实体{family_index:04d}-{number:04d}"
                value = f"区域{rng.randrange(10000):04d}"
                add(entity, "site_region", value, phase)

        filler_until(target_tokens // 4, 2)
        duplicate(0)
        add(project, "project_owner", person_b, 4)
        add(person_b, "member_team", team_b1, 4)
        filler_until(target_tokens // 2, 5)
        duplicate(1)
        add(person_a, "member_team", team_a1, 6)
        filler_until(3 * target_tokens // 4, 6)
        add(project, "project_owner", person_a, 7)
        duplicate(0)
        filler_until(target_tokens, 8)
        names = {
            "history_id": history_id,
            "project": project,
            "person_a": person_a,
            "person_b": person_b,
            "team_a0": team_a0,
            "team_a1": team_a1,
            "team_b0": team_b0,
            "team_b1": team_b1,
        }
        return events, sources, names

    def _chunk(
        self, events: list[PrivateEvent], sources: list[SourceRegistryRecord]
    ) -> tuple[
        tuple[DynamicPublicChunk, ...], list[PrivateEvent], list[SourceRegistryRecord]
    ]:
        target = int(self.config.budget["chunk_target_tokens"])
        maximum = int(self.config.budget["chunk_max_tokens"])
        chunks: list[DynamicPublicChunk] = []
        current: list[SourceRegistryRecord] = []
        current_tokens = 0

        def commit() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            index = len(chunks)
            text = "\n".join(item.text for item in current)
            chunks.append(
                DynamicPublicChunk(
                    chunk_id=_opaque("chunk", current[0].history_id, index),
                    observed_at=index,
                    text=text,
                    source_refs=tuple(item.source_ref for item in current),
                    content_token_count=self.accounting.count_text(text),
                )
            )
            current = []
            current_tokens = 0

        for source in sources:
            cost = self.accounting.count_text(source.text)
            if cost > maximum:
                raise DynamicDataError("one sentence exceeds chunk_max_tokens")
            if current and current_tokens + cost > target:
                commit()
            current.append(source)
            current_tokens += cost
            if current_tokens >= maximum:
                commit()
        commit()
        ref_to_chunk = {
            ref: chunk.observed_at for chunk in chunks for ref in chunk.source_refs
        }
        updated_sources = [
            item.model_copy(update={"observed_at": ref_to_chunk[item.source_ref]})
            for item in sources
        ]
        updated_events = [
            item.model_copy(update={"observed_at": ref_to_chunk[item.source_ref]})
            for item in events
        ]
        return tuple(chunks), updated_events, updated_sources

    def _queries(
        self, events: Sequence[PrivateEvent], names: dict[str, str]
    ) -> tuple[list[DynamicQueryPublic], list[DynamicQueryPrivate]]:
        h, project, a, b = (
            names["history_id"],
            names["project"],
            names["person_a"],
            names["person_b"],
        )
        specs = [
            (
                "current_state",
                "current_state",
                project,
                "project_owner",
                None,
                None,
                f"阅读结束时，{project}的负责人是谁？",
                "entity",
                None,
            ),
            (
                "historical_state",
                "state_at",
                project,
                "project_owner",
                2,
                None,
                f"第 2 日，{project}的负责人是谁？",
                "entity",
                None,
            ),
            (
                "multi_update",
                "state_at",
                project,
                "project_owner",
                5,
                None,
                f"第 5 日，{project}的负责人是谁？",
                "entity",
                None,
            ),
            (
                "multi_update",
                "start_time",
                project,
                "project_owner",
                None,
                None,
                f"{b}从第几日起成为{project}的负责人？",
                "time",
                b,
            ),
            (
                "temporal_join",
                "temporal_join",
                project,
                "project_owner",
                2,
                "member_team",
                f"第 2 日，{project}负责人所属的团队是什么？",
                "entity",
                None,
            ),
            (
                "temporal_join",
                "temporal_join",
                project,
                "project_owner",
                5,
                "member_team",
                f"第 5 日，{project}负责人所属的团队是什么？",
                "entity",
                None,
            ),
            (
                "current_state",
                "current_state",
                a,
                "member_team",
                None,
                None,
                f"阅读结束时，{a}所属的团队是什么？",
                "entity",
                None,
            ),
            (
                "historical_state",
                "state_at",
                a,
                "member_team",
                2,
                None,
                f"第 2 日，{a}所属的团队是什么？",
                "entity",
                None,
            ),
            (
                "retention_under_budget",
                "state_at",
                project,
                "project_owner",
                5,
                None,
                f"请回忆第 5 日{project}由谁负责。",
                "entity",
                None,
            ),
            (
                "retention_under_budget",
                "temporal_join",
                project,
                "project_owner",
                8,
                "member_team",
                f"第 8 日，{project}负责人所属的团队是什么？",
                "entity",
                None,
            ),
        ]
        oracle = WorldOracle(events)
        public: list[DynamicQueryPublic] = []
        private: list[DynamicQueryPrivate] = []
        for number, (
            family,
            kind,
            entity,
            relation,
            time,
            join,
            question,
            answer_type,
            hint,
        ) in enumerate(specs):
            pub, shell = _query_shell(
                history_id=h,
                number=number,
                family=family,
                kind=kind,
                entity=entity,
                relation=relation,
                question=question,
                query_time=time,
                join_relation=join,
                answer_type=answer_type,
                answer_hint=hint,
            )
            draft = DynamicQueryPrivate.model_validate(shell)
            solved = oracle.solve(draft)
            final = draft.model_copy(
                update={
                    "answers": solved.answers,
                    "answer_available_at": solved.answer_available_at,
                    "support_alternatives": (solved.support_event_ids,),
                    "answerable": solved.answerable,
                    "answer_type": "unknown" if not solved.answerable else answer_type,
                }
            )
            public.append(pub)
            private.append(final)
        issues = validate_private_labels(events, private)
        if issues:
            raise DynamicDataError(
                f"independent Oracle validation failed: {issues[:3]}"
            )
        return public, private

    def _d0_rows(self, count: int) -> list[dict[str, Any]]:
        cases = []
        kinds = [
            "current_vs_history",
            "half_open_boundary",
            "a_b_a",
            "temporal_join",
            "unknown",
            "end_life_difference",
        ]
        definitions = {
            "current_vs_history": {
                "events": [[1, "A"], [4, "B"]],
                "queries": [["at", 2, "A"], ["current", None, "B"]],
            },
            "half_open_boundary": {
                "events": [[1, "A"], [4, "B"]],
                "queries": [["at", 3, "A"], ["at", 4, "B"]],
            },
            "a_b_a": {
                "events": [[1, "A"], [4, "B"], [7, "A"]],
                "queries": [["at", 5, "B"]],
            },
            "temporal_join": {
                "events": [
                    ["owner", 1, "Alice"],
                    ["team:Alice", 1, "Red"],
                    ["team:Alice", 4, "Blue"],
                ],
                "queries": [["join", 4, "Blue"]],
            },
            "unknown": {
                "events": [[3, "A"]],
                "queries": [["at", 1, "unknown"]],
            },
            "end_life_difference": {
                "trajectory_a": {
                    "u_M": [[1, 1], [1, 1]],
                    "U_end": 1,
                    "V_retention": 1,
                    "U_life": 1,
                },
                "trajectory_b": {
                    "u_M": [[0, 1], [1, 1]],
                    "U_end": 1,
                    "V_retention": 0.75,
                    "U_life": 0.875,
                },
            },
        }
        for index in range(count):
            kind = kinds[index % len(kinds)]
            cases.append(
                {
                    "fixture_id": _opaque("d0", self.config.experiment["seed"], index),
                    "case_kind": kind,
                    "seed": int(self.config.experiment["seed"]),
                    "contract": definitions[kind],
                }
            )
        return cases

    def build(self, output_root: Path) -> DynamicManifest:
        output_root = output_root.resolve()
        config_json = _canonical(self.config.model_dump(mode="json"))
        config_sha = sha256_text(config_json)
        if output_root.exists() and any(output_root.iterdir()):
            manifest_path = output_root / "manifest.json"
            if manifest_path.is_file():
                existing = DynamicManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                if existing.config_sha256 == config_sha:
                    validate_dynamic_manifest(manifest_path)
                    return existing
            raise DynamicDataError(
                "refusing to overwrite a non-empty output with another identity"
            )
        output_root.mkdir(parents=True, exist_ok=True)

        counts = dict(self.config.data["d1_family_counts"])
        assignment_plan = [
            split
            for split in ("train", "dev", "test")
            for _ in range(int(counts[split]))
        ]
        target_tokens = round(
            float(self.config.data["train_length_ratio"])
            * int(self.config.budget["context_total_tokens"])
        )
        histories: list[DynamicHistoryPublic] = []
        events: list[PrivateEvent] = []
        query_public: list[DynamicQueryPublic] = []
        query_private: list[DynamicQueryPrivate] = []
        sources: list[SourceRegistryRecord] = []
        assignments: list[SplitAssignment] = []
        feasible_reference_tokens: list[int] = []
        feasible_reference_prompt_totals: list[int] = []

        for index, split in enumerate(assignment_plan):
            raw_events, raw_sources, names = self._base_world(index, target_tokens)
            chunks, final_events, final_sources = self._chunk(raw_events, raw_sources)
            history = DynamicHistoryPublic(
                history_id=names["history_id"],
                history_family_id=_opaque(
                    "family", self.config.experiment["seed"], index
                ),
                chunks=chunks,
                context_budget_tokens=int(self.config.budget["context_total_tokens"]),
                memory_budget_tokens=int(
                    self.config.budget["persistent_memory_tokens"]
                ),
            )
            qpub, qpriv = self._queries(final_events, names)
            required_ids = {
                event_id
                for query in qpriv
                for alternative in query.support_alternatives
                for event_id in alternative
            }
            required_events = [
                item for item in final_events if item.event_id in required_ids
            ]
            oracle = WorldOracle(final_events)
            claims = []
            for event in required_events:
                interval = next(
                    item
                    for item in oracle.intervals(event.entity, event.relation)
                    if item.event_id == event.event_id
                )
                claims.append(
                    {
                        "entity": event.entity,
                        "relation": event.relation,
                        "value": event.value,
                        "effective_at": event.effective_at,
                        "valid_from": interval.valid_from,
                        "valid_to": interval.valid_to,
                    }
                )
            reference = {
                "content": "\n".join(
                    next(
                        item.text
                        for item in final_sources
                        if item.source_ref == event.source_ref
                    )
                    for event in required_events
                ),
                "title": "public-rule-timeline",
                "tags": ["timeline"],
                "source_refs": [item.source_ref for item in required_events],
                "claims": claims,
                "custom": {"representation": "ordered_event_timeline"},
            }
            reference_payload = canonical_dynamic_payload(reference)
            feasible_reference_tokens.append(
                self.accounting.count_text(reference_payload)
            )
            feasible_reference_prompt_totals.append(
                max(
                    self.accounting.count_chat(
                        [
                            {
                                "role": "system",
                                "content": "Answer from retrieved memory only.",
                            },
                            {"role": "user", "content": item.question},
                            {
                                "role": "user",
                                "content": "RETRIEVED MEMORY\n" + reference_payload,
                            },
                        ]
                    )
                    + int(self.config.budget["answer_max_new_tokens"])
                    for item in qpub
                )
            )
            histories.append(history)
            events.extend(final_events)
            query_public.extend(qpub)
            query_private.extend(qpriv)
            sources.extend(final_sources)
            assignments.append(
                SplitAssignment(
                    history_id=history.history_id,
                    history_family_id=history.history_family_id,
                    split=split,
                    data_layer="D1",
                )
            )

        file_specs: list[tuple[str, Sequence[Any], str]] = [
            ("histories.public.jsonl", histories, "public"),
            ("queries.public.jsonl", query_public, "public"),
            ("events.private.jsonl", events, "private"),
            ("queries.private.jsonl", query_private, "private"),
            ("source_registry.private.jsonl", sources, "private"),
            ("splits.private.jsonl", assignments, "private"),
            (
                "d0_fixtures.private.jsonl",
                self._d0_rows(int(self.config.data["d0_fixture_count"])),
                "private",
            ),
        ]
        files: list[DynamicManifestFile] = []
        for filename, models, visibility in file_specs:
            path = output_root / filename
            rows = _write_jsonl(
                path,
                (
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump")
                    else item
                    for item in models
                ),
            )
            files.append(
                DynamicManifestFile(
                    path=filename,
                    sha256=_file_digest(path),
                    rows=rows,
                    visibility=visibility,
                )
            )

        split_sets = {
            split: {
                item.history_family_id for item in assignments if item.split == split
            }
            for split in ("train", "dev", "test")
        }
        split_audit = {
            "split_first": True,
            "split_unit": "history_family_id",
            "family_overlap": {
                "train_dev": len(split_sets["train"] & split_sets["dev"]),
                "train_test": len(split_sets["train"] & split_sets["test"]),
                "dev_test": len(split_sets["dev"] & split_sets["test"]),
            },
            "question_selection_depends_on_policy": False,
        }
        audit_path = output_root / "split_audit.json"
        audit_path.write_text(
            json.dumps(split_audit, indent=2) + "\n", encoding="utf-8"
        )
        files.append(
            DynamicManifestFile(
                path="split_audit.json",
                sha256=_file_digest(audit_path),
                rows=1,
                visibility="audit",
            )
        )

        history_tokens = [
            sum(chunk.content_token_count for chunk in item.chunks)
            for item in histories
        ]
        chunk_tokens = [
            chunk.content_token_count for item in histories for chunk in item.chunks
        ]
        event_by_id = {item.event_id: item for item in events}
        evidence_spans = [
            max(event_by_id[event_id].observed_at for event_id in alternative)
            - min(event_by_id[event_id].observed_at for event_id in alternative)
            for query in query_private
            for alternative in query.support_alternatives
        ]
        git_commit, dirty = _git_state(self.repository_root)
        manifest = DynamicManifest(
            build_id=_opaque("build", config_sha),
            config_sha256=config_sha,
            build_seed=int(self.config.experiment["seed"]),
            tokenizer_name=self.accounting.name,
            tokenizer_revision=self.accounting.revision,
            chat_template_sha256=self.accounting.chat_template_sha256,
            split_rule="assign history_family_id before rendering; variants remain in split",
            split_counts={key: int(value) for key, value in counts.items()},
            files=tuple(files),
            statistics={
                "d0_fixture_count": int(self.config.data["d0_fixture_count"]),
                "d1_histories": len(histories),
                "queries": len(query_public),
                "questions_per_candidate_pool": len(query_public) // len(histories),
                "chunks": sum(len(item.chunks) for item in histories),
                "source_registry_records": len(sources),
                "unique_source_texts": len({item.text for item in sources}),
                "duplicate_source_text_records": len(sources)
                - len({item.text for item in sources}),
                "unique_entities": len({item.entity for item in events}),
                "state_update_events": len(events),
                "evidence_span_chunks_max": max(evidence_spans),
                "history_tokens_min": min(history_tokens),
                "history_tokens_mean": sum(history_tokens) / len(history_tokens),
                "history_tokens_max": max(history_tokens),
                "alpha_min": min(history_tokens)
                / int(self.config.budget["context_total_tokens"]),
                "alpha_max": max(history_tokens)
                / int(self.config.budget["context_total_tokens"]),
                "chunk_tokens_min": min(chunk_tokens),
                "chunk_tokens_max": max(chunk_tokens),
                "question_visible_during_ingest": False,
                "gold_visible_to_policy": False,
                "time_semantics": "[valid_from,valid_to), monotonic effective_at",
                "feasible_reference_tokens_min": min(feasible_reference_tokens),
                "feasible_reference_tokens_max": max(feasible_reference_tokens),
                "feasible_reference_within_B": max(feasible_reference_tokens)
                <= int(self.config.budget["persistent_memory_tokens"]),
                "feasible_reference_within_retrieval_cap": max(
                    feasible_reference_tokens
                )
                <= int(self.config.budget["retrieved_payload_tokens"]),
                "feasible_reference_prompt_plus_output_max": max(
                    feasible_reference_prompt_totals
                ),
                "feasible_reference_within_C": max(feasible_reference_prompt_totals)
                <= int(self.config.budget["context_total_tokens"]),
            },
            validation={
                "independent_oracle_issues": 0,
                "cross_split_family_overlap": sum(
                    split_audit["family_overlap"].values()
                ),
                "public_private_separated": True,
            },
            git_commit=git_commit,
            dirty_patch_sha256=dirty,
        )
        (output_root / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest


def validate_dynamic_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = DynamicManifest.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.parent
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for item in manifest.files:
        target = root / item.path
        if not target.is_file() or _file_digest(target) != item.sha256:
            raise DynamicDataError(f"manifest file missing or changed: {item.path}")
        rows = (
            [json.loads(target.read_text(encoding="utf-8"))]
            if target.suffix == ".json"
            else _read_jsonl(target)
        )
        if len(rows) != item.rows:
            raise DynamicDataError(f"manifest row count changed: {item.path}")
        rows_by_file[item.path] = rows
    histories = [
        DynamicHistoryPublic.model_validate(row)
        for row in rows_by_file["histories.public.jsonl"]
    ]
    public_queries = [
        DynamicQueryPublic.model_validate(row)
        for row in rows_by_file["queries.public.jsonl"]
    ]
    private_queries = [
        DynamicQueryPrivate.model_validate(row)
        for row in rows_by_file["queries.private.jsonl"]
    ]
    events = [
        PrivateEvent.model_validate(row) for row in rows_by_file["events.private.jsonl"]
    ]
    registry = [
        SourceRegistryRecord.model_validate(row)
        for row in rows_by_file["source_registry.private.jsonl"]
    ]
    assignments = [
        SplitAssignment.model_validate(row)
        for row in rows_by_file["splits.private.jsonl"]
    ]
    history_ids = {item.history_id for item in histories}
    if len(history_ids) != len(histories):
        raise DynamicDataError("duplicate history_id")
    if {item.query_id for item in public_queries} != {
        item.query_id for item in private_queries
    }:
        raise DynamicDataError("public/private query join mismatch")
    if any(
        item.history_id not in history_ids for item in public_queries + private_queries
    ):
        raise DynamicDataError("query references unknown history")
    registry_map = {item.source_ref: item for item in registry}
    if len(registry_map) != len(registry):
        raise DynamicDataError("duplicate source_ref")
    if any(item.source_ref not in registry_map for item in events):
        raise DynamicDataError("event source missing from private registry")
    if any(
        item.source_text_hash != registry_map[item.source_ref].text_sha256
        for item in events
    ):
        raise DynamicDataError("event/source hash mismatch")
    public_blob = _canonical([item.model_dump(mode="json") for item in histories])
    forbidden = (
        "answers",
        "answer_available_at",
        "support_alternatives",
        "event_id",
        "effective_at",
    )
    if any(f'"{name}"' in public_blob for name in forbidden):
        raise DynamicDataError("private field leaked into public history")
    by_history_events: dict[str, list[PrivateEvent]] = {}
    by_history_queries: dict[str, list[DynamicQueryPrivate]] = {}
    for item in events:
        by_history_events.setdefault(item.history_id, []).append(item)
    for item in private_queries:
        by_history_queries.setdefault(item.history_id, []).append(item)
    issues = [
        issue
        for history_id in history_ids
        for issue in validate_private_labels(
            by_history_events[history_id], by_history_queries[history_id]
        )
    ]
    if issues:
        raise DynamicDataError(f"independent validation failed: {issues[:3]}")
    split_sets = {
        split: {item.history_family_id for item in assignments if item.split == split}
        for split in ("train", "dev", "test")
    }
    overlap = sum(
        len(split_sets[a] & split_sets[b])
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    )
    if overlap:
        raise DynamicDataError("history family leaked across splits")
    return {
        "status": "pass",
        "histories": len(histories),
        "queries": len(public_queries),
        "events": len(events),
        "source_registry_records": len(registry),
        "independent_oracle_issues": 0,
        "cross_split_family_overlap": 0,
        "public_private_separated": True,
    }


__all__ = ["DynamicDataError", "DynamicWorldBuilder", "validate_dynamic_manifest"]
