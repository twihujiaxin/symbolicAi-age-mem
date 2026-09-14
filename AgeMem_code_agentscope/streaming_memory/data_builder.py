"""Deterministic, split-first HotpotQA builder for long streaming histories."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    ManifestFile,
    PublicChunk,
    PublicSourceSpan,
    QueryGold,
    QueryRecord,
    SourceSentenceRef,
    StreamEpisodePublic,
    StreamingBuildConfig,
    StreamingManifest,
)
from .token_budget import TokenAccounting, sha256_text


class StreamingDataError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    rows = [canonical_json(record) for record in records]
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8", newline="\n")
    return len(rows)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paragraph_identity(title: str, sentences: Sequence[str]) -> tuple[str, str, str]:
    body = "".join(sentences).strip()
    exact = sha256_text(f"{title.strip()}\0{body}")
    normalized = " ".join(f"{title} {body}".casefold().split())
    return exact, sha256_text(normalized), body


def _canonical_title(title: str) -> str:
    return " ".join(title.casefold().split())


def _simhash(text: str) -> int:
    terms = " ".join(text.casefold().split()).split()
    weights = [0] * 64
    for term in terms:
        value = int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _sentence_digest(sentence: str) -> str:
    return sha256_text(sentence.strip())


def _git_state(repository: Path) -> tuple[str | None, str | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True,
        ).stdout.decode("ascii").strip()
        patch = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=repository, check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repository,
            check=True, capture_output=True,
        ).stdout
        if patch or untracked:
            digest = hashlib.sha256()
            digest.update(patch)
            for raw_path in filter(None, untracked.split(b"\0")):
                digest.update(b"\0path\0" + raw_path + b"\0content\0")
                candidate = repository / raw_path.decode("utf-8")
                if candidate.is_file():
                    digest.update(candidate.read_bytes())
            dirty = digest.hexdigest()
        else:
            dirty = None
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


@dataclass(frozen=True)
class _Candidate:
    source_split: str
    source_index: int
    row: Mapping[str, Any]

    @property
    def question_id(self) -> str:
        return str(self.row["id"])


class StreamingHotpotBuilder:
    """Build episodes only after raw QA and document identities are assigned.

    The builder never uses supporting labels to select or order public text.
    Labels are resolved only while emitting the separate gold file.
    """

    def __init__(
        self,
        *,
        dataset: Any,
        config: StreamingBuildConfig,
        accounting: TokenAccounting,
        source_path: Path,
        source_fingerprints: Mapping[str, str],
        repository_root: Path,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.accounting = accounting
        self.source_path = source_path.resolve()
        self.source_fingerprints = dict(source_fingerprints)
        self.repository_root = repository_root.resolve()
        self.rejected: Counter[str] = Counter()

    @staticmethod
    def from_local_dataset(
        *, config: StreamingBuildConfig, accounting: TokenAccounting, repository_root: Path
    ) -> "StreamingHotpotBuilder":
        try:
            from datasets import load_from_disk
        except ImportError as exc:  # pragma: no cover
            raise StreamingDataError("datasets>=4,<5 is required") from exc
        configured = config.data.get("source_path")
        source_path = (
            Path(configured).expanduser()
            if configured
            else repository_root.parent / "data" / "hotpot_qa" / "fullwiki"
        )
        if not source_path.is_dir():
            raise StreamingDataError(f"HotpotQA DatasetDict not found: {source_path}")
        dataset = load_from_disk(str(source_path))
        if not {"train", "validation"}.issubset(dataset):
            raise StreamingDataError("source must contain train and validation")
        fingerprints = {
            name: str(getattr(split, "_fingerprint", "unavailable"))
            for name, split in dataset.items()
        }
        return StreamingHotpotBuilder(
            dataset=dataset,
            config=config,
            accounting=accounting,
            source_path=source_path,
            source_fingerprints=fingerprints,
            repository_root=repository_root,
        )

    def _ordered_candidates(self, source_split: str, target_split: str) -> list[_Candidate]:
        source = self.dataset[source_split]
        seed = int(self.config.data["build_seed"])
        identity_source = source.select_columns(["id"]) if hasattr(source, "select_columns") else source
        candidates = [
            _Candidate(source_split, index, {"id": identity_source[index]["id"]})
            for index in range(len(source))
        ]
        candidates.sort(
            key=lambda item: sha256_text(f"{seed}:{target_split}:{item.question_id}")
        )
        return candidates

    def _assign_rows(self) -> dict[str, list[_Candidate]]:
        counts = self.config.data["debug_episode_counts"]
        candidate_target = int(self.config.data["candidate_query_target"])
        m = int(self.config.data["queries_per_snapshot"])
        if candidate_target < m:
            raise StreamingDataError("candidate_query_target must be >= queries_per_snapshot")
        required = {
            split: int(counts[split]) * candidate_target
            for split in ("train", "dev", "test")
        }
        pools = {
            "dev": self._ordered_candidates("validation", "dev"),
            "test": self._ordered_candidates("validation", "test"),
            "train": self._ordered_candidates("train", "train"),
        }
        assigned: dict[str, list[_Candidate]] = {key: [] for key in pools}
        qa_ids: set[str] = set()
        document_owner: dict[str, str] = {}
        normalized_owner: dict[str, str] = {}
        near_by_split: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
        for split in ("dev", "test", "train"):
            for candidate in pools[split]:
                if len(assigned[split]) >= required[split]:
                    break
                row = self.dataset[candidate.source_split][candidate.source_index]
                candidate = _Candidate(candidate.source_split, candidate.source_index, row)
                if candidate.question_id in qa_ids:
                    self.rejected["duplicate_question_id"] += 1
                    continue
                answer = row.get("answer")
                supports = row.get("supporting_facts") or {}
                if not answer or not (supports.get("title") or []):
                    self.rejected["missing_gold"] += 1
                    continue
                context = row.get("context") or {}
                titles = context.get("title") or []
                sentences = context.get("sentences") or []
                if not titles or len(titles) != len(sentences):
                    self.rejected["invalid_context"] += 1
                    continue
                title_to_sentences = {
                    _canonical_title(str(title)): paragraph
                    for title, paragraph in zip(titles, sentences)
                }
                support_pairs = zip(supports.get("title") or [], supports.get("sent_id") or [])
                if any(
                    _canonical_title(str(title)) not in title_to_sentences
                    or int(sentence_index) < 0
                    or int(sentence_index) >= len(title_to_sentences[_canonical_title(str(title))])
                    for title, sentence_index in support_pairs
                ):
                    self.rejected["unresolved_support_pointer"] += 1
                    continue
                identities = [_paragraph_identity(t, s) for t, s in zip(titles, sentences)]
                if any(document_owner.get(exact) not in (None, split) for exact, _, _ in identities):
                    self.rejected["cross_split_exact_paragraph"] += 1
                    continue
                if any(normalized_owner.get(norm) not in (None, split) for _, norm, _ in identities):
                    self.rejected["cross_split_normalized_paragraph"] += 1
                    continue
                signatures = [_simhash(f"{title} {body}") for title, (_, _, body) in zip(titles, identities)]
                other_signatures = [
                    signature for owner, values in near_by_split.items() if owner != split
                    for signature in values
                ]
                if any(
                    _hamming(signature, previous) <= 3
                    for signature in signatures for previous in other_signatures
                ):
                    self.rejected["cross_split_near_duplicate_simhash"] += 1
                    continue
                assigned[split].append(candidate)
                qa_ids.add(candidate.question_id)
                for exact, norm, _ in identities:
                    document_owner.setdefault(exact, split)
                    normalized_owner.setdefault(norm, split)
                near_by_split[split].extend(signatures)
            if len(assigned[split]) != required[split]:
                raise StreamingDataError(
                    f"could not assign {required[split]} split-safe rows for {split}; "
                    f"got {len(assigned[split])}"
                )
        return assigned

    def _chunk_documents(self, documents: Sequence[dict[str, Any]]) -> tuple[PublicChunk, ...]:
        target = int(self.config.environment["chunk_target_tokens"])
        maximum = int(self.config.environment["chunk_max_tokens"])
        atomic: list[tuple[str, PublicSourceSpan]] = []
        for doc in documents:
            title, sentences = doc["title"], doc["sentences"]
            start = 0
            current: list[str] = []
            for index, sentence in enumerate(sentences):
                proposed = f"[{title}]\n" + "".join(current + [sentence]).strip()
                if current and self.accounting.count_text(proposed) > maximum:
                    text = f"[{title}]\n" + "".join(current).strip()
                    atomic.append((text, PublicSourceSpan(
                        document_key=doc["document_key"], title=title,
                        sentence_start=start, sentence_end=index - 1,
                        document_sha256=doc["document_sha256"],
                    )))
                    start, current = index, [sentence]
                else:
                    current.append(sentence)
            if current:
                text = f"[{title}]\n" + "".join(current).strip()
                if self.accounting.count_text(text) > maximum:
                    raise StreamingDataError("one source sentence exceeds chunk_max_tokens")
                atomic.append((text, PublicSourceSpan(
                    document_key=doc["document_key"], title=title,
                    sentence_start=start, sentence_end=len(sentences) - 1,
                    document_sha256=doc["document_sha256"],
                )))
        chunks: list[PublicChunk] = []
        texts: list[str] = []
        spans: list[PublicSourceSpan] = []
        for text, span in atomic:
            proposed = "\n\n".join(texts + [text])
            if texts and self.accounting.count_text(proposed) > target:
                body = "\n\n".join(texts)
                chunks.append(PublicChunk(
                    chunk_id=f"chunk-{len(chunks):04d}", text=body,
                    sources=tuple(spans), content_token_count=self.accounting.count_text(body),
                ))
                texts, spans = [], []
            texts.append(text)
            spans.append(span)
        if texts:
            body = "\n\n".join(texts)
            chunks.append(PublicChunk(
                chunk_id=f"chunk-{len(chunks):04d}", text=body,
                sources=tuple(spans), content_token_count=self.accounting.count_text(body),
            ))
        if any(chunk.content_token_count > maximum for chunk in chunks):
            raise StreamingDataError("constructed chunk exceeds chunk_max_tokens")
        return tuple(chunks)

    def _build_episode(
        self, split: str, episode_index: int, candidates: Sequence[_Candidate]
    ) -> tuple[StreamEpisodePublic, list[QueryRecord], list[QueryGold], dict[str, Any]]:
        episode_id = f"smq-{split}-{episode_index:04d}"
        docs: dict[tuple[str, str], dict[str, Any]] = {}
        row_docs: dict[str, dict[str, dict[str, Any]]] = {}
        for candidate in candidates:
            context = candidate.row["context"]
            row_docs[candidate.question_id] = {}
            for title, sentences in zip(context["title"], context["sentences"]):
                exact, _, body = _paragraph_identity(title, sentences)
                key = (title.strip(), exact)
                doc = docs.setdefault(key, {
                    "document_key": f"doc-{exact[:20]}", "title": title.strip(),
                    "sentences": tuple(str(s) for s in sentences),
                    "document_sha256": exact, "body": body,
                })
                row_docs[candidate.question_id][_canonical_title(title)] = doc
        ordered_docs = list(docs.values())
        rng = random.Random(int(self.config.data["build_seed"]) + episode_index)
        rng.shuffle(ordered_docs)
        chunks = self._chunk_documents(ordered_docs)
        public = StreamEpisodePublic(
            episode_id=episode_id,
            history_family_id=f"history-{split}-{episode_index:04d}",
            chunks=chunks,
            context_budget_tokens=int(self.config.environment["context_total_tokens"]),
            memory_budget_tokens=int(self.config.environment["persistent_memory_tokens"]),
        )
        questions, gold = [], []
        source_revision = str(self.config.data.get("source_revision") or "local-save-to-disk")
        evaluated_candidates = candidates[: int(self.config.data["queries_per_snapshot"])]
        for query_index, candidate in enumerate(evaluated_candidates):
            row = candidate.row
            query_id = f"{episode_id}-q{query_index:02d}"
            questions.append(QueryRecord(
                query_id=query_id, episode_id=episode_id, question=str(row["question"]).strip()
            ))
            refs = []
            supports = row["supporting_facts"]
            for title, sentence_index in zip(supports["title"], supports["sent_id"]):
                doc = row_docs[candidate.question_id].get(_canonical_title(str(title)))
                if doc is None or int(sentence_index) >= len(doc["sentences"]):
                    raise StreamingDataError(
                        "support pointer is absent from public history: "
                        f"question={candidate.question_id}, title={title!r}, "
                        f"sentence_index={sentence_index}"
                    )
                sentence = doc["sentences"][int(sentence_index)]
                refs.append(SourceSentenceRef(
                    source_revision=source_revision,
                    document_key=doc["document_key"], title=str(title),
                    sentence_index=int(sentence_index), sentence_sha256=_sentence_digest(sentence),
                ))
            gold.append(QueryGold(
                query_id=query_id, answer=str(row["answer"]).strip(),
                support_refs=tuple(refs), source_question_id=candidate.question_id,
            ))
        detail = {
            "episode_id": episode_id,
            "split": split,
            "source_question_ids": [c.question_id for c in candidates],
            "document_keys": [doc["document_key"] for doc in ordered_docs],
            "history_tokens": sum(chunk.content_token_count for chunk in chunks),
            "chunk_count": len(chunks),
            "source_registry": [
                {
                    "source_dataset": "hotpot_qa",
                    "source_revision": source_revision,
                    "document_key": doc["document_key"],
                    "title": doc["title"],
                    "sentence_index": index,
                    "sentence": sentence.strip(),
                    "sentence_sha256": _sentence_digest(sentence),
                }
                for doc in ordered_docs
                for index, sentence in enumerate(doc["sentences"])
                if sentence.strip()
            ],
        }
        return public, questions, gold, detail

    def _select_history_prefix(
        self, split: str, episode_index: int, candidates: Sequence[_Candidate]
    ) -> tuple[StreamEpisodePublic, list[QueryRecord], list[QueryGold], dict[str, Any]]:
        m = int(self.config.data["queries_per_snapshot"])
        targets = tuple(float(value) for value in self.config.data["alpha_targets"])
        alpha_target = targets[episode_index % len(targets)]
        tolerance = float(self.config.data["alpha_tolerance"])
        context_budget = int(self.config.environment["context_total_tokens"])
        lower = alpha_target * (1.0 - tolerance) * context_budget
        upper = alpha_target * (1.0 + tolerance) * context_budget
        if alpha_target >= 5.0:
            lower, upper = max(lower, 5.0 * context_budget), min(upper, 10.0 * context_budget)
        best = None
        for size in range(m, len(candidates) + 1):
            built = self._build_episode(split, episode_index, candidates[:size])
            history_tokens = built[3]["history_tokens"]
            distance = abs(history_tokens - alpha_target * context_budget)
            if best is None or distance < best[0]:
                best = (distance, built)
            if lower <= history_tokens <= upper:
                detail = dict(built[3])
                detail.update({
                    "alpha_target": alpha_target,
                    "alpha_actual": history_tokens / context_budget,
                    "candidate_question_count": size,
                    "evaluated_question_count": m,
                })
                return built[0], built[1], built[2], detail
        assert best is not None
        self.rejected["alpha_out_of_tolerance"] += 1
        raise StreamingDataError(
            f"episode {split}/{episode_index} cannot meet alpha={alpha_target} "
            f"within tolerance={tolerance}; closest_tokens={best[1][3]['history_tokens']}"
        )

    def build(self, output_root: Path) -> StreamingManifest:
        output_root = output_root.resolve()
        resolved_identity = {
            "config": self.config.model_dump(mode="json"),
            "source_fingerprints": self.source_fingerprints,
            "tokenizer_name": self.accounting.name,
            "tokenizer_revision": self.accounting.revision,
            "chat_template_sha256": self.accounting.chat_template_sha256,
        }
        config_sha = sha256_text(canonical_json(resolved_identity))
        build_id = f"stream-mq-{config_sha[:12]}"
        if output_root.exists() and any(output_root.iterdir()):
            manifest_path = output_root / "manifest.json"
            if manifest_path.is_file():
                existing = StreamingManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                if existing.config_sha256 == config_sha:
                    validate_manifest(manifest_path)
                    return existing
            raise StreamingDataError(f"refusing to overwrite non-empty output: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        assigned = self._assign_rows()
        m = int(self.config.data["queries_per_snapshot"])
        candidate_target = int(self.config.data["candidate_query_target"])
        publics, questions, golds, details = [], [], [], []
        for split in ("train", "dev", "test"):
            rows = assigned[split]
            for start in range(0, len(rows), candidate_target):
                public, query_rows, gold_rows, detail = self._select_history_prefix(
                    split, start // candidate_target, rows[start : start + candidate_target]
                )
                publics.append(public.model_dump(mode="json"))
                questions.extend(row.model_dump(mode="json") for row in query_rows)
                golds.extend(row.model_dump(mode="json") for row in gold_rows)
                details.append(detail)
        source_registry: dict[tuple[str, int, str], dict[str, Any]] = {}
        public_details = []
        for detail in details:
            clean = dict(detail)
            for record in clean.pop("source_registry"):
                key = (record["document_key"], record["sentence_index"], record["sentence_sha256"])
                source_registry[key] = record
            public_details.append(clean)
        split_audit = {
            "schema_version": "agemem.stream_mq.split_audit.v1",
            "split_first": True,
            "qa_overlap": 0,
            "exact_paragraph_overlap": 0,
            "normalized_paragraph_overlap": 0,
            "near_duplicate_simhash_hamming_le_3_overlap": 0,
            "near_duplicate_method": "64-bit token simhash, cross-split Hamming distance <= 3",
            "episode_sources": public_details,
        }
        files_and_rows = []
        for name, rows in (
            ("episodes.public.jsonl", publics),
            ("episodes.questions.jsonl", questions),
            ("episodes.gold.jsonl", golds),
            ("source_registry.jsonl", [source_registry[key] for key in sorted(source_registry)]),
            ("split_audit.jsonl", [split_audit]),
            ("rejected.jsonl", [{"reason": k, "count": v} for k, v in sorted(self.rejected.items())]),
        ):
            path = output_root / name
            count = _write_jsonl(path, rows)
            files_and_rows.append(ManifestFile(path=name, sha256=_file_digest(path), rows=count))
        split_counts = {
            split: sum(item["split"] == split for item in public_details)
            for split in ("train", "dev", "test")
        }
        query_counts = {
            split: split_counts[split] * m for split in ("train", "dev", "test")
        }
        history_tokens = [d["history_tokens"] for d in public_details]
        chunk_sizes = [c["content_token_count"] for p in publics for c in p["chunks"]]
        git_commit, dirty = _git_state(self.repository_root)
        manifest = StreamingManifest(
            build_id=build_id, config_sha256=config_sha,
            source_path=str(self.source_path),
            source_revision=str(self.config.data.get("source_revision") or "local-save-to-disk"),
            source_fingerprints=self.source_fingerprints,
            tokenizer_name=self.accounting.name, tokenizer_revision=self.accounting.revision,
            chat_template_sha256=self.accounting.chat_template_sha256,
            split_rule="train<-source/train; dev,test<-disjoint source/validation; cross-split paragraph digests forbidden",
            build_seed=int(self.config.data["build_seed"]), split_counts=split_counts,
            query_counts=query_counts, files=tuple(files_and_rows),
            statistics={
                "episodes": len(publics), "queries_per_snapshot": m,
                "candidate_query_target": candidate_target,
                "alpha_targets": list(self.config.data["alpha_targets"]),
                "alpha_actual_min": min(d["alpha_actual"] for d in public_details),
                "alpha_actual_max": max(d["alpha_actual"] for d in public_details),
                "source_registry_sentences": len(source_registry),
                "history_token_min": min(history_tokens), "history_token_max": max(history_tokens),
                "history_token_mean": sum(history_tokens) / len(history_tokens),
                "chunk_token_min": min(chunk_sizes), "chunk_token_max": max(chunk_sizes),
                "chunk_count": len(chunk_sizes), "question_visible_during_ingest": False,
                "gold_visible_to_policy": False,
            },
            rejected=dict(self.rejected), git_commit=git_commit, dirty_patch_sha256=dirty,
        )
        (output_root / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = StreamingManifest.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.parent
    for item in manifest.files:
        target = root / item.path
        if not target.is_file() or _file_digest(target) != item.sha256:
            raise StreamingDataError(f"manifest file missing or changed: {item.path}")
        if len(_load_jsonl(target)) != item.rows:
            raise StreamingDataError(f"manifest row count changed: {item.path}")
    publics = [StreamEpisodePublic.model_validate(row) for row in _load_jsonl(root / "episodes.public.jsonl")]
    questions = [QueryRecord.model_validate(row) for row in _load_jsonl(root / "episodes.questions.jsonl")]
    gold = [QueryGold.model_validate(row) for row in _load_jsonl(root / "episodes.gold.jsonl")]
    registry_rows = _load_jsonl(root / "source_registry.jsonl")
    registry = {
        (row["document_key"], int(row["sentence_index"]), row["sentence_sha256"])
        for row in registry_rows
    }
    episode_ids = {item.episode_id for item in publics}
    if len(episode_ids) != len(publics):
        raise StreamingDataError("episode IDs are not unique")
    if any(item.episode_id not in episode_ids for item in questions):
        raise StreamingDataError("question references an unknown episode")
    qids = {item.query_id for item in questions}
    if len(qids) != len(questions) or qids != {item.query_id for item in gold}:
        raise StreamingDataError("question/gold query IDs do not join exactly")
    missing_refs = [
        ref for item in gold for ref in item.support_refs
        if (ref.document_key, ref.sentence_index, ref.sentence_sha256) not in registry
    ]
    if missing_refs:
        raise StreamingDataError("gold support reference is absent from source registry")
    public_text = canonical_json([p.model_dump(mode="json") for p in publics])
    forbidden = [item.answer for item in gold if item.answer and item.answer in public_text]
    # Answers may naturally occur in source text; this is not leakage. What is
    # forbidden is serializing gold fields or questions into the public schema.
    if any(key in public_text for key in ('"answer":', '"support_refs":', '"question":')):
        raise StreamingDataError("private fields leaked into public episodes")
    return {
        "status": "pass", "episodes": len(publics), "queries": len(questions),
        "natural_answer_occurrences_in_source": len(forbidden),
        "question_gold_join": len(qids), "source_registry_sentences": len(registry),
        "files": len(manifest.files),
    }


__all__ = ["StreamingDataError", "StreamingHotpotBuilder", "validate_manifest"]
