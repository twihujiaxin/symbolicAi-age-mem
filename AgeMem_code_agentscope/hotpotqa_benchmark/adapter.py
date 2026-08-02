"""Offline adapter for a local Hugging Face HotpotQA fullwiki DatasetDict."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from .models import (
    BenchmarkSplit,
    HotpotFact,
    HotpotQAMemoryTask,
    HotpotQARow,
    HotpotQASmokeConfig,
    HotpotQASmokeManifest,
    SmokeSelection,
    SourceFactPointer,
)


class HotpotQADataError(ValueError):
    """Raised when local HotpotQA data violates the M5 Oracle contract."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_hotpotqa_path() -> Path:
    configured = os.getenv("HOTPOTQA_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (repository_root().parent / "data" / "hotpot_qa" / "fullwiki").resolve()


def default_smoke_config_path() -> Path:
    return repository_root() / "configs" / "m5_hotpotqa_smoke.json"


def _canonical_title(title: str) -> str:
    return " ".join(title.split()).casefold()


def stable_fact_id(
    hotpot_id: str,
    title: str,
    sent_id: int,
    sentence: str,
) -> str:
    """Build a stable ID from an exact official sentence pointer and content."""

    payload = f"{hotpot_id}\0{_canonical_title(title)}\0{sent_id}\0{sentence}".encode(
        "utf-8"
    )
    return f"hp-{hotpot_id}-{hashlib.sha256(payload).hexdigest()[:16]}"


def load_smoke_config(path: Optional[str | Path] = None) -> HotpotQASmokeConfig:
    config_path = Path(path or default_smoke_config_path()).expanduser().resolve()
    return HotpotQASmokeConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )


class HotpotQADataAdapter:
    """Read local save-to-disk data and adapt labeled rows without network calls."""

    def __init__(
        self,
        path: Optional[str | Path] = None,
        *,
        dataset_dict: Optional[Any] = None,
    ) -> None:
        self.path = Path(path or default_hotpotqa_path()).expanduser().resolve()
        if dataset_dict is None:
            if not self.path.is_dir():
                raise HotpotQADataError(
                    f"local HotpotQA DatasetDict does not exist: {self.path}"
                )
            try:
                from datasets import load_from_disk
            except ImportError as exc:  # pragma: no cover - exercised in minimal envs.
                raise HotpotQADataError(
                    "reading Arrow data requires `datasets>=4,<5`; "
                    "install AgeMem_code_agentscope/requirements-hotpotqa.txt"
                ) from exc
            dataset_dict = load_from_disk(str(self.path))
        if set(dataset_dict) != {"train", "validation", "test"}:
            raise HotpotQADataError(
                "HotpotQA fullwiki must contain train, validation, and test splits"
            )
        self.dataset = dataset_dict

    def split_size(self, split: str) -> int:
        return len(self.dataset[split])

    def fingerprints(self) -> Dict[str, str]:
        fingerprints = {}
        for split in ("train", "validation", "test"):
            value = getattr(self.dataset[split], "_fingerprint", None)
            if not value:
                digest = hashlib.sha256()
                for index in range(len(self.dataset[split])):
                    try:
                        row = json.dumps(
                            self.dataset[split][index],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    except (TypeError, ValueError) as exc:
                        raise HotpotQADataError(
                            f"cannot fingerprint source row {split}[{index}]"
                        ) from exc
                    digest.update(len(row).to_bytes(8, "big"))
                    digest.update(row)
                value = digest.hexdigest()
            fingerprints[split] = str(value)
        return fingerprints

    def row(self, split: str, index: int) -> HotpotQARow:
        if split not in self.dataset:
            raise HotpotQADataError(f"unknown source split {split!r}")
        if index < 0 or index >= len(self.dataset[split]):
            raise HotpotQADataError(
                f"row index {index} is outside source split {split!r}"
            )
        try:
            return HotpotQARow.model_validate(self.dataset[split][index])
        except ValidationError as exc:
            raise HotpotQADataError(
                f"invalid HotpotQA row at {split}[{index}]: {exc}"
            ) from exc

    def validate_official_test_is_label_blind(self) -> int:
        """Fail if the official test split unexpectedly exposes Oracle labels."""

        split = self.dataset["test"]
        for index in range(len(split)):
            raw = split[index]
            answer = raw.get("answer")
            supporting = raw.get("supporting_facts") or {}
            titles = supporting.get("title") or []
            sent_ids = supporting.get("sent_id") or []
            if answer is not None or titles or sent_ids:
                raise HotpotQADataError(
                    f"official test row {index} unexpectedly contains hidden labels"
                )
        return len(split)

    @staticmethod
    def _resolve_supporting(
        row: HotpotQARow,
    ) -> Tuple[Tuple[SourceFactPointer, str], ...]:
        title_to_index: Dict[str, int] = {}
        for index, title in enumerate(row.context.title):
            canonical = _canonical_title(title)
            if canonical in title_to_index:
                raise HotpotQADataError(
                    f"row {row.id!r} has ambiguous duplicate context title {title!r}"
                )
            title_to_index[canonical] = index

        resolved = []
        for title, sent_id in row.supporting_facts.pairs():
            paragraph_index = title_to_index.get(_canonical_title(title))
            if paragraph_index is None:
                raise HotpotQADataError(
                    f"supporting pointer {(title, sent_id)!r} is absent from row {row.id!r}"
                )
            paragraph = row.context.sentences[paragraph_index]
            if sent_id >= len(paragraph):
                raise HotpotQADataError(
                    f"supporting pointer {(title, sent_id)!r} is out of range in row {row.id!r}"
                )
            sentence = paragraph[sent_id].strip()
            if not sentence:
                raise HotpotQADataError(
                    f"supporting pointer {(title, sent_id)!r} resolves to empty text"
                )
            fact_id = stable_fact_id(row.id, title, sent_id, sentence)
            resolved.append(
                (
                    SourceFactPointer(
                        fact_id=fact_id,
                        title=title,
                        sent_id=sent_id,
                    ),
                    sentence,
                )
            )
        fact_ids = [pointer.fact_id for pointer, _ in resolved]
        if len(fact_ids) != len(set(fact_ids)):
            raise HotpotQADataError(f"stable fact-ID collision in row {row.id!r}")
        return tuple(resolved)

    @classmethod
    def _validate_labeled_row(
        cls,
        row: HotpotQARow,
        *,
        min_supporting: int,
        max_supporting: int,
    ) -> None:
        if not row.answer or not row.answer.strip():
            raise HotpotQADataError(f"labeled row {row.id!r} has no answer")
        if row.type is None or row.level is None:
            raise HotpotQADataError(f"labeled row {row.id!r} has no type/level")
        count = len(row.supporting_facts.title)
        if not min_supporting <= count <= max_supporting:
            raise HotpotQADataError(
                f"row {row.id!r} has {count} supporting facts outside smoke bounds"
            )
        cls._resolve_supporting(row)

    @staticmethod
    def _selection_slots(size: int) -> List[Tuple[str, bool]]:
        slots: List[Tuple[str, bool]] = [("bridge", True), ("comparison", False)]
        while len(slots) < size:
            slots.append(("bridge" if len(slots) % 2 == 0 else "comparison", False))
        return slots[:size]

    def _candidate_metadata(self, split: str) -> List[Dict[str, Any]]:
        source = self.dataset[split]
        if hasattr(source, "select_columns"):
            source = source.select_columns(["id", "type", "level", "supporting_facts"])
        metadata = []
        for index in range(len(source)):
            row = source[index]
            metadata.append(
                {
                    "source_index": index,
                    "hotpot_id": str(row["id"]),
                    "hotpot_type": row.get("type"),
                    "level": row.get("level"),
                    "supporting_fact_count": len(
                        (row.get("supporting_facts") or {}).get("title") or []
                    ),
                }
            )
        return metadata

    def _select(
        self,
        *,
        source_split: str,
        benchmark_split: BenchmarkSplit,
        size: int,
        config: HotpotQASmokeConfig,
        excluded_ids: Set[str],
    ) -> List[SmokeSelection]:
        candidates = self._candidate_metadata(source_split)
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{config.seed}:{benchmark_split}:{item['hotpot_id']}".encode("utf-8")
            ).hexdigest()
        )
        selected = []
        used = set(excluded_ids)
        for required_type, require_multi in self._selection_slots(size):
            match = None
            for candidate in candidates:
                if candidate["hotpot_id"] in used:
                    continue
                if candidate["hotpot_type"] != required_type:
                    continue
                support_count = candidate["supporting_fact_count"]
                if not (
                    config.min_supporting_facts
                    <= support_count
                    <= config.max_supporting_facts
                ):
                    continue
                if require_multi and support_count < 3:
                    continue
                try:
                    row = self.row(source_split, candidate["source_index"])
                    self._validate_labeled_row(
                        row,
                        min_supporting=config.min_supporting_facts,
                        max_supporting=config.max_supporting_facts,
                    )
                except HotpotQADataError:
                    continue
                match = SmokeSelection(
                    benchmark_split=benchmark_split,
                    source_split=source_split,
                    **candidate,
                )
                break
            if match is None:
                raise HotpotQADataError(
                    f"unable to select a valid {required_type} row for {benchmark_split}"
                )
            selected.append(match)
            used.add(match.hotpot_id)
        return selected

    def build_smoke_manifest(
        self,
        config: HotpotQASmokeConfig,
    ) -> HotpotQASmokeManifest:
        self.validate_official_test_is_label_blind()
        used: Set[str] = set()
        train = self._select(
            source_split="train",
            benchmark_split="train",
            size=config.train_size,
            config=config,
            excluded_ids=used,
        )
        used.update(item.hotpot_id for item in train)
        dev = self._select(
            source_split="validation",
            benchmark_split="dev",
            size=config.dev_size,
            config=config,
            excluded_ids=used,
        )
        used.update(item.hotpot_id for item in dev)
        test = self._select(
            source_split="validation",
            benchmark_split="test",
            size=config.test_size,
            config=config,
            excluded_ids=used,
        )
        selections = tuple(train + dev + test)
        return HotpotQASmokeManifest(
            source_fingerprints=self.fingerprints(),
            seed=config.seed,
            smoke_config_digest=smoke_config_digest(config),
            split_sizes={
                "train": config.train_size,
                "dev": config.dev_size,
                "test": config.test_size,
            },
            selections=selections,
        )

    def verify_manifest(
        self,
        manifest: HotpotQASmokeManifest,
        config: Optional[HotpotQASmokeConfig] = None,
    ) -> None:
        if manifest.source_fingerprints != self.fingerprints():
            raise HotpotQADataError("smoke manifest source fingerprints do not match")
        if config is not None:
            expected_sizes = {
                "train": config.train_size,
                "dev": config.dev_size,
                "test": config.test_size,
            }
            if manifest.seed != config.seed or manifest.split_sizes != expected_sizes:
                raise HotpotQADataError(
                    "smoke manifest seed or split sizes do not match the config"
                )
            if manifest.smoke_config_digest != smoke_config_digest(config):
                raise HotpotQADataError(
                    "smoke manifest does not match the complete smoke config"
                )
        for selection in manifest.selections:
            row = self.row(selection.source_split, selection.source_index)
            actual = (row.id, row.type, row.level, len(row.supporting_facts.title))
            expected = (
                selection.hotpot_id,
                selection.hotpot_type,
                selection.level,
                selection.supporting_fact_count,
            )
            if actual != expected:
                raise HotpotQADataError(
                    f"manifest row mismatch: expected {expected!r}, got {actual!r}"
                )

    def adapt(
        self,
        selection: SmokeSelection,
        config: HotpotQASmokeConfig,
    ) -> HotpotQAMemoryTask:
        row = self.row(selection.source_split, selection.source_index)
        self._validate_labeled_row(
            row,
            min_supporting=config.min_supporting_facts,
            max_supporting=config.max_supporting_facts,
        )
        if row.id != selection.hotpot_id:
            raise HotpotQADataError("selection ID does not match the source row")
        actual_metadata = (
            row.type,
            row.level,
            len(row.supporting_facts.title),
        )
        expected_metadata = (
            selection.hotpot_type,
            selection.level,
            selection.supporting_fact_count,
        )
        if actual_metadata != expected_metadata:
            raise HotpotQADataError(
                "selection metadata does not match the source row: "
                f"expected {expected_metadata!r}, got {actual_metadata!r}"
            )
        resolved = self._resolve_supporting(row)
        supporting_pairs = {
            (_canonical_title(pointer.title), pointer.sent_id)
            for pointer, _ in resolved
        }
        supporting_facts = [
            HotpotFact(
                fact_id=pointer.fact_id,
                title=pointer.title,
                sent_id=pointer.sent_id,
                sentence=sentence,
                stage=1,
                role="supporting",
            )
            for pointer, sentence in resolved
        ]

        distractor_candidates = []
        for title, sentences in zip(row.context.title, row.context.sentences):
            for sent_id, raw_sentence in enumerate(sentences):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                if (_canonical_title(title), sent_id) in supporting_pairs:
                    continue
                rank = hashlib.sha256(
                    f"{config.seed}:{row.id}:{_canonical_title(title)}:{sent_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                distractor_candidates.append((rank, title, sent_id, sentence))
        distractor_candidates.sort()
        distractor_limit = config.stage1_distractors + config.stage2_distractors
        if len(distractor_candidates) < distractor_limit:
            raise HotpotQADataError(
                f"row {row.id!r} has only {len(distractor_candidates)} usable "
                f"distractors; {distractor_limit} are required"
            )
        distractor_facts = []
        for offset, (_, title, sent_id, sentence) in enumerate(
            distractor_candidates[:distractor_limit]
        ):
            stage = 1 if offset < config.stage1_distractors else 2
            distractor_facts.append(
                HotpotFact(
                    fact_id=stable_fact_id(row.id, title, sent_id, sentence),
                    title=title,
                    sent_id=sent_id,
                    sentence=sentence,
                    stage=stage,
                    role="distractor",
                )
            )
        facts = tuple(supporting_facts + distractor_facts)
        fact_ids = [fact.fact_id for fact in facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise HotpotQADataError(f"stable fact-ID collision in adapted row {row.id!r}")
        return HotpotQAMemoryTask(
            task_id=f"hotpot-{row.id}",
            split=selection.benchmark_split,
            question=row.question,
            answer=(row.answer or "").strip(),
            facts=facts,
            supporting_fact_ids=tuple(fact.fact_id for fact in supporting_facts),
            distractor_fact_ids=tuple(fact.fact_id for fact in distractor_facts),
            hotpot_id=row.id,
            source_split=selection.source_split,
            source_index=selection.source_index,
            hotpot_type=row.type,
            level=row.level,
            supporting_fact_pointers=tuple(pointer for pointer, _ in resolved),
            source_context_sentence_count=sum(
                len(sentences) for sentences in row.context.sentences
            ),
        )


def write_manifest(manifest: HotpotQASmokeManifest, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def load_manifest(path: str | Path) -> HotpotQASmokeManifest:
    return HotpotQASmokeManifest.model_validate_json(
        Path(path).expanduser().resolve().read_text(encoding="utf-8")
    )


def manifest_digest(manifest: HotpotQASmokeManifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def smoke_config_digest(config: HotpotQASmokeConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "HotpotQADataAdapter",
    "HotpotQADataError",
    "default_hotpotqa_path",
    "default_smoke_config_path",
    "load_manifest",
    "load_smoke_config",
    "manifest_digest",
    "stable_fact_id",
    "smoke_config_digest",
    "write_manifest",
]
