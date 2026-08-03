"""Offline M6 Oracle-vs-extracted benchmark on canonical M5 trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..action_schema import (
    ActionCreditRecord,
    ActionEvent,
    MigrationFileRecord,
    MigrationManifest,
    TrajectoryStepV2,
    load_migration_manifest,
)
from ..hotpotqa_benchmark import HotpotQADataAdapter
from ..hotpotqa_benchmark.metrics import BenchmarkRecord, OracleBenchmarkReport
from .annotations import (
    AnnotationCorpus,
    AnnotationValidationSummary,
    ManualTripleAnnotation,
    load_annotation_corpus,
)
from .cache import GroupExtractionCache
from .extractor import MockTripleExtractor
from .grounding import (
    EvidenceDigestRelevanceResolver,
    ExtractedAPGrounder,
    GroundedAction,
    derive_memory_delta,
)
from .metrics import (
    APEvaluationRecord,
    APMetrics,
    AcceptanceDecision,
    AcceptanceMetrics,
    RewardActionValue,
    RewardPropagationMetrics,
    TripleEvaluationRecord,
    TripleMetrics,
    score_acceptance,
    score_aps,
    score_reward_propagation,
    score_triples,
)
from .models import (
    EXTRACTOR_OUTPUT_SCHEMA_VERSION,
    ActionBinding,
    EvidenceSpan,
    ExtractionRequest,
    TripleCandidate,
    TripleRecord,
    canonical_digest,
    text_digest,
)
from .rewarding import ExtractedRewardReplay
from .state import CategorySpec, StateTracker


M6_BENCHMARK_CONFIG_SCHEMA_VERSION = "agemem.m6_extraction_benchmark.v1"
M6_BENCHMARK_REPORT_SCHEMA_VERSION = "agemem.m6_extraction_benchmark_report.v1"
M6_FAILURE_SCHEMA_VERSION = "agemem.m6_error_propagation.v1"

Cardinality = Literal["single", "multi"]
LLMEvaluationStatus = Literal["not_run"]

T = TypeVar("T", bound=BaseModel)


class M6BenchmarkError(RuntimeError):
    """Raised when benchmark inputs or derived provenance fail closed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_benchmark_config_path() -> Path:
    return repository_root() / "configs" / "m6_extraction_benchmark.json"


class ExtractionProfileConfig(BaseModel):
    """A deterministic fixture profile, including controlled extraction errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    drop_fact_ids: Tuple[str, ...] = ()
    corrupt_values: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_profile(self) -> "ExtractionProfileConfig":
        if not self.name.strip():
            raise ValueError("profile name must be non-blank")
        if len(self.drop_fact_ids) != len(set(self.drop_fact_ids)):
            raise ValueError("drop_fact_ids must be unique")
        if set(self.drop_fact_ids) & set(self.corrupt_values):
            raise ValueError("one fact cannot be both dropped and corrupted")
        if any(not value.strip() for value in self.corrupt_values.values()):
            raise ValueError("corrupt values must be non-blank")
        return self


class M6BenchmarkConfig(BaseModel):
    """Externalized, strict configuration for the model-free M6 benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_BENCHMARK_CONFIG_SCHEMA_VERSION]
    seed: int = Field(ge=0)
    m5_report_path: str = Field(min_length=1)
    m5_runtime_root: str = Field(min_length=1)
    m6_migration_root: str = Field(min_length=1)
    manual_triples_path: str = Field(min_length=1)
    semantic_targets_path: str = Field(min_length=1)
    reward_profile: str = Field(min_length=1)
    allow_human_oracle_semantic_target: Literal[True]
    llm_evaluation: LLMEvaluationStatus
    category_registry: Dict[str, Cardinality]
    profiles: Tuple[ExtractionProfileConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_config(self) -> "M6BenchmarkConfig":
        if not self.category_registry:
            raise ValueError("category_registry cannot be empty")
        if any(not key or key.casefold() != key for key in self.category_registry):
            raise ValueError("category_registry keys must be lowercase non-empty slugs")
        names = [item.name for item in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("profile names must be unique")
        if "human_backed_mock" not in names or "controlled_error" not in names:
            raise ValueError("benchmark requires perfect and controlled-error profiles")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "M6BenchmarkConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class ErrorPropagationRecord(BaseModel):
    """First per-rollout downstream AP→DFA→reward divergence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_FAILURE_SCHEMA_VERSION] = M6_FAILURE_SCHEMA_VERSION
    profile: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    rollout_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    timestep: int = Field(ge=0)
    oracle_only_aps: Tuple[str, ...] = ()
    extracted_only_aps: Tuple[str, ...] = ()
    extracted_triple_ids: Tuple[str, ...] = ()
    extracted_state_fact_ids: Tuple[str, ...] = ()
    oracle_edges: Tuple[str, ...] = ()
    extracted_edges: Tuple[str, ...] = ()
    oracle_state_before: str = Field(min_length=1)
    extracted_state_before: str = Field(min_length=1)
    oracle_state_after: str = Field(min_length=1)
    extracted_state_after: str = Field(min_length=1)
    oracle_status: Literal["running", "accepted", "rejected", "timed_out"]
    extracted_status: Literal["running", "accepted", "rejected", "timed_out"]
    env_reward_error: float
    milestone_reward_error: float
    violation_reward_error: float
    trend_reward_error: float
    format_reward_error: float
    cost_reward_error: float
    reward_error: float
    profile_drop_fact_ids: Tuple[str, ...] = ()
    profile_corrupt_fact_ids: Tuple[str, ...] = ()

    @field_validator(
        "env_reward_error",
        "milestone_reward_error",
        "violation_reward_error",
        "trend_reward_error",
        "format_reward_error",
        "cost_reward_error",
        "reward_error",
    )
    @classmethod
    def reward_error_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward_error must be finite")
        return value

    def to_json_line(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ExtractionProfileReport(BaseModel):
    """Metrics and deterministic runtime diagnostics for one extractor profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    rollout_count: int = Field(ge=1)
    action_count: int = Field(ge=1)
    triple_metrics: TripleMetrics
    ap_metrics: APMetrics
    acceptance: AcceptanceMetrics
    reward_propagation: RewardPropagationMetrics
    cache_hits: int = Field(ge=0)
    cache_misses: int = Field(ge=0)
    cache_size: int = Field(ge=0)
    extractor_call_count: int = Field(ge=0)
    observation_candidates: int = Field(ge=0)
    action_specific_candidates: int = Field(ge=0)
    quarantined_candidates: int = Field(ge=0)
    provenance_valid_count: int = Field(ge=0)
    provenance_total_count: int = Field(ge=0)
    provenance_integrity_rate: float = Field(ge=0.0, le=1.0)
    accepted_rollouts: int = Field(ge=0)
    rejected_rollouts: int = Field(ge=0)
    timed_out_rollouts: int = Field(ge=0)
    action_credit_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "ExtractionProfileReport":
        if self.rollout_count != (
            self.accepted_rollouts + self.rejected_rollouts + self.timed_out_rollouts
        ):
            raise ValueError("terminal status counts must equal rollout_count")
        if self.provenance_valid_count > self.provenance_total_count:
            raise ValueError("valid provenance count cannot exceed total")
        if self.failure_count > self.rollout_count:
            raise ValueError("at most one first divergence is retained per rollout")
        if (
            self.reward_propagation.action_count != self.action_count
            or self.reward_propagation.trajectory_count != self.rollout_count
            or self.acceptance.action_count != self.rollout_count
        ):
            raise ValueError("metric joins do not match profile action/rollout counts")
        expected = (
            1.0
            if self.provenance_total_count == 0
            else self.provenance_valid_count / self.provenance_total_count
        )
        if not math.isclose(self.provenance_integrity_rate, expected, abs_tol=1e-15):
            raise ValueError("provenance_integrity_rate does not match counts")
        return self


class M6ExtractionBenchmarkReport(BaseModel):
    """Compact, source-text-free M6 benchmark report."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[M6_BENCHMARK_REPORT_SCHEMA_VERSION] = (
        M6_BENCHMARK_REPORT_SCHEMA_VERSION
    )
    benchmark_name: Literal["m6-hotpotqa-extracted-ap-smoke"] = (
        "m6-hotpotqa-extracted-ap-smoke"
    )
    seed: int = Field(ge=0)
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_validation: AnnotationValidationSummary
    m5_report_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    migration_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_rollout_count: int = Field(ge=1)
    canonical_action_count: int = Field(ge=1)
    llm_evaluation: LLMEvaluationStatus
    real_llm_call_count: Literal[0] = 0
    human_oracle_semantic_target_used: Literal[True] = True
    profiles: Tuple[ExtractionProfileReport, ...] = Field(min_length=1)
    failures: Tuple[ErrorPropagationRecord, ...] = ()
    limitations: Tuple[str, ...] = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> "M6ExtractionBenchmarkReport":
        names = [item.name for item in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("profile report names must be unique")
        if any(
            item.rollout_count != self.canonical_rollout_count
            or item.action_count != self.canonical_action_count
            for item in self.profiles
        ):
            raise ValueError("every profile must cover the canonical M5 universe")
        failure_keys = [
            (item.profile, item.task_id, item.rollout_id) for item in self.failures
        ]
        if len(failure_keys) != len(set(failure_keys)):
            raise ValueError("only the first divergence per profile/rollout is allowed")
        known_profiles = set(names)
        if any(item.profile not in known_profiles for item in self.failures):
            raise ValueError("failure row references an unknown profile")
        failure_counts = {
            name: sum(item.profile == name for item in self.failures) for name in names
        }
        if any(
            item.failure_count != failure_counts[item.name] for item in self.profiles
        ):
            raise ValueError("profile failure counts must equal failure rows")
        if self.digest != self.expected_digest():
            raise ValueError("report digest does not match payload")
        return self

    def canonical_dict(self, *, include_digest: bool = True) -> Dict[str, object]:
        data = self.model_dump(mode="json")
        if not include_digest:
            data.pop("digest", None)
        return data

    def expected_digest(self) -> str:
        return canonical_digest(self.canonical_dict(include_digest=False))

    def to_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class M6BenchmarkArtifacts:
    """Generated report and audit artifact locations."""

    report_path: Path
    failures_path: Path
    markdown_path: Path
    report: M6ExtractionBenchmarkReport


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise M6BenchmarkError(f"configured path escapes repository: {value!r}")
    return path


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise M6BenchmarkError(f"manifest path escapes configured root: {relative!r}")
    return path


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise M6BenchmarkError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_models(path: Path, model: Type[T]) -> Tuple[T, ...]:
    if not path.is_file():
        raise M6BenchmarkError(f"required JSONL does not exist: {path}")
    rows: List[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise M6BenchmarkError(f"blank JSONL row at {path}:{line_number}")
            try:
                rows.append(model.model_validate_json(line))
            except ValueError as exc:
                raise M6BenchmarkError(
                    f"invalid {model.__name__} at {path}:{line_number}: {exc}"
                ) from exc
    if not rows:
        raise M6BenchmarkError(f"empty JSONL file: {path}")
    return tuple(rows)


def _model_json_line(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _resolve_annotation_texts(
    corpus: AnnotationCorpus,
    adapter: HotpotQADataAdapter,
) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for record in corpus.manual.records:
        row = adapter.row(record.source_split, record.source_index)
        matches = [
            index
            for index, title in enumerate(row.context.title)
            if title == record.title
        ]
        if len(matches) != 1 or record.sent_id >= len(
            row.context.sentences[matches[0]]
        ):
            raise M6BenchmarkError(f"annotation pointer failed: {record.annotation_id}")
        sentence = row.context.sentences[matches[0]][record.sent_id].strip()
        if text_digest(sentence) != record.text_sha256:
            raise M6BenchmarkError(
                f"annotation text hash failed: {record.annotation_id}"
            )
        texts[record.fact_id] = sentence
    return texts


def _profile_triples(
    record: ManualTripleAnnotation,
    profile: ExtractionProfileConfig,
) -> Tuple[Tuple[str, str, str], ...]:
    if record.fact_id in profile.drop_fact_ids:
        return ()
    corrupted = profile.corrupt_values.get(record.fact_id)
    return tuple(
        (
            item.subject,
            item.category,
            corrupted if corrupted is not None else item.value,
        )
        for item in record.triples
    )


def _raw_payload_for_source(
    source_text: str,
    records: Sequence[ManualTripleAnnotation],
    annotation_texts: Mapping[str, str],
    profile: ExtractionProfileConfig,
) -> Dict[str, object]:
    triples: List[Dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.fact_id):
        evidence_text = annotation_texts[record.fact_id]
        start = source_text.find(evidence_text)
        if start < 0:
            continue
        end = start + len(evidence_text)
        for subject, category, value in _profile_triples(record, profile):
            triples.append(
                {
                    "subject": subject,
                    "category": category,
                    "value": value,
                    "confidence": 1.0,
                    "evidence": [
                        {
                            "source": "observation",
                            "text": evidence_text,
                            "start": start,
                            "end": end,
                        }
                    ],
                }
            )
    return {"schema_version": EXTRACTOR_OUTPUT_SCHEMA_VERSION, "triples": triples}


def _action_specific_candidates(
    *,
    tool_text: str,
    records: Sequence[ManualTripleAnnotation],
    annotation_texts: Mapping[str, str],
    profile: ExtractionProfileConfig,
    extractor_version: str,
    model_version: str,
) -> Tuple[TripleCandidate, ...]:
    if not tool_text:
        return ()
    candidates: List[TripleCandidate] = []
    for record in sorted(records, key=lambda item: item.fact_id):
        evidence_text = annotation_texts[record.fact_id]
        start = tool_text.find(evidence_text)
        if start < 0:
            continue
        span = EvidenceSpan.from_source(
            source="tool_result",
            source_text=tool_text,
            start=start,
            end=start + len(evidence_text),
        )
        for subject, category, value in _profile_triples(record, profile):
            candidates.append(
                TripleCandidate.create(
                    subject=subject,
                    category=category,
                    value=value,
                    confidence=1.0,
                    evidence=(span,),
                    extractor_version=extractor_version,
                    extractor_kind="mock",
                    model_version=model_version,
                )
            )
    unique = {item.candidate_id: item for item in candidates}
    return tuple(unique[key] for key in sorted(unique))


def _binding(action: ActionEvent) -> ActionBinding:
    return ActionBinding(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        action_id=action.action_id,
        assistant_turn_id=action.assistant_turn_id,
        action_index_in_turn=action.action_index_in_turn,
    )


def _markdown(report: M6ExtractionBenchmarkReport) -> str:
    lines = [
        "# M6 Natural-Language Triple Extraction and Explicit State Benchmark",
        "",
        "本报告在 M5 的 30 条规范离线轨迹上比较 Oracle AP 与 Extracted AP。",
        "Triple 抽取使用人工标注支持的确定性 mock；LLM adapter 仅由 fake client 测试，",
        "本报告没有调用真实 LLM，也不代表模型质量。相关性与 coverage 使用独立的人工",
        "Oracle semantic target，未进入 Triple candidate cache。",
        "",
        "## Reproducibility",
        "",
        f"- Report digest: `{report.digest}`",
        f"- M5 report digest: `{report.m5_report_digest}`",
        f"- Migration manifest digest: `{report.migration_manifest_digest}`",
        f"- Annotation corpus digest: `{report.annotation_validation.corpus_digest}`",
        f"- Canonical rollouts/actions: {report.canonical_rollout_count}/{report.canonical_action_count}",
        f"- Real LLM calls: {report.real_llm_call_count} (`{report.llm_evaluation}`)",
        "",
        "## Metric definitions",
        "",
        "- Triple exact key: evidence sentence plus normalized subject/category/value; "
        "micro spans 37 gold triples and macro spans 34 annotated sentences.",
        "- AP exact key: action_id plus normalized proposition; macro spans every action "
        "present on either side.",
        "- False Accept uses Oracle-rejected rollouts as denominator; False Reject uses "
        "Oracle-accepted rollouts as denominator.",
        "- Reward error is Extracted minus Oracle over the exact 224-action join; "
        "trajectory error aggregates the same values over 30 task/rollout keys.",
        "",
        "## Metrics",
        "",
        "| Profile | Triple P/R/F1 | AP P/R/F1 | FA | FR | Reward MAE | "
        "Trajectory abs error | Accepted | Cache hit/miss |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for item in report.profiles:
        triple = item.triple_metrics.micro
        ap = item.ap_metrics.micro
        fa = item.acceptance.false_accept_rate
        fr = item.acceptance.false_reject_rate
        lines.append(
            f"| {item.name} | {triple.precision:.3f}/{triple.recall:.3f}/{triple.f1:.3f} | "
            f"{ap.precision:.3f}/{ap.recall:.3f}/{ap.f1:.3f} | "
            f"{('n/a' if fa is None else f'{fa:.3f}')} "
            f"({item.acceptance.false_accept_numerator}/{item.acceptance.false_accept_denominator}) | "
            f"{('n/a' if fr is None else f'{fr:.3f}')} "
            f"({item.acceptance.false_reject_numerator}/{item.acceptance.false_reject_denominator}) | "
            f"{item.reward_propagation.action_total.mae:.4f} | "
            f"{item.reward_propagation.trajectory_absolute_error_total:.4f} | "
            f"{item.accepted_rollouts}/{item.rollout_count} | "
            f"{item.cache_hits}/{item.cache_misses} |"
        )
    lines.extend(
        [
            "",
            "## Error propagation",
            "",
            f"First downstream-divergence audit rows: {len(report.failures)}. 每行使用 "
            "action_id 连接该点已有的 Triple/StateFact/AP、DFA edge 和 reward；"
            "被漏抽的 Triple 不会伪造 ID，报告也不保存原始句子。",
            "",
            "## Limits",
            "",
            *(f"- {item}" for item in report.limitations),
            "",
        ]
    )
    return "\n".join(lines)


class M6ExtractionBenchmark:
    """Run deterministic mock extraction, explicit state, AP, and reward replay."""

    def __init__(
        self,
        *,
        config: M6BenchmarkConfig,
        adapter: HotpotQADataAdapter,
        repository: Optional[str | Path] = None,
    ) -> None:
        self.config = config.model_copy(deep=True)
        self.adapter = adapter
        self.root = Path(repository or repository_root()).resolve()

    def _load_inputs(
        self,
    ) -> Tuple[
        AnnotationCorpus,
        AnnotationValidationSummary,
        Dict[str, str],
        OracleBenchmarkReport,
        MigrationManifest,
    ]:
        corpus = load_annotation_corpus(
            _resolve_repo_path(self.root, self.config.manual_triples_path),
            _resolve_repo_path(self.root, self.config.semantic_targets_path),
        )
        validation = corpus.validate_against_adapter(self.adapter)
        texts = _resolve_annotation_texts(corpus, self.adapter)
        report_path = _resolve_repo_path(self.root, self.config.m5_report_path)
        report = OracleBenchmarkReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        migration_root = _resolve_repo_path(self.root, self.config.m6_migration_root)
        manifest = load_migration_manifest(migration_root / "migration_manifest.json")
        if manifest.source_report_digest != report.digest:
            raise M6BenchmarkError("migration manifest does not match M5 report")
        if manifest.canonical_rollout_count != len(report.records):
            raise M6BenchmarkError("migration/report rollout counts differ")
        return corpus, validation, texts, report, manifest

    def _run_profile(
        self,
        *,
        profile: ExtractionProfileConfig,
        corpus: AnnotationCorpus,
        annotation_texts: Mapping[str, str],
        report: OracleBenchmarkReport,
        manifest: MigrationManifest,
        runtime_output: Optional[Path],
    ) -> Tuple[ExtractionProfileReport, Tuple[ErrorPropagationRecord, ...]]:
        records_by_task: Dict[str, Tuple[ManualTripleAnnotation, ...]] = {}
        for hotpot_id in sorted({item.hotpot_id for item in corpus.manual.records}):
            records_by_task[f"hotpot-{hotpot_id}"] = tuple(
                item for item in corpus.manual.records if item.hotpot_id == hotpot_id
            )
        all_fact_ids = {item.fact_id for item in corpus.manual.records}
        if (set(profile.drop_fact_ids) | set(profile.corrupt_values)) - all_fact_ids:
            raise M6BenchmarkError(
                f"profile {profile.name!r} references unknown fact IDs"
            )

        known_subjects = tuple(
            sorted(
                {
                    triple.subject
                    for item in corpus.manual.records
                    for triple in item.triples
                },
                key=str.casefold,
            )
        )
        allowed_categories = tuple(sorted(self.config.category_registry))
        category_specs = tuple(
            CategorySpec(name=name, cardinality=cardinality)
            for name, cardinality in sorted(self.config.category_registry.items())
        )
        annotated_categories = {
            triple.category for item in corpus.manual.records for triple in item.triples
        }
        if annotated_categories != set(allowed_categories):
            raise M6BenchmarkError(
                "category registry does not exactly cover annotations"
            )

        migration_root = _resolve_repo_path(self.root, self.config.m6_migration_root)
        source_runtime_root = _resolve_repo_path(self.root, self.config.m5_runtime_root)
        report_record_by_source = {
            item.trajectory_path: item for item in report.records
        }
        if len(report_record_by_source) != len(report.records):
            raise M6BenchmarkError("M5 report contains duplicate trajectory paths")
        loaded: List[
            Tuple[
                MigrationFileRecord,
                BenchmarkRecord,
                Tuple[TrajectoryStepV2, ...],
                Tuple[ActionCreditRecord, ...],
            ]
        ] = []
        responses: Dict[str, object] = {}

        def add_response(
            source_text: str, task_records: Sequence[ManualTripleAnnotation]
        ) -> None:
            key = text_digest(source_text)
            payload = _raw_payload_for_source(
                source_text,
                task_records,
                annotation_texts,
                profile,
            )
            previous = responses.setdefault(key, payload)
            if previous != payload:
                raise M6BenchmarkError(
                    "one observation digest mapped to two fixture payloads"
                )

        for item in manifest.files:
            benchmark_record = report_record_by_source.get(item.source_trajectory_path)
            if benchmark_record is None:
                raise M6BenchmarkError("migration file is absent from M5 report")
            if (
                item.source_reward_path != benchmark_record.reward_path
                or item.task_id != benchmark_record.task_id
                or item.policy != benchmark_record.policy
            ):
                raise M6BenchmarkError("migration identity does not match M5 report")
            source_trajectory_path = _resolve_under(
                source_runtime_root, item.source_trajectory_path
            )
            source_reward_path = _resolve_under(
                source_runtime_root, item.source_reward_path
            )
            target_trajectory_path = _resolve_under(
                migration_root, item.target_trajectory_path
            )
            target_credit_path = _resolve_under(migration_root, item.target_credit_path)
            expected_hashes = (
                (source_trajectory_path, item.source_trajectory_sha256),
                (source_reward_path, item.source_reward_sha256),
                (target_trajectory_path, item.target_trajectory_sha256),
                (target_credit_path, item.target_credit_sha256),
            )
            for path, expected_hash in expected_hashes:
                if _sha256_file(path) != expected_hash:
                    raise M6BenchmarkError(f"manifest hash mismatch: {path}")
            steps = _read_jsonl_models(
                target_trajectory_path,
                TrajectoryStepV2,
            )
            oracle_credits = _read_jsonl_models(
                target_credit_path,
                ActionCreditRecord,
            )
            if (
                len(steps) != item.action_count
                or len(oracle_credits) != item.credit_count
                or len(steps) != len(oracle_credits)
            ):
                raise M6BenchmarkError("migrated trajectory/credit row counts differ")
            actions = tuple(action for step in steps for action in step.actions)
            if len(actions) != len(steps):
                raise M6BenchmarkError("M6 benchmark requires one action per M5 step")
            action_coordinates = tuple(
                (
                    action.task_id,
                    action.rollout_id,
                    action.stage_id,
                    action.timestep,
                    action.action_id,
                )
                for action in actions
            )
            credit_coordinates = tuple(
                (
                    credit.task_id,
                    credit.rollout_id,
                    credit.stage_id,
                    credit.timestep,
                    credit.action_id,
                )
                for credit in oracle_credits
            )
            if action_coordinates != credit_coordinates:
                raise M6BenchmarkError(
                    "migrated action/credit coordinates do not join in order"
                )
            expected_identity = item.task_id, item.rollout_id
            if any(
                (step.task_id, step.rollout_id) != expected_identity for step in steps
            ) or any(
                (credit.task_id, credit.rollout_id) != expected_identity
                for credit in oracle_credits
            ):
                raise M6BenchmarkError("migrated rows crossed task/rollout identity")
            task_records = records_by_task.get(item.task_id)
            if task_records is None:
                raise M6BenchmarkError("migration task is absent from annotations")
            for step in steps:
                add_response(step.observation, task_records)
            loaded.append((item, benchmark_record, steps, oracle_credits))
        if len(loaded) != len(report.records):
            raise M6BenchmarkError("migration did not cover the complete M5 report")
        if sum(len(steps) for _, _, steps, _ in loaded) != manifest.action_count:
            raise M6BenchmarkError(
                "loaded action count does not match migration manifest"
            )
        for annotation in corpus.manual.records:
            add_response(annotation_texts[annotation.fact_id], (annotation,))

        extractor_version = f"agemem.{profile.name}.v1"
        model_version = f"manual-corpus-{corpus.digest[:16]}-{profile.name}"
        extractor = MockTripleExtractor(
            responses,
            extractor_version=extractor_version,
            model_version=model_version,
            prompt_version="agemem.triple_prompt.v1",
        )

        gold_triples: List[TripleEvaluationRecord] = []
        predicted_triples: List[TripleEvaluationRecord] = []
        for annotation in corpus.manual.records:
            sentence = annotation_texts[annotation.fact_id]
            for triple in annotation.triples:
                gold_triples.append(
                    TripleEvaluationRecord(
                        evidence_id=annotation.annotation_id,
                        subject=triple.subject,
                        category=triple.category,
                        value=triple.value,
                    )
                )
            request = ExtractionRequest(
                task_id=f"hotpot-{annotation.hotpot_id}",
                split_id=annotation.benchmark_split,
                rollout_id=f"m6-eval-{annotation.fact_id}",
                group_id=annotation.annotation_id,
                stage_id=1,
                observation=sentence,
                known_subjects=known_subjects,
                allowed_categories=allowed_categories,
            )
            result = extractor.extract(request)
            if result.diagnostics.quarantined_count:
                raise M6BenchmarkError("manual mock fixture unexpectedly quarantined")
            for candidate in result.candidates:
                predicted_triples.append(
                    TripleEvaluationRecord(
                        evidence_id=annotation.annotation_id,
                        subject=candidate.subject,
                        category=candidate.category,
                        value=candidate.value,
                    )
                )

        tracker = StateTracker(categories=category_specs, known_subjects=known_subjects)
        cache = GroupExtractionCache()
        oracle_ap_rows: List[APEvaluationRecord] = []
        extracted_ap_rows: List[APEvaluationRecord] = []
        oracle_acceptance: List[AcceptanceDecision] = []
        extracted_acceptance: List[AcceptanceDecision] = []
        oracle_rewards: List[RewardActionValue] = []
        extracted_rewards: List[RewardActionValue] = []
        all_extracted_credits: List[ActionCreditRecord] = []
        failures: List[ErrorPropagationRecord] = []
        observation_candidates = action_candidates_count = quarantine_count = 0
        provenance_valid = provenance_total = 0
        status_counts = {"accepted": 0, "rejected": 0, "timed_out": 0}

        target_by_hotpot = {item.hotpot_id: item for item in corpus.targets.tasks}
        role_by_task_digest: Dict[
            str, Dict[str, Literal["relevant", "irrelevant"]]
        ] = {}
        for annotation in corpus.manual.records:
            task_id = f"hotpot-{annotation.hotpot_id}"
            digest = text_digest(annotation_texts[annotation.fact_id])
            role: Literal["relevant", "irrelevant"] = (
                "relevant"
                if annotation.fact_id in corpus.relevant_fact_ids
                else "irrelevant"
            )
            task_roles = role_by_task_digest.setdefault(task_id, {})
            previous = task_roles.setdefault(digest, role)
            if previous != role:
                raise M6BenchmarkError(
                    "one task/evidence digest has conflicting relevance annotations"
                )

        for migration_file, benchmark_record, steps, oracle_credits in loaded:
            task_records = records_by_task[benchmark_record.task_id]
            hotpot_id = benchmark_record.hotpot_id
            target = target_by_hotpot[hotpot_id]
            required = tuple(
                text_digest(annotation_texts[fact_id])
                for fact_id in target.relevant_fact_ids
            )
            grounder = ExtractedAPGrounder(
                relevance_resolver=EvidenceDigestRelevanceResolver(
                    role_by_task_digest[benchmark_record.task_id]
                ),
                required_relevant_digests=required,
            )
            pairs: List[Tuple[TrajectoryStepV2, GroundedAction]] = []
            triple_history: Dict[str, TripleRecord] = {}
            grounded_by_action: Dict[str, GroundedAction] = {}
            retrieved_relevant_digests: set[str] = set()
            seed: Optional[int] = None

            for step in steps:
                action = step.actions[0]
                raw_seed = action.arguments.get("seed")
                if (
                    isinstance(raw_seed, bool)
                    or not isinstance(raw_seed, int)
                    or raw_seed < 0
                ):
                    raise M6BenchmarkError("canonical M5 action seed is invalid")
                if seed is None:
                    seed = raw_seed
                elif seed != raw_seed:
                    raise M6BenchmarkError("one rollout contains multiple seeds")
                request = ExtractionRequest(
                    task_id=action.task_id,
                    split_id=benchmark_record.split,
                    rollout_id=action.rollout_id,
                    group_id=f"{action.task_id}:stage:{action.stage_id}",
                    stage_id=action.stage_id,
                    observation=step.observation,
                    known_subjects=known_subjects,
                    allowed_categories=allowed_categories,
                )
                lookup = cache.get_or_extract(request, extractor)
                observation_records = cache.materialize(
                    request,
                    lookup.result.candidates,
                    _binding(action),
                )
                observation_candidates += len(observation_records)
                quarantine_count += lookup.result.diagnostics.quarantined_count
                memory_delta = derive_memory_delta(step, action)
                action_candidates = _action_specific_candidates(
                    tool_text=memory_delta.tool_text,
                    records=task_records,
                    annotation_texts=annotation_texts,
                    profile=profile,
                    extractor_version=extractor_version,
                    model_version=model_version,
                )
                action_records = tuple(
                    TripleRecord.from_candidate(
                        candidate,
                        _binding(action),
                        source_texts={"tool_result": memory_delta.tool_text},
                    )
                    for candidate in action_candidates
                )
                action_candidates_count += len(action_records)
                combined_by_id = {
                    item.triple_id: item
                    for item in (*observation_records, *action_records)
                }
                combined = tuple(combined_by_id[key] for key in sorted(combined_by_id))
                delta = tracker.apply(action, combined)
                for item in combined:
                    if item.triple_id in set(delta.accepted_triple_ids):
                        triple_history[item.triple_id] = item
                grounded = grounder.ground(
                    step=step,
                    action=action,
                    triples=combined,
                    state_delta=delta,
                    active_state_facts=tracker.active_facts(action.rollout_id),
                    state_triple_history=triple_history,
                    retrieved_relevant_digests_before=retrieved_relevant_digests,
                )
                active_memory_by_id = {
                    item.memory_id: item for item in memory_delta.active_after
                }
                retrieved_relevant_digests.update(
                    active_memory_by_id[memory_id].content_digest
                    for memory_id in memory_delta.returned_memory_ids
                    if active_memory_by_id[memory_id].content_digest in required
                )
                pairs.append((step, grounded))
                grounded_by_action[action.action_id] = grounded
                active_state_by_id = {
                    item.state_fact_id: item
                    for item in tracker.active_facts(action.rollout_id)
                }
                memory_digests_by_id: Dict[str, set[str]] = {}
                for memory in (*step.memory_before, *step.memory_after):
                    memory_digests_by_id.setdefault(memory.memory_id, set()).add(
                        text_digest(memory.content)
                    )
                provenance_total += len(grounded.atomic_propositions)
                for ap in grounded.atomic_propositions:
                    evidence_triples = {
                        triple_id: triple_history.get(triple_id)
                        for triple_id in ap.evidence_triple_ids
                    }
                    evidence_states = {
                        state_id: active_state_by_id.get(state_id)
                        for state_id in ap.evidence_state_fact_ids
                    }
                    expected_action_ids = {
                        action.action_id,
                        *(
                            triple.action_id
                            for triple in evidence_triples.values()
                            if triple is not None
                        ),
                    }
                    ap_triple_ids = set(ap.evidence_triple_ids)
                    ap_evidence_digests = {
                        text_digest(span.text)
                        for triple in evidence_triples.values()
                        if triple is not None
                        for span in triple.evidence
                    }
                    state_provenance_valid = all(
                        state is not None
                        and set(state.evidence_triple_ids).issubset(triple_history)
                        and bool(set(state.evidence_triple_ids) & ap_triple_ids)
                        and set(state.provenance_action_ids)
                        == {
                            triple_history[triple_id].action_id
                            for triple_id in state.evidence_triple_ids
                        }
                        for state in evidence_states.values()
                    )
                    valid = (
                        ap.action_id == action.action_id
                        and all(item is not None for item in evidence_triples.values())
                        and all(item is not None for item in evidence_states.values())
                        and all(
                            memory_id in memory_digests_by_id
                            and bool(
                                memory_digests_by_id[memory_id] & ap_evidence_digests
                            )
                            for memory_id in ap.evidence_memory_ids
                        )
                        and set(ap.evidence_action_ids) == expected_action_ids
                        and state_provenance_valid
                    )
                    if not valid:
                        raise M6BenchmarkError(
                            f"AP provenance chain is invalid: {ap.ap_id}"
                        )
                    provenance_valid += 1

            assert seed is not None
            extracted = ExtractedRewardReplay.from_config(
                self.config.reward_profile,
                extractor_version=extractor_version,
                reward_version=f"agemem.reward.{profile.name}.v1",
            ).replay(pairs, seed=seed)
            repeated = ExtractedRewardReplay.from_config(
                self.config.reward_profile,
                extractor_version=extractor_version,
                reward_version=f"agemem.reward.{profile.name}.v1",
            ).replay(pairs, seed=seed)
            if extracted != repeated or extracted.to_json() != repeated.to_json():
                raise M6BenchmarkError("extracted reward replay is not deterministic")
            extracted_credits = tuple(item.credit for item in extracted.actions)
            if tuple(item.action_id for item in oracle_credits) != tuple(
                item.action_id for item in extracted_credits
            ):
                raise M6BenchmarkError("Oracle/extracted ActionCredit join mismatch")
            for oracle, derived in zip(oracle_credits, extracted_credits):
                if oracle.reward_breakdown.env != derived.reward_breakdown.env:
                    raise M6BenchmarkError(
                        "extraction changed terminal environment reward"
                    )
                oracle_rewards.append(RewardActionValue.from_action_credit(oracle))
                extracted_rewards.append(RewardActionValue.from_action_credit(derived))
                oracle_ap_rows.extend(
                    APEvaluationRecord(action_id=oracle.action_id, proposition=ap)
                    for ap in oracle.atomic_propositions
                )
                extracted_ap_rows.extend(
                    APEvaluationRecord(action_id=derived.action_id, proposition=ap)
                    for ap in derived.atomic_propositions
                )
            terminal_action_id = oracle_credits[-1].action_id
            oracle_accepted = (
                oracle_credits[-1].reward_breakdown.automaton_status == "accepted"
            )
            oracle_acceptance.append(
                AcceptanceDecision(
                    action_id=terminal_action_id, accepted=oracle_accepted
                )
            )
            extracted_acceptance.append(
                AcceptanceDecision(
                    action_id=terminal_action_id, accepted=extracted.accepted
                )
            )
            if extracted.final_status == "running":
                raise M6BenchmarkError(
                    "complete M5 rollout produced non-terminal replay"
                )
            status_counts[extracted.final_status] += 1
            all_extracted_credits.extend(extracted_credits)

            if runtime_output is not None:
                relative = Path(migration_file.target_credit_path)
                if not relative.parts or relative.parts[0] != "action_credits":
                    raise M6BenchmarkError(
                        "migration credit path has an unexpected namespace"
                    )
                output = runtime_output / profile.name / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    "".join(item.to_json_line() + "\n" for item in extracted_credits),
                    encoding="utf-8",
                    newline="\n",
                )
                ap_output = (
                    runtime_output
                    / profile.name
                    / "ap_records"
                    / Path(*relative.parts[1:])
                )
                ap_output.parent.mkdir(parents=True, exist_ok=True)
                ap_output.write_text(
                    "".join(
                        _model_json_line(ap) + "\n"
                        for _, grounded in pairs
                        for ap in grounded.atomic_propositions
                    ),
                    encoding="utf-8",
                    newline="\n",
                )

            first: Optional[ErrorPropagationRecord] = None
            for oracle, derived in zip(oracle_credits, extracted_credits):
                oracle_aps = set(oracle.atomic_propositions)
                extracted_aps = set(derived.atomic_propositions)
                component_names = (
                    "env",
                    "milestone",
                    "violation",
                    "trend",
                    "format",
                    "cost",
                    "total",
                )
                component_errors = {
                    name: getattr(derived.reward_breakdown, name)
                    - getattr(oracle.reward_breakdown, name)
                    for name in component_names
                }
                reward_equal = all(
                    math.isclose(value, 0.0, rel_tol=0.0, abs_tol=0.0)
                    for value in component_errors.values()
                )
                if (
                    oracle.atomic_propositions == derived.atomic_propositions
                    and oracle.transition_ids == derived.transition_ids
                    and oracle.dfa_state_before == derived.dfa_state_before
                    and oracle.dfa_state_after == derived.dfa_state_after
                    and oracle.reward_breakdown.automaton_status
                    == derived.reward_breakdown.automaton_status
                    and oracle.reward_breakdown.newly_rewarded_edges
                    == derived.reward_breakdown.newly_rewarded_edges
                    and oracle.reward_breakdown.violation_edges
                    == derived.reward_breakdown.violation_edges
                    and reward_equal
                ):
                    continue
                grounded = grounded_by_action[oracle.action_id]
                triple_ids = set(grounded.state_delta.accepted_triple_ids) | {
                    triple_id
                    for ap in grounded.atomic_propositions
                    for triple_id in ap.evidence_triple_ids
                }
                state_ids = set(grounded.state_delta.active_state_fact_ids) | {
                    state_id
                    for ap in grounded.atomic_propositions
                    for state_id in ap.evidence_state_fact_ids
                }
                first = ErrorPropagationRecord(
                    profile=profile.name,
                    task_id=oracle.task_id,
                    rollout_id=oracle.rollout_id,
                    action_id=oracle.action_id,
                    timestep=oracle.timestep,
                    oracle_only_aps=tuple(sorted(oracle_aps - extracted_aps)),
                    extracted_only_aps=tuple(sorted(extracted_aps - oracle_aps)),
                    extracted_triple_ids=tuple(sorted(triple_ids)),
                    extracted_state_fact_ids=tuple(sorted(state_ids)),
                    oracle_edges=oracle.transition_ids,
                    extracted_edges=derived.transition_ids,
                    oracle_state_before=oracle.dfa_state_before,
                    extracted_state_before=derived.dfa_state_before,
                    oracle_state_after=oracle.dfa_state_after,
                    extracted_state_after=derived.dfa_state_after,
                    oracle_status=oracle.reward_breakdown.automaton_status,
                    extracted_status=derived.reward_breakdown.automaton_status,
                    env_reward_error=component_errors["env"],
                    milestone_reward_error=component_errors["milestone"],
                    violation_reward_error=component_errors["violation"],
                    trend_reward_error=component_errors["trend"],
                    format_reward_error=component_errors["format"],
                    cost_reward_error=component_errors["cost"],
                    reward_error=component_errors["total"],
                    profile_drop_fact_ids=profile.drop_fact_ids,
                    profile_corrupt_fact_ids=tuple(sorted(profile.corrupt_values)),
                )
                break
            if first is not None:
                failures.append(first)

        triple_metrics = score_triples(gold_triples, predicted_triples)
        ap_metrics = score_aps(oracle_ap_rows, extracted_ap_rows)
        acceptance = score_acceptance(oracle_acceptance, extracted_acceptance)
        reward_metrics = score_reward_propagation(oracle_rewards, extracted_rewards)
        credit_digest = canonical_digest(
            [item.model_dump(mode="json") for item in all_extracted_credits]
        )
        result = ExtractionProfileReport(
            name=profile.name,
            extractor_version=extractor_version,
            rollout_count=len(loaded),
            action_count=len(all_extracted_credits),
            triple_metrics=triple_metrics,
            ap_metrics=ap_metrics,
            acceptance=acceptance,
            reward_propagation=reward_metrics,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            cache_size=cache.size,
            extractor_call_count=extractor.call_count,
            observation_candidates=observation_candidates,
            action_specific_candidates=action_candidates_count,
            quarantined_candidates=quarantine_count,
            provenance_valid_count=provenance_valid,
            provenance_total_count=provenance_total,
            provenance_integrity_rate=(
                1.0 if provenance_total == 0 else provenance_valid / provenance_total
            ),
            accepted_rollouts=status_counts["accepted"],
            rejected_rollouts=status_counts["rejected"],
            timed_out_rollouts=status_counts["timed_out"],
            action_credit_digest=credit_digest,
            failure_count=len(failures),
        )
        return result, tuple(failures)

    def run(
        self,
        *,
        output_root: str | Path,
        docs_path: Optional[str | Path] = None,
        runtime_output: Optional[str | Path] = None,
    ) -> M6BenchmarkArtifacts:
        corpus, validation, texts, m5_report, manifest = self._load_inputs()
        profile_reports: List[ExtractionProfileReport] = []
        failures: List[ErrorPropagationRecord] = []
        runtime = Path(runtime_output).resolve() if runtime_output is not None else None
        for profile in self.config.profiles:
            result, profile_failures = self._run_profile(
                profile=profile,
                corpus=corpus,
                annotation_texts=texts,
                report=m5_report,
                manifest=manifest,
                runtime_output=runtime,
            )
            profile_reports.append(result)
            failures.extend(profile_failures)

        limitations = (
            "human-backed mock is a Triple-extraction upper bound; downstream AP timing is compared independently with the M4 Oracle",
            "relevance and required coverage slots are Oracle-derived evaluation targets",
            "Triple F1 is scored only on the 34 fully annotated sentences",
            "controlled drop/corrupt errors are synthetic and are not an empirical LLM error distribution",
            "M5 rule/error trajectories contain no token IDs, token logprobs, or policy version",
            "missing-support answer correctness is fail-closed from terminal env reward",
        )
        digest_payload: Dict[str, object] = {
            "schema_version": M6_BENCHMARK_REPORT_SCHEMA_VERSION,
            "benchmark_name": "m6-hotpotqa-extracted-ap-smoke",
            "seed": self.config.seed,
            "config_digest": self.config.digest,
            "annotation_validation": validation.model_dump(mode="json"),
            "m5_report_digest": m5_report.digest,
            "migration_manifest_digest": manifest.digest,
            "canonical_rollout_count": manifest.canonical_rollout_count,
            "canonical_action_count": manifest.action_count,
            "llm_evaluation": self.config.llm_evaluation,
            "real_llm_call_count": 0,
            "human_oracle_semantic_target_used": True,
            "profiles": [item.model_dump(mode="json") for item in profile_reports],
            "failures": [item.model_dump(mode="json") for item in failures],
            "limitations": limitations,
        }
        report = M6ExtractionBenchmarkReport(
            seed=self.config.seed,
            config_digest=self.config.digest,
            annotation_validation=validation,
            m5_report_digest=m5_report.digest,
            migration_manifest_digest=manifest.digest,
            canonical_rollout_count=manifest.canonical_rollout_count,
            canonical_action_count=manifest.action_count,
            llm_evaluation=self.config.llm_evaluation,
            profiles=tuple(profile_reports),
            failures=tuple(failures),
            limitations=limitations,
            digest=canonical_digest(digest_payload),
        )
        output = Path(output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "extraction_benchmark.json"
        failures_path = output / "error_propagation.jsonl"
        markdown_path = output / "extraction_benchmark.md"
        report_path.write_text(report.to_json() + "\n", encoding="utf-8", newline="\n")
        failures_path.write_text(
            "".join(item.to_json_line() + "\n" for item in report.failures),
            encoding="utf-8",
            newline="\n",
        )
        markdown = _markdown(report)
        markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
        if docs_path is not None:
            docs = Path(docs_path).resolve()
            docs.parent.mkdir(parents=True, exist_ok=True)
            docs.write_text(markdown, encoding="utf-8", newline="\n")
        return M6BenchmarkArtifacts(
            report_path=report_path,
            failures_path=failures_path,
            markdown_path=markdown_path,
            report=report,
        )


def run_default_m6_benchmark(
    *,
    data_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    output_root: Optional[str | Path] = None,
    docs_path: Optional[str | Path] = None,
    runtime_output: Optional[str | Path] = None,
) -> M6BenchmarkArtifacts:
    """Run the checked-in benchmark without constructing an LLM client."""

    root = repository_root()
    config = M6BenchmarkConfig.from_json(config_path or default_benchmark_config_path())
    adapter = HotpotQADataAdapter(data_path)
    return M6ExtractionBenchmark(
        config=config,
        adapter=adapter,
        repository=root,
    ).run(
        output_root=output_root or root / "artifacts" / "m6_extraction_benchmark",
        docs_path=docs_path or root / "docs" / "m6_extraction_benchmark.md",
        runtime_output=runtime_output or root / "runs" / "m6_extraction_benchmark",
    )


__all__ = [
    "M6_BENCHMARK_CONFIG_SCHEMA_VERSION",
    "M6_BENCHMARK_REPORT_SCHEMA_VERSION",
    "ErrorPropagationRecord",
    "ExtractionProfileConfig",
    "ExtractionProfileReport",
    "M6BenchmarkArtifacts",
    "M6BenchmarkConfig",
    "M6BenchmarkError",
    "M6ExtractionBenchmark",
    "M6ExtractionBenchmarkReport",
    "default_benchmark_config_path",
    "run_default_m6_benchmark",
]
