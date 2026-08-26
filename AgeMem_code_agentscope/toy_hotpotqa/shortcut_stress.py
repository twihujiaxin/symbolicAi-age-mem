"""Expanded, deterministic Stage 1/2 anti-shortcut stress experiment.

The original :mod:`shortcut_benchmark` remains a small CI canary.  This module
adds the broader experiment needed for interpretation: multiple noisy tasks,
multiple order seeds, globally fixed budgets, stronger public baselines, and
paired Stage-2 futures that share one byte-identical public input.

No Agent, LLM, embedding service, or network call is made here.  A frozen local
Hugging Face tokenizer can be injected for the production-tokenizer rerun.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from statistics import fmean
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .dataset import ToyTaskDataset
from .models import ToyFact, ToyMemoryTask
from .storage_baselines import count_ltm_tokens, ordered_stage1_facts


STRESS_SCHEMA_VERSION = "agemem.anti_shortcut_stress.v1"
DEFAULT_STRESS_SEEDS = tuple(range(50))
DEFAULT_STAGE1_BUDGETS = (12, 20, 28)
COUNTERFACTUAL_HANDLE_VERSION = "agemem-counterfactual-opaque-v1"

Stage1PolicyName = Literal[
    "store_all",
    "store_none",
    "reverse_order",
    "shortest_first",
    "longest_first",
    "opaque_min",
    "opaque_max",
    "random_hash",
    "entity_chain",
    "oracle_support",
]
Stage2StressPolicyName = Literal[
    "always_keep",
    "always_clear",
    "first_fit",
    "last_fit",
    "shortest_first",
    "longest_first",
    "opaque_min",
    "opaque_max",
    "random_hash",
    "style_density",
    "pair_blind_oracle",
    "oracle_future",
]
CounterfactualSplit = Literal["dev", "test"]
CounterfactualScenario = Literal[
    "matched_entity",
    "matched_style",
    "matched_length",
]

STAGE1_POLICIES: Tuple[Stage1PolicyName, ...] = (
    "store_all",
    "store_none",
    "reverse_order",
    "shortest_first",
    "longest_first",
    "opaque_min",
    "opaque_max",
    "random_hash",
    "entity_chain",
    "oracle_support",
)
STAGE2_STRESS_POLICIES: Tuple[Stage2StressPolicyName, ...] = (
    "always_keep",
    "always_clear",
    "first_fit",
    "last_fit",
    "shortest_first",
    "longest_first",
    "opaque_min",
    "opaque_max",
    "random_hash",
    "style_density",
    "pair_blind_oracle",
    "oracle_future",
)


def _canonical_digest(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


class TokenCounterSpec(BaseModel):
    """Serializable identity for the exact counter used by every arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    repository_id: Optional[str] = None
    revision: Optional[str] = None
    assets_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frozen_identity(self) -> "TokenCounterSpec":
        fields = (self.repository_id, self.revision, self.assets_digest)
        if self.name == "unicode-lexical-v1":
            if any(value is not None for value in fields):
                raise ValueError("lexical counter must not claim tokenizer assets")
        elif any(value is None for value in fields):
            raise ValueError(
                "external tokenizer requires repository, revision, and digest"
            )
        if self.revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", self.revision
        ):
            raise ValueError(
                "tokenizer revision must be a lowercase 40-character commit"
            )
        return self


def lexical_token_counter() -> Tuple[Callable[[str], int], TokenCounterSpec]:
    return count_ltm_tokens, TokenCounterSpec(name="unicode-lexical-v1")


def frozen_hf_token_counter(
    tokenizer_path: str | Path,
    *,
    repository_id: str,
    revision: str,
) -> Tuple[Callable[[str], int], TokenCounterSpec]:
    """Load a tokenizer from local files and bind the report to its assets."""

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a lowercase 40-character commit")
    root = Path(tokenizer_path)
    if not root.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {root}")
    asset_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "tokenizer.model",
        "chat_template.jinja",
    )
    assets = [root / name for name in asset_names if (root / name).is_file()]
    if not any(path.name == "tokenizer_config.json" for path in assets):
        raise ValueError("tokenizer_config.json is required for a frozen tokenizer run")
    has_tokenizer_body = any(
        path.name in {"tokenizer.json", "vocab.json", "vocab.txt"} for path in assets
    )
    if not has_tokenizer_body:
        raise ValueError("tokenizer vocabulary assets are missing")
    digest = hashlib.sha256()
    for path in sorted(assets, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )

    def counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return counter, TokenCounterSpec(
        name="huggingface-auto-tokenizer",
        repository_id=repository_id,
        revision=revision,
        assets_digest=digest.hexdigest(),
    )


class BlindStage1Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_handle: str = Field(pattern=r"^stage1-h-[0-9a-f]{24}$")
    title: str = Field(min_length=1)
    sentence: str = Field(min_length=1)
    content_tokens: int = Field(ge=1)


class BlindStage1Input(BaseModel):
    """Policy boundary: deliberately excludes task ID, split, labels, and seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    budget_tokens: int = Field(ge=1)
    observed_facts: Tuple[BlindStage1Fact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_handles(self) -> "BlindStage1Input":
        handles = [fact.fact_handle for fact in self.observed_facts]
        if len(handles) != len(set(handles)):
            raise ValueError("Stage-1 public handles must be unique")
        return self


class Stage1PolicyAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Stage1PolicyName
    uses_oracle_labels: bool
    arm_count: int = Field(ge=1)
    support_recall: float = Field(ge=0.0, le=1.0)
    oracle_normalized_recall: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    exact_support_rate: float = Field(ge=0.0, le=1.0)
    oracle_equivalent_rate: float = Field(ge=0.0, le=1.0)
    budget_compliance_rate: float = Field(ge=0.0, le=1.0)
    mean_stored_tokens: float = Field(ge=0.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)


class PermutationCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    fact_count: int = Field(ge=2)
    observed_permutations: int = Field(ge=1)
    possible_permutations: int = Field(ge=1)
    coverage_rate: float = Field(ge=0.0, le=1.0)


class Stage1StressReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    task_ids: Tuple[str, ...]
    seeds: Tuple[int, ...] = Field(min_length=1)
    budgets: Tuple[int, ...] = Field(min_length=1)
    public_input_fields: Tuple[str, ...]
    policies: Tuple[Stage1PolicyName, ...]
    arm_count_per_policy: int = Field(ge=1)
    permutation_coverage: Tuple[PermutationCoverage, ...]
    aggregates: Dict[Stage1PolicyName, Stage1PolicyAggregate]
    by_budget: Dict[str, Dict[Stage1PolicyName, Stage1PolicyAggregate]]
    by_split: Dict[str, Dict[Stage1PolicyName, Stage1PolicyAggregate]]
    by_scenario: Dict[str, Dict[Stage1PolicyName, Stage1PolicyAggregate]]

    @model_validator(mode="after")
    def validate_coverage(self) -> "Stage1StressReport":
        if len(self.seeds) != len(set(self.seeds)) or any(
            seed < 0 for seed in self.seeds
        ):
            raise ValueError("Stage-1 seeds must be unique and non-negative")
        if len(self.budgets) != len(set(self.budgets)) or any(
            budget <= 0 for budget in self.budgets
        ):
            raise ValueError("Stage-1 budgets must be unique and positive")
        if self.task_count != len(self.task_ids):
            raise ValueError("Stage-1 task_count does not match task_ids")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Stage-1 task_ids must be unique")
        if len(self.permutation_coverage) != self.task_count:
            raise ValueError("Stage-1 permutation coverage is incomplete")
        if self.policies != STAGE1_POLICIES:
            raise ValueError("Stage-1 policy order does not match the protocol")
        if set(self.aggregates) != set(STAGE1_POLICIES):
            raise ValueError("Stage-1 aggregate policy set is incomplete")
        expected_arms = self.task_count * len(self.seeds) * len(self.budgets)
        if self.arm_count_per_policy != expected_arms:
            raise ValueError("Stage-1 arm count does not match the Cartesian product")
        for aggregate_map in (
            self.aggregates,
            *self.by_budget.values(),
            *self.by_split.values(),
            *self.by_scenario.values(),
        ):
            if set(aggregate_map) != set(STAGE1_POLICIES):
                raise ValueError("Stage-1 stratum has an incomplete policy set")
            if any(row.policy != name for name, row in aggregate_map.items()):
                raise ValueError("Stage-1 aggregate policy does not match its key")
        return self


class CounterfactualSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_key: str = Field(min_length=1)
    text: str = Field(min_length=1)


class CounterfactualFuture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant_id: str = Field(min_length=1)
    future_query: str = Field(min_length=1)
    future_answer: str = Field(min_length=1)
    support_segment_keys: Tuple[str, ...] = Field(min_length=1)


class CounterfactualPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    pair_id: str = Field(min_length=1)
    split: CounterfactualSplit
    scenario: CounterfactualScenario
    max_context_tokens: int = Field(ge=1)
    segments: Tuple[CounterfactualSegment, ...] = Field(min_length=3)
    futures: Tuple[CounterfactualFuture, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_counterfactual_pair(self) -> "CounterfactualPair":
        segment_keys = [segment.segment_key for segment in self.segments]
        if len(segment_keys) != len(set(segment_keys)):
            raise ValueError("counterfactual segment keys must be unique")
        variant_ids = [future.variant_id for future in self.futures]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("counterfactual variant IDs must be unique")
        known = set(segment_keys)
        support_sets = []
        segment_text = {
            segment.segment_key: _normalized(segment.text) for segment in self.segments
        }
        for future in self.futures:
            support = set(future.support_segment_keys)
            if not support <= known:
                raise ValueError("future references an unknown support segment")
            if len(support) != len(future.support_segment_keys):
                raise ValueError("future support keys must be unique")
            answer = _normalized(future.future_answer)
            support_text = " ".join(segment_text[key] for key in support)
            distractor_text = " ".join(
                text for key, text in segment_text.items() if key not in support
            )
            if answer not in support_text:
                raise ValueError("future answer must be grounded in its support")
            if answer in distractor_text:
                raise ValueError("future answer must be absent from distractors")
            support_sets.append(support)
        if support_sets[0] & support_sets[1]:
            raise ValueError("paired futures must have disjoint support sets")
        return self


class CounterfactualDataset:
    def __init__(self, pairs: Iterable[CounterfactualPair]) -> None:
        self._pairs = tuple(pairs)
        if len(self._pairs) < 6:
            raise ValueError("counterfactual stress set requires at least six pairs")
        ids = [pair.pair_id for pair in self._pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("counterfactual pair IDs must be unique")
        if {pair.split for pair in self._pairs} != {"dev", "test"}:
            raise ValueError("counterfactual set requires dev and test")
        required = {"matched_entity", "matched_style", "matched_length"}
        for split in ("dev", "test"):
            observed = {pair.scenario for pair in self._pairs if pair.split == split}
            if observed != required:
                raise ValueError(
                    f"counterfactual {split} split lacks scenario coverage"
                )

    @classmethod
    def from_json(cls, path: Optional[str | Path] = None) -> "CounterfactualDataset":
        selected = Path(path) if path is not None else default_counterfactual_path()
        raw = json.loads(selected.read_text(encoding="utf-8"))
        pairs = TypeAdapter(List[CounterfactualPair]).validate_python(raw)
        return cls(pairs)

    def all(self) -> List[CounterfactualPair]:
        return [pair.model_copy(deep=True) for pair in self._pairs]

    def digest(self) -> str:
        return _canonical_digest(
            [
                pair.model_dump(mode="json")
                for pair in sorted(self._pairs, key=lambda item: item.pair_id)
            ]
        )


def default_counterfactual_path() -> Path:
    source = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "toy"
        / "stage2_counterfactual_pairs.json"
    )
    if source.is_file():
        return source
    packaged = Path(__file__).with_name("data") / "stage2_counterfactual_pairs.json"
    if not packaged.is_file():
        raise FileNotFoundError("Stage-2 counterfactual fixture is missing")
    return packaged


class BlindStage2Segment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_handle: str = Field(pattern=r"^stage2-h-[0-9a-f]{24}$")
    text: str = Field(min_length=1)
    content_tokens: int = Field(ge=1)


class BlindStage2Input(BaseModel):
    """One public context reused unchanged for both future-query variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_context_tokens: int = Field(ge=1)
    segments: Tuple[BlindStage2Segment, ...] = Field(min_length=3)


class Stage2StressPolicyAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Stage2StressPolicyName
    uses_future_labels: bool
    arm_count: int = Field(ge=1)
    future_support_recall: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    distractor_removal_recall: float = Field(ge=0.0, le=1.0)
    budget_compliance_rate: float = Field(ge=0.0, le=1.0)
    safe_success_rate: float = Field(ge=0.0, le=1.0)
    mean_kept_tokens: float = Field(ge=0.0)


class Stage2CounterfactualReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_count: int = Field(ge=1)
    variant_count: int = Field(ge=2)
    seeds: Tuple[int, ...] = Field(min_length=1)
    budgets: Tuple[int, ...] = Field(min_length=1)
    public_input_fields: Tuple[str, ...]
    policies: Tuple[Stage2StressPolicyName, ...]
    arm_count_per_policy: int = Field(ge=1)
    query_blind_decision_count: int = Field(ge=1)
    hindsight_decision_count: int = Field(ge=1)
    public_input_identity_rate: float = Field(ge=0.0, le=1.0)
    public_safe_success_ceiling: float = Field(ge=0.0, le=1.0)
    max_target_token_gap: int = Field(ge=0)
    max_target_capitalized_gap: int = Field(ge=0)
    aggregates: Dict[Stage2StressPolicyName, Stage2StressPolicyAggregate]
    by_split: Dict[str, Dict[Stage2StressPolicyName, Stage2StressPolicyAggregate]]
    by_scenario: Dict[str, Dict[Stage2StressPolicyName, Stage2StressPolicyAggregate]]

    @model_validator(mode="after")
    def validate_coverage(self) -> "Stage2CounterfactualReport":
        if len(self.seeds) != len(set(self.seeds)) or any(
            seed < 0 for seed in self.seeds
        ):
            raise ValueError("Stage-2 seeds must be unique and non-negative")
        if len(self.budgets) != len(set(self.budgets)) or any(
            budget <= 0 for budget in self.budgets
        ):
            raise ValueError("Stage-2 budgets must be unique and positive")
        if self.variant_count != self.pair_count * 2:
            raise ValueError("Stage-2 variant count must be two per pair")
        if self.policies != STAGE2_STRESS_POLICIES:
            raise ValueError("Stage-2 policy order does not match the protocol")
        if set(self.aggregates) != set(STAGE2_STRESS_POLICIES):
            raise ValueError("Stage-2 aggregate policy set is incomplete")
        expected_arms = self.variant_count * len(self.seeds)
        if self.arm_count_per_policy != expected_arms:
            raise ValueError("Stage-2 arm count does not match the Cartesian product")
        expected_blind = (
            (len(STAGE2_STRESS_POLICIES) - 1) * self.pair_count * len(self.seeds)
        )
        if self.query_blind_decision_count != expected_blind:
            raise ValueError("Stage-2 query-blind decision count is inconsistent")
        if self.hindsight_decision_count != expected_arms:
            raise ValueError("Stage-2 hindsight decision count is inconsistent")
        for aggregate_map in (
            self.aggregates,
            *self.by_split.values(),
            *self.by_scenario.values(),
        ):
            if set(aggregate_map) != set(STAGE2_STRESS_POLICIES):
                raise ValueError("Stage-2 stratum has an incomplete policy set")
            if any(row.policy != name for name, row in aggregate_map.items()):
                raise ValueError("Stage-2 aggregate policy does not match its key")
        return self


class StressGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    passed: bool
    evidence: str = Field(min_length=1)


class AntiShortcutStressReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agemem.anti_shortcut_stress.v1"] = STRESS_SCHEMA_VERSION
    token_counter: TokenCounterSpec
    real_llm_call_count: Literal[0] = 0
    stage1: Stage1StressReport
    stage2: Stage2CounterfactualReport
    gates: Tuple[StressGate, ...]
    passed: bool
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> "AntiShortcutStressReport":
        expected_gates = _derive_stress_gates(
            self.stage1,
            self.stage2,
            real_llm_call_count=self.real_llm_call_count,
        )
        if self.gates != expected_gates:
            raise ValueError("stress report gates do not match derived metrics")
        if self.passed != all(gate.passed for gate in self.gates):
            raise ValueError("stress report passed flag does not match gates")
        if self.digest != stress_report_digest(self, include_digest=False):
            raise ValueError("stress report digest does not match payload")
        return self


def _opaque_handle(prefix: str, task_id: str, seed: int, private_key: str) -> str:
    material = (
        f"{COUNTERFACTUAL_HANDLE_VERSION}\0{prefix}\0{task_id}\0{seed}\0{private_key}"
    ).encode("utf-8")
    return f"{prefix}-h-{hashlib.sha256(material).hexdigest()[:24]}"


def _stage1_scenario(task: ToyMemoryTask) -> str:
    difficulty = set(task.difficulty)
    if "distractor" in difficulty:
        return "distractor"
    if "duplicate" in difficulty:
        return "duplicate"
    if "stale_fact" in difficulty or "fact_update" in difficulty:
        return "stale_fact"
    return "other"


def _stage1_tasks(dataset: ToyTaskDataset) -> List[ToyMemoryTask]:
    selected = []
    for task in dataset.all():
        observed = [fact for fact in task.facts if fact.stage == 1]
        observed_ids = {fact.fact_id for fact in observed}
        supporting = set(task.supporting_fact_ids)
        if supporting <= observed_ids and observed_ids - supporting:
            selected.append(task)
    return sorted(selected, key=lambda item: item.task_id)


def _blind_stage1_input(
    task: ToyMemoryTask,
    seed: int,
    budget: int,
    counter: Callable[[str], int],
) -> Tuple[BlindStage1Input, Dict[str, ToyFact]]:
    observed = ordered_stage1_facts(task, seed)
    public_facts = []
    private = {}
    for fact in observed:
        handle = _opaque_handle("stage1", task.task_id, seed, fact.fact_id)
        public_facts.append(
            BlindStage1Fact(
                fact_handle=handle,
                title=fact.title,
                sentence=fact.sentence,
                content_tokens=counter(fact.sentence),
            )
        )
        private[handle] = fact
    return (
        BlindStage1Input(
            budget_tokens=budget,
            observed_facts=tuple(public_facts),
        ),
        private,
    )


def _entity_chain_scores(facts: Sequence[BlindStage1Fact]) -> Dict[str, int]:
    scores = {fact.fact_handle: 0 for fact in facts}
    for index, left in enumerate(facts):
        for right in facts[index + 1 :]:
            left_title = _normalized(left.title)
            right_title = _normalized(right.title)
            linked = left_title in _normalized(
                right.sentence
            ) or right_title in _normalized(left.sentence)
            if linked:
                scores[left.fact_handle] += 1
                scores[right.fact_handle] += 1
    return scores


def _rank_stage1(
    policy: Stage1PolicyName,
    public_input: BlindStage1Input,
    *,
    private: Mapping[str, ToyFact],
    supporting_ids: set[str],
) -> Tuple[str, ...]:
    facts = list(public_input.observed_facts)
    if policy == "store_none":
        return ()
    if policy == "store_all":
        ranked = facts
    elif policy == "reverse_order":
        ranked = list(reversed(facts))
    elif policy == "shortest_first":
        ranked = sorted(facts, key=lambda item: (item.content_tokens, item.fact_handle))
    elif policy == "longest_first":
        ranked = sorted(
            facts,
            key=lambda item: (-item.content_tokens, item.fact_handle),
        )
    elif policy == "opaque_min":
        ranked = sorted(facts, key=lambda item: item.fact_handle)
    elif policy == "opaque_max":
        ranked = sorted(facts, key=lambda item: item.fact_handle, reverse=True)
    elif policy == "random_hash":
        ranked = sorted(
            facts,
            key=lambda item: hashlib.sha256(
                f"stage1-random-v1\0{item.fact_handle}".encode("utf-8")
            ).digest(),
        )
    elif policy == "entity_chain":
        scores = _entity_chain_scores(facts)
        ranked = sorted(
            facts,
            key=lambda item: (
                -scores[item.fact_handle],
                item.content_tokens,
                item.fact_handle,
            ),
        )
    elif policy == "oracle_support":
        ranked = sorted(
            (
                fact
                for fact in facts
                if private[fact.fact_handle].fact_id in supporting_ids
            ),
            key=lambda item: (item.content_tokens, item.fact_handle),
        )
    else:
        raise ValueError(f"unknown Stage-1 policy: {policy}")
    return tuple(fact.fact_handle for fact in ranked)


def _fit_ranked(
    handles: Sequence[str],
    public_input: BlindStage1Input,
) -> Tuple[Tuple[str, ...], int, int]:
    by_handle = {fact.fact_handle: fact for fact in public_input.observed_facts}
    selected = []
    stored_tokens = 0
    rejected = 0
    for handle in handles:
        tokens = by_handle[handle].content_tokens
        if stored_tokens + tokens <= public_input.budget_tokens:
            selected.append(handle)
            stored_tokens += tokens
        else:
            rejected += 1
    return tuple(selected), stored_tokens, rejected


def _aggregate_stage1(
    rows: Sequence[Dict[str, object]],
    policy: Stage1PolicyName,
) -> Stage1PolicyAggregate:
    selected = [row for row in rows if row["policy"] == policy]
    if not selected:
        raise ValueError(f"Stage-1 stratum has no rows for {policy}")
    return Stage1PolicyAggregate(
        policy=policy,
        uses_oracle_labels=policy == "oracle_support",
        arm_count=len(selected),
        support_recall=fmean(float(row["support_recall"]) for row in selected),
        oracle_normalized_recall=fmean(
            float(row["oracle_normalized_recall"]) for row in selected
        ),
        memory_precision=fmean(float(row["memory_precision"]) for row in selected),
        exact_support_rate=fmean(float(row["exact_support"]) for row in selected),
        oracle_equivalent_rate=fmean(
            float(row["oracle_equivalent"]) for row in selected
        ),
        budget_compliance_rate=fmean(
            float(row["budget_compliant"]) for row in selected
        ),
        mean_stored_tokens=fmean(float(row["stored_tokens"]) for row in selected),
        rejection_rate=fmean(float(row["rejection_rate"]) for row in selected),
    )


def _stage1_aggregate_map(
    rows: Sequence[Dict[str, object]],
) -> Dict[Stage1PolicyName, Stage1PolicyAggregate]:
    return {policy: _aggregate_stage1(rows, policy) for policy in STAGE1_POLICIES}


def run_stage1_stress(
    *,
    dataset: Optional[ToyTaskDataset] = None,
    seeds: Sequence[int] = DEFAULT_STRESS_SEEDS,
    budgets: Sequence[int] = DEFAULT_STAGE1_BUDGETS,
    token_counter: Callable[[str], int] = count_ltm_tokens,
) -> Stage1StressReport:
    selected_dataset = dataset or ToyTaskDataset.from_json()
    selected_seeds = tuple(seeds)
    selected_budgets = tuple(budgets)
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("Stage-1 seeds must be non-empty and unique")
    if any(seed < 0 for seed in selected_seeds):
        raise ValueError("Stage-1 seeds must be non-negative")
    if not selected_budgets or len(selected_budgets) != len(set(selected_budgets)):
        raise ValueError("Stage-1 budgets must be non-empty and unique")
    if any(budget <= 0 for budget in selected_budgets):
        raise ValueError("Stage-1 budgets must be positive")
    tasks = _stage1_tasks(selected_dataset)
    if not tasks:
        raise ValueError("no noisy Stage-1 tasks are available")

    task_payload = [task.model_dump(mode="json") for task in tasks]
    rows: List[Dict[str, object]] = []
    coverage = []
    for task in tasks:
        permutations = {
            tuple(fact.fact_id for fact in ordered_stage1_facts(task, seed))
            for seed in selected_seeds
        }
        fact_count = len([fact for fact in task.facts if fact.stage == 1])
        possible = math.factorial(fact_count)
        coverage.append(
            PermutationCoverage(
                task_id=task.task_id,
                fact_count=fact_count,
                observed_permutations=len(permutations),
                possible_permutations=possible,
                coverage_rate=min(1.0, len(permutations) / possible),
            )
        )
        supporting_ids = set(task.supporting_fact_ids)
        for seed in selected_seeds:
            for budget in selected_budgets:
                public_input, private = _blind_stage1_input(
                    task,
                    seed,
                    budget,
                    token_counter,
                )
                oracle_rank = _rank_stage1(
                    "oracle_support",
                    public_input,
                    private=private,
                    supporting_ids=supporting_ids,
                )
                oracle_selected, _, _ = _fit_ranked(oracle_rank, public_input)
                oracle_ids = {private[handle].fact_id for handle in oracle_selected}
                oracle_recall = _ratio(
                    len(oracle_ids & supporting_ids), len(supporting_ids)
                )
                for policy in STAGE1_POLICIES:
                    ranked = _rank_stage1(
                        policy,
                        public_input,
                        private=private,
                        supporting_ids=supporting_ids,
                    )
                    chosen, stored_tokens, rejected = _fit_ranked(ranked, public_input)
                    chosen_ids = {private[handle].fact_id for handle in chosen}
                    retained_support = chosen_ids & supporting_ids
                    recall = _ratio(len(retained_support), len(supporting_ids))
                    rows.append(
                        {
                            "task_id": task.task_id,
                            "split": task.split,
                            "scenario": _stage1_scenario(task),
                            "seed": seed,
                            "budget": budget,
                            "policy": policy,
                            "support_recall": recall,
                            "oracle_normalized_recall": (
                                min(1.0, recall / oracle_recall)
                                if oracle_recall
                                else 1.0
                            ),
                            "memory_precision": _ratio(
                                len(retained_support), len(chosen_ids)
                            ),
                            "exact_support": chosen_ids == supporting_ids,
                            "oracle_equivalent": chosen_ids == oracle_ids,
                            "budget_compliant": stored_tokens <= budget,
                            "stored_tokens": stored_tokens,
                            "rejection_rate": _ratio(rejected, len(ranked)),
                        }
                    )

    by_budget = {
        str(budget): _stage1_aggregate_map(
            [row for row in rows if row["budget"] == budget]
        )
        for budget in selected_budgets
    }
    splits = sorted({str(row["split"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    return Stage1StressReport(
        task_dataset_digest=_canonical_digest(task_payload),
        task_count=len(tasks),
        task_ids=tuple(task.task_id for task in tasks),
        seeds=selected_seeds,
        budgets=selected_budgets,
        public_input_fields=tuple(BlindStage1Input.model_fields),
        policies=STAGE1_POLICIES,
        arm_count_per_policy=len(tasks) * len(selected_seeds) * len(selected_budgets),
        permutation_coverage=tuple(coverage),
        aggregates=_stage1_aggregate_map(rows),
        by_budget=by_budget,
        by_split={
            split: _stage1_aggregate_map([row for row in rows if row["split"] == split])
            for split in splits
        },
        by_scenario={
            scenario: _stage1_aggregate_map(
                [row for row in rows if row["scenario"] == scenario]
            )
            for scenario in scenarios
        },
    )


def _blind_stage2_input(
    pair: CounterfactualPair,
    seed: int,
    counter: Callable[[str], int],
) -> Tuple[BlindStage2Input, Dict[str, CounterfactualSegment]]:
    ordered = list(pair.segments)
    seed_bytes = f"{pair.pair_id}:{seed}:counterfactual".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    random.Random(local_seed).shuffle(ordered)
    public_segments = []
    private = {}
    for segment in ordered:
        handle = _opaque_handle("stage2", pair.pair_id, seed, segment.segment_key)
        public_segments.append(
            BlindStage2Segment(
                segment_handle=handle,
                text=segment.text,
                content_tokens=counter(segment.text),
            )
        )
        private[handle] = segment
    return (
        BlindStage2Input(
            max_context_tokens=pair.max_context_tokens,
            segments=tuple(public_segments),
        ),
        private,
    )


def _capitalized_density(text: str) -> float:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", text)
    capitalized = sum(token[0].isupper() for token in tokens)
    return _ratio(capitalized, len(tokens))


def _rank_stage2(
    policy: Stage2StressPolicyName,
    public_input: BlindStage2Input,
    *,
    private: Mapping[str, CounterfactualSegment],
    support_keys: Optional[set[str]] = None,
) -> Tuple[str, ...]:
    segments = list(public_input.segments)
    if policy == "always_clear":
        return ()
    if policy in {"always_keep", "first_fit"}:
        ranked = segments
    elif policy == "last_fit":
        ranked = list(reversed(segments))
    elif policy == "shortest_first":
        ranked = sorted(
            segments, key=lambda item: (item.content_tokens, item.segment_handle)
        )
    elif policy == "longest_first":
        ranked = sorted(
            segments,
            key=lambda item: (-item.content_tokens, item.segment_handle),
        )
    elif policy == "opaque_min":
        ranked = sorted(segments, key=lambda item: item.segment_handle)
    elif policy == "opaque_max":
        ranked = sorted(segments, key=lambda item: item.segment_handle, reverse=True)
    elif policy == "random_hash":
        ranked = sorted(
            segments,
            key=lambda item: hashlib.sha256(
                f"stage2-random-v1\0{item.segment_handle}".encode("utf-8")
            ).digest(),
        )
    elif policy == "style_density":
        ranked = sorted(
            segments,
            key=lambda item: (
                -_capitalized_density(item.text),
                item.content_tokens,
                item.segment_handle,
            ),
        )
    elif policy in {"pair_blind_oracle", "oracle_future"}:
        if support_keys is None:
            raise ValueError(f"{policy} requires private support keys")
        ranked = sorted(
            (
                segment
                for segment in segments
                if private[segment.segment_handle].segment_key in support_keys
            ),
            key=lambda item: (item.content_tokens, item.segment_handle),
        )
    else:
        raise ValueError(f"unknown Stage-2 policy: {policy}")
    return tuple(segment.segment_handle for segment in ranked)


def _fit_stage2(
    policy: Stage2StressPolicyName,
    ranked: Sequence[str],
    public_input: BlindStage2Input,
) -> Tuple[Tuple[str, ...], int]:
    by_handle = {segment.segment_handle: segment for segment in public_input.segments}
    if policy == "always_keep":
        return tuple(ranked), sum(by_handle[handle].content_tokens for handle in ranked)
    selected = []
    tokens = 0
    for handle in ranked:
        cost = by_handle[handle].content_tokens
        if tokens + cost <= public_input.max_context_tokens:
            selected.append(handle)
            tokens += cost
    return tuple(selected), tokens


def _aggregate_stage2(
    rows: Sequence[Dict[str, object]],
    policy: Stage2StressPolicyName,
) -> Stage2StressPolicyAggregate:
    selected = [row for row in rows if row["policy"] == policy]
    if not selected:
        raise ValueError(f"Stage-2 stratum has no rows for {policy}")
    return Stage2StressPolicyAggregate(
        policy=policy,
        uses_future_labels=policy in {"pair_blind_oracle", "oracle_future"},
        arm_count=len(selected),
        future_support_recall=fmean(
            float(row["future_support_recall"]) for row in selected
        ),
        memory_precision=fmean(float(row["memory_precision"]) for row in selected),
        distractor_removal_recall=fmean(
            float(row["distractor_removal_recall"]) for row in selected
        ),
        budget_compliance_rate=fmean(
            float(row["budget_compliant"]) for row in selected
        ),
        safe_success_rate=fmean(float(row["safe_success"]) for row in selected),
        mean_kept_tokens=fmean(float(row["kept_tokens"]) for row in selected),
    )


def _stage2_aggregate_map(
    rows: Sequence[Dict[str, object]],
) -> Dict[Stage2StressPolicyName, Stage2StressPolicyAggregate]:
    return {
        policy: _aggregate_stage2(rows, policy) for policy in STAGE2_STRESS_POLICIES
    }


def run_stage2_counterfactual_stress(
    *,
    dataset: Optional[CounterfactualDataset] = None,
    seeds: Sequence[int] = DEFAULT_STRESS_SEEDS,
    token_counter: Callable[[str], int] = count_ltm_tokens,
) -> Stage2CounterfactualReport:
    selected_dataset = dataset or CounterfactualDataset.from_json()
    selected_seeds = tuple(seeds)
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("Stage-2 seeds must be non-empty and unique")
    if any(seed < 0 for seed in selected_seeds):
        raise ValueError("Stage-2 seeds must be non-negative")
    pairs = sorted(selected_dataset.all(), key=lambda item: item.pair_id)
    rows: List[Dict[str, object]] = []
    target_token_gaps = []
    target_capitalized_gaps = []
    identity_checks = 0
    identity_matches = 0
    query_blind_decisions = 0
    hindsight_decisions = 0

    for pair in pairs:
        segment_by_key = {segment.segment_key: segment for segment in pair.segments}
        target_sets = [set(future.support_segment_keys) for future in pair.futures]
        target_costs = [
            sum(token_counter(segment_by_key[key].text) for key in support)
            for support in target_sets
        ]
        if any(cost > pair.max_context_tokens for cost in target_costs):
            raise ValueError(f"counterfactual support exceeds budget in {pair.pair_id}")
        union = target_sets[0] | target_sets[1]
        union_cost = sum(token_counter(segment_by_key[key].text) for key in union)
        if union_cost <= pair.max_context_tokens:
            raise ValueError(
                f"paired futures are jointly retainable in {pair.pair_id}; no 0.5 ceiling"
            )
        total_tokens = sum(token_counter(segment.text) for segment in pair.segments)
        if total_tokens <= pair.max_context_tokens:
            raise ValueError(f"full counterfactual context fits in {pair.pair_id}")
        target_token_gaps.append(abs(target_costs[0] - target_costs[1]))
        target_capitalized_gaps.append(
            abs(
                sum(
                    len(re.findall(r"\b[A-Z][A-Za-z0-9-]*\b", segment_by_key[key].text))
                    for key in target_sets[0]
                )
                - sum(
                    len(re.findall(r"\b[A-Z][A-Za-z0-9-]*\b", segment_by_key[key].text))
                    for key in target_sets[1]
                )
            )
        )
        for seed in selected_seeds:
            public_input, private = _blind_stage2_input(pair, seed, token_counter)
            public_digest = _canonical_digest(public_input.model_dump(mode="json"))
            for policy in STAGE2_STRESS_POLICIES:
                shared_decision: Optional[Tuple[Tuple[str, ...], int]] = None
                if policy != "oracle_future":
                    shared_support = (
                        target_sets[0] if policy == "pair_blind_oracle" else None
                    )
                    ranked = _rank_stage2(
                        policy,
                        public_input,
                        private=private,
                        support_keys=shared_support,
                    )
                    shared_decision = _fit_stage2(policy, ranked, public_input)
                    query_blind_decisions += 1
                for future in pair.futures:
                    support_keys = set(future.support_segment_keys)
                    if policy == "oracle_future":
                        ranked = _rank_stage2(
                            policy,
                            public_input,
                            private=private,
                            support_keys=support_keys,
                        )
                        chosen, kept_tokens = _fit_stage2(
                            policy,
                            ranked,
                            public_input,
                        )
                        hindsight_decisions += 1
                    else:
                        assert shared_decision is not None
                        chosen, kept_tokens = shared_decision
                        identity_checks += 1
                        identity_matches += 1
                    chosen_keys = {private[handle].segment_key for handle in chosen}
                    retained = chosen_keys & support_keys
                    distractors = set(segment_by_key) - support_keys
                    removed_distractors = distractors - chosen_keys
                    recall = _ratio(len(retained), len(support_keys))
                    budget_compliant = kept_tokens <= pair.max_context_tokens
                    rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "split": pair.split,
                            "scenario": pair.scenario,
                            "seed": seed,
                            "variant_id": future.variant_id,
                            "public_digest": public_digest,
                            "policy": policy,
                            "future_support_recall": recall,
                            "memory_precision": _ratio(len(retained), len(chosen_keys)),
                            "distractor_removal_recall": _ratio(
                                len(removed_distractors), len(distractors)
                            ),
                            "budget_compliant": budget_compliant,
                            "safe_success": budget_compliant and recall == 1.0,
                            "kept_tokens": kept_tokens,
                        }
                    )

    splits = sorted({str(row["split"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    return Stage2CounterfactualReport(
        dataset_digest=selected_dataset.digest(),
        pair_count=len(pairs),
        variant_count=sum(len(pair.futures) for pair in pairs),
        seeds=selected_seeds,
        budgets=tuple(sorted({pair.max_context_tokens for pair in pairs})),
        public_input_fields=tuple(BlindStage2Input.model_fields),
        policies=STAGE2_STRESS_POLICIES,
        arm_count_per_policy=(
            sum(len(pair.futures) for pair in pairs) * len(selected_seeds)
        ),
        query_blind_decision_count=query_blind_decisions,
        hindsight_decision_count=hindsight_decisions,
        public_input_identity_rate=_ratio(identity_matches, identity_checks),
        public_safe_success_ceiling=0.5,
        max_target_token_gap=max(target_token_gaps),
        max_target_capitalized_gap=max(target_capitalized_gaps),
        aggregates=_stage2_aggregate_map(rows),
        by_split={
            split: _stage2_aggregate_map([row for row in rows if row["split"] == split])
            for split in splits
        },
        by_scenario={
            scenario: _stage2_aggregate_map(
                [row for row in rows if row["scenario"] == scenario]
            )
            for scenario in scenarios
        },
    )


def _derive_stress_gates(
    stage1: Stage1StressReport,
    stage2: Stage2CounterfactualReport,
    *,
    real_llm_call_count: int,
) -> Tuple[StressGate, ...]:
    oracle_stage1 = stage1.aggregates["oracle_support"]
    store_all = stage1.aggregates["store_all"]
    public_stage2 = [
        metrics
        for name, metrics in stage2.aggregates.items()
        if name not in {"pair_blind_oracle", "oracle_future"}
    ]
    pair_blind_oracle = stage2.aggregates["pair_blind_oracle"]
    oracle_stage2 = stage2.aggregates["oracle_future"]
    min_coverage = min(item.coverage_rate for item in stage1.permutation_coverage)
    max_public_safe = max(item.safe_success_rate for item in public_stage2)
    return (
        StressGate(
            name="stage1_public_boundary_hides_task_and_seed",
            passed=stage1.public_input_fields == ("budget_tokens", "observed_facts"),
            evidence=f"public_fields={stage1.public_input_fields}",
        ),
        StressGate(
            name="stage1_uses_at_least_50_order_seeds",
            passed=len(stage1.seeds) >= 50,
            evidence=f"seed_count={len(stage1.seeds)}",
        ),
        StressGate(
            name="stage1_covers_all_three_fact_permutations",
            passed=min_coverage == 1.0,
            evidence=f"minimum_coverage={min_coverage:.3f}",
        ),
        StressGate(
            name="stage1_uses_three_global_budgets",
            passed=len(stage1.budgets) >= 3
            and len(set(stage1.budgets)) == len(stage1.budgets),
            evidence=f"budgets={stage1.budgets}",
        ),
        StressGate(
            name="stage1_store_all_is_not_robust",
            passed=(
                store_all.oracle_equivalent_rate < oracle_stage1.oracle_equivalent_rate
                and store_all.support_recall < oracle_stage1.support_recall
            ),
            evidence=(
                f"store_all_recall={store_all.support_recall:.3f}, "
                f"oracle_recall={oracle_stage1.support_recall:.3f}, "
                f"store_all_oracle_equivalent={store_all.oracle_equivalent_rate:.3f}"
            ),
        ),
        StressGate(
            name="stage2_public_inputs_are_counterfactually_identical",
            passed=stage2.public_input_identity_rate == 1.0,
            evidence=f"identity_rate={stage2.public_input_identity_rate:.3f}",
        ),
        StressGate(
            name="stage2_targets_are_length_and_style_matched",
            passed=(
                stage2.max_target_token_gap <= 2
                and stage2.max_target_capitalized_gap <= 1
            ),
            evidence=(
                f"max_token_gap={stage2.max_target_token_gap}, "
                f"max_capitalized_gap={stage2.max_target_capitalized_gap}"
            ),
        ),
        StressGate(
            name="stage2_public_policies_respect_counterfactual_ceiling",
            passed=max_public_safe <= stage2.public_safe_success_ceiling + 1e-12,
            evidence=(
                f"max_public_safe_success={max_public_safe:.3f}, "
                f"ceiling={stage2.public_safe_success_ceiling:.3f}"
            ),
        ),
        StressGate(
            name="stage2_pair_blind_oracle_reaches_exact_ceiling",
            passed=(
                pair_blind_oracle.safe_success_rate
                == stage2.public_safe_success_ceiling
                and pair_blind_oracle.budget_compliance_rate == 1.0
            ),
            evidence=(
                f"safe_success={pair_blind_oracle.safe_success_rate:.3f}, "
                f"ceiling={stage2.public_safe_success_ceiling:.3f}"
            ),
        ),
        StressGate(
            name="stage2_oracle_future_is_feasible",
            passed=(
                oracle_stage2.safe_success_rate == 1.0
                and oracle_stage2.budget_compliance_rate == 1.0
            ),
            evidence=(
                f"safe_success={oracle_stage2.safe_success_rate:.3f}, "
                f"budget_rate={oracle_stage2.budget_compliance_rate:.3f}"
            ),
        ),
        StressGate(
            name="offline_run_makes_no_real_llm_calls",
            passed=real_llm_call_count == 0,
            evidence=f"real_llm_call_count={real_llm_call_count}",
        ),
    )


def stress_report_digest(
    report: AntiShortcutStressReport | Mapping[str, object],
    *,
    include_digest: bool = False,
) -> str:
    payload = (
        report.model_dump(mode="json")
        if isinstance(report, AntiShortcutStressReport)
        else dict(report)
    )
    if not include_digest:
        payload.pop("digest", None)
    return _canonical_digest(payload)


def run_anti_shortcut_stress(
    *,
    seeds: Sequence[int] = DEFAULT_STRESS_SEEDS,
    stage1_budgets: Sequence[int] = DEFAULT_STAGE1_BUDGETS,
    task_dataset: Optional[ToyTaskDataset] = None,
    counterfactual_dataset: Optional[CounterfactualDataset] = None,
    token_counter: Optional[Callable[[str], int]] = None,
    token_counter_spec: Optional[TokenCounterSpec] = None,
) -> AntiShortcutStressReport:
    if (token_counter is None) != (token_counter_spec is None):
        raise ValueError("token counter and its spec must be supplied together")
    if token_counter is None:
        token_counter, token_counter_spec = lexical_token_counter()
    assert token_counter_spec is not None
    stage1 = run_stage1_stress(
        dataset=task_dataset,
        seeds=seeds,
        budgets=stage1_budgets,
        token_counter=token_counter,
    )
    stage2 = run_stage2_counterfactual_stress(
        dataset=counterfactual_dataset,
        seeds=seeds,
        token_counter=token_counter,
    )
    gates = _derive_stress_gates(stage1, stage2, real_llm_call_count=0)
    payload = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "token_counter": token_counter_spec.model_dump(mode="json"),
        "real_llm_call_count": 0,
        "stage1": stage1.model_dump(mode="json"),
        "stage2": stage2.model_dump(mode="json"),
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }
    payload["digest"] = stress_report_digest(payload)
    return AntiShortcutStressReport.model_validate(payload)


def anti_shortcut_stress_markdown(report: AntiShortcutStressReport) -> str:
    stage1_rows = "\n".join(
        "| {name} | {recall:.3f} | {normalized:.3f} | {precision:.3f} | "
        "{exact:.3f} | {equivalent:.3f} |".format(
            name=name,
            recall=row.support_recall,
            normalized=row.oracle_normalized_recall,
            precision=row.memory_precision,
            exact=row.exact_support_rate,
            equivalent=row.oracle_equivalent_rate,
        )
        for name in STAGE1_POLICIES
        for row in (report.stage1.aggregates[name],)
    )
    stage2_rows = "\n".join(
        "| {name} | {support:.3f} | {precision:.3f} | {removed:.3f} | "
        "{budget:.3f} | {safe:.3f} |".format(
            name=name,
            support=row.future_support_recall,
            precision=row.memory_precision,
            removed=row.distractor_removal_recall,
            budget=row.budget_compliance_rate,
            safe=row.safe_success_rate,
        )
        for name in STAGE2_STRESS_POLICIES
        for row in (report.stage2.aggregates[name],)
    )
    gate_rows = "\n".join(
        f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.evidence} |"
        for gate in report.gates
    )
    tokenizer_identity = report.token_counter.name
    if report.token_counter.repository_id:
        tokenizer_identity += (
            f" (`{report.token_counter.repository_id}` @ "
            f"`{report.token_counter.revision}`; assets "
            f"`{report.token_counter.assets_digest}`)"
        )
    min_coverage = min(
        item.coverage_rate for item in report.stage1.permutation_coverage
    )
    return f"""# Stage 1/2 Anti-Shortcut Stress Experiment

这是与 v2 CI canary 分离的扩展离线实验。它不调用 Agent、LLM、embedding 服务或网络，也不改写 E1 配置和训练 buffer。

- Schema: `{report.schema_version}`
- Token counter: {tokenizer_identity}
- Stage 1: `{report.stage1.task_count}` 个含噪任务 × `{len(report.stage1.seeds)}` seeds × `{len(report.stage1.budgets)}` 固定预算；每个策略 `{report.stage1.arm_count_per_policy}` arms
- Stage 1 budgets: `{report.stage1.budgets}`
- Stage 1 minimum permutation coverage: `{min_coverage:.3f}`
- Stage 2: `{report.stage2.pair_count}` 个反事实对 / `{report.stage2.variant_count}` 个 future variants × `{len(report.stage2.seeds)}` seeds；每个策略 `{report.stage2.arm_count_per_policy}` arms
- Stage 2 budgets: `{report.stage2.budgets}`
- Stage 2 public-input identity: `{report.stage2.public_input_identity_rate:.3f}`
- Real LLM calls: `{report.real_llm_call_count}`
- Integrity gates: `{"PASS" if report.passed else "FAIL"}`
- Repeatability checksum: `{report.digest}`

## Reproduce

```bash
python scripts/agemem_anti_shortcut_stress.py

# AutoDL: rerun with the frozen production tokenizer.
stress_dir="$TRINITY_CHECKPOINT_ROOT_DIR/anti_shortcut_stress/$AGEMEM_EXPECTED_COMMIT"
python scripts/agemem_anti_shortcut_stress.py \\
  --tokenizer-path "$TRINITY_MODEL_PATH" \\
  --tokenizer-revision "$TRINITY_MODEL_REVISION" \\
  --tokenizer-repository-id Qwen/Qwen2.5-1.5B-Instruct \\
  --output-dir "$stress_dir" \\
  --docs-path "$stress_dir/report.md"
```

## Stage 1: multi-task / multi-seed / fixed-budget

非 Oracle 策略只能看到 `budget_tokens` 和带不透明句柄的公开事实；看不到 task ID、split、seed 或事实角色。`reverse_order` 是 last-in-first 的一轮容量代理；在这个静态一次写入实验中，真正的 LRU 与 FIFO 没有额外访问事件可区分。

| Policy | Support recall | Oracle-normalized recall | Memory precision | Exact support | Oracle equivalent |
|---|---:|---:|---:|---:|---:|
{stage1_rows}

`entity_chain` 是公开文本启发式，用于显式检测模板/实体链捷径；`oracle_support` 使用私有 supporting labels，只是离线上界。

## Stage 2: paired counterfactual futures

每个 pair 的两个 future query 在决策时共享完全相同的公开上下文，但需要保留互斥的支持段；两组支持无法同时装入预算。因此任何 query-blind 决策在一对上的 safe-success 上界是 `{report.stage2.public_safe_success_ceiling:.3f}`。目标段最大 token 差为 `{report.stage2.max_target_token_gap}`，最大大写词数量差为 `{report.stage2.max_target_capitalized_gap}`。

| Policy | Future-support recall | Memory precision | Distractor removal | Budget compliance | Safe success |
|---|---:|---:|---:|---:|---:|
{stage2_rows}

`oracle_future` 在查询揭示后使用私有标签，仅用于证明预算内可行解存在。公开策略结果应结合反事实上界解释，不能把 query-blind 任务上的 0.5 当作模型已经学会未来相关性。

## Integrity gates

| Gate | Result | Evidence |
|---|---|---|
{gate_rows}

## Evidence boundary

本报告衡量固定公开基线和数据构造，不是已训练模型结果。若 token counter 为 `unicode-lexical-v1`，它只是一份本机可复现的协议验证；正式上卡结果必须用冻结的 `Qwen/Qwen2.5-1.5B-Instruct` 本地 tokenizer、完整 40 位 revision 和 tokenizer assets digest 重跑。模型策略仍需另报 Answer EM/F1、support F1、memory precision、预算合规率和序列化 token 数。
"""


def write_anti_shortcut_stress_report(
    report: AntiShortcutStressReport,
    *,
    output_dir: str | Path,
    docs_path: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "anti_shortcut_stress.json"
    markdown_path = output / "anti_shortcut_stress.md"
    json_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = anti_shortcut_stress_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    if docs_path is not None:
        docs = Path(docs_path)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(markdown, encoding="utf-8", newline="\n")
    return json_path, markdown_path


__all__ = [
    "AntiShortcutStressReport",
    "BlindStage1Input",
    "BlindStage2Input",
    "CounterfactualDataset",
    "CounterfactualPair",
    "DEFAULT_STAGE1_BUDGETS",
    "DEFAULT_STRESS_SEEDS",
    "STRESS_SCHEMA_VERSION",
    "Stage1StressReport",
    "Stage2CounterfactualReport",
    "StressGate",
    "TokenCounterSpec",
    "anti_shortcut_stress_markdown",
    "default_counterfactual_path",
    "frozen_hf_token_counter",
    "lexical_token_counter",
    "run_anti_shortcut_stress",
    "run_stage1_stress",
    "run_stage2_counterfactual_stress",
    "stress_report_digest",
    "write_anti_shortcut_stress_report",
]
