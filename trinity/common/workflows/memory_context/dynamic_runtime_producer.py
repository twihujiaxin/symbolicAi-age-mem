"""Model-backed runtime producer for ``streaming_dynamic_multiquery_v2``.

The trainable policy sees only :class:`DynamicMemoryEnvironment` messages.
Questions are introduced after the immutable memory snapshot is created and are
answered by a separately supplied frozen reader.  Private events, answers and
source-registry bodies are used only after sampling for reward evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from AgeMem_code_agentscope.streaming_memory.dynamic.environment import (
    DynamicMemoryEnvironment,
    DynamicMemorySnapshot,
    canonical_dynamic_payload,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.reward_profiles import (
    DynamicReward,
    aggregate_dynamic_reward,
    exact_match,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import (
    DynamicHistoryPublic,
    DynamicQueryPrivate,
    DynamicQueryPublic,
    PrivateEvent,
    REWARD_VERSION,
    SourceRegistryRecord,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.state_evaluator import (
    DirectStateEvaluator,
    build_semantic_frames,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.temporal_grounder import (
    exposed_payloads_as_memories,
    validate_memory_semantics,
)
from AgeMem_code_agentscope.streaming_memory.query_runner import (
    ANSWER_SYSTEM,
    FrozenLexicalRetriever,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    BudgetError,
    TokenAccounting,
)
from trinity.common.action_event_contract import (
    ActionContractError,
    parse_tool_calls_with_char_spans,
    prepare_experience_action_drafts,
    record_experience_action_result,
    stable_action_id,
)
from trinity.common.dynamic_multiquery_contract import (
    DynamicMemoryRolloutGroupBundle,
    DynamicQueryBranchScore,
    DynamicReadRollout,
)
from trinity.common.experience import EID, Experience
from trinity.common.streaming_multiquery_contract import ReadActorSample


RUNTIME_PRODUCER_VERSION = "agemem.dynamic.runtime_producer.v1"
ACTION_INTERFACE_VERSION = "agemem.dynamic.action_interface.v2"
INGEST_STAGE_ID = 1


class DynamicRuntimeError(RuntimeError):
    """Fail-closed model/runtime contract error."""


@dataclass(frozen=True)
class FrozenReaderOutput:
    answer_text: str
    total_token_count: int
    reader_version: str


@dataclass(frozen=True)
class DynamicReaderBranch:
    query_branch_id: str
    query_id: str
    answer_text: str
    prompt_token_count: int
    total_reader_token_count: int
    reader_version: str
    retrieved_memory_ids: tuple[str, ...]
    retrieved_payloads: tuple[str, ...]


@dataclass(frozen=True)
class ProducedDynamicGroup:
    experiences: tuple[Experience, ...]
    bundle: DynamicMemoryRolloutGroupBundle
    rewards: tuple[DynamicReward, ...]
    receipt: dict[str, Any]


FrozenReader = Callable[
    [tuple[dict[str, str], ...]], Awaitable[FrozenReaderOutput]
]


def _plain(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise DynamicRuntimeError("runtime tensor field is not one-dimensional")
    return value


def actor_sample_from_experience(
    experience: Experience,
    *,
    action_id: str,
    shared_prefix_id: str,
    policy_version: str,
) -> ReadActorSample:
    """Bind one actual model response to the V2 actor-sample contract."""

    tokens = [int(item) for item in _plain(experience.tokens)]
    prompt_length = experience.prompt_length
    if not isinstance(prompt_length, int) or not 0 < prompt_length < len(tokens):
        raise DynamicRuntimeError("invalid prompt/response boundary")
    response_ids = tokens[prompt_length:]
    if experience.logprobs is None or experience.action_mask is None:
        raise DynamicRuntimeError("ingest Experience lacks logprobs/action_mask")
    logprobs = [float(item) for item in _plain(experience.logprobs)]
    action_mask = [bool(item) for item in _plain(experience.action_mask)]
    if len(response_ids) != len(logprobs) or len(response_ids) != len(action_mask):
        raise DynamicRuntimeError("response/logprob/action-mask lengths differ")
    if not all(action_mask):
        raise DynamicRuntimeError("one ingest response contains masked reader tokens")
    return ReadActorSample(
        action_id=action_id,
        shared_prefix_id=shared_prefix_id,
        token_ids=tuple(response_ids),
        old_logprobs=tuple(logprobs),
        action_mask=tuple(action_mask),
        policy_version=policy_version,
    )


def _one_public_action(response_text: str) -> tuple[str, dict[str, Any]] | None:
    # Unlike the legacy tolerant protocol, V2 admits exactly one complete JSON
    # array and no surrounding prose, extra envelopes or synthetic repairs.
    payload = response_text.strip()
    if payload.startswith("<tool_call>") and payload.endswith("</tool_call>"):
        payload = payload[len("<tool_call>") : -len("</tool_call>")].strip()
    try:
        value = json.loads(payload)
        if not isinstance(value, list) or len(value) != 1:
            return None
        calls = parse_tool_calls_with_char_spans(
            response_text, allow_bare_json_array=True
        )
    except (ActionContractError, json.JSONDecodeError):
        return None
    if len(calls) != 1 or not isinstance(calls[0].call, dict):
        return None
    call = calls[0].call
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return name, dict(arguments)


class DynamicRuntimeProducer:
    """Execute K ingest rollouts and materialize one complete Trinity group."""

    def __init__(
        self,
        *,
        policy_model: Any,
        frozen_reader: FrozenReader,
        accounting: TokenAccounting,
        config: Mapping[str, Any],
        policy_version: str,
        generation_args: Mapping[str, Any] | None = None,
    ) -> None:
        if not policy_version:
            raise DynamicRuntimeError("policy_version is required")
        self.policy_model = policy_model
        self.frozen_reader = frozen_reader
        self.accounting = accounting
        self.config = config
        self.policy_version = policy_version
        self.generation_args = dict(generation_args or {})
        self.generation_args["n"] = 1
        self.generation_args["max_tokens"] = int(
            config["budget"]["ingest_max_new_tokens"]
        )
        self.generation_args["record_action_metadata"] = True
        self.retriever = FrozenLexicalRetriever()

    def _environment(
        self, history: DynamicHistoryPublic, read_rollout_id: str
    ) -> DynamicMemoryEnvironment:
        budget = self.config["budget"]
        return DynamicMemoryEnvironment(
            history,
            accounting=self.accounting,
            read_rollout_id=read_rollout_id,
            policy_version=self.policy_version,
            ingest_max_new_tokens=int(budget["ingest_max_new_tokens"]),
            max_decisions_per_chunk=int(budget["max_decisions_per_chunk"]),
            answer_tail_tokens=int(budget["answer_tail_tokens"]),
            retrieval_payload_tokens=int(budget["retrieved_payload_tokens"]),
        )

    async def _reader_branch(
        self,
        *,
        snapshot: DynamicMemorySnapshot,
        query: DynamicQueryPublic,
        branch_index: int,
    ) -> DynamicReaderBranch:
        budget = self.config["budget"]
        ranked = self.retriever.rank(query.question, snapshot.active_memories)
        base = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"QUESTION\n{query.question}"},
        ]
        selected: list[dict[str, Any]] = []
        payloads: list[str] = []
        payload_tokens = 0
        for memory in ranked:
            payload = canonical_dynamic_payload(memory)
            cost = self.accounting.count_text(payload)
            if payload_tokens + cost > int(budget["retrieved_payload_tokens"]):
                continue
            proposed = base + [
                {
                    "role": "user",
                    "content": "RETRIEVED MEMORY\n" + "\n".join(payloads + [payload]),
                }
            ]
            try:
                self.accounting.enforce_context(
                    proposed,
                    max_new_tokens=int(budget["answer_max_new_tokens"]),
                    context_total_tokens=int(budget["context_total_tokens"]),
                )
            except BudgetError:
                continue
            selected.append(memory)
            payloads.append(payload)
            payload_tokens += cost

        retrieval = {
            "role": "user",
            "content": "RETRIEVED MEMORY\n" + ("\n".join(payloads) or "(none)"),
        }
        tail: list[dict[str, str]] = []
        for message in reversed(snapshot.context_tail):
            proposed_tail = [dict(message)] + tail
            try:
                self.accounting.enforce_context(
                    base + proposed_tail + [retrieval],
                    max_new_tokens=int(budget["answer_max_new_tokens"]),
                    context_total_tokens=int(budget["context_total_tokens"]),
                )
            except BudgetError:
                continue
            tail = proposed_tail
        messages = tuple(base + tail + [retrieval])
        prompt_tokens = self.accounting.enforce_context(
            messages,
            max_new_tokens=int(budget["answer_max_new_tokens"]),
            context_total_tokens=int(budget["context_total_tokens"]),
        )
        before = (snapshot.memory_sha256, snapshot.tail_sha256)
        output = await self.frozen_reader(messages)
        if before != (snapshot.memory_sha256, snapshot.tail_sha256):
            raise DynamicRuntimeError("reader mutated its parent snapshot")
        if not output.reader_version or output.total_token_count < prompt_tokens:
            raise DynamicRuntimeError("invalid frozen-reader receipt")
        return DynamicReaderBranch(
            query_branch_id=f"{snapshot.snapshot_id}:branch:{branch_index}",
            query_id=query.query_id,
            answer_text=output.answer_text,
            prompt_token_count=prompt_tokens,
            total_reader_token_count=output.total_token_count,
            reader_version=output.reader_version,
            retrieved_memory_ids=tuple(
                str(item["memory_id"]) for item in selected
            ),
            retrieved_payloads=tuple(payloads),
        )

    async def _rollout(
        self,
        *,
        history: DynamicHistoryPublic,
        public_queries: Sequence[DynamicQueryPublic],
        private_queries: Sequence[DynamicQueryPrivate],
        events_by_id: Mapping[str, PrivateEvent],
        source_registry: Mapping[str, SourceRegistryRecord],
        batch_id: str | int,
        task_id: str | int,
        run_id: int,
    ) -> tuple[DynamicReadRollout, list[Experience], DynamicReward]:
        read_rollout_id = f"{batch_id}/{task_id}/{run_id}"
        shared_prefix_id = "prefix-" + hashlib.sha256(
            read_rollout_id.encode("utf-8")
        ).hexdigest()[:20]
        environment = self._environment(history, read_rollout_id)
        samples: list[ReadActorSample] = []
        experiences: list[Experience] = []
        timestep = 0
        max_decisions = int(self.config["budget"]["max_decisions_per_chunk"])
        for _ in history.chunks:
            messages = environment.admit_next_chunk()
            committed = False
            for decision_index in range(max_decisions):
                generated = await self.policy_model.chat_async(
                    list(messages), **self.generation_args
                )
                if len(generated) != 1:
                    raise DynamicRuntimeError("policy must return exactly one response")
                experience = generated[0]
                if not isinstance(experience.response_text, str):
                    raise DynamicRuntimeError("policy response_text is required")
                experience.eid = EID(
                    batch=batch_id,
                    task=task_id,
                    run=run_id,
                    step=timestep,
                    suffix=hashlib.sha256(
                        f"{read_rollout_id}:{timestep}".encode("utf-8")
                    ).hexdigest()[:6],
                )
                parsed = _one_public_action(experience.response_text)
                if parsed is None:
                    action_name, arguments = "<invalid_tool_call>", {}
                    result = environment.execute(
                        {"type": "<invalid_tool_call>"},
                        response_text=experience.response_text,
                    )
                    action_id = (
                        "agemem-decision-"
                        + hashlib.sha256(
                            f"{read_rollout_id}:{timestep}:invalid".encode("utf-8")
                        ).hexdigest()[:24]
                    )
                else:
                    action_name, arguments = parsed
                    prepare_experience_action_drafts(
                        experience,
                        stage_id=INGEST_STAGE_ID,
                        timestep=timestep,
                        assistant_turn_id=timestep,
                        allow_bare_json_array=True,
                    )
                    action = {"type": action_name, **arguments}
                    result = environment.execute(
                        action, response_text=experience.response_text
                    )
                    trace_call_id = f"{read_rollout_id}:call:{timestep}"
                    record_experience_action_result(
                        [experience],
                        action_index_in_turn=0,
                        trace_call_id=trace_call_id,
                        action_type=action_name,
                        status="success" if result.admitted else "error",
                        result=asdict(result),
                        error=None if result.admitted else result.code,
                    )
                    info = dict(experience.info or {})
                    info["tool_call_ids"] = [trace_call_id]
                    experience.info = info
                    action_id = stable_action_id(
                        rollout_id=experience.eid.rid,
                        stage_id=INGEST_STAGE_ID,
                        timestep=timestep,
                        assistant_turn_id=timestep,
                        action_index_in_turn=0,
                    )
                info = dict(experience.info or {})
                info.update(
                    {
                        "dynamic_runtime_producer": RUNTIME_PRODUCER_VERSION,
                        "dynamic_action_interface": ACTION_INTERFACE_VERSION,
                        "dynamic_action_id": action_id,
                        "dynamic_action_type": action_name,
                        "dynamic_action_admitted": result.admitted,
                        "dynamic_action_code": result.code,
                        "phase": "ingest",
                    }
                )
                info.setdefault("tool_call_ids", [])
                experience.info = info
                samples.append(
                    actor_sample_from_experience(
                        experience,
                        action_id=action_id,
                        shared_prefix_id=shared_prefix_id,
                        policy_version=self.policy_version,
                    )
                )
                experiences.append(experience)
                timestep += 1
                committed = action_name.upper() == "NEXT" and result.admitted
                if committed:
                    break
                messages = environment.current_messages()
                if decision_index == max_decisions - 1:
                    environment.commit_chunk()
                    committed = True
            if not committed:
                raise DynamicRuntimeError("chunk did not reach an external checkpoint")

        snapshot = environment.finalize()
        frames = build_semantic_frames(
            environment.checkpoints,
            queries=private_queries,
            source_registry=source_registry,
            events_by_id=events_by_id,
        )
        state_evaluation = DirectStateEvaluator().evaluate(frames)
        branches: list[DynamicReaderBranch] = []
        exposure: dict[str, float] = {}
        answers: dict[str, str] = {}
        contract_branches: list[DynamicQueryBranchScore] = []
        for index, (public, private) in enumerate(
            zip(public_queries, private_queries, strict=True)
        ):
            if public.query_id != private.query_id:
                raise DynamicRuntimeError("public/private query identity mismatch")
            branch = await self._reader_branch(
                snapshot=snapshot, query=public, branch_index=index
            )
            branches.append(branch)
            answers[private.query_id] = branch.answer_text
            exposure_state = validate_memory_semantics(
                exposed_payloads_as_memories(branch.retrieved_payloads),
                observed_source_refs=snapshot.observed_source_refs,
                source_registry=source_registry,
                events_by_id=events_by_id,
                query=private,
            )
            exposure[private.query_id] = exposure_state.utility
            contract_branches.append(
                DynamicQueryBranchScore(
                    query_branch_id=branch.query_branch_id,
                    query_id=private.query_id,
                    task_score=exact_match(branch.answer_text, private.answers),
                    reader_version=branch.reader_version,
                    reader_token_count=branch.total_reader_token_count,
                )
            )
        reward = aggregate_dynamic_reward(
            profile=str(self.config["reward"]["profile"]),
            queries=private_queries,
            answer_text_by_query=answers,
            exposure_utility_by_query=exposure,
            state_evaluation=state_evaluation,
            lambda_semantic=float(self.config["reward"]["lambda_semantic"]),
        )
        rollout = DynamicReadRollout(
            read_rollout_id=read_rollout_id,
            snapshot_id=snapshot.snapshot_id,
            shared_prefix_id=shared_prefix_id,
            policy_version=self.policy_version,
            samples=tuple(samples),
            branches=tuple(contract_branches),
            task_reward=reward.task_reward,
            semantic_component=reward.semantic_component,
            total_reward=reward.total_reward,
            reward_profile=reward.profile,
        )
        for experience in experiences:
            experience.reward = reward.total_reward
            info = dict(experience.info or {})
            info.update(
                {
                    "dynamic_snapshot_id": snapshot.snapshot_id,
                    "dynamic_history_family_id": history.history_family_id,
                    "dynamic_reward_profile": reward.profile,
                    "dynamic_reward_version": reward.reward_version,
                    "dynamic_task_reward": reward.task_reward,
                    "dynamic_semantic_component": reward.semantic_component,
                    "dynamic_total_reward": reward.total_reward,
                    "dynamic_reader_actor_loss_tokens": 0,
                    "dynamic_query_branches": [
                        {
                            "query_branch_id": item.query_branch_id,
                            "query_id": item.query_id,
                            "prompt_token_count": item.prompt_token_count,
                            "total_reader_token_count": item.total_reader_token_count,
                            "reader_version": item.reader_version,
                            "retrieved_memory_ids": list(item.retrieved_memory_ids),
                        }
                        for item in branches
                    ],
                }
            )
            experience.info = info
            metrics = dict(experience.metrics or {})
            metrics.update(
                {
                    "dynamic_task_reward": reward.task_reward,
                    "dynamic_semantic_component": reward.semantic_component,
                    "dynamic_total_reward": reward.total_reward,
                    "dynamic_reader_tokens": float(
                        sum(item.total_reader_token_count for item in branches)
                    ),
                }
            )
            experience.metrics = metrics
        return rollout, experiences, reward

    async def produce_group(
        self,
        *,
        history: DynamicHistoryPublic,
        public_queries: Sequence[DynamicQueryPublic],
        private_queries: Sequence[DynamicQueryPrivate],
        events: Sequence[PrivateEvent],
        source_registry: Sequence[SourceRegistryRecord],
        batch_id: str | int,
        task_id: str | int,
        run_id_base: int,
        repeat_times: int,
    ) -> ProducedDynamicGroup:
        expected_k = int(self.config["data"]["memory_rollouts_per_group"])
        expected_m = int(self.config["data"]["questions_per_snapshot"])
        if repeat_times != expected_k:
            raise DynamicRuntimeError("runtime K differs from frozen configuration")
        if len(public_queries) != expected_m or len(private_queries) != expected_m:
            raise DynamicRuntimeError("runtime m differs from frozen configuration")
        if any(item.history_id != history.history_id for item in public_queries):
            raise DynamicRuntimeError("public query belongs to another history")
        if any(item.history_id != history.history_id for item in private_queries):
            raise DynamicRuntimeError("private query belongs to another history")
        events_by_id = {
            item.event_id: item for item in events if item.history_id == history.history_id
        }
        registry = {
            item.source_ref: item
            for item in source_registry
            if item.history_id == history.history_id
        }
        if not events_by_id or not registry:
            raise DynamicRuntimeError("private evaluator sidecar is incomplete")
        rollouts: list[DynamicReadRollout] = []
        experiences: list[Experience] = []
        rewards: list[DynamicReward] = []
        for offset in range(repeat_times):
            rollout, rollout_experiences, reward = await self._rollout(
                history=history,
                public_queries=public_queries,
                private_queries=private_queries,
                events_by_id=events_by_id,
                source_registry=registry,
                batch_id=batch_id,
                task_id=task_id,
                run_id=run_id_base + offset,
            )
            rollouts.append(rollout)
            experiences.extend(rollout_experiences)
            rewards.append(reward)
        group_id = f"dynamic-v2:{batch_id}:{task_id}"
        bundle = DynamicMemoryRolloutGroupBundle(
            group_id=group_id,
            history_family_id=history.history_family_id,
            expected_rollouts=expected_k,
            expected_queries_per_rollout=expected_m,
            rollouts=tuple(rollouts),
            reward_version=REWARD_VERSION,
            reward_profile=str(self.config["reward"]["profile"]),
            lambda_semantic=float(self.config["reward"]["lambda_semantic"]),
            std_ddof=int(self.config["training"]["std_ddof"]),
        )
        bundle.validate_complete()
        advantages = bundle.advantages()
        advantage_by_run = {
            rollout.read_rollout_id: advantage
            for rollout, advantage in zip(rollouts, advantages, strict=True)
        }
        for experience in experiences:
            info = dict(experience.info or {})
            rollout_id = experience.eid.rid
            info.update(
                {
                    "dynamic_group_id": group_id,
                    "dynamic_expected_k": expected_k,
                    "dynamic_expected_m": expected_m,
                    "dynamic_precomputed_advantage_ddof0": advantage_by_run[
                        rollout_id
                    ],
                }
            )
            experience.info = info
        receipt = bundle.receipt()
        receipt.update(
            {
                "runtime_producer_version": RUNTIME_PRODUCER_VERSION,
                "experience_count": len(experiences),
                "model_used": True,
                "reader_frozen": True,
                "action_interface_version": ACTION_INTERFACE_VERSION,
                "invalid_response_count": sum(
                    exp.info["dynamic_action_type"] == "<invalid_tool_call>"
                    for exp in experiences
                ),
                "admitted_memory_write_count": sum(
                    exp.info["dynamic_action_type"].upper() in {"ADD", "UPDATE"}
                    and exp.info["dynamic_action_admitted"]
                    for exp in experiences
                ),
            }
        )
        return ProducedDynamicGroup(
            experiences=tuple(experiences),
            bundle=bundle,
            rewards=tuple(rewards),
            receipt=receipt,
        )


__all__ = [
    "DynamicReaderBranch",
    "DynamicRuntimeError",
    "DynamicRuntimeProducer",
    "FrozenReaderOutput",
    "ProducedDynamicGroup",
    "RUNTIME_PRODUCER_VERSION",
    "actor_sample_from_experience",
]
