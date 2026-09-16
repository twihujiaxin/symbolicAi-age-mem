"""Frozen public-chunk diagnostics with independent first-action environments.

This diagnostic is not a GRPO group, complete memory rollout or reader eval.
No question, Oracle, private source registry or trainer is accepted here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict

from .action_parser import parse_public_action
from .environment import DYNAMIC_INGEST_SYSTEM, DynamicMemoryEnvironment
from .schema import DynamicHistoryPublic

PROBE_VERSION = "agemem.dynamic.first_action_probe.v1"
TASK_EXPLICIT_SYSTEM = (
    "You are the memory writer for a later, separate answering reader. "
    "Future questions are hidden until reading finishes. The reader gets retrieved "
    "retained memory and only a limited context tail, not the full stream or your "
    "hidden understanding. Earlier chunks can leave the context. NEXT only ends "
    "this chunk: it does NOT save its text. ADD/UPDATE explicitly write memory. "
    "Select useful concrete facts with entity, relation, value and effective time, "
    "not topic summaries. Preserve historical facts when current values change. "
    "Respect memory B; choose what to keep. You may skip a chunk, and no action "
    "order or per-chunk ADD is required. " + DYNAMIC_INGEST_SYSTEM
)
SYSTEMS = {"legacy_v3": DYNAMIC_INGEST_SYSTEM, "task_explicit_probe_v1": TASK_EXPLICIT_SYSTEM}
SINGLE_ACTION_SYSTEM = TASK_EXPLICIT_SYSTEM + (
    " This call requests ONLY your next single action, not a plan for the whole chunk. "
    "Return a JSON array containing EXACTLY ONE object, then end the response. "
    "If writing, select ONE concrete fact for this action; do not enumerate the chunk "
    "as multiple ADD objects. After the tool result you can decide another action "
    "while this chunk's decision budget remains. NEXT still ends this chunk. "
    "ADD is not mandatory; choose any legal action appropriate to the current memory."
)
SYSTEMS["single_action_probe_v2"] = SINGLE_ACTION_SYSTEM
SYSTEMS["single_action_no_example_v3"] = SINGLE_ACTION_SYSTEM
STRUCTURE_ONLY_SYSTEM = SINGLE_ACTION_SYSTEM + (
    " ADD arguments must contain memory_id (a fresh nonempty string), content "
    "(one complete factual sentence), and source_refs (a nonempty array of source "
    "IDs visible in this chunk). Put the entity, relation, value and effective time "
    "inside content, NOT in separate arguments named entity/relation/value/time. "
    'Shape only: [{"name":"ADD","arguments":{"memory_id":"<fresh_id>",'
    '"content":"<one factual sentence from the visible chunk>",'
    '"source_refs":["<visible_source_id>"]}}]. '
    "Angle-bracket strings show field roles only: replace them, never copy a "
    "placeholder as a memory/source ID or factual content. Do not add any text "
    "or punctuation outside the JSON array. This shape does not require ADD; "
    "NEXT remains legal with empty arguments."
)
SYSTEMS["structure_only_no_example_v4"] = STRUCTURE_ONLY_SYSTEM
NO_CONCRETE_EXAMPLE_PROFILES = {"single_action_no_example_v3", "structure_only_no_example_v4"}
MULTICHUNK_COMPARISON = "single_vs_no_example_v3"
STRUCTURE_COMPARISON = "no_example_vs_structure_v4"
DEFAULT_COMPARISON = "legacy_vs_task_v1"
COMPARISONS = {
    DEFAULT_COMPARISON: ("legacy_v3", "task_explicit_probe_v1"),
    "task_vs_single_v2": ("task_explicit_probe_v1", "single_action_probe_v2"),
    MULTICHUNK_COMPARISON: ("single_action_probe_v2", "single_action_no_example_v3"),
    STRUCTURE_COMPARISON: ("single_action_no_example_v3", "structure_only_no_example_v4"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_environment(history, accounting, budget, profile, sample_index):
    return DynamicMemoryEnvironment(
        history, accounting=accounting, read_rollout_id=f"probe:{profile}:{sample_index}",
        policy_version="frozen-first-action-probe", ingest_system=SYSTEMS[profile],
        include_add_format_example=profile not in NO_CONCRETE_EXAMPLE_PROFILES,
        ingest_max_new_tokens=int(budget["ingest_max_new_tokens"]),
        max_decisions_per_chunk=int(budget["max_decisions_per_chunk"]),
        answer_tail_tokens=int(budget["answer_tail_tokens"]),
        retrieval_payload_tokens=int(budget["retrieved_payload_tokens"]),
    )


def build_plan(history, accounting, budget, sampling, seed=7, comparison=DEFAULT_COMPARISON):
    if comparison not in COMPARISONS:
        raise ValueError("unknown probe comparison")
    profiles = COMPARISONS[comparison]
    for name in ("temperature", "top_p"):
        value = sampling.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("sampling must explicitly resolve finite temperature/top_p")
    if sampling["temperature"] < 0 or not 0 < sampling["top_p"] <= 1:
        raise ValueError("invalid sampling temperature/top_p")
    if type(sampling.get("top_k")) is not int or (sampling["top_k"] != -1 and sampling["top_k"] <= 0):
        raise ValueError("invalid sampling top_k")
    if history.context_budget_tokens != int(budget["context_total_tokens"]):
        raise ValueError("history/config C mismatch")
    if history.memory_budget_tokens != int(budget["persistent_memory_tokens"]):
        raise ValueError("history/config B mismatch")
    maximum = int(budget["ingest_max_new_tokens"])
    if not 0 < maximum <= 512:
        raise ValueError("probe response cap must be <=512")
    multichunk = comparison in {MULTICHUNK_COMPARISON, STRUCTURE_COMPARISON}
    chunk_indices = [0]
    if multichunk:
        if len(history.chunks) < 3:
            raise ValueError("cross-chunk probe requires at least three public chunks")
        chunk_indices = [0, len(history.chunks) // 2, len(history.chunks) - 1]
    cases = []
    for chunk_index, sample_index in (
        (index, sample) for index in chunk_indices for sample in range(2 if multichunk else 4)
    ):
        for profile in profiles:
            isolated = history.model_copy(update={"chunks": (history.chunks[chunk_index],)})
            env = make_environment(isolated, accounting, budget, profile, sample_index)
            messages = env.admit_next_chunk()
            prompt_ids = list(accounting.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True))
            accounting.enforce_context(messages, max_new_tokens=maximum,
                                       context_total_tokens=history.context_budget_tokens)
            cases.append({"profile": profile, "sample_index": sample_index,
                          "seed": seed + sample_index, "messages": messages,
                          "prompt_token_ids": prompt_ids})
            if multichunk:
                cases[-1].update({"chunk_index": chunk_index, "chunk_id": history.chunks[chunk_index].chunk_id})
    plan = {"probe_version": PROBE_VERSION, "history": history.model_dump(mode="json"),
            "history_sha256": digest(history.model_dump(mode="json")),
            "chunk_id": history.chunks[0].chunk_id, "budget": dict(budget),
            "sampling": {**sampling, "n": 1, "max_tokens": maximum},
            "cases": cases, "model_call_count": len(cases), "reader_call_count": 0,
            "optimizer_update_count": 0, "response_token_upper_bound": len(cases) * maximum,
            "prompt_sha256": {profile: digest(SYSTEMS[profile]) for profile in profiles}}
    if comparison != DEFAULT_COMPARISON:
        plan.update({"comparison": comparison, "probe_version": "agemem.dynamic.first_action_probe.v2"})
    if multichunk:
        plan.update({"probe_version": "agemem.dynamic.first_action_probe.v3", "chunk_indices": chunk_indices,
                     "reading_mode": "isolated_public_chunk_empty_memory_not_sequential"})
    if comparison == STRUCTURE_COMPARISON:
        plan["probe_version"] = "agemem.dynamic.first_action_probe.v4"
    return plan


def execute_probe(plan, accounting, generate):
    """generate accepts locked prompts/params and returns actual token receipts."""
    history = DynamicHistoryPublic.model_validate(plan["history"])
    rebuilt = build_plan(history, accounting, plan["budget"], plan["sampling"],
                         seed=plan["cases"][0]["seed"], comparison=plan.get("comparison", DEFAULT_COMPARISON))
    for key in ("probe_version", "cases", "history_sha256", "prompt_sha256", "sampling"):
        if rebuilt[key] != plan[key]:
            raise ValueError("probe plan differs from current tokenizer/prompt/budget")
    count = rebuilt["model_call_count"]
    if plan["model_call_count"] != count or len(plan["cases"]) != count:
        raise ValueError("probe requires the locked count of independent first actions")
    for key in ("chunk_indices", "reading_mode"):
        if plan.get(key) != rebuilt.get(key):
            raise ValueError("probe chunk selection or reading mode changed")
    params = [{**plan["sampling"], "seed": case["seed"]} for case in plan["cases"]]
    outputs = generate([case["prompt_token_ids"] for case in plan["cases"]], params)
    if len(outputs) != count:
        raise ValueError("model output count differs from locked requests")
    rows = []
    for case, output in zip(plan["cases"], outputs):
        chunk = history.chunks[case.get("chunk_index", 0)]
        visible = dict(zip(chunk.source_refs, chunk.text.split("\n")))
        response = output["response_text"]
        token_ids = output["response_token_ids"]
        if output["prompt_token_ids"] != case["prompt_token_ids"]:
            raise ValueError("model used a different prompt")
        if not isinstance(response, str) or len(token_ids) > plan["sampling"]["max_tokens"]:
            raise ValueError("missing response text or response exceeded token cap")
        if len(case["prompt_token_ids"]) + len(token_ids) > history.context_budget_tokens:
            raise ValueError("actual model call exceeds C")
        parsed = parse_public_action(response)
        isolated = history.model_copy(update={"chunks": (chunk,)})
        env = make_environment(isolated, accounting, plan["budget"], case["profile"], case["sample_index"])
        env.admit_next_chunk()
        result = None
        if parsed.code == "ok":
            result = env.execute({**parsed.arguments, "type": parsed.name}, response_text=response)
        refs = parsed.arguments.get("source_refs") or []
        content = parsed.arguments.get("content")
        valid_refs = bool(refs) and isinstance(refs, list) and all(
            isinstance(ref, str) and ref in visible for ref in refs)
        exact_body = valid_refs and isinstance(content, str) and content.strip() == "\n".join(
            visible[ref] for ref in refs)
        rows.append({"profile": case["profile"], "sample_index": case["sample_index"],
                     "seed": case["seed"], "parse_code": parsed.code, "action_name": parsed.name,
                     "arguments": parsed.arguments, "response_text": response,
                     "prompt_token_count": len(case["prompt_token_ids"]),
                     "response_token_ids": token_ids, "response_token_count": len(token_ids),
                     "finish_reason": output.get("finish_reason"),
                     "max_token_hit": len(token_ids) == plan["sampling"]["max_tokens"],
                     "action_result": asdict(result) if result else None,
                     "memory_tokens": env.memory_tokens(),
                     "source_refs_visible": valid_refs, "exact_visible_source_body": bool(exact_body),
                     "semantic_manual_review_required": True,
                     "human_content_review": "", "human_time_review": "", "human_notes": ""})
        if "chunk_index" in case:
            rows[-1].update({"chunk_index": case["chunk_index"], "chunk_id": chunk.chunk_id,
                            "concrete_add_example_present": case["profile"] not in NO_CONCRETE_EXAMPLE_PROFILES,
                            "selects_first_sentence": bool(exact_body and refs == [chunk.source_refs[0]]),
                            "selects_nonfirst_fact": bool(exact_body and any(ref != chunk.source_refs[0] for ref in refs))})
    arms = {}
    for profile in COMPARISONS[plan.get("comparison", DEFAULT_COMPARISON)]:
        selected = [row for row in rows if row["profile"] == profile]
        arms[profile] = {"sample_count": len(selected), "actions": dict(Counter(row["action_name"] for row in selected)),
                         "parse_codes": dict(Counter(row["parse_code"] for row in selected)),
                         "admitted_writes": sum(row["action_name"] in {"ADD", "UPDATE"}
                             and bool(row["action_result"] and row["action_result"]["admitted"])
                             for row in selected),
                         "exact_visible_source_body_count": sum(row["exact_visible_source_body"] for row in selected)}
        if "chunk_indices" in plan:
            arms[profile].update({"first_sentence_count": sum(row["selects_first_sentence"] for row in selected),
                                  "nonfirst_fact_count": sum(row["selects_nonfirst_fact"] for row in selected),
                                  "max_token_hit_count": sum(row["max_token_hit"] for row in selected),
                                  "admitted_writes_by_chunk": {str(index): sum(row["chunk_index"] == index and
                                      row["action_name"] in {"ADD", "UPDATE"} and bool(row["action_result"] and
                                      row["action_result"]["admitted"]) for row in selected)
                                      for index in plan["chunk_indices"]}})
    report = {"probe_version": plan["probe_version"], "model_call_count": count, "reader_call_count": 0,
              "optimizer_update_count": 0, "history_sha256": plan["history_sha256"],
              "chunk_id": plan["chunk_id"], "sampling": plan["sampling"],
              "actual_response_tokens": sum(row["response_token_count"] for row in rows),
              "actual_prompt_tokens": sum(row["prompt_token_count"] for row in rows),
              "arms": arms, "learning_effectiveness_checked": False,
              "whole_stream_retention_checked": False, "oracle_semantics_checked": False}
    if "comparison" in plan:
        report["comparison"] = plan["comparison"]
    if "chunk_indices" in plan:
        report.update({"chunk_indices": plan["chunk_indices"], "reading_mode": plan["reading_mode"],
                       "independent_history_count": 1, "sequential_ingest_checked": False})
    return rows, report
