"""Bounded sequential ingest diagnostic, not a trainer/GRPO/reader experiment."""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter

from .action_parser import parse_public_action
from .first_action_probe import build_plan, digest, make_environment
from .schema import DynamicHistoryPublic

VERSION = "agemem.dynamic.sequential_ingest_diagnostic.v1"
PROFILE = "single_action_no_example_v3"


def build_sequential_plan(history, accounting, budget, sampling):
    if len(history.chunks) < 3:
        raise ValueError("sequential diagnostic requires three public prefix chunks")
    if budget["max_decisions_per_chunk"] != 3:
        raise ValueError("diagnostic locks three decisions per chunk")
    # Reuse the validated sampling/C/B contract without inventing a new parser.
    checked = build_plan(history, accounting, budget, sampling, comparison="single_vs_no_example_v3")
    prefix = history.model_copy(update={"chunks": history.chunks[:3]})
    first = make_environment(prefix, accounting, budget, PROFILE, 0).admit_next_chunk()
    ids = list(accounting.tokenizer.apply_chat_template(first, tokenize=True, add_generation_prompt=True))
    return {"diagnostic_version": VERSION, "profile": PROFILE,
            "original_history_sha256": digest(history.model_dump(mode="json")),
            "history": prefix.model_dump(mode="json"), "prefix_indices": [0, 1, 2],
            "budget": dict(budget), "sampling": checked["sampling"], "rollout_seeds": [7, 8],
            "call_seed_rule": "rollout_seed + 1000 * global_turn_within_rollout",
            "execution_order": "serial_rollout_then_chunk_then_decision",
            "first_prompt_token_ids": ids, "model_call_upper_bound": 18,
            "response_token_upper_bound": 18 * checked["sampling"]["max_tokens"],
            "reader_call_count": 0, "optimizer_update_count": 0,
            "is_grpo_group": False, "original_full_history_coverage": False}


def validate_sequential_plan(plan, accounting):
    history = DynamicHistoryPublic.model_validate(plan["history"])
    rebuilt = build_sequential_plan(history, accounting, plan["budget"], plan["sampling"])
    for key in ("diagnostic_version", "profile", "prefix_indices", "budget", "sampling",
                "rollout_seeds", "call_seed_rule", "execution_order", "first_prompt_token_ids",
                "model_call_upper_bound", "response_token_upper_bound", "is_grpo_group",
                "reader_call_count", "optimizer_update_count", "original_full_history_coverage"):
        if plan[key] != rebuilt[key]:
            raise ValueError(f"sequential plan changed: {key}")
    if len(history.chunks) != 3:
        raise ValueError("only three contiguous prefix chunks permitted")
    return history


def execute_sequential_plan(plan, accounting, generate, on_action=None):
    history = validate_sequential_plan(plan, accounting)
    rows, snapshots = [], []
    for rollout_index, seed in enumerate(plan["rollout_seeds"]):
        env = make_environment(history, accounting, plan["budget"], PROFILE, rollout_index)
        turn = 0
        for chunk_index, chunk in enumerate(history.chunks):
            messages = env.admit_next_chunk()
            for decision in range(int(plan["budget"]["max_decisions_per_chunk"])):
                prompt_ids = list(accounting.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True))
                accounting.enforce_context(messages, max_new_tokens=plan["sampling"]["max_tokens"],
                                           context_total_tokens=history.context_budget_tokens)
                params = {**plan["sampling"], "seed": seed + 1000 * turn}
                output = generate(prompt_ids, params)
                tokens = output["response_token_ids"]
                if output["prompt_token_ids"] != prompt_ids:
                    raise ValueError("actual sequential prompt IDs differ")
                if len(tokens) > params["max_tokens"] or len(prompt_ids) + len(tokens) > history.context_budget_tokens:
                    raise ValueError("actual sequential call exceeds token budget")
                parsed = parse_public_action(output["response_text"])
                action = ({**parsed.arguments, "type": parsed.name} if parsed.code == "ok" else
                          {"type": "<invalid_tool_call>", "format_error": parsed.code})
                result = env.execute(action, response_text=output["response_text"])
                row = {"rollout_index": rollout_index, "chunk_index": chunk_index, "chunk_id": chunk.chunk_id,
                       "turn": turn, "decision_index": decision, "sampling": params,
                       "prompt_messages": messages, "prompt_token_ids": prompt_ids,
                       "response_text": output["response_text"], "response_token_ids": tokens,
                       "finish_reason": output.get("finish_reason"), "parse_code": parsed.code,
                       "action_name": parsed.name, "arguments": parsed.arguments,
                       "action_result": asdict(result), "max_token_hit": len(tokens) == params["max_tokens"],
                       "memory_tokens": env.memory_tokens()}
                if row["memory_tokens"] > history.memory_budget_tokens:
                    raise ValueError("actual sequential memory exceeds B")
                rows.append(row)
                if on_action:
                    on_action(row)
                turn += 1
                if parsed.name == "NEXT" and result.admitted:
                    break
                if decision == int(plan["budget"]["max_decisions_per_chunk"]) - 1:
                    env.commit_chunk()  # external checkpoint, not a fabricated policy NEXT
                else:
                    messages = list(env.current_messages())
        snapshots.append({"rollout_index": rollout_index, "snapshot": asdict(env.finalize()),
                          "checkpoints": [asdict(c) for c in env.checkpoints]})
    report = {"diagnostic_version": VERSION, "profile": PROFILE, "model_call_count": len(rows),
              "actual_prompt_tokens": sum(len(r["prompt_token_ids"]) for r in rows),
              "actual_response_tokens": sum(len(r["response_token_ids"]) for r in rows),
              "parse_codes": dict(Counter(r["parse_code"] for r in rows)),
              "actions": dict(Counter(r["action_name"] for r in rows)),
              "admitted_writes": sum(r["action_name"] in {"ADD", "UPDATE"} and r["action_result"]["admitted"] for r in rows),
              "max_token_hit_count": sum(r["max_token_hit"] for r in rows),
              "snapshot_count": len(snapshots), "reader_call_count": 0, "optimizer_update_count": 0,
              "is_grpo_group": False, "experience_action_contract_checked": False,
              "task_performance_checked": False, "semantic_manual_review_required": True,
              "learning_effectiveness_checked": False, "original_full_history_coverage": False}
    return rows, snapshots, report
