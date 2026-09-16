import importlib.util
import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.environment import (
    DynamicMemoryEnvironment,
    DynamicEnvironmentError,
    canonical_dynamic_payload,
    render_public_chunk,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.schema import (
    DynamicHistoryPublic,
    DynamicPublicChunk,
    DynamicQueryPrivate,
    DynamicQueryPublic,
    PrivateEvent,
    SourceRegistryRecord,
)
from AgeMem_code_agentscope.streaming_memory.query_runner import IndependentQueryRunner
from AgeMem_code_agentscope.streaming_memory.dynamic.temporal_grounder import (
    exposed_payloads_as_memories,
    validate_memory_semantics,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    DebugLexicalTokenizer,
    TokenAccounting,
    sha256_text,
)


TEXT = "自第 1 日起，星河项目的负责人为成员甲。"


def objects():
    source = SourceRegistryRecord(
        source_ref="src",
        history_id="h",
        observed_at=0,
        text=TEXT,
        text_sha256=sha256_text(TEXT),
    )
    event = PrivateEvent(
        event_id="event",
        history_id="h",
        entity="星河项目",
        relation="project_owner",
        value="成员甲",
        observed_at=0,
        effective_at=1,
        source_ref="src",
        source_text_hash=source.text_sha256,
    )
    query = DynamicQueryPrivate(
        query_id="q",
        history_id="h",
        task_family="historical_state",
        query_kind="state_at",
        entity="星河项目",
        relation="project_owner",
        query_time=2,
        answers=("成员甲",),
        answer_available_at=0,
        support_alternatives=(("event",),),
        answer_type="entity",
        answerable=True,
    )
    return source, event, query


def history(memory=300):
    return DynamicHistoryPublic(
        history_id="h",
        history_family_id="f",
        context_budget_tokens=260,
        memory_budget_tokens=memory,
        chunks=(
            DynamicPublicChunk(
                chunk_id="c",
                observed_at=0,
                text=TEXT,
                source_refs=("src",),
                content_token_count=20,
            ),
        ),
    )


def good_action(content=TEXT):
    return {
        "type": "ADD",
        "memory_id": "m",
        "content": content,
        "source_refs": ["src"],
        "claims": [
            {
                "entity": "星河项目",
                "relation": "project_owner",
                "value": "成员甲",
                "effective_at": 1,
            }
        ],
        "custom": {"note": "visible metadata counts"},
    }


class DynamicEnvironmentGroundingTest(unittest.TestCase):
    def test_public_sentence_source_map_is_visible_and_unambiguous(self):
        chunk = history().chunks[0]
        self.assertEqual(render_public_chunk(chunk), "STREAM CHUNK\n[src] " + TEXT)
        ambiguous = chunk.model_copy(update={"source_refs": ("src", "other")})
        with self.assertRaisesRegex(DynamicEnvironmentError, "ambiguous"):
            render_public_chunk(ambiguous)

    def test_cpu_prompt_preflight_counts_source_labels_and_rejects_small_c(self):
        path = Path(__file__).resolve().parents[2] / "scripts/agemem_dynamic_v2_prompt_preflight.py"
        spec = importlib.util.spec_from_file_location("dynamic_prompt_preflight_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check_history = module.check_history
        accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())
        budget = {
            "ingest_max_new_tokens": 40, "max_decisions_per_chunk": 3,
            "answer_tail_tokens": 40, "retrieved_payload_tokens": 80,
        }
        maximum = check_history(history(), accounting, budget)
        self.assertGreater(maximum, 40)
        self.assertLessEqual(maximum, 260)
        too_small = history().model_copy(update={"context_budget_tokens": 40})
        with self.assertRaisesRegex(DynamicEnvironmentError, "satisfy C"):
            check_history(too_small, accounting, budget)

    def setUp(self):
        self.accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())
        self.source, self.event, self.query = objects()

    def env(self, memory=300, retrieval=120):
        return DynamicMemoryEnvironment(
            history(memory),
            accounting=self.accounting,
            read_rollout_id="r",
            policy_version="p",
            ingest_max_new_tokens=40,
            max_decisions_per_chunk=3,
            answer_tail_tokens=40,
            retrieval_payload_tokens=retrieval,
        )

    def score(self, memories):
        return validate_memory_semantics(
            memories,
            observed_source_refs=("src",),
            source_registry={"src": self.source},
            events_by_id={"event": self.event},
            query=self.query,
        )

    def test_correct_pointer_wrong_body_is_not_supported(self):
        env = self.env()
        env.admit_next_chunk()
        result = env.execute(good_action("泛化概述，不包含具体事实。"))
        self.assertTrue(result.admitted)
        scored = self.score(env.policy_memory())
        self.assertTrue(scored.decision.source_ok)
        self.assertFalse(scored.decision.content_ok)
        self.assertEqual(scored.utility, 0)

    def test_correct_content_with_wrong_effective_time_is_not_supported(self):
        env = self.env()
        env.admit_next_chunk()
        action = good_action()
        action["claims"][0]["effective_at"] = 2
        self.assertTrue(env.execute(action).admitted)
        scored = self.score(env.policy_memory())
        self.assertTrue(scored.decision.content_ok)
        self.assertFalse(scored.decision.temporal_ok)
        self.assertEqual(scored.utility, 0)

    def test_failed_update_and_delete_do_not_advance_revision(self):
        env = self.env()
        env.admit_next_chunk()
        env.execute(good_action())
        before = env.memory_revision
        bad = dict(good_action())
        bad.update(type="UPDATE", content="x " * 1000)
        self.assertFalse(env.execute(bad).admitted)
        self.assertEqual(env.memory_revision, before)
        self.assertFalse(
            env.execute({"type": "DELETE", "memory_id": "missing"}).admitted
        )
        self.assertEqual(env.memory_revision, before)

    def test_update_revalidates_body_and_old_revision_is_not_policy_readable(self):
        env = self.env()
        env.admit_next_chunk()
        env.execute(good_action())
        old_body = env.policy_memory()[0]["content"]
        bad = dict(good_action("泛化概述，不包含具体事实。"))
        bad["type"] = "UPDATE"
        self.assertTrue(env.execute(bad).admitted)
        self.assertEqual(self.score(env.policy_memory()).utility, 0)
        self.assertNotIn(old_body, str(env.policy_memory()))
        self.assertIn(old_body, str(env.private_audit_ledger()))

    def test_b_counts_claims_sources_and_custom_metadata(self):
        action = good_action()
        env = self.env()
        env.admit_next_chunk()
        env.execute(action)
        content_only = self.accounting.count_text(TEXT)
        self.assertGreater(env.memory_tokens(), content_only)
        payload = canonical_dynamic_payload(env.policy_memory()[0])
        for key in ("content", "source_refs", "claims", "custom"):
            self.assertIn(key, payload)

    def test_retrieval_uses_active_revision_and_truncated_payload_fails_closed(self):
        env = self.env(retrieval=8)
        env.admit_next_chunk()
        env.execute(good_action())
        result = env.execute({"type": "RETRIEVE", "memory_id": "m"})
        self.assertTrue(result.truncated)
        exposed = exposed_payloads_as_memories((result.displayed_payload,))
        self.assertEqual(self.score(exposed).utility, 0)

    def test_delete_last_representation_reduces_coverage(self):
        env = self.env()
        env.admit_next_chunk()
        env.execute(good_action())
        self.assertEqual(self.score(env.policy_memory()).utility, 1)
        env.execute({"type": "DELETE", "memory_id": "m"})
        self.assertEqual(self.score(env.policy_memory()).utility, 0)

    def test_removing_redundant_representation_keeps_coverage(self):
        env = self.env(memory=600)
        env.admit_next_chunk()
        env.execute(good_action())
        duplicate = good_action()
        duplicate["memory_id"] = "m2"
        env.execute(duplicate)
        self.assertEqual(self.score(env.policy_memory()).utility, 1)
        env.execute({"type": "DELETE", "memory_id": "m2"})
        self.assertEqual(self.score(env.policy_memory()).utility, 1)

    def test_question_and_private_labels_are_absent_from_observation(self):
        env = self.env()
        env.admit_next_chunk()
        visible = str(env.policy_observation())
        for forbidden in (
            "answers",
            "support_alternatives",
            "event_id",
            "answer_available_at",
        ):
            self.assertNotIn(forbidden, visible)

    def test_source_ref_is_not_a_raw_text_retrieval_key(self):
        env = self.env()
        env.admit_next_chunk()
        result = env.execute({"type": "RETRIEVE", "memory_id": "src"})
        self.assertFalse(result.admitted)
        self.assertNotIn(TEXT, result.message)

    def test_v1_query_runner_reuses_one_immutable_snapshot_for_independent_branches(
        self,
    ):
        env = self.env()
        env.admit_next_chunk()
        env.execute(good_action())
        env.execute({"type": "NEXT"})
        snapshot = env.finalize()
        runner = IndependentQueryRunner(
            accounting=self.accounting,
            reader=lambda _: "<answer>成员甲</answer>",
            reader_version="fixed-reader",
            context_total_tokens=260,
            answer_max_new_tokens=40,
            retrieval_payload_tokens=160,
            retrieval_top_k=2,
        )
        query = DynamicQueryPublic(
            query_id="q", history_id="h", question="负责人是谁？"
        )
        first, second = runner.run(snapshot, query, 0), runner.run(snapshot, query, 1)
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        self.assertNotEqual(first.query_branch_id, second.query_branch_id)
        self.assertEqual(snapshot.memory_sha256, env.finalize().memory_sha256)


if __name__ == "__main__":
    unittest.main()
