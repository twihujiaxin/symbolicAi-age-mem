import unittest

from AgeMem_code_agentscope.streaming_memory.environment import (
    StreamingEnvironmentError,
    StreamingMemoryEnvironment,
)
from AgeMem_code_agentscope.streaming_memory.query_runner import IndependentQueryRunner
from AgeMem_code_agentscope.streaming_memory.schema import (
    PublicChunk, PublicSourceSpan, QueryRecord, StreamEpisodePublic,
)
from AgeMem_code_agentscope.streaming_memory.token_budget import (
    DebugLexicalTokenizer, TokenAccounting, canonical_memory_payload,
)


def episode(context=180, memory=70):
    sources = lambda key: (PublicSourceSpan(
        document_key=key, title=key, sentence_start=0, sentence_end=1,
        document_sha256="a" * 64,
    ),)
    return StreamEpisodePublic(
        episode_id="episode", history_family_id="history",
        chunks=(
            PublicChunk(chunk_id="c0", text="Alpha is in Paris. Paris is in France.", sources=sources("d0"), content_token_count=10),
            PublicChunk(chunk_id="c1", text="Beta is elsewhere. More distractor text.", sources=sources("d1"), content_token_count=9),
        ), context_budget_tokens=context, memory_budget_tokens=memory,
    )


class StreamingEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.accounting = TokenAccounting.from_tokenizer(DebugLexicalTokenizer())

    def env(self, **kwargs):
        return StreamingMemoryEnvironment(
            episode(**kwargs), accounting=self.accounting, read_rollout_id="r0",
            policy_version="model_version:0", ingest_max_new_tokens=24,
            decisions_per_chunk=2, answer_tail_tokens=30,
        )

    def test_question_is_unrepresentable_in_ingest_object(self):
        env = self.env(); messages = env.admit_next_chunk()
        rendered = self.accounting.render_chat(messages)
        self.assertNotIn("What is", rendered)
        self.assertNotIn("answer", rendered.casefold())

    def test_source_must_be_observed(self):
        env = self.env(); env.admit_next_chunk()
        result = env.execute({"type": "ADD", "memory_id": "m", "content": "secret", "source_refs": [{"document_key": "future", "sentence_index": 0}]})
        self.assertFalse(result.admitted)
        self.assertEqual(env.memories, {})

    def test_b_counts_metadata_and_failed_update_is_atomic(self):
        env = self.env(memory=100); env.admit_next_chunk()
        added = env.execute({"type": "ADD", "memory_id": "m", "content": "Alpha is in Paris.", "title": "Alpha", "tags": ["fact"], "custom": {"note": "x"}, "source_refs": [{"document_key": "d0", "sentence_index": 0}]})
        self.assertTrue(added.admitted)
        before = dict(env.memories)
        update = env.execute({"type": "UPDATE", "memory_id": "m", "content": "word " * 80, "title": "huge", "source_refs": [{"document_key": "d0", "sentence_index": 0}]})
        self.assertFalse(update.admitted)
        self.assertEqual(env.memories, before)
        content_only = self.accounting.count_text(env.memories["m"]["content"])
        self.assertGreater(env.memory_tokens(), content_only)

    def test_every_call_and_prompt_satisfies_c_with_fifo(self):
        env = self.env(context=120)
        for _ in episode().chunks:
            messages = env.admit_next_chunk()
            self.assertLessEqual(self.accounting.count_chat(messages) + 24, 120)
            result = env.execute({"type": "NEXT"})
            self.assertLessEqual(result.context_tokens + 24, 120)
        self.assertEqual(len(env.observed_sources), 4)

    def test_snapshot_is_final_and_branches_are_independent(self):
        env = self.env()
        for i, chunk in enumerate(episode().chunks):
            env.admit_next_chunk()
            env.execute({"type": "ADD", "memory_id": f"m{i}", "content": chunk.text, "source_refs": [{"document_key": chunk.sources[0].document_key, "sentence_index": 0}]})
        snapshot = env.finalize()
        runner = IndependentQueryRunner(
            accounting=self.accounting, reader=lambda _: "<answer>France</answer>", reader_version="reader0",
            context_total_tokens=180, answer_max_new_tokens=24, retrieval_payload_tokens=70, retrieval_top_k=2,
        )
        query = QueryRecord(query_id="q", episode_id="episode", question="Where is Alpha?")
        a, b = runner.run(snapshot, query, 0), runner.run(snapshot, query, 1)
        self.assertNotEqual(a.query_branch_id, b.query_branch_id)
        self.assertEqual(a.snapshot_id, b.snapshot_id)
        self.assertTrue(a.actor_loss_masked and b.actor_loss_masked)

    def test_canonical_payload_includes_every_recoverable_field(self):
        value = canonical_memory_payload({"memory_id": "opaque", "content": "c", "title": "t", "tags": ["x"], "source_refs": [{"document_key": "d", "sentence_index": 0}], "custom": {"k": "v"}})
        self.assertNotIn("opaque", value)
        for expected in ("content", "title", "tags", "source_refs", "custom"):
            self.assertIn(expected, value)

    def test_snapshot_before_end_fails_closed(self):
        env = self.env(); env.admit_next_chunk()
        with self.assertRaises(StreamingEnvironmentError): env.finalize()


if __name__ == "__main__": unittest.main()
