import hashlib
import unittest

from AgeMem_code_agentscope.streaming_memory.environment import MemorySnapshot
from AgeMem_code_agentscope.streaming_memory.query_runner import QueryBranchResult
from AgeMem_code_agentscope.streaming_memory.reward_replay import score_snapshot, state_transition
from AgeMem_code_agentscope.streaming_memory.schema import QueryGold, SourceSentenceRef


def ref(document, index, sentence):
    return SourceSentenceRef(
        source_revision="fixture", document_key=document, title=document,
        sentence_index=index, sentence_sha256=hashlib.sha256(sentence.encode()).hexdigest(),
    )


def branch(qid, answer, ids=()):
    return QueryBranchResult(
        query_branch_id=f"b-{qid}", query_id=qid, snapshot_id="s", reader_version="r",
        answer_text=f"<answer>{answer}</answer>", prompt_tokens=10,
        retrieved_memory_ids=tuple(ids), retrieved_payloads=(), tail_messages=(),
    )


class StreamingRewardTest(unittest.TestCase):
    def setUp(self):
        self.s1, self.s2 = "Alpha is in Paris.", "Paris is in France."
        self.refs = (ref("d", 0, self.s1), ref("d", 1, self.s2))
        self.gold = QueryGold(query_id="q", answer="France", support_refs=self.refs, source_question_id="private-q")
        self.registry = {
            (r.document_key, r.sentence_index, r.sentence_sha256): s
            for r, s in zip(self.refs, (self.s1, self.s2))
        }

    def snapshot(self, second_content=None):
        memories = [
            {"memory_id": "m1", "content": self.s1, "source_refs": [{"document_key": "d", "sentence_index": 0}]},
            {"memory_id": "m2", "content": second_content or self.s2, "source_refs": [{"document_key": "d", "sentence_index": 1}]},
        ]
        return MemorySnapshot("s", "e", "r", tuple(memories), (), (("d", 0), ("d", 1)), "m", "t", "p")

    def test_flat_and_dfa_are_exactly_equivalent_in_static_protocol(self):
        result = score_snapshot(
            snapshot=self.snapshot(), branch_results=[branch("q", "France", ("m1", "m2"))],
            gold_by_query={"q": self.gold}, source_sentences=self.registry, semantic_lambda=.25,
        )
        self.assertEqual(result.terminal, 1.0)
        self.assertEqual(result.flat_state, 1.25)
        self.assertEqual(result.flat_state, result.dfa)
        self.assertTrue(result.flat_dfa_equivalent)

    def test_correct_pointer_with_unrelated_body_gets_no_credit(self):
        result = score_snapshot(
            snapshot=self.snapshot("A generic overview with no fact."),
            branch_results=[branch("q", "wrong", ("m2",))], gold_by_query={"q": self.gold},
            source_sentences=self.registry, semantic_lambda=1,
        )
        self.assertEqual(result.per_query[0].retained_coverage, .5)
        self.assertEqual(result.per_query[0].exposed_coverage, 0)

    def test_multiquery_scores_are_averaged_not_summed(self):
        gold2 = self.gold.model_copy(update={"query_id": "q2"})
        result = score_snapshot(
            snapshot=self.snapshot(), branch_results=[branch("q", "France"), branch("q2", "wrong")],
            gold_by_query={"q": self.gold, "q2": gold2}, source_sentences=self.registry,
            semantic_lambda=0,
        )
        self.assertEqual(result.terminal, .5)

    def test_state_rolls_back_after_delete_or_bad_update(self):
        self.assertEqual(state_transition("absent", "valid_store"), "retained")
        self.assertEqual(state_transition("retained", "valid_exposure"), "exposed_valid")
        self.assertEqual(state_transition("exposed_valid", "delete"), "absent")
        self.assertEqual(state_transition("retained", "invalid_update"), "absent")
        self.assertEqual(state_transition("retained", "delete", True), "retained")


if __name__ == "__main__": unittest.main()
