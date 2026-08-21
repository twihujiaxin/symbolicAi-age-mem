import unittest

from AgeMem_code_agentscope.memory_store import (
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
)
from AgeMem_code_agentscope.toy_hotpotqa import (
    MemoryBudgetExceeded,
    OracleSafeStorePolicy,
    Stage1StorageBenchmark,
    StoreAllPolicy,
    StoreNonePolicy,
    TokenBudgetMemoryStore,
    ToyTaskDataset,
    count_ltm_tokens,
)


class TokenBudgetMemoryStoreTest(unittest.TestCase):
    def make_store(self, budget: int) -> TokenBudgetMemoryStore:
        return TokenBudgetMemoryStore(
            InMemoryStore("budget-rollout"),
            token_budget=budget,
        )

    @staticmethod
    def record(memory_id: str, content: str, step: int = 0) -> MemoryRecord:
        return MemoryRecord(
            memory_id=memory_id,
            content=content,
            source_rollout_id="budget-rollout",
            source_step=step,
        )

    def test_wrapper_still_implements_memory_store_protocol(self):
        self.assertIsInstance(self.make_store(10), MemoryStore)

    def test_add_over_budget_fails_closed_and_records_exact_cost(self):
        packed = self.make_store(3)
        with self.assertRaises(MemoryBudgetExceeded):
            packed.add(self.record("packed", "one two three four"))
        self.assertEqual(packed.size(), 0)
        self.assertEqual(packed.active_tokens(), 0)

        store = self.make_store(3)
        store.add(self.record("m1", "alpha beta"))
        snapshot_before = store.snapshot()

        with self.assertRaises(MemoryBudgetExceeded) as raised:
            store.add(self.record("m2", "gamma delta", step=1))

        self.assertEqual(store.snapshot(), snapshot_before)
        self.assertIsNone(store.get("m2"))
        self.assertEqual(store.active_tokens(), 2)
        event = raised.exception.event
        self.assertEqual(event.operation, "add")
        self.assertEqual(event.reason, "budget_exceeded")
        self.assertEqual(event.active_tokens_before, 2)
        self.assertEqual(event.attempted_content_tokens, 2)
        self.assertEqual(event.projected_active_tokens, 4)
        self.assertEqual(event.active_tokens_after, 2)

    def test_update_over_budget_preserves_content_and_version_history(self):
        store = self.make_store(4)
        store.add(self.record("m1", "one two", step=0))
        history_before = store.history("m1")

        with self.assertRaises(MemoryBudgetExceeded) as raised:
            store.update(
                "m1",
                content="one two three four five",
                source_step=1,
            )

        self.assertEqual(store.get("m1").content, "one two")
        self.assertEqual(store.history("m1"), history_before)
        self.assertEqual(store.get("m1").version, 1)
        event = raised.exception.event
        self.assertEqual(event.operation, "update")
        self.assertEqual(event.previous_content_tokens, 2)
        self.assertEqual(event.attempted_content_tokens, 5)
        self.assertEqual(event.projected_active_tokens, 5)
        self.assertEqual(event.version_before, 1)
        self.assertIsNone(event.version_after)

    def test_update_charges_delta_and_keeps_m2_version_semantics(self):
        store = self.make_store(5)
        store.add(self.record("m1", "one two three", step=0))
        updated = store.update("m1", content="one two", source_step=1)

        self.assertIsNotNone(updated)
        self.assertEqual(store.active_tokens(), 2)
        self.assertEqual(
            [record.status for record in store.history("m1")],
            ["superseded", "active"],
        )
        event = store.audit_log()[-1]
        self.assertEqual(event.token_delta, -1)
        self.assertEqual(event.version_before, 1)
        self.assertEqual(event.version_after, 2)

    def test_delete_frees_capacity_but_soft_delete_history_is_retained(self):
        store = self.make_store(2)
        store.add(self.record("m1", "one two"))
        store.delete("m1", source_step=1)
        store.add(self.record("m2", "three four", step=2))

        self.assertEqual(store.active_tokens(), 2)
        self.assertEqual(
            [record.status for record in store.history("m1")],
            ["superseded", "discarded"],
        )
        self.assertEqual(store.report().remaining_tokens, 0)

    def test_restore_above_budget_is_rejected_without_changing_target(self):
        source = InMemoryStore("budget-rollout")
        source.add(self.record("large", "one two three four"))
        oversized = source.snapshot()
        target = self.make_store(2)
        target.add(self.record("small", "one"))
        before = target.snapshot()

        with self.assertRaises(MemoryBudgetExceeded):
            target.restore(oversized)

        self.assertEqual(target.snapshot(), before)
        self.assertEqual(target.audit_log()[-1].operation, "restore")
        self.assertEqual(target.audit_log()[-1].reason, "budget_exceeded")

    def test_injected_counter_is_the_admission_contract(self):
        store = TokenBudgetMemoryStore(
            InMemoryStore("budget-rollout"),
            token_budget=4,
            token_counter=lambda content: len(content.encode("utf-8")),
            token_counter_name="utf8-bytes-test",
        )
        store.add(self.record("m1", "abcd"))
        with self.assertRaises(MemoryBudgetExceeded):
            store.add(self.record("m2", "e", step=1))
        self.assertEqual(store.report().token_counter, "utf8-bytes-test")

        task = ToyTaskDataset.from_json().get("toy-train-005")
        benchmark = Stage1StorageBenchmark(
            token_budget=100,
            token_counter=lambda _content: -1,
            token_counter_name="invalid-test-counter",
        )
        with self.assertRaisesRegex(
            ValueError,
            "token_counter must return a non-negative integer",
        ):
            benchmark.run(
                task,
                StoreAllPolicy(),
                rollout_id="invalid-counter-rollout",
                seed=7,
            )


class Stage1StorageBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = ToyTaskDataset.from_json().get("toy-train-005")
        cls.support_budget = sum(
            count_ltm_tokens(cls.task.fact(fact_id).sentence)
            for fact_id in cls.task.supporting_fact_ids
        )

    def run_policy(self, policy, rollout_id):
        return Stage1StorageBenchmark(token_budget=self.support_budget).run(
            self.task,
            policy,
            rollout_id=rollout_id,
            seed=7,
        )

    def test_three_policies_have_distinct_declared_information_access(self):
        store_all = StoreAllPolicy()
        store_none = StoreNonePolicy()
        oracle = OracleSafeStorePolicy()

        self.assertFalse(store_all.uses_oracle_labels)
        self.assertFalse(store_none.uses_oracle_labels)
        self.assertTrue(oracle.uses_oracle_labels)
        store_all_result = self.run_policy(store_all, "policy-access-store-all")
        store_none_result = self.run_policy(store_none, "policy-access-store-none")
        oracle_result = self.run_policy(oracle, "policy-access-oracle")
        self.assertEqual(
            store_all_result.selected_fact_ids,
            ("t005-a", "t005-d1", "t005-b"),
        )
        self.assertEqual(store_none_result.selected_fact_ids, ())
        self.assertEqual(
            oracle_result.selected_fact_ids,
            ("t005-a", "t005-b"),
        )

        class LeakageProbePolicy:
            def __init__(self, claimed_oracle_access):
                self.name = f"public-boundary-probe-{claimed_oracle_access}"
                self.uses_oracle_labels = claimed_oracle_access
                self.received = None

            def actions(self, public_input):
                self.received = public_input
                return ()

        for claimed_oracle_access in (False, True):
            with self.subTest(claimed_oracle_access=claimed_oracle_access):
                probe = LeakageProbePolicy(claimed_oracle_access)
                result = Stage1StorageBenchmark(token_budget=self.support_budget).run(
                    self.task,
                    probe,
                    rollout_id=f"public-boundary-{claimed_oracle_access}",
                    seed=7,
                )

                self.assertFalse(result.uses_oracle_labels)
                self.assertIsNotNone(probe.received)
                public_payload = probe.received.model_dump(mode="json")
                self.assertEqual(
                    set(public_payload), {"task_id", "seed", "observed_facts"}
                )
                self.assertTrue(
                    all(
                        set(fact) == {"fact_handle", "title", "sentence"}
                        for fact in public_payload["observed_facts"]
                    )
                )
                for private_field in (
                    "question",
                    "answer",
                    "supporting_fact_ids",
                    "distractor_fact_ids",
                ):
                    self.assertFalse(hasattr(probe.received, private_field))
                    with self.assertRaises(AttributeError):
                        getattr(probe.received, private_field)
                serialized_public = probe.received.model_dump_json()
                for fact in self.task.facts:
                    self.assertNotIn(fact.fact_id, serialized_public)

    def test_store_all_hits_fixed_budget_and_rejection_is_auditable(self):
        result = self.run_policy(StoreAllPolicy(), "store-all-rollout")

        self.assertEqual(result.policy, "store-all")
        self.assertEqual(
            result.selected_fact_ids,
            ("t005-a", "t005-d1", "t005-b"),
        )
        self.assertEqual(result.stored_supporting_fact_ids, ("t005-a",))
        self.assertEqual(result.stored_non_supporting_fact_ids, ("t005-d1",))
        self.assertEqual(
            [item.admitted for item in result.decisions],
            [True, True, False],
        )
        self.assertEqual(result.decisions[-1].reason, "budget_exceeded")
        self.assertEqual(result.decisions[-1].action.fact_id, "t005-b")
        self.assertEqual(result.active_tokens, self.support_budget)
        self.assertEqual(result.remaining_tokens, 0)
        self.assertEqual(
            result.audit_events[-1].active_tokens_after,
            self.support_budget,
        )

    def test_store_none_is_zero_cost_and_has_no_write_events(self):
        result = self.run_policy(StoreNonePolicy(), "store-none-rollout")

        self.assertEqual(result.policy, "store-none")
        self.assertEqual(result.selected_fact_ids, ())
        self.assertEqual(result.active_memories, ())
        self.assertEqual(result.decisions, ())
        self.assertEqual(result.audit_events, ())
        self.assertEqual(result.active_tokens, 0)
        self.assertEqual(result.remaining_tokens, self.support_budget)

    def test_oracle_safe_store_uses_minimal_support_set_within_same_budget(self):
        result = self.run_policy(
            OracleSafeStorePolicy(),
            "oracle-safe-store-rollout",
        )

        self.assertEqual(result.policy, "oracle-safe-store")
        self.assertTrue(result.uses_oracle_labels)
        self.assertEqual(result.selected_fact_ids, ("t005-a", "t005-b"))
        self.assertEqual(result.stored_supporting_fact_ids, ("t005-a", "t005-b"))
        self.assertEqual(result.stored_non_supporting_fact_ids, ())
        self.assertTrue(all(item.admitted for item in result.decisions))
        self.assertEqual(result.active_tokens, self.support_budget)
        self.assertEqual(len(result.memory_snapshot["records"]), 2)

    def test_same_inputs_produce_identical_serialized_audit(self):
        benchmark = Stage1StorageBenchmark(token_budget=self.support_budget)
        first = benchmark.run(
            self.task,
            OracleSafeStorePolicy(),
            rollout_id="deterministic-storage",
            seed=11,
        )
        second = benchmark.run(
            self.task,
            OracleSafeStorePolicy(),
            rollout_id="deterministic-storage",
            seed=11,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_distinct_rollouts_have_no_shared_state(self):
        benchmark = Stage1StorageBenchmark(token_budget=self.support_budget)
        first = benchmark.run(
            self.task,
            StoreAllPolicy(),
            rollout_id="isolated-storage-a",
            seed=2,
        )
        second = benchmark.run(
            self.task,
            StoreNonePolicy(),
            rollout_id="isolated-storage-b",
            seed=2,
        )

        self.assertTrue(first.active_memories)
        self.assertFalse(second.active_memories)
        self.assertTrue(
            all(
                record["source_rollout_id"] == "isolated-storage-a"
                for record in first.memory_snapshot["records"]
            )
        )
        self.assertEqual(second.memory_snapshot["records"], [])


if __name__ == "__main__":
    unittest.main()
