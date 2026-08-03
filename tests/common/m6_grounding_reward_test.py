import unittest

from AgeMem_code_agentscope.action_schema.models import (
    ActionEvent,
    TrajectoryStepV2,
)
from AgeMem_code_agentscope.memory_extraction.grounding import (
    EvidenceDigestRelevanceResolver,
    ExtractedAPGrounder,
    GroundedAction,
    MemoryDelta,
)
from AgeMem_code_agentscope.memory_extraction.models import (
    APRecord,
    ActionBinding,
    EvidenceSpan,
    TripleCandidate,
    TripleRecord,
    text_digest,
)
from AgeMem_code_agentscope.memory_extraction.rewarding import (
    DEFAULT_EXTRACTED_REWARD_VERSION,
    EXTRACTED_REPLAY_SCHEMA_VERSION,
    EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION,
    ExtractedRewardReplay,
    ExtractedRewardReplayError,
)
from AgeMem_code_agentscope.memory_extraction.state import (
    ActionPosition,
    CategorySpec,
    StateDelta,
    StateTracker,
)
from AgeMem_code_agentscope.memory_oracle.models import AP_ORDER
from AgeMem_code_agentscope.trajectory import MemorySnapshotItem


TASK_ID = "m6-grounding-task"
ROLLOUT_ID = "m6-grounding-rollout"
OLD_SENTENCE = "Project Atlas has status planned."
NEW_SENTENCE = "Project Atlas has status active."


def _action(
    action_id,
    *,
    timestep,
    action_type,
    rollout_id=ROLLOUT_ID,
    result_text="",
    private_metadata=None,
):
    return ActionEvent(
        action_id=action_id,
        task_id=TASK_ID,
        rollout_id=rollout_id,
        stage_id=1,
        timestep=timestep,
        assistant_turn_id=timestep,
        action_index_in_turn=0,
        source="rule",
        action_type=action_type,
        action_text="{}",
        arguments={},
        result={
            "content": (
                ({"type": "text", "text": result_text},) if result_text else ()
            ),
            "metadata": private_metadata or {},
            "is_interrupted": False,
        },
    )


def _memory(
    content,
    *,
    version=1,
    status="active",
    private_metadata=None,
):
    return MemorySnapshotItem(
        memory_id="memory-project-status",
        content=content,
        metadata=private_metadata or {},
        version=version,
        status=status,
        source_rollout_id=ROLLOUT_ID,
        source_step=0,
    )


def _step(action, *, before=(), after=(), env_reward=0.0, done=False):
    return TrajectoryStepV2(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        observation="public observation",
        actions=(action,),
        memory_before=tuple(before),
        memory_after=tuple(after),
        env_reward=env_reward,
        done=done,
    )


def _triple(action, sentence, *, value, confidence=0.9):
    evidence = EvidenceSpan.from_source(
        source="observation",
        source_text=sentence,
        start=0,
        end=len(sentence),
    )
    candidate = TripleCandidate.create(
        subject="Project Atlas",
        category="status",
        value=value,
        confidence=confidence,
        evidence=(evidence,),
        extractor_version="agemem.test_extractor.v1",
        extractor_kind="mock",
        model_version="deterministic-fixture-v1",
    )
    binding = ActionBinding(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        action_id=action.action_id,
        assistant_turn_id=action.assistant_turn_id,
        action_index_in_turn=action.action_index_in_turn,
    )
    return TripleRecord.from_candidate(
        candidate,
        binding,
        source_texts={"observation": sentence},
    )


def _state_tracker():
    return StateTracker(
        categories=(CategorySpec(name="status", cardinality="single"),),
        known_subjects=("Project Atlas",),
    )


def _grounder(*sentences):
    roles = {text_digest(sentence): "relevant" for sentence in sentences}
    return ExtractedAPGrounder(
        relevance_resolver=EvidenceDigestRelevanceResolver(roles),
        required_relevant_digests=(text_digest(sentences[-1]),),
    )


def _propositions(grounded):
    return tuple(item.proposition for item in grounded.atomic_propositions)


def _empty_grounded_pair(
    timestep,
    propositions=(),
    *,
    action_type="Noop",
    done=False,
    env_reward=0.0,
    rollout_id="reward-rollout",
):
    action = _action(
        f"{rollout_id}:action:{timestep}",
        timestep=timestep,
        action_type=action_type,
        rollout_id=rollout_id,
    )
    step = _step(action, env_reward=env_reward, done=done)
    position = ActionPosition.from_action(action)
    state_delta = StateDelta(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        action_id=action.action_id,
        position=position,
    )
    memory_delta = MemoryDelta(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        action_id=action.action_id,
    )
    ordered = tuple(ap for ap in AP_ORDER if ap in set(propositions))
    records = tuple(
        APRecord.create(
            task_id=action.task_id,
            rollout_id=action.rollout_id,
            stage_id=action.stage_id,
            timestep=action.timestep,
            action_id=action.action_id,
            proposition=proposition,
            confidence=1.0,
            grounder_version="agemem.test_grounder.v1",
        )
        for proposition in ordered
    )
    grounded = GroundedAction(
        task_id=action.task_id,
        rollout_id=action.rollout_id,
        stage_id=action.stage_id,
        timestep=action.timestep,
        action_id=action.action_id,
        memory_delta=memory_delta,
        state_delta=state_delta,
        atomic_propositions=records,
    )
    return step, grounded


class M6ExtractedAPGroundingTest(unittest.TestCase):
    def test_bare_add_and_retrieve_do_not_ground_semantic_APs(self):
        tracker = _state_tracker()
        grounder = _grounder(NEW_SENTENCE)

        bare_add = _action("bare-add", timestep=0, action_type="Add_memory")
        add_step = _step(bare_add)
        add_state = tracker.apply(bare_add, ())
        add_grounded = grounder.ground(
            step=add_step,
            action=bare_add,
            triples=(),
            state_delta=add_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
        )
        self.assertEqual(add_grounded.atomic_propositions, ())

        stored = _memory(NEW_SENTENCE)
        real_add = _action(
            "real-add",
            timestep=1,
            action_type="Add_memory",
            result_text=f"Stored: {NEW_SENTENCE}",
        )
        real_triple = _triple(real_add, NEW_SENTENCE, value="active")
        real_state = tracker.apply(real_add, (real_triple,))
        history = {real_triple.triple_id: real_triple}
        real_grounded = grounder.ground(
            step=_step(real_add, after=(stored,)),
            action=real_add,
            triples=(real_triple,),
            state_delta=real_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        self.assertIn("stored_supporting_fact", _propositions(real_grounded))
        self.assertNotIn("supporting_coverage_complete", _propositions(real_grounded))

        bare_retrieve = _action(
            "bare-retrieve",
            timestep=2,
            action_type="Retrieve_memory",
        )
        retrieve_state = tracker.apply(bare_retrieve, ())
        retrieve_grounded = grounder.ground(
            step=_step(bare_retrieve, before=(stored,), after=(stored,)),
            action=bare_retrieve,
            triples=(),
            state_delta=retrieve_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        self.assertNotIn("retrieved_supporting_fact", _propositions(retrieve_grounded))

    def test_private_oracle_and_role_poison_do_not_change_grounding(self):
        tracker = _state_tracker()
        grounder = _grounder(NEW_SENTENCE)
        clean_action = _action(
            "poison-invariant",
            timestep=0,
            action_type="Add_memory",
            result_text=f"Stored: {NEW_SENTENCE}",
            private_metadata={"oracle_labels": {"answer_correct": True}},
        )
        poisoned_action = _action(
            "poison-invariant",
            timestep=0,
            action_type="Add_memory",
            result_text=f"Stored: {NEW_SENTENCE}",
            private_metadata={
                "oracle_labels": {"answer_correct": False},
                "role": "irrelevant",
            },
        )
        triple = _triple(clean_action, NEW_SENTENCE, value="active")
        state_delta = tracker.apply(clean_action, (triple,))
        clean_memory = _memory(
            NEW_SENTENCE,
            private_metadata={"role": "supporting", "oracle": True},
        )
        poisoned_memory = _memory(
            NEW_SENTENCE,
            private_metadata={"role": "irrelevant", "oracle": False},
        )

        clean = grounder.ground(
            step=_step(clean_action, after=(clean_memory,)),
            action=clean_action,
            triples=(triple,),
            state_delta=state_delta,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
        )
        poisoned = grounder.ground(
            step=_step(poisoned_action, after=(poisoned_memory,)),
            action=poisoned_action,
            triples=(triple,),
            state_delta=state_delta,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
        )
        self.assertEqual(clean, poisoned)

    def test_update_overwrite_and_soft_delete_are_grounded_from_real_deltas(self):
        tracker = _state_tracker()
        grounder = _grounder(OLD_SENTENCE, NEW_SENTENCE)

        add = _action(
            "update-add",
            timestep=0,
            action_type="Add_memory",
            result_text=f"Stored: {OLD_SENTENCE}",
        )
        old_triple = _triple(add, OLD_SENTENCE, value="planned")
        add_state = tracker.apply(add, (old_triple,))
        history = {old_triple.triple_id: old_triple}
        old_active = _memory(OLD_SENTENCE, version=1, status="active")
        grounder.ground(
            step=_step(add, after=(old_active,)),
            action=add,
            triples=(old_triple,),
            state_delta=add_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )

        update = _action(
            "update-real",
            timestep=1,
            action_type="Update_memory",
            result_text=f"Updated: {NEW_SENTENCE}",
        )
        new_triple = _triple(update, NEW_SENTENCE, value="active")
        update_state = tracker.apply(update, (new_triple,))
        history[new_triple.triple_id] = new_triple
        old_closed = _memory(OLD_SENTENCE, version=1, status="superseded")
        new_active = _memory(NEW_SENTENCE, version=2, status="active")
        updated = grounder.ground(
            step=_step(
                update,
                before=(old_active,),
                after=(old_closed, new_active),
            ),
            action=update,
            triples=(new_triple,),
            state_delta=update_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        self.assertIn("updated_stale_fact", _propositions(updated))
        self.assertEqual(len(update_state.superseded), 1)
        self.assertEqual(update_state.superseded[0].valid_to, update_state.position)

        delete = _action(
            "delete-real",
            timestep=2,
            action_type="Delete_memory",
            result_text="Soft deleted memory.",
        )
        delete_triple = _triple(delete, NEW_SENTENCE, value="active")
        delete_state = tracker.apply(delete, (delete_triple,))
        history[delete_triple.triple_id] = delete_triple
        new_discarded = _memory(NEW_SENTENCE, version=2, status="discarded")
        deleted = grounder.ground(
            step=_step(
                delete,
                before=(old_closed, new_active),
                after=(old_closed, new_discarded),
            ),
            action=delete,
            triples=(delete_triple,),
            state_delta=delete_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        self.assertIn("deleted_supporting_fact", _propositions(deleted))


class M6ExtractedRewardReplayTest(unittest.TestCase):
    def setUp(self):
        self.replay = ExtractedRewardReplay.from_config(
            "terminal_dfa",
            extractor_version="agemem.test_extractor.v1",
        )

    def test_m4_formula_double_edge_credit_join_and_jsonl_are_deterministic(self):
        pairs = (
            _empty_grounded_pair(
                0,
                ("stored_supporting_fact",),
                action_type="Add_memory",
                rollout_id="reward-success",
            ),
            _empty_grounded_pair(
                1,
                (
                    "retrieved_supporting_fact",
                    "supporting_coverage_complete",
                ),
                action_type="Retrieve_memory",
                rollout_id="reward-success",
            ),
            _empty_grounded_pair(
                2,
                ("answered_correctly",),
                action_type="Answer",
                done=True,
                env_reward=1.0,
                rollout_id="reward-success",
            ),
        )
        first = self.replay.replay(pairs, seed=17)
        second = self.replay.replay(pairs, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.to_jsonl().encode(), second.to_jsonl().encode())
        self.assertTrue(first.accepted)
        self.assertEqual(first.env_total, 1.0)
        self.assertEqual(first.milestone_total, 1.0)
        self.assertEqual(first.total_reward, 2.0)
        self.assertEqual(first.reward_version, DEFAULT_EXTRACTED_REWARD_VERSION)
        self.assertEqual(first.extractor_version, "agemem.test_extractor.v1")
        self.assertEqual(first.schema_version, EXTRACTED_REPLAY_SCHEMA_VERSION)
        self.assertEqual(
            first.actions[0].schema_version,
            EXTRACTED_REWARDED_ACTION_SCHEMA_VERSION,
        )

        credits = [item.credit for item in first.actions]
        self.assertEqual(len({item.action_id for item in credits}), 3)
        double = credits[1]
        self.assertEqual(
            double.transition_ids,
            ("progress_support_coverage", "progress_retrieve_support"),
        )
        self.assertIsNone(double.transition_id)
        self.assertTrue(
            all(
                evidence == (ap.ap_id,)
                for ap in first.actions[1].atomic_propositions
                for evidence in (double.atomic_proposition_evidence[ap.proposition],)
            )
        )
        self.assertTrue(
            all(
                item.reward_breakdown.cost == 0.0
                and item.return_to_go is None
                and item.advantage is None
                for item in credits
            )
        )

    def test_repeated_add_retrieve_and_loop_cannot_farm_milestones(self):
        tracker = _state_tracker()
        grounder = _grounder(NEW_SENTENCE)
        stored = _memory(NEW_SENTENCE)
        pairs = []
        history = {}

        add = _action(
            "loop-add",
            timestep=0,
            action_type="Add_memory",
            result_text=f"Stored: {NEW_SENTENCE}",
        )
        add_triple = _triple(add, NEW_SENTENCE, value="active")
        add_state = tracker.apply(add, (add_triple,))
        history[add_triple.triple_id] = add_triple
        add_grounded = grounder.ground(
            step=_step(add, after=(stored,)),
            action=add,
            triples=(add_triple,),
            state_delta=add_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        pairs.append((_step(add, after=(stored,)), add_grounded))

        repeated_add = _action(
            "loop-add-repeat",
            timestep=1,
            action_type="Add_memory",
            result_text=f"Already stored: {NEW_SENTENCE}",
        )
        repeated_add_triple = _triple(repeated_add, NEW_SENTENCE, value="active")
        repeated_add_state = tracker.apply(repeated_add, (repeated_add_triple,))
        history[repeated_add_triple.triple_id] = repeated_add_triple
        repeated_add_grounded = grounder.ground(
            step=_step(repeated_add, before=(stored,), after=(stored,)),
            action=repeated_add,
            triples=(repeated_add_triple,),
            state_delta=repeated_add_state,
            active_state_facts=tracker.active_facts(ROLLOUT_ID),
            state_triple_history=history,
        )
        self.assertNotIn("stored_supporting_fact", _propositions(repeated_add_grounded))
        pairs.append(
            (
                _step(repeated_add, before=(stored,), after=(stored,)),
                repeated_add_grounded,
            )
        )

        for timestep, action_id in ((2, "loop-retrieve"), (3, "loop-retrieve-repeat")):
            retrieve = _action(
                action_id,
                timestep=timestep,
                action_type="Retrieve_memory",
                result_text=(
                    f"Retrieved: {NEW_SENTENCE} (Memory ID: memory-project-status)"
                ),
            )
            retrieve_triple = _triple(retrieve, NEW_SENTENCE, value="active")
            retrieve_state = tracker.apply(retrieve, (retrieve_triple,))
            history[retrieve_triple.triple_id] = retrieve_triple
            retrieve_step = _step(retrieve, before=(stored,), after=(stored,))
            retrieve_grounded = grounder.ground(
                step=retrieve_step,
                action=retrieve,
                triples=(retrieve_triple,),
                state_delta=retrieve_state,
                active_state_facts=tracker.active_facts(ROLLOUT_ID),
                state_triple_history=history,
            )
            self.assertIn("retrieved_supporting_fact", _propositions(retrieve_grounded))
            self.assertIn(
                "supporting_coverage_complete", _propositions(retrieve_grounded)
            )
            pairs.append((retrieve_step, retrieve_grounded))

        pairs.extend(
            _empty_grounded_pair(
                timestep,
                rollout_id=ROLLOUT_ID,
            )
            for timestep in range(4, 12)
        )
        result = self.replay.replay(tuple(pairs), seed=3)

        self.assertEqual(result.final_status, "timed_out")
        self.assertFalse(result.accepted)
        self.assertEqual(result.milestone_total, 0.75)
        self.assertEqual(result.total_reward, 0.75)
        self.assertEqual(result.actions[0].credit.reward_breakdown.milestone, 0.25)
        self.assertEqual(result.actions[1].credit.reward_breakdown.milestone, 0.0)
        self.assertEqual(result.actions[2].credit.reward_breakdown.milestone, 0.5)
        self.assertEqual(result.actions[3].credit.reward_breakdown.milestone, 0.0)
        self.assertTrue(
            all(
                item.credit.reward_breakdown.milestone == 0.0
                for item in result.actions[4:]
            )
        )

    def test_action_grounded_join_and_order_fail_closed(self):
        first = _empty_grounded_pair(0, rollout_id="join-rollout")
        second = _empty_grounded_pair(1, rollout_id="join-rollout")
        with self.assertRaisesRegex(ExtractedRewardReplayError, "identity mismatch"):
            self.replay.replay(((first[0], second[1]),), seed=1)
        with self.assertRaisesRegex(ExtractedRewardReplayError, "strictly ordered"):
            self.replay.replay((second, first), seed=1)
        with self.assertRaisesRegex(ExtractedRewardReplayError, "duplicate action_id"):
            self.replay.replay((first, first), seed=1)


if __name__ == "__main__":
    unittest.main()
