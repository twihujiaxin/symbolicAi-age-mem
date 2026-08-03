import unittest

from pydantic import ValidationError

from AgeMem_code_agentscope.action_schema.models import ActionEvent
from AgeMem_code_agentscope.memory_extraction.models import (
    ActionBinding,
    EvidenceSpan,
    TripleCandidate,
    TripleRecord,
)
from AgeMem_code_agentscope.memory_extraction.state import (
    QUARANTINE_SCHEMA_VERSION,
    STATE_DELTA_SCHEMA_VERSION,
    STATE_FACT_SCHEMA_VERSION,
    STATE_SNAPSHOT_SCHEMA_VERSION,
    STATE_TRACKER_VERSION,
    ActionPosition,
    CategorySpec,
    StateSnapshot,
    StateTracker,
    StateTrackerError,
)


def _action(
    action_id,
    *,
    rollout_id="rollout-a",
    task_id="task-a",
    timestep=0,
    assistant_turn_id=None,
    action_index_in_turn=0,
):
    turn = timestep if assistant_turn_id is None else assistant_turn_id
    return ActionEvent(
        action_id=action_id,
        task_id=task_id,
        rollout_id=rollout_id,
        stage_id=1,
        timestep=timestep,
        assistant_turn_id=turn,
        action_index_in_turn=action_index_in_turn,
        source="rule",
        action_type="Observe",
        action_text="{}",
        arguments={},
        result={},
    )


def _triple(action, subject, category, value, confidence=0.8):
    source_text = f"{subject} | {category} | {value}"
    evidence = EvidenceSpan.from_source(
        source="observation",
        source_text=source_text,
        start=0,
        end=len(source_text),
    )
    candidate = TripleCandidate.create(
        subject=subject,
        category=category,
        value=value,
        confidence=confidence,
        evidence=(evidence,),
        extractor_version="test-extractor-v1",
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
        source_texts={"observation": source_text},
    )


def _tracker(*, subjects=("Project Atlas", "Alice")):
    return StateTracker(
        categories=(
            CategorySpec(name="status", cardinality="single"),
            CategorySpec(name="member_of", cardinality="multi"),
        ),
        known_subjects=subjects,
    )


class M6StateSchemaAndQuarantineTest(unittest.TestCase):
    def test_position_and_category_contracts_are_strict(self):
        self.assertEqual(STATE_TRACKER_VERSION, "agemem.state_tracker.v1")
        self.assertEqual(STATE_FACT_SCHEMA_VERSION, "agemem.state_fact.v1")
        self.assertEqual(STATE_DELTA_SCHEMA_VERSION, "agemem.state_delta.v1")
        self.assertEqual(STATE_SNAPSHOT_SCHEMA_VERSION, "agemem.state_snapshot.v1")
        self.assertEqual(QUARANTINE_SCHEMA_VERSION, "agemem.state_quarantine.v1")
        position = ActionPosition(
            timestep=1,
            assistant_turn_id=2,
            action_index_in_turn=3,
        )
        self.assertEqual(position.key(), (1, 2, 3))
        with self.assertRaises(ValidationError):
            ActionPosition(
                timestep=-1,
                assistant_turn_id=0,
                action_index_in_turn=0,
            )
        with self.assertRaises(ValidationError):
            ActionPosition(
                timestep=0,
                assistant_turn_id=0,
                action_index_in_turn=0,
                invented=True,
            )
        with self.assertRaises(ValidationError):
            CategorySpec(name="Status", cardinality="single")
        with self.assertRaises(ValidationError):
            CategorySpec(name="status", cardinality="one")

    def test_unknown_category_subject_and_pronoun_are_quarantined(self):
        tracker = _tracker()
        action = _action("quarantine-action")
        triples = (
            _triple(action, "Project Atlas", "invented_category", "active"),
            _triple(action, "Unknown Project", "status", "active"),
            _triple(action, "It", "status", "active"),
        )
        delta = tracker.apply(action, triples)

        self.assertEqual(delta.accepted_triple_ids, ())
        self.assertEqual(
            set(delta.quarantined_triple_ids), {t.triple_id for t in triples}
        )
        self.assertEqual(
            {item.reason for item in delta.quarantine},
            {"unknown_category", "unknown_subject", "unresolved_pronoun"},
        )
        self.assertEqual(tracker.active_facts(action.rollout_id), ())
        self.assertEqual(tracker.quarantine(action.rollout_id), delta.quarantine)


class M6StateVersioningTest(unittest.TestCase):
    def test_single_value_reinforcement_then_half_open_overwrite(self):
        tracker = _tracker()
        first_action = _action("single-0", timestep=0)
        first_record = _triple(first_action, "Project Atlas", "status", "planned", 0.6)
        first = tracker.apply(first_action, (first_record,))
        original = first.inserted[0]
        self.assertEqual(original.version, 1)
        self.assertEqual(original.status, "active")

        repeat_action = _action("single-1", timestep=1)
        repeat_record = _triple(
            repeat_action, "Project Atlas", "status", "planned", 0.95
        )
        repeated = tracker.apply(repeat_action, (repeat_record,))
        reinforced = repeated.reinforced[0]
        self.assertEqual(repeated.inserted, ())
        self.assertEqual(reinforced.state_fact_id, original.state_fact_id)
        self.assertEqual(reinforced.version, 1)
        self.assertEqual(reinforced.confidence, 0.95)
        self.assertEqual(reinforced.provenance_action_ids, ("single-0", "single-1"))
        self.assertEqual(
            set(reinforced.evidence_triple_ids),
            {first_record.triple_id, repeat_record.triple_id},
        )

        update_action = _action("single-2", timestep=2)
        update_record = _triple(update_action, "Project Atlas", "status", "active", 0.9)
        updated = tracker.apply(update_action, (update_record,))
        self.assertEqual(len(updated.superseded), 1)
        self.assertEqual(len(updated.inserted), 1)
        closed = updated.superseded[0]
        current = updated.inserted[0]
        self.assertEqual(closed.state_fact_id, original.state_fact_id)
        self.assertEqual(closed.status, "superseded")
        self.assertEqual(closed.valid_from.key(), (0, 0, 0))
        self.assertEqual(closed.valid_to.key(), (2, 2, 0))
        self.assertEqual(closed.provenance_action_ids, ("single-0", "single-1"))
        self.assertEqual(current.version, 2)
        self.assertEqual(current.value, "active")
        self.assertEqual(current.valid_from, closed.valid_to)
        self.assertEqual(len(tracker.history("rollout-a")), 2)
        self.assertEqual(tracker.active_facts("rollout-a"), (current,))

    def test_multi_value_category_keeps_multiple_active_values(self):
        tracker = _tracker()
        first_action = _action("multi-0", timestep=0)
        team_a = _triple(first_action, "Alice", "member_of", "Team A", 0.7)
        team_b = _triple(first_action, "Alice", "member_of", "Team B", 0.8)
        first = tracker.apply(first_action, (team_b, team_a))

        self.assertEqual(len(first.inserted), 2)
        self.assertEqual({item.value for item in first.inserted}, {"Team A", "Team B"})
        self.assertTrue(all(item.version == 1 for item in first.inserted))
        self.assertEqual(len(tracker.active_facts("rollout-a")), 2)

        repeat_action = _action("multi-1", timestep=1)
        repeated_a = _triple(repeat_action, "Alice", "member_of", "Team A", 0.99)
        repeated = tracker.apply(repeat_action, (repeated_a,))
        self.assertEqual(len(repeated.reinforced), 1)
        self.assertEqual(repeated.reinforced[0].value, "Team A")
        self.assertEqual(repeated.reinforced[0].version, 1)
        self.assertEqual(len(tracker.active_facts("rollout-a")), 2)

    def test_same_action_single_key_conflict_is_quarantined_atomically(self):
        tracker = _tracker()
        seed_action = _action("conflict-0", timestep=0)
        tracker.apply(
            seed_action,
            (_triple(seed_action, "Project Atlas", "status", "planned"),),
        )
        before = tracker.active_facts("rollout-a")

        conflict_action = _action("conflict-1", timestep=1)
        active = _triple(conflict_action, "Project Atlas", "status", "active")
        paused = _triple(conflict_action, "Project Atlas", "status", "paused")
        delta = tracker.apply(conflict_action, (active, paused))

        self.assertEqual(delta.accepted_triple_ids, ())
        self.assertEqual(delta.inserted, ())
        self.assertEqual(delta.superseded, ())
        self.assertEqual(len(delta.quarantine), 2)
        self.assertTrue(
            all(item.reason == "same_action_conflict" for item in delta.quarantine)
        )
        self.assertEqual(tracker.active_facts("rollout-a"), before)
        self.assertEqual(len(tracker.history("rollout-a")), 1)


class M6StateIsolationAndReplayTest(unittest.TestCase):
    def test_rollouts_are_isolated_and_action_order_fails_closed(self):
        tracker = _tracker(subjects=("Project Atlas",))
        action_a = _action("isolation-a", rollout_id="rollout-a", timestep=0)
        action_b = _action("isolation-b", rollout_id="rollout-b", timestep=0)
        tracker.apply(
            action_a,
            (_triple(action_a, "Project Atlas", "status", "alpha"),),
        )
        tracker.apply(
            action_b,
            (_triple(action_b, "Project Atlas", "status", "beta"),),
        )
        self.assertEqual(tracker.active_facts("rollout-a")[0].value, "alpha")
        self.assertEqual(tracker.active_facts("rollout-b")[0].value, "beta")

        same_turn_next = _action(
            "isolation-a-next",
            rollout_id="rollout-a",
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=1,
        )
        tracker.apply(
            same_turn_next,
            (_triple(same_turn_next, "Project Atlas", "status", "gamma"),),
        )
        before_failure = tracker.snapshot("rollout-a")

        out_of_order = _action(
            "isolation-a-old",
            rollout_id="rollout-a",
            timestep=0,
            assistant_turn_id=0,
            action_index_in_turn=0,
        )
        with self.assertRaisesRegex(StateTrackerError, "positions must increase"):
            tracker.apply(
                out_of_order,
                (_triple(out_of_order, "Project Atlas", "status", "old"),),
            )
        with self.assertRaisesRegex(StateTrackerError, "duplicate action_id"):
            tracker.apply(same_turn_next, ())
        self.assertEqual(tracker.snapshot("rollout-a"), before_failure)
        self.assertEqual(tracker.active_facts("rollout-b")[0].value, "beta")

    def test_snapshot_restore_reset_and_replay_are_deterministic(self):
        categories = (
            CategorySpec(name="status", cardinality="single"),
            CategorySpec(name="member_of", cardinality="multi"),
        )
        tracker_one = StateTracker(
            categories=categories,
            known_subjects=("Project Atlas",),
        )
        tracker_two = StateTracker(
            categories=categories,
            known_subjects=("Project Atlas",),
        )
        actions = (
            _action("replay-0", timestep=0),
            _action("replay-1", timestep=1),
        )
        records = (
            _triple(actions[0], "Project Atlas", "status", "planned", 0.5),
            _triple(actions[1], "Project Atlas", "status", "active", 0.9),
        )
        for tracker in (tracker_one, tracker_two):
            tracker.apply(actions[0], (records[0],))
            tracker.apply(actions[1], (records[1],))

        first_snapshot = tracker_one.snapshot("rollout-a")
        second_snapshot = tracker_two.snapshot("rollout-a")
        self.assertEqual(first_snapshot, second_snapshot)
        self.assertEqual(first_snapshot.to_json(), second_snapshot.to_json())
        self.assertEqual(
            StateSnapshot.model_validate_json(first_snapshot.to_json()),
            first_snapshot,
        )

        other_action = _action("other-0", rollout_id="rollout-b", timestep=0)
        tracker_one.apply(
            other_action,
            (_triple(other_action, "Project Atlas", "status", "other"),),
        )
        tracker_one.reset("rollout-a")
        self.assertEqual(tracker_one.active_facts("rollout-a"), ())
        self.assertEqual(tracker_one.active_facts("rollout-b")[0].value, "other")
        tracker_one.restore(first_snapshot)
        self.assertEqual(tracker_one.snapshot("rollout-a"), first_snapshot)

        incompatible = StateTracker(
            categories=categories,
            known_subjects=("Project Atlas", "Different Subject"),
        )
        with self.assertRaisesRegex(StateTrackerError, "configuration"):
            incompatible.restore(first_snapshot)

        tracker_one.reset()
        self.assertEqual(tracker_one.active_facts("rollout-a"), ())
        self.assertEqual(tracker_one.active_facts("rollout-b"), ())


if __name__ == "__main__":
    unittest.main()
