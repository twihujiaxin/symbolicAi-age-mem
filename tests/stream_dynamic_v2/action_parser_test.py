import importlib.util
import json
import unittest
from pathlib import Path

from AgeMem_code_agentscope.streaming_memory.dynamic.action_parser import (
    format_error_feedback, parse_public_action,
)
from AgeMem_code_agentscope.streaming_memory.dynamic.environment import DYNAMIC_INGEST_SYSTEM


class PublicActionParserTest(unittest.TestCase):
    def test_retry3_response_shapes_are_rejected_without_alias_or_repair(self):
        # Independent public fixtures reproduce observed envelopes, not private
        # audit data or a claim that their content/source semantics are gold.
        args = {"memory_id": "1", "content": "fixture fact", "source_refs": ["fixture-source"]}
        self.assertEqual(parse_public_action(json.dumps({"ACTION": "ADD", "arguments": args})).code,
                         "not_array")
        self.assertEqual(parse_public_action(json.dumps([{"name": "ACTION", "arguments": {
            "type": "ADD", **args}}])).code, "unknown_action")
        self.assertEqual(parse_public_action(json.dumps([{"name": "ADD", "arguments": {
            "type": "NEXT", **args}}])).code, "reserved_type_argument")

    def test_every_allowed_name_and_original_arguments_survive(self):
        for name in ("ADD", "UPDATE", "DELETE", "RETRIEVE", "NEXT"):
            args = {} if name == "NEXT" else {"memory_id": "m"}
            text = json.dumps([{"name": name, "arguments": args}])
            for response in (text, "<tool_call>" + text + "</tool_call>"):
                parsed = parse_public_action(response)
                self.assertEqual((parsed.name, parsed.arguments, parsed.code), (name, args, "ok"))

    def test_duplicate_keys_nonfinite_batches_and_bad_envelopes_fail_closed(self):
        invalid = (
            '[{"name":"NEXT","name":"ADD","arguments":{}}]',
            '[{"name":"ADD","arguments":{"memory_id":"a","memory_id":"b"}}]',
            '[{"name":"ADD","arguments":{"custom":{"x":NaN}}}]',
            '[{"name":"NEXT","arguments":{}},]', '[]',
            '[{"name":"NEXT","arguments":{}},{"name":"NEXT","arguments":{}}]',
            '[{"name":"NEXT"}]', '[{"name":"NEXT","arguments":{},"type":"ADD"}]',
            '[{"name":"NEXT","arguments":{"memory_id":"m"}}]',
            '[{"name":"NEXT","arguments":{}}',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertNotEqual(parse_public_action(text).code, "ok")

    def test_feedback_is_specific_and_system_has_executable_example(self):
        self.assertIn('not a single object', format_error_feedback("not_array"))
        self.assertIn('Remove arguments.type', format_error_feedback("reserved_type_argument"))
        self.assertNotIn('"name":"ACTION"', DYNAMIC_INGEST_SYSTEM)
        self.assertIn('[{"name":"NEXT","arguments":{}}]', DYNAMIC_INGEST_SYSTEM)

    def test_offline_retry3_shape_counts_do_not_execute_or_modify_rows(self):
        path = Path(__file__).resolve().parents[2] / "scripts/agemem_dynamic_v2_response_audit.py"
        spec = importlib.util.spec_from_file_location("dynamic_response_audit_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [{"response_text": '{"ACTION":"ADD","arguments":{}}'} for _ in range(203)]
        rows.append({"response_text": '[{"name":"ACTION","arguments":{"type":"ADD"}}]'})
        before = json.dumps(rows)
        report = module.classify_rows(rows)
        self.assertEqual(report["strict_parse_codes"], {"not_array": 203, "unknown_action": 1})
        self.assertEqual(report["responses_with_legacy_type_override"], 1)
        self.assertFalse(report["action_execution"])
        self.assertEqual(json.dumps(rows), before)
