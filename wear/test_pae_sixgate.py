"""Unit tests for the transcript, prompt and verdict handling of pae_sixgate.py (no judge calls).

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_pae_sixgate.py
"""
import unittest

from pae_sixgate import (
    GATES, SYS_TEMPLATE, build_groundtruth, build_transcript,
    build_user_prompt, parse_verdict, tau2_utility,
)


class TestTranscript(unittest.TestCase):
    def test_full_untruncated_tool_result_and_turns(self):
        sim = {"messages": [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "change my order"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "get_order_details",
                             "arguments": {"order_id": "#W1"}, "requestor": "assistant"}]},
            {"role": "tool", "id": "c1", "content": "X" * 5000, "error": False},
            {"role": "tool", "id": "c2", "content": "boom", "error": True},
        ]}
        t = build_transcript(sim)
        self.assertIn("USER: change my order", t)
        self.assertIn('AGENT_TOOL_CALL[id=None]: get_order_details({"order_id": "#W1"})', t)
        self.assertIn("X" * 5000, t)
        self.assertIn("TOOL_RESULT[id=c2] (ERROR): boom", t)
        self.assertIn("TOOL_RESULT[id=c1]", t)
        self.assertIn("[turn 1]", t)

    def test_user_requestor_tool_call_labelled(self):
        sim = {"messages": [
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "x", "arguments": {}, "requestor": "user"}]},
        ]}
        self.assertIn("USER_TOOL_CALL", build_transcript(sim))


class TestPrompts(unittest.TestCase):
    def test_six_gates_and_schema(self):
        self.assertEqual(
            set(GATES), {"i_pc", "i_pf", "i_ec", "i_df", "i_intent", "i_qf"})
        s = SYS_TEMPLATE.format(gate=GATES["i_ec"])
        self.assertIn("Execution Consistency", s)
        self.assertIn('"pass"', s)

    def test_groundtruth_contains_gold_and_scenario(self):
        task = {"user_scenario": {"instructions": {"reason_for_call": "trip to Dallas"}},
                "evaluation_criteria": {"actions": [
                    {"name": "update_reservation_baggages",
                     "arguments": {"reservation_id": "ZZ0001"}}]}}
        gt = build_groundtruth(task)
        self.assertIn("trip to Dallas", gt)
        self.assertIn("ZZ0001", gt)

    def test_groundtruth_scenario_mode_excludes_gold_ids(self):
        task = {"user_scenario": {"instructions": {"reason_for_call": "trip to Dallas"}},
                "evaluation_criteria": {"actions": [
                    {"name": "w", "arguments": {"reservation_id": "ZZ0001"}}]}}
        gt = build_groundtruth(task, mode="scenario")
        self.assertIn("trip to Dallas", gt)
        self.assertNotIn("ZZ0001", gt)

    def test_user_prompt_sections(self):
        up = build_user_prompt("POLICY-TEXT", "GT-TEXT", "TR-TEXT")
        self.assertLess(up.index("POLICY-TEXT"), up.index("GT-TEXT"))
        self.assertLess(up.index("GT-TEXT"), up.index("TR-TEXT"))
        up2 = build_user_prompt("P", None, "T")
        self.assertNotIn("TASK GROUNDTRUTH", up2)


class TestVerdict(unittest.TestCase):
    def test_strict_json(self):
        v = parse_verdict('{"pass": true, "violations": []}')
        self.assertIs(v["pass"], True)
        self.assertEqual(v["violations"], [])

    def test_fenced_json(self):
        v = parse_verdict('blah\n{"pass": false, "violations": [{"turn": 3, '
                          '"description": "claimed without executing"}]}\nend')
        self.assertIs(v["pass"], False)
        self.assertEqual(v["violations"][0]["turn"], 3)

    def test_garbage_is_none_not_pass(self):
        v = parse_verdict("I think it looks fine")
        self.assertIsNone(v["pass"])

    def test_missing_pass_key_is_none(self):
        v = parse_verdict('{"violations": []}')
        self.assertIsNone(v["pass"])

    def test_string_false_is_fail_not_pass(self):
        self.assertIs(parse_verdict('{"pass": "false", "violations": []}')["pass"], False)
        self.assertIs(parse_verdict('{"pass": "true", "violations": []}')["pass"], True)

    def test_nonbool_garbage_pass_value_is_none(self):
        self.assertIsNone(parse_verdict('{"pass": "maybe"}')["pass"])
        self.assertIsNone(parse_verdict('{"pass": 1}')["pass"])


class TestUtility(unittest.TestCase):
    def test_reads_tau2_reward(self):
        self.assertEqual(tau2_utility({"reward_info": {"reward": 1.0}}), 1.0)
        self.assertIsNone(tau2_utility({}))


if __name__ == "__main__":
    unittest.main()
