"""Unit tests for eligibility.py (the timing check) on constructed trajectories.

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_eligibility.py
"""
import unittest

from eligibility import Eligibility, assess, assess_results, load_perturb_index, aggregate

WRITE_TOOLS = {"update_reservation_baggages"}
SPEC = {
    "trigger_arg": "reservation_id",
    "trigger_arg_value": "NO6JO3",
    "referent_id": "NO6JO3",
    "referent_new_id": "ZZ0001",
    "distractor_id": "HKEG34",
}


def _asst(name, args, cid, requestor="assistant"):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "name": name, "arguments": args,
                            "requestor": requestor}]}


def _tool(cid, content, error=False):
    return {"role": "tool", "id": cid, "content": content, "error": error}


def _sim(messages, task_id="t", sim_id="s1"):
    return {"task_id": task_id, "id": sim_id, "messages": messages}


def _assess(msgs, **kw):
    """Run the timing check with SPEC, the binding read being get_reservation_details."""
    return assess(_sim(msgs), SPEC, WRITE_TOOLS,
                  target_tool="get_reservation_details", **kw)


class TestAssess(unittest.TestCase):
    def test_no_recheck_guardrail_satisfied(self):
        e = _assess([
            _asst("get_user_details", {"user_id": "u"}, "c0"),
            _tool("c0", "{...reservations: NO6JO3, HKEG34}"),
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", '{"reservation_id": "NO6JO3", "destination": "DFW"}'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertTrue(e.trigger_found)
        self.assertFalse(e.recheck_any)
        self.assertEqual(e.label, "guardrail_satisfied")
        self.assertEqual(e.first_write_name, "update_reservation_baggages")
        self.assertFalse(e.new_id_observed)

    def test_recheck_without_new_id(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "orig world"),
            _asst("get_reservation_details", {"reservation_id": "HKEG34"}, "c1"),
            _tool("c1", '{"reservation_id": "HKEG34"}'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertEqual(e.label, "rechecked")
        self.assertEqual(e.recheck_reads, ["get_reservation_details"])
        self.assertFalse(e.new_id_observed)

    def test_recheck_observes_new_id(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "orig world"),
            _asst("get_user_details", {"user_id": "u"}, "c1"),
            _tool("c1", '{"reservations": ["ZZ0001", "NO6JO3"]}'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertEqual(e.label, "rechecked")
        self.assertTrue(e.new_id_observed)

    def test_no_trigger(self):
        e = _assess([
            _asst("get_user_details", {"user_id": "u"}, "c0"),
            _tool("c0", "x"),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", "ok"),
        ])
        self.assertEqual(e.label, "no_trigger")

    def test_write_before_trigger(self):
        e = _assess([
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "ok"),
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", "x"),
        ])
        self.assertEqual(e.label, "write_before_trigger")

    def test_no_write_window_extends_to_end(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "x"),
            _asst("get_user_details", {"user_id": "u"}, "c1"),
            _tool("c1", "no ids here"),
        ])
        self.assertIsNone(e.first_write_name)
        self.assertEqual(e.label, "rechecked")

    def test_case_insensitive_trigger_and_new_id(self):
        spec = dict(SPEC, trigger_arg="order_id", trigger_arg_value="#W8327915",
                    referent_new_id="#W0000001")
        e = assess(_sim([
            _asst("get_order_details", {"order_id": "#w8327915"}, "c0"),
            _tool("c0", "orig"),
            _asst("get_order_details", {"order_id": "#w8327915"}, "c1"),
            _tool("c1", 'order not found; user orders: ["#w0000001"]'),
            _asst("modify_pending_order_address", {"order_id": "#w8327915"}, "c2"),
            _tool("c2", "ok"),
        ]), spec, {"modify_pending_order_address"},
            target_tool="get_order_details")
        self.assertTrue(e.trigger_found)
        self.assertEqual(e.label, "rechecked")
        self.assertTrue(e.new_id_observed)

    def test_new_id_boundary_rejects_substring(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "orig world"),
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", '{"reservations": ["ZZ00010", "XZZ0001"]}'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertEqual(e.label, "rechecked")
        self.assertFalse(e.new_id_observed)

    def test_fire_after_n_calls_2(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "1st read: not trigger yet"),
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", "2nd read: trigger"),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ], fire_after_n_calls=2)
        self.assertTrue(e.trigger_found)
        self.assertEqual(e.trigger_flat_index, 1)
        self.assertEqual(e.label, "guardrail_satisfied")

    def test_user_requestor_read_not_recheck_but_info_counts(self):
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
            _tool("c0", "orig"),
            _asst("get_user_details", {"user_id": "u"}, "c1", requestor="user"),
            _tool("c1", 'reservations now: ZZ0001'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertFalse(e.recheck_any)
        self.assertEqual(e.label, "guardrail_satisfied")
        self.assertTrue(e.new_id_observed)


class TestBatch(unittest.TestCase):
    def test_assess_results_joins_by_task_id(self):
        results = {"simulations": [
            _sim([
                _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0"),
                _tool("c0", "x"),
                _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c1"),
                _tool("c1", "ok"),
            ], task_id="airline_idreuse_dallas"),
            _sim([], task_id="unknown_task"),
        ]}
        pidx = {"airline_idreuse_dallas": {
            "knob": "id_reuse", "target_tool": "get_reservation_details",
            "fire_after_n_calls": 1, "spec": SPEC}}
        rows = assess_results(results, pidx, WRITE_TOOLS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].label, "guardrail_satisfied")
        agg = aggregate(rows)
        self.assertEqual(agg["n"], 1)
        self.assertEqual(agg["guardrail_satisfied"], 1)

    def test_user_requestor_call_does_not_trigger(self):
        """A call requested by the user simulator never counts as the binding read, as in
        the perturbation hook of perturb.py."""
        e = _assess([
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c0",
                  requestor="user"),
            _tool("c0", '{"reservation_id": "NO6JO3"}'),
            _asst("get_reservation_details", {"reservation_id": "NO6JO3"}, "c1"),
            _tool("c1", '{"reservation_id": "NO6JO3", "destination": "DFW"}'),
            _asst("update_reservation_baggages", {"reservation_id": "NO6JO3"}, "c2"),
            _tool("c2", "ok"),
        ])
        self.assertTrue(e.trigger_found)
        self.assertEqual(e.trigger_flat_index, 1)
        self.assertFalse(e.recheck_any)
        self.assertEqual(e.label, "guardrail_satisfied")


if __name__ == "__main__":
    unittest.main()
