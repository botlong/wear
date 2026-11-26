"""Unit tests for score_wear_ts.py, the WEAR scorer of the ToolSandbox replication.

Standard library only; ToolSandbox and polars are not needed. The conversations
are small hand-written examples in both message formats the scorer accepts.

Usage: python wear/toolsandbox/test_score_wear_ts.py
   or: tau2-bench/.venv/bin/python -m pytest wear/toolsandbox/test_score_wear_ts.py
"""

import json

from score_wear_ts import score_trajectory, WRITE_TOOL_ID_ARG

GOLD = "gold-person-uuid"
WRONG = "wrong-person-uuid"


def _tool_msg(tool_name, arguments, errored=False):
    """Build an internal-format message (execution environment to agent) with one tool_trace entry."""
    return {
        "sender": "execution_environment",
        "recipient": "agent",
        "tool_trace": [json.dumps({"tool_name": tool_name, "arguments": arguments})],
        "tool_call_exception": "some error" if errored else None,
    }


def test_correct_write_is_clean():
    conv = [_tool_msg("modify_contact", {"person_id": GOLD, "phone_number": "+1"})]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1
    assert r.wear == 0.0 and r.effective_wear == 0.0


def test_wrong_write_successful_is_attempted_and_effective():
    conv = [_tool_msg("modify_contact", {"person_id": WRONG, "phone_number": "+1"})]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1
    assert r.wear == 1.0 and r.effective_wear == 1.0


def test_wrong_write_errored_is_attempted_not_effective():
    conv = [_tool_msg("modify_contact", {"person_id": WRONG}, errored=True)]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1
    assert r.wear == 1.0 and r.effective_wear == 0.0


def test_no_write_calls_is_undefined():
    conv = [{"sender": "agent", "recipient": "user", "content": "hi", "tool_trace": None}]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 0
    assert r.wear is None and r.effective_wear is None


def test_read_tools_are_ignored():
    conv = [
        {
            "sender": "execution_environment",
            "recipient": "agent",
            "tool_trace": [json.dumps({"tool_name": "search_contacts", "arguments": {"name": "John"}})],
            "tool_call_exception": None,
        }
    ]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 0


def test_mixed_one_correct_one_wrong_is_half():
    conv = [
        _tool_msg("modify_contact", {"person_id": GOLD}),
        _tool_msg("remove_contact", {"person_id": WRONG}),
    ]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 2
    assert r.wear == 0.5 and r.effective_wear == 0.5


def test_reminder_write_target_arg():
    conv = [_tool_msg("modify_reminder", {"reminder_id": WRONG, "content": "x"})]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 1.0


def _openai_msgs(tool_name, args, result_content=None, call_id="call_1"):
    """Build the OpenAI-format pair for one call: the assistant tool call and its role "tool" result."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": tool_name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "name": tool_name, "tool_call_id": call_id, "content": result_content},
    ]


def test_openai_format_correct_write():
    conv = _openai_msgs("modify_contact", {"person_id": GOLD, "phone_number": "+1"})
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 0.0


def test_openai_format_wrong_write():
    conv = _openai_msgs("modify_contact", {"person_id": WRONG, "phone_number": "+1"})
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 1.0 and r.effective_wear == 1.0


def test_openai_format_wrong_write_errored_content():
    conv = _openai_msgs("modify_contact", {"person_id": WRONG}, result_content="ValueError: contact not found")
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 1.0 and r.effective_wear == 0.0


def test_openai_format_polars_nodataerror_detected():
    real = ("polars.exceptions.NoDataError: No db entry matching "
            "person_id='e3570ab6-0819-5032-be1e-2b366390c8ef' found")
    conv = _openai_msgs("modify_contact", {"person_id": WRONG}, result_content=real)
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 1.0 and r.effective_wear == 0.0


def test_openai_format_search_is_ignored():
    conv = _openai_msgs("search_contacts", {"name": "John"}, result_content="[{...}]")
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 0


def test_tool_trace_as_dict_not_string():
    conv = [{
        "sender": "execution_environment", "recipient": "agent",
        "tool_trace": [{"tool_name": "modify_contact", "arguments": {"person_id": WRONG}}],
        "tool_call_exception": None,
    }]
    r = score_trajectory(conv, GOLD)
    assert r.n_write_calls == 1 and r.wear == 1.0


# Run as a script, this executes the tests defined above this block; pytest also
# collects the test defined below it.
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} ToolSandbox WEAR-scorer tests passed. WRITE tools: {list(WRITE_TOOL_ID_ARG)}")


def test_write_without_target_does_not_dilute_wear():
    """A write call that names no target cannot be compared with the gold target, so it
    is not an eligible write and stays out of the denominator of WEAR."""
    from score_wear_ts import score_trajectory
    conv = [{"role": "assistant", "tool_calls": [
        {"id": "a", "function": {"name": "remove_reminder", "arguments": "{\"reminder_id\": \"r2\"}"}},
        {"id": "b", "function": {"name": "remove_reminder", "arguments": "{}"}},
    ]}]
    r = score_trajectory(conv, "r1")
    assert (r.n_write_calls, r.n_eligible, r.n_wrong_entity) == (2, 1, 1)
    assert r.wear == 1.0
