"""Unit tests for scorer.py on constructed trajectories (no network).

Run from the repository root: tau2-bench/.venv/bin/python -m pytest wear/test_scorer.py
"""
from __future__ import annotations

from scorer import (
    score_simulation,
    score_results,
    contingency_table,
    extract_agent_write_calls,
)

WRITE_TOOLS = {"cancel_pending_order", "modify_user_address", "exchange_delivered_order_items"}
OBJ_ID_KEYS = {"order_id", "user_id"}


def _asst(tool, args, requestor="assistant"):
    """An assistant message with one tool call (call id "x")."""
    return {"role": "assistant", "tool_calls": [
        {"id": "x", "name": tool, "arguments": args, "requestor": requestor}]}


def _tool_obs(content="{}"):
    """A tool message without a call id, so it is attached to no call."""
    return {"role": "tool", "content": content, "requestor": "assistant"}


def _sim(messages, reward=1.0, task_id="t1", sid="s1"):
    return {"id": sid, "task_id": task_id, "messages": messages,
            "reward_info": {"reward": reward}}


def _task(actions, task_id="t1"):
    return {"id": task_id, "evaluation_criteria": {"actions": actions}}


def test_correct_binding():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W1"})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_write_calls == 1 and r.n_shape_ok == 1
    assert r.n_correct == 1 and r.n_wrong_entity == 0
    assert r.wear == 0.0 and r.has_wrong_bind is False


def test_wrong_entity():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W2"})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_wrong_entity == 1 and r.wear == 1.0 and r.has_wrong_bind is True
    assert r.wrong_examples and r.wrong_examples[0]["tool"] == "cancel_pending_order"


def test_errored_wrong_write_attempted_vs_effective():
    msgs = [
        _asst("cancel_pending_order", {"order_id": "#W2"}),
        {"role": "tool", "id": "x", "error": True, "content": "Error: not found"},
    ]
    sim = _sim(msgs)
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_wrong_entity == 1 and r.wear == 1.0
    assert r.n_wrong_entity_effective == 0 and r.effective_wear == 0.0
    assert r.wrong_examples[0]["errored"] is True


def test_successful_wrong_write_is_effective():
    msgs = [
        _asst("cancel_pending_order", {"order_id": "#W2"}),
        {"role": "tool", "id": "x", "error": False, "content": "{}"},
    ]
    sim = _sim(msgs)
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_wrong_entity == 1 and r.n_wrong_entity_effective == 1
    assert r.wear == 1.0 and r.effective_wear == 1.0


def test_case_and_whitespace_insensitive():
    sim = _sim([_asst("cancel_pending_order", {"order_id": " #W1 "})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#w1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_correct == 1 and r.n_wrong_entity == 0


def test_spurious_action_not_shape_ok():
    sim = _sim([_asst("modify_user_address", {"user_id": "U9"})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_write_calls == 1 and r.n_spurious == 1
    assert r.n_shape_ok == 0 and r.n_wrong_entity == 0
    assert r.wear is None and r.no_eligible_write  # no eligible write: WEAR is undefined
    assert r.wear_all_writes == 0.0
    assert r.wear_shapeok is None


def test_user_initiated_ignored():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W2"}, requestor="user")])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_write_calls == 0 and r.n_wrong_entity == 0


def test_read_tools_ignored():
    sim = _sim([_asst("get_order_details", {"order_id": "#W2"})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_write_calls == 0


def test_no_id_check():
    sim = _sim([_asst("modify_user_address", {"street": "Main St"})])
    task = _task([{"name": "modify_user_address", "arguments": {"street": "Oak St"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_shape_ok == 1 and r.n_no_id_check == 1
    assert r.n_wrong_entity == 0 and r.n_correct == 0


def test_multiple_gold_same_name_is_ambiguous():
    """With several gold writes for the tool and no alignment annotation, a write cannot
    be aligned. Matching any gold write would certify it, so it is counted as ambiguous."""
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W2"})])
    task = _task([
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W2"}},
    ])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert (r.n_correct, r.n_wrong_entity, r.n_ambiguous) == (0, 0, 1)
    assert r.wear is None and r.no_eligible_write


def test_compare_args_override():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W1", "reason": "no_longer_needed"})])
    task = _task([{"name": "cancel_pending_order",
                   "arguments": {"order_id": "#W1", "reason": "ordered_by_mistake"},
                   "compare_args": ["order_id"]}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_correct == 1 and r.n_wrong_entity == 0


def test_mixed_calls_wear_fraction():
    sim = _sim([
        _asst("cancel_pending_order", {"order_id": "#W1"}),
        _tool_obs(),
        _asst("exchange_delivered_order_items", {"order_id": "#W9"}),
    ])
    task = _task([
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
        {"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W8"}},
    ])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_write_calls == 2 and r.n_correct == 1 and r.n_wrong_entity == 1
    assert r.wear == 0.5 and r.wear_shapeok == 0.5


def test_contingency_table():
    wt = {"retail": list(WRITE_TOOLS), "_object_id_keys": {"retail": list(OBJ_ID_KEYS)}}
    results = {
        "tasks": [_task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}], task_id="t1")],
        "simulations": [
            _sim([_asst("cancel_pending_order", {"order_id": "#W2"}),
                  _asst("cancel_pending_order", {"order_id": "#W1"})], reward=1.0, sid="a"),
            _sim([_asst("cancel_pending_order", {"order_id": "#W1"})], reward=0.0, sid="b"),
        ],
    }
    rs = score_results(results, "retail", wt)
    ct = contingency_table(rs)
    assert ct["cells"]["success/wrong"] == 1
    assert ct["cells"]["fail/clean"] == 1
    assert ct["key_cell_success_and_wrong"] == 1


def test_empty_compare_args_is_name_only():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W9"})])
    task = _task([{"name": "cancel_pending_order",
                   "arguments": {"order_id": "#W1"}, "compare_args": []}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_shape_ok == 1 and r.n_correct == 1
    assert r.n_wrong_entity == 0 and r.n_no_id_check == 0


def test_call_missing_gold_id_is_wrong():
    sim = _sim([_asst("cancel_pending_order", {"reason": "no_longer_needed"})])
    task = _task([{"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_shape_ok == 1 and r.n_wrong_entity == 1 and r.n_no_id_check == 0


def test_extra_id_arg_does_not_poison_match():
    sim = _sim([_asst("cancel_pending_order",
                      {"order_id": "#W1", "payment_method_id": "PX"})])
    task = _task([{"name": "cancel_pending_order",
                   "arguments": {"order_id": "#W1"}, "compare_args": ["order_id"]}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_correct == 1 and r.n_wrong_entity == 0


def test_mixed_golds_without_alignment_are_ambiguous():
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W9"})])
    task = _task([
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W2"}, "compare_args": []},
    ])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert (r.n_correct, r.n_wrong_entity, r.n_ambiguous) == (0, 0, 1)


def test_two_specific_golds_no_match_is_ambiguous_not_wrong():
    """Without alignment the scorer cannot say which gold write was meant, so it does not
    call the write wrong either; scorer_extended.py aligns such tasks by annotation."""
    sim = _sim([_asst("cancel_pending_order", {"order_id": "#W9"})])
    task = _task([
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}},
        {"name": "cancel_pending_order", "arguments": {"order_id": "#W2"}},
    ])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert r.n_wrong_entity == 0 and r.n_ambiguous == 1 and r.wear is None


def test_shipped_task_golds_use_only_object_id_keys():
    """Every gold write action in the shipped tasks (deictic_tasks/) names its target only
    through the domain's target-identifier keys, and has at least one such key."""
    import json as _json
    from pathlib import Path as _P
    from scorer import _entity_keys, NAME_ONLY, load_write_tools

    wt = load_write_tools()
    task_files = sorted((_P(__file__).parent / "deictic_tasks").glob("*.json"))
    assert task_files, "deictic_tasks/*.json missing"
    n_gold = 0
    for tf in task_files:
        if tf.name.startswith("airline"):
            domain = "airline"
        elif tf.name.startswith("retail"):
            domain = "retail"
        else:
            continue
        wts, oik = set(wt[domain]), set(wt["_object_id_keys"][domain])
        for t in _json.load(open(tf)):
            for a in (t.get("evaluation_criteria") or {}).get("actions") or []:
                if a.get("name") not in wts:
                    continue
                keys = _entity_keys(dict(a.get("arguments") or {}), oik,
                                    a.get("compare_args"))
                assert keys is not NAME_ONLY and keys, \
                    f"{tf.name}:{t.get('id')} gold write action has no entity key"
                assert set(keys) <= oik, \
                    f"{tf.name}:{t.get('id')} fallback selected non-object_id_keys keys: {keys}"
                n_gold += 1
    assert n_gold >= 39, f"too few gold write actions covered: {n_gold}"


def _run_all():
    """Run the test functions defined above this point without pytest."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()


def test_no_eligible_write_marks_undefined_wear():
    """Writes that all use tools outside the gold actions leave the eligible set empty,
    so WEAR is undefined and the trajectory is left out of the task mean."""
    from scorer import SimWear
    assert SimWear(sim_id=1, task_id="t", n_write_calls=2, n_spurious=2).no_eligible_write
    assert not SimWear(sim_id=1, task_id="t", n_write_calls=0).no_eligible_write
    assert not SimWear(sim_id=1, task_id="t", n_write_calls=2, n_shape_ok=1, n_correct=1,
                       n_spurious=1).no_eligible_write


def test_unmatched_write_stays_out_of_the_denominator():
    """WEAR is |W|/|E|. A write with a tool that no gold write uses is unmatched: it is
    counted separately and must not lower WEAR."""
    sim = _sim([
        _asst("exchange_delivered_order_items", {"order_id": "#W9"}),
        _tool_obs(),
        _asst("modify_user_address", {"user_id": "U9"}),
    ])
    task = _task([{"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W8"}}])
    r = score_simulation(sim, task, WRITE_TOOLS, OBJ_ID_KEYS)
    assert (r.n_write_calls, r.n_eligible, r.n_wrong_entity, r.n_spurious) == (2, 1, 1, 1)
    assert r.wear == 1.0 and r.wear_all_writes == 0.5
