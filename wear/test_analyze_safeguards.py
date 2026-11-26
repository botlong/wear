"""Tests for analyze_safeguards.py (tau2 env).

Synthetic rollouts cover the edge cases of the rollout measures; one test checks
that the analysis reproduces the verification-policy effects of the main
experiment recorded in wear/paper/NUMBERS.json, when those runs are present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
pytest.importorskip("tau2")
import analyze_safeguards as a  # noqa: E402
from run_perturbed import apply_gold_remap  # noqa: E402

RUNS = HERE / "runs"
TASK = HERE / "deictic_tasks" / "airline_idreuse_boston.json"


def _src():
    """The source task airline_idreuse_boston."""
    t = json.load(open(TASK))
    return t[0] if isinstance(t, list) else t


def _sim(calls, fired_at, fired=True):
    """calls: (name, args, error) in order; each gets a unique id and a tool reply."""
    msgs = [{"role": "assistant", "content": "Hi", "usage": None}]
    for i, (name, args, err) in enumerate(calls):
        msgs.append({"role": "assistant", "usage": {"prompt_tokens": 1}, "raw_data": {"model": "m"},
                     "tool_calls": [{"id": f"c{i}", "name": name, "arguments": args,
                                     "requestor": "assistant"}]})
        msgs.append({"role": "tool", "id": f"c{i}", "content": "x", "error": err})
    ev = [{"kind": "fired", "call_index": fired_at, "call_id": f"c{fired_at}"}] \
        if fired_at is not None else []
    return {"seed": 300, "termination_reason": "user_stop", "messages": msgs,
            "_perturb_fired": fired, "_perturb_events": ev, "_final_db_hash": None}


def _gold_args(src):
    """The task's first gold write action."""
    return next(x for x in src["evaluation_criteria"]["actions"] if x["action_id"].startswith("w"))


def test_failed_trigger_read_and_pre_change_write():
    """Failed read, successful pre-change write, successful re-read, rejected stale
    write. No stale write took effect; the pre-change write was correct."""
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    sim = _sim([("get_reservation_details", {"reservation_id": old}, True),
                (w["name"], dict(w["arguments"]), False),
                ("get_reservation_details", {"reservation_id": old}, False),
                (w["name"], dict(w["arguments"]), True)], fired_at=3)
    stored = apply_gold_remap(src, "perturbed", "airline")
    m = a.rollout_measures(sim, stored, src, "airline", "perturbed")
    assert m["stale_writes"] == 0 and m["passed_after_reread"] == 0
    assert m["attempted"] == 0.5 and m["effective"] == 0.0
    assert m["completion"] is True


def test_stale_write_after_successful_reread_is_counted():
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    sim = _sim([("get_reservation_details", {"reservation_id": old}, False),
                ("get_reservation_details", {"reservation_id": old}, False),
                (w["name"], dict(w["arguments"]), False)], fired_at=1)
    stored = apply_gold_remap(src, "perturbed", "airline")
    m = a.rollout_measures(sim, stored, src, "airline", "perturbed")
    assert (m["stale_writes"], m["passed_after_reread"], m["other_wrong"]) == (1, 1, 0)
    assert m["effective"] == 1.0 and m["completion"] is False


def test_unexposed_rollout_contributes_nothing():
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    sim = _sim([("get_reservation_details", {"reservation_id": old}, True),
                (w["name"], dict(w["arguments"]), False)], fired_at=None, fired=False)
    m = a.rollout_measures(sim, apply_gold_remap(src, "perturbed", "airline"), src, "airline", "perturbed")
    assert not m["exposed"] and m["attempted"] is None and m["stale_writes"] == 0


def test_unmatched_responses_are_unresolved_not_zero():
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    sim = _sim([("get_reservation_details", {"reservation_id": old}, False),
                (w["name"], dict(w["arguments"]), False)], fired_at=1)
    sim["messages"] = [m for m in sim["messages"] if not (m["role"] == "tool" and m["id"] == "c1")]
    m = a.rollout_measures(sim, apply_gold_remap(src, "perturbed", "airline"), src, "airline", "perturbed")
    assert m["stale_writes"] == 0 and m["stale_unresolved"] == 1 and m["writes_unresolved"] == 1


def test_paired_needs_eight_pairs():
    cells = {("x",): {f"t{i}": [{"attempted": 1.0}] for i in range(7)},
             ("y",): {f"t{i}": [{"attempted": 0.0}] for i in range(7)}}
    assert "p_two_sided" not in a.paired(cells, ("x",), ("y",), "attempted")


@pytest.mark.skipif(not (RUNS / "policy").is_dir(), reason="policy runs not present")
def test_reproduces_mitigation_effects():
    """The policy-vs-none test on the main suite matches the verification-policy effects in NUMBERS.json."""
    allnums = json.load(open(HERE / "paper" / "NUMBERS.json"))
    nums, be = allnums["mitigation"], allnums["binding_extended"]["cells"]
    res = a.analyze({"none": {"main": str(RUNS / "main_experiment")},
                     "policy": {"main": str(RUNS / "policy")}},
                    replay=False, legacy_fire=True)
    for slug in ("openai_gpt_5_5", "openai_deepseek_v4_pro"):
        t = res["tests"][f"main/{slug}"]["policy_vs_none_attempted"]
        # analyze_safeguards scores the runs without a safeguard on the eligible denominator,
        # so the expected difference uses the binding_extended task means, which are
        # computed the same way. They differ from the simpler all-writes rule only
        # for the one DeepSeek airline trajectory whose writes are all unmatched.
        unmit = sum(be[f"{slug}/{d}/perturbed"]["task_mean_wear"] for d in ("airline", "retail")) / 2
        want = unmit - allnums["mitigation_task_means"][slug]["policy_perturbed"]["task_mean_wear"]
        assert t["n_pairs"] == 20
        assert abs(-t["mean_diff_b_minus_a"] - want) < 1e-4
        assert t["p_two_sided"] < 0.01


def test_boundary_located_by_call_id_not_position():
    """A call whose arguments fail to bind never reaches make_tool_call, so perturb.py's
    call_index (here 1) is not the trajectory position (2); the logged call id is."""
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    sim = _sim([("get_reservation_details", {"requestor": "x"}, True),  # never bound
                ("get_reservation_details", {"reservation_id": old}, False),  # trigger
                (w["name"], dict(w["arguments"]), False)], fired_at=2)
    sim["_perturb_events"][0]["call_index"] = 1
    m = a.rollout_measures(sim, apply_gold_remap(src, "perturbed", "airline"), src, "airline", "perturbed")
    assert (m["stale_writes"], m["passed_after_reread"]) == (1, 0)


def test_boundary_unresolved_when_call_id_missing():
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    sim = _sim([("get_reservation_details", {"reservation_id": old}, False)], fired_at=0)
    sim["_perturb_events"][0]["call_id"] = None
    m = a.rollout_measures(sim, apply_gold_remap(src, "perturbed", "airline"), src, "airline", "perturbed")
    assert m["boundary_unresolved"] and not m["exposed"] and m["attempted"] is None


def test_other_wrong_counts_eligible_non_stale_writes_only():
    """An unmatched stale cancellation plus an eligible write to another wrong
    reservation: other_wrong is 1."""
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    other = dict(w["arguments"], reservation_id="ZZZZZZ")
    sim = _sim([("get_reservation_details", {"reservation_id": old}, False),
                ("cancel_reservation", {"reservation_id": old}, False),
                (w["name"], other, False)], fired_at=1)
    m = a.rollout_measures(sim, apply_gold_remap(src, "perturbed", "airline"), src, "airline", "perturbed")
    assert m["effective"] == 1.0 and m["stale_writes"] == 1 and m["other_wrong"] == 1


def test_cost_measures_count_reads_and_skip_unexposed():
    src = _src()
    old = src["_perturb"]["spec"]["referent_id"]
    w = _gold_args(src)
    calls = [("get_reservation_details", {"reservation_id": old}, False),
             ("get_reservation_details", {"reservation_id": old}, False),
             (w["name"], dict(w["arguments"]), False)]
    stored = apply_gold_remap(src, "perturbed", "airline")
    sim = {**_sim(calls, fired_at=0), "duration": 12.5}
    m = a.rollout_measures(sim, stored, src, "airline", "perturbed")
    assert (m["tool_calls"], m["read_calls"], m["duration_s"]) == (3, 2, 12.5)
    sim = {**_sim(calls, fired_at=None, fired=False), "duration": 12.5}
    m = a.rollout_measures(sim, stored, src, "airline", "perturbed")
    assert not m["exposed"] and m["tool_calls"] is None and m["duration_s"] is None
