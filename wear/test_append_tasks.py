"""Offline tests of the APPEND and REORDER task files written by gen_append.py (no API calls).

The tests load the tasks from wear/deictic_tasks/, build the tau2-bench airline
environment with the perturbation layer, and check the mechanics: the decoy is
in the database but not in the user's list, the first list read hides it and
the second shows it, a write to the decoy succeeds, the expected end-state hash
is the same in the baseline and perturbed conditions, and REORDER puts another
reservation in position 2 from the second list read on.

Usage: tau2-bench/.venv/bin/python -m pytest wear/test_append_tasks.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

from expected_hash import expected_hash  # noqa: E402
from perturb import wrap_environment  # noqa: E402
from run_perturbed import schedule_from_task  # noqa: E402

from tau2.registry import registry  # noqa: E402


def _load(pattern):
    """Load the tasks of all files in wear/deictic_tasks/ that match pattern."""
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "deictic_tasks", pattern))):
        out.extend(json.load(open(f)))
    return out

APPEND_TASKS = _load("airline_append_*.json")
# The 9 generated REORDER tasks; the hand-built airline_reorder_ordinal is left out.
REORDER_TASKS = [t for t in _load("airline_reorder_*.json")
                 if t["id"] != "airline_reorder_ordinal"]


def _fresh_env(task):
    """Build an airline environment with the task's initial agent data (the APPEND decoy) applied."""
    env = registry.get_env_constructor("airline")()
    init = (task.get("initial_state") or {}).get("initialization_data") or {}
    if init.get("agent_data"):
        env.tools.update_db(init["agent_data"])
    return env


def _gold(task):
    """Return the user id and the arguments of the gold bag write."""
    acts = task["evaluation_criteria"]["actions"]
    uid = next(a for a in acts if a["name"] == "get_user_details")["arguments"]["user_id"]
    w0 = next(a for a in acts if a["name"] == "update_reservation_baggages")["arguments"]
    return uid, w0


def test_generated_counts():
    """gen_append.py wrote 10 APPEND and 9 REORDER tasks."""
    assert len(APPEND_TASKS) == 10
    assert len(REORDER_TASKS) == 9


@pytest.mark.parametrize("task", APPEND_TASKS, ids=lambda t: t["id"])
def test_append_mechanics(task):
    """The decoy is a same-user copy of the target with the same flights on later dates,
    stored but not listed; the second list read shows it, and a write to it succeeds."""
    uid, w0 = _gold(task)
    decoy_id = task["_perturb"]["spec"]["append_item"]
    ref_id = w0["reservation_id"]
    env = _fresh_env(task)
    db = env.tools.db
    assert decoy_id in db.reservations
    assert ref_id in db.reservations
    assert decoy_id not in db.users[uid].reservations
    d, r = db.reservations[decoy_id], db.reservations[ref_id]
    assert d.user_id == uid
    assert [f.flight_number for f in d.flights] == [f.flight_number for f in r.flights]
    assert all(df.date > rf.date for df, rf in zip(d.flights, r.flights))
    env = wrap_environment(env, schedule_from_task(task))
    r1 = env.make_tool_call("get_user_details", user_id=uid)
    assert decoy_id not in r1.reservations
    r2 = env.make_tool_call("get_user_details", user_id=uid)
    assert decoy_id in r2.reservations
    assert decoy_id not in env.tools.db.users[uid].reservations
    w = env.make_tool_call(
        "update_reservation_baggages", reservation_id=decoy_id,
        total_baggages=w0["total_baggages"], nonfree_baggages=0,
        payment_id=w0["payment_id"])
    assert getattr(w, "reservation_id", None) == decoy_id


@pytest.mark.parametrize("task", APPEND_TASKS, ids=lambda t: t["id"])
def test_append_expected_hash_arms_equal(task):
    """APPEND changes only list reads, so the expected end-state hash is the same in
    the baseline and perturbed conditions."""
    hb = expected_hash(task, "baseline", "airline")
    hp = expected_hash(task, "perturbed", "airline")
    assert hb == hp


@pytest.mark.parametrize("task", REORDER_TASKS, ids=lambda t: t["id"])
def test_reorder_mechanics(task):
    """The first list read keeps the stored order, with the gold target in position 2;
    the second is rotated, so position 2 holds another reservation."""
    uid, w0 = _gold(task)
    env = _fresh_env(task)
    orig = list(env.tools.db.users[uid].reservations)
    assert len(orig) >= 3
    assert w0["reservation_id"] == orig[1]
    env = wrap_environment(env, schedule_from_task(task))
    r1 = env.make_tool_call("get_user_details", user_id=uid)
    assert list(r1.reservations) == orig
    r2 = env.make_tool_call("get_user_details", user_id=uid)
    assert list(r2.reservations) != orig
    assert r2.reservations[1] != orig[1]
    assert expected_hash(task, "perturbed", "airline")


def test_no_user_overlap_with_suite():
    """Every APPEND and REORDER task has its own user."""
    seen = set()
    for t in APPEND_TASKS + REORDER_TASKS:
        uid, _ = _gold(t)
        assert uid not in seen
        seen.add(uid)
