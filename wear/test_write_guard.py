"""Offline tests of both write guards (write_guard.py) on the real tau2 databases.

Covers the CAS and identity guards with and without the ID-REUSE change, the
required wrapping order (guard inside the perturbation wrapper), and the
held-out tasks. Run in the tau2 env: pytest wear/test_write_guard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perturb import wrap_environment  # noqa: E402
from run_perturbed import schedule_from_task  # noqa: E402
from write_guard import wrap_write_guard  # noqa: E402

from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.runner.build import build_environment  # noqa: E402

WT = json.load(open(HERE / "write_tools.json"))
CASES = {
    "airline": ("airline_idreuse_dallas", "get_reservation_details", "reservation_id"),
    "retail": ("retail_idreuse_laptop", "get_order_details", "order_id"),
}


def _task(name):
    """Load one task of the main suite."""
    d = json.load(open(HERE / "deictic_tasks" / f"{name}.json"))
    return d[0] if isinstance(d, list) else d


def _gold_write(task):
    """The task's first gold write action (action ids starting with "w")."""
    return next(a for a in task["evaluation_criteria"]["actions"] if a["action_id"].startswith("w"))


def _env(domain, guard=True, perturb=None, guard_outside=False, mode="cas"):
    """Fresh environment with a guard and an optional perturbation, wrapped in the given order."""
    env = build_environment(domain)
    if guard and not guard_outside:
        env = wrap_write_guard(env, set(WT[domain]), mode)
    if perturb is not None:
        env = wrap_environment(env, perturb)
    if guard and guard_outside:
        env = wrap_write_guard(env, set(WT[domain]), mode)
    return env


def _call(env, name, args):
    """Execute one agent tool call and return the tool message."""
    return env.get_response(ToolCall(id="t", name=name, arguments=args, requestor="assistant"))


def _setup(domain, **kw):
    """Environment and task data for a domain's test task; perturbed=True adds its schedule."""
    name, read_tool, id_field = CASES[domain]
    task = _task(name)
    spec = task["_perturb"]["spec"]
    sched = schedule_from_task(task) if kw.pop("perturbed", False) else None
    env = _env(domain, perturb=sched, **kw)
    return env, task, spec, read_tool, id_field


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_unchanged_world_write_passes(domain):
    env, task, spec, read_tool, idf = _setup(domain)
    assert not _call(env, read_tool, {idf: spec["referent_id"]}).error
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error, tm.content


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_stale_write_rejected_and_distractor_untouched(domain):
    env, task, spec, read_tool, idf = _setup(domain, perturbed=True)
    old, new = spec["referent_id"], spec["referent_new_id"]
    assert not _call(env, read_tool, {idf: old}).error  # binding read arms id_reuse
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])  # the change is applied, then the guard checks
    assert env._perturb._fired
    assert tm.error and "Precondition failed" in tm.content
    assert env._write_guard.events[-1]["kind"] == "write_rejected"
    # reading the new identifier and writing to it passes
    assert not _call(env, read_tool, {idf: new}).error
    args = dict(w["arguments"], **{idf: new})
    tm2 = _call(env, w["name"], args)
    assert not tm2.error, tm2.content


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_rejected_write_leaves_db_unchanged(domain):
    env, task, spec, read_tool, idf = _setup(domain, perturbed=True)
    _call(env, read_tool, {idf: spec["referent_id"]})
    w = _gold_write(task)
    env._perturb.before(w["name"], "assistant", w["arguments"])  # apply the change
    h = env.get_db_hash()
    tm = _call(env, w["name"], w["arguments"])
    assert tm.error and env.get_db_hash() == h


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_unread_id_passes(domain):
    env, task, spec, read_tool, idf = _setup(domain)
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error, tm.content


def test_successive_writes_to_same_entity_pass():
    env, task, spec, read_tool, idf = _setup("airline")
    _call(env, read_tool, {idf: spec["referent_id"]})
    w = _gold_write(task)
    assert not _call(env, w["name"], w["arguments"]).error
    args = dict(w["arguments"], total_baggages=w["arguments"]["total_baggages"] + 1)
    tm = _call(env, w["name"], args)
    assert not tm.error, tm.content


def test_guard_outside_perturbation_misses_the_change():
    """Documents the required order: the guard must be wrapped first (inside)."""
    env, task, spec, read_tool, idf = _setup("airline", perturbed=True, guard_outside=True)
    _call(env, read_tool, {idf: spec["referent_id"]})
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error  # wrong order: the stale write goes through


def test_reread_of_changed_entity_then_write_passes():
    """Compare-and-set checks staleness, not intent: after re-reading the stale id
    (now the distractor), a write to it is based on the current version and passes."""
    env, task, spec, read_tool, idf = _setup("airline", perturbed=True)
    _call(env, read_tool, {idf: spec["referent_id"]})
    _call(env, read_tool, {idf: spec["referent_id"]})  # change applied; guard records the distractor
    w = _gold_write(task)
    assert not _call(env, w["name"], w["arguments"]).error


# --- identity mode ----------------------------------------------------------- #

@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_identity_unchanged_world_and_follow_up_writes_pass(domain):
    env, task, spec, read_tool, idf = _setup(domain, mode="identity")
    assert not _call(env, read_tool, {idf: spec["referent_id"]}).error
    w = _gold_write(task)
    assert not _call(env, w["name"], w["arguments"]).error
    assert not _call(env, read_tool, {idf: spec["referent_id"]}).error
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error, tm.content


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_identity_rejects_stale_write_even_after_reread(domain):
    """The case compare-and-set passes (test_reread_of_changed_entity_then_write_passes)."""
    env, task, spec, read_tool, idf = _setup(domain, perturbed=True, mode="identity")
    old, new = spec["referent_id"], spec["referent_new_id"]
    _call(env, read_tool, {idf: old})                 # binds old -> target record
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])        # change applied, then the guard checks
    assert env._perturb._fired
    assert tm.error and "now refers to a different" in tm.content
    h = env.get_db_hash()
    assert not _call(env, read_tool, {idf: old}).error  # re-read shows the distractor
    tm = _call(env, w["name"], w["arguments"])
    assert tm.error and "now refers to a different" in tm.content
    assert env.get_db_hash() == h                      # distractor untouched
    kinds = [e["kind"] for e in env._write_guard.events]
    assert kinds.count("write_rejected_identity") == 2
    # the target record under its new identifier is writable after reading it
    assert not _call(env, read_tool, {idf: new}).error
    tm = _call(env, w["name"], dict(w["arguments"], **{idf: new}))
    assert not tm.error, tm.content


def test_identity_first_read_after_change_binds_current_entity():
    """The binding forms at the agent's first read: an identifier first read after the
    change is bound to the record it then names (a stated limit of the guard)."""
    env, task, spec, read_tool, idf = _setup("airline", perturbed=True, mode="identity")
    env._perturb._fire_id_reuse()
    env._perturb._fired = True  # applied by hand; do not apply again
    _call(env, read_tool, {idf: spec["referent_id"]})
    w = _gold_write(task)
    assert not _call(env, w["name"], w["arguments"]).error


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_identity_unread_id_passes(domain):
    env, task, spec, read_tool, idf = _setup(domain, mode="identity")
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error, tm.content


@pytest.mark.parametrize("mode", ["cas", "identity"])
@pytest.mark.parametrize("fname", ["airline_heldout_boston_chen", "retail_heldout_backpack_harper"])
def test_heldout_stale_write_rejected(mode, fname):
    """Both guards reject the stale write on held-out tasks of both domains."""
    d = json.load(open(HERE / "deictic_tasks_heldout" / f"{fname}.json"))
    task = d[0] if isinstance(d, list) else d
    domain = fname.split("_", 1)[0]
    _, read_tool, idf = CASES[domain]
    spec = task["_perturb"]["spec"]
    env = _env(domain, perturb=schedule_from_task(task), mode=mode)
    _call(env, read_tool, {idf: spec["referent_id"]})
    w = _gold_write(task)
    tm = _call(env, w["name"], w["arguments"])
    assert env._perturb._fired
    assert tm.error and "Precondition failed" in tm.content
