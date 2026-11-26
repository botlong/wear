"""Offline tests of the held-out suite (gen_heldout.py) on the tau2-bench databases (no API calls).

The tests check that the 12 committed task files equal the generator's output;
that without the change the gold write passes the CAS guard; and that after the
change the old identifier names a similar record (same metro area or same
product), the CAS guard rejects the gold write through it, and a write through
the new identifier passes once that identifier has been read.

Usage: tau2-bench/.venv/bin/python -m pytest wear/test_gen_heldout.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_heldout as gh  # noqa: E402
import gen_suite as gs  # noqa: E402
from perturb import wrap_environment  # noqa: E402
from run_perturbed import schedule_from_task  # noqa: E402
from write_guard import wrap_write_guard  # noqa: E402

from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.runner.build import build_environment  # noqa: E402

WT = json.load(open(HERE / "write_tools.json"))
FILES = sorted((HERE / "deictic_tasks_heldout").glob("*.json"))


def _built():
    """Build the 12 held-out tasks from the databases."""
    adb = json.load(open(os.path.join(gs.BASE, "airline", "db.json")))
    rdb = json.load(open(os.path.join(gs.BASE, "retail", "db.json")))
    return gh.build_airline(adb) + gh.build_retail(rdb)


def test_files_match_generator():
    """The committed task files equal the generator's output."""
    built = {t["id"]: t for t in _built()}
    assert len(FILES) == 12 and set(built) == {f.stem for f in FILES}
    for f in FILES:
        assert json.load(open(f)) == [built[f.stem]], f.name


def _call(env, name, args):
    """Send one assistant tool call to the environment; return the tool message."""
    return env.get_response(ToolCall(id="t", name=name, arguments=args, requestor="assistant"))


def _parts(path):
    """Load a task file; return (task, domain, spec, read tool, id argument, gold write)."""
    task = json.load(open(path))[0]
    dom = task["id"].split("_", 1)[0]
    spec = task["_perturb"]["spec"]
    read = task["_perturb"]["target_tool"]
    idf = spec["trigger_arg"]
    w = next(a for a in task["evaluation_criteria"]["actions"] if a["action_id"] == "w0")
    return task, dom, spec, read, idf, w


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_baseline_gold_write_passes(path):
    """Without the change, the gold write passes the CAS guard after the binding read."""
    task, dom, spec, read, idf, w = _parts(path)
    env = wrap_write_guard(build_environment(dom), set(WT[dom]))
    assert not _call(env, read, {idf: spec["referent_id"]}).error
    tm = _call(env, w["name"], w["arguments"])
    assert not tm.error, tm.content


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_perturbed_distractor_is_hard_and_guard_rejects(path):
    """After the change the old identifier names a record with the same metro area or
    product. The CAS guard rejects the gold write through it, because the stored
    record differs from what the binding read returned; a write through the new
    identifier passes after that identifier is read."""
    task, dom, spec, read, idf, w = _parts(path)
    env = wrap_write_guard(build_environment(dom), set(WT[dom]))
    env = wrap_environment(env, schedule_from_task(task))
    ref = json.loads(_call(env, read, {idf: spec["referent_id"]}).content)  # binding read; the change is now pending
    env._perturb.before(w["name"], "assistant", w["arguments"])  # apply it, as the next tool call would
    coll = getattr(env.tools.db, spec["collection"])
    stale = json.loads(json.dumps(coll[spec["referent_id"]].model_dump(), default=str))
    assert env._perturb._fired and stale != ref
    if dom == "airline":
        assert gs.METRO[stale["destination"]] == gs.METRO[ref["destination"]]
    else:
        names = lambda o: {it["name"].lower() for it in o["items"]}
        assert names(stale) & names(ref)
    tm = _call(env, w["name"], w["arguments"])
    assert tm.error and "Precondition failed" in tm.content
    new = spec["referent_new_id"]
    assert not _call(env, read, {idf: new}).error
    tm2 = _call(env, w["name"], dict(w["arguments"], **{idf: new}))
    assert not tm2.error, tm2.content
