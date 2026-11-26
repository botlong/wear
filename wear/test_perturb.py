"""Offline tests for perturb.py: timing and database change of ID-REUSE.

Run in the tau2 env: pytest wear/test_perturb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import PerturbSchedule, wrap_environment  # noqa: E402

from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.runner.build import build_environment  # noqa: E402


def _sched():
    """ID-REUSE schedule armed by the first read of NO6JO3: then NO6JO3 moves to ZZ0001
    and the distractor HKEG34 takes over NO6JO3."""
    return PerturbSchedule(
        knob="id_reuse",
        target_tool="get_reservation_details",
        fire_after_n_calls=1,
        spec={
            "trigger_arg": "reservation_id",
            "trigger_arg_value": "NO6JO3",
            "collection": "reservations",
            "owner_list_attr": "reservations",
            "referent_id": "NO6JO3",
            "referent_new_id": "ZZ0001",
            "distractor_id": "HKEG34",
        },
    )


def _sched_at_start():
    s = _sched()
    s.fire_at_start = True
    return s


def _read(env, rid):
    tm = env.get_response(
        ToolCall(id="t", name="get_reservation_details",
                 arguments={"reservation_id": rid}, requestor="assistant")
    )
    return tm


def _content(tm):
    try:
        return json.loads(tm.content)
    except Exception:
        return tm.content


def test_id_reuse_timing_and_mutation():
    env = build_environment("airline")
    env = wrap_environment(env, _sched())
    db = env.tools.db

    assert db.reservations["NO6JO3"].destination == "DFW"
    assert db.reservations["HKEG34"].destination == "LAS"
    assert "ZZ0001" not in db.reservations
    assert db.users["mia_li_3668"].reservations[:3] == ["NO6JO3", "AIXC49", "HKEG34"]

    tm1 = _read(env, "NO6JO3")
    c1 = _content(tm1)
    assert not tm1.error, tm1.content
    assert c1["destination"] == "DFW", "trigger read already shows the distractor (timing bug)"
    assert env._perturb._armed and not env._perturb._fired

    tm2 = _read(env, "NO6JO3")
    c2 = _content(tm2)
    assert not tm2.error, tm2.content
    assert env._perturb._fired
    assert c2["destination"] == "LAS", "NO6JO3 was not repointed to the distractor"
    assert c2["reservation_id"] == "NO6JO3"

    tmg = _read(env, "ZZ0001")
    cg = _content(tmg)
    assert not tmg.error, tmg.content
    assert cg["destination"] == "DFW"
    assert cg["reservation_id"] == "ZZ0001"

    assert db.reservations["ZZ0001"].destination == "DFW"
    assert db.reservations["NO6JO3"].destination == "LAS"
    assert "HKEG34" not in db.reservations
    lst = db.users["mia_li_3668"].reservations
    assert "ZZ0001" in lst and "NO6JO3" in lst and "HKEG34" not in lst

    kinds = [e["kind"] for e in env._perturb.events]
    assert kinds.count("armed") == 1
    assert kinds.count("id_reuse.referent_moved") == 1
    assert kinds.count("id_reuse.distractor_moved") == 1
    print("OK test_id_reuse_timing_and_mutation")


def test_non_referent_read_does_not_trigger():
    env = build_environment("airline")
    env = wrap_environment(env, _sched())
    _read(env, "AIXC49")
    assert not env._perturb._armed and not env._perturb._fired
    assert env.tools.db.reservations["NO6JO3"].destination == "DFW"
    print("OK test_non_referent_read_does_not_trigger")


def test_fire_at_start_no_stale_cache():
    """fire_at_start applies the change before the rollout (the control condition), so the
    first read of the old identifier already returns the distractor (no stale cache)."""
    env = build_environment("airline")
    env = wrap_environment(env, _sched_at_start())
    db = env.tools.db
    assert env._perturb._fired
    assert db.reservations["ZZ0001"].destination == "DFW"
    assert db.reservations["NO6JO3"].destination == "LAS"
    c = _content(_read(env, "NO6JO3"))
    assert c["destination"] == "LAS"
    assert _content(_read(env, "ZZ0001"))["destination"] == "DFW"
    print("OK test_fire_at_start_no_stale_cache")


if __name__ == "__main__":
    test_id_reuse_timing_and_mutation()
    test_non_referent_read_does_not_trigger()
    test_fire_at_start_no_stale_cache()
    print("ALL PASS")
