"""End-to-end offline tests of run_perturbed.py through the real tau2 Orchestrator.

The agent and user-simulator LLMs are replaced by a script (no API calls). The tests
cover the write guards, the analysis of live orchestrator output, failure records
and checkpointing. Run in the tau2 env: pytest wear/test_run_perturbed_e2e.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_perturbed as rp  # noqa: E402

import tau2.agent.llm_agent as llm_agent  # noqa: E402
import tau2.user.user_simulator as user_sim  # noqa: E402
from tau2.data_model.message import AssistantMessage, ToolCall  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402

TASK_FILE = HERE / "deictic_tasks" / "airline_idreuse_boston.json"
USAGE = {"prompt_tokens": 10, "completion_tokens": 5}


def _raw():
    """The source task airline_idreuse_boston as a dict."""
    td = json.load(open(TASK_FILE))
    return td[0] if isinstance(td, list) else td


def _gold_write(td):
    """The task's first gold write action."""
    return next(a for a in td["evaluation_criteria"]["actions"] if a["action_id"].startswith("w"))


def _script(td, fail_at=None):
    """Scripted generate() functions for the agent and the user simulator.

    The agent reads the target record, writes, re-reads the same identifier, writes
    again, then stops; with fail_at, its call number fail_at raises instead.
    """
    old = td["_perturb"]["spec"]["referent_id"]
    w = _gold_write(td)
    steps = [("get_reservation_details", {"reservation_id": old}),
             (w["name"], dict(w["arguments"])),
             ("get_reservation_details", {"reservation_id": old}),
             (w["name"], dict(w["arguments"])),
             None]
    state = {"agent": 0, "user": 0}

    def agent_generate(*a, **k):
        i = state["agent"]
        state["agent"] += 1
        if fail_at is not None and i == fail_at:
            raise RuntimeError("relay 502")
        step = steps[min(i, len(steps) - 1)]
        if step is None:
            return AssistantMessage(role="assistant", content="Done.", usage=USAGE,
                                    raw_data={"model": "fake-1"})
        return AssistantMessage(role="assistant", content=None, usage=USAGE,
                                raw_data={"model": "fake-1"},
                                tool_calls=[ToolCall(id=f"c{i}", name=step[0], arguments=step[1],
                                                     requestor="assistant")])

    def user_generate(*a, **k):
        state["user"] += 1
        text = "Please add one checked bag to my Boston trip." if state["user"] == 1 else "###STOP###"
        return AssistantMessage(role="assistant", content=text, usage=USAGE,
                                raw_data={"model": "fake-1"})

    return agent_generate, user_generate


def _run(monkeypatch, arm, guard, fail_at=None):
    """Run one scripted rollout through run_perturbed.run_one and return its dict."""
    td = _raw()
    ag, ug = _script(td, fail_at)
    monkeypatch.setattr(llm_agent, "generate", ag)
    monkeypatch.setattr(user_sim, "generate", ug)
    sched = rp.schedule_from_task(td) if arm == "perturbed" else None
    t = rp.apply_gold_remap(td, arm, "airline")
    task = Task.model_validate({k: v for k, v in t.items() if not k.startswith("_")})
    return rp.run_one("airline", task, "openai/fake-1", sched, 300, 60, 300, write_guard=guard)


def _guard_kinds(sim):
    return [e["kind"] for e in sim.get("_write_guard_events") or []]


def test_identity_guard_blocks_both_stale_writes(monkeypatch):
    sim = _run(monkeypatch, "perturbed", "identity")
    assert sim["_perturb_fired"] and sim["_write_guard_mode"] == "identity"
    assert _guard_kinds(sim).count("write_rejected_identity") == 2
    fired = [e for e in sim["_perturb_events"] if e["kind"] == "fired"]
    assert len(fired) == 1 and fired[0]["call_id"] == "c1"  # the first write's call id


def test_analysis_on_live_orchestrator_output(monkeypatch):
    import analyze_safeguards as a
    td = _raw()
    t = rp.apply_gold_remap(td, "perturbed", "airline")
    for guard, stale, passed in ((None, 2, 1), ("cas", 1, 1), ("identity", 0, 0)):
        sim = _run(monkeypatch, "perturbed", guard)
        m = a.rollout_measures(sim, t, td, "airline", "perturbed")
        assert m["exposed"] and (m["stale_writes"], m["passed_after_reread"]) == (stale, passed), guard


def test_cas_guard_blocks_first_then_passes_after_reread(monkeypatch):
    sim = _run(monkeypatch, "perturbed", "cas")
    kinds = _guard_kinds(sim)
    assert kinds[0] == "write_rejected" and kinds[-1] == "write_allowed"


def test_no_guard_writes_the_distractor(monkeypatch):
    sim = _run(monkeypatch, "perturbed", None)
    assert sim["_perturb_fired"] and "_write_guard_events" not in sim
    tool_errors = [m for m in sim["messages"] if m["role"] == "tool" and m.get("error")]
    assert not tool_errors


@pytest.mark.parametrize("guard", ["cas", "identity"])
def test_baseline_guard_never_rejects(monkeypatch, guard):
    sim = _run(monkeypatch, "baseline", guard)
    assert not any(k.startswith("write_rejected") for k in _guard_kinds(sim))


def test_exception_returns_failure_record_with_partial_messages(monkeypatch):
    sim = _run(monkeypatch, "perturbed", "identity", fail_at=2)
    assert sim["_failure"].startswith("RuntimeError") and sim["seed"] == 300
    assert any(m.get("tool_calls") for m in sim["messages"])  # work done before the failure


def test_main_checkpoints_and_keeps_failures(monkeypatch, tmp_path):
    td = _raw()
    ag, ug = _script(td, fail_at=0)  # first rollout fails at once; later ones run
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return ag(*a, **k) if calls["n"] == 1 else _script(td)[0](*a, **k)

    monkeypatch.setattr(llm_agent, "generate", flaky)
    monkeypatch.setattr(user_sim, "generate", ug)
    out = tmp_path / "u.json"
    for k in [k for k in __import__("os").environ if k.startswith("TAU2_LLM_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("WEAR_RUN_CONFIG", json.dumps({"model": "openai/fake-1", "llm_env": {}}))
    monkeypatch.setattr(sys, "argv", ["run_perturbed.py", "--domain", "airline", "--tasks",
                                      str(TASK_FILE), "--arm", "perturbed", "--model",
                                      "openai/fake-1", "--k", "2", "--seed", "300",
                                      "--write-guard", "identity", "--out", str(out)])
    rp.main()
    d = json.loads(out.read_text())
    assert [f["seed"] for f in d["failures"]] == [300]
    assert [s["seed"] for s in d["simulations"]] == [301]
    assert d["_config"] == {"model": "openai/fake-1", "llm_env": {}}
    assert d["started"] == [300, 301]  # seeds reserved before each rollout


def test_main_refuses_when_effective_settings_differ(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_LLM_DROP_KWARGS", '["seed"]')  # e.g. restored by a dotenv file
    monkeypatch.setenv("WEAR_RUN_CONFIG", json.dumps({"llm_env": {}}))
    monkeypatch.setattr(sys, "argv", ["run_perturbed.py", "--domain", "airline", "--tasks",
                                      str(TASK_FILE), "--arm", "perturbed", "--model", "m",
                                      "--k", "1", "--out", str(tmp_path / "u.json")])
    with pytest.raises(SystemExit):
        rp.main()
    assert not (tmp_path / "u.json").exists()
