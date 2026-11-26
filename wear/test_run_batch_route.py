"""Offline tests for run_batch.py (no API calls).

Covers endpoint routing (--route, --model-kwargs), seed allocation on resume,
merging of top-up parts, and the configuration checks that stop a unit from being
resumed under different settings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch as rb  # noqa: E402


def test_route_uses_prefix_vars(monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/qwen3.7-plus", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test/")
    key, base, kn, bn, extra = rb._endpoint_for("openai/qwen3.7-plus")
    assert (key, base, kn, bn) == ("k", "https://example.test/v1", "RELAY_API_KEY", "RELAY_BASE_URL")
    assert "max_tokens" in extra["TAU2_LLM_EXTRA_KWARGS"]


def test_base_with_v1_not_doubled(monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/deepseek-v4-pro", "DEEPSEEK")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1")
    assert rb._endpoint_for("openai/deepseek-v4-pro")[1] == "https://example.test/v1"


def test_unrouted_model_keeps_relay(monkeypatch):
    monkeypatch.setenv("RELAY_API_KEY", "a")
    monkeypatch.setenv("RELAY_BASE_URL", "https://relay-a.test")
    key, base, kn, _, _ = rb._endpoint_for("openai/qwen3.7-plus")
    assert (key, base, kn) == ("a", "https://relay-a.test/v1", "RELAY_API_KEY")


def test_missing_route_key_reported(monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/x", "NOPE")
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    monkeypatch.delenv("NOPE_BASE_URL", raising=False)
    key, base, kn, bn, _ = rb._endpoint_for("openai/x")
    assert key is None and base is None and kn == "NOPE_API_KEY"


def test_routed_gemini_uses_low_reasoning(monkeypatch):
    import json
    monkeypatch.setitem(rb.ROUTES, "openai/gemini-3.5-flash", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    extra = rb._endpoint_for("openai/gemini-3.5-flash")[4]
    pm = json.loads(extra["TAU2_LLM_PER_MODEL_KWARGS"])["openai/gemini-3.5-flash"]
    assert pm == {"reasoning_effort": "low", "max_tokens": 16384,
                  "allowed_openai_params": ["reasoning_effort"]}


def test_routed_gpt_gets_no_injected_kwargs(monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/gpt-5.5", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    extra = rb._endpoint_for("openai/gpt-5.5")[4]
    assert set(extra) == {"TAU2_LLM_EXTRA_KWARGS"}  # no decoding kwargs, only a deadline
    assert json.loads(extra["TAU2_LLM_EXTRA_KWARGS"]) == {"timeout": rb.REQUEST_TIMEOUT_S}


def test_model_kwargs_merge_into_routed_model(monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/gpt-5.5", "RELAY")
    monkeypatch.setitem(rb.MODEL_KWARGS, "openai/gpt-5.5", {"reasoning_effort": "low"})
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    extra = rb._endpoint_for("openai/gpt-5.5")[4]
    assert json.loads(extra["TAU2_LLM_PER_MODEL_KWARGS"]) == {"openai/gpt-5.5": {"reasoning_effort": "low"}}


def test_next_seed_never_reuses_an_attempted_seed(tmp_path):
    p = tmp_path / "u.json"
    assert rb.next_seed(str(p)) == 300  # fresh unit
    p.write_text(json.dumps({"simulations": [{"seed": s} for s in (300, 301)]}))
    assert rb.next_seed(str(p)) == 302  # plain resume
    # seed 304 was moved out and 305 failed: neither is reused
    p.write_text(json.dumps({"simulations": [{"seed": s} for s in (300, 301, 302, 303)],
                             "failures": [{"seed": 305}]}))
    (tmp_path / "u.quarantine.json").write_text(json.dumps({"simulations": [{"seed": 304}]}))
    assert rb.next_seed(str(p)) == 306
    # every rollout moved out: still no restart at 300
    p.write_text(json.dumps({"simulations": []}))
    assert rb.next_seed(str(p)) == 305


def test_merge_part_keeps_failures_and_config(tmp_path):
    out, part = tmp_path / "u.json", tmp_path / "u.json.part2"
    out.write_text(json.dumps({"tasks": [1], "simulations": [{"seed": 300}],
                               "failures": [{"seed": 301}], "_config": {"a": 1}}))
    part.write_text(json.dumps({"tasks": [1], "simulations": [{"seed": 302}],
                                "failures": [{"seed": 303}], "_config": {"a": 1}}))
    rb.merge_part(str(out), str(part))
    d = json.loads(out.read_text())
    assert [s["seed"] for s in d["simulations"]] == [300, 302]
    assert [s["seed"] for s in d["failures"]] == [301, 303]
    assert d["_config"] == {"a": 1} and not part.exists()


def test_run_unit_refuses_resume_with_different_config(tmp_path, monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/deepseek-v4-pro", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    task = tmp_path / "airline_idreuse_x.json"
    task.write_text("[]")
    u = rb.unit_spec("airline", str(task), "perturbed", "openai/deepseek-v4-pro",
                     str(tmp_path / "out"), 5, write_guard="cas")
    import os
    os.makedirs(os.path.dirname(u["out"]))
    cfg = rb.unit_config(dict(u, write_guard="identity"), "https://example.test/v1",
                         rb._endpoint_for(u["model"])[4])
    with open(u["out"], "w") as f:
        json.dump({"simulations": [], "_config": cfg}, f)
    name, ok, err = rb.run_unit(u, 10)
    assert not ok and "config differs" in err


def test_unit_config_has_no_key(monkeypatch):
    monkeypatch.setenv("RELAY_GEMINI_API_KEY", "secret")
    monkeypatch.setenv("RELAY_BASE_URL", "https://relay-b.test")
    u = rb.unit_spec("airline", __file__, "baseline", "openai/gemini-x", "o", 5)
    cfg = rb.unit_config(u, "https://relay-a.test/v1", rb._endpoint_for(u["model"])[4])
    assert "secret" not in json.dumps(cfg)


def test_unit_spec_carries_guard_mode():
    u = rb.unit_spec("airline", "x/airline_idreuse_boston.json", "perturbed", "m", "out", 5,
                     write_guard="identity")
    assert u["write_guard"] == "identity"


def test_orphan_part_recovered_and_started_seeds_retired(tmp_path):
    out, part = tmp_path / "u.json", tmp_path / "u.json.part2"
    out.write_text(json.dumps({"simulations": [{"seed": 300}, {"seed": 301}],
                               "started": [300, 301]}))
    # a killed driver left a top-up with one finished rollout and one in flight
    part.write_text(json.dumps({"simulations": [{"seed": 302}], "started": [302, 303]}))
    assert rb.recover_parts(str(out)) == 1 and not part.exists()
    d = json.loads(out.read_text())
    assert [s["seed"] for s in d["simulations"]] == [300, 301, 302]
    assert rb.next_seed(str(out)) == 304  # 303 was started, never reused


def test_check_config_covers_completed_units(tmp_path, monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/deepseek-v4-pro", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    task = tmp_path / "airline_idreuse_x.json"
    task.write_text("[]")
    u = rb.unit_spec("airline", str(task), "perturbed", "openai/deepseek-v4-pro",
                     str(tmp_path / "out"), 1)
    import os
    os.makedirs(os.path.dirname(u["out"]))
    _, cfg, _ = rb.resolve(u)
    with open(u["out"], "w") as f:  # completed unit
        json.dump({"simulations": [{"seed": 300}], "_config": cfg}, f)
    assert rb.done(u) and rb.check_config(u) is None
    monkeypatch.setitem(rb.MODEL_KWARGS, "openai/deepseek-v4-pro", {"temperature": 0})
    assert "config differs" in rb.check_config(u)


def test_interrupted_merge_is_not_applied_twice(tmp_path):
    out, part = tmp_path / "u.json", tmp_path / "u.json.part1"
    out.write_text(json.dumps({"simulations": [{"id": "a", "seed": 300}, {"id": "b", "seed": 301}],
                               "started": [300, 301]}))
    # the merge had replaced u.json but died before removing the part
    part.write_text(json.dumps({"simulations": [{"id": "b", "seed": 301}], "started": [301]}))
    rb.recover_parts(str(out))
    assert [s["seed"] for s in json.loads(out.read_text())["simulations"]] == [300, 301]


def test_part_with_other_config_is_set_aside(tmp_path):
    out, part = tmp_path / "u.json", tmp_path / "u.json.part1"
    out.write_text(json.dumps({"simulations": [], "_config": {"a": 1, "code": "x"}}))
    part.write_text(json.dumps({"simulations": [{"id": "c", "seed": 300}], "_config": {"a": 2}}))
    import pytest
    with pytest.raises(RuntimeError):
        rb.recover_parts(str(out))
    assert (tmp_path / "u.json.part1.conflict").exists()
    assert json.loads(out.read_text())["simulations"] == []


def test_existing_output_without_config_is_unverifiable(tmp_path, monkeypatch):
    monkeypatch.setitem(rb.ROUTES, "openai/deepseek-v4-pro", "RELAY")
    monkeypatch.setenv("RELAY_API_KEY", "k")
    monkeypatch.setenv("RELAY_BASE_URL", "https://example.test")
    task = tmp_path / "airline_idreuse_x.json"
    task.write_text("[]")
    u = rb.unit_spec("airline", str(task), "perturbed", "openai/deepseek-v4-pro", str(tmp_path / "o"), 1)
    import os
    os.makedirs(os.path.dirname(u["out"]))
    open(u["out"], "w").write(json.dumps({"simulations": [{"seed": 300}]}))
    assert "unverifiable" in rb.check_config(u)
