"""Offline tests for the unit plan of run_safeguards.py (the safeguard study); run: pytest wear/test_run_safeguards.py."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_safeguards as rv  # noqa: E402

MODELS = ["openai/gpt-5.5", "openai/gemini-3.1-pro-preview", "openai/gemini-3.5-flash",
          "openai/deepseek-v4-pro", "openai/qwen3.7-plus"]


def test_full_plan_size_and_arms():
    units = rv.plan(MODELS, "out", 5)
    s = rv.suites()
    assert (len(s["main"]), len(s["heldout"])) == (20, 12)
    by_arm = Counter(u["name"].split("/")[0] for u in units)
    assert by_arm == {"none": 320, "policy": 320, "guard_cas": 160, "guard_identity": 160}
    assert sum(u["k"] for u in units) == 4800
    assert len({u["out"] for u in units}) == len(units)  # no output collisions
    for u in units:
        arm = u["name"].split("/")[0]
        assert (u.get("write_guard") or None) == rv.ARMS[arm].get("write_guard")
        assert bool(u.get("agent_suffix_file")) == (arm == "policy")
        assert u["sim_timeout"] == rv.SIM_TIMEOUT_S
        assert f"{os.sep}{arm}{os.sep}" in u["out"]


def test_blocks_keep_arms_together():
    units = rv.plan(MODELS, "out", 5)
    key = lambda u: (u["name"].split("/")[1], u["task_path"], u["model"], u["condition"])  # noqa: E731
    seen, prev = set(), None
    for u in units:
        if key(u) != prev:
            assert key(u) not in seen  # each block appears once, contiguously
            seen.add(key(u))
            prev = key(u)
    assert rv.plan(MODELS, "out", 5) == units  # fixed shuffle


def test_smoke_plan():
    units = rv.plan(MODELS, "out", 1, smoke=True)
    assert len(units) == 5 * 2 * 4 and all(u["condition"] == "perturbed" for u in units)
