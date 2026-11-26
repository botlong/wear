#!/usr/bin/env python3
"""Expected end-state DB hash of a task, for the end-state comparison (no model calls).

Replays the task's gold write actions in a fresh tau2 environment and returns the
database hash. In the perturbed arm an ID-REUSE change is applied first (APPEND
and REORDER leave the database unchanged) and the gold writes are remapped to the
new identifier, so the hash is that of the gold state after the change. Reads a task file (one task or a list); with --results,
also compares each saved rollout's `_final_db_hash` with the expected hash.
analyze_safeguards.py uses expected_hash() for its end-state success measure.

Run (tau2 venv): tau2-bench/.venv/bin/python wear/expected_hash.py --task TASK.json --arm baseline|perturbed [--results RESULTS.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import PerturbSchedule, wrap_environment  # noqa: E402
from run_perturbed import apply_gold_remap, schedule_from_task  # noqa: E402

from tau2.registry import registry  # noqa: E402


def build_environment(domain: str):
    """Fresh tau2 environment of a domain."""
    return registry.get_env_constructor(domain)()


def expected_hash(task_dict: dict, arm: str, domain: str) -> str:
    """Replay the gold writes for one arm and return the resulting DB hash.

    The writes start from the task's initial state, after the change in the perturbed
    arm. Raises if a gold write returns an error.
    """
    env = build_environment(domain)
    # the task's initial data is loaded for both arms, before any change is applied
    init_state = task_dict.get("initial_state") or {}
    if init_state.get("initialization_actions"):
        raise NotImplementedError("initialization_actions replay not supported here")
    init = init_state.get("initialization_data") or {}
    if init.get("agent_data"):
        env.tools.update_db(init["agent_data"])
    if init.get("user_data"):
        env.user_tools.update_db(init["user_data"])
    if arm == "perturbed":
        sched = schedule_from_task(task_dict)
        assert sched is not None, "perturbed arm requires a task with _perturb"
        if sched.knob == "id_reuse":
            env = wrap_environment(env, sched)
            controller = env._perturb
            # control tasks (fire_at_start) were already changed inside wrap_environment;
            # applying the change twice would raise KeyError
            if not controller._fired:
                controller._fire_id_reuse()
                controller._fired = True
        # APPEND and REORDER change only responses, never the database
    td = apply_gold_remap(task_dict, arm, domain)
    gold = (td.get("evaluation_criteria") or {}).get("actions") or []
    from tau2.environment.tool import Tool  # noqa: F401
    write_names = _write_tools(domain)
    for a in gold:
        if a.get("name") not in write_names:
            continue
        resp = env.make_tool_call(a["name"], **(a.get("arguments") or {}))
        if getattr(resp, "error", False):
            raise RuntimeError(f"gold write errored: {a['name']} -> {resp}")
    h = env.get_db_hash()
    assert h, "empty db hash"
    return h


def _write_tools(domain: str) -> set[str]:
    """Names of the domain's write tools, from write_tools.json."""
    wt = json.load(open(Path(__file__).resolve().parent / "write_tools.json"))
    return set(wt[domain])


def main() -> None:
    """Print the expected hash of every task in --task and, with --results, compare rollouts to it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="deictic task json (single task or list)")
    ap.add_argument("--arm", required=True, choices=["baseline", "perturbed"])
    ap.add_argument("--domain", default=None, choices=[None, "airline", "retail"])
    ap.add_argument("--results", default=None,
                    help="runner output json; if given, compare _final_db_hash per sim")
    args = ap.parse_args()

    tasks = json.load(open(args.task))
    if isinstance(tasks, dict):
        tasks = [tasks]
    out = {}
    for t in tasks:
        domain = args.domain or t["id"].split("_", 1)[0]
        exp = expected_hash(t, args.arm, domain)
        out[t["id"]] = exp
        print(f"{t['id']} [{args.arm}] expected={exp}")

    if args.results:
        d = json.load(open(args.results))
        for s in d.get("simulations") or []:
            tid = s.get("task_id")
            actual = s.get("_final_db_hash")
            exp = out.get(tid)
            if exp is None:
                continue
            print(f"  sim {str(s.get('id'))[:8]} actual={str(actual)[:16]}... "
                  f"match={actual == exp}")


if __name__ == "__main__":
    main()
