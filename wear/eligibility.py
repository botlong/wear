#!/usr/bin/env python3
"""Timing check: did the agent read anything between the binding read and its first write?

The binding read is the call that triggers the task's perturbation: the n-th agent
call to the task's target tool whose trigger argument has the trigger value (from
the task's _perturb entry). Each trajectory gets one label:
  rechecked             the agent made at least one read after the binding read and
                        before its first write
  guardrail_satisfied   no such read (the paper's "not rechecked")
  no_trigger            the binding read never happened
  write_before_trigger  the first write came at or before the binding read
The check also records verified exposure: whether the new identifier (ID-REUSE) or
the decoy's identifier (APPEND) appeared in a tool result in that window.

Usage: python3 wear/eligibility.py --results RESULTS.json --tasks TASKS.json... --domain airline|retail
       [--write-tools write_tools.json]
Prints one JSON line per trajectory, then one line with the aggregate counts.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Per-trajectory assessment
# --------------------------------------------------------------------------- #


@dataclass
class Eligibility:
    """Timing-check result for one trajectory.

    The *_flat_index fields are positions in the flattened list of tool calls
    (_flat_calls); recheck_reads names the agent reads in the window.
    """
    task_id: str
    sim_id: Optional[str]
    trigger_found: bool
    trigger_flat_index: Optional[int]
    first_write_flat_index: Optional[int]
    first_write_name: Optional[str]
    recheck_any: bool
    recheck_reads: list[str] = field(default_factory=list)
    new_id_observed: bool = False
    anomaly: Optional[str] = None  # no_trigger | write_before_trigger

    @property
    def label(self) -> str:
        """The anomaly if any, else "rechecked" or "guardrail_satisfied" (not rechecked)."""
        if self.anomaly:
            return self.anomaly
        return "rechecked" if self.recheck_any else "guardrail_satisfied"


def _flat_calls(messages: list[dict]) -> list[dict]:
    """Flatten the tool calls of all assistant-role messages, in order, into dicts with
    name, arguments, requestor and call_id."""
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            out.append({
                "name": tc.get("name"),
                "arguments": dict(tc.get("arguments") or {}),
                "requestor": tc.get("requestor", "assistant"),
                "call_id": tc.get("id"),
            })
    return out


def _results_by_call_id(messages: list[dict]) -> dict[str, str]:
    """Map each tool-call id to the content of its tool message."""
    out: dict[str, str] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("id") is not None:
            out[m["id"]] = str(m.get("content") or "")
    return out


def _norm(v: Any) -> str:
    """Normalize an identifier: str(), strip whitespace, lowercase."""
    return str(v).strip().lower()


def assess(
    sim: dict,
    perturb_spec: dict,
    write_tools: set[str],
    fire_after_n_calls: int = 1,
    target_tool: Optional[str] = None,
) -> Eligibility:
    """Run the timing check on one trajectory and return its Eligibility.

    perturb_spec, fire_after_n_calls and target_tool come from the task's _perturb
    entry. Calls requested by the user simulator never count as the binding read.
    """
    messages = sim.get("messages") or []
    calls = _flat_calls(messages)
    results = _results_by_call_id(messages)

    trig_arg = perturb_spec.get("trigger_arg")
    trig_val = _norm(perturb_spec.get("trigger_arg_value"))
    new_id = (_norm(perturb_spec.get("referent_new_id", ""))
              or _norm(perturb_spec.get("append_item", ""))) or None

    trigger_idx: Optional[int] = None
    hits = 0
    for i, c in enumerate(calls):
        if c["requestor"] != "assistant":
            continue
        if target_tool and c["name"] != target_tool:
            continue
        if trig_arg and _norm(c["arguments"].get(trig_arg)) != trig_val:
            continue
        hits += 1
        if hits >= fire_after_n_calls:
            trigger_idx = i
            break

    write_idx: Optional[int] = None
    for i, c in enumerate(calls):
        if c["requestor"] == "assistant" and c["name"] in write_tools:
            write_idx = i
            break

    e = Eligibility(
        task_id=sim.get("task_id", "?"),
        sim_id=sim.get("id"),
        trigger_found=trigger_idx is not None,
        trigger_flat_index=trigger_idx,
        first_write_flat_index=write_idx,
        first_write_name=calls[write_idx]["name"] if write_idx is not None else None,
        recheck_any=False,
    )
    if trigger_idx is None:
        e.anomaly = "no_trigger"
        return e
    if write_idx is not None and write_idx <= trigger_idx:
        e.anomaly = "write_before_trigger"
        return e

    # Window: calls after the binding read and before the first agent write (to the
    # end if there is no write). Only agent reads count as a re-check; the tool
    # result of any call, also one the user simulator made, counts for exposure.
    lo, hi = trigger_idx + 1, (write_idx if write_idx is not None else len(calls))
    for c in calls[lo:hi]:
        if c["name"] in write_tools:
            continue
        if c["requestor"] == "assistant":
            e.recheck_any = True
            e.recheck_reads.append(c["name"])
        if new_id and c["call_id"] is not None:
            if re.search(r"(?<![a-z0-9])" + re.escape(new_id) + r"(?![a-z0-9])",
                         _norm(results.get(c["call_id"], ""))):
                e.new_id_observed = True
    return e


# --------------------------------------------------------------------------- #
# Batch CLI
# --------------------------------------------------------------------------- #


def load_perturb_index(task_files: list[str]) -> dict[str, dict]:
    """Map task id to its _perturb entry, read from task JSON files (lists of tasks,
    as in deictic_tasks/); tasks without a _perturb entry are skipped."""
    idx: dict[str, dict] = {}
    for path in task_files:
        for t in json.load(open(path)):
            if t.get("_perturb"):
                idx[t["id"]] = t["_perturb"]
    return idx


def assess_results(
    results: dict, perturb_index: dict[str, dict], write_tools: set[str]
) -> list[Eligibility]:
    """Run the timing check on every simulation of a results dict whose task is in
    perturb_index."""
    out = []
    for sim in results.get("simulations") or []:
        p = perturb_index.get(sim.get("task_id"))
        if p is None:
            continue
        out.append(assess(
            sim,
            p.get("spec") or {},
            write_tools,
            fire_after_n_calls=int(p.get("fire_after_n_calls", 1)),
            target_tool=p.get("target_tool"),
        ))
    return out


def aggregate(rows: list[Eligibility]) -> dict[str, Any]:
    """Count the labels, and the trajectories in which the new identifier was observed."""
    n = len(rows)
    lab = {}
    for r in rows:
        lab[r.label] = lab.get(r.label, 0) + 1
    return {
        "n": n,
        "labels": lab,
        "guardrail_satisfied": lab.get("guardrail_satisfied", 0),
        "rechecked": lab.get("rechecked", 0),
        "new_id_observed": sum(r.new_id_observed for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="runner results.json")
    ap.add_argument("--tasks", required=True, nargs="+",
                    help="source deictic task json(s) containing _perturb")
    ap.add_argument("--domain", required=True, choices=["airline", "retail"])
    ap.add_argument("--write-tools", default=None,
                    help="path to write_tools.json (default: alongside this file)")
    args = ap.parse_args()

    import os
    wt_path = args.write_tools or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "write_tools.json")
    write_tools = set(json.load(open(wt_path))[args.domain])

    results = json.load(open(args.results))
    pidx = load_perturb_index(args.tasks)
    rows = assess_results(results, pidx, write_tools)
    for r in rows:
        print(json.dumps({
            "task_id": r.task_id, "sim_id": r.sim_id, "label": r.label,
            "recheck_reads": r.recheck_reads,
            "new_id_observed": r.new_id_observed,
            "first_write": r.first_write_name,
        }, ensure_ascii=False))
    print(json.dumps({"aggregate": aggregate(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
