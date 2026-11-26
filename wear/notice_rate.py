#!/usr/bin/env python3
"""Notice rate of perturbed ID-REUSE trajectories (offline; no judge or API calls).

For each perturbed trajectory, runs the timing check (eligibility.assess) and
records whether the new identifier appeared in a tool result between the binding
read and the first write (exposed), whether the agent noticed it, i.e. the new
identifier appears in the agent's message text or tool-call arguments before its
first write (notice), and whether the first write went through the old identifier
(wrong_first). The notice rate is the share of trajectories with notice.

Usage: python wear/notice_rate.py <outdir> --task-files f1.json f2.json ... [--out out.json]
Reads <outdir>/<model>/perturbed/<task>.json; only tasks covered by --task-files are
scored, and a task file's domain is taken from its name (airline* or retail*).
Prints a markdown table with one row per (model, task); --out writes the rows and the
aggregates as JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wear.eligibility import assess, _flat_calls  # noqa: E402

WRITE_TOOLS = {d: set(v) for d, v in json.load(
    open(os.path.join(HERE, "write_tools.json"))).items() if not d.startswith("_")}


def task_index(task_files: list[str]) -> dict[str, dict]:
    """Map task id to its perturbation details: old and new identifier, spec, target
    tool, fire_after count, domain and the identifier argument (id_arg)."""
    out = {}
    for tf in task_files:
        for t in json.load(open(tf)):
            p = t["_perturb"]
            spec = p["spec"]
            domain = "airline" if os.path.basename(tf).startswith("airline") else "retail"
            out[t["id"]] = {
                "old": spec["referent_id"],
                "new": spec["referent_new_id"],
                "spec": spec,
                "target_tool": p.get("target_tool"),
                "fire_after": p.get("fire_after_n_calls", 1),
                "domain": domain,
                "id_arg": spec.get("trigger_arg"),
            }
    return out


def _contains_id(text: str, eid: str) -> bool:
    """True if eid occurs in text as a whole token (no letter or digit on either side),
    ignoring case."""
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(eid) + r"(?![A-Za-z0-9])",
                     text, re.IGNORECASE) is not None


def sim_metrics(sim: dict, ti: dict) -> dict:
    """Return the per-trajectory fields: timing-check label, exposed, notice,
    has_write, first_write_id, wrong_first and the re-check reads."""
    wt = WRITE_TOOLS[ti["domain"]]
    e = assess(sim, ti["spec"], wt, ti["fire_after"], ti["target_tool"])
    msgs = sim.get("messages") or []
    calls = _flat_calls(msgs)
    w_idx = e.first_write_flat_index

    first_write_id = None
    w_call_id = None
    if w_idx is not None:
        wc = calls[w_idx]
        first_write_id = (wc.get("arguments") or {}).get(ti["id_arg"])
        w_call_id = wc.get("call_id")

    # notice in tool-call arguments: any agent call before the first write
    notice_args = False
    if w_idx is not None:
        for c in calls[:w_idx]:
            if c["requestor"] == "assistant" and _contains_id(json.dumps(
                    c.get("arguments") or {}, ensure_ascii=False), ti["new"]):
                notice_args = True
                break

    # notice in message text: assistant messages up to and including the one that
    # makes the first write
    notice_text = False
    if w_idx is not None:
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            is_write_msg = any(
                (tc.get("id") == w_call_id or tc.get("call_id") == w_call_id)
                if w_call_id is not None else (tc.get("name") in wt)
                for tc in tcs)
            c = m.get("content")
            if isinstance(c, str) and _contains_id(c, ti["new"]):
                notice_text = True
            if is_write_msg:
                break

    return {
        "label": e.label,
        "exposed": e.new_id_observed,
        "notice": (notice_text or notice_args) if w_idx is not None else False,
        "has_write": w_idx is not None,
        "first_write_id": first_write_id,
        "wrong_first": (first_write_id == ti["old"]) if first_write_id else None,
        "recheck_reads": e.recheck_reads,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--task-files", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    idx = task_index(args.task_files)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.outdir, "*", "perturbed", "*.json"))):
        tid = os.path.basename(f)[:-5]
        if tid not in idx:
            continue
        model = f.split(os.sep)[-3]
        d = json.load(open(f))
        for s in d.get("simulations") or []:
            m = sim_metrics(s, idx[tid])
            rows.append({"model": model, "task": tid, **m})

    key = lambda r: (r["model"], r["task"])
    agg: dict = {}
    for r in rows:
        a = agg.setdefault(key(r), dict(n=0, write=0, wrong=0, rechecked=0,
                                        exposed=0, notice=0, anomalies=0))
        a["n"] += 1
        a["write"] += r["has_write"]
        a["wrong"] += bool(r["wrong_first"])
        a["rechecked"] += r["label"] == "rechecked"
        a["exposed"] += r["exposed"]
        a["notice"] += r["notice"]
        a["anomalies"] += r["label"] not in ("rechecked", "guardrail_satisfied")

    print("| model | task | n | writes | wrong1st | rechecked | exposed | notice | anom |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (mo, ta), a in sorted(agg.items()):
        print(f"| {mo} | {ta} | {a['n']} | {a['write']} | {a['wrong']} | "
              f"{a['rechecked']} | {a['exposed']} | {a['notice']} | {a['anomalies']} |")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows,
                       "agg": {f"{mo}/{ta}": a for (mo, ta), a in agg.items()}},
                      f, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
