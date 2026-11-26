#!/usr/bin/env python3
"""Stale writes and recovery from them in perturbed ID-REUSE runs (offline; no judge or API calls).

Written for the no-distractor variant, in which no record holds the old identifier,
so a write through it is rejected. For each perturbed trajectory it records whether
the first write went through the old identifier (first_write_old), whether any write
through the old identifier returned no error (effective_old), whether, after such a
first write, a later write to the new identifier returned no error (recovery), and
whether the first write went elsewhere but a later one went through the old
identifier (odd_path). Identifiers are compared exactly. A write without a tool
response counts as an error.

Usage: python wear/m2_recovery.py <outdir> --task-files f1.json f2.json ... [--out out.json]
Reads <outdir>/<model>/perturbed/<task>.json; only tasks covered by --task-files are
scored, and a task file's domain is taken from its name (airline* or retail*).
Prints a markdown table with one row per (model, task) and per-model totals; --out
writes the rows and the aggregates as JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wear.eligibility import _flat_calls  # noqa: E402

WRITE_TOOLS = {d: set(v) for d, v in json.load(
    open(os.path.join(HERE, "write_tools.json"))).items() if not d.startswith("_")}


def task_index(task_files: list[str]) -> dict[str, dict]:
    """Map task id to its old and new identifier, identifier argument and domain."""
    out = {}
    for tf in task_files:
        for t in json.load(open(tf, encoding="utf-8")):
            spec = t["_perturb"]["spec"]
            domain = "airline" if os.path.basename(tf).startswith("airline") else "retail"
            out[t["id"]] = {"old": spec["referent_id"],
                            "new": spec["referent_new_id"],
                            "id_arg": spec.get("trigger_arg"),
                            "domain": domain}
    return out


def sim_metrics(sim: dict, ti: dict) -> dict:
    """Return the per-trajectory flags described in the module docstring, plus the
    number of writes without a tool response (resp_missing)."""
    wt = WRITE_TOOLS[ti["domain"]]
    msgs = sim.get("messages") or []
    resp_err: dict = {}
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("id") is not None:
            resp_err[m["id"]] = bool(m.get("error"))

    writes = []
    for c in _flat_calls(msgs):
        if c.get("requestor") != "assistant" or c.get("name") not in wt:
            continue
        cid = c.get("call_id") if c.get("call_id") is not None else c.get("id")
        tgt = (c.get("arguments") or {}).get(ti["id_arg"])
        # a write without a tool response counts as an error, so it can be neither
        # effective nor a recovery
        err = resp_err.get(cid, True)
        writes.append({"target": tgt, "error": err,
                       "resp_missing": cid not in resp_err})

    if not writes:
        return {"no_write": True, "first_write_old": False,
                "effective_old": False, "recovery": False,
                "odd_path": False, "resp_missing": 0}
    first_old = writes[0]["target"] == ti["old"]
    return {
        "no_write": False,
        "first_write_old": first_old,
        "effective_old": any(w["target"] == ti["old"] and not w["error"]
                             for w in writes),
        "recovery": first_old and any(
            w["target"] == ti["new"] and not w["error"] for w in writes[1:]),
        "odd_path": (not first_old) and any(w["target"] == ti["old"]
                                            for w in writes[1:]),
        "resp_missing": sum(w["resp_missing"] for w in writes),
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
        for s in json.load(open(f, encoding="utf-8")).get("simulations") or []:
            rows.append({"model": model, "task": tid, **sim_metrics(s, idx[tid])})

    KEYS = ("n", "no_write", "first_old", "eff_old", "recov", "odd_path",
            "resp_missing")
    agg: dict = {}
    for r in rows:
        a = agg.setdefault((r["model"], r["task"]), {k: 0 for k in KEYS})
        a["n"] += 1
        a["no_write"] += r["no_write"]
        a["first_old"] += r["first_write_old"]
        a["eff_old"] += r["effective_old"]
        a["recov"] += r["recovery"]
        a["odd_path"] += r["odd_path"]
        a["resp_missing"] += r["resp_missing"]

    print("| model | task | n | no_write | first_write_OLD | effective_OLD | recovery | odd_path | resp_missing |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot: dict = {}
    for (mo, ta), a in sorted(agg.items()):
        t = tot.setdefault(mo, {k: 0 for k in KEYS})
        for k in KEYS:
            t[k] += a[k]
        print(f"| {mo} | {ta} | {a['n']} | {a['no_write']} | {a['first_old']} | "
              f"{a['eff_old']} | {a['recov']} | {a['odd_path']} | {a['resp_missing']} |")
    for mo, t in sorted(tot.items()):
        nw = t["n"] - t["no_write"]
        print(f"| {mo} | TOTAL | {t['n']} | {t['no_write']} | {t['first_old']} | "
              f"{t['eff_old']} | {t['recov']}/{t['first_old']} duped (with-writes {nw}) | "
              f"{t['odd_path']} | {t['resp_missing']} |")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows,
                       "agg": {f"{mo}/{ta}": a for (mo, ta), a in agg.items()}},
                      fh, ensure_ascii=False, indent=1)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
