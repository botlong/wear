#!/usr/bin/env python3
"""Aggregate a run_batch.py output tree into per-(model, condition, domain) WEAR cells.

Reads every <outdir>/<model_slug>/<condition>/<task>.json results file whose task id
starts with airline_ or retail_, scores it with scorer.py, and, for the perturbed and
control conditions, runs the timing check (eligibility.py) with the task
specifications in --tasks-dir. For each cell it reports attempted and effective WEAR
(the mean of task means, with a two-stage bootstrap 95% interval), the number of
tasks with at least one wrong-entity write (with a Wilson 95% interval), and the
counts of timing-check labels. For each model and domain it also reports a
two-sided paired sign-flip p-value (Monte Carlo) over task means, for baseline vs
perturbed and control vs perturbed. Trajectories without a write, or without an
eligible write, have no defined WEAR and are left out. No API calls.

Usage: python wear/score_batch.py <outdir> [--tasks-dir wear/deictic_tasks] [--out report.json]
Prints a markdown table to stdout; --out also writes the report as JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligibility import assess_results, load_perturb_index  # noqa: E402
from scorer import load_write_tools, score_results  # noqa: E402
from stats import paired_permutation, task_means, two_stage_bootstrap_ci, wilson  # noqa: E402


def walk_units(outdir: str, keep=None) -> list[dict]:
    """List the results files outdir/<model>/<condition>/<task>.json of the airline and
    retail domains, as dicts with path, model, condition, task and domain.

    keep(task), if given, selects tasks by id.
    """
    units = []
    for p in sorted(glob.glob(os.path.join(outdir, "*", "*", "*.json"))):
        rel = os.path.relpath(p, outdir).split(os.sep)
        if len(rel) != 3:
            continue
        mslug, condition, fname = rel
        task = fname[:-5]
        domain = task.split("_", 1)[0]
        if domain not in ("airline", "retail"):
            continue
        if keep is not None and not keep(task):
            continue
        units.append({"path": p, "model": mslug, "condition": condition,
                      "task": task, "domain": domain})
    return units


def collect(outdir: str, tasks_dir: str, keep=None) -> dict[str, Any]:
    """Score all results files and group the per-trajectory values by
    (model, condition, domain); also list the trajectories without a write."""
    wt = load_write_tools()
    write_tools_by_domain = {d: set(wt[d]) for d in ("airline", "retail")}
    pidx = load_perturb_index(sorted(glob.glob(os.path.join(tasks_dir, "*.json"))))

    # cells[(model, condition, domain)] -> {"wear": {task: [values]},
    #   "eff": {task: [values]}, "wrongtask": {task: any wrong write},
    #   "elig": {timing-check label: n}, "n_sims": n}
    cells: dict[tuple, dict[str, Any]] = {}
    n_sims_total = 0
    missing_writes: list[str] = []
    for u in walk_units(outdir, keep):
        results = json.load(open(u["path"]))
        rows = score_results(results, u["domain"])
        key = (u["model"], u["condition"], u["domain"])
        c = cells.setdefault(key, {"wear": {}, "eff": {}, "wrongtask": {},
                                   "elig": {}, "n_sims": 0})
        for r in rows:
            n_sims_total += 1
            c["n_sims"] += 1
            if r.n_write_calls == 0:
                missing_writes.append(f"{u['model']}/{u['condition']}/{u['task']}")
                continue
            if r.no_eligible_write:  # WEAR is undefined: leave the trajectory out
                continue
            c["wear"].setdefault(u["task"], []).append(r.wear)
            eff = r.effective_wear
            c["eff"].setdefault(u["task"], []).append(
                eff if eff is not None else 0.0)
            c["wrongtask"][u["task"]] = (
                c["wrongtask"].get(u["task"], False) or bool(r.has_wrong_bind))
        if u["condition"] in ("perturbed", "control"):
            for e in assess_results(results, pidx,
                                    write_tools_by_domain[u["domain"]]):
                c["elig"][e.label] = c["elig"].get(e.label, 0) + 1
    return {"cells": cells, "n_sims_total": n_sims_total,
            "no_write_sims": missing_writes}


def cell_summary(c: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Summarize one cell: mean of task means with bootstrap intervals for attempted
    and effective WEAR, tasks with a wrong write with a Wilson interval, and the
    timing-check labels. None if no trajectory in the cell has a defined WEAR."""
    if not c["wear"]:
        return None
    point, lo, hi = two_stage_bootstrap_ci(c["wear"])
    epoint, elo, ehi = two_stage_bootstrap_ci(c["eff"])
    n_tasks = len(c["wear"])
    k_wrong = sum(bool(v) for v in c["wrongtask"].values())
    wlo, whi = wilson(k_wrong, n_tasks)
    return {
        "n_tasks": n_tasks, "n_sims": c["n_sims"],
        "wear_mean": round(point, 4), "wear_ci95": [round(lo, 4), round(hi, 4)],
        "eff_wear_mean": round(epoint, 4), "eff_wear_ci95": [round(elo, 4), round(ehi, 4)],
        "tasks_with_wrong_bind": f"{k_wrong}/{n_tasks}",
        "wilson95_taskwrong": [round(wlo, 4), round(whi, 4)],
        "eligibility": c["elig"] or None,
    }


def paired_p(cells: dict, model: str, domain: str, cond_a: str, cond_b: str) -> Optional[float]:
    """Two-sided paired sign-flip p-value (Monte Carlo) between two conditions of one
    model and domain, over the tasks scored in both. None if fewer than two tasks."""
    a = cells.get((model, cond_a, domain))
    b = cells.get((model, cond_b, domain))
    if not a or not b:
        return None
    ta, tb = task_means(a["wear"]), task_means(b["wear"])
    common = sorted(set(ta) & set(tb))
    if len(common) < 2:
        return None
    _, p = paired_permutation({t: (ta[t], tb[t]) for t in common})
    return round(p, 5)


def build_report(outdir: str, tasks_dir: str, keep=None) -> dict[str, Any]:
    """Build the report dict (cells, paired_p, n_sims_total, no_write_sims) from the
    results files under outdir."""
    data = collect(outdir, tasks_dir, keep)
    cells = data["cells"]
    report: dict[str, Any] = {"cells": {}, "paired_p": {},
                              "n_sims_total": data["n_sims_total"],
                              "no_write_sims": data["no_write_sims"]}
    for key in sorted(cells):
        s = cell_summary(cells[key])
        if s is None:
            continue
        report["cells"]["/".join(key)] = s
    models = sorted({k[0] for k in cells})
    domains = sorted({k[2] for k in cells})
    for m in models:
        for d in domains:
            for a, b in (("baseline", "perturbed"), ("control", "perturbed")):
                p = paired_p(cells, m, d, a, b)
                if p is not None:
                    report["paired_p"][f"{m}/{d}/{a}-vs-{b}"] = p
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--tasks-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deictic_tasks"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = build_report(args.outdir, args.tasks_dir)

    print("| model | domain | condition | tasks | sims | WEAR [CI95] | "
          "eff-WEAR | wrong-tasks (Wilson95) | eligibility |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, s in report["cells"].items():
        model, condition, domain = key.split("/")
        elig = " ".join(f"{k}:{v}" for k, v in sorted((s["eligibility"] or {}).items()))
        print(f"| {model} | {domain} | {condition} | {s['n_tasks']} | {s['n_sims']} "
              f"| {s['wear_mean']} {s['wear_ci95']} | {s['eff_wear_mean']} "
              f"| {s['tasks_with_wrong_bind']} {s['wilson95_taskwrong']} | {elig} |")
    print("\npaired permutation p (task-level, two-sided):")
    for k, p in report["paired_p"].items():
        m, d, ab = k.split("/")
        a, b = ab.split("-vs-")
        print(f"  {m} {d} {a} vs {b}: p={p}")
    if report["no_write_sims"]:
        print(f"\nzero-write trajectories: {len(report['no_write_sims'])} (WEAR undefined, excluded)",
              file=sys.stderr)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
