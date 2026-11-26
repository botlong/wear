"""Label the recovery-suite trajectories that do not return to the gold state.

In the recovery suite (GPT-5.5, retail, perturbed condition) the write through the
old identifier succeeds on a distractor, and a later user turn prompts the agent to
check again. A wrong-binding trajectory is "fail_wrong" when its final database hash
differs from the expected one (the gold actions replayed on the perturbed database).
Each fail_wrong trajectory is labeled by the order of its writes to the old
identifier (OLD) and to the new identifier (NEW, the gold target after the change):
  poisoned_recovery    an OLD write comes after the last NEW write
  incomplete_recovery  a NEW write with no OLD write after it; the end state is
                       still wrong
  single_no_recovery   no NEW write at all
These are the three outcomes of the unrecovered trajectories reported for the
recovery regime in the harm analysis (7 / 4 / 1 of 12).

Usage (repo root): tau2-bench/.venv/bin/python wear/paper/forensic_recover.py
Run directly, it prints one Markdown row per trajectory and checks the 7/4/1 split
and the per-task counts against NUMBERS.json (so collect_numbers.py must have run
first); it writes no file.
Reads: the recovery-suite run files under wear/runs/ (RECOVER_POOL_GLOBS), the task
files in wear/deictic_tasks/, and wear/paper/NUMBERS.json.
collect_numbers.py imports classify_recover() and stores its result under
recover.forensic in NUMBERS.json.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEAR = ROOT / "wear"
sys.path.insert(0, str(WEAR))

from scorer import (  # noqa: E402
    _norm, extract_agent_write_calls, extract_gold_write_actions,
    load_write_tools, score_results,
)
from expected_hash import expected_hash  # noqa: E402

RUNS = WEAR / "runs"
TASKS = WEAR / "deictic_tasks"
NUMBERS_JSON = WEAR / "paper" / "NUMBERS.json"

# The recovery-suite run files. collect_numbers.py imports this list, so both
# scripts score the same files.
RECOVER_POOL_GLOBS = [
    "recovery_suite/*_pert_gpt55_s300.json",
    "recovery_suite/*_k4_s301.json",
    "recovery_suite/retail_recover_ereader_pert_gpt55.json",
    "recovery_suite/retail_recover_backpack_pert_gpt55.json",
]

_ORDER_KEY = "order_id"          # target-identifier argument of the retail write tools
LABELS = ("poisoned_recovery", "incomplete_recovery", "single_no_recovery")


def _require(cond: bool, msg: str) -> None:
    """Raise AssertionError with msg unless cond holds; unlike assert, also under python -O."""
    if not cond:
        raise AssertionError(msg)


def _old_new(sim: dict) -> tuple[str, str]:
    """Return (old identifier, new identifier) from the rollout's single
    id_reuse.referent_moved perturbation event; raises if there is not exactly one."""
    moved = [ev for ev in (sim.get("_perturb_events") or [])
             if ev.get("kind") == "id_reuse.referent_moved"]
    _require(len(moved) == 1,
             f"expected exactly 1 id_reuse.referent_moved event, got {len(moved)}: {moved}")
    old, new = moved[0].get("old"), moved[0].get("new")
    _require(bool(old) and bool(new), f"referent_moved missing old/new: {moved[0]}")
    return old, new


def _write_seq(sim: dict, gold_names: set[str], old: str, new: str) -> list[dict]:
    """Return the agent's writes through gold-action tools, in order, as
    [{order_id, kind, errored}] with kind OLD, NEW or OTHER (any other target)."""
    o_norm, n_norm = _norm(old), _norm(new)
    seq: list[dict] = []
    for c in extract_agent_write_calls(sim.get("messages") or [], gold_names):
        oid = c.arguments.get(_ORDER_KEY)
        k = _norm(oid)
        kind = "NEW" if k == n_norm else ("OLD" if k == o_norm else "OTHER")
        seq.append({"order_id": oid, "kind": kind, "errored": bool(c.errored)})
    return seq


def _classify(seq: list[dict]) -> str:
    """Return the label of one fail_wrong write sequence (see the module docstring);
    raises if the sequence is empty or contains an OTHER target."""
    kinds = [w["kind"] for w in seq]
    _require(bool(kinds), "fail_wrong sim has an empty shape_ok write sequence (contradicts has_wrong_bind)")
    _require("OTHER" not in kinds,
             f"fail_wrong write sequence contains non-OLD/NEW targets (unexpected): {kinds}")
    if "NEW" not in kinds:
        return "single_no_recovery"                 # never wrote to the new identifier
    last_new = len(kinds) - 1 - kinds[::-1].index("NEW")
    return ("poisoned_recovery" if "OLD" in kinds[last_new + 1:]
            else "incomplete_recovery")


def classify_recover() -> dict:
    """Find and label the fail_wrong trajectories of the recovery suite.

    Uses the same files and the same fail_wrong rule as collect_numbers.collect_recover.
    Returns {counts (per label), per_traj (one row per trajectory), n_fail_wrong,
    n_pool_files}.
    """
    wt_data = load_write_tools()
    write_tools = set(wt_data["retail"])

    files: list[str] = []
    for g in RECOVER_POOL_GLOBS:
        files.extend(glob.glob(str(RUNS / g)))
    files = sorted(set(files))

    exp_cache: dict[str, str] = {}
    gold_cache: dict[str, set[str]] = {}
    per_traj: list[dict] = []
    for f in files:
        results = json.load(open(f))
        sims_raw = {str(s.get("id")): s for s in results.get("simulations") or []}
        for sw in score_results(results, "retail", wt_data):
            tid = sw.task_id
            if tid not in exp_cache:
                task = json.load(open(TASKS / f"{tid}.json"))[0]
                exp_cache[tid] = expected_hash(task, "perturbed", "retail")
                gold_cache[tid] = {g.name for g in
                                   extract_gold_write_actions(task, write_tools)}
            if not sw.has_wrong_bind:
                continue
            raw = sims_raw.get(str(sw.sim_id))
            _require(raw is not None, f"sim {sw.sim_id} of {tid} not in {f}")
            final = raw.get("_final_db_hash")
            _require(final is not None,
                     f"sim {sw.sim_id} of {tid} lacks _final_db_hash (must not silently count as fail_wrong)")
            if final == exp_cache[tid]:
                continue                            # end state is the gold state: recovered
            old, new = _old_new(raw)
            seq = _write_seq(raw, gold_cache[tid], old, new)
            per_traj.append({
                "task": tid,
                "seed": raw.get("seed"),
                "sim_id": str(sw.sim_id),
                "label": _classify(seq),
                "writes": [w["kind"] for w in seq],
                "n_write": len(seq),
                "any_errored": any(w["errored"] for w in seq),
            })

    per_traj.sort(key=lambda r: (r["task"], r["seed"]))
    counts = {lab: sum(t["label"] == lab for t in per_traj) for lab in LABELS}
    return {
        "counts": counts,
        "per_traj": per_traj,
        "n_fail_wrong": len(per_traj),
        "n_pool_files": len(files),
    }


def _self_check(res: dict) -> None:
    """Check 12 fail_wrong trajectories split 7/4/1, no write with an error, and per-task
    fail_wrong counts equal to recover.per_task in NUMBERS.json; raises on any mismatch."""
    c = res["counts"]
    _require(res["n_fail_wrong"] == 12, f"n_fail_wrong should be 12, got {res['n_fail_wrong']}")
    _require((c["poisoned_recovery"], c["incomplete_recovery"], c["single_no_recovery"])
             == (7, 4, 1), f"split should be 7/4/1, got {c}")
    # the labels look only at write targets, which is valid only if no write returned an error
    errored = [t for t in res["per_traj"] if t["any_errored"]]
    _require(not errored, f"errored write inside the 12 - recheck the target-based criterion: {errored}")

    _require(NUMBERS_JSON.exists(),
             f"{NUMBERS_JSON} missing; run collect_numbers.py first")
    rec = json.load(open(NUMBERS_JSON)).get("recover", {}).get("per_task", {})
    by_task = Counter(t["task"] for t in res["per_traj"])
    fw_tasks = {tid for tid, v in rec.items() if v.get("fail_wrong", 0) > 0}
    _require(set(by_task) == fw_tasks,
             f"fail_wrong task sets diverge: this script {set(by_task)} vs NUMBERS {fw_tasks}")
    for tid, v in rec.items():
        _require(by_task.get(tid, 0) == v.get("fail_wrong", 0),
                 f"{tid}: this script fail_wrong={by_task.get(tid, 0)} "
                 f"≠ NUMBERS {v.get('fail_wrong')}")


def main() -> None:
    """Label the trajectories, run the self-check, and print the table and the totals."""
    res = classify_recover()
    _self_check(res)

    print(f"# Unrecovered recovery trajectories, per-trajectory split ({res['n_fail_wrong']} fail_wrong / "
          f"{res['n_pool_files']} pool files)\n")
    print("| task | seed | writes | label |")
    print("|---|---|---|---|")
    for t in res["per_traj"]:
        print(f"| {t['task'].replace('retail_recover_', '')} | {t['seed']} | "
              f"{'→'.join(t['writes'])} | {t['label']} |")
    c = res["counts"]
    print(f"\n**total**: poisoned={c['poisoned_recovery']} / "
          f"incomplete={c['incomplete_recovery']} / "
          f"single={c['single_no_recovery']}  (= {res['n_fail_wrong']})")
    # prints only; the stored copy of these labels is recover.forensic in NUMBERS.json


if __name__ == "__main__":
    main()
