#!/usr/bin/env python3
"""Analysis of the safeguard study.

Reads the unit files that run_safeguards.py writes, ROOT/<arm>/<suite>/<model>/<condition>/
<task>.json, together with the source task files, and prints a JSON report (also
written to --out): per-setting measures and counters ("cells"), paired tests
("tests"), and the false-rejection replay of the write guards.

Writes are scored with the eligible-denominator WEAR scorer
(scorer_extended.score_binding). In the perturbed condition, a write made before
the ID-REUSE change is scored against the original gold target and a write made at
or after it against the remapped one. The change point is the call named by the
`fired` event that perturb.py logs.
  attempted WEAR = wrong-target eligible writes / eligible writes
  effective WEAR = wrong-target eligible writes with a non-error response / eligible writes
  stale-ID write = a write to the reused identifier at or after the change with a
      non-error response; one with an unmatched response is counted as unresolved
  passed-after-re-read = a stale-ID write preceded, at or after the change, by a
      non-error read of the reused identifier
  other wrong-target writes = effective wrong-target writes that are not stale-ID writes
  completion = a non-error write to the applicable gold target (scorer status "correct")
  exact completion (secondary) = such a write that also equals the gold write in every
      gold argument after the scorer's identifier normalisation
  end-state success = the final DB hash equals the expected hash that expected_hash.py
      computes for the task and condition
Cost per rollout is also reported: agent tool calls, the non-write calls among
them, and recorded run time.

A perturbed rollout in which the change was never applied, or whose change point
cannot be located, is *unexposed*: it is counted, and excluded from every perturbed
measure and counter. A rollout with no eligible write has undefined WEAR.
Aggregation: rollout -> task mean -> mean over the tasks
with a defined value. Paired tests use the tasks defined in both settings, pooled
over domains within a suite: an exact two-sided sign-flip test by full enumeration,
run only with at least 8 pairs.

Usage (tau2 env): python wear/analyze_safeguards.py [--root wear/runs/safeguard_study] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_perturbed import schedule_from_task  # noqa: E402
from scorer import extract_gold_write_actions  # noqa: E402
from scorer_extended import normalize_id, score_binding  # noqa: E402
from stats import paired_sign_flip_exact  # noqa: E402

WT = json.load(open(os.path.join(HERE, "write_tools.json")))
READ_TOOL = {"airline": ("get_reservation_details", "reservation_id"),
             "retail": ("get_order_details", "order_id")}
ARMS = ("none", "policy", "guard_cas", "guard_identity")
SUITE_DIRS = {"main": "deictic_tasks", "heldout": "deictic_tasks_heldout"}
MIN_PAIRS = 8
PERTURBED_MEASURES = ("attempted", "effective", "completion", "completion_exact",
                      "end_state_success")
# Descriptive cost per rollout: agent tool calls, the non-write calls among them,
# and recorded run time.
COST_MEASURES = ("tool_calls", "read_calls", "duration_s")
# Per-rollout counts, summed over the exposed rollouts of a setting.
COUNTERS = ("stale_writes", "stale_unresolved", "passed_after_reread", "other_wrong",
            "rejections", "rejections_identity", "writes_unresolved")


def source_task(path: str) -> dict:
    """Load the source task, from either suite directory, that a unit file was run on."""
    base = os.path.basename(path)
    for d in SUITE_DIRS.values():
        p = os.path.join(HERE, d, base)
        if os.path.exists(p):
            t = json.load(open(p))
            return t[0] if isinstance(t, list) else t
    raise FileNotFoundError(base)


def tool_calls(sim: dict) -> list[dict]:
    """Return the agent's tool calls in execution order, each with its response status.

    The status is 'error' or 'nonerror' when the call id is unique and exactly one
    later tool message, with an error field, carries it; otherwise 'unknown'.
    """
    msgs = sim.get("messages") or []
    resp: dict = defaultdict(list)
    ids: Counter = Counter()
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and m.get("id") is not None:
            resp[m["id"]].append((i, m))
    calls = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c.get("requestor", "assistant") == "assistant":
                ids[c.get("id")] += 1
                calls.append((i, c))
    out = []
    for i, c in calls:
        cand = resp.get(c.get("id"), [])
        r = "unknown"
        if c.get("id") is not None and ids[c.get("id")] == 1 and len(cand) == 1 \
                and cand[0][0] > i and "error" in cand[0][1]:
            r = "error" if cand[0][1]["error"] else "nonerror"
        out.append({"name": c.get("name"), "arguments": dict(c.get("arguments") or {}),
                    "id": c.get("id"), "response": r})
    return out


def fire_index(sim: dict, calls: list[dict]) -> int | str | None:
    """Return the position in `calls` of the call before which the change was applied.

    The call is located by the call id that perturb.py logged with the `fired` event.
    Returns "unresolved" if that id is missing or not unique, and None if the
    rollout has no `fired` event.
    """
    for e in sim.get("_perturb_events") or []:
        if e.get("kind") == "fired":
            pos = [i for i, c in enumerate(calls) if e.get("call_id") is not None
                   and c["id"] == e.get("call_id")]
            return pos[0] if len(pos) == 1 else "unresolved"
    return None


def fire_index_reconstructed(calls: list[dict], src: dict) -> int | None:
    """Reconstruct the change point for run files without a `fired` event.

    Most runs of the main experiment and of the verification-policy runs carry no
    such event. The change is applied before the call that follows the n-th trigger
    read that returned without error.
    """
    s = schedule_from_task(src)
    if s is None:
        return None
    hits, armed = 0, False
    targ, val = s.spec.get("trigger_arg"), s.spec.get("trigger_arg_value")
    for i, c in enumerate(calls):
        if armed:
            return i
        if c["name"] == s.target_tool and c["response"] != "error" \
                and (targ is None or c["arguments"].get(targ) == val):
            hits += 1
        if hits >= s.fire_after_n_calls:
            armed = True
    return None


_EXPECTED: dict = {}


def expected_end_hash(src: dict, cond: str, domain: str) -> str | None:
    """Expected final DB hash for a task and condition (expected_hash.py), cached; None on failure."""
    key = (src.get("id"), cond)
    if key not in _EXPECTED:
        try:
            from expected_hash import expected_hash
            _EXPECTED[key] = expected_hash(src, cond, domain)
        except Exception as e:  # reported, never silently treated as success
            print(f"[analyze] expected hash failed for {key}: {e}", file=sys.stderr)
            _EXPECTED[key] = None
    return _EXPECTED[key]


def _exact(args: dict, gold) -> bool:
    """True if every gold argument is present in args with the same normalised value."""
    return all(k in args and normalize_id(args[k]) == normalize_id(v)
               for k, v in gold.arguments.items())


def rollout_measures(sim: dict, stored_task: dict, src: dict, domain: str, cond: str,
                     legacy_fire: bool = False) -> dict:
    """Measures and counters of one rollout (see the module docstring).

    `stored_task` is the task as the runner saved it (gold remapped in the perturbed
    condition) and `src` the source task with the original gold. The last flag
    reconstructs the change point (fire_index_reconstructed) for run files without
    a `fired` event.
    """
    wts = set(WT[domain])
    keys = set(WT["_object_id_keys"][domain])
    calls = tool_calls(sim)
    write_pos = [i for i, c in enumerate(calls) if c["name"] in wts]
    out = {"seed": sim.get("seed"), "termination": sim.get("termination_reason"),
           "exposed": True, **{k: 0 for k in COUNTERS},
           "tool_calls": len(calls), "read_calls": len(calls) - len(write_pos),
           "duration_s": sim.get("duration")}
    fi = None
    if cond == "perturbed":
        fi = fire_index_reconstructed(calls, src) if legacy_fire else fire_index(sim, calls)
        out["boundary_unresolved"] = fi == "unresolved"
        if fi == "unresolved":
            fi = None
        out["exposed"] = bool(sim.get("_perturb_fired")) and fi is not None
        if not out["exposed"]:
            out.update({m: None for m in PERTURBED_MEASURES + COST_MEASURES})
            return out
    after = score_binding(sim, stored_task, wts, keys).to_dict()["writes"]
    before = score_binding(sim, src, wts, keys).to_dict()["writes"] if fi is not None else after
    golds_after = extract_gold_write_actions(stored_task, wts)
    golds_before = extract_gold_write_actions(src, wts)
    elig = wrong = wrong_eff = 0
    completion = exact = False
    old = src["_perturb"]["spec"]["referent_id"] if fi is not None else None
    read_tool, idf = READ_TOOL[domain]
    out["other_wrong"] = 0
    for j, pos in enumerate(write_pos):
        post = fi is not None and pos >= fi
        w = (after if (fi is None or post) else before)[j]
        gold = (golds_after if (fi is None or post) else golds_before)[0]
        c = calls[pos]
        out["writes_unresolved"] += int(c["response"] == "unknown")
        if w["status"] in ("correct", "wrong_target"):
            elig += 1
        if w["status"] == "wrong_target":
            wrong += 1
            if c["response"] == "nonerror":
                wrong_eff += 1
                stale = post and normalize_id(c["arguments"].get(idf, "")) == normalize_id(old)
                out["other_wrong"] += int(not stale)
        if w["status"] == "correct" and c["response"] == "nonerror":
            completion = True
            exact |= c["name"] == gold.name and _exact(c["arguments"], gold)
    out["attempted"] = wrong / elig if elig else None
    out["effective"] = wrong_eff / elig if elig else None
    out["n_eligible"] = elig
    out["completion"], out["completion_exact"] = completion, exact
    if fi is not None:
        reread = False
        for i, c in enumerate(calls):
            if i < fi:
                continue
            target = normalize_id(c["arguments"].get(idf, ""))
            if c["name"] == read_tool and target == normalize_id(old) and c["response"] == "nonerror":
                reread = True
            if c["name"] in wts and target == normalize_id(old):
                if c["response"] == "nonerror":
                    out["stale_writes"] += 1
                    out["passed_after_reread"] += int(reread)
                elif c["response"] == "unknown":
                    out["stale_unresolved"] += 1
    kinds = [e.get("kind") for e in sim.get("_write_guard_events") or []]
    out["rejections"] = sum(k.startswith("write_rejected") for k in kinds)
    out["rejections_identity"] = kinds.count("write_rejected_identity")
    exp = expected_end_hash(src, cond, domain)
    out["end_state_success"] = None if exp is None or sim.get("_final_db_hash") is None \
        else sim["_final_db_hash"] == exp
    return out


def _unit_files(root: str) -> list[str]:
    """Unit files root/<model>/<condition>/<task>.json, without quarantine files."""
    return [p for p in sorted(glob.glob(os.path.join(root, "*", "*", "*.json")))
            if not p.endswith(".quarantine.json")]


def load_cells(root_map: dict[str, dict[str, str]], legacy_fire: bool = False):
    """Score every rollout under root_map[arm][suite].

    Returns ({(arm, suite, model, cond): {task: [rollout measures]}}, per-setting
    counts of units, rollouts, unexposed rollouts and termination reasons).
    """
    cells: dict = defaultdict(lambda: defaultdict(list))
    meta: dict = defaultdict(Counter)
    for arm, suites in root_map.items():
        for suite, root in suites.items():
            for path in _unit_files(root):
                model, cond = path.split(os.sep)[-3], path.split(os.sep)[-2]
                if cond not in ("baseline", "perturbed"):
                    continue
                domain = os.path.basename(path).split("_", 1)[0]
                src = source_task(path)
                d = json.load(open(path))
                tmap = {t["id"]: t for t in d.get("tasks") or []}
                key = (arm, suite, model, cond)
                meta[key]["units"] += 1
                for sim in d.get("simulations") or []:
                    meta[key]["rollouts"] += 1
                    m = rollout_measures(sim, tmap[sim["task_id"]], src, domain, cond, legacy_fire)
                    meta[key]["unexposed"] += int(not m["exposed"])
                    meta[key]["boundary_unresolved"] += int(bool(m.get("boundary_unresolved")))
                    meta[key]["termination_" + str(m["termination"])] += 1
                    cells[key][os.path.basename(path)[:-5]].append(m)
    return cells, meta


def task_means(rolls_by_task: dict, measure: str) -> dict[str, float]:
    """Mean of a measure over each task's rollouts, for the tasks where it is defined."""
    out = {}
    for task, rolls in rolls_by_task.items():
        vals = [float(r[measure]) for r in rolls if r.get(measure) is not None]
        if vals:
            out[task] = sum(vals) / len(vals)
    return out


def cell_mean(tm: dict[str, float]) -> float | None:
    """Mean over tasks, or None if no task has a value."""
    return sum(tm.values()) / len(tm) if tm else None


def paired(cells, a: tuple, b: tuple, measure: str) -> dict:
    """Compare settings a and b on a measure, over the tasks defined in both.

    Returns the two means and, with at least MIN_PAIRS pairs, the mean difference
    b - a and the exact two-sided sign-flip p-value.
    """
    ta, tb = task_means(cells.get(a, {}), measure), task_means(cells.get(b, {}), measure)
    common = sorted(set(ta) & set(tb))
    res = {"n_pairs": len(common), "mean_a": cell_mean({t: ta[t] for t in common}),
           "mean_b": cell_mean({t: tb[t] for t in common})}
    if len(common) >= MIN_PAIRS:
        diff, p = paired_sign_flip_exact({t: (ta[t], tb[t]) for t in common})
        res.update(mean_diff_b_minus_a=diff, p_two_sided=float(p), p_exact=str(Fraction(p)))
    else:
        res["note"] = f"fewer than {MIN_PAIRS} pairs: descriptive only"
    return res


def replay_false_rejections(root_map: dict[str, dict[str, str]]) -> dict:
    """Measure false rejections of the write guards.

    Replays the agent tool calls of every baseline rollout without a safeguard
    through a fresh environment wrapped with each guard (no perturbation), and
    counts the rollouts in which a write is rejected.
    """
    from tau2.data_model.message import ToolCall
    from tau2.runner.build import build_environment
    from write_guard import wrap_write_guard
    out: dict = defaultdict(Counter)
    for suite, root in (root_map.get("none") or {}).items():
        for path in _unit_files(root):
            if path.split(os.sep)[-2] != "baseline":
                continue
            model = path.split(os.sep)[-3]
            domain = os.path.basename(path).split("_", 1)[0]
            for sim in json.load(open(path)).get("simulations") or []:
                for mode in ("cas", "identity"):
                    env = wrap_write_guard(build_environment(domain), set(WT[domain]), mode)
                    rejected = False
                    for i, c in enumerate(tool_calls(sim)):
                        r = env.get_response(ToolCall(id=str(i), name=c["name"],
                                                      arguments=c["arguments"], requestor="assistant"))
                        if r.error and "Precondition failed" in (r.content or ""):
                            rejected = True
                            break
                    k = f"{suite}/{model}/{mode}"
                    out[k]["rollouts"] += 1
                    out[k]["rejected"] += int(rejected)
    return {k: dict(v) for k, v in sorted(out.items())}


def analyze(root_map: dict[str, dict[str, str]], replay: bool = True,
            legacy_fire: bool = False) -> dict:
    """Build the full report: cells, the false-rejection replay, and the paired tests.

    Per suite and model the tests are: policy vs none on attempted WEAR, identity vs
    CAS guard on effective WEAR, and each guard vs policy on completion, all in the
    perturbed condition.
    """
    cells, meta = load_cells(root_map, legacy_fire)
    out: dict = {"cells": {}, "tests": {}}
    for key in sorted(set(meta) | set(cells)):
        by_task = cells.get(key, {})
        rolls = [r for rs in by_task.values() for r in rs]
        out["cells"]["/".join(key)] = {
            **{k: 0 for k in ("units", "rollouts", "unexposed", "boundary_unresolved")},
            **dict(meta.get(key, {})), "n_tasks": len(by_task),
            **{f"{m}_mean": cell_mean(task_means(by_task, m)) for m in PERTURBED_MEASURES},
            **{f"n_tasks_defined_{m}": len(task_means(by_task, m)) for m in PERTURBED_MEASURES},
            **{f"{m}_mean": cell_mean(task_means(by_task, m)) for m in COST_MEASURES},
            **{c: sum(r[c] for r in rolls if r["exposed"]) for c in COUNTERS},
            "no_eligible_write": sum(r.get("n_eligible") == 0 for r in rolls if r["exposed"]),
        }
    if replay:
        out["h3c_false_rejection_replay"] = replay_false_rejections(root_map)
    models = sorted({k[2] for k in meta})
    suites = sorted({k[1] for k in meta})
    for s in suites:
        for m in models:
            k = lambda arm, cond: (arm, s, m, cond)  # noqa: E731
            t = out["tests"].setdefault(f"{s}/{m}", {})
            t["policy_vs_none_attempted"] = paired(cells, k("none", "perturbed"), k("policy", "perturbed"), "attempted")
            t["identity_vs_cas_effective"] = paired(cells, k("guard_cas", "perturbed"), k("guard_identity", "perturbed"), "effective")
            for g in ("guard_cas", "guard_identity"):
                t[f"{g}_vs_policy_completion"] = paired(cells, k("policy", "perturbed"), k(g, "perturbed"), "completion")
    return out


def main() -> None:
    """Analyze the arms and suites found under --root and print the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "runs", "safeguard_study"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root_map = {arm: {s: os.path.join(args.root, arm, s) for s in SUITE_DIRS
                      if os.path.isdir(os.path.join(args.root, arm, s))} for arm in ARMS}
    res = analyze(root_map)
    txt = json.dumps(res, indent=1, default=str)
    if args.out:
        open(args.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
