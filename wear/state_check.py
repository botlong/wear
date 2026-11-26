#!/usr/bin/env python3
"""The state check: an offline replay that tests whether each write changed the stored record.

Replays every listed trajectory's tool calls, in their recorded order, in a fresh
tau2 environment wrapped with the same perturbation schedule (no model calls).
For each write it compares the stored record of the target just before the write
(after any change has been applied) with the record just after it. A replay is
faithful only if every replayed tool response equals the recorded one and the
final DB hash equals the recorded `_final_db_hash`; only then is the per-write
evidence passed to the scorer (scorer_extended.score_binding).

Reads --sources, a JSON list of {source, sim_id, condition, domain} rows; source is
a run file relative to the repository root, and the optional field `extended`
holds scorer counts to compare against. Writes --out with the totals (trajectories,
faithful replays, disagreements with `extended`, and writes counted by condition,
status, response and whether the target changed), the SHA-256 of every source file
and of this script, and each trajectory's rescored writes with their replay
evidence. The totals are also printed.

Run (tau2 env):
  tau2-bench/.venv/bin/python wear/state_check.py --sources wear/audit/state_check_20261124/sources.json \
      --out wear/audit/state_check_20261124/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WEAR = Path(__file__).resolve().parent
ROOT = WEAR.parent
sys.path.insert(0, str(WEAR))
from perturb import PerturbSchedule, wrap_environment  # noqa: E402
from run_perturbed import apply_gold_remap  # noqa: E402
from scorer_extended import score_binding  # noqa: E402

from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.registry import registry  # noqa: E402

WT = json.load(open(WEAR / "write_tools.json"))
COLLECTION = {"reservation_id": "reservations", "order_id": "orders", "user_id": "users"}


def schedule_for(task_id: str, condition: str) -> PerturbSchedule | None:
    """Perturbation schedule of a rollout's task file in wear/deictic_tasks/; None for baseline."""
    if condition == "baseline":
        return None
    name = task_id if condition == "perturbed" else task_id  # a control task id names its control file
    spec = json.load(open(WEAR / "deictic_tasks" / f"{name}.json"))
    if isinstance(spec, list):
        spec = spec[0]
    p = spec.get("_perturb")
    if not p:
        raise ValueError(f"{name}: no _perturb")
    return PerturbSchedule(knob=p["knob"], target_tool=p["target_tool"],
                           fire_after_n_calls=p.get("fire_after_n_calls", 1),
                           spec=p.get("spec", {}), fire_at_start=p.get("fire_at_start", False))


def target_key(domain: str, args: dict):
    """(id field, identifier) of the record a write targets, or (None, None)."""
    for k in ("reservation_id", "order_id", "user_id"):
        if k in WT["_object_id_keys"][domain] and k in args and args[k] is not None:
            return k, str(args[k])
    return None, None


def payload(env, key, value):
    """Canonical JSON of the record stored under an identifier, or None if there is none."""
    coll = getattr(env.tools.db, COLLECTION[key])
    obj = coll.get(value)
    if obj is None:
        return None
    return json.dumps(obj.model_dump(mode="json"), sort_keys=True)


def replay(sim: dict, task: dict, domain: str, condition: str) -> dict:
    """Replay one trajectory and return per-write evidence, mismatched call ids and the final-hash check."""
    env = registry.get_env_constructor(domain)()
    init = task.get("initial_state")
    if init:
        raise NotImplementedError(f"{task['id']}: initial_state replay not supported")
    writes = set(WT[domain])
    evidence: dict[str, dict] = {}
    current: dict = {}

    # Inner hook: sits inside the perturbation wrapper, so an armed change has
    # already been applied when the "before" payload is read.
    inner = env.make_tool_call

    def recording(tool_name, requestor="assistant", **kwargs):
        rec = None
        if requestor == "assistant" and tool_name in writes:
            k, v = target_key(domain, kwargs)
            rec = {"tool": tool_name, "key": k, "id": v,
                   "before": payload(env, k, v) if k else None}
        resp = inner(tool_name, requestor=requestor, **kwargs)
        if rec is not None:
            rec["after"] = payload(env, rec["key"], rec["id"]) if rec["key"] else None
            current["rec"] = rec
        return resp

    env.make_tool_call = recording
    sched = schedule_for(sim["task_id"], condition)
    if sched is not None:
        env = wrap_environment(env, sched)

    recorded = {m["id"]: m for m in sim["messages"] if m.get("role") == "tool" and m.get("id")}
    mismatches = []
    for m in sim["messages"]:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            current.clear()
            call = ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments") or {},
                            requestor=c.get("requestor", "assistant"))
            resp = env.get_response(call)
            r = recorded.get(c["id"])
            if r is None or resp.content != r.get("content") or bool(resp.error) != bool(r.get("error")):
                mismatches.append(c["id"])
            if c["name"] in writes and c.get("requestor", "assistant") == "assistant":
                rec = current.get("rec")
                evidence[c["id"]] = {
                    "tool": c["name"], "target_key": rec["key"] if rec else None,
                    "target_id": rec["id"] if rec else None,
                    "reached_env": rec is not None, "error": bool(resp.error),
                    "target_existed": rec is not None and rec["before"] is not None,
                    "target_changed": (rec is not None and rec["before"] is not None
                                       and rec["before"] != rec["after"]),
                }
    final_ok = env.get_db_hash() == sim.get("_final_db_hash")
    return {"evidence": evidence, "response_mismatches": mismatches, "final_hash_match": final_ok}


def main() -> None:
    """Replay every listed trajectory, rescore its writes, and write the results file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="JSON list of {source, sim_id, condition, domain}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.sources))
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["source"]].append(r)

    out, totals = [], Counter()
    for src, rs in sorted(by_file.items()):
        data = json.load(open(ROOT / src))
        sims = {s["id"]: s for s in data["simulations"]}
        tasks = {t["id"]: t for t in (data["tasks"] if isinstance(data["tasks"], list) else [data["tasks"]])}
        for r in rs:
            sim, cond, dom = sims[r["sim_id"]], r["condition"], r["domain"]
            task = tasks[sim["task_id"]]
            rep = replay(sim, task, dom, cond)
            faithful = rep["final_hash_match"] and not rep["response_mismatches"]
            mutations = {cid: e["target_changed"] for cid, e in rep["evidence"].items()} if faithful else None
            spec = json.load(open(WEAR / "deictic_tasks" / f"{sim['task_id']}.json"))
            spec = spec[0] if isinstance(spec, list) else spec
            # a control rollout has the same change applied before the conversation,
            # so its gold is remapped too
            arm = "baseline" if cond == "baseline" else "perturbed"
            scored = score_binding(sim, apply_gold_remap(spec, arm, dom), set(WT[dom]),
                                   set(WT["_object_id_keys"][dom]), mutations=mutations).to_dict()
            ref = r.get("extended") or {}
            for k in ("n_writes", "n_wrong_target", "n_wrong_nonerror", "n_correct", "n_unmatched"):
                if k in ref and ref[k] != scored[k]:
                    totals["score_disagrees_with_audit"] += 1
                    break
            totals["trajectories"] += 1
            totals["faithful"] += faithful
            for w in scored["writes"]:
                ev = rep["evidence"].get(w["call_id"], {})
                key = (cond, w["status"], w["response"],
                       {True: "changed", False: "unchanged", None: "unverified"}[w["target_changed"]])
                totals["|".join(key)] += 1
                w["replay"] = ev
            out.append({"source": src, "sim_id": r["sim_id"], "task_id": sim["task_id"],
                        "model": r.get("model"), "domain": dom, "condition": cond,
                        "faithful": faithful, "final_hash_match": rep["final_hash_match"],
                        "response_mismatches": rep["response_mismatches"], "scored": scored})
            if totals["trajectories"] % 100 == 0:
                print(totals["trajectories"], "done", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    src_hashes = {s: hashlib.sha256(open(ROOT / s, "rb").read()).hexdigest() for s in by_file}
    json.dump({"totals": dict(sorted(totals.items())), "source_sha256": src_hashes,
               "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               "trajectories": out}, open(args.out, "w"), indent=1)
    for k, v in sorted(totals.items()):
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
