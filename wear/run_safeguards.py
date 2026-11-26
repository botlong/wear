#!/usr/bin/env python3
"""Driver for the safeguard study.

The study compares four safeguards, called arms in the code: none, policy (the
verification policy in mitigation_prompt.txt, appended to the agent's domain
policy), guard_cas (the CAS write guard) and guard_identity (the identity write
guard). It runs two task suites: main (the 20 ID-REUSE tasks of wear/deictic_tasks/)
and heldout (the 12 tasks of wear/deictic_tasks_heldout/).

The script builds every unit (arm x suite x task x model x condition) and runs them
through run_batch.run_unit in matched blocks. A block holds the arms of one
(suite, task, model, condition); they are started together, and the order of the
blocks is shuffled with a fixed seed. The arms are therefore collected over the
same period, so drift in the serving cannot pass for a safeguard effect.

Writes <outroot>/<arm>/<suite>/<model slug>/<condition>/<task>.json (with a .log),
and appends failed units to <outroot>/failures.txt. Every model must be given a
--route, which names the environment-variable prefix of the relay's base URL and key.
A short trial plan (k=1, perturbed condition, one task per suite, every arm) can be run into
a separate output root first; see --help.

Usage: python3 wear/run_safeguards.py --models M... --route M=PREFIX... [--model-kwargs M=JSON...]
       [--outroot DIR] [--k 5] [--concurrency 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_batch as rb  # noqa: E402

# Safeguard (arm) -> the run_batch unit options that implement it.
ARMS = {
    "none": {},
    "policy": {"agent_suffix_file": os.path.join(HERE, "mitigation_prompt.txt")},
    "guard_cas": {"write_guard": "cas"},
    "guard_identity": {"write_guard": "identity"},
}
# The guards cannot act unless a record changes, so they run only in the perturbed
# condition; their false rejections are measured offline, by replaying baseline
# rollouts through each guard (analyze_safeguards.replay_false_rejections).
CONDITIONS = {"none": ("baseline", "perturbed"), "policy": ("baseline", "perturbed"),
              "guard_cas": ("perturbed",), "guard_identity": ("perturbed",)}
SMOKE_TASKS = ("airline_idreuse_boston", "retail_heldout_backpack_harper")
BLOCK_SEED = 2026  # seed of the block shuffle
SIM_TIMEOUT_S = 900  # wall-clock budget per rollout, seconds


def suites() -> dict[str, list[tuple[str, str]]]:
    """Task files of the main and held-out suites, as (domain, path) pairs."""
    main = [(d, p) for d in ("airline", "retail") for p in rb.suite_tasks(d)]
    held = [(os.path.basename(p).split("_", 1)[0], p)
            for p in sorted(glob.glob(os.path.join(HERE, "deictic_tasks_heldout", "*.json")))]
    return {"main": main, "heldout": held}


def plan(models: list[str], outroot: str, k: int, smoke: bool = False) -> list[dict]:
    """Return every unit of the study, block by block.

    Blocks are (suite, task, model, condition); their order is shuffled with
    BLOCK_SEED, and so is the order of the arms inside each block. The last argument
    restricts the plan to the trial run.
    """
    blocks: dict[tuple, list[dict]] = {}
    for suite, tasks in suites().items():
        for domain, tp in tasks:
            base = os.path.basename(tp)[:-5]
            if smoke and base not in SMOKE_TASKS:
                continue
            for model in models:
                for arm, flags in ARMS.items():
                    for cond in CONDITIONS[arm]:
                        if smoke and cond != "perturbed":
                            continue
                        u = rb.unit_spec(domain, tp, cond, model,
                                         os.path.join(outroot, arm, suite), k,
                                         flags.get("agent_suffix_file"),
                                         flags.get("write_guard"), SIM_TIMEOUT_S)
                        u["name"] = f"{arm}/{suite}/{u['name']}"
                        blocks.setdefault((suite, base, model, cond), []).append(u)
    order = sorted(blocks)
    rng = random.Random(BLOCK_SEED)
    rng.shuffle(order)
    out = []
    for key in order:
        arms = blocks[key][:]
        rng.shuffle(arms)  # no arm is always first within a block
        for u in arms:
            u["block"] = "/".join(key)
        out += arms
    return out


def run_block(units: list[dict], timeout_s: int) -> list[tuple[str, bool, str]]:
    """Start every arm of a block at the same time; return their (name, success, error) results."""
    with ThreadPoolExecutor(max_workers=len(units)) as ex:
        return list(ex.map(lambda u: rb.run_unit(u, timeout_s), units))


def main() -> None:
    """Plan the units, check existing outputs against the current configuration, and run the pending blocks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--route", nargs="+", default=[], metavar="MODEL=PREFIX")
    ap.add_argument("--model-kwargs", nargs="+", default=[], metavar="MODEL=JSON")
    ap.add_argument("--outroot", default=os.path.join(HERE, "runs", "safeguard_study"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="short trial run: k=1, perturbed, the two SMOKE_TASKS, every safeguard")
    ap.add_argument("--smoke-attempt", type=int, default=1,
                    help="each trial run gets its own outroot")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="concurrent units; blocks run whole, so this is rounded to blocks")
    ap.add_argument("--unit-timeout", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for r in args.route:
        m, _, prefix = r.partition("=")
        assert m and prefix, f"bad --route {r!r}"
        rb.ROUTES[m] = prefix
    for r in args.model_kwargs:
        m, _, js = r.partition("=")
        rb.MODEL_KWARGS[m] = json.loads(js)
    missing = [m for m in args.models if m not in rb.ROUTES]
    assert not missing, f"every model needs --route: {missing}"
    rb.GIT_HEAD = rb._git_head()
    k = 1 if args.smoke else args.k
    outroot = args.outroot + (f"_smoke{args.smoke_attempt}" if args.smoke else "")

    units = plan(args.models, outroot, k, args.smoke)
    # Every existing unit must match the current configuration, completed or not,
    # so settings can never be mixed within an output tree.
    bad = [e for e in (rb.check_config(u) for u in units) if e]
    if bad:
        raise SystemExit("config check failed:\n  " + "\n  ".join(bad[:10]))
    pending = [u for u in units if not rb.done(u)]
    print(f"[plan] units={len(units)} pending={len(pending)} k={k} "
          f"~{sum(u['k'] - rb.n_sims(u['out']) for u in pending)} rollouts -> {outroot}")
    if args.dry_run:
        for u in pending[:12]:
            print("  ", u["name"])
        return
    failures: list[str] = []
    ok = 0
    blocks: dict[str, list[dict]] = {}
    for u in pending:
        blocks.setdefault(u["block"], []).append(u)
    n_par = max(1, args.concurrency // len(ARMS))
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futs = [ex.submit(run_block, us, args.unit_timeout) for us in blocks.values()]
        for f in as_completed(futs):
            for name, success, err in f.result():
                if success:
                    ok += 1
                    print(f"[done {ok}/{len(pending)}] {name}", flush=True)
                else:
                    failures.append(f"{name}: {err}")
                    print(f"[FAIL] {name}: {err}", file=sys.stderr, flush=True)
    if failures:
        os.makedirs(outroot, exist_ok=True)
        with open(os.path.join(outroot, "failures.txt"), "a") as fh:
            fh.write("\n".join(failures) + "\n")
        print(f"[summary] ok={ok} FAILED={len(failures)}", file=sys.stderr)
        sys.exit(1)
    print(f"[summary] ok={ok} failed=0")


if __name__ == "__main__":
    main()
