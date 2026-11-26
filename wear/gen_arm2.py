#!/usr/bin/env python3
"""Generate two variants of airline main-suite ID-REUSE tasks.

Single-character variant (airline_idreuse_<city>_simnew.json, 5 tasks): the
target's new identifier differs from the old one only in its last character
(for example NO6JO3 -> NO6JO0) instead of being an unrelated code. It is chosen
so that it is neither a reservation id in the airline database nor a new
identifier in any task file. Everything else is as in the base task.

No-distractor variant (airline_idreuse_<city>_distC_none.json, 4 tasks): no
record takes over the old identifier, so after the change a write through the
old identifier fails with an error. The fifth task of this variant,
airline_idreuse_dallas_distC_none.json, already exists and is not written here.

The script reads the base tasks wear/deictic_tasks/airline_idreuse_<city>.json
and the tau2-bench airline database, and writes into wear/deictic_tasks/. It is
deterministic and safe to rerun: a missing output file is written, an identical
one is skipped, and one that differs stops the script without being overwritten.

Usage: python3 wear/gen_arm2.py
"""
from __future__ import annotations

import json
import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "deictic_tasks")
DB = os.path.join(os.path.dirname(HERE), "tau2-bench", "data", "tau2",
                  "domains", "airline", "db.json")

M1_BASES = ["dallas", "boston", "denver", "houston", "miami"]  # single-character variant
M2_BASES = ["boston", "denver", "houston", "miami"]  # no-distractor variant; dallas exists already


def db_reservation_ids() -> set[str]:
    """Return the reservation ids of the tau2-bench airline database."""
    db = json.load(open(DB))
    res = db["reservations"]
    ids = set(res.keys()) if isinstance(res, dict) else {r["reservation_id"] for r in res}
    assert ids, "empty reservation ids?"
    return ids


def suite_new_ids() -> set[str]:
    """Return the new identifiers (spec.referent_new_id) used by any task file in wear/deictic_tasks/."""
    out = set()
    for fn in os.listdir(TASKS):
        if not fn.endswith(".json"):
            continue
        for t in json.load(open(os.path.join(TASKS, fn))):
            spec = (t.get("_perturb") or {}).get("spec") or {}
            if spec.get("referent_new_id"):
                out.add(spec["referent_new_id"])
    return out


def simnew_id(old: str, taken: set[str]) -> str:
    """Replace the last character of old with the first of 0-9, A-Z that differs from it
    and gives an id not in taken."""
    assert len(old) == 6, old
    for ch in string.digits + string.ascii_uppercase:
        if ch == old[-1]:
            continue
        cand = old[:-1] + ch
        if cand not in taken:
            return cand
    raise RuntimeError(f"no simnew candidate for {old}")


def load_single(path: str) -> dict:
    """Load a task file that holds exactly one task."""
    arr = json.load(open(path))
    assert isinstance(arr, list) and len(arr) == 1, path
    return arr[0]


def dump(path: str, task: dict) -> None:
    """Write [task] to path if the file is missing; skip it if identical; exit if it differs."""
    payload = [task]
    if os.path.exists(path):
        if json.load(open(path)) != payload:
            sys.exit(f"REFUSE overwrite (drifted): {path}")
        print(f"[skip] identical: {os.path.basename(path)}")
        return
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[write] {os.path.basename(path)}")


def main() -> None:
    """Write (or check) the 5 single-character and 4 no-distractor tasks."""
    taken = db_reservation_ids() | suite_new_ids()
    for base in M1_BASES:
        src = os.path.join(TASKS, f"airline_idreuse_{base}.json")
        t = load_single(src)
        spec = t["_perturb"]["spec"]
        old, prev_new = spec["referent_id"], spec["referent_new_id"]
        outp = os.path.join(TASKS, f"airline_idreuse_{base}_simnew.json")
        # Reuse the new identifier of an existing file, so that a rerun reproduces it.
        if os.path.exists(outp):
            new = load_single(outp)["_perturb"]["spec"]["referent_new_id"]
            assert new != old and len(new) == 6, (base, new)
        else:
            new = simnew_id(old, taken | {old})
        taken.add(new)
        t["id"] = f"airline_idreuse_{base}_simnew"
        t["_doc"] = (f"M1 similar-new-id mechanism variant of airline_idreuse_{base}: "
                     f"referent_new_id {prev_new}->{new} (edit distance 1 from OLD {old}; "
                     f"collision-checked against DB and suite). Primary metric notice-rate, "
                     f"wrong-rate secondary. Gold stores the baseline id; the runner maps it "
                     f"via _gold_remap.")
        spec["referent_new_id"] = new
        assert list(t["_gold_remap"]) == [old], t["_gold_remap"]
        t["_gold_remap"] = {old: new}
        dump(os.path.join(TASKS, f"airline_idreuse_{base}_simnew.json"), t)

    for base in M2_BASES:
        src = os.path.join(TASKS, f"airline_idreuse_{base}.json")
        t = load_single(src)
        spec = t["_perturb"]["spec"]
        assert "distractor_id" in spec, src
        old = spec["referent_id"]
        t["id"] = f"airline_idreuse_{base}_distC_none"
        t["_doc"] = (f"M2 dangling-handle mechanism variant of airline_idreuse_{base}: "
                     f"same task without a distractor parked at OLD — after the referent "
                     f"moves {old}->{spec['referent_new_id']}, writes to the cached {old} "
                     f"fail loudly. Metrics: attempted/effective split and "
                     f"recovery-to-correct.")
        del spec["distractor_id"]
        dump(os.path.join(TASKS, f"airline_idreuse_{base}_distC_none.json"), t)

    print("[done] M1 ×5, M2 ×4")


if __name__ == "__main__":
    main()
