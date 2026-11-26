#!/usr/bin/env python3
"""Generate the control version of each of the 20 main-suite ID-REUSE tasks.

A control task is a copy of its base task with _perturb.fire_at_start set, so
the same change is applied before the conversation starts: the target record
already has its new identifier when the agent first reads it, and the old
identifier already names the distractor. Run control tasks with --arm perturbed;
run_perturbed.py builds the perturbation and remaps the gold write only in that
condition.

The script reads wear/deictic_tasks/*_idreuse_*.json, skipping the variant and
control files, and writes <base>_control.json next to each base task. Existing
control files are always overwritten; each depends only on its base task file.

Usage: python3 wear/gen_controls.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deictic_tasks")

# The 20 main-suite tasks: every *_idreuse_* file except variants and controls.
BASES = sorted(
    p for p in glob.glob(os.path.join(OUT, "*_idreuse_*.json"))
    if not any(s in p for s in ("_control", "_signal", "_distB", "_distC", "_simnew"))
)


def main() -> None:
    """Write one control task per base task."""
    made = 0
    for path in BASES:
        tasks = json.load(open(path))
        assert len(tasks) == 1, path
        t = tasks[0]
        assert t.get("_perturb"), f"{path}: no _perturb"
        assert t["_perturb"].get("knob") == "id_reuse", path
        cpath = path.replace(".json", "_control.json")
        c = json.loads(json.dumps(t))  # deep copy
        c["id"] = t["id"] + "_control"
        c["_perturb"]["fire_at_start"] = True
        c["_doc"] = (
            f"CONTROL (no-stale-cache) twin of {t['id']}: same perturbation & "
            f"final world, fired BEFORE rollout start -> referent already at "
            f"{list(t['_gold_remap'].values())[0]} when agent first reads; "
            f"expected WEAR=0."
        )
        with open(cpath, "w") as f:
            json.dump([c], f, indent=2, ensure_ascii=False)
            f.write("\n")
        made += 1
        print("wrote", os.path.relpath(cpath, HERE))
    print(f"controls: {made} written (unconditional regen), {len(BASES)} base tasks")


if __name__ == "__main__":
    main()
