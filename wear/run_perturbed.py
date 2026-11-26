"""Runner: tau2-bench rollouts of one task file, with an optional perturbation and write guard.

Reads a task file (one task or a list). A task's `_perturb` field gives the
perturbation schedule (perturb.py) and `_gold_remap` maps old to new identifiers
for the gold writes. With --arm baseline the task runs unchanged; with --arm
perturbed the schedule is applied and the gold writes are remapped. The control
condition is the perturbed arm on a task's control file, whose schedule applies
the change before the conversation starts. --write-guard wraps the write tools
(write_guard.py); --agent-policy-suffix-file appends a text, such as the
verification policy, to the domain policy the agent receives.

Writes one JSON file {tasks, simulations, failures, started} that wear/scorer.py
and the analysis scripts read. It is rewritten after every rollout. Each
simulation also carries its perturbation events, write-guard events and final DB
hash. A rollout that raises is kept as a failure record with its partial messages.
When started by run_batch.py (WEAR_RUN_CONFIG set), the file also stores the
unit's configuration as `_config`.

Run under the tau2 venv:
  tau2-bench/.venv/bin/python wear/run_perturbed.py --domain airline --tasks TASKS.json --arm perturbed --model MODEL --k 3 --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import PerturbSchedule, wrap_environment  # noqa: E402
from write_guard import wrap_write_guard  # noqa: E402

import traceback  # noqa: E402

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # noqa: E402
from tau2.orchestrator.modes import CommunicationMode  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau2.runner.build import (  # noqa: E402
    build_agent,
    build_environment,
    build_user,
)


def load_tasks(path: str) -> list[dict]:
    """Load a task file holding one task, a list, or {"tasks": [...]}; return a list of task dicts."""
    raw = json.load(open(path))
    if isinstance(raw, dict) and "tasks" in raw:
        raw = raw["tasks"]
    if isinstance(raw, dict):
        raw = [raw]
    return raw


_ID_FIELDS = ("reservation_id", "order_id")


def _load_write_tools(domain: str) -> set[str]:
    """Names of the domain's write tools, from write_tools.json."""
    wt = json.load(open(Path(__file__).resolve().parent / "write_tools.json"))
    return set(wt.get(domain, []))


def apply_gold_remap(task_dict: dict, arm: str, domain: str) -> dict:
    """Return the task with its gold writes pointed at the new identifiers, for the perturbed arm.

    The reservation_id / order_id arguments of gold write actions are mapped through
    `_gold_remap`, so that the reference write targets the target record under its new
    identifier. Other arms, and tasks without a remap, are returned unchanged.
    """
    remap = task_dict.get("_gold_remap") or {}
    if arm != "perturbed" or not remap:
        return task_dict
    write_tools = _load_write_tools(domain)
    td = json.loads(json.dumps(task_dict))
    for a in (td.get("evaluation_criteria") or {}).get("actions") or []:
        if a.get("name") not in write_tools:
            continue
        args = a.get("arguments") or {}
        for f in _ID_FIELDS:
            if f in args and args[f] in remap:
                args[f] = remap[args[f]]
    return td


def schedule_from_task(task_dict: dict) -> PerturbSchedule | None:
    """Build the PerturbSchedule from a task's `_perturb` field, or None if it has none."""
    p = task_dict.get("_perturb")
    if not p:
        return None
    return PerturbSchedule(
        knob=p["knob"],
        target_tool=p["target_tool"],
        fire_after_n_calls=p.get("fire_after_n_calls", 1),
        spec=p.get("spec", {}),
        fire_at_start=p.get("fire_at_start", False),
    )


def run_one(
    domain: str,
    task: Task,
    model: str,
    schedule: PerturbSchedule | None,
    seed: int,
    max_steps: int,
    timeout: int,
    policy_suffix: str | None = None,
    write_guard: str | None = None,
) -> dict:
    """Run one rollout and return the tau2 simulation as a dict.

    The dict also holds the perturbation events, the write-guard events and the final DB
    hash. A rollout that raises returns a failure record instead (`_failure`, seed,
    task id and the messages generated so far).
    """
    env = build_environment(domain)
    guard = None
    if write_guard:
        # Write guard: wrapped first, so the perturbation wrapper sits outside it and an
        # armed change is applied before the guard checks the write.
        env = wrap_write_guard(env, _load_write_tools(domain), write_guard)
        guard = env._write_guard
    controller = None
    if schedule is not None:
        env = wrap_environment(env, schedule)
        controller = env._perturb

    if policy_suffix:
        # Policy suffix (e.g. the verification policy): appended to the domain policy
        # that the agent receives and the saved simulation records.
        _orig_policy = env.get_policy()
        env.get_policy = lambda: _orig_policy + "\n\n" + policy_suffix

    agent = build_agent("llm_agent", env, llm=model, task=task)
    user = build_user("user_simulator", env, task, llm=model)

    sim_id = str(uuid.uuid4())
    orch = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=max_steps,
        max_errors=5,
        seed=seed,
        solo_mode=False,
        simulation_id=sim_id,
        timeout=timeout,
    )
    try:
        sim = orch.run()
    except Exception as e:
        # Unfinished rollout (API, transport or other error): keep the messages
        # generated so far, so the failure can be audited; its seed is not used again.
        partial = []
        for m in getattr(orch, "trajectory", None) or []:
            try:
                partial.append(json.loads(m.model_dump_json()))
            except Exception:
                pass
        return {"_failure": f"{type(e).__name__}: {e}", "seed": seed, "task_id": task.id,
                "messages": partial,
                "_perturb_fired": controller._fired if controller is not None else None}
    try:
        sim.policy = env.get_policy()
    except Exception:
        pass
    try:
        sim.reward_info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=domain,
            mode=CommunicationMode.HALF_DUPLEX,
        )
    except Exception as e:
        print(f"[run] eval skipped (kept messages): {type(e).__name__}: {e}",
              file=sys.stderr)

    sim_d = json.loads(sim.model_dump_json())
    if controller is not None:
        sim_d["_perturb_events"] = controller.events
        sim_d["_perturb_fired"] = controller._fired
    if guard is not None:
        sim_d["_write_guard_events"] = guard.events
        sim_d["_write_guard_mode"] = guard.mode
    try:
        sim_d["_final_db_hash"] = env.get_db_hash()
    except Exception:
        sim_d["_final_db_hash"] = None
    return sim_d


def main():
    """Run k rollouts per task and write the output file after every rollout."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["retail", "airline"])
    ap.add_argument("--tasks", required=True, help="deictic task json (list or single)")
    ap.add_argument("--arm", required=True, choices=["baseline", "perturbed"])
    ap.add_argument("--model", required=True, help="e.g. openai/gemini-3.5-flash")
    ap.add_argument("--k", type=int, default=3, help="trials per task")
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-policy-suffix-file", default=None,
                    help="text file appended to the agent domain policy (verification policy runs)")
    ap.add_argument("--write-guard", nargs="?", const="cas", default=None,
                    choices=["cas", "identity"],
                    help="cas: reject writes to an entity that changed since the agent last "
                         "read it; identity: also reject when the id now names a different "
                         "entity")
    args = ap.parse_args()

    policy_suffix = None
    if args.agent_policy_suffix_file:
        policy_suffix = Path(args.agent_policy_suffix_file).read_text().strip()
        assert policy_suffix, "empty --agent-policy-suffix-file"

    task_dicts = load_tasks(args.tasks)
    tasks_out: list[dict] = []
    sims_out: list[dict] = []
    failures_out: list[dict] = []
    started: list[int] = []  # seeds reserved before each rollout starts
    config = json.loads(os.environ.get("WEAR_RUN_CONFIG") or "null")
    if config is not None:
        # Started by run_batch.py: the effective tau2 LLM settings of this process
        # (after tau2's load_dotenv) must be the ones the driver resolved and recorded.
        eff = {k: json.loads(v) for k, v in os.environ.items() if k.startswith("TAU2_LLM_")
               and k not in ("TAU2_LLM_NL_ASSERTIONS", "TAU2_LLM_ENV_INTERFACE")}
        for m in (eff.get("TAU2_LLM_PER_MODEL_KWARGS") or {}).values():
            m.pop("api_key", None)
        if eff != config.get("llm_env"):
            raise SystemExit(f"effective TAU2_LLM_* settings differ from the driver's: {sorted(eff)}")

    def checkpoint() -> None:
        # Written after every rollout, so a killed unit loses at most the one in flight.
        out = {"tasks": tasks_out, "simulations": sims_out, "failures": failures_out,
               "started": started}
        if config is not None:
            out["_config"] = config
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out + ".tmp", "w") as f:
            json.dump(out, f, default=str)
        os.replace(args.out + ".tmp", args.out)

    for td in task_dicts:
        sched = schedule_from_task(td) if args.arm == "perturbed" else None
        td = apply_gold_remap(td, args.arm, args.domain)
        td_clean = {k: v for k, v in td.items() if not k.startswith("_")}
        task = Task.model_validate(td_clean)
        tasks_out.append(json.loads(task.model_dump_json()))
        for i in range(args.k):
            seed = args.seed + i
            print(
                f"[run] task={task.id} arm={args.arm} trial={i} seed={seed} "
                f"knob={sched.knob if sched else '-'} guard={args.write_guard}",
                file=sys.stderr,
            )
            started.append(seed)
            checkpoint()
            try:
                sim_d = run_one(
                    args.domain, task, args.model, sched, seed,
                    args.max_steps, args.timeout,
                    policy_suffix=policy_suffix,
                    write_guard=args.write_guard,
                )
            except Exception as e:
                sim_d = {"_failure": f"{type(e).__name__}: {e}", "seed": seed,
                         "task_id": task.id, "messages": []}
            if "_failure" in sim_d:
                print(f"[run] task={task.id} trial={i} FAILED: {sim_d['_failure']}",
                      file=sys.stderr)
                failures_out.append(sim_d)
            else:
                sims_out.append(sim_d)
            checkpoint()

    checkpoint()
    print(f"[run] wrote {len(sims_out)} sims, {len(failures_out)} failures -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
