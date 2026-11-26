"""WEAR (Wrong-Entity Action Rate) scorer for tau2-bench results.json files.

For each trajectory in a results file, every write call the agent made is compared
with the task's gold write actions. A write is eligible when it is aligned to
exactly one gold write that names a target record, and it is a wrong-entity write
when its target identifier differs from that gold write's. Attempted WEAR is
wrong-entity writes over eligible writes; effective WEAR counts only the
wrong-entity writes whose tool response reported no error. Identifiers are
compared after str(), whitespace stripping and lowercasing.

This scorer aligns a write only when the task has a single gold write for the
write's tool. scorer_extended.py adds annotated alignment for tasks with several.

The write tools of each domain, and the argument keys that name a target record
("_object_id_keys"), are read from write_tools.json next to this file. This is a
library module; score_batch.py (a run directory) is the command-line entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


def load_write_tools(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load write_tools.json (by default the one next to this file).

    Returns a dict mapping each domain to its write-tool names, plus
    "_object_id_keys" mapping each domain to the argument keys that name a target record.
    """
    if path is None:
        path = Path(__file__).with_name("write_tools.json")
    with open(path) as f:
        data = json.load(f)
    return data


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One agent write call; errored is True when its tool response reported an error."""
    name: str
    arguments: dict[str, Any]
    errored: bool = False


def extract_agent_write_calls(
    messages: Iterable[dict], write_tools: set[str]
) -> list[ToolCall]:
    """Return the write calls the agent made, in message order.

    Read calls and calls requested by the user simulator are skipped. A call is
    marked errored when the tool message with its call id has a true "error" field.
    """
    err_by_id: dict[str, bool] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("id") is not None:
            err_by_id[m["id"]] = bool(m.get("error"))

    calls: list[ToolCall] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            if tc.get("requestor", "assistant") != "assistant":
                continue
            name = tc.get("name")
            if name in write_tools:
                calls.append(ToolCall(
                    name=name,
                    arguments=dict(tc.get("arguments") or {}),
                    errored=err_by_id.get(tc.get("id"), False),
                ))
    return calls


@dataclass
class GoldAction:
    """One gold write action; compare_args, if set, lists the arguments tau2 compares."""
    name: str
    arguments: dict[str, Any]
    compare_args: Optional[list[str]] = None


def extract_gold_write_actions(task: dict, write_tools: set[str]) -> list[GoldAction]:
    """Return the gold write actions listed in task["evaluation_criteria"]["actions"]."""
    ec = task.get("evaluation_criteria") or {}
    out: list[GoldAction] = []
    for a in ec.get("actions") or []:
        if a.get("name") in write_tools:
            out.append(
                GoldAction(
                    name=a["name"],
                    arguments=dict(a.get("arguments") or {}),
                    compare_args=a.get("compare_args"),
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Target identifiers and matching
# --------------------------------------------------------------------------- #
# Returned by _entity_keys when a gold action has compare_args == []: tau2 then
# matches by tool name only, so the write counts as correct whatever its target.
NAME_ONLY = object()


def _entity_keys(
    arguments: dict[str, Any],
    object_id_keys: set[str],
    compare_args: Optional[list[str]],
):
    """Return the target-identifier keys to compare for one gold action.

    NAME_ONLY if compare_args is []; otherwise the keys among compare_args (or, if
    compare_args is None, among the gold arguments) that are in object_id_keys. An
    empty list means the gold action names no target, so the write cannot be assessed.
    """
    if compare_args is not None:
        if len(compare_args) == 0:
            return NAME_ONLY
        return [k for k in compare_args if k in object_id_keys]
    return [k for k in arguments if k in object_id_keys]


def _signature(args: dict[str, Any], keys: list[str]) -> Optional[tuple]:
    """Return the sorted (key, normalized value) pairs for keys, or None if keys is empty.

    A key missing from args is normalized as the string "none".
    """
    if not keys:
        return None
    return tuple(sorted((k, _norm(args.get(k))) for k in keys))


def _norm(v: Any) -> str:
    """Normalize an identifier for comparison: str(), strip whitespace, lowercase."""
    return str(v).strip().lower()


# --------------------------------------------------------------------------- #
# Per-simulation scoring
# --------------------------------------------------------------------------- #
@dataclass
class SimWear:
    """WEAR counts for one trajectory.

    n_write_calls counts all agent write calls. n_spurious counts writes whose tool
    is used by no gold write (unmatched); the rest are n_shape_ok. Of those,
    n_ambiguous use a tool with several gold writes, n_no_id_check align to a gold
    write that names no target, and n_correct plus n_wrong_entity form the eligible
    set (n_correct includes name-only matches). n_wrong_entity_effective counts the
    wrong-entity writes that returned no tool error.
    """
    sim_id: Any
    task_id: Any
    n_write_calls: int = 0
    n_shape_ok: int = 0
    n_correct: int = 0
    n_wrong_entity: int = 0
    n_wrong_entity_effective: int = 0
    n_spurious: int = 0
    n_no_id_check: int = 0
    n_ambiguous: int = 0  # tool has several gold writes and no alignment annotation
    has_wrong_bind: bool = False
    task_success: Optional[bool] = None
    reward: Optional[float] = None
    wrong_examples: list[dict] = field(default_factory=list)

    @property
    def n_eligible(self) -> int:
        """|E|: writes aligned to exactly one gold write that names a target (correct + wrong)."""
        return self.n_correct + self.n_wrong_entity

    @property
    def wear(self) -> Optional[float]:
        """Attempted WEAR, |W| / |E|: wrong-entity writes over eligible writes.

        Unmatched, ambiguous and unassessable writes are outside E and are counted
        separately. None when E is empty (WEAR undefined).
        """
        if self.n_eligible == 0:
            return None
        return self.n_wrong_entity / self.n_eligible

    @property
    def effective_wear(self) -> Optional[float]:
        """Effective WEAR, |W_ok| / |E|: wrong-entity writes without a tool error over
        eligible writes. None when E is empty."""
        if self.n_eligible == 0:
            return None
        return self.n_wrong_entity_effective / self.n_eligible

    @property
    def wear_all_writes(self) -> Optional[float]:
        """Wrong-entity writes over all write calls: the simpler rule the paper compares
        WEAR against. Not WEAR, because unmatched writes enlarge the denominator."""
        if self.n_write_calls == 0:
            return None
        return self.n_wrong_entity / self.n_write_calls

    @property
    def no_eligible_write(self) -> bool:
        """True when the trajectory has writes but none is eligible, so WEAR is undefined."""
        return self.n_write_calls > 0 and self.n_eligible == 0

    @property
    def wear_shapeok(self) -> Optional[float]:
        """Secondary rate: wrong-entity writes over writes whose tool has a gold write."""
        if self.n_shape_ok == 0:
            return None
        return self.n_wrong_entity / self.n_shape_ok


def score_simulation(
    sim: dict,
    task: dict,
    write_tools: set[str],
    object_id_keys: set[str],
) -> SimWear:
    """Score one trajectory against its task and return its SimWear counts."""
    calls = extract_agent_write_calls(sim.get("messages") or [], write_tools)
    golds = extract_gold_write_actions(task, write_tools)

    reward = (sim.get("reward_info") or {}).get("reward")
    res = SimWear(
        sim_id=sim.get("id"),
        task_id=sim.get("task_id"),
        reward=reward,
        task_success=(None if reward is None else float(reward) >= 1.0),
    )

    gold_by_name: dict[str, list[GoldAction]] = {}
    for g in golds:
        gold_by_name.setdefault(g.name, []).append(g)

    for c in calls:
        res.n_write_calls += 1
        cands = gold_by_name.get(c.name)
        if not cands:
            res.n_spurious += 1
            continue
        res.n_shape_ok += 1
        if len(cands) > 1:
            # Several gold writes use this tool. Without an alignment annotation the
            # write cannot be aligned; scorer_extended.py aligns such writes by annotation.
            res.n_ambiguous += 1
            continue

        # Exactly one gold write uses this tool: compare the target fields.
        matched = False
        assessable = False
        gold_sigs: list[dict] = []
        bound_vals: Optional[dict] = None
        for g in cands:
            keys = _entity_keys(g.arguments, object_id_keys, g.compare_args)
            if keys is NAME_ONLY:
                matched = True
                break
            if not keys:
                continue
            assessable = True
            gsig = _signature(g.arguments, keys)
            csig = _signature(c.arguments, keys)
            gold_sigs.append(dict(gsig))
            if bound_vals is None:
                bound_vals = dict(csig)
            if csig == gsig:
                matched = True
                break

        if matched:
            res.n_correct += 1
        elif assessable:
            res.n_wrong_entity += 1
            if not c.errored:
                res.n_wrong_entity_effective += 1
            res.wrong_examples.append(
                {"tool": c.name, "bound": bound_vals, "gold": gold_sigs,
                 "errored": c.errored}
            )
        else:
            res.n_no_id_check += 1

    res.has_wrong_bind = res.n_wrong_entity > 0
    return res


# --------------------------------------------------------------------------- #
# Whole results.json
# --------------------------------------------------------------------------- #
def score_results(
    results: dict,
    domain: str,
    write_tools_data: Optional[dict] = None,
) -> list[SimWear]:
    """Score every simulation of a results.json dict for one domain.

    Simulations whose task is missing from results["tasks"] are skipped.
    """
    if write_tools_data is None:
        write_tools_data = load_write_tools()
    write_tools = set(write_tools_data[domain])
    object_id_keys = set((write_tools_data.get("_object_id_keys") or {}).get(domain, []))

    tasks_by_id = {t.get("id"): t for t in (results.get("tasks") or [])}
    out: list[SimWear] = []
    for sim in results.get("simulations") or []:
        task = tasks_by_id.get(sim.get("task_id"))
        if task is None:
            continue
        out.append(score_simulation(sim, task, write_tools, object_id_keys))
    return out


def contingency_table(sim_results: list[SimWear]) -> dict[str, Any]:
    """Summarize the trajectories that contain at least one write.

    Returns a 2x2 table of task success (reward >= 1) by wrong-entity write (any vs
    none), mean attempted and effective WEAR over the trajectories where WEAR is
    defined, and write-call totals. The key cell counts trajectories that succeed
    although they contain a wrong-entity write. Trajectories without a reward are
    counted in n_unknown_success and left out of the table.
    """
    cells = {
        ("success", "wrong"): 0,
        ("success", "clean"): 0,
        ("fail", "wrong"): 0,
        ("fail", "clean"): 0,
    }
    n_unknown_success = 0
    scored = [r for r in sim_results if r.n_write_calls > 0]
    for r in scored:
        if r.task_success is None:
            n_unknown_success += 1
            continue
        s = "success" if r.task_success else "fail"
        w = "wrong" if r.has_wrong_bind else "clean"
        cells[(s, w)] += 1

    wears = [r.wear for r in scored if r.wear is not None]
    return {
        "n_sims_total": len(sim_results),
        "n_sims_with_write": len(scored),
        "n_unknown_success": n_unknown_success,
        "cells": {f"{s}/{w}": v for (s, w), v in cells.items()},
        "key_cell_success_and_wrong": cells[("success", "wrong")],
        "mean_wear": (sum(wears) / len(wears)) if wears else None,
        "mean_effective_wear": (
            sum(r.effective_wear for r in scored if r.effective_wear is not None)
            / len(wears)
        ) if wears else None,
        "n_sims_with_any_wrong_bind": sum(1 for r in scored if r.has_wrong_bind),
        "total_write_calls": sum(r.n_write_calls for r in scored),
        "total_wrong_entity": sum(r.n_wrong_entity for r in scored),
        "total_wrong_entity_effective": sum(r.n_wrong_entity_effective for r in scored),
        "total_spurious": sum(r.n_spurious for r in scored),
        "total_no_id_check": sum(r.n_no_id_check for r in scored),
    }


def sim_to_dict(r: SimWear) -> dict:
    """Return a SimWear as a plain dict, including its computed rates."""
    d = asdict(r)
    d["wear"] = r.wear
    d["effective_wear"] = r.effective_wear
    d["wear_shapeok"] = r.wear_shapeok
    return d
