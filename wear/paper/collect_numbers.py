"""Recompute the numbers reported in the paper from the raw run files and write NUMBERS.json.

Each collect_* function re-scores one experiment from its trajectories (WEAR, the
timing check, expected end-state hashes from an offline replay of the gold actions)
or aggregates stored judge outputs, and returns a dict; main() stores each under one
top-level key. Two inputs are stored summaries rather than trajectories: the
notice-rate rows of the cue-present variant (runs/cue_present/notice.json) and of
the five base tasks of the ID-REUSE variants (runs/nodistractor_singlechar/notice_distinct.json).
make_figures.py renders the figures and generated tables from NUMBERS.json, and
test_paper_numbers.py checks the manuscript against it. Running the script again on
the same run files gives a byte-identical NUMBERS.json.

Main top-level keys (short internal names):
  fig2                   main experiment: 20 ID-REUSE tasks, 3 conditions, 5 models
                         (the main results table)
  fig3                   APPEND and REORDER tasks
  judge_prompts                   general-purpose judge prompts
  sixgate_judge                   six-gate judge, with the end-state comparison column
  sixgate_gemini_judge        six-gate judge with Gemini Pro as the judge
  fig4_ts                ToolSandbox replication
  recover                recovery suite, with the labels of forensic_recover.py
  mitigation, mitigation_placebo, mitigation_task_means, mitigation_overhead
                         verification policy and placebo
  cue_present               cue-present variant
  nodistractor_singlechar              no-distractor and single-character variants
  reread_timing          timing of reads between the binding read and the first write
  binding_extended, recovery_authorization
                         extended scoring and authorized repairs
  exact_p                exact paired sign-flip p values
  flash_sampling         Flash's first five rollouts per task
  safeguard_study          safeguard study (policy, CAS guard, identity guard)
  stats_params           parameters of the statistical procedures

Usage (repo root): tau2-bench/.venv/bin/python wear/paper/collect_numbers.py
Reads: wear/runs/ (ToolSandbox runs under wear/runs/toolsandbox/), the task files in
wear/deictic_tasks/, wear/annotations/recovery_authorization_plans.json, and the
tau2-bench domain sources (to list the read tools).
Writes: wear/paper/NUMBERS.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# use litellm's bundled model cost map, so that importing tau2 needs no network access
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

ROOT = Path(__file__).resolve().parents[2]
WEAR = ROOT / "wear"
sys.path.insert(0, str(WEAR))
sys.path.insert(0, str(WEAR / "toolsandbox"))
sys.path.insert(0, str(WEAR / "paper"))

from eligibility import (assess, assess_results, load_perturb_index,  # noqa: E402
                         _flat_calls, _results_by_call_id)
from scorer import (load_write_tools, score_results,  # noqa: E402
                    extract_gold_write_actions, score_simulation)
from scorer_extended import score_binding  # noqa: E402
from score_batch import build_report as fig2_build_report  # noqa: E402
from stats import paired_permutation, paired_sign_flip_exact, wilson  # noqa: E402
from expected_hash import expected_hash  # noqa: E402
from score_wear_ts import score_file as ts_score_file  # noqa: E402
from forensic_recover import RECOVER_POOL_GLOBS, classify_recover  # noqa: E402

RUNS = WEAR / "runs"
TASKS = WEAR / "deictic_tasks"
OUT_JSON = WEAR / "paper" / "NUMBERS.json"

# agent models of the main experiment, by run-directory name
MODELS = ["openai_gpt_5_5", "openai_gemini_3_1_pro", "openai_gemini_3_5_flash",
          "openai_deepseek_v4_pro", "openai_qwen3_7_plus"]
MODEL_SHORT = {
    "openai_gpt_5_5": "gpt-5.5",
    "openai_gemini_3_1_pro": "gemini-3.1-pro",
    "openai_gemini_3_5_flash": "gemini-3.5-flash",
    "openai_deepseek_v4_pro": "deepseek-v4-pro",
    "openai_qwen3_7_plus": "qwen3.7-plus",
}

# ---------------------------------------------------------------- main experiment (fig2)
def collect_fig2() -> dict:
    """Main experiment: per-cell WEAR with bootstrap and Wilson intervals, and Monte-Carlo
    paired p values against baseline and control (score_batch.build_report)."""
    return fig2_build_report(str(RUNS / "main_experiment"), str(TASKS))


# ---------------------------------------------------------------- APPEND and REORDER (fig3)
FIG3_MODELS = ["openai_gpt_5_5", "openai_gemini_3_5_flash"]


def collect_fig3() -> dict:
    """APPEND and REORDER tasks (GPT-5.5 and Flash), per model, condition and perturbation.

    Counts rollouts, writes, wrong writes and affected tasks. In the perturbed condition
    it adds the timing-check labels and verified exposure: decoy_seen counts the
    write-bearing rollouts (decoy_seen_tasks the tasks) in which the decoy's identifier
    appeared in a tool response after the change was applied and before the first write.
    """
    wt_data = load_write_tools()
    write_tools = set(wt_data["airline"])
    out: dict = {}
    for model in FIG3_MODELS:
        for arm in ("baseline", "perturbed"):
            cell = defaultdict(lambda: {
                "n_sims_total": 0, "n_sims_with_writes": 0, "n_sims_wrong": 0,
                "n_write_calls": 0, "n_wrong_writes": 0, "n_wrong_effective": 0,
                "tasks": {}, "eligibility": None, "decoy_seen": None,
            })
            elig_rows = defaultdict(list)
            seen_tasks: dict = defaultdict(set)
            for f in sorted(glob.glob(str(RUNS / "append_reorder" / model / arm / "*.json"))):
                task_id = Path(f).stem
                knob = "append" if "_append_" in task_id else "reorder"
                results = json.load(open(f))
                sims = score_results(results, "airline", wt_data)
                c = cell[knob]
                t_wrong = 0
                for s in sims:
                    c["n_sims_total"] += 1
                    if s.n_write_calls > 0:
                        c["n_sims_with_writes"] += 1
                    c["n_write_calls"] += s.n_write_calls
                    c["n_wrong_writes"] += s.n_wrong_entity
                    c["n_wrong_effective"] += s.n_wrong_entity_effective
                    if s.has_wrong_bind:
                        c["n_sims_wrong"] += 1
                        t_wrong += 1
                c["tasks"][task_id] = {"n_sims": len(sims), "n_wrong": t_wrong}
                if arm == "perturbed":
                    pidx = load_perturb_index([str(TASKS / f"{task_id}.json")])
                    rows_t = assess_results(results, pidx, write_tools)
                    elig_rows[knob].extend(rows_t)
                    # tasks in which at least one write-bearing rollout saw the decoy
                    if any(r.first_write_flat_index is not None and r.new_id_observed
                           for r in rows_t):
                        seen_tasks[knob].add(task_id)
            assert cell, f"no fig3 unit files under {model}/{arm}"
            for knob, c in cell.items():
                n_task = len(c["tasks"])
                assert n_task == (10 if knob == "append" else 9), (knob, n_task)
                wrong_tasks = sum(1 for v in c["tasks"].values() if v["n_wrong"] > 0)
                c["n_tasks"] = n_task
                c["tasks_with_wrong_bind"] = f"{wrong_tasks}/{n_task}"
                c["wilson95_taskwrong"] = list(wilson(wrong_tasks, n_task))
                if arm == "perturbed":
                    rows = [r for r in elig_rows[knob]
                            if r.first_write_flat_index is not None]
                    lab: dict = defaultdict(int)
                    for r in rows:
                        lab[r.label] += 1
                    c["eligibility"] = dict(lab)
                    c["decoy_seen"] = f"{sum(r.new_id_observed for r in rows)}/{len(rows)}"
                    c["decoy_seen_tasks"] = f"{len(seen_tasks[knob])}/{n_task}"
            out[f"{model}/{arm}"] = {k: dict(v) for k, v in cell.items()}
    return out


# ---------------------------------------------------------------- general-purpose judge prompts (judge_prompts)
def collect_judge_prompts() -> dict:
    """General-purpose judge prompts: per judge-and-prompt directory under runs/judge_prompts, the
    wrong-binding trajectories, how many of them the judge passed, and the miss rate."""
    out = {}
    for cfg_dir in sorted((RUNS / "judge_prompts").iterdir()):
        if not cfg_dir.is_dir():
            continue
        n_wrong = n_missed = n_files = 0
        for f in sorted(cfg_dir.glob("*.json")):
            s = json.load(open(f))["summary"]
            n_wrong += s["n_wrong_bind"]
            n_missed += s["n_pae_pass_and_wrong"]
            n_files += 1
        out[cfg_dir.name] = {
            "n_task_files": n_files, "n_wrong_bind": n_wrong,
            "n_judge_pass_and_wrong": n_missed,
            "miss_rate": round(n_missed / n_wrong, 4) if n_wrong else None,
        }
    return out


# ---------------------------------------------------------------- six-gate judge (sixgate_judge)
# judge verdict (pass/fail) x WEAR's wrong-binding flag, plus units without a
# verdict or without a WEAR score
_CONT_KEYS = ("pass_wrong", "pass_clean", "fail_wrong", "fail_clean",
              "judge_error", "wear_missing")
# the six gates; per-gate failure counts are taken on wrong-binding trajectories
_GATE_KEYS = ("i_pc", "i_pf", "i_ec", "i_df", "i_intent", "i_qf")
# agent models whose main-experiment trajectories the six-gate judge and the
# end-state comparison both cover
FIG2_DBHASH_MODELS = ("openai_gpt_5_5", "openai_gemini_3_1_pro",
                      "openai_gemini_3_5_flash")


def _sixgate_judge_units_aggregate() -> dict:
    """Sum the stored per-unit six-gate judge outputs in runs/sixgate_judge/<pool>/<gold>/*.json.

    pool is an agent model or the recovery suite; gold is what the judge is told: full
    (original gold), full_remap (corrected gold), scenario (instructions only) or none.
    Returns {"pool|gold": {contingency, gate_catch}}.
    """
    agg: dict = {}
    base = RUNS / "sixgate_judge"
    for pool_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for gt_dir in sorted(p for p in pool_dir.iterdir() if p.is_dir()):
            files = sorted(gt_dir.glob("*.json"))
            if not files:
                continue
            cont = {k: 0 for k in _CONT_KEYS}
            gate = {k: 0 for k in _GATE_KEYS}
            for f in files:
                s = json.load(open(f)).get("summary") or {}
                c = s.get("contingency_pae_gate_x_wrong_bind") or {}
                g = s.get("per_gate_fail_counts_on_wrong_bind_sims") or {}
                for k in _CONT_KEYS:
                    cont[k] += int(c.get(k, 0))
                for k in _GATE_KEYS:
                    gate[k] += int(g.get(k, 0))
            agg[f"{pool_dir.name}|{gt_dir.name}"] = {"contingency": cont, "gate_catch": gate}
    return agg


def _sixgate_judge_db_hash_column(recover: dict, wt_data: dict) -> dict:
    """End-state comparison: compare each perturbed trajectory's final database hash with
    the expected hash (gold actions replayed on the perturbed database).

    Per model: visible = wrong-binding trajectories whose end state differs (flagged),
    blind = wrong-binding trajectories whose end state matches (missed), clean_flagged =
    trajectories without a wrong write whose end state differs. The recovery suite adds
    its recovered (KEY) and fail_wrong counts from collect_recover.
    """
    fig2: dict = {}
    exp_cache: dict[str, str] = {}
    for model in FIG2_DBHASH_MODELS:
        visible = blind = clean_flagged = 0
        for f in sorted(glob.glob(str(RUNS / "main_experiment" / model / "perturbed" / "*.json"))):
            domain = Path(f).stem.split("_", 1)[0]
            results = json.load(open(f))
            sims_raw = {str(s.get("id")): s for s in results.get("simulations") or []}
            for sw in score_results(results, domain, wt_data):
                tid = sw.task_id
                if tid not in exp_cache:
                    task = json.load(open(TASKS / f"{tid}.json"))
                    exp_cache[tid] = expected_hash(
                        task[0] if isinstance(task, list) else task, "perturbed", domain)
                raw = sims_raw.get(str(sw.sim_id))
                final = raw.get("_final_db_hash") if raw else None
                match = final is not None and final == exp_cache[tid]
                if sw.has_wrong_bind:
                    blind += int(match)
                    visible += int(not match)
                elif not match:
                    clean_flagged += 1
        fig2[model] = {"visible": visible, "blind": blind, "clean_flagged": clean_flagged}
    rt = recover["total"]
    return {
        "fig2_perturbed": fig2,
        "recover_pool": {"KEY_wrong_and_recovered": rt["KEY"], "fail_wrong": rt["fail_wrong"]},
    }


def collect_sixgate_judge(recover: dict) -> dict:
    """Six-gate judge: the aggregated outputs, the miss rate per pool and gold condition,
    the end-state column, and coverage totals (wrong-binding trajectories, those passed
    by the judge under original and corrected gold, those missed by end-state comparison)."""
    wt_data = load_write_tools()
    agg = _sixgate_judge_units_aggregate()
    agg["db_hash_column"] = _sixgate_judge_db_hash_column(recover, wt_data)
    derived = {}
    total_wrong = total_missed_full = total_missed_full_remap = 0
    for key, v in agg.items():
        if key == "db_hash_column":
            continue
        cont = v["contingency"]
        wrong = cont["pass_wrong"] + cont["fail_wrong"]
        derived[key] = {
            "n_wrong": wrong, "n_clean": cont["pass_clean"] + cont["fail_clean"],
            "miss_rate": round(cont["pass_wrong"] / wrong, 4) if wrong else None,
        }
        if key.endswith("|full"):  # endswith excludes |full_remap
            total_wrong += wrong
            total_missed_full += cont["pass_wrong"]
        elif key.endswith("|full_remap"):
            total_missed_full_remap += cont["pass_wrong"]
    dbh = agg["db_hash_column"]
    db_blind = (sum(m.get("blind", 0) for m in dbh["fig2_perturbed"].values())
                + dbh["recover_pool"]["KEY_wrong_and_recovered"])
    return {
        "aggregate": agg,
        "derived": derived,
        "coverage": {
            "total_wrong_bind_trajectories": total_wrong,
            "sixgate_fullGT_missed": total_missed_full,
            "sixgate_correctedGT_missed": total_missed_full_remap,
            "db_diff_blind_transient": db_blind,
            "wear_caught": total_wrong,
        },
    }


# ---------------------------------------------------------------- ToolSandbox replication (fig4_ts)
# gold target identifier of each scenario (the record named by its gold milestone)
TS_GOLD = {
    "modify_contact_disambiguation_binding_attack": "c1b6eeb7-144c-5c70-9989-a4379f015ee6",
    "modify_contact_coreference_stale_handle_binding_attack": "c1b6eeb7-144c-5c70-9989-a4379f015ee6",
    "modify_reminder_disambiguation_static_control": "da2b856e-9390-511f-bcbe-7005f72cbf5e",
    "modify_contact_idreuse_stale_cache_binding_attack": "cf1697b7-bc76-572c-8927-e3e9268cce9f",
    "modify_reminder_idreuse_stale_cache_binding_attack": "38e978a9-0a6b-5837-a86a-4bdd41b7debe",
    "modify_contact_idreuse_swapped_pair_binding_attack": "1b310f28-cb61-58fc-ab99-3725c005326d",
    "modify_contact_idreuse_cue_present_binding_attack": "cf1697b7-bc76-572c-8927-e3e9268cce9f",
    "modify_contact_idreuse_no_distractor_binding_attack": "14e2f13a-548f-5e25-8b7d-65c86c5d8abb",
}
# scenario group, as in the ToolSandbox figure
TS_GROUP = {
    "modify_contact_disambiguation_binding_attack": "static",
    "modify_contact_coreference_stale_handle_binding_attack": "static",
    "modify_reminder_disambiguation_static_control": "static",
    "modify_contact_idreuse_stale_cache_binding_attack": "idreuse",
    "modify_reminder_idreuse_stale_cache_binding_attack": "idreuse",
    "modify_contact_idreuse_swapped_pair_binding_attack": "idreuse",
    "modify_contact_idreuse_cue_present_binding_attack": "cue_present",
    "modify_contact_idreuse_no_distractor_binding_attack": "no_distractor",
}


def collect_fig4_ts() -> dict:
    """ToolSandbox replication: score every saved run of each scenario against its gold
    target (score_wear_ts). Per model and scenario: mean attempted and effective WEAR
    over runs with a write, and counts of runs without a write or without a saved
    conversation; _groups holds the mean over the scenarios of each group."""
    base = RUNS / "toolsandbox"
    out: dict = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        per_scen = defaultdict(lambda: {"runs": [], "missing": 0})
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir() or "_run" not in run_dir.name:
                continue
            scen = run_dir.name.rsplit("_run", 1)[0]
            if scen not in TS_GOLD:
                continue
            convs = sorted(run_dir.rglob("conversation.json"))
            if not convs:
                per_scen[scen]["missing"] += 1
                continue
            # a run dir may hold several candidate trees: score all; divergence fails loudly
            scored_all = [ts_score_file(str(c), TS_GOLD[scen]) for c in convs]
            triples = {(x.n_write_calls, x.wear, x.effective_wear)
                       for x in scored_all}
            if len(triples) != 1:
                raise RuntimeError(
                    f"{run_dir}: {len(convs)} conversation.json candidates "
                    f"score differently ({sorted(triples)}); resolve explicitly")
            r = scored_all[0]
            per_scen[scen]["runs"].append({
                "run": run_dir.name, "n_write_calls": r.n_write_calls,
                "attempted": r.wear, "effective": r.effective_wear,
            })
        model_out = {}
        for scen, v in per_scen.items():
            scored = [x for x in v["runs"] if x["attempted"] is not None]
            model_out[scen] = {
                "group": TS_GROUP[scen],
                "n_runs_scored": len(scored),
                "n_zero_write": len(v["runs"]) - len(scored),
                "n_missing_conv": v["missing"],
                "attempted_mean": (round(sum(x["attempted"] for x in scored) / len(scored), 4)
                                   if scored else None),
                "effective_mean": (round(sum(x["effective"] for x in scored) / len(scored), 4)
                                   if scored else None),
            }
        grp_scens: dict[str, list] = defaultdict(list)
        for scen, v in model_out.items():
            if v["attempted_mean"] is not None and v["effective_mean"] is not None:
                grp_scens[v["group"]].append(v)
        model_out["_groups"] = {
            g: {"attempted_mean": sum(x["attempted_mean"] for x in xs) / len(xs),
                "effective_mean": sum(x["effective_mean"] for x in xs) / len(xs),
                "n_scenarios": len(xs)}
            for g, xs in grp_scens.items()}
        out[model_dir.name] = model_out
    return out


# ---------------------------------------------------------------- Gemini Pro as six-gate judge (sixgate_gemini_judge)
def collect_crossjudge() -> dict | None:
    """Six-gate judge with Gemini Pro judging GPT-5.5's trajectories (runs/sixgate_gemini_judge):
    contingency and miss rate under original gold, and under corrected gold as
    full_remap; None if the runs are absent."""
    base = RUNS / "sixgate_gemini_judge" / "openai_gpt_5_5"

    def _one(gt: str) -> dict | None:
        d = base / gt
        if not d.exists():
            return None
        cont = {"pass_wrong": 0, "pass_clean": 0, "fail_wrong": 0, "fail_clean": 0,
                "judge_error": 0, "wear_missing": 0}
        n_files = 0
        for f in sorted(d.glob("*.json")):
            s = json.load(open(f))["summary"]
            assert s["judge"] != "gpt-5.5", f"unexpected judge in {f}"
            for k in cont:
                cont[k] += s["contingency_pae_gate_x_wrong_bind"].get(k, 0)
            n_files += 1
        wrong = cont["pass_wrong"] + cont["fail_wrong"]
        return {"n_unit_files": n_files, "contingency": cont,
                "miss_rate": round(cont["pass_wrong"] / wrong, 4) if wrong else None}

    full = _one("full")
    if full is None:
        return None
    out = {"judge": "gemini-3.1-pro", "gt": "full", "n_unit_files": full["n_unit_files"],
           "contingency": full["contingency"], "miss_rate": full["miss_rate"]}
    remap = _one("full_remap")
    if remap is not None:
        out["full_remap"] = remap
    return out


# ---------------------------------------------------------------- verification policy and placebo
MITIGATION_MODELS = ["openai_gpt_5_5", "openai_deepseek_v4_pro"]


def collect_mitigation() -> dict | None:
    """Verification policy (runs/policy): _collect_mitigation_for each model, or None
    if no model has runs."""
    out = {}
    for mslug in MITIGATION_MODELS:
        v = _collect_mitigation_for(mslug)
        if v:
            out[mslug] = v
    return out or None


def _collect_mitigation_for(mslug: str) -> dict | None:
    """One model under the verification policy. Per condition: rollout counts, affected
    tasks with their Wilson interval, and mean task reward. Also a Monte-Carlo paired
    test of perturbed task-mean WEAR against the main experiment (no safeguard), and
    the main experiment's baseline reward."""
    base = RUNS / "policy" / mslug
    if not base.exists():
        return None
    wt_data = load_write_tools()
    out: dict = {}
    task_means: dict[str, dict[str, float]] = {}
    for cond in ("baseline", "perturbed"):
        cd = base / cond
        if not cd.exists():
            continue
        n_sims = n_with_writes = n_wrong = n_zero_write = 0
        rewards: list[float] = []
        tasks: dict[str, int] = {}
        means: dict[str, float] = {}
        for f in sorted(cd.glob("*.json")):
            task_id = Path(f).stem
            domain = task_id.split("_", 1)[0]
            wears: list[float] = []
            for s in score_results(json.load(open(f)), domain, wt_data):
                n_sims += 1
                if s.n_write_calls > 0:
                    n_with_writes += 1
                    if not s.no_eligible_write:  # else WEAR is undefined: no eligible write
                        wears.append(s.wear)
                else:
                    n_zero_write += 1
                if s.has_wrong_bind:
                    n_wrong += 1
                    tasks[task_id] = tasks.get(task_id, 0) + 1
                if s.reward is not None:
                    rewards.append(float(s.reward))
            tasks.setdefault(task_id, 0)
            if wears:
                means[task_id] = sum(wears) / len(wears)
        task_means[cond] = means
        n_task = len(tasks)
        wrong_tasks = sum(1 for v in tasks.values() if v > 0)
        out[cond] = {
            "n_tasks": n_task, "n_sims": n_sims, "n_sims_with_writes": n_with_writes,
            "n_sims_zero_write": n_zero_write, "n_sims_wrong": n_wrong,
            "tasks_with_wrong_bind": f"{wrong_tasks}/{n_task}",
            "wilson95_taskwrong": list(wilson(wrong_tasks, n_task)) if n_task else None,
            "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "n_reward": len(rewards),
        }
    if not out:
        return None
    if "perturbed" in task_means:
        unmit: dict[str, float] = {}
        for f in sorted((RUNS / "main_experiment" / mslug / "perturbed").glob("*.json")):
            task_id = Path(f).stem
            domain = task_id.split("_", 1)[0]
            ws = [s.wear for s in score_results(json.load(open(f)), domain, wt_data)
                  if s.wear is not None and not s.no_eligible_write]
            if ws:
                unmit[task_id] = sum(ws) / len(ws)
        common = sorted(set(unmit) & set(task_means["perturbed"]))
        if common:
            obs, p = paired_permutation(
                {t: (task_means["perturbed"][t], unmit[t]) for t in common})
            out["paired_vs_unmitigated_perturbed"] = {
                "n_task_pairs": len(common), "mean_diff_unmit_minus_mit": round(obs, 4),
                "p_two_sided": p,
            }
    unmit_rewards: list[float] = []
    for f in sorted((RUNS / "main_experiment" / mslug / "baseline").glob("*.json")):
        domain = Path(f).stem.split("_", 1)[0]
        for s in score_results(json.load(open(f)), domain, wt_data):
            if s.reward is not None:
                unmit_rewards.append(float(s.reward))
    if unmit_rewards:
        out["unmitigated_baseline"] = {
            "reward_mean": round(sum(unmit_rewards) / len(unmit_rewards), 4),
            "n_reward": len(unmit_rewards),
        }
    return out


def collect_mitigation_placebo() -> dict | None:
    """Placebo control (runs/placebo), perturbed condition only: per model,
    rollout counts, affected tasks, task-mean WEAR, and a Monte-Carlo paired test
    against the verification policy; None if there are no placebo runs."""
    wt_data = load_write_tools()
    out: dict = {}
    for mslug in MITIGATION_MODELS:
        base = RUNS / "placebo" / mslug / "perturbed"
        if not base.exists():
            continue
        n_sims = n_with_writes = n_wrong = n_zero_write = 0
        tasks: dict[str, int] = {}
        means: dict[str, float] = {}
        for f in sorted(base.glob("*.json")):
            task_id = Path(f).stem
            domain = task_id.split("_", 1)[0]
            wears: list[float] = []
            for s in score_results(json.load(open(f)), domain, wt_data):
                n_sims += 1
                if s.n_write_calls > 0:
                    n_with_writes += 1
                    if not s.no_eligible_write:  # else WEAR is undefined: no eligible write
                        wears.append(s.wear)
                else:
                    n_zero_write += 1
                if s.has_wrong_bind:
                    n_wrong += 1
                    tasks[task_id] = tasks.get(task_id, 0) + 1
            tasks.setdefault(task_id, 0)
            if wears:
                means[task_id] = sum(wears) / len(wears)
        if not n_sims:
            continue
        n_task = len(tasks)
        wrong_tasks = sum(1 for v in tasks.values() if v > 0)
        entry: dict = {
            "perturbed": {
                "n_tasks": n_task, "n_sims": n_sims,
                "n_sims_with_writes": n_with_writes,
                "n_sims_zero_write": n_zero_write, "n_sims_wrong": n_wrong,
                "tasks_with_wrong_bind": f"{wrong_tasks}/{n_task}",
                "wilson95_taskwrong": list(wilson(wrong_tasks, n_task)) if n_task else None,
                "task_mean_wear": round(sum(means.values()) / len(means), 4) if means else None,
            }
        }
        mit_means: dict[str, float] = {}
        for f in sorted((RUNS / "policy" / mslug / "perturbed").glob("*.json")):
            task_id = Path(f).stem
            domain = task_id.split("_", 1)[0]
            ws = [s.wear for s in score_results(json.load(open(f)), domain, wt_data)
                  if s.wear is not None and not s.no_eligible_write]
            if ws:
                mit_means[task_id] = sum(ws) / len(ws)
        common = sorted(set(mit_means) & set(means))
        if common:
            obs, p = paired_permutation(
                {t: (mit_means[t], means[t]) for t in common})
            entry["paired_vs_mitigation_perturbed"] = {
                "n_task_pairs": len(common),
                "mean_diff_placebo_minus_mit": round(obs, 4),
                "p_two_sided": p,
            }
        out[mslug] = entry
    return out or None


# ---------------------------------------------------------------- exploratory single-task cue-present run
def collect_pilot_signal() -> dict | None:
    """Rescore an exploratory cue-present run on one airline task, in which the user says
    that the trip was canceled and rebooked; None if the run directory is absent."""
    d = RUNS / "pilot_signal"
    if not d.exists():
        return None
    wt_data = load_write_tools()
    out = {}
    for f in sorted(d.rglob("airline_idreuse_dallas_signal.json")):
        arm = f.parent.name
        sims = score_results(json.load(open(f)), "airline", wt_data)
        with_writes = [s for s in sims if s.n_write_calls > 0]
        out[f"{f.parent.parent.name}/{arm}"] = {
            "n_sims": len(sims),
            "n_sims_with_writes": len(with_writes),
            "n_sims_wrong": sum(1 for s in with_writes if s.has_wrong_bind),
            "note": "exploratory: single task, cue-present (user states cancel-and-rebook)",
        }
    return out or None


# ---------------------------------------------------------------- recovery suite (recover)
def collect_recover() -> dict:
    """Recovery suite: per task, rollouts and wrong-binding rollouts, the latter split into
    KEY (final database equals the expected end state, so the wrong write was undone) and
    fail_wrong (it differs). forensic holds forensic_recover.classify_recover()'s labels
    of the fail_wrong trajectories."""
    wt_data = load_write_tools()
    files: list[str] = []
    for g in RECOVER_POOL_GLOBS:
        files.extend(glob.glob(str(RUNS / g)))
    files = sorted(set(files))
    exp_cache: dict[str, str] = {}
    per_task = defaultdict(lambda: {"n_sims": 0, "n_wrong": 0, "KEY": 0, "fail_wrong": 0})
    for f in files:
        results = json.load(open(f))
        sims_raw = {str(s.get("id")): s for s in results.get("simulations") or []}
        for sw in score_results(results, "retail", wt_data):
            tid = sw.task_id
            if tid not in exp_cache:
                task = json.load(open(TASKS / f"{tid}.json"))[0]
                exp_cache[tid] = expected_hash(task, "perturbed", "retail")
            c = per_task[tid]
            c["n_sims"] += 1
            if not sw.has_wrong_bind:
                continue
            c["n_wrong"] += 1
            raw = sims_raw.get(str(sw.sim_id))
            if raw is None:
                raise ValueError(f"sim {sw.sim_id} of {tid} not found in {f}")
            final = raw.get("_final_db_hash")
            if final == exp_cache[tid]:
                c["KEY"] += 1  # recovered: final db state == expected
            else:
                c["fail_wrong"] += 1
    tot = {k: sum(v[k] for v in per_task.values())
           for k in ("n_sims", "n_wrong", "KEY", "fail_wrong")}
    return {"per_task": dict(per_task), "total": tot, "n_pool_files": len(files),
            "forensic": classify_recover()}


# ---------------------------------------------------------------- reads, extended scoring, exact tests, variants
def _reread_target_split(batch: list[tuple], pidx: dict, wt_data: dict,
                         require_write: bool) -> dict:
    """Classify each agent read between the binding read and the first write by what it queries.

    batch lists (run file, task id, domain). fresh: an argument equals the new
    identifier; owner_relist: get_user_details (a re-list of the user's records); misc:
    get_flight_status or list_all_airports; stale_ctx: any other read.
    stale_ctx_pre_seen counts the stale_ctx reads whose first argument also appears in
    a tool result at or before the binding read, which shows that the identifier was
    known before the change. With require_write, trajectories without a write are
    skipped; otherwise their window runs to the end of the trajectory. Returns the counts.
    """
    split = {"stale_ctx": 0, "owner_relist": 0, "fresh": 0, "misc": 0,
             "total": 0, "stale_ctx_pre_seen": 0}
    for path, task, dom in batch:
        p = pidx[task]
        wts = set(wt_data[dom])
        new_id = str(p["spec"]["referent_new_id"]).strip().lower()
        for sim in json.load(open(path)).get("simulations") or []:
            if sim.get("task_id") != task:
                continue
            e = assess(sim, p["spec"], wts,
                       fire_after_n_calls=int(p.get("fire_after_n_calls", 1)),
                       target_tool=p.get("target_tool"))
            if e.anomaly or (require_write and e.first_write_flat_index is None):
                continue
            msgs = sim.get("messages") or []
            calls = _flat_calls(msgs)
            results = _results_by_call_id(msgs)
            hi = (e.first_write_flat_index
                  if e.first_write_flat_index is not None else len(calls))
            pre = " ".join(results.get(c["call_id"], "")
                           for c in calls[:e.trigger_flat_index + 1]
                           if c["call_id"]).lower()
            for c in calls[e.trigger_flat_index + 1:hi]:
                if c["requestor"] != "assistant" or c["name"] in wts:
                    continue
                vals = [str(v).strip().lower() for v in c["arguments"].values()]
                split["total"] += 1
                if new_id in vals:
                    split["fresh"] += 1
                elif c["name"] == "get_user_details":
                    split["owner_relist"] += 1
                elif c["name"] in ("get_flight_status", "list_all_airports"):
                    split["misc"] += 1
                else:
                    split["stale_ctx"] += 1
                    tok = vals[0] if vals else ""
                    if tok and re.search(r"(?<![a-z0-9])" + re.escape(tok)
                                         + r"(?![a-z0-9])", pre):
                        split["stale_ctx_pre_seen"] += 1
    return split


def _call_msg_index(msgs: list) -> list[int]:
    """Message index of each flat call, aligned with _flat_calls order."""
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        out.extend(i for tc in m.get("tool_calls") or [] if isinstance(tc, dict))
    return out


def collect_reread_timing() -> dict:
    """Timing of the agent reads between the binding read and the first write, in the
    write-bearing perturbed trajectories of the main experiment (the re-read table in the
    supplementary material).

    A read is same_msg if issued in the binding read's own assistant message (a parallel
    lookup, before any result was seen), after_user if a user turn intervenes, else
    later_no_user. notfound counts trajectories in which such a read returned a
    not-found error; notfound_wrote_stale counts those whose first write then went
    through the old identifier. Returns the counts per model.
    """
    wt_data = load_write_tools()
    pidx = load_perturb_index(sorted(glob.glob(str(TASKS / "*.json"))))
    out: dict = {}
    for model in MODELS:
        c = {"write_bearing": 0, "any_reread": 0, "reread_after_user": 0,
             "reads_same_msg": 0, "reads_later_no_user": 0, "reads_after_user": 0,
             "notfound": 0, "notfound_wrote_stale": 0}
        for f in sorted(glob.glob(str(RUNS / "main_experiment" / model / "perturbed" / "*.json"))):
            task = Path(f).stem
            p = pidx.get(task)
            if not p or "referent_new_id" not in p["spec"]:
                continue
            spec = p["spec"]
            wts = set(wt_data["airline" if task.startswith("airline") else "retail"])
            for sim in json.load(open(f)).get("simulations") or []:
                if sim.get("task_id") != task:
                    continue
                e = assess(sim, spec, wts,
                           fire_after_n_calls=int(p.get("fire_after_n_calls", 1)),
                           target_tool=p.get("target_tool"))
                if e.anomaly or e.first_write_flat_index is None:
                    continue
                msgs = sim.get("messages") or []
                calls = _flat_calls(msgs)
                midx = _call_msg_index(msgs)
                results = _results_by_call_id(msgs)
                t, w = e.trigger_flat_index, e.first_write_flat_index
                c["write_bearing"] += 1
                kinds, nf = set(), False
                for k in range(t + 1, w):
                    cl = calls[k]
                    if cl["requestor"] != "assistant" or cl["name"] in wts:
                        continue
                    if midx[k] == midx[t]:
                        kind = "same_msg"
                    elif any(isinstance(m, dict) and m.get("role") == "user"
                             for m in msgs[midx[t]:midx[k]]):
                        kind = "after_user"
                    else:
                        kind = "later_no_user"
                    c[f"reads_{kind}"] += 1
                    kinds.add(kind)
                    r = results.get(cl["call_id"], "").lower()
                    nf = nf or (r.startswith("error") and "not found" in r)
                c["any_reread"] += int(bool(kinds))
                c["reread_after_user"] += int("after_user" in kinds)
                stale = (str(calls[w]["arguments"].get(spec["trigger_arg"], "")).strip().lower()
                         == str(spec["referent_id"]).strip().lower())
                c["notfound"] += int(nf)
                c["notfound_wrote_stale"] += int(nf and stale)
        out[model] = c
    return out


def collect_binding_extended() -> dict:
    """Extended scoring (scorer_extended) of every main-experiment trajectory: the counts
    of the extended-scoring table in the supplementary material, and task-mean WEAR per cell.

    Every main task has exactly one gold write, so the extended scorer's WEAR must equal
    scorer.py's for every trajectory (asserted). It may differ from the simpler rule (all
    writes as the denominator) only when every write is unmatched; such trajectories are
    listed in diverging_trajectories, and a cell whose task mean changes also keeps the
    simpler-rule mean as task_mean_wear_submitted.
    """
    wt = load_write_tools()
    keys_all = ("n_traj", "no_write", "n_writes", "n_eligible", "n_wrong",
                "n_unmatched", "n_unassessed", "n_wrong_nonerror")
    cells: dict = {}
    diverging: list = []
    for f in sorted(glob.glob(str(RUNS / "main_experiment" / "*" / "*" / "*.json"))):
        model, cond = Path(f).parts[-3], Path(f).parts[-2]
        dom = Path(f).stem.split("_", 1)[0]
        wts, keys = set(wt[dom]), set(wt["_object_id_keys"][dom])
        data = json.load(open(f))
        tmap = {t["id"]: t for t in data.get("tasks") or []}
        c = cells.setdefault(f"{model}/{dom}/{cond}",
                             {**{k: 0 for k in keys_all}, "_ext": defaultdict(list),
                              "_orig": defaultdict(list)})
        for sim in data.get("simulations") or []:
            task = tmap[sim["task_id"]]
            assert len(extract_gold_write_actions(task, wts)) == 1, sim["task_id"]
            r = score_binding(sim, task, wts, keys).to_dict()
            o = score_simulation(sim, task, wts, keys)
            assert o.n_write_calls == r["n_writes"] and o.n_wrong_entity == r["n_wrong_target"]
            c["n_traj"] += 1
            c["no_write"] += int(r["no_write"])
            c["n_writes"] += r["n_writes"]
            c["n_eligible"] += r["n_eligible"]
            c["n_wrong"] += r["n_wrong_target"]
            c["n_unmatched"] += r["n_unmatched"]
            c["n_unassessed"] += r["n_unassessable"] + r["n_ambiguous"]
            c["n_wrong_nonerror"] += r["n_wrong_nonerror"]
            if r["binding_error_rate"] is not None:
                c["_ext"][sim["task_id"]].append(r["binding_error_rate"])
            if o.wear_all_writes is not None:  # the simpler rule: wrong writes over all writes
                c["_orig"][sim["task_id"]].append(o.wear_all_writes)
            assert o.wear == r["binding_error_rate"]  # both use the eligible denominator
            if r["binding_error_rate"] != o.wear_all_writes:
                # The only admitted divergence: every write is unmatched (a tool outside
                # the gold actions), so the eligible denominator is empty while the
                # simpler rule counts the writes as not wrong (WEAR 0).
                assert o.wear_all_writes == 0 and r["binding_error_rate"] is None \
                    and r["n_eligible"] == 0 and r["n_unmatched"] == r["n_writes"] > 0, \
                    f"eligible-denominator WEAR diverges: {f} {sim['id']}"
                diverging.append({"cell": f"{model}/{dom}/{cond}", "task_id": sim["task_id"],
                                  "seed": sim.get("seed"), "sim_id": sim["id"],
                                  "submitted_wear": o.wear_all_writes, "eligible_wear": None,
                                  "n_writes": r["n_writes"], "n_unmatched": r["n_unmatched"]})
    for c in cells.values():
        ext, orig = c.pop("_ext"), c.pop("_orig")
        tm = lambda d: sum(sum(v) / len(v) for v in d.values()) / len(d) if d else None
        c["task_mean_wear"] = tm(ext)
        if c["task_mean_wear"] != tm(orig):
            c["task_mean_wear_submitted"] = tm(orig)
    tot = {cond: {k: sum(v[k] for key, v in cells.items() if key.endswith("/" + cond))
                  for k in keys_all} for cond in ("baseline", "control", "perturbed")}
    tot["all"] = {k: sum(t[k] for t in tot.values()) for k in keys_all}
    return {"cells": cells, "totals": tot, "diverging_trajectories": diverging,
            "task_means_unchanged": all("task_mean_wear_submitted" not in c
                                        for c in cells.values())}


# stepwise authorization annotations of the recovery-suite trajectories that recovered
AUTH_PLANS = WEAR / "annotations" / "recovery_authorization_plans.json"


def collect_recovery_authorization() -> dict:
    """Authorized repairs: re-score the 13 recovery-suite trajectories that return to the
    gold state, without and with their annotations of the user turn that authorizes
    undoing the wrong write. Returns the write statuses and wrong-label counts of both."""
    wt = load_write_tools()
    wts, keys = set(wt["retail"]), set(wt["_object_id_keys"]["retail"])
    rows = []
    for a in json.load(open(AUTH_PLANS)):
        data = json.load(open(ROOT / a["source"]))
        sim = next(s for s in data["simulations"] if s["id"] == a["plan"]["sim_id"])
        task = next(t for t in data["tasks"] if t["id"] == sim["task_id"])
        static = score_binding(sim, task, wts, keys).to_dict()
        annotated = score_binding(sim, task, wts, keys, plan=a["plan"]).to_dict()
        rows.append({
            "task": sim["task_id"], "seed": sim.get("seed"),
            "static": [w["status"] for w in static["writes"]],
            "annotated": [w["status"] for w in annotated["writes"]],
        })
    first_wrong = all(r["static"][0] == r["annotated"][0] == "wrong_target" for r in rows)
    return {
        "n_traj": len(rows),
        "n_writes": sum(len(r["static"]) for r in rows),
        "wrong_labels_static": sum(r["static"].count("wrong_target") for r in rows),
        "wrong_labels_annotated": sum(r["annotated"].count("wrong_target") for r in rows),
        "first_write_still_wrong": first_wrong,
        "rows": rows,
    }


def _task_mean_wear(files: list) -> dict[str, float]:
    """Return {task: mean WEAR over its rollouts with a defined WEAR} for the given run
    files (one task per file); tasks without such a rollout are left out."""
    wt = load_write_tools()
    out = {}
    for f in files:
        task = Path(f).stem
        ws = [s.wear for s in score_results(json.load(open(f)), task.split("_", 1)[0], wt)
              if s.wear is not None and not s.no_eligible_write]
        if ws:
            out[task] = sum(ws) / len(ws)
    return out


def _exact(a: dict, b: dict) -> dict:
    """Exact paired sign-flip test of task means a and b over their common tasks;
    mean_diff is the mean of b - a, and p is given as a fraction and as a float."""
    common = sorted(set(a) & set(b))
    obs, p = paired_sign_flip_exact({t: (a[t], b[t]) for t in common})
    return {"n_pairs": len(common), "mean_diff": round(obs, 4),
            "p_num": p.numerator, "p_den": p.denominator, "p": float(p)}


def collect_exact_p() -> dict:
    """Exact paired sign-flip p values (all 2^n sign patterns) for the claim-bearing
    comparisons: baseline and control against perturbed in every main-experiment cell,
    policy against no safeguard, and policy against placebo. The Monte-Carlo p values
    of the same comparisons are stored under fig2, mitigation and mitigation_placebo."""
    out: dict = {"fig2": {}, "mitigation": {}, "placebo": {}}
    for m in MODELS:
        for dom in ("airline", "retail"):
            tm = {c: _task_mean_wear(sorted(glob.glob(
                str(RUNS / "main_experiment" / m / c / f"{dom}_*.json"))))
                for c in ("baseline", "control", "perturbed")}
            for a in ("baseline", "control"):
                out["fig2"][f"{m}/{dom}/{a}-vs-perturbed"] = _exact(tm[a], tm["perturbed"])
    for m in MITIGATION_MODELS:
        mit = _task_mean_wear(sorted(glob.glob(str(RUNS / "policy" / m / "perturbed" / "*.json"))))
        unmit = _task_mean_wear(sorted(glob.glob(str(RUNS / "main_experiment" / m / "perturbed" / "*.json"))))
        plac = _task_mean_wear(sorted(glob.glob(str(RUNS / "placebo" / m / "perturbed" / "*.json"))))
        if mit and unmit:
            out["mitigation"][m] = _exact(mit, unmit)
        if mit and plac:
            out["placebo"][m] = _exact(mit, plac)
    return out


def collect_mitigation_task_means() -> dict:
    """Verification-policy table: task-mean WEAR per arm (rollouts with a defined WEAR; a
    task whose rollouts all wrote nothing has no mean and is dropped)."""
    out = {}
    for m in MITIGATION_MODELS:
        g = lambda *p: _task_mean_wear(sorted(glob.glob(str(RUNS.joinpath(*p) / "*.json"))))
        arms = {"policy_perturbed": g("policy", m, "perturbed"),
                "policy_baseline": g("policy", m, "baseline"),
                "placebo_perturbed": g("placebo", m, "perturbed"),
                "unmitigated_perturbed": g("main_experiment", m, "perturbed"),
                "unmitigated_baseline": g("main_experiment", m, "baseline")}
        out[m] = {k: {"n_tasks": len(v), "task_mean_wear": round(sum(v.values()) / len(v), 4)}
                  for k, v in arms.items() if v}
    return out


def _read_tools(dom: str) -> set[str]:
    """Tools annotated @is_tool(ToolType.READ) in the tau2-bench source of the domain."""
    src = (ROOT / "tau2-bench" / "src" / "tau2" / "domains" / dom / "tools.py").read_text()
    return set(re.findall(r"@is_tool\(ToolType\.READ\)\s*\n\s*def (\w+)", src))


def collect_mitigation_overhead() -> dict:
    """Descriptive cost of the verification policy: mean READ calls and
    saved simulation duration per rollout (task mean, then over tasks), policy
    perturbed arm vs the unmitigated and placebo perturbed arms. Durations come
    from separate batches and include user-simulator and service time."""
    reads = {d: _read_tools(d) for d in ("airline", "retail")}

    def arm(path: Path) -> dict | None:
        files = sorted(path.glob("*.json"))
        if not files:
            return None
        per_task = []
        for f in files:
            dom = f.stem.split("_", 1)[0]
            sims = [s for s in json.load(open(f)).get("simulations") or []
                    if s.get("task_id") == f.stem]
            if not sims:
                continue
            r = [sum(1 for c in _flat_calls(s.get("messages") or [])
                     if c["requestor"] == "assistant" and c["name"] in reads[dom]) for s in sims]
            d = [float(s["duration"]) for s in sims]
            per_task.append((sum(r) / len(r), sum(d) / len(d)))
        return {"n_tasks": len(per_task),
                "mean_read_calls": sum(x for x, _ in per_task) / len(per_task),
                "mean_duration_s": sum(y for _, y in per_task) / len(per_task)}

    out = {}
    for m in MITIGATION_MODELS:
        pol = arm(RUNS / "policy" / m / "perturbed")
        ref = {"unmitigated": arm(RUNS / "main_experiment" / m / "perturbed"),
               "placebo": arm(RUNS / "placebo" / m / "perturbed")}
        if not pol:
            continue
        out[m] = {"policy": pol, **{k: v for k, v in ref.items() if v}}
        for k, v in ref.items():
            if v:
                out[m][f"delta_vs_{k}"] = {
                    "read_calls": round(pol["mean_read_calls"] - v["mean_read_calls"], 2),
                    "duration_s": round(pol["mean_duration_s"] - v["mean_duration_s"], 1)}
    return out


def collect_flash_sampling() -> dict:
    """Flash's extension toward 10 rollouts per task (statistical-procedures appendix):
    per domain and condition, the task mean and the number of affected tasks over the
    first five rollouts per task (seeds 300-304), and the number of saved rollouts."""
    wt = load_write_tools()
    m = "openai_gemini_3_5_flash"
    out: dict = {}
    for dom in ("airline", "retail"):
        cell = {}
        for cond in ("baseline", "control", "perturbed"):
            means, affected, saved = {}, 0, 0
            for f in sorted(glob.glob(str(RUNS / "main_experiment" / m / cond / f"{dom}_*.json"))):
                res = json.load(open(f))
                seeds = {s["id"]: s.get("seed") for s in res.get("simulations") or []}
                scored = list(score_results(res, dom, wt))
                saved += len(scored)
                init = [s for s in scored if 300 <= int(seeds.get(s.sim_id, -1) or -1) <= 304]
                ws = [s.wear for s in init if s.wear is not None and not s.no_eligible_write]
                if ws:
                    means[Path(f).stem] = sum(ws) / len(ws)
                affected += int(any(s.has_wrong_bind for s in init))
            cell[cond] = {"initial_task_mean": round(sum(means.values()) / len(means), 4),
                          "initial_affected_tasks": affected, "n_tasks": len(means),
                          "saved_rollouts": saved}
        out[dom] = cell
    out["saved_total"] = sum(c[k]["saved_rollouts"] for c in (out["airline"], out["retail"])
                             for k in ("baseline", "control", "perturbed"))
    return out


# the five main-suite tasks from which the cue-present variant is built
CUE_BASE_TASKS = [
    ("airline_idreuse_chicago", "airline"),
    ("airline_idreuse_dallas", "airline"),
    ("airline_idreuse_phoenix", "airline"),
    ("retail_idreuse_espresso", "retail"),
    ("retail_idreuse_laptop", "retail"),
]


def collect_cue_present() -> dict | None:
    """Cue-present variant: the stored notice-rate rows of its rollouts, the rechecked
    share on the same five tasks without the cue, and the re-read targets of the
    cue-present rollouts; None if the stored rows are absent."""
    p = RUNS / "cue_present" / "notice.json"
    if not p.exists():
        return None
    out = json.load(open(p))
    # Comparison without the cue: the same five tasks in the main experiment's perturbed
    # condition. The rechecked share is counted over write-bearing rollouts, with the
    # same timing check as in the main experiment.
    wt_data = load_write_tools()
    pidx = load_perturb_index(sorted(glob.glob(str(TASKS / "*.json"))))
    base: dict = {}
    for model in ("openai_gpt_5_5", "openai_deepseek_v4_pro",
                  "openai_gemini_3_1_pro"):
        tot = rech = 0
        for t, dom in CUE_BASE_TASKS:
            f = RUNS / "main_experiment" / model / "perturbed" / f"{t}.json"
            rows = assess_results(json.load(open(f)), pidx, set(wt_data[dom]))
            rows = [r for r in rows if r.first_write_flat_index is not None]
            tot += len(rows)
            rech += sum(1 for r in rows if r.label == "rechecked")
        base[model] = {"rechecked": rech, "write_bearing": tot,
                       "rate": round(rech / tot, 4)}
    out["recheck_nocue_baseline"] = base
    cue_batch = [
        (RUNS / "cue_present" / model / "perturbed" / f"{t}_signal.json",
         f"{t}_signal", dom)
        for model in ("openai_gpt_5_5", "openai_deepseek_v4_pro",
                      "openai_gemini_3_1_pro")
        for t, dom in CUE_BASE_TASKS]
    out["recheck_read_targets"] = _reread_target_split(
        cue_batch, pidx, wt_data, require_write=True)
    return out


def _score_cell_from_files(files: list[str], domain: str, wt_data: dict) -> dict:
    """Rescore one cell from raw run files (per-task sim mean, then mean over tasks)."""
    per_task: dict[str, list[tuple[float, float]]] = {}
    n_sims = 0
    for f in files:
        t = Path(f).stem
        for s in score_results(json.load(open(f)), domain, wt_data):
            n_sims += 1
            if s.wear is None or s.no_eligible_write:  # WEAR undefined, excluded
                continue
            per_task.setdefault(t, []).append(
                (s.wear, s.effective_wear if s.effective_wear is not None else 0.0))
    att = [sum(x[0] for x in v) / len(v) for v in per_task.values()]
    eff = [sum(x[1] for x in v) / len(v) for v in per_task.values()]
    return {"n_tasks": len(per_task), "n_sims": n_sims,
            "wear_mean": round(sum(att) / len(att), 4),
            "eff_wear_mean": round(sum(eff) / len(eff), 4)}


def _variant_script(script: str, task_files: list[str]) -> dict:
    """Run a variant analysis script (notice_rate.py or m2_recovery.py) on the run files
    under runs/nodistractor_singlechar for the given tasks and return its JSON output."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        subprocess.run([sys.executable, str(WEAR / script), str(RUNS / "nodistractor_singlechar"),
                        "--task-files", *task_files, "--out", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        return json.load(open(out))


def collect_nodistractor_singlechar() -> dict | None:
    """No-distractor (tasks *_distC_none) and single-character (*_simnew) variants of
    ID-REUSE, computed from the run files (stored summaries in runs/nodistractor_singlechar are not read,
    except notice_distinct.json).

    Returns the notice rate of both variants (notice_mech), the no-distractor outcomes
    (m2_recovery: first write through the old identifier, recovery to the new one), WEAR
    of the no-distractor variant (score_m2), how many no-distractor trajectories write
    through the old identifier at least twice (score_m2_retries), WEAR of the
    single-character variant beside the same tasks in the main experiment (score_m1),
    and the stored notice-rate rows of the five base tasks in the main experiment with
    their re-read targets (notice_distinct). None if an input is absent.
    """
    p_dist = RUNS / "nodistractor_singlechar" / "notice_distinct.json"
    nd_files = sorted(glob.glob(str(TASKS / "airline_idreuse_*_distC_none.json")))
    sn_files = sorted(glob.glob(str(TASKS / "airline_idreuse_*_simnew.json")))
    if not (p_dist.exists() and nd_files and sn_files):
        return None
    notice_mech = _variant_script("notice_rate.py", nd_files + sn_files)
    m2_recovery = _variant_script("m2_recovery.py", nd_files)
    score_m2 = fig2_build_report(str(RUNS / "nodistractor_singlechar"), str(TASKS),
                                 keep=lambda t: t.endswith("_distC_none"))
    # no-distractor: how many write-bearing trajectories write through the old
    # identifier at least twice (repeated attempts after the rejection)
    m2_retries: dict = {}
    for model in ("openai_gpt_5_5", "openai_gemini_3_1_pro"):
        n = k = 0
        for f in sorted(glob.glob(str(RUNS / "nodistractor_singlechar" / model / "perturbed"
                                      / "*_distC_none.json"))):
            for sw in score_results(json.load(open(f)), "airline", load_write_tools()):
                if sw.n_write_calls == 0:
                    continue
                n += 1
                k += int(sw.n_wrong_entity >= 2)
        m2_retries[model] = {"write_bearing": n, "old_id_writes_ge2": k}
    wt_data = load_write_tools()
    score_m1: dict = {}
    m1_tasks: list[str] = []
    for model in ("openai_gpt_5_5", "openai_gemini_3_1_pro"):
        files = sorted(glob.glob(
            str(RUNS / "nodistractor_singlechar" / model / "perturbed" / "*_simnew.json")))
        if files:
            score_m1[model] = _score_cell_from_files(files, "airline", wt_data)
            m1_tasks = [Path(f).stem.replace("_simnew", "") for f in files]
    for model in ("openai_gpt_5_5", "openai_gemini_3_1_pro"):
        files = [str(RUNS / "main_experiment" / model / "perturbed" / f"{t}.json")
                 for t in m1_tasks]
        files = [f for f in files if Path(f).exists()]
        if files:
            score_m1[f"{model}@fig2subset"] = _score_cell_from_files(
                files, "airline", wt_data)
    nd = json.load(open(p_dist))
    pidx = load_perturb_index(sorted(glob.glob(str(TASKS / "*.json"))))
    dist_batch = [
        (RUNS / "main_experiment" / model / "perturbed" / f"{t}.json", t, "airline")
        for model in MODELS for t in sorted({r["task"] for r in nd["rows"]})]
    nd["reread_targets"] = _reread_target_split(
        dist_batch, pidx, wt_data, require_write=False)
    return {
        "notice_mech": notice_mech,
        "m2_recovery": m2_recovery,
        "notice_distinct": nd,
        "score_m2": score_m2,
        "score_m2_retries": m2_retries,
        "score_m1": score_m1,
    }


# ---------------------------------------------------------------- safeguard study (safeguard_study)
# per-cell counters summed over all cells into safeguard_study.totals
SG_TOTAL_KEYS = ("units", "rollouts", "unexposed", "boundary_unresolved",
                 "writes_unresolved")


# rollout-level outcome counters of _v2_rollout_outcomes
SG_OUTCOME_KEYS = ("exposed", "completed", "stale", "completed_and_stale", "neither", "rejected",
                    "relisted_completed", "relisted_stale", "relisted_neither",
                    "not_relisted_completed", "not_relisted_stale", "not_relisted_neither")


def _v2_rollout_outcomes(av, root_map: dict) -> dict:
    """Rollout outcomes in the perturbed condition of the safeguard study (descriptive),
    with the same per-rollout measures as analyze_safeguards.

    Per safeguard/suite/model: exposed rollouts (the change was applied); those that
    completed (a non-error write to the gold target); those with at least one stale write
    that returned no error; both; neither. For rollouts in which a guard rejected a write:
    whether the agent re-listed the user's records (a get_user_details call) after its
    first write that returned an error, by outcome (stale, else completed, else neither).
    These counts feed the guard-flow figure.
    """
    out: dict = {}
    for arm, suites in root_map.items():
        for suite, root in suites.items():
            for path in av._unit_files(root):
                model, cond = path.split(os.sep)[-3], path.split(os.sep)[-2]
                if cond != "perturbed":
                    continue
                domain = os.path.basename(path).split("_", 1)[0]
                src = av.source_task(path)
                d = json.load(open(path))
                tmap = {t["id"]: t for t in d.get("tasks") or []}
                c = out.setdefault(f"{arm}/{suite}/{model}", Counter())
                for sim in d.get("simulations") or []:
                    m = av.rollout_measures(sim, tmap[sim["task_id"]], src, domain, cond)
                    if not m["exposed"]:
                        continue
                    done, stale = bool(m["completion"]), m["stale_writes"] > 0
                    kind = "stale" if stale else "completed" if done else "neither"
                    c["exposed"] += 1
                    c["completed"] += done
                    c["stale"] += stale
                    c["completed_and_stale"] += done and stale
                    c["neither"] += not (done or stale)
                    if m["rejections"]:
                        calls = av.tool_calls(sim)
                        first = next((i for i, cl in enumerate(calls)
                                      if cl["name"] in av.WT[domain] and cl["response"] == "error"), None)
                        relist = first is not None and any(
                            cl["name"] == "get_user_details" for cl in calls[first + 1:])
                        c["rejected"] += 1
                        c[("relisted_" if relist else "not_relisted_") + kind] += 1
    return {k: {x: int(v.get(x, 0)) for x in SG_OUTCOME_KEYS} for k, v in sorted(out.items())}


def collect_safeguard_study() -> dict | None:
    """Safeguard study: no safeguard, the verification policy, the compare-and-set (CAS)
    guard and the identity guard on the main and held-out suites, for GPT-5.5, DeepSeek
    and Qwen. Cells, paired tests and the false-rejection replay of baseline tool calls
    through both guards come from analyze_safeguards.analyze on runs/safeguard_study; the replay is
    also run on the main experiment's baseline rollouts (h3c_paper_baselines). Adds the
    totals and the rollout outcomes; None if the run directory is absent."""
    root = RUNS / "safeguard_study"
    if not root.is_dir():
        return None
    import analyze_safeguards as av
    root_map = {arm: {s: str(root / arm / s) for s in av.SUITE_DIRS
                      if (root / arm / s).is_dir()} for arm in av.ARMS}
    res = av.analyze(root_map)
    res["h3c_paper_baselines"] = av.replay_false_rejections(
        {"none": {"paper": str(RUNS / "main_experiment")}})
    res["totals"] = {k: sum(c.get(k, 0) for c in res["cells"].values()) for k in SG_TOTAL_KEYS}
    res["rollout_outcomes"] = _v2_rollout_outcomes(av, root_map)
    return res


# ---------------------------------------------------------------- output
def main() -> None:
    """Run every collector and write NUMBERS.json; optional parts whose runs are absent
    are left out."""
    recover = collect_recover()
    num = {
        "fig2": collect_fig2(),
        "fig3": collect_fig3(),
        "judge_prompts": collect_judge_prompts(),
        "sixgate_judge": collect_sixgate_judge(recover),
        "fig4_ts": collect_fig4_ts(),
        "recover": recover,
        "stats_params": {  # must match the defaults in wear/stats.py
            "two_stage_bootstrap": {"n_boot": 10000, "seed": 0},
            "paired_sign_flip": {"n_perm": 10000, "seed": 0, "smoothing": "add-one"},
            "wilson_level": 0.95,
            "exact_two_sided_floor_10_saturated_pairs": 2 / 2 ** 10,
        },
    }
    pilot = collect_pilot_signal()
    if pilot:
        num["pilot_signal_exploratory"] = pilot
    cj = collect_crossjudge()
    if cj:
        num["sixgate_gemini_judge"] = cj
    mit = collect_mitigation()
    if mit:
        num["mitigation"] = mit
    mit_placebo = collect_mitigation_placebo()
    if mit_placebo:
        num["mitigation_placebo"] = mit_placebo
    cue_present = collect_cue_present()
    if cue_present:
        num["cue_present"] = cue_present
    nodistractor_singlechar = collect_nodistractor_singlechar()
    if nodistractor_singlechar:
        num["nodistractor_singlechar"] = nodistractor_singlechar
    num["reread_timing"] = collect_reread_timing()
    num["binding_extended"] = collect_binding_extended()
    num["recovery_authorization"] = collect_recovery_authorization()
    num["exact_p"] = collect_exact_p()
    num["mitigation_overhead"] = collect_mitigation_overhead()
    num["mitigation_task_means"] = collect_mitigation_task_means()
    num["flash_sampling"] = collect_flash_sampling()
    mit_sg = collect_safeguard_study()
    if mit_sg:
        num["safeguard_study"] = mit_sg
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(num, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
