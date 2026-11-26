"""Tests that keep the manuscript, generated tables and figures consistent with NUMBERS.json.

Three kinds of check: the generated tables in tables/ match what make_figures.py
renders from NUMBERS.json now; load-bearing values in NUMBERS.json (and in the
state-check and control-comparability results) have the values the paper reports;
and the sentences of the manuscript that quote them contain exactly these numbers
(_assert_in_tex).

Usage (repo root): tau2-bench/.venv/bin/python -m pytest wear/paper/test_paper_numbers.py
Reads, never writes: wear/paper/NUMBERS.json, tables/*.tex, journal/wear.tex and
journal/wear_supp.tex, and the state-check and control-comparability results under
wear/audit/. When the manuscript files are absent, the text checks are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_figures as mf  # noqa: E402

NUM = json.load(open(HERE / "NUMBERS.json"))
# The paper and its supplementary material are checked as one manuscript: a pinned
# sentence may sit in either file, so moving text between them does not break a pin.
_TEX_PATHS = [HERE / "journal" / "wear.tex", HERE / "journal" / "wear_supp.tex"]
# manuscripts hard-wrap at ~80 cols: assert substrings on a whitespace-normalized view
_present = [p for p in _TEX_PATHS if p.exists()]
TEXNS = ({"journal/wear.tex+wear_supp.tex": " ".join(" ".join(p.read_text().split()) for p in _present)}
         if _present else {})


def _assert_in_tex(s: str) -> None:
    """Assert that s occurs in the whitespace-normalized manuscript; a no-op when the
    manuscript files are absent."""
    for name, texn in TEXNS.items():
        assert s in texn, f"{name} missing expected text: {s!r}"


# ------------------------------------------------------------------ table artifacts
def test_tab_cells_artifact_fresh():
    """tables/tab_cells.tex (the main results table) equals a fresh rendering."""
    want = "\n".join(mf._tab_cells_lines(NUM)) + "\n"
    got = (HERE / "tables" / "tab_cells.tex").read_text()
    assert got == want, "tables/tab_cells.tex diverges from NUMBERS.json - rerun make_figures.py"


def test_tab_mitigation_artifact_fresh():
    """tables/tab_mitigation.tex (the verification-policy table) equals a fresh rendering."""
    want = "\n".join(mf._tab_mitigation_lines(NUM)) + "\n"
    got = (HERE / "tables" / "tab_mitigation.tex").read_text()
    assert got == want, "tables/tab_mitigation.tex diverges from NUMBERS.json - rerun make_figures.py"


def test_main_tex_inputs_generated_tables():
    """The manuscript includes the generated tables instead of hand-written copies."""
    if not TEXNS:
        pytest.skip("manuscripts not shipped in the supplementary package")
    for name, texn in TEXNS.items():
        for t in ("tab_cells", "tab_mitigation", "tab_safeguard_study", "tab_ladder", "tab_sixgate"):
            assert f"\\input{{../tables/{t}}}" in texn, f"{name} does not \\input {t}"
        assert "& 0.0 / 0.0 &" not in texn, f"{name} contains a hand-written tab:cells value row"


# ------------------------------------------------------------------ load-bearing numbers
def _f2(m, cond, dom):
    """Main-experiment cell of model m, condition cond and domain dom."""
    return NUM["fig2"]["cells"][f"{m}/{cond}/{dom}"]


def test_rollout_total_and_p():
    """Main experiment: 1,798 rollouts, and exact p = 1/512 for all 20 comparisons of
    baseline or control against perturbed."""
    assert NUM["fig2"]["n_sims_total"] == 1798
    _assert_in_tex("1{,}798 rollouts")
    pp = NUM["fig2"]["paired_p"]
    # pin the count first: all() over an empty dict is vacuously true
    assert len(pp) == 20
    assert all(v == 0.0025 for v in pp.values())
    ex = NUM["exact_p"]["fig2"]
    assert set(ex) == set(pp)  # same 20 claim-bearing comparisons
    assert all((v["p_num"], v["p_den"], v["n_pairs"]) == (1, 512, 10) for v in ex.values())
    _assert_in_tex("$p = 1/512 \\approx 0.002$")
    _assert_in_tex("$p = 2/2^{10} = 1/512 \\approx 0.00195$")


def test_perturbed_range_and_flash_n99():
    """Perturbed WEAR range, the Wilson interval on affected tasks, and Flash's 99-rollout cells."""
    vals = [_f2(m, "perturbed", d)["wear_mean"]
            for m in mf.MODELS for d in ("airline", "retail")]
    assert mf._num3(min(vals)) == "0.907" and max(vals) == 1.0
    _assert_in_tex("0.907--1.0")
    for m in mf.MODELS:
        for d in ("airline", "retail"):
            assert _f2(m, "perturbed", d)["wilson95_taskwrong"] == [0.7225, 1.0]
    _assert_in_tex("fraction of affected tasks: $[0.72, 1.0]$)")
    n99 = [(m, d) for m in mf.MODELS for d in ("airline", "retail")
           if _f2(m, "perturbed", d)["n_sims"] == 99]
    assert len(n99) == 2 and all(m == "openai_gemini_3_5_flash" for m, _ in n99)
    _assert_in_tex("Each perturbed Flash cell therefore has 99 rollouts")


def test_ts_run_counts_13_14_and_15_15():
    """ToolSandbox dynamic ID-REUSE scenarios: wrong-write runs per model (13/14, 15/15, 15/15)."""
    ts = NUM["fig4_ts"]
    for model, want in (("gpt-5.5", (13, 14)), ("gemini-3.1-pro", (15, 15)),
                        ("gemini-3.5-flash", (15, 15))):
        wrong = n = 0
        for scen, v in ts[model].items():
            if scen == "_groups" or v["group"] != "idreuse":
                continue
            n += v["n_runs_scored"]
            wrong += round(v["attempted_mean"] * v["n_runs_scored"])
        assert (wrong, n) == want, (model, wrong, n)
    _assert_in_tex("\\textbf{15/15}")
    _assert_in_tex("\\textbf{13/14}")
    # one GPT-5.5 rollout of the swapped-pair scenario crashed before it was saved
    sw = ts["gpt-5.5"]["modify_contact_idreuse_swapped_pair_binding_attack"]
    assert sw["n_runs_scored"] == 4 and sw["n_missing_conv"] == 1
    _assert_in_tex("so that scenario has 4 rollouts for GPT-5.5")


def test_mech_rereads_340_over_150():
    """Five base tasks in the main experiment: 150 rollouts, 340 reads between the binding
    read and the first write, and no notice of the new identifier."""
    rows = NUM["nodistractor_singlechar"]["notice_distinct"]["rows"]
    assert len(rows) == 150
    assert sum(len(r["recheck_reads"]) for r in rows) == 340
    assert all(not r["notice"] for r in rows)
    _assert_in_tex("340 reads between the binding read and the first write")
    _assert_in_tex("150 rollouts")


def test_mech_simnew_0_of_50():
    """Single-character variant: notice rate 0/50 (GPT-5.5 and Gemini Pro)."""
    # the _simnew tasks of the variant runs
    rows = [r for r in NUM["nodistractor_singlechar"]["notice_mech"]["rows"]
            if "_simnew" in r["task"]]
    assert len(rows) == 50 and len({r["model"] for r in rows}) == 2
    assert sum(r["notice"] for r in rows) == 0
    assert {r["model"] for r in rows} == {"openai_gpt_5_5", "openai_gemini_3_1_pro"}
    _assert_in_tex("ran with GPT-5.5 and Gemini Pro on the same five tasks (50 trajectories)")
    _assert_in_tex("is 0 (0/50)")


def test_m2_recovery_loud_reject_and_stall():
    """No-distractor variant: every first write goes through the old identifier and is
    rejected; share that never completes, WEAR, and repeated old-identifier writes."""
    agg = NUM["nodistractor_singlechar"]["m2_recovery"]["agg"]
    fo = sum(v["first_old"] for v in agg.values())
    eo = sum(v["eff_old"] for v in agg.values())
    assert (fo, eo) == (50, 0)
    _assert_in_tex("50/50 write-bearing trajectories")
    _assert_in_tex("all 50 trajectories contain a write, and in all 50 the first write goes "
                   "through the old identifier and is rejected")
    by: dict = {}
    for k, v in agg.items():
        d = by.setdefault(k.split("/")[0], {"fo": 0, "rc": 0})
        d["fo"] += v["first_old"]; d["rc"] += v["recov"]
    stall = sorted(round(100 * (1 - d["rc"] / d["fo"])) for d in by.values())
    assert stall == [12, 40]
    _assert_in_tex("12--40\\%")
    _assert_in_tex("only 60\\% of GPT-5.5's 25 trajectories and 88\\% of Gemini Pro's 25")
    _assert_in_tex("12--40\\% of these trajectories then never complete the task")
    assert {k: d["fo"] for k, d in by.items()} == {"openai_gpt_5_5": 25, "openai_gemini_3_1_pro": 25}
    cells = NUM["nodistractor_singlechar"]["score_m2"]["cells"]
    assert cells["openai_gpt_5_5/perturbed/airline"]["wear_mean"] == 0.7
    assert cells["openai_gemini_3_1_pro/perturbed/airline"]["wear_mean"] == 0.7067
    _assert_in_tex("On $\\tau^2$ it is 0.70 for GPT-5.5 and\n0.71 for Gemini Pro".replace("\n", " "))
    _assert_in_tex("0.70--0.71 on $\\tau^2$")
    rt = NUM["nodistractor_singlechar"]["score_m2_retries"]["openai_gemini_3_1_pro"]
    assert (rt["old_id_writes_ge2"], rt["write_bearing"]) == (22, 25)
    _assert_in_tex("22 of its 25 trajectories write through it at least twice")


def test_cue_owner_relist_12_of_75():
    """Cue-present variant: the new identifier is seen in 12 of 75 trajectories, and
    exactly those 12 write to the target record."""
    rows = [r for r in NUM["cue_present"]["rows"] if r["has_write"]]
    assert len(rows) == 75
    exposed = [r for r in rows if r["exposed"]]
    assert len(exposed) == 12 and all(not r["wrong_first"] for r in exposed)
    _assert_in_tex("12/75")
    assert all(r["wrong_first"] for r in rows if not r["exposed"])  # 63/63
    assert sum(r["wrong_first"] for r in rows) == 63
    # the paper quotes the same 12 and 63 in several places
    _assert_in_tex("a read returned the new identifier in 12 of the 75 trajectories, and all 12 "
                   "wrote to the target record")
    assert sum(r["notice"] for r in rows) == 12 and all(r["notice"] == r["exposed"] for r in rows)
    _assert_in_tex("is also 12/75 (0.16)")
    _assert_in_tex("still writes to the wrong record in 63 of 75 trajectories")
    _assert_in_tex("In 63 of the 75 trajectories (0.84) the agent still writes to the wrong record")
    _assert_in_tex("against 63/75 on $\\tau^2$")


def test_tab_diag_numbers():
    """APPEND/REORDER table: zero wrong writes, verified exposure, and the task counts."""
    f3 = NUM["fig3"]
    gpt = f3["openai_gpt_5_5/perturbed"]
    fla = f3["openai_gemini_3_5_flash/perturbed"]
    for d in (gpt, fla):
        assert d["append"]["n_sims_wrong"] == 0 == d["reorder"]["n_sims_wrong"]
        assert d["append"]["n_sims_with_writes"] == 49
        assert d["reorder"]["n_sims_with_writes"] == 45
        assert d["append"]["n_tasks"] == 10 and d["reorder"]["n_tasks"] == 9
    assert gpt["append"]["decoy_seen"] == "49/49"
    assert fla["append"]["decoy_seen"] == "28/49"
    _assert_in_tex("APPEND, same-as, all write-bearing & 0/49 & 0/49 & 49/49 & 28/49")
    _assert_in_tex("APPEND, same-as, saw the decoy & \\textbf{0/49} & 0/28 & --- & ---")
    # a claim-bearing zero needs >= 10 tasks: GPT-5.5's exposed rollouts cover 10, Flash's 9
    assert (gpt["append"]["decoy_seen_tasks"], fla["append"]["decoy_seen_tasks"]) == ("10/10", "9/10")
    _assert_in_tex("spread over 9 of the 10 tasks")
    _assert_in_tex("Flash saw the decoy in 28 of its 49, spread over 9 of the 10 tasks")
    _assert_in_tex("0/45 & 0/45 & ---")
    _assert_in_tex("10 APPEND tasks, 9 REORDER tasks")
    # the comparison with Flash's ID-REUSE WEAR from the main results table
    a = mf._num3(_f2("openai_gemini_3_5_flash", "perturbed", "airline")["wear_mean"])
    r = mf._num3(_f2("openai_gemini_3_5_flash", "perturbed", "retail")["wear_mean"])
    _assert_in_tex(f"WEAR of 1.0 for GPT-5.5 and {a}--{r} for Flash")
    assert all("\\textbf{0/28}" not in t for t in TEXNS.values())


def test_mitigation_caption_numbers():
    """Verification policy: table note values, task reward, and exact p values against no
    safeguard and against the placebo."""
    mit = NUM["mitigation"]
    gpt, ds = mit["openai_gpt_5_5"], mit["openai_deepseek_v4_pro"]
    assert gpt["perturbed"]["n_sims_zero_write"] == 4
    assert ds["perturbed"]["n_sims_zero_write"] == 2
    _assert_in_tex("split into 92 correct, 2 without a write")
    for m, lo, hi in ((gpt, "0.01", "0.24"), (ds, "0.08", "0.42")):
        w = m["perturbed"]["wilson95_taskwrong"]
        assert f"{w[0]:.2f}" == lo and f"{w[1]:.2f}" == hi
    _assert_in_tex("[0.01, 0.24] for GPT-5.5 and [0.08, 0.42] for DeepSeek")
    assert (gpt["unmitigated_baseline"]["reward_mean"],
            gpt["baseline"]["reward_mean"]) == (0.99, 0.98)
    assert (ds["unmitigated_baseline"]["reward_mean"],
            ds["baseline"]["reward_mean"]) == (0.99, 0.99)
    _assert_in_tex("task reward is 0.99 without the policy and 0.98 with it for GPT-5.5, "
                   "and 0.99 in both cases for DeepSeek")
    for m in (gpt, ds):
        assert m["paired_vs_unmitigated_perturbed"]["p_two_sided"] < 1.01e-4
    em, ep = NUM["exact_p"]["mitigation"], NUM["exact_p"]["placebo"]
    assert [(em[m]["p_num"], em[m]["p_den"]) for m in ("openai_gpt_5_5", "openai_deepseek_v4_pro")] \
        == [(1, 524288), (1, 524288)]
    _assert_in_tex("$p = 1/524288 \\approx 2 \\times 10^{-6}$ for each model")
    assert [(ep[m]["p_num"], ep[m]["p_den"]) for m in ("openai_gpt_5_5", "openai_deepseek_v4_pro")] \
        == [(1, 524288), (1, 524288)]
    _assert_in_tex("$p = 1/524288$ for each model")
    red = [round(m["paired_vs_unmitigated_perturbed"]["mean_diff_unmit_minus_mit"], 2)
           for m in (gpt, ds)]
    assert red == [0.99, 0.94]


def test_mitigation_placebo_saturation():
    """Placebo: nearly every write-bearing rollout still contains a wrong write."""
    mp = NUM["mitigation_placebo"]
    gpt, ds = mp["openai_gpt_5_5"], mp["openai_deepseek_v4_pro"]
    assert (gpt["perturbed"]["n_sims_wrong"],
            gpt["perturbed"]["n_sims_with_writes"]) == (100, 100)
    assert (ds["perturbed"]["n_sims_wrong"],
            ds["perturbed"]["n_sims_with_writes"]) == (99, 100)
    assert f'{gpt["perturbed"]["task_mean_wear"]:.2f}' == "0.99"
    assert f'{ds["perturbed"]["task_mean_wear"]:.2f}' == "0.94"
    assert gpt["perturbed"]["tasks_with_wrong_bind"] == "20/20"
    assert ds["perturbed"]["tasks_with_wrong_bind"] == "20/20"
    for m in (gpt, ds):
        assert m["paired_vs_mitigation_perturbed"]["p_two_sided"] < 1.01e-4
    _assert_in_tex("task-mean WEAR is 0.99 for GPT-5.5 and 0.94 for DeepSeek")
    _assert_in_tex("100/100 write-bearing rollouts of GPT-5.5 and 99/100 of DeepSeek contained at "
                   "least one wrong write")

def test_reread_and_recheck_target_provenance():
    """What the reads before the first write query, without the cue (base tasks) and with it."""
    m = NUM["nodistractor_singlechar"]["notice_distinct"]["reread_targets"]
    assert (m["stale_ctx"], m["owner_relist"], m["fresh"], m["misc"],
            m["total"]) == (339, 0, 0, 1, 340)
    assert m["stale_ctx_pre_seen"] == 339
    c = NUM["cue_present"]["recheck_read_targets"]
    assert (c["stale_ctx"], c["owner_relist"], c["fresh"], c["misc"],
            c["total"]) == (137, 12, 12, 1, 162)
    assert c["stale_ctx_pre_seen"] == 137
    _assert_in_tex("Of these, 339 query identifiers seen before the change")
    _assert_in_tex("137 of the 162 reads that count toward the rechecked label query identifiers "
                   "that the agent had seen before the change")


def test_cue_recheck_nocue_baseline():
    """Cue-present variant: share of rechecked trajectories with and without the cue."""
    b = NUM["cue_present"]["recheck_nocue_baseline"]
    assert b["openai_gpt_5_5"]["rechecked"] == 0
    assert b["openai_gpt_5_5"]["write_bearing"] == 25
    assert round(b["openai_deepseek_v4_pro"]["rate"] * 100) == 76
    assert round(b["openai_gemini_3_1_pro"]["rate"] * 100) == 76
    cue = {r["model"]: 0 for r in NUM["cue_present"]["rows"]}
    for r in NUM["cue_present"]["rows"]:
        cue[r["model"]] += 1 if r["label"] == "rechecked" else 0
    assert cue["openai_gpt_5_5"] == 25 and cue["openai_deepseek_v4_pro"] == 21
    assert cue["openai_gemini_3_1_pro"] == 21
    _assert_in_tex("from 0\\% to 100\\% for GPT-5.5, and from 76\\% to 84\\% for DeepSeek and for "
                   "Gemini Pro")


def test_reread_timing_provenance():
    """Timing of the reads between the binding read and the first write: the re-read
    table in the supplementary material and the sentences that quote it."""
    t = NUM["reread_timing"]
    names = {"openai_gpt_5_5": "GPT-5.5", "openai_gemini_3_1_pro": "Gemini Pro",
             "openai_gemini_3_5_flash": "Flash",
             "openai_deepseek_v4_pro": "DeepSeek", "openai_qwen3_7_plus": "Qwen"}
    for m, label in names.items():
        r = t[m]
        # rechecked = any read in the interval; not rechecked = the rest
        _assert_in_tex(
            f"{label} & {r['write_bearing']} & {r['any_reread']} & "
            f"{r['write_bearing'] - r['any_reread']} & {r['reread_after_user']} & "
            f"{r['reads_same_msg']} & {r['reads_later_no_user']} & {r['reads_after_user']} & "
            f"{r['notfound']} & {r['notfound_wrote_stale']} \\\\")
    tot = {k: sum(r[k] for r in t.values()) for k in (
        "write_bearing", "any_reread", "reread_after_user", "reads_same_msg",
        "reads_later_no_user", "reads_after_user", "notfound", "notfound_wrote_stale")}
    _assert_in_tex(
        f"all & {tot['write_bearing']} & {tot['any_reread']} & "
        f"{tot['write_bearing'] - tot['any_reread']} & {tot['reread_after_user']} & "
        f"{tot['reads_same_msg']} & {tot['reads_later_no_user']} & {tot['reads_after_user']} & "
        f"{tot['notfound']} & {tot['notfound_wrote_stale']} \\\\")
    wb = sum(r["write_bearing"] for r in t.values())
    after = sum(r["reread_after_user"] for r in t.values())
    nf = sum(r["notfound"] for r in t.values())
    assert all(r["notfound"] == r["notfound_wrote_stale"] for r in t.values())
    g, p = t["openai_gpt_5_5"], t["openai_gemini_3_1_pro"]
    gem_reads = p["reads_same_msg"] + p["reads_later_no_user"] + p["reads_after_user"]
    _assert_in_tex(f"Of the {wb} write-bearing perturbed trajectories, only {after} contain a read")
    _assert_in_tex(f"in {g['write_bearing'] - g['any_reread']} of {g['write_bearing']} trajectories")
    _assert_in_tex(f"labeled rechecked in {p['any_reread']} of {p['write_bearing']}, but "
                   f"{p['reads_same_msg']} of its {gem_reads} such reads")
    _assert_in_tex(f"In {nf} of the {wb} write-bearing perturbed trajectories, a lookup of another "
                   "of the user's records")
    _assert_in_tex(f"All {nf} still wrote through the old identifier")
    f = t["openai_gemini_3_5_flash"]
    _assert_in_tex(f"appears in {p['notfound']} of Gemini Pro's {p['write_bearing']} trajectories, "
                   f"in {f['notfound']} of Flash's {f['write_bearing']}, and in none of GPT-5.5's")
    assert g["notfound"] == 0
    _assert_in_tex(f"only {after} contain a read that came after the binding read and a further user turn")


def test_batch2_extended_scoring_and_reporting():
    """Eligible-denominator WEAR and extended scoring (the extended-scoring table in the
    supplementary material), authorized repairs, the cost of the verification policy,
    and Flash's first five rollouts per task."""
    be = NUM["binding_extended"]
    # One DeepSeek trajectory has only unmatched (rejected) writes: its eligible-denominator
    # WEAR is undefined, while the simpler rule scores it 0.
    assert not be["task_means_unchanged"]
    (dv,) = be["diverging_trajectories"]
    assert (dv["cell"], dv["task_id"], dv["seed"], dv["submitted_wear"], dv["eligible_wear"]) \
        == ("openai_deepseek_v4_pro/airline/perturbed", "airline_idreuse_denver", 304, 0.0, None)
    changed = {k: (c["task_mean_wear"], c["task_mean_wear_submitted"])
               for k, c in be["cells"].items() if "task_mean_wear_submitted" in c}
    assert changed == {"openai_deepseek_v4_pro/airline/perturbed": (1.0, 0.975)}
    _assert_in_tex("is therefore 1.0 under the eligible denominator, which all our tables and "
                   "figures use, and 0.975 under the simpler rule")
    # the tables and figures use the eligible denominator of the WEAR definition
    c = NUM["fig2"]["cells"]["openai_deepseek_v4_pro/perturbed/airline"]
    assert (c["wear_mean"], c["eff_wear_mean"]) == (1.0, 1.0)
    assert "DeepSeek & 1.000 [1.00, 1.00] & 1.000 [1.00, 1.00] & 0.000 / 0.000 & 0.000 / 0.000" in (HERE / "tables" / "tab_cells.tex").read_text()
    for cond, row in be["totals"].items():
        _assert_in_tex(f"{cond} & {row['n_traj']} & {row['no_write']} & {row['n_writes']} & "
                       f"{row['n_eligible']} & {row['n_wrong']} & {row['n_unmatched']} \\\\")
    t = be["totals"]
    assert t["all"]["n_traj"] == 1798 and t["all"]["n_unmatched"] == 4
    assert t["perturbed"]["n_eligible"] == t["perturbed"]["n_writes"] - 2
    assert t["perturbed"]["n_unmatched"] == 2 and t["baseline"]["n_unmatched"] == 2
    assert t["perturbed"]["n_wrong_nonerror"] == t["perturbed"]["n_wrong"] == 587
    _assert_in_tex("four writes in the 1{,}798 trajectories are unmatched")
    ra = NUM["recovery_authorization"]
    assert (ra["n_traj"], ra["n_writes"], ra["wrong_labels_static"],
            ra["wrong_labels_annotated"]) == (13, 39, 26, 13)
    assert ra["first_write_still_wrong"]
    _assert_in_tex("Their 39 writes then carry 13 wrong labels instead of 26")
    tm = NUM["mitigation_task_means"]
    g, d = tm["openai_gpt_5_5"], tm["openai_deepseek_v4_pro"]
    f = lambda x: f"{x:.2f}"
    _assert_in_tex(
        f"task-mean WEAR falls from {f(g['unmitigated_perturbed']['task_mean_wear'])} to "
        f"{f(g['policy_perturbed']['task_mean_wear'])} for GPT-5.5 and from "
        f"{f(d['unmitigated_perturbed']['task_mean_wear'])} to "
        f"{f(d['policy_perturbed']['task_mean_wear'])} for DeepSeek")
    oh = NUM["mitigation_overhead"]
    og, od = oh["openai_gpt_5_5"]["delta_vs_unmitigated"], oh["openai_deepseek_v4_pro"]["delta_vs_unmitigated"]
    _assert_in_tex(f"In the perturbed condition, GPT-5.5 makes {og['read_calls']:.2f} more read "
                   f"calls per rollout with the policy than without it, and DeepSeek "
                   f"{od['read_calls']:.2f} more")
    _assert_in_tex(f"is {og['duration_s']:.1f} seconds longer for GPT-5.5 and "
                   f"{od['duration_s']:.1f} seconds longer for DeepSeek")
    fs = NUM["flash_sampling"]
    _assert_in_tex(f"task means of {fs['airline']['perturbed']['initial_task_mean']:.2f} (airline) and "
                   f"{fs['retail']['perturbed']['initial_task_mean']:.2f} (retail)")
    assert all(fs[dm]["perturbed"]["initial_affected_tasks"] == 10
               and fs[dm]["baseline"]["initial_affected_tasks"] == 0
               and fs[dm]["control"]["initial_affected_tasks"] == 0 for dm in ("airline", "retail"))
    _assert_in_tex(f"{fs['saved_total']} of the 600 targeted rollouts were saved")


def test_control_comparability_journal():
    """Control and perturbed rollouts are about equally long (user turns, tool calls,
    tool errors); rollouts without a write per condition (12, 15, 13)."""
    texn = next(iter(TEXNS.values()), None)
    if texn is None:
        pytest.skip("journal manuscript not present")
    path = HERE.parent / "audit" / "control_comparability_20261124" / "results.json"
    cc = json.load(open(path))
    c, p = cc["control"], cc["perturbed"]
    assert (cc["baseline"]["no_write"], c["no_write"], p["no_write"]) == (12, 15, 13)
    for s in (f"a control rollout has {c['mean_user_turns']:.2f} user turns and "
              f"{c['mean_tool_calls']:.2f} tool calls, and a perturbed rollout has "
              f"{p['mean_user_turns']:.2f} user turns and {p['mean_tool_calls']:.2f} tool calls",
              f"return an error after the change: {p['mean_tool_errors']:.2f} per rollout, "
              f"against {c['mean_tool_errors']:.3f} in the control"):
        assert s in texn, s


# ------------------------------------------------------------------ safeguard study
# Values come from wear/analyze_safeguards.py through collect_numbers.collect_safeguard_study.
SG = NUM["safeguard_study"]
SG_MODELS = ("openai_gpt_5_5", "openai_deepseek_v4_pro", "openai_qwen3_7_plus")
SG_SUITES = {"main": 20, "heldout": 12}


def _sg(arm, suite, m, cond="perturbed"):
    """Safeguard-study cell of safeguard arm, suite, model m and condition cond."""
    return SG["cells"][f"{arm}/{suite}/{m}/{cond}"]


def test_tab_excluded_artifact_fresh_and_totals():
    """tables/tab_excluded.tex equals a fresh rendering; 35 perturbed rollouts lack a binding read."""
    want = "\n".join(mf._tab_excluded_lines(NUM)) + "\n"
    assert (HERE / "tables" / "tab_excluded.tex").read_text() == want
    cells = NUM["safeguard_study"]["cells"]
    per = {k: v["unexposed"] for k, v in cells.items() if k.endswith("/perturbed")}
    assert sum(per.values()) == 35
    assert sum(v for k, v in per.items() if "qwen" in k) == 26
    _assert_in_tex("In 35 of the 1{,}920 perturbed rollouts the")


def test_tab_safeguard_study_artifact_fresh():
    """tables/tab_safeguard_study.tex (the safeguard table) equals a fresh rendering."""
    want = "\n".join(mf._tab_safeguard_study_lines(NUM)) + "\n"
    got = (HERE / "tables" / "tab_safeguard_study.tex").read_text()
    assert got == want, "tables/tab_safeguard_study.tex diverges from NUMBERS.json - rerun make_figures.py"
    # Appendix D cites this table for the tool-call ranges
    assert "\\emph{Tool calls per rollout}" in got


def test_sg_coverage():
    """Safeguard study: all cells and rollouts present, none unresolved."""
    # 3 models x 2 suites x 6 settings (none and policy in baseline and perturbed,
    # each guard in perturbed only)
    assert len(SG["cells"]) == 36
    t = SG["totals"]
    assert (t["units"], t["rollouts"]) == (576, 2880)
    assert t["boundary_unresolved"] == t["writes_unresolved"] == 0
    for k, c in SG["cells"].items():
        suite = k.split("/")[1]
        assert (c["n_tasks"], c["rollouts"]) == (SG_SUITES[suite], 5 * SG_SUITES[suite]), k


def test_sg_h0_validity():
    """Without a safeguard the failure replicates: baseline WEAR <= 0.1, perturbed >= 0.8."""
    for m in SG_MODELS:
        for s in SG_SUITES:
            assert _sg("none", s, m, "baseline")["attempted_mean"] <= 0.1, (m, s)
            assert _sg("none", s, m)["attempted_mean"] >= 0.8, (m, s)


def test_sg_h1_h2_policy():
    """Verification policy against no safeguard: exact p per suite and model, and Qwen's WEAR."""
    tests = SG["tests"]
    p = lambda s, m: tests[f"{s}/{m}"]["policy_vs_none_attempted"]  # noqa: E731
    want = {("main", "openai_gpt_5_5"): "1/524288", ("main", "openai_deepseek_v4_pro"): "1/524288",
            ("main", "openai_qwen3_7_plus"): "1/16384",
            ("heldout", "openai_gpt_5_5"): "1/2048", ("heldout", "openai_deepseek_v4_pro"): "1/2048",
            ("heldout", "openai_qwen3_7_plus"): "1/2"}
    for (s, m), pe in want.items():
        r = p(s, m)
        assert r["n_pairs"] == SG_SUITES[s] and r["p_exact"] == pe, (s, m, r)
    # a fixed threshold (p < 0.01 and held-out WEAR under the policy <= 0.25) holds
    # for GPT-5.5 and DeepSeek and fails for Qwen
    for m, transfers in (("openai_gpt_5_5", True), ("openai_deepseek_v4_pro", True),
                         ("openai_qwen3_7_plus", False)):
        r = p("heldout", m)
        assert (r["p_two_sided"] < 0.01 and r["mean_b"] <= 0.25) is transfers, m
    assert round(_sg("policy", "main", "openai_qwen3_7_plus")["attempted_mean"], 2) == 0.72
    assert round(_sg("policy", "heldout", "openai_qwen3_7_plus")["attempted_mean"], 2) == 0.94


def test_sg_h3_guards():
    """Write guards: no stale write under the identity guard; every stale write accepted by
    the CAS guard follows a re-read of the reused identifier (160 in all)."""
    cas_stale = 0
    for m in SG_MODELS:
        for s in SG_SUITES:
            ident, cas = _sg("guard_identity", s, m), _sg("guard_cas", s, m)
            # identity guard: no stale write, and none with an unresolved response
            assert ident["stale_writes"] == ident["stale_unresolved"] == 0, (m, s)
            assert ident["effective_mean"] == 0.0, (m, s)
            # CAS guard: every accepted stale write followed a re-read of the reused identifier
            assert cas["stale_writes"] == cas["passed_after_reread"] > 0, (m, s)
            assert cas["other_wrong"] == ident["other_wrong"] == 0, (m, s)
            cas_stale += cas["stale_writes"]
    assert cas_stale == 160


def test_sg_h3c_no_false_rejections():
    """Replaying baseline tool calls through either guard gives no rejection (480 + 600 rollouts)."""
    for key in ("h3c_false_rejection_replay", "h3c_paper_baselines"):
        rep = SG[key]
        assert rep and all(v["rejected"] == 0 for v in rep.values()), key
    sg = SG["h3c_false_rejection_replay"]
    assert sum(v["rollouts"] for k, v in sg.items() if k.endswith("/cas")) == 480
    paper = SG["h3c_paper_baselines"]
    assert sum(v["rollouts"] for k, v in paper.items() if k.endswith("/cas")) == 600


# ------------------------------------------------------------------ safeguard study: sentences of the paper
SGO = SG["rollout_outcomes"]
SG_ARMS = ("none", "policy", "guard_cas", "guard_identity")


def _v2o(arm, key):
    """Rollout-outcome counter key of safeguard arm, summed over suites and models."""
    return sum(SGO[f"{arm}/{s}/{m}"][key] for s in SG_SUITES for m in SG_MODELS)


def _sg_range(arm, key, cond="perturbed", fmt="{:.2f}"):
    """Range 'min--max' of a cell measure over the six model-suite combinations."""
    vals = [_sg(arm, s, m, cond)[key] for s in SG_SUITES for m in SG_MODELS]
    return f"{fmt.format(min(vals))}--{fmt.format(max(vals))}"


def test_sg_prose_design_and_validity():
    """Safeguard study design: rollout counts, excluded rollouts, and the replicated failure."""
    pert = sum(c["rollouts"] for k, c in SG["cells"].items() if k.endswith("/perturbed"))
    assert (SG["totals"]["rollouts"], pert, SG["totals"]["unexposed"]) == (2880, 1920, 35)
    qwen = sum(c["unexposed"] for k, c in SG["cells"].items() if "qwen" in k)
    assert qwen == 26
    assert sum(v["exposed"] for v in SGO.values()) == 1920 - 35
    _assert_in_tex("2{,}880 rollouts in all")
    _assert_in_tex("In 35 of the 1{,}920 perturbed rollouts")
    _assert_in_tex("(26 of them are Qwen's; \\supptab{tab:excluded})")
    base = [_sg("none", s, m, "baseline")["attempted_mean"] for s in SG_SUITES for m in SG_MODELS]
    assert round(max(base), 2) == 0.01
    assert _sg_range("none", "attempted_mean") == "0.98--1.00"
    _assert_in_tex("baseline WEAR is at most 0.01 and perturbed WEAR is 0.98--1.00 for all six "
                   "model--suite combinations")
    # attempted WEAR range under the two guards
    att = [_sg(a, s, m)["attempted_mean"] for a in ("guard_cas", "guard_identity")
           for s in SG_SUITES for m in SG_MODELS]
    assert (f"{min(att):.2f}", f"{max(att):.2f}") == ("0.50", "0.92")
    _assert_in_tex("Under the guards attempted WEAR is 0.50--0.92")


def test_sg_prose_policy():
    """Transfer of the verification policy to held-out tasks, its failure modes and its cost."""
    g, d, q = SG_MODELS
    f = lambda a, s, m: f"{_sg(a, s, m)['attempted_mean']:.2f}"  # noqa: E731
    assert (f("policy", "main", g), f("policy", "main", d)) == ("0.00", "0.06")
    assert (f("policy", "heldout", g), f("policy", "heldout", d)) == ("0.00", "0.02")
    assert (f("none", "main", q), f("policy", "main", q)) == ("0.98", "0.72")
    assert (f("none", "heldout", q), f("policy", "heldout", q)) == ("0.99", "0.94")
    _assert_in_tex("lowers WEAR to 0.00 for GPT-5.5 and 0.06 for DeepSeek ($p = 1/524288$ for each)")
    _assert_in_tex("0.00 for GPT-5.5 and 0.02 for DeepSeek ($p = 1/2048$ for each)")
    _assert_in_tex("only from 0.98 to 0.72 on the main suite ($p = 1/16384$) and from 0.99 to "
                   "0.94 on the held-out suite ($p = 1/2$")
    _assert_in_tex("lowers WEAR to at most 0.06 for GPT-5.5 and DeepSeek v4 Pro, also on held-out tasks")
    _assert_in_tex("but only to 0.72--0.94 for Qwen3.7 Plus")
    # the verification-policy table values that this paragraph compares against
    tm = NUM["mitigation_task_means"]
    assert [f"{tm[m]['policy_perturbed']['task_mean_wear']:.2f}" for m in (g, d)] == ["0.01", "0.06"]
    _assert_in_tex("close to the 0.01 and 0.06 of Table~\\ref{tab:mitigation}")
    # failure modes
    st = lambda m, k: sum(_sg("policy", s, m)[k] for s in SG_SUITES)  # noqa: E731
    assert (st(q, "stale_writes"), st(q, "passed_after_reread")) == (119, 48)
    assert (st(d, "stale_writes"), st(d, "passed_after_reread")) == (7, 7)
    assert st(g, "stale_writes") == 0
    _assert_in_tex("In 71 of Qwen's 119 stale writes under the policy")
    _assert_in_tex("In the other 48, and in all 7 of DeepSeek's")
    o = {s: SGO[f"policy/{s}/{g}"] for s in SG_SUITES}
    assert [(o[s]["neither"], o[s]["exposed"]) for s in SG_SUITES] == [(7, 99), (10, 58)]
    assert [f"{_sg('policy', s, g)['completion_mean']:.2f}" for s in SG_SUITES] == ["0.93", "0.83"]
    _assert_in_tex("7 of 99 on the main suite and 10 of 58 on the held-out suite. Its completion "
                   "is therefore 0.93 on the main suite and 0.83 on the held-out suite")
    extra = [_sg("policy", s, m, "baseline")["tool_calls_mean"]
             - _sg("none", s, m, "baseline")["tool_calls_mean"] for s in SG_SUITES for m in SG_MODELS]
    _assert_in_tex(f"adds {min(extra):.2f}--{max(extra):.2f} tool calls per rollout")


def test_sg_prose_cas():
    """CAS guard: rejections, accepted stale writes, re-list paths, and paired tests against
    the identity guard."""
    cas = lambda k: _v2o("guard_cas", k)  # noqa: E731
    assert (cas("rejected"), cas("exposed"), cas("stale")) == (468, 474, 155)
    assert all(SGO[f"guard_cas/{s}/{m}"]["stale"] > 0 for s in SG_SUITES for m in SG_MODELS)
    _assert_in_tex("The CAS guard rejects a write in 468 of its 474 rollouts")
    _assert_in_tex("155 rollouts contain an accepted stale write (160 such writes in all, with at "
                   "least one in every row)")
    assert _sg_range("guard_cas", "effective_mean") == "0.10--0.40"
    ps = [SG["tests"][f"{s}/{m}"]["identity_vs_cas_effective"]["p_exact"]
          for s in SG_SUITES for m in SG_MODELS]
    assert sorted(ps, key=lambda x: 1 / int(x.split("/")[1]))[0] == "1/32768"
    assert max(1 / int(x.split("/")[1]) for x in ps) == 1 / 16 and all(x.startswith("1/") for x in ps)
    _assert_in_tex("effective WEAR under CAS is 0.10--0.40")
    # identity vs CAS: p <= 1/32 in five rows; the sixth (held-out GPT-5.5) is 1/16. Identity WEAR
    # is 0 in every task, so every nonzero difference has one sign and p = 2/2^k with k tasks that
    # differ: 1/16 means CAS accepted a stale write in exactly 5 of the 12 held-out tasks.
    pmap = {f"{s}/{m}": SG["tests"][f"{s}/{m}"]["identity_vs_cas_effective"] for s in SG_SUITES
            for m in SG_MODELS}
    assert all(t["mean_b"] == 0.0 for t in pmap.values())
    assert pmap["heldout/openai_gpt_5_5"]["p_exact"] == "1/16"
    assert pmap["heldout/openai_gpt_5_5"]["p_two_sided"] == 2 / 2**5
    assert all(t["p_two_sided"] <= 1 / 32 for k, t in pmap.items() if k != "heldout/openai_gpt_5_5")
    _assert_in_tex("$p \\le 1/32$ in five of the six rows")
    _assert_in_tex("CAS accepted a stale write in only 5 of the 12 tasks, and $p = 1/16$")
    assert cas("relisted_completed") == 227 and cas("not_relisted_completed") == 0
    not_relisted = cas("not_relisted_completed") + cas("not_relisted_stale") + cas("not_relisted_neither")
    assert (not_relisted, cas("not_relisted_stale")) == (225, 144)
    assert cas("completed") - cas("completed_and_stale") == 227
    relisted = cas("relisted_completed") + cas("relisted_stale") + cas("relisted_neither")
    assert (relisted, cas("not_relisted_neither")) == (243, 81) and relisted + not_relisted == cas("rejected")
    _assert_in_tex("In 243 of the 468 rejected rollouts the agent re-lists, and in 225 it does not")
    assert (cas("relisted_stale"), cas("relisted_neither")) == (11, 5)
    _assert_in_tex("rejected, then re-lists & 243 & 227 & 11 & 5")
    _assert_in_tex("rejected, does not re-list & 225 & 0 & 144 & 81")
    no_write = sum(_sg("guard_cas", s, m)["no_eligible_write"] for s in SG_SUITES for m in SG_MODELS)
    assert no_write == 6 == cas("exposed") - cas("rejected")
    assert cas("neither") == 92 == cas("relisted_neither") + cas("not_relisted_neither") + no_write
    _assert_in_tex("makes no write & 6 & 0 & 0 & 6")
    _assert_in_tex("all & 474 & 227 & 155 & 92")
    assert cas("completed_and_stale") == 7
    _assert_in_tex("in 7\nof these 155 rollouts the agent later also wrote to the correct record".replace("\n", " "))
    _assert_in_tex("Of the 225 that do not re-list, none completes")
    assert cas("relisted_stale") == 11 == cas("stale") - cas("not_relisted_stale")
    assert cas("relisted_stale") + cas("not_relisted_stale") == cas("stale") == 155
    g = [SGO[f"guard_cas/{s}/openai_gpt_5_5"] for s in SG_SUITES]
    assert (sum(x["neither"] for x in g), sum(x["exposed"] for x in g)) == (75, 159)
    assert all(x["rejected"] == x["exposed"] for x in g)
    assert [f"{_sg('guard_cas', s, 'openai_gpt_5_5')['completion_mean']:.2f}" for s in SG_SUITES] \
        == ["0.26", "0.33"]
    _assert_in_tex("In 75 of its 159 rollouts it makes no successful write after the rejection, and "
                   "its completion is 0.26 on the main suite and 0.33 on the held-out suite")
    for s in ("Nevertheless, 155 rollouts contain an accepted stale write (160 such writes in all",
              "A compare-and-set guard accepted 160 stale writes"):
        _assert_in_tex(s)


def test_sg_prose_identity():
    """Identity guard: no accepted stale write, its re-list paths, and completion against
    the policy."""
    idn = lambda k: _v2o("guard_identity", k)  # noqa: E731
    assert (idn("exposed"), idn("stale"), idn("rejected"), idn("completed")) == (473, 0, 457, 445)
    relisted = idn("relisted_completed") + idn("relisted_stale") + idn("relisted_neither")
    assert relisted == 454 and idn("relisted_completed") == 445
    _assert_in_tex("no stale write is accepted in any of the six rows (473 rollouts)")
    _assert_in_tex("In 457 rollouts the agent still tries the old identifier first and is rejected")
    assert (idn("relisted_neither"), idn("not_relisted_neither"), idn("neither")) == (9, 3, 28)
    _assert_in_tex("rejected, then re-lists & 454 & 445 & 0 & 9")
    _assert_in_tex("rejected, does not re-list & 3 & 0 & 0 & 3")
    no_write = sum(_sg("guard_identity", s, m)["no_eligible_write"] for s in SG_SUITES for m in SG_MODELS)
    assert no_write == 16 == idn("exposed") - idn("rejected")
    _assert_in_tex("makes no write & 16 & 0 & 0 & 16")
    _assert_in_tex("all & 473 & 445 & 0 & 28")
    assert _sg_range("guard_identity", "attempted_mean") == "0.50--0.60"
    _assert_in_tex("attempted WEAR stays at 0.50--0.60")
    assert _sg_range("guard_identity", "completion_mean") == "0.84--1.00"
    for s in ("Completion is 0.84--1.00. In no row is it lower than under the policy",
              "a correct write follows in 84--100\\% of rollouts"):
        _assert_in_tex(s)
    for s in SG_SUITES:
        for m in SG_MODELS:
            assert _sg("guard_identity", s, m)["completion_mean"] >= _sg("policy", s, m)["completion_mean"]
            t = SG["tests"][f"{s}/{m}"]["guard_identity_vs_policy_completion"]
            if "qwen" in m:
                assert t["p_exact"] == {"main": "1/65536", "heldout": "1/2048"}[s]
            else:
                assert t["p_two_sided"] >= 1 / 8
    _assert_in_tex("For Qwen it is higher ($p = 1/65536$ on the main suite and $1/2048$ on the "
                   "held-out suite)")
    _assert_in_tex("the two are not distinguishable ($p \\geq 1/8$)")


def test_sg_prose_replay_and_cost():
    """False-rejection replay counts and tool calls per rollout under each safeguard."""
    n_sg = sum(v["rollouts"] for k, v in SG["h3c_false_rejection_replay"].items() if k.endswith("/cas"))
    n_paper = sum(v["rollouts"] for k, v in SG["h3c_paper_baselines"].items() if k.endswith("/cas"))
    assert (n_sg, n_paper) == (480, 600)
    _assert_in_tex("the 480 baseline rollouts without a safeguard from the safeguard study, and "
                   "the 600 baseline rollouts")
    for s in ("the tool calls of 1{,}080 baseline rollouts, in which nothing changes, produce no "
              "rejection",):
        _assert_in_tex(s)
    r = lambda a: _sg_range(a, "tool_calls_mean")  # noqa: E731
    _assert_in_tex(f"A rollout makes {r('none')} tool calls without a safeguard, {r('policy')} "
                   f"with the policy, {r('guard_cas')} with CAS and {r('guard_identity')} with "
                   "the identity guard")


def test_state_check_write_totals():
    """State check: 587 wrong writes (555 changed a record, 32 did not), 1,204 other writes
    without an error (all changed their target), 4 writes with an error (none changed anything)."""
    t = json.load(open(HERE.parent / "audit" / "state_check_20261124" / "results.json"))["totals"]
    w = {k: v for k, v in t.items() if "|" in k}
    assert sum(w.values()) == 1795 and t["trajectories"] == t["faithful"] == 1798
    wrong = {k: v for k, v in w.items() if "|wrong_target|" in k}
    assert wrong == {"perturbed|wrong_target|nonerror|changed": 555,
                     "perturbed|wrong_target|nonerror|unchanged": 32}
    other_ok = {k: v for k, v in w.items() if "|wrong_target|" not in k and "|nonerror|" in k}
    assert sum(other_ok.values()) == 1204 and all(k.endswith("|changed") for k in other_ok)
    err = {k: v for k, v in w.items() if "|error|" in k}
    assert sum(err.values()) == 4 and all(k.endswith("|unchanged") for k in err)
    _assert_in_tex("555 of the 587 wrong writes changed the wrong record")
    _assert_in_tex("Of the 1{,}795 writes of the main experiment, 587 are the wrong writes above")
    _assert_in_tex("1{,}204 writes returned no error, and all of them changed their target")
    _assert_in_tex("The last 4 returned an error and changed nothing")


# ------------------------------------------------------------------ figures drawn by figs_v3.py
def test_new_figures_match_text_and_tables():
    """The data figures read NUMBERS.json through figs_v3's *_data helpers; their values
    must be the ones the text and the supplementary tables state."""
    import figs_v3
    r = figs_v3.regimes_data(NUM)
    assert (r["r3a"], r["r3b"], r["r3c"], r["n_recovery"]) == (13, 7, 5, 25)
    assert r["r2_stall"] == (12, 40) and round(r["r1_lo"], 3) == 0.907 and r["r1_hi"] == 1.0
    g = figs_v3.guard_flow_data(NUM)
    for guard, rows in (("guard_cas", [("rejected, then re-lists", "relisted"),
                                       ("rejected, does not re-list", "not_relisted")]),
                        ("guard_identity", [("rejected, then re-lists", "relisted"),
                                            ("rejected, does not re-list", "not_relisted")])):
        t = g[guard]
        for label, k in rows:
            row = (f"{label} & {t[k]} & {t[k + '_completed']} & {t[k + '_stale']} & "
                   f"{t[k + '_neither']}")
            _assert_in_tex(row)
        _assert_in_tex(f"makes no write & {t['no_write']} & 0 & 0 & {t['no_write']}")
    s = figs_v3.safeguard_data(NUM)
    tab = (HERE / "tables" / "tab_safeguard_study.tex").read_text()
    for (m, suite, arm), (eff, comp) in s.items():
        assert f"{eff:.2f}" in tab and f"{comp:.2f}" in tab
    j = figs_v3.judge_data(NUM)
    assert round(100 * j["cross"]["full"]) == 96 and round(100 * j["cross"]["full_remap"]) == 94
    six = (HERE / "tables" / "tab_sixgate.tex").read_text()
    for v in j["six"].values():
        assert f"{100 * v:.1f}\\%" in six
    lad = (HERE / "tables" / "tab_ladder.tex").read_text()
    for v in j["ladder"].values():
        assert f"{100 * v:.0f}\\%" in lad
