"""Render the paper's figures and generated LaTeX tables from NUMBERS.json.

Every measured value comes from NUMBERS.json; nothing is computed from run files
here. This file draws the ToolSandbox figure and writes the tables;
figs_v3.render_all() draws the framework, trajectory, harm-regime, safeguard,
guard-flow and judge figures. The mechanism figure is drawn separately by
figs/make_fig1.py.

Usage (repo root, after collect_numbers.py):
  tau2-bench/.venv/bin/python wear/paper/make_figures.py
Reads: wear/paper/NUMBERS.json.
Writes: wear/paper/figs/*.{pdf,png} and wear/paper/tables/*.tex (included by the
manuscript).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
NUM = json.load(open(HERE / "NUMBERS.json"))
FIGS = HERE / "figs"
TABLES = HERE / "tables"
FIGS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

MODELS = ["openai_gpt_5_5", "openai_gemini_3_1_pro", "openai_gemini_3_5_flash",
          "openai_deepseek_v4_pro", "openai_qwen3_7_plus"]
SHORT = {
    "openai_gpt_5_5": "GPT-5.5",
    "openai_gemini_3_1_pro": "Gemini 3.1 Pro",
    "openai_gemini_3_5_flash": "Gemini 3.5 Flash",
    "openai_deepseek_v4_pro": "DeepSeek v4 Pro",
    "openai_qwen3_7_plus": "Qwen3.7 Plus",
}


XTICK = {"openai_gpt_5_5": "GPT-5.5", "openai_gemini_3_1_pro": "Gemini Pro",
         "openai_gemini_3_5_flash": "Flash", "openai_deepseek_v4_pro": "DeepSeek",
         "openai_qwen3_7_plus": "Qwen"}


def xtick(m: str) -> str:
    """Short model name used in figures and tables (Gemini Pro, Flash, DeepSeek, Qwen)."""
    return XTICK[m]
# a light-to-dark gray ramp plus one accent: the three conditions stay apart in grayscale
COND_COLORS = {"baseline": "#d9d9d9", "control": "#969696", "perturbed": "#d95f02"}
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    # embed TrueType (Type 42) serif fonts; no Type 3 fonts in the PDFs
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
})


def fmt_pct(v, digits: int = 1) -> str:
    """Format a fraction as a LaTeX percentage, or n/a for None."""
    return f"{v*100:.{digits}f}\\%" if v is not None else "n/a"


def cell(d: dict, key: str) -> dict:
    """Return d[key]; a missing entry raises KeyError naming it."""
    c = d.get(key)
    if c is None:
        raise KeyError(f"NUMBERS.json missing expected cell: {key}")
    return c


def savefig(fig, name: str) -> None:
    """Save fig as figs/<name>.pdf and a 200-dpi figs/<name>.png, then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote figs/{name}.pdf/.png")


# ---------------------------------------------------------------- ToolSandbox figure
# scenario groups in plotting order, with their axis labels
TS_ORDER = [
    ("static", "static\n(3 scenarios)"),
    ("idreuse", "dynamic ID-REUSE\n(3 scenarios)"),
    ("cue_present", "cue-present\n(1 scenario)"),
    ("no_distractor", "no-distractor\n(1 scenario)"),
]
# agent model -> its directory under runs/toolsandbox (the fig4_ts keys)
TS_MODEL_DIR = {
    "openai_gpt_5_5": "gpt-5.5",
    "openai_gemini_3_1_pro": "gemini-3.1-pro",
    "openai_gemini_3_5_flash": "gemini-3.5-flash",
}
# model palette must stay hue-disjoint from COND_COLORS (gray ramp plus
# orange for perturbed): one color = one meaning across figures
from figs_v3 import MODEL_COLORS as TS_MODEL_COLORS  # noqa: E402  (shared with the judge figure)


def fig4() -> None:
    """ToolSandbox figure: attempted WEAR per scenario group and model.

    Each bar is the mean over the scenarios of its group. Where effective WEAR is
    below attempted WEAR, the solid bar shows effective WEAR and a hatched outline
    extends it to attempted WEAR. Sized to print at about 0.8 of the text width."""
    ts = NUM["fig4_ts"]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ts_models = list(TS_MODEL_DIR)
    ngrp = len(TS_ORDER)
    for mi, m in enumerate(ts_models):
        md = cell(ts, TS_MODEL_DIR[m])
        groups = cell(md, "_groups")
        att = [groups[grp]["attempted_mean"] if grp in groups else 0.0
               for grp, _ in TS_ORDER]
        eff = [groups[grp]["effective_mean"] if grp in groups else 0.0
               for grp, _ in TS_ORDER]
        xs = [gi + (mi - 1) * 0.26 for gi in range(ngrp)]
        col = TS_MODEL_COLORS[m]
        # bar height = attempted; solid fill = effective, hatched where they
        # differ (a y=0 tick is invisible against the axis)
        att_only = [a if e != a else 0.0 for a, e in zip(att, eff)]
        solid = [a if e == a else e for a, e in zip(att, eff)]
        ax.bar(xs, att_only, width=0.24, facecolor="none", edgecolor=col,
               linewidth=0.6, hatch="///")
        ax.bar(xs, solid, width=0.24, label=xtick(m), edgecolor="black",
               linewidth=0.4, color=col)
        for x, a in zip(xs, att):
            if a == 0:
                ax.text(x, 0.015, "0", ha="center", va="bottom", fontsize=7,
                        color="black")
    ax.set_xticks(range(ngrp))
    ax.set_xticklabels([lbl for _, lbl in TS_ORDER], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("attempted WEAR", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="0.35", hatch="///",
                         label="attempted, all rejected"))
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    savefig(fig, "fig4_generalization")


# ---------------------------------------------------------------- tables
def judge_prompts_tex() -> None:
    """Write tables/tab_ladder.tex, the general-purpose judge prompt table of the
    supplementary material: miss rate per judge model and prompt."""
    t1 = NUM["judge_prompts"]
    order = [("gpt-5.4-mini", "GPT-5.4-mini"), ("gemini-3.1-pro", "Gemini Pro"),
             ("gpt-5.5", "GPT-5.5")]
    tex = ["\\begin{tabular}{lrr}", "\\toprule",
           "judge & consistency prompt & fidelity prompt \\\\", "\\midrule"]
    for key, label in order:
        c = cell(t1, f"{key}_consistency")["miss_rate"]
        f = cell(t1, f"{key}_fidelity")["miss_rate"]
        tex.append(f"{label} & {fmt_pct(c, 0)} & {fmt_pct(f, 0)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    # NUMBERS.json stores these results under the key judge_prompts
    (TABLES / "tab_ladder.tex").write_text("\n".join(tex) + "\n")
    print("wrote tables/tab_ladder.tex")


def sixgate_judge_tex() -> None:
    """Write tables/tab_sixgate.tex, the six-gate judge table of the supplementary
    material: miss rate per trajectory source and what the judge is told, plus a row
    with Gemini Pro as the judge. A trailing LaTeX comment records the coverage totals."""
    t2 = NUM["sixgate_judge"]
    pools = [("openai_gpt_5_5", "GPT-5.5"), ("openai_gemini_3_1_pro", "Gemini Pro"),
             ("openai_gemini_3_5_flash", "Flash"), ("recover", "recovery suite")]
    gts = ("full", "full_remap", "scenario", "none")
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "trajectories from & traj. & original gold & corrected gold & "
           "instructions only & nothing \\\\",
           "\\midrule"]
    for key, label in pools:
        n = cell(t2["derived"], f"{key}|full")["n_wrong"]
        row = [(t2["derived"].get(f"{key}|{gt}") or {}).get("miss_rate") for gt in gts]
        tex.append(f"{label} & {n} & " + " & ".join(fmt_pct(r) for r in row) + " \\\\")
    cj = NUM["sixgate_gemini_judge"]
    k = cj["contingency"]
    n_cj = k["pass_wrong"] + k["fail_wrong"]
    tex += ["\\midrule",
            f"GPT-5.5, Gemini Pro as judge & {n_cj} & {fmt_pct(cj['miss_rate'])} & "
            f"{fmt_pct(cj['full_remap']['miss_rate'])} & --- & --- \\\\"]
    cov = t2["coverage"]
    foot = (f"wrong-bind total {cov['total_wrong_bind_trajectories']}; six-gate missed: "
            f"task-gold {cov['sixgate_fullGT_missed']}, corrected-gold "
            f"{cov.get('sixgate_correctedGT_missed', '-')}; DB-diff blind "
            f"{cov['db_diff_blind_transient']}; "
            f"WEAR {cov['wear_caught']}/{cov['total_wrong_bind_trajectories']}")
    tex += ["\\bottomrule", "\\end{tabular}", f"% {foot}"]
    # NUMBERS.json stores these results under the key sixgate_judge
    (TABLES / "tab_sixgate.tex").write_text("\n".join(tex) + "\n")
    print("wrote tables/tab_sixgate.tex")


# The *_lines table builders below are pure functions of NUMBERS.json:
# test_paper_numbers.py calls them and compares the result with the files in tables/.
MIT_ORDER = [("openai_gpt_5_5", "GPT-5.5"), ("openai_deepseek_v4_pro", "DeepSeek")]


def _num3(v: float) -> str:
    """Table cell number style: round(v, 3) rendered as the Python float string."""
    return str(round(v, 3))


def _fix3(v: float) -> str:
    """Point estimates of the main results table: always three decimals, as the text
    quotes them (0.907)."""
    return f"{v:.3f}"


def _ci2(c: dict) -> str:
    """A cell's bootstrap 95% interval as [lo, hi] with two decimals."""
    lo, hi = c["wear_ci95"]
    return f"[{lo:.2f}, {hi:.2f}]"


def _tab_cells_lines(num: dict) -> list[str]:
    """Rows of the main results table: perturbed WEAR with its bootstrap 95% interval per
    domain, and the baseline and control values (0 in every cell, interval [0, 0])."""
    cells = num["fig2"]["cells"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{perturbed: WEAR [95\\% CI]} & baseline & control \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-5}",
             "model & airline & retail & airline / retail & airline / retail \\\\", "\\midrule"]
    for m in MODELS:
        row = [XTICK[m]]
        for dom in ("airline", "retail"):
            c = cell(cells, f"{m}/perturbed/{dom}")
            assert c["eff_wear_mean"] == c["wear_mean"], (m, dom)  # effective == attempted WEAR
            row.append(f"{_fix3(c['wear_mean'])} {_ci2(c)}")
        for cond in ("baseline", "control"):
            pair = []
            for dom in ("airline", "retail"):
                c = cell(cells, f"{m}/{cond}/{dom}")
                assert c["wear_mean"] == 0 and c["wear_ci95"] == [0.0, 0.0], (m, cond, dom)
                pair.append(_fix3(c["wear_mean"]))
            row.append(f"{pair[0]} / {pair[1]}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def _pool_tasks(a: str, b: str) -> str:
    """Add two "k/n" counts, e.g. the affected-task counts of the two domains."""
    n1, d1 = a.split("/")
    n2, d2 = b.split("/")
    return f"{int(n1) + int(n2)}/{int(d1) + int(d2)}"


def _tab_mitigation_lines(num: dict) -> list[str]:
    """Rows of the verification-policy table: task-mean WEAR in every column; in
    parentheses, the tasks with at least one wrong-binding rollout."""
    tm, mit, pla = num["mitigation_task_means"], num["mitigation"], num["mitigation_placebo"]
    f = lambda x: f"{x:.2f}"
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             " & no safeguard & verification policy & placebo \\\\", "\\midrule"]
    for m, label in MIT_ORDER:
        t = tm[m]
        assert t["unmitigated_baseline"]["task_mean_wear"] == 0.0, m
        f2 = num["fig2"]["cells"]
        unmit_tasks = _pool_tasks(*[cell(f2, f"{m}/perturbed/{d}")["tasks_with_wrong_bind"]
                                    for d in ("airline", "retail")])
        lines.append(
            f"{label}, perturbed & {f(t['unmitigated_perturbed']['task_mean_wear'])} ({unmit_tasks}) & "
            f"\\textbf{{{f(t['policy_perturbed']['task_mean_wear'])}}} "
            f"({mit[m]['perturbed']['tasks_with_wrong_bind']}) & "
            f"{f(t['placebo_perturbed']['task_mean_wear'])} "
            f"({pla[m]['perturbed']['tasks_with_wrong_bind']}) \\\\")
        unmit_base = _pool_tasks(*[cell(f2, f"{m}/baseline/{d}")["tasks_with_wrong_bind"]
                                   for d in ("airline", "retail")])
        lines.append(
            f"{label}, baseline & 0.00 ({unmit_base}) & {f(t['policy_baseline']['task_mean_wear'])} "
            f"({mit[m]['baseline']['tasks_with_wrong_bind']}) & --- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


SG_ORDER = [("openai_gpt_5_5", "GPT-5.5"), ("openai_deepseek_v4_pro", "DeepSeek"),
            ("openai_qwen3_7_plus", "Qwen")]
SG_ARMS = ("none", "policy", "guard_cas", "guard_identity")


def _tab_safeguard_study_lines(num: dict) -> list[str]:
    """Rows of the safeguard table in the supplementary material: in the perturbed
    condition, by model and suite, three panels with the task means of effective WEAR,
    completion and tool calls per rollout. Under none and the policy, attempted and
    effective WEAR agree to two decimals (asserted); one Qwen write under the policy on
    the main suite returned an error."""
    cells = num["safeguard_study"]["cells"]
    f = lambda x: f"{x:.2f}"  # noqa: E731
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "model & suite & none & policy & CAS & identity \\\\"]
    panels = (("effective_mean", "Effective WEAR"), ("completion_mean", "Completion"),
              ("tool_calls_mean", "Tool calls per rollout"))
    for key, title in panels:
        lines += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for m, label in SG_ORDER:
            for suite, slabel in (("main", "main"), ("heldout", "held-out")):
                cs = [cells[f"{a}/{suite}/{m}/perturbed"] for a in SG_ARMS]
                if key == "effective_mean":
                    for c in cs[:2]:
                        assert f(c["effective_mean"]) == f(c["attempted_mean"]), (m, suite)
                row = [label if suite == "main" else "", slabel] + [f(c[key]) for c in cs]
                lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


SG_SAFEGUARDS = [("none", "none"), ("policy", "policy"), ("guard_cas", "CAS"),
                 ("guard_identity", "identity")]
SG_ROWS = [("openai_gpt_5_5", "GPT-5.5"), ("openai_deepseek_v4_pro", "DeepSeek"),
           ("openai_qwen3_7_plus", "Qwen")]


def _tab_excluded_lines(num: dict) -> list[str]:
    """Rows of the excluded-rollouts table in the supplementary material: perturbed
    rollouts of the safeguard study without a binding read (the change was never
    applied), by model, suite and safeguard; they are excluded from the perturbed measures."""
    cells = num["safeguard_study"]["cells"]
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             " & & & \\multicolumn{4}{c}{safeguard} \\\\", "\\cmidrule(lr){4-7}",
             "model & suite & rollouts & " + " & ".join(l for _, l in SG_SAFEGUARDS) + " \\\\",
             "\\midrule"]
    for m, mlab in SG_ROWS:
        for su, slab in (("main", "main"), ("heldout", "held-out")):
            cs = [cell(cells, f"{a}/{su}/{m}/perturbed") for a, _ in SG_SAFEGUARDS]
            n = {c["rollouts"] for c in cs}
            assert len(n) == 1, (m, su, n)
            lines.append(f"{mlab} & {slab} & {n.pop()} & "
                         + " & ".join(str(c["unexposed"]) for c in cs) + " \\\\")
    total = sum(cell(cells, f"{a}/{su}/{m}/perturbed")["unexposed"]
                for a, _ in SG_SAFEGUARDS for su in ("main", "heldout") for m, _ in SG_ROWS)
    lines += ["\\midrule", f"all & & & \\multicolumn{{4}}{{c}}{{{total}}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    return lines


def tab_excluded_tex() -> None:
    (TABLES / "tab_excluded.tex").write_text("\n".join(_tab_excluded_lines(NUM)) + "\n")
    print("wrote tables/tab_excluded.tex")


def tab_safeguard_study_tex() -> None:
    (TABLES / "tab_safeguard_study.tex").write_text(
        "\n".join(_tab_safeguard_study_lines(NUM)) + "\n")
    print("wrote tables/tab_safeguard_study.tex")


def tab_cells_tex() -> None:
    (TABLES / "tab_cells.tex").write_text("\n".join(_tab_cells_lines(NUM)) + "\n")
    print("wrote tables/tab_cells.tex")


def tab_mitigation_tex() -> None:
    (TABLES / "tab_mitigation.tex").write_text(
        "\n".join(_tab_mitigation_lines(NUM)) + "\n")
    print("wrote tables/tab_mitigation.tex")


if __name__ == "__main__":
    import figs_v3  # schematics and the remaining data figures
    figs_v3.render_all(NUM)
    fig4()
    judge_prompts_tex()
    sixgate_judge_tex()
    tab_cells_tex()
    tab_mitigation_tex()
    tab_safeguard_study_tex()
    tab_excluded_tex()
