"""Six figures of the paper: the framework and trajectory figures, and the harm-regime,
safeguard, guard-flow and judge figures, which show measured values.

Not run directly: make_figures.py calls render_all(NUM). The figures with measured
values read NUMBERS.json through the pure *_data() helpers below, which
test_paper_numbers.py also calls, so a value drawn in a figure is the value checked
against the text. The framework figure shows no measured value; the trajectory
figure quotes one recorded trajectory as fixed text.
Writes: wear/paper/figs/<name>.pdf and .png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"

# palette shared with the mechanism figure (figs/make_fig1.py)
INK, GREY, LINE, BLUE, RED, GREEN = "#1a1a1a", "#9aa3ae", "#555555", "#2f6db5", "#b03a2e", "#2e7d4f"
FILL, REDFILL, BLUEFILL, GREENFILL = "#f3f5f8", "#fbeae8", "#e8f0fa", "#e7f3ec"
SANS = {"family": "DejaVu Sans"}
MONO = {"family": "DejaVu Sans Mono"}
# models and suites of the safeguard study, and the four safeguards (key, label, color, marker)
SG_MODELS = [("openai_gpt_5_5", "GPT-5.5"), ("openai_deepseek_v4_pro", "DeepSeek"),
             ("openai_qwen3_7_plus", "Qwen")]
SG_SUITES = [("main", "main suite (20 tasks)"), ("heldout", "held-out suite (12 tasks)")]
SAFEGUARDS = [("none", "none", "#8c8c8c", "o"), ("policy", "policy", BLUE, "s"),
              ("guard_cas", "CAS guard", "#d95f02", "^"), ("guard_identity", "identity guard", GREEN, "D")]
# one color per agent model in every figure that colors by model (the ToolSandbox
# and judge figures);
# hue-disjoint from the condition colors of make_figures.py (gray ramp, orange = perturbed)
MODEL_COLORS = {
    "openai_gpt_5_5": "#4d4d4d",
    "openai_gemini_3_1_pro": "#e7298a",
    "openai_gemini_3_5_flash": "#a6761d",
}


def _save(fig, name: str) -> None:
    """Save fig as figs/<name>.pdf and a 300-dpi figs/<name>.png, then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)  # reproducible bytes
    plt.close(fig)
    print(f"wrote figs/{name}.pdf/.png")


def _canvas(w: float, h: float):
    """Return (fig, ax): a w x h inch figure whose hidden axes use inch coordinates."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, title=None, body=None, fc=FILL, ec=GREY, size=7.5,
         body_font=None, title_color=INK, body_color=INK, lw=0.8, align="center"):
    """Draw a rounded box with lower-left corner (x, y), an optional bold title and body."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    cx = x + w / 2 if align == "center" else x + 0.08
    ha = "center" if align == "center" else "left"
    if title and body:
        ax.text(cx, y + h - 0.13, title, ha=ha, va="top", fontsize=size, fontweight="bold",
                color=title_color, **SANS)
        ax.text(cx, y + h - 0.34, body, ha=ha, va="top", fontsize=size - 0.7,
                color=body_color, linespacing=1.35, **(body_font or SANS))
    elif title:
        ax.text(cx, y + h / 2, title, ha=ha, va="center", fontsize=size, fontweight="bold",
                color=title_color, linespacing=1.3, **SANS)


def _arrow(ax, x0, y0, x1, y1, color=LINE, both=False, lw=0.9, label=None, lpos=0.5,
           loff=(0, 0.07), size=6.5, style="-|>"):
    """Draw an arrow from (x0, y0) to (x1, y1), two-headed if both, with an optional label."""
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=("<|-|>" if both else style),
                        mutation_scale=7, color=color, lw=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        ax.text(x0 + (x1 - x0) * lpos + loff[0], y0 + (y1 - y0) * lpos + loff[1], label,
                ha="center", va="bottom", fontsize=size, color=LINE, **SANS)


# ------------------------------------------------------------------ framework
def fig_framework() -> None:
    """Framework figure: where the perturbation layer sits, the three conditions, and
    offline scoring.

    Drawn at its printed width (5.38 in, set at \\textwidth = 5.42 in), so every
    label prints at 7 pt or more."""
    W, H = 5.38, 3.30
    fig, ax = _canvas(W, H)
    T, B, S = 8, 7.5, 7  # section titles, box titles, all other text (pt)

    def cbox(x, y, w, h, title, body, fc=FILL, ec=GREY, tc=INK, bc=INK):
        """Box with its title and body centered together vertically."""
        _box(ax, x, y, w, h, None, None, fc=fc, ec=ec)
        n = body.count("\n") + 1
        tb, lh = B * 1.25 / 72, S * 1.3 / 72
        top = y + h / 2 + (tb + 0.06 + n * lh) / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=B, fontweight="bold",
                color=tc, **SANS)
        ax.text(x + w / 2, top - tb - 0.06, body, ha="center", va="top", fontsize=S, color=bc,
                linespacing=1.3, **SANS)

    ax.text(0.0, H - 0.01, "Rollout", fontsize=T, fontweight="bold", va="top", color=INK, **SANS)
    h = 1.02
    y0 = H - 0.22 - h
    # widths = measured 7 pt text + padding; gaps fit the arrow labels (messages 0.49 in,
    # tool calls 0.45 in) with room on both sides
    xa, wa = 0.0, 0.92
    xb, wb = 1.58, 0.64
    xc, wc = 2.84, 1.30
    xd, wd = 4.52, 0.86
    cbox(xa, y0, wa, h, "User simulator", "LLM, reads\nthe user\ninstructions")
    cbox(xb, y0, wb, h, "Agent", "LLM under\ntest, binds\nand writes")
    cbox(xc, y0, wc, h, "Perturbation layer",
         "counts calls to the\ntrigger tool, applies\nthe scripted change\nonce, logs it, and\nupdates the gold target",
         fc=BLUEFILL, ec=BLUE, tc=BLUE)
    cbox(xd, y0, wd, h, "Benchmark", "tools and\ndatabase (code\nunchanged)")
    ym = y0 + h / 2
    _arrow(ax, xa + wa, ym, xb, ym, both=True, label="messages", size=S, loff=(0, 0.05))
    _arrow(ax, xb + wb, ym, xc, ym, both=True, label="tool calls", size=S, loff=(0, 0.05))
    _arrow(ax, xc + wc, ym, xd, ym, both=True)
    # conditions
    yh = y0 - 0.12
    ax.text(0.0, yh, "Conditions (same change, different time)", fontsize=T, fontweight="bold",
            va="top", color=INK, **SANS)
    xs, xe = 0.68, 3.20
    t_start, t_bind, t_write = 1.02, 2.02, 2.92
    for i, (name, when) in enumerate((("baseline", None), ("control", "start"), ("perturbed", "bind"))):
        yy = yh - 0.42 - i * 0.30
        ax.text(0.0, yy, name, fontsize=S, va="center", color=INK, **SANS)
        ax.plot([xs, xe], [yy, yy], color=GREY, lw=0.8)
        for t in (t_start, t_bind, t_write):
            ax.plot([t, t], [yy - 0.05, yy + 0.05], color=LINE, lw=0.8)
        if when:
            xm = (xs + t_start) / 2 if when == "start" else t_bind + 0.22
            # blue diamond = the change, as in the mechanism figure (red marks wrong writes
            # in the mechanism, harm-regime and guard-flow figures)
            ax.plot(xm, yy, marker="D", ms=5, color=BLUE, mec=BLUE)
    ytl = yh - 0.42 - 2 * 0.30 - 0.14
    for t, lab in ((t_start, "conversation\nstarts"), (t_bind, "binding\nread"), (t_write, "write")):
        ax.text(t, ytl, lab, ha="center", va="top", fontsize=S, color=LINE, linespacing=1.15, **SANS)
    ax.plot(0.06, 0.38, marker="D", ms=5, color=BLUE, mec=BLUE)
    ax.text(0.16, 0.38, "the change", fontsize=S, va="center", color=BLUE, **SANS)
    # offline scoring: title and subtitle centered together in each box
    bx, bw = 3.50, 1.88
    yt, ht = 0.98, 0.56
    yw, hw = 0.24, 0.50
    for (yb, hb, title, sub, fc, ec, tc) in (
            (yt, ht, "Timing check", "binding read before\nthe first write?", FILL, GREY, INK),
            (yw, hw, "WEAR scorer", "each write vs. the gold target", GREENFILL, GREEN, GREEN)):
        cbox(bx, yb, bw, hb, title, sub, fc=fc, ec=ec, tc=tc, bc=LINE)
    _arrow(ax, bx + bw / 2, yt, bx + bw / 2, yw + hw)
    ax.text(bx + bw / 2, yw - 0.05, "attempted and effective WEAR", ha="center", va="top", fontsize=S,
            color=INK, **SANS)
    xv = 3.62
    _arrow(ax, xv, y0, xv, yt + ht, color=GREY)
    ax.text(xv + 0.08, (y0 + yt + ht) / 2, "after the rollout:\ntrajectory and change log", fontsize=S,
            color=LINE, ha="left", va="center", linespacing=1.15, **SANS)
    _save(fig, "fig_framework")


# ------------------------------------------------------------------ trajectory
def fig_trajectory() -> None:
    """Trajectory figure: one recorded perturbed GPT-5.5 trajectory
    (airline_idreuse_dallas, first rollout), quoted as fixed text."""
    W, H = 6.3, 3.0
    fig, ax = _canvas(W, H)
    rows = [
        ("1  Bind", FILL, GREY,
         [("get_reservation_details(NO6JO3)", MONO, INK), ("  →  {origin: JFK, destination: DFW, ...}", MONO, INK),
          ("agent: “I found your trip to Dallas: Reservation NO6JO3 ...”", SANS, INK)]),
        ("2  The records\n    change", BLUEFILL, BLUE,
         [("Dallas reservation re-created under ZZ0001", SANS, INK),
          ("NO6JO3 taken over by the user's DEN→LAS trip (formerly HKEG34)", SANS, INK)]),
        ("3  Write through\n    the old identifier", REDFILL, RED,
         [("update_reservation_baggages(reservation_id=\"NO6JO3\", total_baggages=3, ...)", MONO, INK),
          ("  →  {reservation_id: NO6JO3, origin: DEN, destination: LAS, ...}", MONO, RED)]),
        ("4  Confident\n    success", FILL, GREY,
         [("agent: “Your checked bag update is complete. Reservation: NO6JO3.”", SANS, INK)]),
    ]
    y = H - 0.05
    hs = [0.66, 0.52, 0.52, 0.40]
    for (lab, fc, ec, lines), h in zip(rows, hs):
        y -= h + 0.06
        ax.add_patch(FancyBboxPatch((0.05, y), W - 0.10, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(0.15, y + h / 2, lab, va="center", ha="left", fontsize=7.2, fontweight="bold",
                color=ec if ec != GREY else INK, linespacing=1.1, **SANS)
        for k, (txt, font, col) in enumerate(lines):
            ax.text(1.55, y + h - 0.12 - k * 0.165, txt, va="top", ha="left",
                    fontsize=6.3 if font is MONO else 6.8, color=col, **font)
    y -= 0.32
    ax.text(0.15, y + 0.12, "WEAR", fontsize=7.2, fontweight="bold", color=GREEN, va="center", **SANS)
    ax.text(1.55, y + 0.12, "NO6JO3 ≠ gold target ZZ0001: a wrong write that returned no error (attempted and effective)", fontsize=6.8, va="center", color=INK, **SANS)
    _save(fig, "fig_trajectory")


# ------------------------------------------------------------------ harm regimes
def regimes_data(num: dict) -> dict:
    """Values drawn in the harm-regime figure: the range of perturbed WEAR over the
    main-experiment cells (regime 1), the range over models of the share of no-distractor
    trajectories that never complete (regime 2), and the recovery suite's rollouts and
    split (regime 3: 3a undone, 3b a second wrong write, 3c incomplete or none)."""
    cells = num["fig2"]["cells"]
    pert = [c["wear_mean"] for k, c in cells.items() if k.split("/")[1] == "perturbed"]
    agg = num["nodistractor_singlechar"]["m2_recovery"]["agg"]
    by: dict = {}
    for k, v in agg.items():
        d = by.setdefault(k.split("/")[0], {"fo": 0, "rc": 0})
        d["fo"] += v["first_old"]; d["rc"] += v["recov"]
    stall = sorted(round(100 * (1 - d["rc"] / d["fo"])) for d in by.values())
    rec = num["recover"]
    fc = rec["forensic"]["counts"]
    return {"r1_lo": min(pert), "r1_hi": max(pert), "r2_stall": (stall[0], stall[-1]),
            "n_recovery": rec["total"]["n_sims"], "r3a": rec["total"]["KEY"],
            "r3b": fc["poisoned_recovery"],
            "r3c": fc["incomplete_recovery"] + fc["single_no_recovery"]}


def fig_regimes(num: dict) -> None:
    """Harm-regime figure: the three regimes of a write through the old identifier, and
    which evaluation method detects each.

    Drawn at its printed width (5.38 in, set at \\textwidth = 5.42 in), so every
    label prints at 7 pt or more."""
    d = regimes_data(num)
    W, H = 5.38, 3.92
    fig, ax = _canvas(W, H)
    B, S = 7.5, 7  # box titles, all other text (pt)

    def cbox(x, y, w, h, title, body=None, fc=FILL, ec=GREY, tc=INK, tsize=B):
        """Box with its title and body centered together vertically."""
        _box(ax, x, y, w, h, None, None, fc=fc, ec=ec)
        n = 0 if body is None else body.count("\n") + 1
        tb, lh = tsize * 1.25 / 72, S * 1.3 / 72
        top = y + h / 2 + (tb + (0.06 + n * lh if n else 0)) / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=tsize, fontweight="bold",
                color=tc, **SANS)
        if body:
            ax.text(x + w / 2, top - tb - 0.06, body, ha="center", va="top", fontsize=S, color=INK,
                    linespacing=1.3, **SANS)

    LW = 2.62            # width of each column
    L0, R0 = 0.0, W - LW
    lc, rc = L0 + LW / 2, R0 + LW / 2
    top3 = 1.30          # top of the regime-3 panel
    y2, h2 = top3 + 0.36, 0.62
    y1, h1 = y2 + h2 + 0.26, 0.58
    y0, h0 = y1 + h1 + 0.30, 0.42
    cbox(W / 2 - 1.30, y0, 2.60, h0, "Write through the old identifier", tsize=8)
    cbox(L0, y1, LW, h1, "Another record now holds it", "a distractor of the same user")
    cbox(R0, y1, LW, h1, "No record holds it", "no-distractor variant")
    _arrow(ax, W / 2 - 0.40, y0, lc, y1 + h1); _arrow(ax, W / 2 + 0.40, y0, rc, y1 + h1)
    r1 = f"{d['r1_lo']:.3f}".rstrip("0").rstrip(".") + f"–{d['r1_hi']:.1f}"
    cbox(L0, y2, LW, h2, "Regime 1: silent",
         f"succeeds on the wrong record\neffective WEAR {r1}", fc=REDFILL, ec=RED, tc=RED)
    cbox(R0, y2, LW, h2, "Regime 2: loud",
         f"rejected with an error; effective WEAR 0\n{d['r2_stall'][0]}–{d['r2_stall'][1]}% never complete the task")
    _arrow(ax, lc, y1, lc, y2 + h2); _arrow(ax, rc, y1, rc, y2 + h2)
    # regime 3 continues regime 1: a later user turn prompts a re-check
    _arrow(ax, lc, y2, lc, top3)
    ax.text(lc + 0.08, (y2 + top3) / 2, "a later user turn\nprompts a re-check", fontsize=S,
            color=LINE, ha="left", va="center", linespacing=1.15, **SANS)
    _box(ax, L0, 0.02, LW, top3 - 0.02, None, None, fc="white", ec=GREY)
    ax.text(lc, top3 - 0.07, "Regime 3: recovery", ha="center", va="top", fontsize=B,
            fontweight="bold", color=INK, **SANS)
    ax.text(lc, top3 - 0.24, f"recovery suite, {d['n_recovery']} trajectories", ha="center",
            va="top", fontsize=S, color=LINE, **SANS)
    lw_, gap = 0.80, 0.06
    x_first = L0 + (LW - 3 * lw_ - 2 * gap) / 2
    leaves = [("3a", f"{d['r3a']}", "undone:\nback to the\ngold state"),
              ("3b", f"{d['r3b']}", "a second\nwrong write"),
              ("3c", f"{d['r3c']}", "incomplete\nor none")]
    for k, (tag, n, txt) in enumerate(leaves):
        cbox(x_first + k * (lw_ + gap), 0.08, lw_, 0.74, f"{tag}  ({n})", txt, tsize=7.2)
    # which evaluation method detects each regime
    ax.text(R0, top3, "Who detects it", fontsize=B, fontweight="bold", color=INK, va="top", **SANS)
    cols = ["end-state\ncomparison", "LLM\njudge", "WEAR"]
    rows = [("1", ["yes*", "varies", "yes"]), ("2", ["task fails", "—", "yes"]),
            ("3a", ["no", "yes", "yes"]), ("3b", ["yes†", "—", "yes"]), ("3c", ["yes", "—", "yes"])]
    cx = [R0 + 0.82, R0 + 1.62, R0 + 2.30]
    for c, lab in zip(cx, cols):
        ax.text(c, top3 - 0.21, lab, ha="center", va="top", fontsize=S, color=LINE, linespacing=1.0, **SANS)
    for i, (r, vals) in enumerate(rows):
        yy = top3 - 0.62 - i * 0.145
        ax.text(R0 + 0.04, yy, r, fontsize=S, va="center", color=INK, **SANS)
        for c, v in zip(cx, vals):
            col = GREEN if v.startswith("yes") else (RED if v == "no" else LINE)
            ax.text(c, yy, v, ha="center", va="center", fontsize=S, color=col, **SANS)
    _save(fig, "fig_regimes")


# ------------------------------------------------------------------ safeguards
def safeguard_data(num: dict) -> dict:
    """Return {(model, suite, safeguard): (effective WEAR, completion)} for the perturbed
    cells of the safeguard study."""
    cells = num["safeguard_study"]["cells"]
    out = {}
    for m, _ in SG_MODELS:
        for s, _ in SG_SUITES:
            for a, *_ in SAFEGUARDS:
                c = cells[f"{a}/{s}/{m}/perturbed"]
                out[(m, s, a)] = (c["effective_mean"], c["completion_mean"])
    return out


def fig_safeguards(num: dict) -> None:
    """Safeguard figure: effective WEAR and completion per model and suite under the four
    safeguards."""
    d = safeguard_data(num)
    plt.rcParams.update({"font.family": "STIXGeneral"})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), sharey=True)
    ylab, yt = [], []
    k = 0
    for m, mlab in SG_MODELS:
        for s, slab in SG_SUITES:
            yt.append(k); ylab.append(f"{mlab}, {'main' if s == 'main' else 'held-out'}")
            k += 1
        k += 0.6
    for ai, (ax, metric, title) in enumerate(zip(axes, (0, 1), ("effective WEAR (lower is better)",
                                                               "completion (higher is better)"))):
        for j, (a, lab, col, mk) in enumerate(SAFEGUARDS):
            ys = []; xs = []
            k = 0
            for m, _ in SG_MODELS:
                for s, _ in SG_SUITES:
                    xs.append(d[(m, s, a)][metric]); ys.append(k + (j - 1.5) * 0.17)
                    k += 1
                k += 0.6
            ax.scatter(xs, ys, s=22, marker=mk, color=col, edgecolor="white", linewidth=0.4,
                       label=lab, zorder=3)
        ax.set_xlim(-0.04, 1.04)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
        for yv in yt:
            ax.axhline(yv, color="#f0f0f0", lw=0.5, zorder=0)
    axes[0].set_yticks(yt); axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout()
    _save(fig, "fig_safeguards")


# ------------------------------------------------------------------ guard flow
def guard_flow_data(num: dict) -> dict:
    """Rollout outcomes under each write guard, summed over models and suites, with the
    totals of the three paths: rejected then re-lists (relisted), rejected without a
    re-list (not_relisted), and no rejection (no_write = exposed - rejected, drawn as
    "makes no write")."""
    ro = num["safeguard_study"]["rollout_outcomes"]
    keys = ["exposed", "rejected", "relisted_completed", "relisted_stale", "relisted_neither",
            "not_relisted_completed", "not_relisted_stale", "not_relisted_neither", "completed",
            "stale", "neither"]
    out = {}
    for g in ("guard_cas", "guard_identity"):
        tot = {k: 0 for k in keys}
        for kk, v in ro.items():
            if kk.startswith(g + "/"):
                for k in keys:
                    tot[k] += v[k]
        tot["relisted"] = tot["relisted_completed"] + tot["relisted_stale"] + tot["relisted_neither"]
        tot["not_relisted"] = tot["not_relisted_completed"] + tot["not_relisted_stale"] + tot["not_relisted_neither"]
        tot["no_write"] = tot["exposed"] - tot["rejected"]
        out[g] = tot
    return out


def fig_guard_flow(num: dict) -> None:
    """Guard-flow figure: how the perturbed rollouts under each write guard end.

    Drawn at its printed width (5.38 in, set at \\textwidth = 5.42 in), so every
    label prints at 7 pt or more. The two guards are stacked and share one bar scale."""
    d = guard_flow_data(num)
    S, T = 7, 8                       # all text, panel titles (pt)
    W = 5.38
    BOXW, BOXH, PITCH, TITLE = 1.72, 0.40, 0.47, 0.25
    panel_h = TITLE + 3 * PITCH
    LEG = 0.26                        # legend strip at the bottom
    H = LEG + 2 * panel_h + 0.12
    fig, ax = _canvas(W, H)
    bx = BOXW + 0.14                  # bars start here
    bw = W - bx - 0.32                # bar length of the largest path; both guards share one scale
    nmax = max(max(t["relisted"], t["not_relisted"], t["no_write"]) for t in d.values())
    for gi, (g, name) in enumerate((("guard_cas", "CAS guard"), ("guard_identity", "identity guard"))):
        t = d[g]
        top = H - gi * (panel_h + 0.12)
        ax.text(0.0, top - 0.01, f"{name}: {t['exposed']} perturbed rollouts", fontsize=T,
                fontweight="bold", va="top", color=INK, **SANS)
        paths = [("rejected, then\nre-lists the records", t["relisted"],
                  (t["relisted_completed"], t["relisted_stale"], t["relisted_neither"])),
                 ("rejected, does\nnot re-list", t["not_relisted"],
                  (t["not_relisted_completed"], t["not_relisted_stale"], t["not_relisted_neither"])),
                 ("makes no write", t["no_write"], (0, 0, t["no_write"]))]
        for pi, (lab, n, (c, s, ne)) in enumerate(paths):
            y = top - TITLE - (pi + 1) * PITCH + (PITCH - BOXH) / 2
            ym = y + BOXH / 2
            _box(ax, 0.0, y, BOXW, BOXH, None, None, fc=FILL, ec=GREY)
            ax.text(0.07, ym, lab, fontsize=S, va="center", color=INK, linespacing=1.15, **SANS)
            ax.text(BOXW - 0.07, ym, str(n), fontsize=T, fontweight="bold", va="center",
                    ha="right", color=INK, **SANS)
            # outcome bar, length proportional to the number of rollouts on this path
            segs = [(c, GREEN), (s, RED), (ne, "#bdbdbd")]
            cur = bx
            small = []
            for v, col in segs:
                if v == 0:
                    continue
                wseg = bw * v / nmax
                ax.add_patch(plt.Rectangle((cur, ym - 0.13), wseg, 0.26, fc=col, ec="white", lw=0.3))
                if wseg > 0.26:
                    ax.text(cur + wseg / 2, ym, str(v), ha="center", va="center", fontsize=S,
                            color="white", fontweight="bold", **SANS)
                else:
                    small.append((v, col))
                cur += wseg
            for si, (v, col) in enumerate(small):
                yy = ym if len(small) == 1 else ym + 0.08 - si * 0.16
                ax.text(cur + 0.05, yy, str(v), ha="left", va="center", fontsize=S,
                        color=("#7f7f7f" if col == "#bdbdbd" else col), fontweight="bold", **SANS)
            _arrow(ax, BOXW, ym, bx, ym, color=GREY)
    legend = ((GREEN, "completed, no stale write"), (RED, "stale write accepted"), ("#bdbdbd", "neither"))
    for (col, lab), x in zip(legend, (0.0, 2.05, 3.75)):
        ax.add_patch(plt.Rectangle((x, 0.04), 0.14, 0.11, fc=col, ec="none"))
        ax.text(x + 0.20, 0.095, lab, fontsize=S, va="center", color=INK, **SANS)
    _save(fig, "fig_guard_flow")


# ------------------------------------------------------------------ judges
# trajectory sources, what the judge is told, and the judges of the general-purpose prompts
JUDGE_SOURCES = [("openai_gpt_5_5", "GPT-5.5"), ("openai_gemini_3_1_pro", "Gemini Pro"),
                 ("openai_gemini_3_5_flash", "Flash"), ("recover", "recovery suite")]
JUDGE_LEVELS = [("full", "original gold"), ("full_remap", "corrected gold"),
                ("scenario", "instructions only"), ("none", "nothing")]
LADDER = [("gpt-5.4-mini", "GPT-5.4-mini"), ("gemini-3.1-pro", "Gemini Pro"), ("gpt-5.5", "GPT-5.5")]


def judge_data(num: dict) -> dict:
    """Miss rates drawn in the judge figure: six-gate judge by trajectory source and what
    it is told (six), Gemini Pro as six-gate judge (cross), and the general-purpose
    prompts by judge (ladder)."""
    t2 = num["sixgate_judge"]["derived"]
    six = {(s, lv): t2[f"{s}|{lv}"]["miss_rate"] for s, _ in JUDGE_SOURCES for lv, _ in JUDGE_LEVELS}
    cj = num["sixgate_gemini_judge"]
    cross = {"full": cj["miss_rate"], "full_remap": cj["full_remap"]["miss_rate"]}
    t1 = num["judge_prompts"]
    ladder = {(j, p): t1[f"{j}_{p}"]["miss_rate"] for j, _ in LADDER for p in ("consistency", "fidelity")}
    return {"six": six, "cross": cross, "ladder": ladder}


def fig_judges(num: dict) -> None:
    """Judge figure: (a) six-gate judge miss rates, (b) general-purpose prompt miss rates."""
    d = judge_data(num)
    plt.rcParams.update({"font.family": "STIXGeneral"})
    # The figure is 6.3 in wide and set at \textwidth (5.42 in); FS = 8.4 pt prints at about 7.2 pt.
    FS, FL = 8.4, 9.2   # smallest text; axis labels and panel titles
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw={"width_ratios": [2.1, 1]})
    # agents keep their colors from the ToolSandbox figure; the recovery suite is not
    # one model, so it is hatched
    styles = [dict(color=MODEL_COLORS[s]) if s in MODEL_COLORS else
              dict(color="white", hatch="////") for s, _ in JUDGE_SOURCES]
    wid = 0.19
    for i, (s, lab) in enumerate(JUDGE_SOURCES):
        xs = [k + (i - 1.5) * wid for k in range(len(JUDGE_LEVELS))]
        ys = [100 * d["six"][(s, lv)] for lv, _ in JUDGE_LEVELS]
        a1.bar(xs, ys, width=wid, label=lab, edgecolor="black", lw=0.3, **styles[i])
        for x, y in zip(xs, ys):
            a1.text(x, y + 1.2, f"{y:.0f}", ha="center", va="bottom", fontsize=FS)
    for k, lv in enumerate(("full", "full_remap")):
        a1.plot(k - 1.5 * wid, 100 * d["cross"][lv], marker="*", ms=9, color="black", zorder=4,
                linestyle="none")
        a1.text(k - 1.5 * wid + 0.08, 100 * d["cross"][lv], f"{100 * d['cross'][lv]:.0f}", fontsize=FS,
                va="center", ha="left")
    # say on the plot what the stars are; the x category gives the gold condition
    a1.text(1.12, 95, "stars: Gemini Pro as judge\non GPT-5.5's trajectories", fontsize=FS,
            ha="left", va="center", linespacing=1.05)
    a1.set_xticks(range(len(JUDGE_LEVELS)))
    a1.set_xticklabels([l.replace(" ", "\n") for _, l in JUDGE_LEVELS], fontsize=FS, linespacing=0.95)
    a1.set_xlabel("what the judge is told", fontsize=FL)
    a1.set_ylabel("miss rate (%)", fontsize=FL); a1.set_ylim(0, 100); a1.tick_params(labelsize=FS)
    a1.set_title("(a) six-gate judge, GPT-5.5 as judge", fontsize=FL)
    # framed and below the axis title, so it cannot be read as an x-axis label
    leg = a1.legend(fontsize=FS, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2,
                    title="trajectories from", title_fontsize=FS, borderpad=0.5, handlelength=1.6,
                    columnspacing=1.2)
    leg.get_frame().set_linewidth(0.4); leg.get_frame().set_edgecolor("#bdbdbd")
    leg._legend_box.align = "left"
    wid2 = 0.36
    for i, (p, lab, col) in enumerate((("consistency", "consistency prompt", "#bdbdbd"),
                                        ("fidelity", "fidelity prompt", "#2f6db5"))):
        xs = [k + (i - 0.5) * wid2 for k in range(len(LADDER))]
        ys = [100 * d["ladder"][(j, p)] for j, _ in LADDER]
        a2.bar(xs, ys, width=wid2, color=col, label=lab, edgecolor="black", lw=0.3)
        for x, y in zip(xs, ys):
            a2.text(x, y + 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=FS)
    a2.set_xticks(range(len(LADDER)))
    a2.set_xticklabels([l.replace("GPT-5.4-mini", "GPT-5.4-\nmini").replace("Gemini Pro", "Gemini\nPro")
                        for _, l in LADDER], fontsize=FS, linespacing=0.95)
    a2.set_ylim(0, 100); a2.tick_params(labelsize=FS)
    a2.set_title("(b) general-purpose prompts, by judge", fontsize=FL)
    leg2 = a2.legend(fontsize=FS, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1,
                     borderpad=0.5, handlelength=1.6)
    leg2.get_frame().set_linewidth(0.4); leg2.get_frame().set_edgecolor("#bdbdbd")
    # fixed margins: tight_layout would shrink the axes to make room for the legends
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.385, wspace=0.17)
    _save(fig, "fig_judges")


def render_all(num: dict) -> None:
    """Draw all six figures of this module."""
    fig_framework(); fig_trajectory(); fig_regimes(num); fig_safeguards(num)
    fig_guard_flow(num); fig_judges(num)
