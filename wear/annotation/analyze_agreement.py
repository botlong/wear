#!/usr/bin/env python3
"""Human validation of the WEAR scorer: inter-annotator and scorer-vs-human agreement per write.

Inputs: the two annotators' label files (one label per write: right, wrong, repair
or unsure); the sample manifest (sample_manifest.json: each sampled trajectory with its
source results file under wear/, stratum, agent and stored scorer counts); and the
annotation cards (annotation_cards.json), whose write entries are checked against the
re-scored writes. Every write is re-scored from its source trajectory, and the
script asserts that write order, error flags and wrong-write counts agree with the
cards and the manifest.

Prints Cohen's kappa between the annotators (four classes, and wrong vs other) with
the confusion matrix; the scorer's sensitivity and specificity against each
annotator and against the writes on which both agree, with Wilson intervals (repairs
excluded, then counted as not wrong); repair counts; effective vs refused writes
among those both annotators mark wrong; trajectory-level agreement; counts per
stratum; and every disagreement. --out writes the per-write rows as JSONL.

Usage: python wear/annotation/analyze_agreement.py wear/annotation/labels_A.json wear/annotation/labels_B.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEAR = HERE.parent
sys.path.insert(0, str(WEAR))
from scorer import (NAME_ONLY, _entity_keys, _signature, extract_agent_write_calls,  # noqa: E402
                    extract_gold_write_actions, load_write_tools)

LABELS = ["right", "wrong", "repair", "unsure"]


def per_call_wear(sim, task, write_tools, oid_keys):
    """Classify each agent write of one trajectory with the scorer's matching rules.

    Returns one (kind, errored) pair per write, kind being spurious, correct, wrong
    or no_id. Unlike scorer.score_simulation, a tool with several gold writes is
    matched against any of them; no task in the human-validation sample has such
    a tool, so the two agree on it.
    """
    golds = extract_gold_write_actions(task, write_tools)
    by = defaultdict(list)
    for g in golds:
        by[g.name].append(g)
    out = []
    for c in extract_agent_write_calls(sim.get("messages") or [], write_tools):
        cands = by.get(c.name)
        if not cands:
            out.append(("spurious", c.errored)); continue
        matched = assessable = False
        for g in cands:
            keys = _entity_keys(g.arguments, oid_keys, g.compare_args)
            if keys is NAME_ONLY:
                matched = True; break
            if not keys:
                continue
            assessable = True
            if _signature(c.arguments, keys) == _signature(g.arguments, keys):
                matched = True; break
        out.append(("correct" if matched else "wrong" if assessable else "no_id", c.errored))
    return out


def kappa(a, b, cats):
    """Return (raw agreement, Cohen's kappa) of two label lists over categories cats."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in cats) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k, n, z=1.96):
    """Return (k/n, lo, hi) with the 95% Wilson score interval; NaNs if n is 0."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fmt(k, n):
    """Format k/n with its Wilson interval."""
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs=2)
    ap.add_argument("--manifest", default=str(WEAR / "annotation/sample_manifest.json"))
    ap.add_argument("--cards", default=str(HERE / "annotation_cards.json"))
    ap.add_argument("--out", default=None, help="write per-call rows as JSONL")
    args = ap.parse_args()

    wt = load_write_tools()
    man = json.load(open(args.manifest))["items"]
    cards = {c["card_id"]: c for c in json.load(open(args.cards))["cards"]}
    ann = [json.load(open(f)) for f in args.labels]
    names = [a["annotator"] for a in ann]

    rows, cache = [], {}
    for it in man:
        res = cache.setdefault(it["file"], json.load(open(WEAR / it["file"])))
        sim = next(s for s in res["simulations"] if s["id"] == it["sim_id"])
        task = next(t for t in res["tasks"] if t["id"] == it["task_id"])
        dom = it["domain"]
        calls = per_call_wear(sim, task, set(wt[dom]), set(wt["_object_id_keys"][dom]))
        writes = [e for e in cards[it["card_id"]]["timeline"] if e.get("write")]
        assert len(calls) == len(writes), it["card_id"]
        assert sum(k == "wrong" for k, _ in calls) == it["wear"]["n_wrong_entity"], it["card_id"]
        for i, ((kind, err), e) in enumerate(zip(calls, writes), 1):
            assert e["error"] == err and e["write_no"] == i
            rows.append({"card_id": it["card_id"], "stratum": it["stratum"], "agent": it["agent"],
                         "write_no": i, "tool": e["name"], "errored": err, "wear": kind,
                         "h": [a["cards"][it["card_id"]]["w"][str(i)] for a in ann],
                         "notes": [a["cards"][it["card_id"]].get("note", "") for a in ann]})

    n = len(rows)
    print(f"annotators: {names}; cards {len(man)}; write calls {n}")
    print("scorer kinds:", dict(Counter(r["wear"] for r in rows)))
    for j, nm in enumerate(names):
        print(f"{nm} labels:", dict(Counter(r["h"][j] for r in rows)))

    print("\n## Inter-annotator")
    a, b = [r["h"][0] for r in rows], [r["h"][1] for r in rows]
    po, k4 = kappa(a, b, LABELS)
    print(f"4-class: raw agreement {po:.3f}, Cohen's kappa {k4:.3f}")
    bw = lambda x: "wrong" if x == "wrong" else "other"
    po2, k2 = kappa([bw(x) for x in a], [bw(x) for x in b], ["wrong", "other"])
    print(f"wrong vs other: raw agreement {po2:.3f}, Cohen's kappa {k2:.3f}")
    print("confusion (A rows, B cols):")
    cm = Counter(zip(a, b))
    print("        " + " ".join(f"{x:>7}" for x in LABELS))
    for x in LABELS:
        print(f"{x:>7} " + " ".join(f"{cm[(x, y)]:>7}" for y in LABELS))

    def vs_scorer(sel, title, merge_repair):
        """Print the scorer's sensitivity and specificity against the human label in
        r["hh"]; unsure is skipped, and repair is skipped unless merge_repair
        counts it as not wrong."""
        print(f"\n### {title}")
        tp = fn = fp = tn = 0
        for r in sel:
            h = r["hh"]
            if h == "unsure" or (h == "repair" and not merge_repair):
                continue
            hw = h == "wrong"
            sw = r["wear"] == "wrong"
            tp += hw and sw; fn += hw and not sw; fp += (not hw) and sw; tn += (not hw) and not sw
        print(f"  sensitivity {fmt(tp, tp + fn)}   specificity {fmt(tn, tn + fp)}   "
              f"(TP {tp} FN {fn} FP {fp} TN {tn})")

    print("\n## Scorer vs human, per write call (human = reference)")
    views = [(names[0], lambda r: r["h"][0]), (names[1], lambda r: r["h"][1]),
             ("both agree", lambda r: r["h"][0] if r["h"][0] == r["h"][1] else None)]
    for nm, f in views:
        sel = [dict(r, hh=f(r)) for r in rows if f(r) is not None]
        vs_scorer(sel, f"{nm} (n={len(sel)}), primary: repair excluded", False)
        vs_scorer(sel, f"{nm}, merged: repair counted as not-wrong", True)

    print("\n## Repair-labelled calls (scorer over-count on authorised fixes)")
    for j, nm in enumerate(names):
        rep = [r for r in rows if r["h"][j] == "repair"]
        print(f"  {nm}: {len(rep)} repair; scorer marks wrong on {sum(r['wear'] == 'wrong' for r in rep)}")

    print("\n## Effective vs refused (calls both annotators label wrong)")
    ww = [r for r in rows if r["h"] == ["wrong", "wrong"]]
    print(f"  {len(ww)} calls: effective {sum(not r['errored'] for r in ww)}, refused {sum(r['errored'] for r in ww)}; "
          f"scorer wrong on {sum(r['wear'] == 'wrong' for r in ww)}")

    print("\n## Trajectory level (any human 'wrong' vs has_wrong_bind)")
    for j, nm in enumerate(names):
        cm2 = Counter()
        for it in man:
            hs = [r for r in rows if r["card_id"] == it["card_id"]]
            cm2[(any(r["h"][j] == "wrong" for r in hs), it["wear"]["has_wrong_bind"])] += 1
        print(f"  {nm}: both wrong {cm2[(True, True)]}, both clean {cm2[(False, False)]}, "
              f"human-only {cm2[(True, False)]}, scorer-only {cm2[(False, True)]}")

    print("\n## By stratum (calls; scorer wrong / A wrong / B wrong / A repair / B repair)")
    for s in sorted({r["stratum"] for r in rows}):
        sel = [r for r in rows if r["stratum"] == s]
        print(f"  {s:20s} n={len(sel):3d}  {sum(r['wear'] == 'wrong' for r in sel):3d} "
              f"{sum(r['h'][0] == 'wrong' for r in sel):3d} {sum(r['h'][1] == 'wrong' for r in sel):3d} "
              f"{sum(r['h'][0] == 'repair' for r in sel):3d} {sum(r['h'][1] == 'repair' for r in sel):3d}")

    print("\n## Disagreements")
    for r in rows:
        hw = ["wrong" if x == "wrong" else x for x in r["h"]]
        scorer_h = "wrong" if r["wear"] == "wrong" else "right"
        if r["h"][0] != r["h"][1] or any(x != scorer_h and x != "repair" for x in r["h"]):
            print(f"  {r['card_id']} #{r['write_no']} {r['stratum']:18s} {r['tool']:32s} "
                  f"err={r['errored']!s:5s} scorer={r['wear']:8s} {names[0]}={r['h'][0]:6s} {names[1]}={r['h'][1]}")

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
