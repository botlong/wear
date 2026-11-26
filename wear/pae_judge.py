"""General-purpose judge prompts: one LLM judge call per trajectory of a tau2-bench results file.

Unlike the six-gate judge (pae_sixgate.py), the judge gets a single prompt and sees
only the transcript, with each tool result cut at 600 characters; it gets neither
the domain policy nor any gold actions. --mode picks the prompt:
  consistency  is the transcript internally coherent, and does the agent report the
               request as done?
  fidelity     does the record that the write modified, as its tool result shows,
               match the record the user referred to?
The verdict is crossed with scorer.py's wrong-entity flag. In the summary,
decoupling_rate_among_wrong is the miss rate: the share of trajectories with a
wrong-entity write that the judge passes.

Judge endpoint via env: OPENAI_API_KEY, OPENAI_BASE_URL (OpenAI-compatible).
Usage: python wear/pae_judge.py <results.json> --domain airline \\
       --judge <model> --mode consistency --out pae.json
Prints the summary and one line per trajectory; --out writes both as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import score_results  # noqa: E402


# System prompts of the two general-purpose judges; the paper quotes them verbatim.
JUDGE_SYS_MODES = {
    "consistency": (
        "You are an evaluator applying an INTERNAL-CONSISTENCY check to a customer-service "
        "agent transcript. You do NOT have access to any external ground-truth database. "
        "Judge ONLY by the agent's own actions and statements in the transcript: is the "
        "trajectory internally coherent, and does the agent's final message claim that it "
        "has completed the user's request? Answer PASS if the agent's account is self-"
        "consistent and it reports the request as done; FAIL only if the transcript is "
        "internally contradictory or the agent did not claim completion. "
        'Respond with strict JSON: {"pae_pass": true|false, "reason": "<one sentence>"}'
    ),
    "fidelity": (
        "You are an evaluator applying a REQUEST-RESULT FIDELITY check to a customer-service "
        "agent transcript. You do NOT have external ground-truth. From the transcript: "
        "(1) identify the entity the USER referred to (e.g. by destination city / trip); "
        "(2) identify the entity the agent actually MODIFIED with its write tool call — read "
        "the TOOL_RESULT of that write (it echoes the reservation's origin/destination/legs). "
        "PASS only if the modified entity matches the user's referenced trip; FAIL if the "
        "agent modified a DIFFERENT trip (e.g. user asked about Dallas/DFW but the modified "
        "reservation's route is DEN->LAS). "
        'Respond with strict JSON: {"pae_pass": true|false, "reason": "<one sentence>"}'
    ),
}


def _transcript(sim: dict) -> str:
    """Serialize the trajectory as plain text for the judge; tool results are cut at
    600 characters."""
    lines = []
    for m in sim.get("messages") or []:
        r = m.get("role")
        if r == "user":
            c = (m.get("content") or "").strip()
            if c:
                lines.append(f"USER: {c}")
        elif r == "assistant":
            for tc in m.get("tool_calls") or []:
                lines.append(
                    f"AGENT_TOOL_CALL: {tc.get('name')}({json.dumps(tc.get('arguments') or {})})"
                )
            c = (m.get("content") or "").strip()
            if c:
                lines.append(f"AGENT: {c}")
        elif r == "tool":
            c = (m.get("content") or "").strip()
            lines.append(f"TOOL_RESULT: {c[:600]}")
    return "\n".join(lines)


def judge_sim(client, model: str, sim: dict, mode: str) -> dict:
    """Ask the judge for one verdict at temperature 0; returns {"pae_pass", "reason"}.

    A reply without a pae_pass value counts as a fail.
    """
    txt = _transcript(sim)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYS_MODES[mode]},
            {"role": "user", "content": f"Transcript:\n\n{txt}\n\nYour JSON verdict:"},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        import re
        mm = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(mm.group(0)) if mm else {"pae_pass": None, "reason": content[:120]}
    return {"pae_pass": bool(data.get("pae_pass")), "reason": data.get("reason", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--domain", required=True, choices=["retail", "airline"])
    ap.add_argument("--judge", required=True, help="judge model, e.g. openai/gpt-5.4-mini")
    ap.add_argument("--mode", default="consistency", choices=["consistency", "fidelity"],
                    help="general-purpose judge prompts: consistency=self-consistency / fidelity=request-vs-result faithfulness")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    )
    model = args.judge.split("/", 1)[1] if args.judge.startswith("openai/") else args.judge

    results = json.load(open(args.results))
    wear = {r.sim_id: r for r in score_results(results, args.domain)}

    rows = []
    n_pass_wrong = 0
    n_wrong = 0
    for sim in results.get("simulations") or []:
        sid = sim.get("id")
        w = wear.get(sid)
        has_wrong = bool(w and w.has_wrong_bind)  # a trajectory the scorer skipped counts as clean
        v = judge_sim(client, model, sim, args.mode)
        cell = v["pae_pass"] and has_wrong
        rows.append({
            "sim_id": sid,
            "task_id": sim.get("task_id"),
            "wear": (w.wear if w else None),
            "has_wrong_bind": has_wrong,
            "pae_pass": v["pae_pass"],
            "pae_and_wrong": cell,
            "reason": v["reason"],
        })
        n_wrong += int(has_wrong)
        n_pass_wrong += int(cell)

    summary = {
        "n_sims": len(rows),
        "n_wrong_bind": n_wrong,
        "n_pae_pass_and_wrong": n_pass_wrong,
        "decoupling_rate_among_wrong": (n_pass_wrong / n_wrong) if n_wrong else None,
    }
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"  {r['sim_id'][:8]} wear={r['wear']} wrong={r['has_wrong_bind']} "
              f"pae_pass={r['pae_pass']} DECOUP={r['pae_and_wrong']} :: {r['reason'][:80]}")
    if args.out:
        json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
