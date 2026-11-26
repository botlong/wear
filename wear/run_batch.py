#!/usr/bin/env python3
"""Batch driver for tau2-bench collections such as the main experiment.

A unit is one task under one condition for one model. For every unit this script
runs wear/run_perturbed.py in a subprocess until the unit file holds k rollouts.
The three conditions differ only in when the same change to the records happens:
  baseline   no change;
  control    the change is applied before the conversation starts (the task's
             *_control.json file, whose schedule sets fire_at_start);
  perturbed  the change is applied right after the agent's binding read.

Reads: the ID-REUSE task files wear/deictic_tasks/{airline,retail}_idreuse_*.json
(or the files given with --task-files) and, for the control condition, their
*_control.json versions.
Writes: <outdir>/<model slug>/<condition>/<task>.json with a .log next to it;
failed units are appended to <outdir>/failures.txt.

The script can be re-run on the same --outdir: completed units are skipped and
incomplete ones are topped up with seeds that no earlier attempt used. Each unit
file records the resolved, key-free configuration of its rollouts (`_config`), and a
unit is not resumed under a different configuration. Models are reached through the
relay, an OpenAI-compatible service whose base URL and keys are read from RELAY_*
environment variables (see README); --route MODEL=PREFIX reads
<PREFIX>_BASE_URL and <PREFIX>_API_KEY instead.

Usage: python3 wear/run_batch.py --outdir OUT --models MODEL... [--conditions ...] [--k N]
       [--route MODEL=PREFIX ...] [--write-guard cas|identity] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNNER = os.path.join(HERE, "run_perturbed.py")
PYTHON = os.environ.get("WEAR_PYTHON") or os.path.join(REPO, "tau2-bench", ".venv", "bin", "python")

CONDITIONS = ("baseline", "perturbed", "control")


def suite_tasks(domain: str) -> list[str]:
    """Return the domain's base task files (control and variant files excluded)."""
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "deictic_tasks",
                                           f"{domain}_idreuse_*.json"))):
        if any(s in os.path.basename(p) for s in
               ("_control", "_signal", "_distB", "_distC", "_simnew")):
            continue
        out.append(p)
    return out


def _mslug(model: str) -> str:
    """Sanitize a model string into a filesystem-safe slug."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def unit_spec(domain: str, task_path: str, condition: str, model: str,
              outdir: str, k: int, agent_suffix_file: str | None = None,
              write_guard: str | None = None, sim_timeout: int | None = None) -> dict:
    """Describe one unit as a dict: task file, runner arm, model, output path, k and options.

    The control condition runs the runner's perturbed arm on the task's control file.
    """
    base = os.path.basename(task_path).replace(".json", "")
    if condition == "control":
        task_path = task_path.replace(".json", "_control.json")
    arm = "baseline" if condition == "baseline" else "perturbed"
    mslug = _mslug(model)
    out = os.path.join(outdir, mslug, condition, f"{base}.json")
    return {"domain": domain, "task_path": task_path, "condition": condition,
            "arm": arm, "model": model, "out": out, "k": k,
            "agent_suffix_file": agent_suffix_file, "write_guard": write_guard,
            "sim_timeout": sim_timeout, "name":
            f"{mslug}/{condition}/{base}"}


def n_sims(path: str) -> int:
    """Number of completed rollouts in a unit file; 0 if it is missing or unreadable."""
    try:
        d = json.load(open(path))
        return len(d.get("simulations") or [])
    except Exception:
        return 0


def done(u: dict) -> bool:
    """True if the unit file already holds at least k completed rollouts."""
    return n_sims(u["out"]) >= u["k"]


def attempted_seeds(path: str) -> set[int]:
    """Return the seeds of every attempt recorded for a unit.

    This covers completed rollouts, failure records, rollouts moved aside to
    <unit>.quarantine.json, and the seeds listed as started.
    """
    seeds: set[int] = set()
    for p in (path, path.replace(".json", ".quarantine.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for key in ("simulations", "failures"):
            for s in d.get(key) or []:
                if s.get("seed") is not None:
                    seeds.add(int(s["seed"]))
        seeds.update(int(x) for x in d.get("started") or [])  # includes a rollout killed in flight
    return seeds


def next_seed(path: str) -> int:
    """Return the first seed for a unit's next attempts.

    It lies above every recorded attempt, so no seed is ever reused; a fresh unit
    starts at BASE_SEED.
    """
    used = attempted_seeds(path)
    return max(used) + 1 if used else BASE_SEED


def _strip_code(cfg):
    """Drop the `code` entry (git commit) so configurations from different commits compare equal."""
    return None if cfg is None else {k: v for k, v in cfg.items() if k != "code"}


def merge_part(out: str, part: str) -> None:
    """Append a top-up part's rollouts, failure records and started seeds to the unit file.

    The merge is atomic and idempotent: an entry already present (for example after a
    merge interrupted before the part was removed) is not added twice. A part
    collected under another configuration is renamed to <part>.conflict and
    RuntimeError is raised; it is never merged.
    """
    old = json.load(open(out))
    new = json.load(open(part))
    if old.get("_config") is not None and new.get("_config") is not None \
            and _strip_code(old["_config"]) != _strip_code(new["_config"]):
        os.replace(part, part + ".conflict")
        raise RuntimeError(f"{part}: configuration differs from {out}; set aside")
    skey = lambda s: s.get("id") or ("seed", s.get("seed"), s.get("task_id"))  # noqa: E731
    sim_ids = {skey(s) for s in old.get("simulations") or []}
    old["simulations"] = (old.get("simulations") or []) + [
        s for s in new.get("simulations") or [] if skey(s) not in sim_ids]
    fkey = lambda f: (f.get("seed"), f.get("_failure"))  # noqa: E731
    have_f = {fkey(f) for f in old.get("failures") or []}
    old["failures"] = (old.get("failures") or []) + [
        f for f in new.get("failures") or [] if fkey(f) not in have_f]
    old["started"] = sorted(set(old.get("started") or []) | set(new.get("started") or []))
    if not old.get("tasks"):
        old["tasks"] = new.get("tasks")
    if "_config" in new:
        old.setdefault("_config", new["_config"])
    with open(out + ".merge", "w") as f:
        json.dump(old, f, default=str)
    os.replace(out + ".merge", out)
    os.remove(part)


BASE_SEED = 300


GIT_HEAD = ""  # set in main()
ROUTES: dict[str, str] = {}  # model -> environment-variable prefix, from --route
MODEL_KWARGS: dict[str, dict] = {}  # model -> extra litellm kwargs, from --model-kwargs
# Seconds per LLM request on routed models; the observed p99 latency was at most 23 s.
REQUEST_TIMEOUT_S = 180


def _endpoint_for(model: str) -> tuple[str | None, str | None, str, str, dict]:
    """Resolve how a model is served.

    Returns (API key, base URL ending in /v1, name of the key variable, name of the
    base-URL variable, extra TAU2_LLM_* settings). The key or URL is None when its
    variable is unset.
    """
    name = model.split("/", 1)[-1].lower()
    extra: dict = {}
    if model in ROUTES:
        # Routed model: endpoint from <PREFIX>_BASE_URL / <PREFIX>_API_KEY, with the
        # same max_tokens default as the default route at the end.
        kn, bn = f"{ROUTES[model]}_API_KEY", f"{ROUTES[model]}_BASE_URL"
        per: dict = {}
        if name.startswith("gpt"):
            # GPT: no decoding kwargs are injected, as on the unrouted GPT route
            # below; only a per-request timeout.
            extra = {"TAU2_LLM_EXTRA_KWARGS": json.dumps({"timeout": REQUEST_TIMEOUT_S})}
        else:
            extra = {"TAU2_LLM_EXTRA_KWARGS": json.dumps(
                {"max_tokens": 8192, "timeout": REQUEST_TIMEOUT_S})}
        if name.startswith("gemini"):
            # Gemini: low reasoning effort and a 16384-token cap, as on the unrouted
            # Gemini route below. LiteLLM drops reasoning_effort for openai/gemini-*
            # unless it is listed in allowed_openai_params.
            per = {"reasoning_effort": "low", "max_tokens": 16384,
                   "allowed_openai_params": ["reasoning_effort"]}
        per.update(MODEL_KWARGS.get(model, {}))  # --model-kwargs override the defaults
        if per:
            extra["TAU2_LLM_PER_MODEL_KWARGS"] = json.dumps({model: per})
    elif name.startswith("gemini"):
        # Unrouted Gemini: the model's own calls use the relay's Gemini key, passed
        # with its base URL as per-model kwargs (api_base, api_key), at most one
        # request every 7 s per process. The default key and URL set below serve
        # the auxiliary GPT-5.4-mini calls.
        kn, bn = "RELAY_API_KEY", "RELAY_BASE_URL"
        xk = os.environ.get("RELAY_GEMINI_API_KEY")
        xb = os.environ.get("RELAY_BASE_URL", "").rstrip("/")
        if not xk or not xb:
            miss = "RELAY_GEMINI_API_KEY" if not xk else "RELAY_BASE_URL"
            return None, None, miss, miss, {}
        extra = {
            "TAU2_LLM_EXTRA_KWARGS": '{"max_tokens": 8192}',
            "TAU2_LLM_PER_MODEL_KWARGS": json.dumps({model: {
                "api_base": xb + "/v1", "api_key": xk,
                "reasoning_effort": "low", "max_tokens": 16384}}),
            "TAU2_LLM_MIN_INTERVAL_S": json.dumps({model: 7}),
        }
    elif name.startswith("gpt"):
        # Unrouted GPT: the relay's GPT key. This endpoint accepts seed and
        # non-streamed tool calls, so no extra kwargs are injected.
        kn, bn = "RELAY_GPT_API_KEY", "RELAY_BASE_URL"
    else:
        # Default route: the relay's default key, with a max_tokens cap.
        kn, bn = "RELAY_API_KEY", "RELAY_BASE_URL"
        extra = {"TAU2_LLM_EXTRA_KWARGS": '{"max_tokens": 8192}'}
    base = os.environ.get(bn, "").rstrip("/")
    if base and not base.endswith("/v1"):
        base += "/v1"
    return os.environ.get(kn), (base or None), kn, bn, extra


def _git_head() -> str:
    """Current commit of the repository, or '' if git is unavailable."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _sha256(path: str | None) -> str | None:
    """SHA-256 of a file's bytes, or None when no file is given."""
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if path else None


def unit_config(u: dict, base_url: str, extra_env: dict) -> dict:
    """Return the resolved configuration of a unit, without any API key.

    It holds the model, route, endpoint host, TAU2_LLM_* settings, condition, write
    guard, rollout timeout and hashes of the task and policy-suffix files. Later
    rollouts of the unit must be collected under the same configuration.
    """
    from urllib.parse import urlparse
    kw = {k: json.loads(v) for k, v in extra_env.items()}
    for m in (kw.get("TAU2_LLM_PER_MODEL_KWARGS") or {}).values():
        m.pop("api_key", None)
    return {"model": u["model"], "route": ROUTES.get(u["model"]),
            "base_host": urlparse(base_url).netloc, "llm_env": kw,
            "condition": u["condition"], "task_sha256": _sha256(u["task_path"]),
            "agent_suffix_sha256": _sha256(u.get("agent_suffix_file")),
            "write_guard": u.get("write_guard"), "sim_timeout": u.get("sim_timeout")}


def recover_parts(out: str) -> int:
    """Merge the top-up parts (<out>.part*) that a killed driver left behind.

    They hold checkpointed rollouts. An unreadable part is renamed to .unreadable,
    never silently dropped. Returns the number of parts merged or moved into place.
    """
    n = 0
    for part in sorted(glob.glob(out + ".part*")):
        if part.endswith(".tmp"):
            continue
        try:
            json.load(open(part))
        except Exception:
            os.replace(part, part + ".unreadable")
            continue
        if not os.path.exists(out):
            os.replace(part, out)
        else:
            merge_part(out, part)
        n += 1
    return n


def resolve(u: dict):
    """Build the environment of a unit's runner process.

    The child starts from the parent environment minus every inherited TAU2_LLM_*
    variable, so only the settings resolved here reach the runner. Returns
    (env, config, None), or (None, None, error) when a key or base URL is missing.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TAU2_LLM_")}
    key, base_url, keyname, basename, extra_env = _endpoint_for(u["model"])
    if not key or not base_url:
        return None, None, f"missing {keyname if not key else basename} in env (see README)"
    env.update(extra_env)
    env.update({"OPENAI_API_KEY": key, "OPENAI_BASE_URL": base_url, "OPENAI_API_BASE": base_url,
                "TAU2_LLM_NL_ASSERTIONS": "openai/gpt-5.4-mini",
                "TAU2_LLM_ENV_INTERFACE": "openai/gpt-5.4-mini"})
    return env, unit_config(u, base_url, extra_env), None


def check_config(u: dict) -> str | None:
    """Return an error message if the unit's existing rollouts cannot be resumed, else None.

    Leftover top-up parts are merged first. Resuming is refused when the stored
    configuration differs from the current one or when rollouts carry none.
    """
    try:
        recover_parts(u["out"])
    except RuntimeError as e:
        return str(e)
    if not os.path.exists(u["out"]):
        return None
    _, cfg, err = resolve(u)
    if err:
        return err
    d = json.load(open(u["out"]))
    prev = d.get("_config")
    if prev is None:
        if d.get("simulations") or d.get("failures"):
            return f"{u['name']}: existing rollouts carry no _config (unverifiable)"
        return None
    if _strip_code(prev) != cfg:
        return f"{u['name']}: config differs from the unit's earlier rollouts"
    return None


def run_unit(u: dict, timeout_s: int) -> tuple[str, bool, str]:
    """Collect the missing rollouts of one unit with run_perturbed.py.

    A unit that already has rollouts is topped up into <out>.part<n>, which is merged
    back even when the run is cut short. Returns (unit name, success, error message).
    """
    err = check_config(u)
    if err:
        return u["name"], False, err
    env, cfg, err = resolve(u)
    if err:
        return u["name"], False, err
    os.makedirs(os.path.dirname(u["out"]), exist_ok=True)
    log = u["out"].replace(".json", ".log")

    have = n_sims(u["out"])
    need = u["k"] - have
    if need <= 0:
        return u["name"], True, ""
    exists = os.path.exists(u["out"])
    env["WEAR_RUN_CONFIG"] = json.dumps(dict(cfg, code=GIT_HEAD))
    part = u["out"] + f".part{have}"
    cmd = [PYTHON, RUNNER, "--domain", u["domain"], "--tasks", u["task_path"],
           "--arm", u["arm"], "--model", u["model"], "--k", str(need),
           "--seed", str(next_seed(u["out"])),
           "--out", part if exists else u["out"]]
    if u.get("agent_suffix_file"):
        cmd += ["--agent-policy-suffix-file", u["agent_suffix_file"]]
    if u.get("write_guard"):
        cmd += ["--write-guard", u["write_guard"]]
    if u.get("sim_timeout"):
        cmd += ["--timeout", str(u["sim_timeout"])]
    try:
        with open(log, "a") as lf:
            p = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=lf,
                               timeout=timeout_s)
        if p.returncode != 0:
            return u["name"], False, f"exit {p.returncode} (log: {log})"
    except subprocess.TimeoutExpired:
        return u["name"], False, f"unit timeout {timeout_s}s"
    except Exception as e:
        return u["name"], False, f"{type(e).__name__}: {e}"
    finally:
        # The runner checkpoints after every rollout: keep whatever it finished,
        # even when the unit was cut short.
        if exists and os.path.exists(part):
            try:
                merge_part(u["out"], part)
            except Exception as e:  # set aside or unreadable: never lose it silently
                print(f"[merge] {u['name']}: {e}", file=sys.stderr)
    if not done(u):
        return u["name"], False, f"output incomplete (<k sims; log: {log})"
    return u["name"], True, ""


def main() -> None:
    """Plan the units, check existing outputs against the current configuration, and run the pending units."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--domains", nargs="+", default=["airline", "retail"],
                    choices=["airline", "retail"])
    ap.add_argument("--task-filter", default=None,
                    help="substring filter on task filenames (e.g. boston)")
    ap.add_argument("--task-files", nargs="+", default=None,
                    help="explicit task files (overrides the idreuse suite glob; "
                         "domain inferred from the airline_/retail_ filename prefix)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--unit-timeout", type=int, default=3600,
                    help="wall-clock limit per unit (one task x one condition x k rollouts), seconds")
    ap.add_argument("--agent-suffix-file", default=None,
                    help="text file appended to the agent domain policy (mitigation runs; use a dedicated --outdir)")
    ap.add_argument("--write-guard", nargs="?", const="cas", default=None,
                    choices=["cas", "identity"],
                    help="tool-layer write guard, cas or identity (use a dedicated --outdir)")
    ap.add_argument("--route", nargs="+", default=[], metavar="MODEL=PREFIX",
                    help="serve MODEL via <PREFIX>_BASE_URL / <PREFIX>_API_KEY (e.g. openai/deepseek-v4-pro=DEEPSEEK)")
    ap.add_argument("--model-kwargs", nargs="+", default=[], metavar="MODEL=JSON",
                    help="extra decoding kwargs for a routed MODEL "
                         "(e.g. 'openai/gpt-5.5={\"reasoning_effort\": \"low\"}')")
    ap.add_argument("--sim-timeout", type=int, default=None,
                    help="wall-clock budget per rollout in seconds (runner default 300)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global GIT_HEAD
    GIT_HEAD = _git_head()
    for r in args.model_kwargs:
        m, _, js = r.partition("=")
        MODEL_KWARGS[m] = json.loads(js)
    for r in args.route:
        m, _, prefix = r.partition("=")
        assert m and prefix, f"bad --route {r!r}"
        ROUTES[m] = prefix
    if args.agent_suffix_file:
        assert os.path.isfile(args.agent_suffix_file), args.agent_suffix_file

    domain_tasks: list[tuple[str, str]] = []
    if args.task_files:
        for tp in args.task_files:
            domain = os.path.basename(tp).split("_", 1)[0]
            assert domain in ("airline", "retail"), f"can't infer domain: {tp}"
            domain_tasks.append((domain, tp))
    else:
        for domain in args.domains:
            for tp in suite_tasks(domain):
                domain_tasks.append((domain, tp))

    units: list[dict] = []
    for domain, tp in domain_tasks:
        if args.task_filter and args.task_filter not in os.path.basename(tp):
            continue
        for cond in args.conditions:
            if cond == "control" and not os.path.exists(
                    tp.replace(".json", "_control.json")):
                print(f"[plan] SKIP control (no twin): {os.path.basename(tp)}",
                      file=sys.stderr)
                continue
            for m in args.models:
                units.append(unit_spec(domain, tp, cond, m, args.outdir, args.k,
                                       args.agent_suffix_file, args.write_guard,
                                       args.sim_timeout))

    bad = [e for e in (check_config(u) for u in units) if e]
    if bad:
        raise SystemExit("config check failed:\n  " + "\n  ".join(bad[:10]))
    pending = [u for u in units if not done(u)]
    print(f"[plan] units total={len(units)} pending={len(pending)} "
          f"(k={args.k}, ~{len(pending) * args.k} rollouts)")
    if args.dry_run:
        for u in pending:
            print("  ", u["name"])
        return

    os.makedirs(args.outdir, exist_ok=True)
    failures: list[str] = []
    ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_unit, u, args.unit_timeout): u for u in pending}
        for f in as_completed(futs):
            name, success, err = f.result()
            if success:
                ok += 1
                print(f"[done {ok}/{len(pending)}] {name}")
            else:
                failures.append(f"{name}: {err}")
                print(f"[FAIL] {name}: {err}", file=sys.stderr)
    if failures:
        fp = os.path.join(args.outdir, "failures.txt")
        with open(fp, "a") as f:
            f.write("\n".join(failures) + "\n")
        print(f"[summary] ok={ok} FAILED={len(failures)} -> {fp}", file=sys.stderr)
        sys.exit(1)
    print(f"[summary] ok={ok} failed=0")


if __name__ == "__main__":
    main()
