# WEAR: code and data

Code and data for the paper *Acting on the Wrong Entity: Measuring and Mitigating
Stale-Identifier Writes in LLM Tool-Use Agents*.

WEAR (Wrong-Entity Action Rate) is the share of an agent's writes, among those matched to
the task's reference solution, whose target identifier differs from the reference. A
perturbation layer runs each task in three conditions that differ only in when the same
change to the records happens: never (baseline), before the conversation (control), or
right after the agent reads the identifier (perturbed).

## Contents

| Path | Contents |
|---|---|
| `wear/perturb.py`, `wear/run_perturbed.py`, `wear/run_batch.py`, `wear/run_safeguards.py` | perturbation layer and run drivers for tau2-bench |
| `wear/analyze_safeguards.py` | safeguard study: analysis |
| `wear/scorer.py`, `wear/scorer_extended.py`, `wear/score_batch.py`, `wear/eligibility.py` | WEAR scorer (static and extended scoring) and the timing check |
| `wear/write_guard.py` | compare-and-set and identity write guards |
| `wear/pae_sixgate.py`, `wear/pae_judge.py`, `wear/run_sixgate_judge_sweep.sh` | six-gate judge, general-purpose judge prompts, and the six-gate sweep |
| `wear/state_check.py`, `wear/expected_hash.py`, `wear/stats.py` | state check, end-state replay, statistics |
| `wear/gen_*.py`, `wear/deictic_tasks/`, `wear/deictic_tasks_heldout/` | task generators and the task suites |
| `wear/mitigation_prompt.txt`, `wear/mitigation_placebo.txt` | verification policy and placebo paragraph |
| `wear/toolsandbox/` | ToolSandbox scenarios, WEAR scorer for ToolSandbox, and driver |
| `wear/annotation/` | human validation: `sample_manifest.json`, `labels_A.json`, `labels_B.json`, `annotation_cards.json`, `analyze_agreement.py` |
| `wear/annotations/` | authorization annotations for extended scoring of the recovery suite |
| `wear/paper/` | `collect_numbers.py`, `NUMBERS.json`, `make_figures.py`, generated tables, tests |
| `wear/runs/` | run files of every reported result (see below) |
| `wear/audit/state_check_20261124/` | per-write output of the state check |
| `patches/` | changes to tau2-bench and ToolSandbox, with the base commits |
| `wear/test_*.py` | tests |

### Run data (`wear/runs/`)

| Directory | Result |
|---|---|
| `main_experiment/` | main experiment: five models, airline and retail, baseline / control / perturbed |
| `append_reorder/` | APPEND and REORDER |
| `cue_present/` | cue-present variant |
| `nodistractor_singlechar/` | no-distractor and single-character variants |
| `recovery_suite/` | recovery suite |
| `toolsandbox/` | ToolSandbox replication: each rollout's `conversation.json`; the one rollout that crashed before saving its conversation keeps ToolSandbox's `result_summary.json` |
| `sixgate_judge/`, `sixgate_gemini_judge/` | six-gate judge outputs (GPT-5.5 and Gemini Pro as judge) |
| `judge_prompts/` | general-purpose judge prompts |
| `policy/`, `placebo/` | verification policy and placebo |
| `safeguard_study/` | safeguard study: no safeguard, policy, CAS guard, identity guard |
| `deepseek_preview/` | only the 25 run files from which the human-validation sample drew its 34 DeepSeek trajectories (an earlier collection on DeepSeek's preview release, retired; it enters no other result) |

All models were reached through a third-party OpenAI-compatible relay service; the drivers
read its endpoint from the `RELAY_*` environment variables. The run files keep the fields
that the analysis reads: the conversation, the tool calls and responses, and the run
outcome. Provider response metadata, timestamps and costs are not included. In the safeguard study, attempts interrupted by an API or transport error
were collected again under a new seed and enter no result; their partial records are not
included (`"failures": []`). In `wear/annotation/`, `annotation_cards.json` keeps the write calls
that the annotators labeled, and the label files omit the annotators' free-text notes.

## Reproducing the reported numbers

Every number in the paper's tables and figures is computed from `wear/runs/` by
`wear/paper/collect_numbers.py`, which writes `wear/paper/NUMBERS.json`;
`make_figures.py` renders the figures and generated tables from that file.

```bash
# tau2-bench at the base commit, with the patch applied
git clone https://github.com/sierra-research/tau2-bench.git
git -C tau2-bench checkout $(awk '/tau2-bench/ {print $2}' patches/BASE_COMMITS.txt)
git -C tau2-bench apply ../patches/tau2-bench.patch
cd tau2-bench && python -m venv .venv && .venv/bin/pip install -e . && cd ..

tau2-bench/.venv/bin/python wear/paper/collect_numbers.py      # rewrites NUMBERS.json
tau2-bench/.venv/bin/python wear/paper/make_figures.py         # figures and tables
tau2-bench/.venv/bin/python -m pytest wear --ignore=wear/toolsandbox/test_binding_scenarios.py
```

The human-validation agreement statistics are printed by

```bash
tau2-bench/.venv/bin/python wear/annotation/analyze_agreement.py \
    wear/annotation/labels_A.json wear/annotation/labels_B.json
```

## Running new rollouts

Set `RELAY_BASE_URL` and the API keys the drivers read (`RELAY_API_KEY`, and for the shell
drivers `RELAY_GPT_API_KEY` and `RELAY_GEMINI_API_KEY`), then run, for example,

```bash
tau2-bench/.venv/bin/python wear/run_batch.py --outdir wear/runs/new \
    --models openai/gpt-5.5 --conditions baseline control perturbed --k 5
```

Both Gemini models run at a low reasoning effort, which `run_batch.py` sets. In the safeguard
study GPT-5.5 runs with its reasoning effort set to `none`:

```bash
tau2-bench/.venv/bin/python wear/run_safeguards.py --outroot wear/runs/new_safeguards \
    --models openai/gpt-5.5 openai/deepseek-v4-pro openai/qwen3.7-plus \
    --route openai/gpt-5.5=RELAY openai/deepseek-v4-pro=RELAY openai/qwen3.7-plus=RELAY \
    --model-kwargs 'openai/gpt-5.5={"reasoning_effort": "none"}'
```

ToolSandbox runs need ToolSandbox cloned into `./ToolSandbox` at its base commit, with
`patches/ToolSandbox.patch` applied and installed in `ToolSandbox/.venv`;
`wear/toolsandbox/step6_run.sh` writes the runs to `wear/runs/toolsandbox/`. The scenario
test `wear/toolsandbox/test_binding_scenarios.py` also needs this installation.

## License

Code: MIT (see `LICENSE`). Data (`wear/runs/`, the task suites, and `wear/annotation/`):
CC BY 4.0. tau2-bench and ToolSandbox keep their own licenses; the task suites are built
from tau2-bench tasks.
