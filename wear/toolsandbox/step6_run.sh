#!/usr/bin/env bash
# Run the ToolSandbox replication: K rollouts of each selected scenario with one
# model, then print, for every rollout, WEAR (from score_wear_ts.py) and
# ToolSandbox's own milestone and minefield scores.
#
# Usage: bash wear/toolsandbox/step6_run.sh [MODEL] [K] [SET]
#   MODEL  model name sent to the API (default gpt-5.5); it plays both the agent
#          and the user simulator.
#   K      rollouts per scenario (default 3).
#   SET    all (default): the 8 scenarios of coreference_scenarios.py;
#          static: the 3 static scenarios;
#          idreuse: the 5 scenarios that apply ID-REUSE;
#          new: the reminder, swapped-pair, cue-present and no-distractor
#          scenarios plus the static reminder scenario.
# Requires: ToolSandbox cloned at ./ToolSandbox, with the ToolSandbox patch
#   applied, coreference_scenarios.py and stale_contact_tool.py in place, and
#   ToolSandbox installed in ToolSandbox/.venv (pip install -e .).
# Input: ENV_FILE, a shell file (default ~/.wear_credentials.env) that exports
#   RELAY_BASE_URL (base URL of an OpenAI-compatible API, without /v1),
#   RELAY_GPT_API_KEY and RELAY_GEMINI_API_KEY.
# Output: one ToolSandbox output directory per rollout,
#   wear/runs/toolsandbox/<MODEL>/<scenario>_run<i>/ (with conversation.json and
#   result_summary.json), and the per-rollout summary on stdout.
set -euo pipefail

EB=$(cd "$(dirname "$0")/../.." && pwd)   # repo root
TS=$EB/ToolSandbox
VENV=$TS/.venv/bin
MODEL="${1:-gpt-5.5}"
K="${2:-3}"
WHICH="${3:-all}"
OUT_BASE=$EB/wear/runs/toolsandbox/$MODEL

# SCENS[i] is scored against GOLDS[i], the identifier of its target record (the one
# its milestone names); identifiers are ToolSandbox deterministic_uuid values.
# Both static contact scenarios target deterministic_uuid of John Myung.
SCENS=()
GOLDS=()
if [[ "$WHICH" == "all" || "$WHICH" == "static" ]]; then
  SCENS+=("modify_contact_disambiguation_binding_attack" "modify_contact_coreference_stale_handle_binding_attack")
  GOLDS+=("c1b6eeb7-144c-5c70-9989-a4379f015ee6" "c1b6eeb7-144c-5c70-9989-a4379f015ee6")
  SCENS+=("modify_reminder_disambiguation_static_control")
  GOLDS+=("da2b856e-9390-511f-bcbe-7005f72cbf5e")  # deterministic_uuid("reminder_1")
fi
if [[ "$WHICH" == "all" || "$WHICH" == "idreuse" ]]; then
  SCENS+=("modify_contact_idreuse_stale_cache_binding_attack")
  GOLDS+=("cf1697b7-bc76-572c-8927-e3e9268cce9f")  # deterministic_uuid("Fredrik Thordendal [reissued]")
fi
if [[ "$WHICH" == "all" || "$WHICH" == "idreuse" || "$WHICH" == "new" ]]; then
  SCENS+=("modify_reminder_idreuse_stale_cache_binding_attack")
  GOLDS+=("38e978a9-0a6b-5837-a86a-4bdd41b7debe")  # deterministic_uuid("reminder_2 [reissued]")
  SCENS+=("modify_contact_idreuse_swapped_pair_binding_attack")
  GOLDS+=("1b310f28-cb61-58fc-ab99-3725c005326d")  # deterministic_uuid("John Petrucci [reissued]")
  SCENS+=("modify_contact_idreuse_cue_present_binding_attack")
  GOLDS+=("cf1697b7-bc76-572c-8927-e3e9268cce9f")  # deterministic_uuid("Fredrik Thordendal [reissued]")
  SCENS+=("modify_contact_idreuse_no_distractor_binding_attack")
  GOLDS+=("14e2f13a-548f-5e25-8b7d-65c86c5d8abb")  # deterministic_uuid("Homer S [reissued]")
fi
if [[ "$WHICH" == "new" ]]; then
  SCENS+=("modify_reminder_disambiguation_static_control")
  GOLDS+=("da2b856e-9390-511f-bcbe-7005f72cbf5e")
fi

# Check that ToolSandbox and its dependencies import.
"$VENV/python" - <<'PY'
import tool_sandbox, polars, dill  # noqa
print("[step6] deps OK")
PY

# Build the scenarios offline (no API calls): all 8 must exist, each with one
# milestone and one minefield, and the perturbed lookup tool must be registered.
"$VENV/python" - <<'PY'
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios.coreference_scenarios import named_binding_attack_scenarios
scen = named_binding_attack_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
expect = {
    "modify_contact_disambiguation_binding_attack",
    "modify_contact_coreference_stale_handle_binding_attack",
    "modify_contact_idreuse_stale_cache_binding_attack",
    "modify_reminder_idreuse_stale_cache_binding_attack",
    "modify_reminder_disambiguation_static_control",
    "modify_contact_idreuse_swapped_pair_binding_attack",
    "modify_contact_idreuse_cue_present_binding_attack",
    "modify_contact_idreuse_no_distractor_binding_attack",
}
assert set(scen) == expect, f"unexpected names: {sorted(scen)}"
for name, s in scen.items():
    ms = s.evaluation.milestone_matcher.milestones
    mf = s.evaluation.minefield_matcher.milestones
    assert len(ms) == 1 and len(mf) == 1, (name, len(ms), len(mf))
print("[step6] offline build OK:", ", ".join(sorted(scen)))
from tool_sandbox.scenarios import named_scenarios
allscen = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
mine = sorted(n for n in allscen if "binding_attack" in n)
print(f"[step6] named_scenarios OK: {len(allscen)} total, base-name ours: {[m for m in mine if m in expect]}")
# Is the perturbed lookup tool discovered as a ToolSandbox tool?
from tool_sandbox.common.tool_discovery import find_tools_by_module
import tool_sandbox.tools
tools = find_tools_by_module(tool_sandbox.tools, preferred_tool_backend=ToolBackend.DEFAULT)
assert "lookup_contacts" in tools, "lookup_contacts not discovered as a tool"
print("[step6] lookup_contacts tool discovered OK")
PY

# API endpoint and key.
# shellcheck disable=SC1090
ENV_FILE="${ENV_FILE:-$HOME/.wear_credentials.env}"   # API keys; override with ENV_FILE=
source "$ENV_FILE"
# Models whose name starts with gemini use the Gemini key, all others the GPT key.
if [[ "$MODEL" == gemini* ]]; then
  export OPENAI_API_KEY="$RELAY_GEMINI_API_KEY"
else
  export OPENAI_API_KEY="$RELAY_GPT_API_KEY"
fi
export OPENAI_BASE_URL="$RELAY_BASE_URL/v1"
export OPENAI_API_BASE="$RELAY_BASE_URL/v1"
# With the ToolSandbox patch, the GPT_4_o_2024_05_13 agent and user roles read the
# endpoint from OPENAI_BASE_URL and the model name from OPENAI_MODEL.
export OPENAI_MODEL="$MODEL"

# K rollouts per scenario, one tool_sandbox process each; a failed rollout is
# reported and skipped.
cd "$TS"
for scen in "${SCENS[@]}"; do
  for i in $(seq 1 "$K"); do
    out="$OUT_BASE/${scen}_run$i"
    mkdir -p "$out"
    echo "[step6] RUN $scen #$i -> $out"
    "$VENV/tool_sandbox" \
      --agent GPT_4_o_2024_05_13 --user GPT_4_o_2024_05_13 \
      -s "$scen" -p 1 -o "$out" || echo "[step6] WARN: run failed, continuing"
  done
done

# Per-rollout summary: WEAR from conversation.json, then ToolSandbox's milestone
# and minefield similarity from result_summary.json.
echo ""
echo "================ STEP6 WEAR SUMMARY ================"
for idx in "${!SCENS[@]}"; do
  scen="${SCENS[$idx]}"; gold="${GOLDS[$idx]}"
  echo "--- $scen (gold=$gold) ---"
  find "$OUT_BASE" -path "*${scen}_run*" -name conversation.json | sort | while read -r conv; do
    python3 "$EB/wear/toolsandbox/score_wear_ts.py" "$conv" --gold "$gold" \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  run: writes={d[\"n_write_calls\"]} attempted={d[\"attempted_wear\"]} effective={d[\"effective_wear\"]} targets={[ (c[\"target_id\"] or \"\")[:8] for c in d[\"write_calls\"] ]}')"
  done
  echo "  native eval (milestone/minefield):"
  find "$OUT_BASE" -path "*${scen}_run*" -name result_summary.json | sort | while read -r rs; do
    python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['per_scenario_results'][0]; print(f'    milestone={r[\"milestone_similarity\"]} minefield={r[\"minefield_similarity\"]}')" "$rs"
  done
done
