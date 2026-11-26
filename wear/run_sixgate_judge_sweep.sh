#!/usr/bin/env bash
# Six-gate judge sweep: runs the six-gate judge (wear/pae_sixgate.py) on perturbed
# rollouts, once for each ground-truth mode in GTS, with 4 jobs in parallel.
# Idempotent: jobs whose output file exists are skipped.
#
# Inputs: the perturbed run files of the main experiment for each MODEL_SLUG and,
#   with INCLUDE_RECOVER=1, the GPT-5.5 rollouts of the recovery suite (the
#   directories are named in the loops below); the source tasks in wear/deictic_tasks/.
# Outputs: $OUTROOT/<model>/<mode>/<task>.json, with a .log next to each, and
#   $OUTROOT/recover/<mode>/<file>.json for the recovery suite; failed jobs are
#   listed in $OUTROOT/failures.txt.
# Credentials: read from $ENV_FILE; the judge is reached through the relay.
# Usage: bash wear/run_sixgate_judge_sweep.sh [MODEL_SLUG...]  (default: openai_gpt_5_5 openai_gemini_3_1_pro)
set -uo pipefail
EB=$(cd "$(dirname "$0")/.." && pwd)   # repo root
cd "$EB"
ENV_FILE="${ENV_FILE:-$HOME/.wear_credentials.env}"   # credentials file; override with ENV_FILE
set -a; source "$ENV_FILE"; set +a
# Settings from the environment (defaults below):
#   JUDGE            judge model, gpt-5.5 or a Gemini model such as gemini-3.1-pro
#   GTS              ground-truth modes passed to --groundtruth: full (user scenario and
#                    original gold actions), scenario (scenario only), none (policy and transcript only)
#   OUTROOT          output directory
#   INCLUDE_RECOVER  1 to judge the recovery suite as well
JUDGE="${JUDGE:-gpt-5.5}"
GTS="${GTS:-full scenario none}"
OUTROOT="${OUTROOT:-wear/runs/sixgate_judge}"
INCLUDE_RECOVER="${INCLUDE_RECOVER:-1}"
# Judge credentials: the relay's key for the judge's model family, and the relay's base URL.
case "$JUDGE" in
  *gemini*) export OPENAI_API_KEY="$RELAY_GEMINI_API_KEY" ;;
  *)        export OPENAI_API_KEY="$RELAY_GPT_API_KEY" ;;
esac
export OPENAI_BASE_URL="${RELAY_BASE_URL%/}/v1"
P=$EB/tau2-bench/.venv/bin/python
MODELS=("${@:-openai_gpt_5_5 openai_gemini_3_1_pro}")
[ $# -eq 0 ] && MODELS=(openai_gpt_5_5 openai_gemini_3_1_pro)

# One job per line: run file|domain|task file|ground-truth mode|output file
jobs_file=$(mktemp)
for gt in $GTS; do
  # main experiment: perturbed run files of each model
  for m in "${MODELS[@]}"; do
    for f in wear/runs/main_experiment/"$m"/perturbed/*.json; do
      b=$(basename "$f" .json)
      dom=${b%%_*}   # domain from the file-name prefix
      task=wear/deictic_tasks/$b.json
      out=$OUTROOT/$m/$gt/$b.json
      [ -f "$out" ] && continue
      echo "$f|$dom|$task|$gt|$out" >> "$jobs_file"
    done
  done
  # recovery suite: GPT-5.5 perturbed run files (all retail)
  [ "$INCLUDE_RECOVER" = "1" ] || continue
  for f in wear/runs/recovery_suite/*_pert_gpt55_s300.json wear/runs/recovery_suite/*_k4_s301.json \
           wear/runs/recovery_suite/retail_recover_ereader_pert_gpt55.json \
           wear/runs/recovery_suite/retail_recover_backpack_pert_gpt55.json; do
    [ -f "$f" ] || continue
    b=$(basename "$f" .json)
    slug=$(echo "$b" | sed 's/_pert_gpt55.*//')
    task=wear/deictic_tasks/$slug.json
    out=$OUTROOT/recover/$gt/$b.json
    [ -f "$out" ] && continue
    echo "$f|retail|$task|$gt|$out" >> "$jobs_file"
  done
done

echo "[sixgate_judge] $(wc -l < "$jobs_file") jobs (judge=$JUDGE outroot=$OUTROOT)"
export JUDGE OUTROOT
xargs -P 4 -L1 -a "$jobs_file" bash -c '
  IFS="|" read -r f dom task gt out <<< "$0"
  mkdir -p "$(dirname "$out")"
  '"$P"' wear/pae_sixgate.py "$f" --domain "$dom" --tasks "$task" \
    --judge "$JUDGE" --groundtruth "$gt" --workers 4 --out "$out" \
    > "${out%.json}.log" 2>&1 || echo "FAIL $out" >> "$OUTROOT"/failures.txt
'
rm -f "$jobs_file"
echo "[sixgate_judge] sweep done; failures: $(wc -l < "$OUTROOT"/failures.txt 2>/dev/null || echo 0)"
