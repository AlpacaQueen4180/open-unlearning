#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/user/alpaca/open-unlearning
ENV="$ROOT/.venv-transformers445"
PY="$ENV/bin/python"
QUEUE_ID=openunlearning_npo_1xAda6000_5seeds_all6_20260829
QUEUE_STATUS="$ROOT/${QUEUE_ID}.status.json"
FINAL_SUMMARY="$ROOT/openunlearning_npo_1xAda6000_5seeds_all6_20260829.summary.json"
CLAIM=/home/user/.open-unlearning-ada6000-5seed-claim

cd "$ROOT" || exit 2
if [[ -e "$QUEUE_STATUS" || -e "$FINAL_SUMMARY" ]]; then
  printf '{"status":"REFUSED_EXISTING_QUEUE","queue":"%s"}\n' "$QUEUE_ID" > "$ROOT/${QUEUE_ID}.launch-refused.json"
  exit 3
fi
if ! mkdir "$CLAIM" 2>/dev/null; then
  printf '{"status":"REFUSED_QUEUE_CLAIMED","queue":"%s"}\n' "$QUEUE_ID" > "$QUEUE_STATUS"
  exit 4
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

base_cells=(
  "Llama-2-7b-chat-hf|forget01|holdout01|retain99"
  "Llama-2-7b-chat-hf|forget05|holdout05|retain95"
  "Llama-2-7b-chat-hf|forget10|holdout10|retain90"
  "Llama-3.2-1B-Instruct|forget01|holdout01|retain99"
  "Llama-3.2-1B-Instruct|forget05|holdout05|retain95"
  "Llama-3.2-1B-Instruct|forget10|holdout10|retain90"
)

for seed in 0 1 2 3 4; do
  for cell in "${base_cells[@]}"; do
    IFS='|' read -r model forget _holdout _retain <<< "$cell"
    task="tofu_${model}_${forget}_NPO_1xAda6000_flash2_micro8_acc4_e10_seed${seed}_20260829"
    if [[ -e "$ROOT/${task}.status.json" || -e "$ROOT/saves/unlearn/$task" || -e "$ROOT/saves/eval/${task}_eval" ]]; then
      printf '{"status":"REFUSED_EXISTING_CELL","queue":"%s","task":"%s","seed":%s}\n' \
        "$QUEUE_ID" "$task" "$seed" > "$QUEUE_STATUS"
      exit 5
    fi
  done
done

cell_number=1
for seed in 0 1 2 3 4; do
  for cell in "${base_cells[@]}"; do
    IFS='|' read -r model forget holdout retain <<< "$cell"
    task="tofu_${model}_${forget}_NPO_1xAda6000_flash2_micro8_acc4_e10_seed${seed}_20260829"
    available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 { print $4 }')"
    if (( available_kb < 52428800 )); then
      printf '{"status":"FAILED_LOW_DISK","queue":"%s","cell":%s,"task":"%s","seed":%s,"available_kb":%s,"required_kb":52428800}\n' \
        "$QUEUE_ID" "$cell_number" "$task" "$seed" "$available_kb" > "$QUEUE_STATUS"
      exit 6
    fi

    printf '{"status":"WAITING_FOR_IDLE_GPU","queue":"%s","cell":%s,"cell_count":30,"task":"%s","model":"%s","forget_split":"%s","seed":%s,"required_idle_seconds":45}\n' \
      "$QUEUE_ID" "$cell_number" "$task" "$model" "$forget" "$seed" > "$QUEUE_STATUS"
    idle_checks=0
    while (( idle_checks < 4 )); do
      process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
      if [[ "$process_count" == 0 ]]; then
        idle_checks=$((idle_checks + 1))
      else
        idle_checks=0
      fi
      if (( idle_checks < 4 )); then
        sleep 15
      fi
    done

    printf '{"status":"RUNNING_CELL","queue":"%s","cell":%s,"cell_count":30,"task":"%s","model":"%s","forget_split":"%s","seed":%s}\n' \
      "$QUEUE_ID" "$cell_number" "$task" "$model" "$forget" "$seed" > "$QUEUE_STATUS"
    "$ROOT/run_ada6000_seed_cell.sh" "$model" "$forget" "$holdout" "$retain" "$seed" "$task"
    cell_rc=$?
    if [[ "$cell_rc" -ne 0 ]]; then
      printf '{"status":"FAILED_CELL","queue":"%s","cell":%s,"task":"%s","seed":%s,"exit_code":%s,"cell_status":"%s"}\n' \
        "$QUEUE_ID" "$cell_number" "$task" "$seed" "$cell_rc" "$ROOT/${task}.status.json" > "$QUEUE_STATUS"
      exit "$cell_rc"
    fi
    cell_number=$((cell_number + 1))
  done
done

"$PY" "$ROOT/summarize_ada6000_npo_5seeds.py" > "$FINAL_SUMMARY.tmp"
summary_rc=$?
if [[ "$summary_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_SUMMARY","queue":"%s","exit_code":%s}\n' "$QUEUE_ID" "$summary_rc" > "$QUEUE_STATUS"
  exit "$summary_rc"
fi
mv "$FINAL_SUMMARY.tmp" "$FINAL_SUMMARY"
"$PY" -c 'import json,sys; q=json.load(open(sys.argv[1])); q["queue"] = sys.argv[2]; q["summary_path"] = sys.argv[1]; print(json.dumps(q, indent=2))' \
  "$FINAL_SUMMARY" "$QUEUE_ID" > "$QUEUE_STATUS.tmp"
mv "$QUEUE_STATUS.tmp" "$QUEUE_STATUS"
