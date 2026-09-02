#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ai/alpaca
ENV=/home/ai/miniforge3/envs/open-unlearning-repro
PY="$ENV/bin/python"
QUEUE_ID=openunlearning_npo_2xH100_llama2_7b_llama3_1b_forget01_05_10_20260828
QUEUE_STATUS="$ROOT/${QUEUE_ID}.status.json"
CLAIM=/home/ai/.open-unlearning-npo-matrix-claim
EXISTING_L2_F05="$ROOT/tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827_eval_retry2.status.json"

cd "$ROOT" || exit 2
if [[ -e "$QUEUE_STATUS" ]]; then
  printf '{"status":"REFUSED_EXISTING_QUEUE","queue":"%s"}\n' "$QUEUE_ID" > "$ROOT/${QUEUE_ID}.launch-refused.json"
  exit 3
fi
if ! mkdir "$CLAIM" 2>/dev/null; then
  printf '{"status":"REFUSED_QUEUE_CLAIMED","queue":"%s"}\n' "$QUEUE_ID" > "$QUEUE_STATUS"
  exit 4
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

if ! "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "DONE"' "$EXISTING_L2_F05"; then
  printf '{"status":"REFUSED_EXISTING_L2_F05_NOT_DONE","path":"%s"}\n' "$EXISTING_L2_F05" > "$QUEUE_STATUS"
  exit 5
fi

cells=(
  "Llama-2-7b-chat-hf|forget01|holdout01|retain99|tofu_Llama-2-7b-chat-hf_forget01_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828"
  "Llama-2-7b-chat-hf|forget10|holdout10|retain90|tofu_Llama-2-7b-chat-hf_forget10_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828"
  "Llama-3.2-1B-Instruct|forget01|holdout01|retain99|tofu_Llama-3.2-1B-Instruct_forget01_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828"
  "Llama-3.2-1B-Instruct|forget05|holdout05|retain95|tofu_Llama-3.2-1B-Instruct_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828"
  "Llama-3.2-1B-Instruct|forget10|holdout10|retain90|tofu_Llama-3.2-1B-Instruct_forget10_NPO_2xH100_zero3_flash2_micro4_acc4_e10_seed0_20260828"
)

cell_number=1
for cell in "${cells[@]}"; do
  IFS='|' read -r model forget holdout retain task <<< "$cell"
  printf '{"status":"WAITING_FOR_TWO_IDLE_GPUS","queue":"%s","cell":%s,"cell_count":5,"task":"%s","model":"%s","forget_split":"%s","required_idle_seconds":120}\n' \
    "$QUEUE_ID" "$cell_number" "$task" "$model" "$forget" > "$QUEUE_STATUS"

  idle_checks=0
  while (( idle_checks < 4 )); do
    process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
    if [[ "$process_count" == 0 ]]; then
      idle_checks=$((idle_checks + 1))
    else
      idle_checks=0
    fi
    if (( idle_checks < 4 )); then
      sleep 30
    fi
  done

  printf '{"status":"RUNNING_CELL","queue":"%s","cell":%s,"cell_count":5,"task":"%s","model":"%s","forget_split":"%s"}\n' \
    "$QUEUE_ID" "$cell_number" "$task" "$model" "$forget" > "$QUEUE_STATUS"
  "$ROOT/run_h100_matrix_cell.sh" "$model" "$forget" "$holdout" "$retain" "$task"
  cell_rc=$?
  if [[ "$cell_rc" -ne 0 ]]; then
    printf '{"status":"FAILED_CELL","queue":"%s","cell":%s,"task":"%s","exit_code":%s,"cell_status":"%s"}\n' \
      "$QUEUE_ID" "$cell_number" "$task" "$cell_rc" "$ROOT/${task}.status.json" > "$QUEUE_STATUS"
    exit "$cell_rc"
  fi
  cell_number=$((cell_number + 1))
done

"$PY" "$ROOT/summarize_h100_npo_matrix.py" > "$QUEUE_STATUS.tmp"
summary_rc=$?
if [[ "$summary_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_SUMMARY","queue":"%s","exit_code":%s}\n' "$QUEUE_ID" "$summary_rc" > "$QUEUE_STATUS"
  exit "$summary_rc"
fi
mv "$QUEUE_STATUS.tmp" "$QUEUE_STATUS"
