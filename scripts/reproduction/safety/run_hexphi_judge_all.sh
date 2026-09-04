#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
PYTHON_BIN="${PYTHON_BIN:-/home/ai/miniforge3/envs/open-unlearning-repro/bin/python}"
SECRET_ENV="${SECRET_ENV:-/home/ai/.config/open-unlearning/safety.env}"
ARTIFACT_ROOT="$ROOT/safety_artifacts"
STATUS="$ARTIFACT_ROOT/judge_all.status.json"
HISTORICAL_RAW="$ARTIFACT_ROOT/historical/tofu_Llama-3.1-8B-Instruct_forget05_NPO_hexphi.json"
HISTORICAL_MINI="$ARTIFACT_ROOT/historical/tofu_Llama-3.1-8B-Instruct_forget05_NPO_hexphi_gpt5mini.jsonl"

models=(original tofu_full npo_no_retain npo_retain95 retain95_oracle)
protocols=(legacy-300 formal-300)

write_status() {
  local state="$1" detail="${2:-}"
  printf '{"status":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$state" "$detail" "$(date -Is)" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}

on_error() {
  local rc=$?
  write_status FAILED "exit_code=$rc"
  exit "$rc"
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
trap on_error ERR

if [[ ! -f "$SECRET_ENV" || "$(stat -c '%a' "$SECRET_ENV")" != 600 ]]; then
  write_status BLOCKED_MISSING_API_KEY_OR_MODE "$SECRET_ENV"
  exit 2
fi

for model_name in "${models[@]}"; do
  for protocol in "${protocols[@]}"; do
    raw="$ARTIFACT_ROOT/raw/${model_name}_${protocol}_seed0.jsonl"
    if [[ ! -f "$raw" || "$(wc -l < "$raw")" -ne 300 ]]; then
      write_status BLOCKED_INCOMPLETE_GENERATION "${model_name}:${protocol}"
      exit 3
    fi
  done
done

mkdir -p "$ARTIFACT_ROOT/judged" "$ARTIFACT_ROOT/summaries" "$ARTIFACT_ROOT/logs"
for model_name in "${models[@]}"; do
  for protocol in "${protocols[@]}"; do
    raw="$ARTIFACT_ROOT/raw/${model_name}_${protocol}_seed0.jsonl"
    judged="$ARTIFACT_ROOT/judged/${model_name}_${protocol}_seed0.jsonl"
    summary="$ARTIFACT_ROOT/summaries/${model_name}_${protocol}_seed0.json"
    write_status RUNNING "${model_name}:${protocol}"
    "$PYTHON_BIN" -m evals.safety judge --input "$raw" --output "$judged" \
      --model gpt-5.6-terra --reasoning-effort medium \
      --concurrency 8 --retries 3 --timeout 60 --env-file "$SECRET_ENV" \
      > "$ARTIFACT_ROOT/logs/${model_name}_${protocol}.judge.log" 2>&1
    "$PYTHON_BIN" -m evals.safety summarize --input "$judged" --output "$summary" \
      > "$ARTIFACT_ROOT/logs/${model_name}_${protocol}.summary.log" 2>&1
  done
done

if [[ -f "$HISTORICAL_RAW" && -f "$HISTORICAL_MINI" ]]; then
  historical_terra="$ARTIFACT_ROOT/judged/historical_npo_legacy300_terra.jsonl"
  write_status RUNNING historical_calibration
  "$PYTHON_BIN" -m evals.safety judge --input "$HISTORICAL_RAW" \
    --output "$historical_terra" --model gpt-5.6-terra \
    --reasoning-effort medium --concurrency 8 --retries 3 --timeout 60 \
    --env-file "$SECRET_ENV" \
    > "$ARTIFACT_ROOT/logs/historical_npo_terra.judge.log" 2>&1
  "$PYTHON_BIN" -m evals.safety calibrate --old "$HISTORICAL_MINI" \
    --new "$historical_terra" \
    --output "$ARTIFACT_ROOT/summaries/gpt5mini_vs_gpt56terra_calibration.json" \
    > "$ARTIFACT_ROOT/logs/historical_calibration.log" 2>&1
else
  write_status DONE_WITHOUT_HISTORICAL_CALIBRATION historical_inputs_missing
  trap - ERR
  exit 0
fi

write_status DONE complete
trap - ERR
