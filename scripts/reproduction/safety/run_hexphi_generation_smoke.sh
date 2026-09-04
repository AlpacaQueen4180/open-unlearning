#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
PYTHON_BIN="${PYTHON_BIN:-/home/ai/miniforge3/envs/open-unlearning-repro/bin/python}"
MODEL="${MODEL:-$ROOT/saves/unlearn/tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_no_retain_2xH100_seed0_20260904}"
DATASET="${DATASET:-$ROOT/safety_artifacts/datasets/HEx-PHI-legacy-300.json}"
OUTPUT="${OUTPUT:-$ROOT/safety_artifacts/raw/npo_no_retain_legacy300_smoke10.jsonl}"
STATUS="${STATUS:-$ROOT/safety_artifacts/npo_no_retain_legacy300_smoke10.status.json}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

printf '{"status":"RUNNING","started_at":"%s"}\n' "$(date -Is)" > "$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"

set +e
"$PYTHON_BIN" -m evals.safety generate \
  --model "$MODEL" \
  --input "$DATASET" \
  --output "$OUTPUT" \
  --run-id llama31_8b_npo_no_retain_legacy300_smoke10 \
  --protocol legacy-300 \
  --checkpoint-id tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_no_retain_2xH100_seed0_20260904 \
  --seed 0 \
  --max-new-tokens 512 \
  --limit 10
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  rows="$(wc -l < "$OUTPUT")"
  printf '{"status":"DONE","rows":%s,"output":"%s","finished_at":"%s"}\n' "$rows" "$OUTPUT" "$(date -Is)" > "$STATUS.tmp"
else
  printf '{"status":"FAILED","exit_code":%s,"output":"%s","finished_at":"%s"}\n' "$rc" "$OUTPUT" "$(date -Is)" > "$STATUS.tmp"
fi
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
