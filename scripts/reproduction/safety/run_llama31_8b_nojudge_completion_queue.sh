#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
QUEUE_ID="${QUEUE_ID:-llama31_8b_nojudge_completion_20260911}"
STATUS="$ROOT/safety_artifacts/$QUEUE_ID.status.json"
LOG_DIR="$ROOT/safety_artifacts/$QUEUE_ID.logs"
REGISTRY="$ROOT/configs/experiment/safety/llama31_8b_capability_31_nojudge.json"
FASTCHAT_COMMIT=587d5cfa1609a43d192cedb8441cac3c17db105d
MT_BENCH_SHA256=119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7
QUESTIONS="$ROOT/safety_artifacts/datasets/MT-Bench-${FASTCHAT_COMMIT}.jsonl"

if [[ "${ENABLE_API_JUDGE:-0}" != 0 ]]; then
  printf 'Refusing to run: ENABLE_API_JUDGE must be 0.\n' >&2
  exit 2
fi
export ENABLE_API_JUDGE=0
unset OPENAI_API_KEY AZURE_OPENAI_API_KEY OPENAI_BASE_URL OPENAI_ORG_ID

write_status() {
  "$PY" - "$STATUS.tmp" "$1" "$2" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2], "stage": sys.argv[3], "api_judge_enabled": False,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}, sort_keys=True) + "\n")
PY
  mv "$STATUS.tmp" "$STATUS"
}

mkdir -p "$LOG_DIR" "$(dirname "$QUESTIONS")"
cd "$ROOT"

if [[ "${1:-}" == --dry-run ]]; then
  bash scripts/reproduction/safety/run_llama31_8b_public_f05_seeds1_4_nojudge_queue.sh --dry-run
  bash scripts/reproduction/safety/run_llama31_8b_nojudge_baselines.sh --dry-run
  "$PY" scripts/reproduction/safety/run_llama31_8b_capability_31_nojudge.py \
    --root "$ROOT" --registry "$REGISTRY" --questions "$QUESTIONS" --dry-run
  exit 0
fi

[[ ! -e "$STATUS" ]] || { printf 'Refusing existing parent status: %s\n' "$STATUS" >&2; exit 2; }
if [[ ! -s "$QUESTIONS" ]]; then
  curl -fsSL "https://raw.githubusercontent.com/lm-sys/FastChat/$FASTCHAT_COMMIT/fastchat/llm_judge/data/mt_bench/question.jsonl" -o "$QUESTIONS.tmp"
  [[ "$(sha256sum "$QUESTIONS.tmp" | cut -d' ' -f1)" == "$MT_BENCH_SHA256" ]] || { rm -f "$QUESTIONS.tmp"; exit 2; }
  mv "$QUESTIONS.tmp" "$QUESTIONS"
fi
[[ "$(sha256sum "$QUESTIONS" | cut -d' ' -f1)" == "$MT_BENCH_SHA256" ]] || { printf 'MT-Bench checksum mismatch\n' >&2; exit 2; }

write_status RUNNING public_f05_training
bash scripts/reproduction/safety/run_llama31_8b_public_f05_seeds1_4_nojudge_queue.sh > "$LOG_DIR/public_f05.log" 2>&1

write_status RUNNING baseline_tofu_and_hexphi_generation
bash scripts/reproduction/safety/run_llama31_8b_nojudge_baselines.sh > "$LOG_DIR/baselines.log" 2>&1

write_status RUNNING capability_31
"$PY" scripts/reproduction/safety/run_llama31_8b_capability_31_nojudge.py \
  --root "$ROOT" --registry "$REGISTRY" --questions "$QUESTIONS" \
  --queue-id llama31_8b_capability_31_nojudge_20260911 --workers 2 \
  > "$LOG_DIR/capability.log" 2>&1

write_status DONE pending_judge_artifacts_ready
