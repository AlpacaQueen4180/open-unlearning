#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
PY="${PY:-/home/ai/miniforge3/envs/open-unlearning-repro/bin/python}"
SECRET_ENV="${SECRET_ENV:-/home/ai/.config/open-unlearning/safety.env}"
QUEUE_ID="${QUEUE_ID:-llama31_8b_credit_resume_20260914}"
STATUS="$ROOT/safety_artifacts/$QUEUE_ID.status.json"
LOG_DIR="$ROOT/safety_artifacts/$QUEUE_ID.logs"
FASTCHAT_COMMIT=587d5cfa1609a43d192cedb8441cac3c17db105d
QUESTIONS="$ROOT/safety_artifacts/datasets/MT-Bench-${FASTCHAT_COMMIT}.jsonl"
JUDGE_PROMPTS="$ROOT/safety_artifacts/datasets/MT-Bench-judge-prompts-${FASTCHAT_COMMIT}.jsonl"
REFERENCES="$ROOT/safety_artifacts/datasets/MT-Bench-reference-gpt4-${FASTCHAT_COMMIT}.jsonl"
PROMPTS_SHA=fd283293406d024f44c174b094ef48031d0687a4682fd3a56b29b138f80281b6
REFERENCES_SHA=f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f

write_status() {
  "$PY" - "$STATUS.tmp" "$1" "$2" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2], "stage": sys.argv[3],
    "judge_model": "gpt-5.6-terra", "reasoning_effort": "medium",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}, sort_keys=True) + "\n")
PY
  mv "$STATUS.tmp" "$STATUS"
}

download_verified() {
  local url="$1" destination="$2" expected="$3"
  if [[ ! -s "$destination" ]]; then
    curl -fsSL "$url" -o "$destination.tmp"
    [[ "$(sha256sum "$destination.tmp" | cut -d' ' -f1)" == "$expected" ]] || { rm -f "$destination.tmp"; return 1; }
    mv "$destination.tmp" "$destination"
  fi
  [[ "$(sha256sum "$destination" | cut -d' ' -f1)" == "$expected" ]]
}

cd "$ROOT"
mkdir -p "$LOG_DIR" "$(dirname "$QUESTIONS")"
export PYTHONPATH="$ROOT/src"
[[ -x "$PY" ]] || { printf 'Missing Python runtime\n' >&2; exit 2; }
[[ -f "$SECRET_ENV" && "$(stat -c '%a' "$SECRET_ENV")" == 600 ]] || { printf 'Invalid secret env file\n' >&2; exit 2; }
[[ -s "$QUESTIONS" ]] || { printf 'Missing pinned MT-Bench questions\n' >&2; exit 2; }
download_verified "https://raw.githubusercontent.com/lm-sys/FastChat/$FASTCHAT_COMMIT/fastchat/llm_judge/data/judge_prompts.jsonl" "$JUDGE_PROMPTS" "$PROMPTS_SHA"
download_verified "https://raw.githubusercontent.com/lm-sys/FastChat/$FASTCHAT_COMMIT/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl" "$REFERENCES" "$REFERENCES_SHA"
if [[ -e "$STATUS" && "${RESUME:-0}" != 1 ]]; then
  printf 'Refusing existing status without RESUME=1: %s\n' "$STATUS" >&2
  exit 2
fi

on_error() {
  local rc=$?
  trap - ERR
  write_status FAILED "exit_code=$rc"
  exit "$rc"
}
trap on_error ERR

write_status RUNNING capability_and_hexphi
ENABLE_API_JUDGE=0 "$PY" scripts/reproduction/safety/run_llama31_8b_capability_31_nojudge.py \
  --root "$ROOT" --registry configs/experiment/safety/llama31_8b_capability_31_nojudge.json \
  --questions "$QUESTIONS" --queue-id llama31_8b_capability_31_nojudge_20260911 --workers 2 \
  > "$LOG_DIR/capability.log" 2>&1 &
capability_pid=$!
ENABLE_API_JUDGE=1 bash scripts/reproduction/safety/run_llama31_8b_pending_hexphi_judge.sh \
  > "$LOG_DIR/hexphi_judge.log" 2>&1 &
hexphi_pid=$!

set +e
wait "$capability_pid"; capability_rc=$?
wait "$hexphi_pid"; hexphi_rc=$?
set -e
if (( capability_rc != 0 || hexphi_rc != 0 )); then
  write_status FAILED "capability_rc=$capability_rc,hexphi_rc=$hexphi_rc"
  exit 1
fi

write_status RUNNING mtbench_judge
ENABLE_API_JUDGE=1 "$PY" -m evals.mt_bench_judge \
  --answers-dir "$ROOT/safety_artifacts/llama31_8b_capability_31_nojudge_20260911/mt_bench_raw" \
  --questions "$QUESTIONS" --references "$REFERENCES" --judge-prompts "$JUDGE_PROMPTS" \
  --output "$ROOT/safety_artifacts/llama31_8b_mtbench_terra_20260914/judged.jsonl" \
  --summary "$ROOT/results/reproduction/capability/llama31_8b_mtbench_terra_20260914/aggregate.json" \
  --env-file "$SECRET_ENV" --model gpt-5.6-terra --reasoning-effort medium \
  --concurrency 8 --retries 3 --timeout 60 --expected-models 31 \
  --fastchat-commit "$FASTCHAT_COMMIT" > "$LOG_DIR/mtbench_judge.log" 2>&1

write_status DONE complete
trap - ERR
