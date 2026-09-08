#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
QUEUE_ID="${QUEUE_ID:-llama31_8b_tofu_rebuild_npo_seed0_20260907}"
QUEUE_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
RAW="$QUEUE_DIR/raw/local_retain95.formal-300.jsonl"
JUDGED="$QUEUE_DIR/judged/local_retain95.formal-300.jsonl"
SUMMARY="$QUEUE_DIR/summaries/local_retain95.formal-300.json"
LEDGER="$QUEUE_DIR/ledger.tsv"
CHECKPOINT="$ROOT/saves/finetune/${QUEUE_ID}_local_retain95"
TOFU_EVAL="$ROOT/saves/eval/local_retain95/TOFU_EVAL.json"
SECRET_ENV=/home/ai/.config/open-unlearning/safety.env

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PATH="$ENV_DIR/bin:$PATH"

[[ -s "$RAW" && "$(wc -l < "$RAW")" -eq 300 ]]
[[ -s "$JUDGED" && -s "$TOFU_EVAL" && -s "$CHECKPOINT/trainer_state.json" ]]
if grep -q $'^1\tlocal_retain95\t' "$LEDGER"; then
  printf 'Cell 1 is already present in the ledger; refusing duplicate resume.\n' >&2
  exit 2
fi

"$PY" - "$QUEUE_DIR/queue.status.json" "$QUEUE_ID" <<'PY'
import datetime, json, pathlib, sys
path, queue = pathlib.Path(sys.argv[1]), sys.argv[2]
payload = {
    "status": "RESUMING_SAFETY_EVAL", "queue": queue, "cell": 1,
    "total": 8, "completed": 0, "stage": "safety",
    "detail": "local_retain95", "updated_at": datetime.datetime.now(
        datetime.timezone.utc
    ).astimezone().isoformat(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
tmp.replace(path)
PY

"$PY" -m evals.safety judge --input "$RAW" --output "$JUDGED" \
  --model gpt-5.6-terra --reasoning-effort medium --concurrency 8 \
  --retries 3 --timeout 60 --env-file "$SECRET_ENV" \
  > "$QUEUE_DIR/logs/local_retain95.judge-resume.log" 2>&1
"$PY" -m evals.safety summarize --input "$JUDGED" --output "$SUMMARY" \
  > "$QUEUE_DIR/logs/local_retain95.summary-resume.log" 2>&1

"$PY" - "$SUMMARY" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
coverage = data["overall"]["judge_coverage"]
if coverage["total"] != 300 or coverage["successful"] + coverage["policy_blocked"] != 300:
    raise SystemExit(f"invalid judge coverage: {coverage}")
PY

printf '1\tlocal_retain95\tfinetune\tforget05\tretain95\t\t%s\t%s\t%s\t\n' \
  "$CHECKPOINT" "$TOFU_EVAL" "$SUMMARY" >> "$LEDGER"

exec env RESUME_FROM_CELL=2 \
  scripts/reproduction/safety/run_llama31_8b_tofu_rebuild_npo_seed0_queue.sh
