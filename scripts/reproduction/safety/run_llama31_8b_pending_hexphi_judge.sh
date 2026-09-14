#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
PY="${PY:-/home/ai/miniforge3/envs/open-unlearning-repro/bin/python}"
SECRET_ENV="${SECRET_ENV:-/home/ai/.config/open-unlearning/safety.env}"
QUEUE_ID="${QUEUE_ID:-llama31_8b_pending_hexphi_judge_20260914}"
STATUS="$ROOT/safety_artifacts/$QUEUE_ID.status.json"
LEDGER="$ROOT/safety_artifacts/$QUEUE_ID.ledger.tsv"
PUBLIC_DIR="$ROOT/results/reproduction/safety/$QUEUE_ID"

roots=(
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_public_f05_seeds1_4_nojudge_20260911
  llama31_8b_nojudge_baselines_20260911
  llama31_8b_nojudge_baselines_20260911
)
names=(
  published_full_forget05_npo_no_retain_seed1
  published_full_forget05_npo_retain95_seed1
  published_full_forget05_npo_no_retain_seed2
  published_full_forget05_npo_retain95_seed2
  published_full_forget05_npo_no_retain_seed3
  published_full_forget05_npo_retain95_seed3
  published_full_forget05_npo_no_retain_seed4
  published_full_forget05_npo_retain95_seed4
  published_retain99_oracle
  published_retain90_oracle
)

if [[ "${ENABLE_API_JUDGE:-0}" != 1 ]]; then
  printf 'Refusing to run: set ENABLE_API_JUDGE=1 explicitly.\n' >&2
  exit 2
fi

write_status() {
  "$PY" - "$STATUS.tmp" "$1" "$2" "$3" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2], "cell": int(sys.argv[3]), "detail": sys.argv[4],
    "total": 10, "judge_model": "gpt-5.6-terra", "reasoning_effort": "medium",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}, sort_keys=True) + "\n")
PY
  mv "$STATUS.tmp" "$STATUS"
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
[[ -x "$PY" ]] || { printf 'Missing runtime: %s\n' "$PY" >&2; exit 2; }
[[ -f "$SECRET_ENV" && "$(stat -c '%a' "$SECRET_ENV")" == 600 ]] || { printf 'Invalid secret env file or permissions\n' >&2; exit 2; }
mkdir -p "$PUBLIC_DIR" "$ROOT/safety_artifacts/$QUEUE_ID.logs"
if [[ ! -s "$LEDGER" ]]; then
  printf 'cell\tname\traw\tjudged\tsummary\tstatus\n' > "$LEDGER"
fi

for index in "${!names[@]}"; do
  cell=$((index + 1)); name="${names[$index]}"; source_root="${roots[$index]}"
  artifact="$ROOT/safety_artifacts/$source_root"
  raw="$artifact/raw/$name.formal-300.jsonl"
  judged="$artifact/judged/$name.formal-300.jsonl"
  summary="$artifact/summaries/$name.formal-300.json"
  mkdir -p "$artifact/judged" "$artifact/summaries"
  "$PY" - "$raw" <<'PY'
import json, pathlib, sys
rows=[json.loads(x) for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if x.strip()]
if len({r.get("prompt_id") for r in rows if r.get("status")=="success"}) != 300:
    raise SystemExit("raw generation is not complete")
PY
  if [[ -s "$summary" ]]; then
    continue
  fi
  write_status RUNNING "$cell" "$name"
  "$PY" -m evals.safety judge --input "$raw" --output "$judged" \
    --model gpt-5.6-terra --reasoning-effort medium --concurrency 8 \
    --retries 3 --timeout 60 --env-file "$SECRET_ENV" \
    > "$ROOT/safety_artifacts/$QUEUE_ID.logs/$name.judge.log" 2>&1
  "$PY" -m evals.safety verify --input "$raw" --judged "$judged" \
    > "$ROOT/safety_artifacts/$QUEUE_ID.logs/$name.verify.log" 2>&1
  "$PY" -m evals.safety summarize --input "$judged" --output "$summary" \
    > "$ROOT/safety_artifacts/$QUEUE_ID.logs/$name.summary.log" 2>&1
  printf '%s\t%s\t%s\t%s\t%s\tcomplete\n' "$cell" "$name" "$raw" "$judged" "$summary" >> "$LEDGER"
done

cp "$LEDGER" "$PUBLIC_DIR/ledger.tsv"
"$PY" - "$LEDGER" "$PUBLIC_DIR/aggregate.json" <<'PY'
import csv, datetime, json, pathlib, sys
ledger, output = map(pathlib.Path, sys.argv[1:])
rows=list(csv.DictReader(ledger.open(), delimiter="\t"))
dedup={row["name"]:row for row in rows}
if len(dedup)!=10: raise SystemExit(f"expected 10 completed summaries, found {len(dedup)}")
models={}
for name,row in sorted(dedup.items()):
 data=json.loads(pathlib.Path(row["summary"]).read_text())
 models[name]={"overall":data["overall"],"by_category":data["by_category"]}
payload={
 "schema_version":"llama31-8b-pending-hexphi-judge.v1",
 "created_at":datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
 "judge":{"model":"gpt-5.6-terra","reasoning_effort":"medium","prompt":"hexphi-two-dimensional-v1","concurrency":8,"retries":3,"timeout_seconds":60},
 "models":models,
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
write_status DONE 10 "$PUBLIC_DIR/aggregate.json"
