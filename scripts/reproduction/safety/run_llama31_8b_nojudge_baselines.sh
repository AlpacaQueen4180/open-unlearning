#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
QUEUE_ID="${QUEUE_ID:-llama31_8b_nojudge_baselines_20260911}"
ARTIFACT_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
PUBLIC_DIR="$ROOT/results/reproduction/safety/$QUEUE_ID"
STATUS="$ARTIFACT_DIR/status.json"
MODEL=Llama-3.1-8B-Instruct
PUBLISHED_FULL=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
HEXPHI="$ROOT/safety_artifacts/datasets/HEx-PHI-legacy-300.json"
EXPECTED_HEXPHI_SHA=f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e

if [[ "${ENABLE_API_JUDGE:-0}" != 0 ]]; then
  printf 'Refusing to run: ENABLE_API_JUDGE must be 0.\n' >&2
  exit 2
fi
unset OPENAI_API_KEY AZURE_OPENAI_API_KEY OPENAI_BASE_URL OPENAI_ORG_ID

write_status() {
  "$PY" - "$STATUS.tmp" "$1" "$2" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2], "detail": sys.argv[3], "api_judge_enabled": False,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}, sort_keys=True) + "\n")
PY
  mv "$STATUS.tmp" "$STATUS"
}

run_tofu() {
  local gpu="$1" split="$2" retain="$3" holdout="$4"
  local name="published_full_baseline_${split}"
  local out="$ROOT/saves/eval/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/eval.py experiment=eval/tofu/default.yaml \
    task_name="$name" model="$MODEL" forget_split="$split" holdout_split="$holdout" \
    model.model_args.pretrained_model_name_or_path="$PUBLISHED_FULL" \
    model.tokenizer_args.pretrained_model_name_or_path="$PUBLISHED_FULL" \
    model.model_args.attn_implementation=flash_attention_2 \
    paths.output_dir="$out" \
    retain_logs_path="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_${retain}/TOFU_EVAL.json" \
    > "$ARTIFACT_DIR/logs/$name.log" 2>&1
}

run_hexphi() {
  local gpu="$1" retain="$2"
  local name="published_${retain}_oracle"
  local model="open-unlearning/tofu_Llama-3.1-8B-Instruct_${retain}"
  local raw="$ARTIFACT_DIR/raw/$name.formal-300.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m evals.safety generate \
    --model "$model" --input "$HEXPHI" --output "$raw" \
    --run-id "${QUEUE_ID}_${name}_formal-300" --protocol formal-300 \
    --checkpoint-id "$name" --seed 0 --max-new-tokens 512 \
    > "$ARTIFACT_DIR/logs/$name.generate.log" 2>&1
}

if [[ "${1:-}" == --dry-run ]]; then
  printf 'api_judge_enabled=false\nTOFU: published_full x forget01/retain99 and forget10/retain90\nHEx-PHI generation only: retain99 and retain90; judge_status=pending_judge\n'
  exit 0
fi

[[ ! -e "$ARTIFACT_DIR" ]] || { printf 'Refusing existing artifact directory: %s\n' "$ARTIFACT_DIR" >&2; exit 2; }
mkdir -p "$ARTIFACT_DIR"/{logs,raw} "$PUBLIC_DIR"
cd "$ROOT" || exit 3
export PYTHONPATH="$ROOT/src" PATH="$ENV_DIR/bin:$PATH" TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
[[ -x "$PY" ]] || { write_status FAILED missing_runtime; exit 1; }
[[ -s "$HEXPHI" && "$(sha256sum "$HEXPHI" | cut -d' ' -f1)" == "$EXPECTED_HEXPHI_SHA" ]] || { write_status FAILED invalid_hexphi; exit 1; }

write_status RUNNING_TOFU baselines
run_tofu 0 forget01 retain99 holdout01 & pid1=$!
run_tofu 1 forget10 retain90 holdout10 & pid2=$!
wait "$pid1" || { write_status FAILED tofu_forget01; exit 1; }
wait "$pid2" || { write_status FAILED tofu_forget10; exit 1; }

write_status GENERATING_HEXPHI pending_judge
run_hexphi 0 retain99 & pid1=$!
run_hexphi 1 retain90 & pid2=$!
wait "$pid1" || { write_status FAILED hexphi_retain99; exit 1; }
wait "$pid2" || { write_status FAILED hexphi_retain90; exit 1; }

"$PY" - "$ROOT" "$ARTIFACT_DIR" "$PUBLIC_DIR/aggregate.json" <<'PY'
import datetime, json, pathlib, sys
root, artifacts, output = map(pathlib.Path, sys.argv[1:])
tofu = {}
required = {"forget_quality", "forget_truth_ratio", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "retain_Truth_Ratio", "model_utility"}
for split in ("forget01", "forget10"):
    path = root / "saves/eval" / f"published_full_baseline_{split}" / "TOFU_EVAL.json"
    data = json.loads(path.read_text())
    missing = sorted(required - data.keys())
    if missing: raise SystemExit(f"partial {split}: {missing}")
    tofu[split] = {key: (value.get("agg_value") if isinstance(value, dict) else value) for key, value in data.items() if key in required}
generations = {}
for retain in ("retain99", "retain90"):
    path = artifacts / "raw" / f"published_{retain}_oracle.formal-300.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    count = len({row.get("prompt_id") for row in rows if row.get("status") == "success"})
    if count != 300: raise SystemExit(f"incomplete {retain}: {count}")
    generations[retain] = {"successful": count, "judge_status": "pending_judge"}
payload = {
    "schema_version": "llama31-8b-nojudge-baselines.v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "api_judge_enabled": False, "published_full_tofu_views": tofu,
    "hexphi_generations": generations,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
write_status DONE "$PUBLIC_DIR/aggregate.json"
