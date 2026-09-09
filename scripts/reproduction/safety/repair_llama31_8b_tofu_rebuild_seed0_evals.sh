#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
QUEUE_ID=llama31_8b_tofu_rebuild_npo_seed0_20260907
QUEUE_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
PUBLIC_DIR="$ROOT/results/reproduction/safety/$QUEUE_ID"
REFERENCE="$ROOT/saves/eval/local_retain95/TOFU_EVAL.json"
LEDGER="$QUEUE_DIR/ledger.corrected.tsv"
MODEL=Llama-3.1-8B-Instruct

cd "$ROOT"
export PYTHONPATH="$ROOT/src" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
[[ -s "$REFERENCE" ]]
if pgrep -f 'src/train.py|src/eval.py' >/dev/null; then
  printf 'Refusing repair while another train/eval process is active.\n' >&2
  exit 2
fi

run_eval() {
  local gpu="$1" name="$2" checkpoint="$3"
  local out="$ROOT/saves/eval/${name}_corrected"
  [[ ! -e "$out" ]] || { printf 'Refusing existing corrected output: %s\n' "$out" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/eval.py experiment=eval/tofu/default.yaml \
    task_name="${name}_corrected" model="$MODEL" forget_split=forget05 holdout_split=holdout05 \
    model.model_args.pretrained_model_name_or_path="$checkpoint" \
    model.tokenizer_args.pretrained_model_name_or_path="$checkpoint" \
    model.model_args.attn_implementation=flash_attention_2 \
    retain_logs_path="$REFERENCE" paths.output_dir="$out" \
    > "$QUEUE_DIR/logs/${name}.tofu-corrected.log" 2>&1
  "$PY" - "$out/TOFU_EVAL.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
required = {"forget_quality", "forget_truth_ratio", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "retain_Truth_Ratio", "model_utility"}
data = json.loads(path.read_text())
missing = sorted(required - data.keys())
if missing:
    raise SystemExit(f"partial TOFU evaluation {path}: missing {missing}")
PY
}

run_eval 0 local_full "$ROOT/saves/finetune/${QUEUE_ID}_local_full" &
pid_full=$!
run_eval 1 local_full_forget05_npo_no_retain "$ROOT/saves/unlearn/local_full_forget05_npo_no_retain" &
pid_npo=$!
wait "$pid_full"
wait "$pid_npo"
run_eval 0 local_full_forget05_npo_retain95 "$ROOT/saves/unlearn/local_full_forget05_npo_retain95"

"$PY" - "$QUEUE_DIR/ledger.tsv" "$LEDGER" <<'PY'
import csv, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
fields = list(rows[0])
for row in rows:
    if row["name"] in {"local_full", "local_full_forget05_npo_no_retain", "local_full_forget05_npo_retain95"}:
        row["tofu_eval"] = str(source.parent.parent.parent / "saves" / "eval" / f'{row["name"]}_corrected' / "TOFU_EVAL.json")
        row["retain_reference"] = str(source.parent.parent.parent / "saves" / "eval" / "local_retain95" / "TOFU_EVAL.json")
with target.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
PY

"$PY" scripts/reproduction/safety/summarize_llama31_8b_tofu_rebuild_npo_seed0.py \
  --root "$ROOT" --queue-dir "$QUEUE_DIR" --ledger "$LEDGER" \
  --output "$PUBLIC_DIR/aggregate.corrected.json" \
  > "$QUEUE_DIR/logs/summarize-corrected.log" 2>&1
cp "$LEDGER" "$PUBLIC_DIR/ledger.corrected.tsv"
