#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
ACCELERATE="$ENV_DIR/bin/accelerate"
QUEUE_ID="${QUEUE_ID:-llama31_8b_npo_seeds1_4_20260909}"
ARTIFACT_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
PUBLIC_DIR="$ROOT/results/reproduction/safety/$QUEUE_ID"
QUEUE_STATUS="$ARTIFACT_DIR/queue.status.json"
LEDGER="$ARTIFACT_DIR/ledger.tsv"
CLAIM="$ROOT/.${QUEUE_ID}.gpu-claim"

MODEL=Llama-3.1-8B-Instruct
LOCAL_FULL="$ROOT/saves/finetune/llama31_8b_tofu_rebuild_npo_seed0_20260907_local_full"
PUBLISHED_FULL=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
LOCAL_RETAIN_EVAL="$ROOT/saves/eval/local_retain95/TOFU_EVAL.json"
PUBLISHED_RETAIN99_EVAL="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain99/TOFU_EVAL.json"
PUBLISHED_RETAIN90_EVAL="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain90/TOFU_EVAL.json"

HEXPHI="$ROOT/safety_artifacts/datasets/HEx-PHI-legacy-300.json"
EXPECTED_HEXPHI_SHA=f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e
SECRET_ENV=/home/ai/.config/open-unlearning/safety.env

GB10_HOST=140.113.13.133
GB10_PORT=22026
GB10_USER=gb10
GB10_KEY=/home/ai/.ssh/open_unlearning_to_gb10
GB10_ARCHIVE="/home/gb10/alpaca/checkpoint_archive/$QUEUE_ID"
SSH_GB10=(ssh -i "$GB10_KEY" -o BatchMode=yes -o ConnectTimeout=15 -p "$GB10_PORT" "$GB10_USER@$GB10_HOST")
RSYNC_SSH="ssh -i $GB10_KEY -o BatchMode=yes -o ConnectTimeout=15 -p $GB10_PORT"

TOTAL_CELLS=24
RESUME_FROM_CELL="${RESUME_FROM_CELL:-1}"
H100_MIN_FREE_KB=$((60 * 1024 * 1024))
# 24 * 16,077,917,649 bytes plus 100 GiB of reserve on GB10.
GB10_REQUIRED_FREE_KB=481683795

write_status() {
  local state="$1" cell="$2" completed="$3" stage="$4" detail="${5:-}"
  "$PY" - "$QUEUE_STATUS.tmp" "$state" "$cell" "$completed" "$stage" "$detail" "$QUEUE_ID" "$TOTAL_CELLS" <<'PY'
import datetime, json, pathlib, sys
path, state, cell, completed, stage, detail, queue, total = sys.argv[1:]
payload = {
    "status": state,
    "queue": queue,
    "cell": int(cell),
    "total": int(total),
    "completed": int(completed),
    "stage": stage,
    "detail": detail,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
pathlib.Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n")
PY
  mv "$QUEUE_STATUS.tmp" "$QUEUE_STATUS"
}

fail() {
  local state="$1" cell="$2" completed="$3" stage="$4" detail="${5:-}"
  write_status "$state" "$cell" "$completed" "$stage" "$detail"
  exit 1
}

wait_for_idle_gpus() {
  local idle_checks=0 process_count
  write_status WAITING_FOR_TWO_IDLE_GPUS "$RESUME_FROM_CELL" "$completed" preflight ""
  while (( idle_checks < 4 )); do
    process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {count++} END {print count+0}')"
    if [[ "$process_count" == 0 ]]; then
      idle_checks=$((idle_checks + 1))
    else
      idle_checks=0
    fi
    if (( idle_checks < 4 )); then sleep 15; fi
  done
}

resolve_cell() {
  local cell="$1"
  seed=$((1 + (cell - 1) / 6))
  variant=$((1 + (cell - 1) % 6))
  case "$variant" in
    1) stem=local_full_forget05_npo_no_retain; forget=forget05; holdout=holdout05; retain=retain95; alpha=0; base="$LOCAL_FULL"; retain_log="$LOCAL_RETAIN_EVAL" ;;
    2) stem=local_full_forget05_npo_retain95; forget=forget05; holdout=holdout05; retain=retain95; alpha=1; base="$LOCAL_FULL"; retain_log="$LOCAL_RETAIN_EVAL" ;;
    3) stem=published_full_forget01_npo_no_retain; forget=forget01; holdout=holdout01; retain=retain99; alpha=0; base="$PUBLISHED_FULL"; retain_log="$PUBLISHED_RETAIN99_EVAL" ;;
    4) stem=published_full_forget01_npo_retain99; forget=forget01; holdout=holdout01; retain=retain99; alpha=1; base="$PUBLISHED_FULL"; retain_log="$PUBLISHED_RETAIN99_EVAL" ;;
    5) stem=published_full_forget10_npo_no_retain; forget=forget10; holdout=holdout10; retain=retain90; alpha=0; base="$PUBLISHED_FULL"; retain_log="$PUBLISHED_RETAIN90_EVAL" ;;
    6) stem=published_full_forget10_npo_retain90; forget=forget10; holdout=holdout10; retain=retain90; alpha=1; base="$PUBLISHED_FULL"; retain_log="$PUBLISHED_RETAIN90_EVAL" ;;
  esac
  name="${stem}_seed${seed}"
  checkpoint="$ROOT/saves/unlearn/$name"
}

validate_checkpoint() {
  local checkpoint_dir="$1"
  [[ -s "$checkpoint_dir/trainer_state.json" ]] || return 1
  find "$checkpoint_dir" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q .
}

write_manifest() {
  local name="$1" checkpoint_dir="$2" seed="$3" forget="$4" retain="$5" alpha="$6"
  local manifest="$ARTIFACT_DIR/manifests/$name.json"
  "$PY" - "$manifest" "$name" "$checkpoint_dir" "$seed" "$forget" "$retain" "$alpha" <<'PY'
import datetime, hashlib, json, pathlib, sys
out, name, checkpoint, seed, forget, retain, alpha = sys.argv[1:]
root = pathlib.Path(checkpoint)
state = json.loads((root / "trainer_state.json").read_text())
files = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    files.append({
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
payload = {
    "name": name,
    "kind": "npo",
    "checkpoint": str(root),
    "seed": int(seed),
    "forget_split": forget,
    "retain_split": retain,
    "alpha": float(alpha),
    "optimizer_steps": state.get("global_step"),
    "files": files,
    "checkpoint_bytes": sum(item["bytes"] for item in files),
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_accelerate() {
  local log="$1"
  shift
  local master_port
  master_port="$($PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")"
  "$ACCELERATE" launch --config_file configs/accelerate/default_config.yaml \
    --main_process_port "$master_port" "$@" > "$log" 2>&1
}

run_npo() {
  local name="$1" model_path="$2" forget="$3" retain="$4" alpha="$5" retain_log="$6" seed="$7" log="$8"
  run_accelerate "$log" src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml trainer=NPO task_name="$name" model="$MODEL" \
    forget_split="$forget" retain_split="$retain" \
    model.model_args.pretrained_model_name_or_path="$model_path" \
    model.tokenizer_args.pretrained_model_name_or_path="$model_path" \
    model.model_args.attn_implementation=flash_attention_2 retain_logs_path="$retain_log" \
    trainer.method_args.alpha="$alpha" trainer.method_args.gamma=1.0 trainer.method_args.beta=0.1 \
    trainer.args.per_device_train_batch_size=4 trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=10 trainer.args.learning_rate=1e-5 trainer.args.weight_decay=0.01 \
    trainer.args.ddp_find_unused_parameters=true trainer.args.gradient_checkpointing=true trainer.args.seed="$seed"
}

run_tofu_eval() {
  local name="$1" checkpoint_dir="$2" forget="$3" holdout="$4" retain_log="$5"
  local out="$ROOT/saves/eval/$name"
  local log="$ARTIFACT_DIR/logs/$name.tofu.log"
  CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py experiment=eval/tofu/default.yaml \
    task_name="$name" model="$MODEL" forget_split="$forget" holdout_split="$holdout" \
    model.model_args.pretrained_model_name_or_path="$checkpoint_dir" \
    model.tokenizer_args.pretrained_model_name_or_path="$checkpoint_dir" \
    model.model_args.attn_implementation=flash_attention_2 \
    paths.output_dir="$out" retain_logs_path="$retain_log" > "$log" 2>&1 || return 1
  "$PY" - "$out/TOFU_EVAL.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
required = {
    "forget_quality", "forget_truth_ratio", "forget_Q_A_Prob",
    "forget_Q_A_ROUGE", "retain_Truth_Ratio", "model_utility",
}
data = json.loads(path.read_text())
missing = sorted(required - data.keys())
if missing:
    raise SystemExit(f"partial TOFU evaluation {path}: missing {missing}")
PY
}

run_safety() {
  local name="$1" checkpoint_dir="$2" seed="$3"
  local raw="$ARTIFACT_DIR/raw/$name.formal-300.jsonl"
  local judged="$ARTIFACT_DIR/judged/$name.formal-300.jsonl"
  local summary="$ARTIFACT_DIR/summaries/$name.formal-300.json"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m evals.safety generate \
    --model "$checkpoint_dir" --input "$HEXPHI" --output "$raw" \
    --run-id "${QUEUE_ID}_${name}_formal-300" --protocol formal-300 \
    --checkpoint-id "$name" --seed "$seed" --max-new-tokens 512 \
    > "$ARTIFACT_DIR/logs/$name.generate.log" 2>&1 || return 1
  [[ "$(wc -l < "$raw")" -eq 300 ]] || return 1
  "$PY" -m evals.safety judge --input "$raw" --output "$judged" \
    --model gpt-5.6-terra --reasoning-effort medium --concurrency 8 \
    --retries 3 --timeout 60 --env-file "$SECRET_ENV" \
    > "$ARTIFACT_DIR/logs/$name.judge.log" 2>&1 || return 1
  "$PY" -m evals.safety summarize --input "$judged" --output "$summary" \
    > "$ARTIFACT_DIR/logs/$name.summary.log" 2>&1 || return 1
  [[ -s "$summary" ]]
}

archive_checkpoint() {
  local name="$1" checkpoint_dir="$2" manifest="$3"
  local incoming="$GB10_ARCHIVE/.incoming/${name}.${QUEUE_ID}"
  local final="$GB10_ARCHIVE/checkpoints/$name"
  local remote_manifest="$GB10_ARCHIVE/manifests/$name.json"

  "${SSH_GB10[@]}" "test ! -e '$final' && rm -rf -- '$incoming' && mkdir -p '$incoming' '$GB10_ARCHIVE/manifests' '$GB10_ARCHIVE/checkpoints'" || return 1
  rsync -a --partial -e "$RSYNC_SSH" "$checkpoint_dir/" "$GB10_USER@$GB10_HOST:$incoming/" || return 1
  rsync -a -e "$RSYNC_SSH" "$manifest" "$GB10_USER@$GB10_HOST:$remote_manifest" || return 1
  "${SSH_GB10[@]}" python3 - "$incoming" "$remote_manifest" > "$ARTIFACT_DIR/logs/$name.archive-verify.log" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise SystemExit(f"archive size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise SystemExit(f"archive checksum mismatch: {path}")
print("verified", manifest["name"], manifest["checkpoint_bytes"])
PY
  "${SSH_GB10[@]}" "mv '$incoming' '$final'" || return 1

  local resolved
  resolved="$(realpath -m "$checkpoint_dir")"
  case "$resolved" in
    "$ROOT/saves/unlearn/"*) rm -rf -- "$resolved" ;;
    *) printf 'Refusing to remove unexpected path: %s\n' "$resolved" >&2; return 1 ;;
  esac
  printf '%s' "$final"
}

append_ledger() {
  local cell="$1" name="$2" seed="$3" forget="$4" retain="$5" alpha="$6" archive="$7" retain_log="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "$name" "$seed" "$forget" "$retain" "$alpha" ARCHIVED "$archive" \
    "$ROOT/saves/eval/$name/TOFU_EVAL.json" \
    "$ARTIFACT_DIR/summaries/$name.formal-300.json" "$retain_log" >> "$LEDGER"
}

if [[ "${1:-}" == --dry-run ]]; then
  printf 'cell\tname\tseed\tforget\tretain\talpha\tbase\n'
  for cell in $(seq 1 "$TOTAL_CELLS"); do
    resolve_cell "$cell"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$name" "$seed" "$forget" "$retain" "$alpha" "$base"
  done
  exit 0
fi

if ! [[ "$RESUME_FROM_CELL" =~ ^([1-9]|1[0-9]|2[0-4])$ ]]; then
  printf 'RESUME_FROM_CELL must be an integer from 1 through 24\n' >&2
  exit 2
fi
if [[ "$RESUME_FROM_CELL" == 1 && -e "$ARTIFACT_DIR" ]]; then
  printf 'Refusing existing queue directory: %s\n' "$ARTIFACT_DIR" >&2
  exit 2
fi
if [[ "$RESUME_FROM_CELL" != 1 && ! -s "$LEDGER" ]]; then
  printf 'Cannot resume without an existing ledger: %s\n' "$LEDGER" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"/{logs,manifests,raw,judged,summaries} "$PUBLIC_DIR"
cd "$ROOT" || exit 3
export CUDA_VISIBLE_DEVICES=0,1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src" PATH="$ENV_DIR/bin:$PATH"

[[ -x "$PY" && -x "$ACCELERATE" ]] || fail FAILED_PREFLIGHT 0 0 environment missing_runtime
[[ -d "$LOCAL_FULL" ]] || fail FAILED_PREFLIGHT 0 0 model missing_local_full
[[ -s "$LOCAL_RETAIN_EVAL" && -s "$PUBLISHED_RETAIN99_EVAL" && -s "$PUBLISHED_RETAIN90_EVAL" ]] || fail FAILED_PREFLIGHT 0 0 reference missing_retain_eval
[[ -s "$HEXPHI" ]] || fail FAILED_PREFLIGHT 0 0 dataset missing_hexphi
[[ "$(sha256sum "$HEXPHI" | cut -d' ' -f1)" == "$EXPECTED_HEXPHI_SHA" ]] || fail FAILED_PREFLIGHT 0 0 dataset checksum_mismatch
[[ -f "$SECRET_ENV" && "$(stat -c '%a' "$SECRET_ENV")" == 600 ]] || fail FAILED_PREFLIGHT 0 0 secret invalid_env_file
[[ -f "$GB10_KEY" && "$(stat -c '%a' "$GB10_KEY")" == 600 ]] || fail FAILED_PREFLIGHT 0 0 archive invalid_ssh_key

h100_free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
(( h100_free_kb >= H100_MIN_FREE_KB )) || fail FAILED_LOW_DISK 0 0 preflight "h100:$h100_free_kb"
gb10_free_kb="$("${SSH_GB10[@]}" "mkdir -p '$GB10_ARCHIVE' && df -Pk '$GB10_ARCHIVE' | awk 'NR==2 {print \$4}'")" || fail FAILED_ARCHIVE_PREFLIGHT 0 0 archive unavailable
(( gb10_free_kb >= GB10_REQUIRED_FREE_KB )) || fail FAILED_LOW_DISK 0 0 preflight "gb10:$gb10_free_kb"

mkdir "$CLAIM" 2>/dev/null || fail REFUSED_GPU_CLAIMED 0 0 preflight "$CLAIM"
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT
if [[ "$RESUME_FROM_CELL" == 1 ]]; then
  printf 'cell\tname\tseed\tforget\tretain\talpha\tcheckpoint_state\tarchive_checkpoint\ttofu_eval\tsafety_summary\tretain_reference\n' > "$LEDGER"
  completed=0
else
  completed="$(awk 'NR > 1 {count++} END {print count+0}' "$LEDGER")"
  (( completed == RESUME_FROM_CELL - 1 )) || fail FAILED_RESUME_PREFLIGHT "$RESUME_FROM_CELL" "$completed" ledger unexpected_completed_count
fi

wait_for_idle_gpus

for cell in $(seq "$RESUME_FROM_CELL" "$TOTAL_CELLS"); do
  resolve_cell "$cell"
  [[ ! -e "$checkpoint" && ! -e "$ROOT/saves/eval/$name" ]] || fail REFUSED_EXISTING_CELL "$cell" "$completed" training "$name"
  h100_free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  (( h100_free_kb >= H100_MIN_FREE_KB )) || fail FAILED_LOW_DISK "$cell" "$completed" training "h100:$h100_free_kb"

  write_status TRAINING "$cell" "$completed" training "$name"
  run_npo "$name" "$base" "$forget" "$retain" "$alpha" "$retain_log" "$seed" "$ARTIFACT_DIR/logs/$name.train.log" \
    || fail FAILED_TRAIN "$cell" "$completed" training "$name"
  validate_checkpoint "$checkpoint" || fail FAILED_SAVE "$cell" "$completed" checkpoint_validation "$name"
  manifest="$ARTIFACT_DIR/manifests/$name.json"
  write_manifest "$name" "$checkpoint" "$seed" "$forget" "$retain" "$alpha" \
    || fail FAILED_MANIFEST "$cell" "$completed" manifest "$name"

  write_status EVALUATING_TOFU "$cell" "$completed" tofu "$name"
  run_tofu_eval "$name" "$checkpoint" "$forget" "$holdout" "$retain_log" \
    || fail FAILED_TOFU_EVAL "$cell" "$completed" tofu "$name"
  write_status EVALUATING_SAFETY "$cell" "$completed" safety "$name"
  run_safety "$name" "$checkpoint" "$seed" \
    || fail FAILED_SAFETY_EVAL "$cell" "$completed" safety "$name"

  write_status ARCHIVING "$cell" "$completed" archive "$name"
  archive_path="$(archive_checkpoint "$name" "$checkpoint" "$manifest")" \
    || fail FAILED_ARCHIVE "$cell" "$completed" archive "$name"
  append_ledger "$cell" "$name" "$seed" "$forget" "$retain" "$alpha" "$archive_path" "$retain_log"
  completed=$((completed + 1))
  write_status CELL_DONE "$cell" "$completed" complete "$name"
done

write_status SUMMARIZING "$TOTAL_CELLS" "$TOTAL_CELLS" summary ""
"$PY" scripts/reproduction/safety/summarize_llama31_8b_npo_seeds1_4.py \
  --ledger "$LEDGER" --output "$PUBLIC_DIR/aggregate.json" \
  > "$ARTIFACT_DIR/logs/summarize.log" 2>&1 \
  || fail FAILED_SUMMARY "$TOTAL_CELLS" "$TOTAL_CELLS" summary aggregate
cp "$LEDGER" "$PUBLIC_DIR/ledger.tsv"
write_status DONE "$TOTAL_CELLS" "$TOTAL_CELLS" complete "$PUBLIC_DIR/aggregate.json"
