#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV_DIR="${ENV_DIR:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PY="$ENV_DIR/bin/python"
ACCELERATE="$ENV_DIR/bin/accelerate"
QUEUE_ID="${QUEUE_ID:-llama31_8b_tofu_rebuild_npo_seed0_20260907}"
ARTIFACT_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
PUBLIC_DIR="$ROOT/results/reproduction/safety/$QUEUE_ID"
QUEUE_STATUS="$ARTIFACT_DIR/queue.status.json"
LEDGER="$ARTIFACT_DIR/ledger.tsv"
CLAIM="$ROOT/.${QUEUE_ID}.gpu-claim"
MODEL=Llama-3.1-8B-Instruct
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
PUBLISHED_FULL=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
HEXPHI="$ROOT/safety_artifacts/datasets/HEx-PHI-legacy-300.json"
SECRET_ENV=/home/ai/.config/open-unlearning/safety.env
MIN_FREE_KB=188743680
EXPECTED_HEXPHI_SHA=f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e

LOCAL_RETAIN_TASK="${QUEUE_ID}_local_retain95"
LOCAL_FULL_TASK="${QUEUE_ID}_local_full"
LOCAL_RETAIN_DIR="$ROOT/saves/finetune/$LOCAL_RETAIN_TASK"
LOCAL_FULL_DIR="$ROOT/saves/finetune/$LOCAL_FULL_TASK"
LOCAL_RETAIN_EVAL="$ROOT/saves/eval/$LOCAL_RETAIN_TASK/TOFU_EVAL.json"

cell_names=(
  local_retain95
  local_full
  local_full_forget05_npo_no_retain
  local_full_forget05_npo_retain95
  published_full_forget01_npo_no_retain
  published_full_forget01_npo_retain99
  published_full_forget10_npo_no_retain
  published_full_forget10_npo_retain90
)

write_status() {
  local state="$1" cell="$2" completed="$3" stage="$4" detail="${5:-}"
  "$PY" - "$QUEUE_STATUS.tmp" "$state" "$cell" "$completed" "$stage" "$detail" "$QUEUE_ID" <<'PY'
import datetime, json, pathlib, sys
path, state, cell, completed, stage, detail, queue = sys.argv[1:]
payload = {
    "status": state, "queue": queue, "cell": int(cell), "total": 8,
    "completed": int(completed), "stage": stage, "detail": detail,
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
  write_status WAITING_FOR_TWO_IDLE_GPUS 0 0 preflight ""
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

validate_checkpoint() {
  local checkpoint="$1"
  [[ -s "$checkpoint/trainer_state.json" ]] || return 1
  find "$checkpoint" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q .
}

write_manifest() {
  local name="$1" kind="$2" checkpoint="$3" forget="$4" retain="$5" alpha="$6"
  local manifest="$ARTIFACT_DIR/manifests/$name.json"
  "$PY" - "$manifest" "$name" "$kind" "$checkpoint" "$forget" "$retain" "$alpha" <<'PY'
import datetime, hashlib, json, pathlib, sys
out, name, kind, checkpoint, forget, retain, alpha = sys.argv[1:]
root = pathlib.Path(checkpoint)
state = json.loads((root / "trainer_state.json").read_text())
shards = []
for path in sorted(root.glob("*.safetensors")):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    shards.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
payload = {
    "name": name, "kind": kind, "checkpoint": str(root), "seed": 0,
    "forget_split": forget or None, "retain_split": retain or None,
    "alpha": None if alpha == "" else float(alpha), "optimizer_steps": state.get("global_step"),
    "safetensors": shards, "checkpoint_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    "created_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_accelerate() {
  local log="$1"; shift
  local master_port
  master_port="$($PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")"
  "$ACCELERATE" launch --config_file configs/accelerate/default_config.yaml \
    --main_process_port "$master_port" "$@" > "$log" 2>&1
}

run_finetune() {
  local name="$1" dataset="$2" split="$3" task="$4" log="$5"
  local dataset_override
  if [[ "$dataset" == full ]]; then
    dataset_override=TOFU_QA_full
  else
    dataset_override=TOFU_QA_retain
  fi
  run_accelerate "$log" src/train.py experiment=finetune/tofu/default.yaml \
    task_name="$task" model="$MODEL" \
    data/datasets@data.train="$dataset_override" \
    data.train."$dataset_override".args.hf_args.name="$split" \
    model.model_args.pretrained_model_name_or_path="$BASE_MODEL" \
    model.tokenizer_args.pretrained_model_name_or_path="$BASE_MODEL" \
    model.model_args.attn_implementation=flash_attention_2 \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=5 trainer.args.learning_rate=1e-5 \
    trainer.args.weight_decay=0.01 trainer.args.warmup_epochs=1.0 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true trainer.args.seed=0
}

run_npo() {
  local task="$1" model_path="$2" forget="$3" retain="$4" alpha="$5" retain_log="$6" log="$7"
  run_accelerate "$log" src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml trainer=NPO task_name="$task" model="$MODEL" \
    forget_split="$forget" retain_split="$retain" \
    model.model_args.pretrained_model_name_or_path="$model_path" \
    model.tokenizer_args.pretrained_model_name_or_path="$model_path" \
    model.model_args.attn_implementation=flash_attention_2 retain_logs_path="$retain_log" \
    trainer.method_args.alpha="$alpha" trainer.method_args.gamma=1.0 trainer.method_args.beta=0.1 \
    trainer.args.per_device_train_batch_size=4 trainer.args.gradient_accumulation_steps=4 \
    trainer.args.num_train_epochs=10 trainer.args.learning_rate=1e-5 trainer.args.weight_decay=0.01 \
    trainer.args.ddp_find_unused_parameters=true trainer.args.gradient_checkpointing=true trainer.args.seed=0
}

run_tofu_eval() {
  local name="$1" checkpoint="$2" forget="$3" holdout="$4" retain_log="$5"
  local out="$ROOT/saves/eval/$name"
  local log="$ARTIFACT_DIR/logs/$name.tofu.log"
  local args=(src/eval.py experiment=eval/tofu/default.yaml task_name="$name" model="$MODEL"
    forget_split="$forget" holdout_split="$holdout"
    model.model_args.pretrained_model_name_or_path="$checkpoint"
    model.tokenizer_args.pretrained_model_name_or_path="$checkpoint"
    model.model_args.attn_implementation=flash_attention_2 paths.output_dir="$out")
  if [[ -n "$retain_log" ]]; then args+=(retain_logs_path="$retain_log"); fi
  CUDA_VISIBLE_DEVICES=0 "$PY" "${args[@]}" > "$log" 2>&1
  [[ -s "$out/TOFU_EVAL.json" ]]
}

run_safety() {
  local name="$1" checkpoint="$2"
  local raw="$ARTIFACT_DIR/raw/$name.formal-300.jsonl"
  local judged="$ARTIFACT_DIR/judged/$name.formal-300.jsonl"
  local summary="$ARTIFACT_DIR/summaries/$name.formal-300.json"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m evals.safety generate \
    --model "$checkpoint" --input "$HEXPHI" --output "$raw" \
    --run-id "${QUEUE_ID}_${name}_formal-300" --protocol formal-300 \
    --checkpoint-id "$name" --seed 0 --max-new-tokens 512 \
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

append_ledger() {
  local cell="$1" name="$2" kind="$3" forget="$4" retain="$5" alpha="$6" checkpoint="$7" retain_log="$8"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "$name" "$kind" "$forget" "$retain" "$alpha" "$checkpoint" \
    "$ROOT/saves/eval/$name/TOFU_EVAL.json" \
    "$ARTIFACT_DIR/summaries/$name.formal-300.json" "$retain_log" >> "$LEDGER"
}

if [[ "${1:-}" == --dry-run ]]; then
  printf 'cell\tname\tkind\tforget\tretain\talpha\tbase\n'
  printf '1\tlocal_retain95\tfinetune\tforget05\tretain95\t\t%s\n' "$BASE_MODEL"
  printf '2\tlocal_full\tfinetune\tforget05\tretain95\t\t%s\n' "$BASE_MODEL"
  printf '3\tlocal_full_forget05_npo_no_retain\tnpo\tforget05\tretain95\t0\t%s\n' "$LOCAL_FULL_DIR"
  printf '4\tlocal_full_forget05_npo_retain95\tnpo\tforget05\tretain95\t1\t%s\n' "$LOCAL_FULL_DIR"
  printf '5\tpublished_full_forget01_npo_no_retain\tnpo\tforget01\tretain99\t0\t%s\n' "$PUBLISHED_FULL"
  printf '6\tpublished_full_forget01_npo_retain99\tnpo\tforget01\tretain99\t1\t%s\n' "$PUBLISHED_FULL"
  printf '7\tpublished_full_forget10_npo_no_retain\tnpo\tforget10\tretain90\t0\t%s\n' "$PUBLISHED_FULL"
  printf '8\tpublished_full_forget10_npo_retain90\tnpo\tforget10\tretain90\t1\t%s\n' "$PUBLISHED_FULL"
  exit 0
fi

if [[ -e "$ARTIFACT_DIR" ]]; then
  printf 'Refusing existing queue directory: %s\n' "$ARTIFACT_DIR" >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR"/{logs,status,manifests,raw,judged,summaries} "$PUBLIC_DIR"
cd "$ROOT" || exit 3
export CUDA_VISIBLE_DEVICES=0,1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src" PATH="$ENV_DIR/bin:$PATH"

[[ -x "$PY" && -x "$ACCELERATE" ]] || fail FAILED_PREFLIGHT 0 0 environment missing_runtime
[[ -s "$HEXPHI" ]] || fail FAILED_PREFLIGHT 0 0 dataset missing_hexphi
[[ "$(sha256sum "$HEXPHI" | cut -d' ' -f1)" == "$EXPECTED_HEXPHI_SHA" ]] || fail FAILED_PREFLIGHT 0 0 dataset checksum_mismatch
[[ -f "$SECRET_ENV" && "$(stat -c '%a' "$SECRET_ENV")" == 600 ]] || fail FAILED_PREFLIGHT 0 0 secret invalid_env_file
[[ -s "$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain99/TOFU_EVAL.json" ]] || fail FAILED_PREFLIGHT 0 0 reference missing_retain99
[[ -s "$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain90/TOFU_EVAL.json" ]] || fail FAILED_PREFLIGHT 0 0 reference missing_retain90
available_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
(( available_kb >= MIN_FREE_KB )) || fail FAILED_LOW_DISK 0 0 preflight "$available_kb"
mkdir "$CLAIM" 2>/dev/null || fail REFUSED_GPU_CLAIMED 0 0 preflight "$CLAIM"
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT
printf 'cell\tname\tkind\tforget\tretain\talpha\tcheckpoint\ttofu_eval\tsafety_summary\tretain_reference\n' > "$LEDGER"
wait_for_idle_gpus

completed=0
for cell in $(seq 1 8); do
  name="${cell_names[$((cell - 1))]}"
  kind=npo; forget=forget05; holdout=holdout05; retain=retain95; alpha=; base=; retain_log=; task="$name"
  case "$cell" in
    1) kind=finetune; task="$LOCAL_RETAIN_TASK"; base="$BASE_MODEL"; checkpoint="$LOCAL_RETAIN_DIR" ;;
    2) kind=finetune; task="$LOCAL_FULL_TASK"; base="$BASE_MODEL"; checkpoint="$LOCAL_FULL_DIR"; retain_log="$LOCAL_RETAIN_EVAL" ;;
    3) alpha=0; base="$LOCAL_FULL_DIR"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$LOCAL_RETAIN_EVAL" ;;
    4) alpha=1; base="$LOCAL_FULL_DIR"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$LOCAL_RETAIN_EVAL" ;;
    5) forget=forget01; holdout=holdout01; retain=retain99; alpha=0; base="$PUBLISHED_FULL"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain99/TOFU_EVAL.json" ;;
    6) forget=forget01; holdout=holdout01; retain=retain99; alpha=1; base="$PUBLISHED_FULL"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain99/TOFU_EVAL.json" ;;
    7) forget=forget10; holdout=holdout10; retain=retain90; alpha=0; base="$PUBLISHED_FULL"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain90/TOFU_EVAL.json" ;;
    8) forget=forget10; holdout=holdout10; retain=retain90; alpha=1; base="$PUBLISHED_FULL"; checkpoint="$ROOT/saves/unlearn/$name"; retain_log="$ROOT/saves/eval/tofu_Llama-3.1-8B-Instruct_retain90/TOFU_EVAL.json" ;;
  esac
  [[ ! -e "$checkpoint" && ! -e "$ROOT/saves/eval/$name" ]] || fail REFUSED_EXISTING_CELL "$cell" "$completed" training "$name"
  write_status TRAINING "$cell" "$completed" training "$name"
  if [[ "$cell" == 1 ]]; then
    run_finetune "$name" retain retain95 "$task" "$ARTIFACT_DIR/logs/$name.train.log" || fail FAILED_TRAIN "$cell" "$completed" training "$name"
  elif [[ "$cell" == 2 ]]; then
    run_finetune "$name" full full "$task" "$ARTIFACT_DIR/logs/$name.train.log" || fail FAILED_TRAIN "$cell" "$completed" training "$name"
  else
    run_npo "$task" "$base" "$forget" "$retain" "$alpha" "$retain_log" "$ARTIFACT_DIR/logs/$name.train.log" || fail FAILED_TRAIN "$cell" "$completed" training "$name"
  fi
  validate_checkpoint "$checkpoint" || fail FAILED_SAVE "$cell" "$completed" checkpoint_validation "$name"
  write_manifest "$name" "$kind" "$checkpoint" "$forget" "$retain" "$alpha" || fail FAILED_MANIFEST "$cell" "$completed" manifest "$name"
  write_status EVALUATING_TOFU "$cell" "$completed" tofu "$name"
  run_tofu_eval "$name" "$checkpoint" "$forget" "$holdout" "$retain_log" || fail FAILED_TOFU_EVAL "$cell" "$completed" tofu "$name"
  write_status EVALUATING_SAFETY "$cell" "$completed" safety "$name"
  run_safety "$name" "$checkpoint" || fail FAILED_SAFETY_EVAL "$cell" "$completed" safety "$name"
  append_ledger "$cell" "$name" "$kind" "$forget" "$retain" "$alpha" "$checkpoint" "$retain_log"
  completed=$((completed + 1))
  write_status CELL_DONE "$cell" "$completed" complete "$name"
done

write_status SUMMARIZING 8 8 summary ""
"$PY" scripts/reproduction/safety/summarize_llama31_8b_tofu_rebuild_npo_seed0.py \
  --root "$ROOT" --queue-dir "$ARTIFACT_DIR" --output "$PUBLIC_DIR/aggregate.json" \
  > "$ARTIFACT_DIR/logs/summarize.log" 2>&1 || fail FAILED_SUMMARY 8 8 summary aggregate
cp "$LEDGER" "$PUBLIC_DIR/ledger.tsv"
write_status DONE 8 8 complete "$PUBLIC_DIR/aggregate.json"

