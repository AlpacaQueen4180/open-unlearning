#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 MODEL FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT SEED TASK" >&2
  exit 2
fi

MODEL="$1"
FORGET_SPLIT="$2"
HOLDOUT_SPLIT="$3"
RETAIN_SPLIT="$4"
SEED="$5"
TASK="$6"

ROOT=/home/ai/alpaca
ENV=/home/ai/miniforge3/envs/open-unlearning-repro
PY="$ENV/bin/python"
ACCELERATE="$ENV/bin/accelerate"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TRAIN_DIR="$ROOT/saves/unlearn/$TASK"
EVAL_TASK=${TASK}_eval
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
TRAIN_LOG="$ROOT/${TASK}.log"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${TASK}.status.json"
TRAINER_STATE="$ROOT/${TASK}.trainer_state.json"
CLEANUP_STATUS="$ROOT/${TASK}.cleanup.status.json"
RETAIN_LOG="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

cd "$ROOT" || exit 2
if [[ ! "$SEED" =~ ^[1-4]$ ]]; then
  printf '{"status":"REFUSED_INVALID_SEED","task":"%s","seed":"%s"}\n' "$TASK" "$SEED" > "$STATUS"
  exit 3
fi
if [[ -e "$STATUS" || -e "$TRAIN_DIR" || -e "$EVAL_DIR" || -e "$TRAINER_STATE" ]]; then
  printf '{"status":"REFUSED_EXISTING_CELL","task":"%s"}\n' "$TASK" > "$ROOT/${TASK}.launch-refused.json"
  exit 4
fi
if [[ ! -s "$RETAIN_LOG" ]]; then
  printf '{"status":"REFUSED_MISSING_RETAIN_LOG","task":"%s","path":"%s"}\n' "$TASK" "$RETAIN_LOG" > "$STATUS"
  exit 5
fi

master_port="$($PY -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
printf '{"status":"TRAINING","task":"%s","model":"%s","forget_split":"%s","seed":%s,"master_port":%s}\n' \
  "$TASK" "$MODEL" "$FORGET_SPLIT" "$SEED" "$master_port" > "$STATUS"

export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

"$ACCELERATE" launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port "$master_port" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=NPO \
  task_name="$TASK" \
  model="$MODEL" \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
  retain_logs_path="$RETAIN_LOG" \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.seed="$SEED" \
  > "$TRAIN_LOG" 2>&1
train_rc=$?

if [[ "$train_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_TRAIN","task":"%s","seed":%s,"exit_code":%s,"log":"%s"}\n' \
    "$TASK" "$SEED" "$train_rc" "$TRAIN_LOG" > "$STATUS"
  exit "$train_rc"
fi
if ! find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q . || [[ ! -s "$TRAIN_DIR/trainer_state.json" ]]; then
  printf '{"status":"FAILED_SAVE","task":"%s","seed":%s,"log":"%s"}\n' "$TASK" "$SEED" "$TRAIN_LOG" > "$STATUS"
  exit 6
fi
cp -- "$TRAIN_DIR/trainer_state.json" "$TRAINER_STATE"

printf '{"status":"EVALUATING","task":"%s","model":"%s","forget_split":"%s","seed":%s}\n' \
  "$TASK" "$MODEL" "$FORGET_SPLIT" "$SEED" > "$STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default.yaml \
  model="$MODEL" \
  forget_split="$FORGET_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  task_name="$EVAL_TASK" \
  model.model_args.pretrained_model_name_or_path="$TRAIN_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
  retain_logs_path="$RETAIN_LOG" \
  seed="$SEED" \
  > "$EVAL_LOG" 2>&1
eval_rc=$?

if [[ "$eval_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_EVAL","task":"%s","seed":%s,"exit_code":%s,"log":"%s"}\n' \
    "$TASK" "$SEED" "$eval_rc" "$EVAL_LOG" > "$STATUS"
  exit "$eval_rc"
fi

"$PY" "$ROOT/finalize_h100_reproduction.py" \
  "$EVAL_DIR/TOFU_SUMMARY.json" \
  "$EVAL_DIR/TOFU_EVAL.json" \
  "$ROOT/$RETAIN_LOG" \
  "$TRAINER_STATE" \
  > "$STATUS.tmp"
final_rc=$?
if [[ "$final_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_FINALIZE","task":"%s","seed":%s,"exit_code":%s}\n' "$TASK" "$SEED" "$final_rc" > "$STATUS"
  exit "$final_rc"
fi
mv "$STATUS.tmp" "$STATUS"

if ! "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "DONE"' "$STATUS"; then
  printf '{"status":"REFUSED_CLEANUP_STATUS_NOT_DONE","task":"%s","seed":%s}\n' "$TASK" "$SEED" > "$CLEANUP_STATUS"
  exit 7
fi
resolved_train_dir="$(realpath -m -- "$TRAIN_DIR")"
expected_suffix="_seed${SEED}_20260828"
if [[ "$resolved_train_dir" != "$ROOT/saves/unlearn/tofu_"* || "$resolved_train_dir" != *"$expected_suffix" ]]; then
  printf '{"status":"REFUSED_UNSAFE_CLEANUP_TARGET","task":"%s","seed":%s,"path":"%s"}\n' \
    "$TASK" "$SEED" "$resolved_train_dir" > "$CLEANUP_STATUS"
  exit 8
fi
rm -rf -- "$resolved_train_dir"
if [[ -e "$resolved_train_dir" ]]; then
  printf '{"status":"FAILED_CHECKPOINT_CLEANUP","task":"%s","seed":%s,"path":"%s"}\n' \
    "$TASK" "$SEED" "$resolved_train_dir" > "$CLEANUP_STATUS"
  exit 9
fi
printf '{"status":"DONE","task":"%s","seed":%s,"checkpoint_removed":true,"trainer_state":"%s"}\n' \
  "$TASK" "$SEED" "$TRAINER_STATE" > "$CLEANUP_STATUS"
