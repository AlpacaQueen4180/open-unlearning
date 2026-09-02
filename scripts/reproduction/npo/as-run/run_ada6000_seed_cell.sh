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

ROOT=/home/user/alpaca/open-unlearning
ENV="$ROOT/.venv-transformers445"
OVERLAY="$ROOT/.python-overlay-openunlearning451"
PY="$ENV/bin/python"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TRAIN_DIR="$ROOT/saves/unlearn/$TASK"
EVAL_TASK=${TASK}_eval
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
TRAIN_LOG="$ROOT/${TASK}.log"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${TASK}.status.json"
RETAIN_LOG="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

cd "$ROOT" || exit 2
if [[ ! "$SEED" =~ ^[0-4]$ ]]; then
  printf '{"status":"REFUSED_INVALID_SEED","task":"%s","seed":"%s"}\n' "$TASK" "$SEED" > "$STATUS"
  exit 3
fi
if [[ -e "$STATUS" || -e "$TRAIN_DIR" || -e "$EVAL_DIR" ]]; then
  printf '{"status":"REFUSED_EXISTING_CELL","task":"%s"}\n' "$TASK" > "$ROOT/${TASK}.launch-refused.json"
  exit 4
fi
if [[ ! -s "$RETAIN_LOG" ]]; then
  printf '{"status":"REFUSED_MISSING_RETAIN_LOG","task":"%s","path":"%s"}\n' "$TASK" "$RETAIN_LOG" > "$STATUS"
  exit 5
fi
if [[ ! -d "$OVERLAY/transformers" || ! -d "$OVERLAY/flash_attn" ]]; then
  printf '{"status":"REFUSED_MISSING_VERSION_OVERLAY","task":"%s","path":"%s"}\n' "$TASK" "$OVERLAY" > "$STATUS"
  exit 6
fi

printf '{"status":"TRAINING","task":"%s","model":"%s","forget_split":"%s","seed":%s,"topology":"1xAda6000","microbatch":8,"gradient_accumulation":4}\n' \
  "$TASK" "$MODEL" "$FORGET_SPLIT" "$SEED" > "$STATUS"

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$OVERLAY"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -u src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=NPO \
  task_name="$TASK" \
  model="$MODEL" \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
  model.model_args.attn_implementation=flash_attention_2 \
  retain_logs_path="$RETAIN_LOG" \
  trainer.args.per_device_train_batch_size=8 \
  trainer.args.gradient_accumulation_steps=4 \
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
  exit 7
fi

printf '{"status":"EVALUATING","task":"%s","model":"%s","forget_split":"%s","seed":%s,"eval_batch_size":8}\n' \
  "$TASK" "$MODEL" "$FORGET_SPLIT" "$SEED" > "$STATUS"
"$PY" -u src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default.yaml \
  model="$MODEL" \
  forget_split="$FORGET_SPLIT" \
  holdout_split="$HOLDOUT_SPLIT" \
  task_name="$EVAL_TASK" \
  model.model_args.pretrained_model_name_or_path="$TRAIN_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
  model.model_args.attn_implementation=flash_attention_2 \
  retain_logs_path="$RETAIN_LOG" \
  eval.tofu.batch_size=8 \
  seed="$SEED" \
  > "$EVAL_LOG" 2>&1
eval_rc=$?

if [[ "$eval_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_EVAL","task":"%s","seed":%s,"exit_code":%s,"log":"%s"}\n' \
    "$TASK" "$SEED" "$eval_rc" "$EVAL_LOG" > "$STATUS"
  exit "$eval_rc"
fi

"$PY" "$ROOT/finalize_ada6000_reproduction.py" \
  "$EVAL_DIR/TOFU_SUMMARY.json" \
  "$EVAL_DIR/TOFU_EVAL.json" \
  "$ROOT/$RETAIN_LOG" \
  "$TRAIN_DIR/trainer_state.json" \
  > "$STATUS.tmp"
final_rc=$?
if [[ "$final_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_FINALIZE","task":"%s","seed":%s,"exit_code":%s}\n' "$TASK" "$SEED" "$final_rc" > "$STATUS"
  exit "$final_rc"
fi
mv "$STATUS.tmp" "$STATUS"
