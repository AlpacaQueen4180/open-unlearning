#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/gb10/open-unlearning-retain-dup-control
PY=/home/gb10/miniconda3/envs/open-unlearning/bin/python
TASK=tofu_Llama-2-7b-chat-hf_forget05_NPO_rankdup_diag_micro8_acc4_e10_20260827
EVAL_TASK=${TASK}_eval
TRAIN_DIR="$ROOT/saves/unlearn/$TASK"
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
TRAIN_LOG="$ROOT/${TASK}.log"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${TASK}.status.json"

cd "$ROOT" || exit 2
if test -e "$TRAIN_DIR" || test -e "$EVAL_DIR" || test -e "$STATUS"; then
  echo '{"status":"REFUSED_EXISTING_TASK"}' > "$ROOT/${TASK}.launch-refused.json"
  exit 3
fi

mkdir -p saves/eval
if ! test -e saves/eval/tofu_Llama-2-7b-chat-hf_retain95; then
  ln -s /home/gb10/open-unlearning/saves/eval/tofu_Llama-2-7b-chat-hf_retain95 \
    saves/eval/tofu_Llama-2-7b-chat-hf_retain95
fi

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 "$PY" -u src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=NPO \
  collator=DuplicateRetainDataCollator \
  task_name="$TASK" \
  model=Llama-2-7b-chat-hf \
  forget_split=forget05 \
  retain_split=retain95 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  retain_logs_path=saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=8 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.gradient_checkpointing=true \
  trainer.args.eval_on_start=false \
  trainer.args.eval_strategy=no \
  trainer.args.do_eval=false \
  model.model_args.attn_implementation=eager \
  > "$TRAIN_LOG" 2>&1
train_rc=$?

if test "$train_rc" -ne 0; then
  printf '{"status":"FAILED_TRAIN","exit_code":%s,"log":"%s"}\n' "$train_rc" "$TRAIN_LOG" > "$STATUS"
  exit "$train_rc"
fi

if ! find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q . || ! test -s "$TRAIN_DIR/trainer_state.json"; then
  printf '{"status":"FAILED_SAVE","log":"%s"}\n' "$TRAIN_LOG" > "$STATUS"
  exit 4
fi

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 "$PY" -u src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default.yaml \
  model=Llama-2-7b-chat-hf \
  forget_split=forget05 \
  holdout_split=holdout05 \
  task_name="$EVAL_TASK" \
  model.model_args.pretrained_model_name_or_path="$TRAIN_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  retain_logs_path=saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json \
  model.model_args.attn_implementation=eager \
  > "$EVAL_LOG" 2>&1
eval_rc=$?

if test "$eval_rc" -ne 0; then
  printf '{"status":"FAILED_EVAL","exit_code":%s,"log":"%s"}\n' "$eval_rc" "$EVAL_LOG" > "$STATUS"
  exit "$eval_rc"
fi

"$PY" "$ROOT/finalize_rankdup_diagnostic.py" \
  "$EVAL_DIR/TOFU_SUMMARY.json" \
  "$EVAL_DIR/TOFU_EVAL.json" \
  "$ROOT/saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json" \
  "$TRAIN_DIR/trainer_state.json" \
  > "$STATUS.tmp"
final_rc=$?
if test "$final_rc" -ne 0; then
  printf '{"status":"FAILED_FINALIZE","exit_code":%s}\n' "$final_rc" > "$STATUS"
  exit "$final_rc"
fi
mv "$STATUS.tmp" "$STATUS"
