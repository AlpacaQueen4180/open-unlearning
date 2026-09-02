#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/gb10/open-unlearning-retain-dup-control
PY=/home/gb10/miniconda3/envs/open-unlearning/bin/python
TRAIN_TASK=tofu_Llama-2-7b-chat-hf_forget05_NPO_rankdup_diag_micro8_acc4_e10_20260827
EVAL_TASK=${TRAIN_TASK}_eval_retry2
TRAIN_DIR="$ROOT/saves/unlearn/$TRAIN_TASK"
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${EVAL_TASK}.status.json"

cd "$ROOT" || exit 2
if test -e "$EVAL_DIR" || test -e "$STATUS"; then
  echo '{"status":"REFUSED_EXISTING_TASK"}' > "$ROOT/${EVAL_TASK}.launch-refused.json"
  exit 3
fi
if ! find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q .; then
  echo '{"status":"REFUSED_MISSING_MODEL"}' > "$STATUS"
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
