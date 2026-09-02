#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ai/alpaca
ENV=/home/ai/miniforge3/envs/open-unlearning-repro
PY="$ENV/bin/python"
TRAIN_TASK=tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827
EVAL_TASK=${TRAIN_TASK}_eval_retry2
TRAIN_DIR="$ROOT/saves/unlearn/$TRAIN_TASK"
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${EVAL_TASK}.status.json"
CLAIM=/home/ai/.open-unlearning-two-gpu-claim

cd "$ROOT" || exit 2
if [[ -e "$EVAL_DIR" || -e "$STATUS" ]]; then
  printf '{"status":"REFUSED_EXISTING_TASK"}\n' > "$ROOT/${EVAL_TASK}.launch-refused.json"
  exit 3
fi
if ! find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q . || [[ ! -s "$TRAIN_DIR/trainer_state.json" ]]; then
  printf '{"status":"REFUSED_MISSING_MODEL"}\n' > "$STATUS"
  exit 4
fi
if ! mkdir "$CLAIM" 2>/dev/null; then
  printf '{"status":"REFUSED_GPU_CLAIMED"}\n' > "$STATUS"
  exit 5
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
if [[ "$process_count" != 0 ]]; then
  printf '{"status":"REFUSED_GPUS_BUSY"}\n' > "$STATUS"
  exit 6
fi

printf '{"status":"EVALUATING"}\n' > "$STATUS"
export HYDRA_FULL_ERROR=1
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "$PY" -u src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default.yaml \
  model=Llama-2-7b-chat-hf \
  forget_split=forget05 \
  holdout_split=holdout05 \
  task_name="$EVAL_TASK" \
  model.model_args.pretrained_model_name_or_path="$TRAIN_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  retain_logs_path=saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json \
  > "$EVAL_LOG" 2>&1
eval_rc=$?

if [[ "$eval_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_EVAL","exit_code":%s,"log":"%s"}\n' "$eval_rc" "$EVAL_LOG" > "$STATUS"
  exit "$eval_rc"
fi

"$PY" "$ROOT/finalize_h100_reproduction.py" \
  "$EVAL_DIR/TOFU_SUMMARY.json" \
  "$EVAL_DIR/TOFU_EVAL.json" \
  "$ROOT/saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json" \
  "$TRAIN_DIR/trainer_state.json" \
  > "$STATUS.tmp"
final_rc=$?
if [[ "$final_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_FINALIZE","exit_code":%s}\n' "$final_rc" > "$STATUS"
  exit "$final_rc"
fi
mv "$STATUS.tmp" "$STATUS"
