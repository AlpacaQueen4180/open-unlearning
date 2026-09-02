#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ai/alpaca
ENV=/home/ai/miniforge3/envs/open-unlearning-repro
PY="$ENV/bin/python"
ACCELERATE="$ENV/bin/accelerate"
TASK=tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_micro4_acc4_e10_20260827
EVAL_TASK=${TASK}_eval
TRAIN_DIR="$ROOT/saves/unlearn/$TASK"
EVAL_DIR="$ROOT/saves/eval/$EVAL_TASK"
TRAIN_LOG="$ROOT/${TASK}.log"
EVAL_LOG="$ROOT/${EVAL_TASK}.log"
STATUS="$ROOT/${TASK}.status.json"
CLAIM=/home/ai/.open-unlearning-two-gpu-claim

cd "$ROOT" || exit 2
if [[ -e "$TRAIN_DIR" || -e "$EVAL_DIR" || -e "$STATUS" ]]; then
  printf '{"status":"REFUSED_EXISTING_TASK"}\n' > "$ROOT/${TASK}.launch-refused.json"
  exit 3
fi
if [[ ! -s saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json ]]; then
  printf '{"status":"REFUSED_MISSING_RETAIN_LOG"}\n' > "$STATUS"
  exit 4
fi

printf '{"status":"WAITING_FOR_TWO_IDLE_GPUS","required_idle_seconds":120}\n' > "$STATUS"
idle_checks=0
while (( idle_checks < 4 )); do
  process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
  if [[ "$process_count" == 0 ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  if (( idle_checks < 4 )); then
    sleep 30
  fi
done

if ! mkdir "$CLAIM" 2>/dev/null; then
  printf '{"status":"REFUSED_GPU_CLAIMED"}\n' > "$STATUS"
  exit 5
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
if [[ "$process_count" != 0 ]]; then
  printf '{"status":"REFUSED_RACE_GPU_BECAME_BUSY"}\n' > "$STATUS"
  exit 6
fi

master_port="$($PY -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
printf '{"status":"TRAINING","master_port":%s}\n' "$master_port" > "$STATUS"

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
  model=Llama-2-7b-chat-hf \
  forget_split=forget05 \
  retain_split=retain95 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  retain_logs_path=saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  > "$TRAIN_LOG" 2>&1
train_rc=$?

if [[ "$train_rc" -ne 0 ]]; then
  printf '{"status":"FAILED_TRAIN","exit_code":%s,"log":"%s"}\n' "$train_rc" "$TRAIN_LOG" > "$STATUS"
  exit "$train_rc"
fi

if ! find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q . || [[ ! -s "$TRAIN_DIR/trainer_state.json" ]]; then
  printf '{"status":"FAILED_SAVE","log":"%s"}\n' "$TRAIN_LOG" > "$STATUS"
  exit 7
fi

printf '{"status":"EVALUATING"}\n' > "$STATUS"
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
