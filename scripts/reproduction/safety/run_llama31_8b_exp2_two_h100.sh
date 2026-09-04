#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/ai/alpaca
ENV=/home/ai/miniforge3/envs/open-unlearning-repro
PY="$ENV/bin/python"
ACCELERATE="$ENV/bin/accelerate"
QUEUE_ID=tofu_safety_exp2_llama31_8b_2xH100_seed0_20260904
ARTIFACT_DIR="$ROOT/$QUEUE_ID"
QUEUE_STATUS="$ARTIFACT_DIR/queue.status.json"
LEDGER="$ARTIFACT_DIR/ledger.tsv"
CLAIM=/home/ai/.open-unlearning-safety-exp2-two-h100-claim
MODEL=Llama-3.1-8B-Instruct
MODEL_PATH=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
RETAIN_LOG=saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json

conditions=(
  'NPO_no_retain|0.0'
  'NPO_retain95|1.0'
)

write_queue_status() {
  local state="$1" cell="$2" completed="$3" task="$4"
  printf '{"status":"%s","queue":"%s","cell":%s,"total":2,"completed":%s,"task":"%s","topology":"2xH100_zero3","updated_at":"%s"}\n' \
    "$state" "$QUEUE_ID" "$cell" "$completed" "$task" "$(date -Is)" > "$QUEUE_STATUS.tmp"
  mv "$QUEUE_STATUS.tmp" "$QUEUE_STATUS"
}

if [[ -e "$ARTIFACT_DIR" ]]; then
  printf '{"status":"REFUSED_EXISTING_QUEUE","queue":"%s","path":"%s"}\n' \
    "$QUEUE_ID" "$ARTIFACT_DIR" > "$ROOT/${QUEUE_ID}.launch-refused.json"
  exit 2
fi
mkdir -p "$ARTIFACT_DIR/logs" "$ARTIFACT_DIR/status"

if [[ ! -s "$RETAIN_LOG" ]]; then
  write_queue_status REFUSED_MISSING_RETAIN_LOG 0 0 ''
  exit 3
fi
if ! mkdir "$CLAIM" 2>/dev/null; then
  write_queue_status REFUSED_GPU_CLAIMED 0 0 ''
  exit 4
fi
trap 'rmdir "$CLAIM" 2>/dev/null || true' EXIT

cd "$ROOT" || exit 5
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

printf 'cell\tcondition\talpha\tseed\tstatus\ttask\tcheckpoint\n' > "$LEDGER"
write_queue_status WAITING_FOR_TWO_IDLE_GPUS 0 0 ''
idle_checks=0
while (( idle_checks < 4 )); do
  process_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF { count++ } END { print count + 0 }')"
  if [[ "$process_count" == 0 ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  if (( idle_checks < 4 )); then
    sleep 15
  fi
done

completed=0
cell=0
for condition_spec in "${conditions[@]}"; do
  cell=$((cell + 1))
  IFS='|' read -r condition alpha <<< "$condition_spec"
  task="tofu_safety_exp2_${MODEL}_forget05_${condition}_2xH100_seed0_20260904"
  train_dir="$ROOT/saves/unlearn/$task"
  train_log="$ARTIFACT_DIR/logs/${task}.train.log"
  status_path="$ARTIFACT_DIR/status/${task}.status.json"

  if [[ -e "$train_dir" || -e "$status_path" ]]; then
    write_queue_status REFUSED_EXISTING_CELL "$cell" "$completed" "$task"
    exit 6
  fi
  available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 { print $4 }')"
  if (( available_kb < 20971520 )); then
    write_queue_status FAILED_LOW_DISK "$cell" "$completed" "$task"
    exit 7
  fi

  master_port="$($PY -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
  write_queue_status TRAINING "$cell" "$completed" "$task"
  "$ACCELERATE" launch \
    --config_file configs/accelerate/default_config.yaml \
    --main_process_port "$master_port" \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=NPO \
    task_name="$task" \
    model="$MODEL" \
    forget_split=forget05 \
    retain_split=retain95 \
    model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
    model.model_args.attn_implementation=flash_attention_2 \
    retain_logs_path="$RETAIN_LOG" \
    trainer.method_args.alpha="$alpha" \
    trainer.method_args.gamma=1.0 \
    trainer.method_args.beta=0.1 \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=4 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.seed=0 \
    > "$train_log" 2>&1
  train_rc=$?

  if [[ "$train_rc" -ne 0 ]]; then
    printf '{"status":"FAILED_TRAIN","condition":"%s","alpha":%s,"task":"%s","exit_code":%s,"log":"%s"}\n' \
      "$condition" "$alpha" "$task" "$train_rc" "$train_log" > "$status_path"
    write_queue_status FAILED_TRAIN "$cell" "$completed" "$task"
    exit "$train_rc"
  fi
  if ! find "$train_dir" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q . || [[ ! -s "$train_dir/trainer_state.json" ]]; then
    printf '{"status":"FAILED_SAVE","condition":"%s","alpha":%s,"task":"%s","log":"%s"}\n' \
      "$condition" "$alpha" "$task" "$train_log" > "$status_path"
    write_queue_status FAILED_SAVE "$cell" "$completed" "$task"
    exit 8
  fi

  shard_count="$(find "$train_dir" -maxdepth 1 -type f -name '*.safetensors' | wc -l)"
  checkpoint_bytes="$(du -sb "$train_dir" | cut -f1)"
  printf '{"status":"DONE","condition":"%s","alpha":%s,"task":"%s","seed":0,"model":"%s","forget_split":"forget05","retain_split":"retain95","topology":"2xH100_zero3","microbatch":4,"gradient_accumulation":4,"global_batch":32,"checkpoint":"%s","checkpoint_bytes":%s,"safetensor_shards":%s,"trainer_state":"%s","finished_at":"%s"}\n' \
    "$condition" "$alpha" "$task" "$MODEL" "$train_dir" "$checkpoint_bytes" "$shard_count" "$train_dir/trainer_state.json" "$(date -Is)" > "$status_path"
  completed=$((completed + 1))
  printf '%s\t%s\t%s\t0\tDONE\t%s\t%s\n' "$cell" "$condition" "$alpha" "$task" "$train_dir" >> "$LEDGER"
  write_queue_status CELL_DONE "$cell" "$completed" "$task"
done

write_queue_status DONE 2 2 ''
