#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ai/alpaca}"
ENV="${ENV:-/home/ai/miniforge3/envs/open-unlearning-repro}"
PYTHON_BIN="$ENV/bin/python"
ACCELERATE="$ENV/bin/accelerate"
QUEUE_ID=llama31_8b_retain95_and_hexphi_seed0_20260905
ARTIFACT_DIR="$ROOT/safety_artifacts/$QUEUE_ID"
STATUS="$ARTIFACT_DIR/status.json"
LEGACY_DATA="$ROOT/safety_artifacts/datasets/HEx-PHI-legacy-300.json"
SECRET_ENV=/home/ai/.config/open-unlearning/safety.env
RETAIN_TASK=tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_retain95_2xH100_seed0_20260904
RETAIN_DIR="$ROOT/saves/unlearn/$RETAIN_TASK"
NO_RETAIN_DIR="$ROOT/saves/unlearn/tofu_safety_exp2_Llama-3.1-8B-Instruct_forget05_NPO_no_retain_2xH100_seed0_20260904"
FULL_MODEL=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
RETAIN_LOG=saves/eval/tofu_Llama-3.1-8B-Instruct_retain95/TOFU_EVAL.json

models=(
  "original|meta-llama/Llama-3.1-8B-Instruct"
  "tofu_full|open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
  "npo_no_retain|$NO_RETAIN_DIR"
  "npo_retain95|$RETAIN_DIR"
  "retain95_oracle|open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95"
)
protocols=("legacy-300" "formal-300")

write_status() {
  local state="$1" stage="$2" detail="${3:-}"
  printf '{"status":"%s","stage":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$state" "$stage" "$detail" "$(date -Is)" > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}

on_error() {
  local rc=$?
  write_status FAILED unexpected_exit "exit_code=$rc"
  exit "$rc"
}

if [[ -e "$ARTIFACT_DIR" || -e "$RETAIN_DIR" ]]; then
  printf 'Refusing existing queue or retain checkpoint\n' >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR/logs" "$ROOT/safety_artifacts/raw" \
  "$ROOT/safety_artifacts/judged" "$ROOT/safety_artifacts/summaries"
trap on_error ERR
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

if [[ "$(sha256sum "$LEGACY_DATA" | cut -d' ' -f1)" != "f72785518afa1dde3c1324987e123ef307a6e2ee2b69a8646c738c06e051db2e" ]]; then
  write_status FAILED dataset_checksum
  exit 3
fi
if (( $(df -Pk "$ROOT" | awk 'NR == 2 {print $4}') < 104857600 )); then
  write_status FAILED disk_preflight
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  write_status REFUSED gpu_busy
  exit 5
fi

write_status RUNNING training_retain95 "$RETAIN_TASK"
export CUDA_VISIBLE_DEVICES=0,1
master_port="$($PYTHON_BIN -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")"
"$ACCELERATE" launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port "$master_port" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml trainer=NPO task_name="$RETAIN_TASK" \
  model=Llama-3.1-8B-Instruct forget_split=forget05 retain_split=retain95 \
  model.model_args.pretrained_model_name_or_path="$FULL_MODEL" \
  model.tokenizer_args.pretrained_model_name_or_path="$FULL_MODEL" \
  model.model_args.attn_implementation=flash_attention_2 \
  retain_logs_path="$RETAIN_LOG" trainer.method_args.alpha=1.0 \
  trainer.method_args.gamma=1.0 trainer.method_args.beta=0.1 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true trainer.args.seed=0 \
  > "$ARTIFACT_DIR/logs/retain95.train.log" 2>&1

if ! find "$RETAIN_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +100M -print -quit | grep -q .; then
  write_status FAILED validate_retain95
  exit 6
fi

for model_spec in "${models[@]}"; do
  IFS='|' read -r model_name model_path <<< "$model_spec"
  for protocol in "${protocols[@]}"; do
    output="$ROOT/safety_artifacts/raw/${model_name}_${protocol}_seed0.jsonl"
    write_status RUNNING generation "${model_name}:${protocol}"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m evals.safety generate \
      --model "$model_path" --input "$LEGACY_DATA" --output "$output" \
      --run-id "llama31_8b_${model_name}_${protocol}_seed0" \
      --protocol "$protocol" --checkpoint-id "$model_name" \
      --seed 0 --max-new-tokens 512 \
      > "$ARTIFACT_DIR/logs/${model_name}_${protocol}.generate.log" 2>&1
    if [[ "$(wc -l < "$output")" -ne 300 ]]; then
      write_status FAILED generation_count "${model_name}:${protocol}"
      exit 7
    fi
  done
done

if [[ ! -f "$SECRET_ENV" ]]; then
  write_status WAITING_FOR_API_KEY generation_complete
  exit 0
fi
if [[ "$(stat -c '%a' "$SECRET_ENV")" != 600 ]]; then
  write_status FAILED secret_permissions
  exit 8
fi

for model_spec in "${models[@]}"; do
  IFS='|' read -r model_name _ <<< "$model_spec"
  for protocol in "${protocols[@]}"; do
    raw="$ROOT/safety_artifacts/raw/${model_name}_${protocol}_seed0.jsonl"
    judged="$ROOT/safety_artifacts/judged/${model_name}_${protocol}_seed0.jsonl"
    summary="$ROOT/safety_artifacts/summaries/${model_name}_${protocol}_seed0.json"
    write_status RUNNING judge "${model_name}:${protocol}"
    "$PYTHON_BIN" -m evals.safety judge --input "$raw" --output "$judged" \
      --model gpt-5.6-terra --reasoning-effort medium \
      --concurrency 8 --retries 3 --timeout 60 --env-file "$SECRET_ENV" \
      > "$ARTIFACT_DIR/logs/${model_name}_${protocol}.judge.log" 2>&1
    "$PYTHON_BIN" -m evals.safety summarize --input "$judged" --output "$summary" \
      > "$ARTIFACT_DIR/logs/${model_name}_${protocol}.summary.log" 2>&1
  done
done

write_status DONE complete
trap - ERR
