#!/usr/bin/env bash
set -euo pipefail
export NPO_ROOT=/data/npo-gb200-20261004
export NPO_VENV=$NPO_ROOT/venv-2gpu
exec > >(tee -a "$NPO_ROOT/two-gpu-workload.log") 2>&1
nvidia-smi
bash "$NPO_ROOT/repo/scripts/reproduction/npo/gb200/bootstrap.sh"
"$NPO_VENV/bin/python" "$NPO_ROOT/repo/scripts/reproduction/npo/gb200/run_experiment.py" \
  --world-size 2 --stages 3 --attention flash_attention_2 --tag two-gpu
