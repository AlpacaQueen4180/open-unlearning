#!/usr/bin/env bash
set -euo pipefail

repo=/home/ai/alpaca
conda_root=/home/ai/miniforge3
env_name=open-unlearning-repro

if [[ ! -x "$conda_root/bin/conda" ]]; then
  installer="$(mktemp /tmp/miniforge-open-unlearning.XXXXXX.sh)"
  trap 'rm -f "$installer"' EXIT
  wget -q --show-progress \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O "$installer"
  bash "$installer" -b -p "$conda_root"
fi

if ! "$conda_root/bin/conda" env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  "$conda_root/bin/conda" create -y -n "$env_name" python=3.11 pip
fi

cd "$repo"
export PIP_NO_CACHE_DIR=1
"$conda_root/envs/$env_name/bin/python" -m pip install --upgrade pip setuptools wheel
"$conda_root/envs/$env_name/bin/python" -m pip install -e .

"$conda_root/envs/$env_name/bin/python" - <<'PY'
import accelerate
import bitsandbytes
import datasets
import deepspeed
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_devices={torch.cuda.device_count()}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"deepspeed={deepspeed.__version__}")
print(f"bitsandbytes={bitsandbytes.__version__}")
print(f"datasets={datasets.__version__}")
PY

touch /home/ai/alpaca/.h100_environment_ready
