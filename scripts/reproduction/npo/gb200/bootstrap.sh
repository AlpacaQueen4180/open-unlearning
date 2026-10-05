#!/usr/bin/env bash
set -euo pipefail
ROOT=${NPO_ROOT:-/data/npo-gb200-20261004}
mkdir -p "$ROOT"
cd "$ROOT"
if [[ ! -d repo/.git ]]; then
  git clone --single-branch --branch repro/npo-h100-ada-5seed https://github.com/AlpacaQueen4180/open-unlearning.git repo
fi
git -C repo checkout --detach 820102411091abba2c5203207391bc1be0f3a863
VENV=${NPO_VENV:-$ROOT/venv}
python -m venv --system-site-packages "$VENV"
export DS_BUILD_OPS=0
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
"$VENV/bin/python" -m pip install \
  transformers==4.51.3 accelerate==0.34.2 deepspeed==0.15.4 \
  huggingface-hub==0.36.0 datasets==3.0.1 hydra-core==1.3.2 hydra-colorlog==1.2.0 \
  rouge-score==0.1.2 scipy==1.14.1 tensorboard==2.18.0 scikit-learn==1.5.2 \
  wandb==0.21.4 sentencepiece lm-eval==0.4.11 peft==0.15.2
"$VENV/bin/python" -m pip install --ignore-installed --no-deps bitsandbytes==0.50.2
"$VENV/bin/python" -m pip freeze > "$VENV/packages.txt"
"$VENV/bin/python" - <<'PY'
import json, torch, importlib.metadata as m
from pathlib import Path
import os
root = Path(os.environ.get('NPO_ROOT', '/data/npo-gb200-20261004'))
info = {n: m.version(n) for n in ('torch','transformers','accelerate','deepspeed','bitsandbytes','flash-attn','datasets')}
info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    flash_attn_func(q,q,q,causal=True).sum().backward()
    torch.cuda.synchronize()
    info['flash_attention_kernel'] = 'PASS'
except Exception as e:
    info['flash_attention_kernel'] = repr(e)
try:
    import bitsandbytes as bnb
    p = torch.nn.Parameter(torch.ones(8192, device='cuda', dtype=torch.bfloat16))
    opt = bnb.optim.PagedAdamW32bit([p], lr=1e-5)
    p.sum().backward(); opt.step(); torch.cuda.synchronize()
    info['paged_adamw_32bit_kernel'] = 'PASS'
except Exception as e:
    info['paged_adamw_32bit_kernel'] = repr(e)
(root/'preflight.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
PY
