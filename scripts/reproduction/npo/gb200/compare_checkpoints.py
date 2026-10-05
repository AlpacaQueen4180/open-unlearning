"""Compare selected tensors with the common starting checkpoint on CPU.

This is a descriptive audit, not an isolated causal test of optimizer precision.
"""
import json
from pathlib import Path
import torch
from safetensors import safe_open

root = Path('/data/npo-gb200-20261004')
assets = json.loads((root/'assets.json').read_text())
base = root/'hf/hub/models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/snapshots'/assets['model_revision']
names = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.15.self_attn.q_proj.weight',
         'model.layers.31.self_attn.q_proj.weight', 'model.layers.0.input_layernorm.weight']


def tensor(directory, key):
    index_path = directory/'model.safetensors.index.json'
    shard = json.loads(index_path.read_text())['weight_map'][key] if index_path.exists() else 'model.safetensors'
    with safe_open(directory/shard, framework='pt', device='cpu') as handle:
        return handle.get_tensor(key).float()


results = []
for run in sorted((root/'runs').glob('*_main')):
    state = json.loads((run/'status.json').read_text())
    if state['phase'] != 'DONE':
        continue
    entry = {'run': run.name, 'tensors': {}}
    for name in names:
        start = tensor(base, name)
        end = tensor(run/'checkpoint', name)
        delta = end - start
        entry['tensors'][name] = {'elements': start.numel(), 'changed_fraction': (delta != 0).float().mean().item(),
                                 'delta_l2': delta.norm().item(), 'relative_l2': (delta.norm()/start.norm()).item(),
                                 'max_abs_change': delta.abs().max().item()}
    results.append(entry)
(root/'checkpoint_comparison.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
