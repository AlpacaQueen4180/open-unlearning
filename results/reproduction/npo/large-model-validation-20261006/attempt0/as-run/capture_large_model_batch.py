"""Capture model-specific TOFU tokens with the archived legacy data/collator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path('/data/npo-gb200-20261004')
repo = root / 'repo'
sys.path.insert(0, str(repo / 'src'))
os.environ['HF_HOME'] = str(root / 'hf')
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from data import get_collators, get_data

assets = json.loads(args.assets.read_text())
assert not args.output.exists(), 'Refuse overwrite of frozen batch'
with initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
    cfg = compose(config_name='unlearn', overrides=[
        'experiment=unlearn/tofu/default.yaml', 'trainer=NPO', 'model=' + args.model,
        'forget_split=forget01', 'retain_split=retain99', 'holdout_split=holdout01',
        'task_name=large_model_frozen_tofu_20261006',
        '+data.forget.TOFU_QA_forget.args.hf_args.revision=' + assets['dataset_revision'],
        '+data.retain.TOFU_QA_retain.args.hf_args.revision=' + assets['dataset_revision'],
    ])
tokenizer = AutoTokenizer.from_pretrained(assets['model_path'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
dataset = get_data(cfg.data, mode='unlearn', tokenizer=tokenizer, template_args=cfg.model.template_args)['train']
assert len(dataset) == 40
collator = get_collators(cfg.collator, tokenizer=tokenizer)
batch = collator([dataset[i] for i in range(40)])
assert all(x['input_ids'].shape[0] == 40 for x in batch.values())
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(batch, args.output)
metadata = {
    'model': args.model, 'assets': assets, 'indices': list(range(40)),
    'retain_sampling_seed': 0, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
    'effective_tokens': {k: [int(n) for n in (v['labels'][:, 1:] != -100).sum(1)] for k, v in batch.items()},
    'batch_file': str(args.output), 'full_window_rows': [0, 32], 'tail_window_rows': [32, 40],
    'collator_repo': str(repo), 'resolved_config': OmegaConf.to_container(cfg, resolve=True),
    'template_source_sha256': hashlib.sha256((repo / 'configs/model' / (args.model + '.yaml')).read_bytes()).hexdigest(),
}
args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, default=str))
print(json.dumps(metadata, default=str))
