"""Freeze real TOFU examples and repo masks/collator once for paired gradients."""
import hashlib,json,os
from pathlib import Path
import sys
ROOT=Path('/data/npo-gb200-20261004')
repo=ROOT/'repo';sys.path.insert(0,str(repo/'src'))
import torch
from hydra import initialize_config_dir,compose
from transformers import AutoTokenizer
from data import get_data,get_collators
assets=json.loads((ROOT/'assets_Llama-3.2-1B-Instruct_forget01.json').read_text())
model=ROOT/'hf/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots'/assets['model_revision']
os.environ['HF_HOME']=str(ROOT/'hf')
with initialize_config_dir(version_base=None,config_dir=str(repo/'configs')):
    cfg=compose(config_name='unlearn',overrides=['experiment=unlearn/tofu/default.yaml','trainer=NPO','model=Llama-3.2-1B-Instruct','forget_split=forget01','retain_split=retain99','holdout_split=holdout01','task_name=gradient_frozen_tofu_20261006',f'+data.forget.TOFU_QA_forget.args.hf_args.revision={assets["dataset_revision"]}',f'+data.retain.TOFU_QA_retain.args.hf_args.revision={assets["dataset_revision"]}'])
tokenizer=AutoTokenizer.from_pretrained(model)
if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
torch.manual_seed(0)
dataset=get_data(cfg.data,mode='unlearn',tokenizer=tokenizer,template_args=cfg.model.template_args)['train']
collator=get_collators(cfg.collator,tokenizer=tokenizer)
batch=collator([dataset[i] for i in range(40)])
path=ROOT/'validation-20261006/tofu-frozen-40.pt'
assert not path.exists(),'Refuse overwrite of frozen data'
torch.save(batch,path)
metadata={'source_repo':'820102411091abba2c5203207391bc1be0f3a863','assets':assets,'indices':list(range(40)),'retain_sampling_seed':0,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'effective_tokens':{k:[int(n) for n in (v['labels'][:,1:]!=-100).sum(1)] for k,v in batch.items()},'batch_file':str(path),'full_window_rows':[0,32],'tail_window_rows':[32,40]}
path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata))
