import subprocess,os,json
code=r'''import json,hashlib,traceback,os
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
site=Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers')
assert sha((site/'training_args.py').read_bytes())=='67e63d9b8a68a1547c0f3d17eac51034592da669d01cf03b4471b0b77f22a756'
import torch
before=torch.cuda.is_initialized();assert not before and torch.cuda.device_count()==1
events=[];original=torch.cuda._lazy_init
def observe(*args,**kwargs):
 stack=traceback.extract_stack()[:-1]
 events.append([dict(filename=x.filename,line=x.lineno,name=x.name,code=x.line) for x in stack])
 return original(*args,**kwargs)
torch.cuda._lazy_init=observe
try:
 from transformers import TrainingArguments
finally:torch.cuda._lazy_init=original
sources={}
for stack in events:
 for x in stack:
  p=Path(x['filename'])
  if p.is_file():
   raw=p.read_bytes();assert len(raw)<2000000;sources[str(p)]=dict(size=len(raw),sha256=sha(raw))
after=torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_VISIBLE_GPU_TRAINING_ARGUMENTS_IMPORT_TRACE',cuda_initialized_before=before,cuda_initialized_after=after,events=events,source_sha256=sources,training_arguments_instantiated=False,weights_read=False,generation_calls=0,npo_training_executed=False,peak_allocated_bytes=torch.cuda.max_memory_allocated() if after else None,observer_restored=torch.cuda._lazy_init is original)))
'''
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HOME':'/data/smart-mfg/jimmy-lin/hf_cache/hub','HF_HUB_CACHE':'/data/smart-mfg/jimmy-lin/hf_cache/hub','WANDB_MODE':'disabled'}
p=subprocess.run(['/data/npo-gb200-20261004/validation-20261006/venv/bin/python','-c',code],env=env,capture_output=True,text=True)
print(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,capture='native subprocess text decoded by Python; not original stream bytes')))
