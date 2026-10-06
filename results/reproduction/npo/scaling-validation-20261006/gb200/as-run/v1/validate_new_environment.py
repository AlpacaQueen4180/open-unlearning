"""Finite, guarded validation queue; no polling loop or modification of old venv."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/data/npo-gb200-20261004')
TASK=ROOT/'validation-20261006'
TASK.mkdir(exist_ok=True)
lock=(TASK/'validation.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (TASK/'status.json').exists():raise RuntimeError('Refuse duplicate queue; retain failed evidence and use a new tag on retry')
state={'phase':'STARTING','pid':os.getpid(),'stages':[],'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
    tmp=TASK/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(TASK/'status.json')
def run(label,cmd,cwd=None,env=None):
    if str(script) in cmd:
        cmd=cmd[:1]+['-m','torch.distributed.run','--nproc_per_node=1','--master_port=29661']+cmd[1:]
    state.update(phase='RUNNING',current=label);save()
    (TASK/(label+'.command.json')).write_text(json.dumps(cmd,indent=2))
    with (TASK/(label+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['stages'].append({'label':label,'returncode':result.returncode})
    save()
    if result.returncode:raise RuntimeError(label+' failed; see saved log')
save()
try:
    old=ROOT/'venv/bin/python'
    script=ROOT/'repo/scripts/reproduction/npo/gb200/gradient_audit.py'
    # FP32 exact reference, bf16 actual DS path, full/tail; GAS4 is H100-recipe
    # scaling control on GB200, NOT a two-H100 hardware measurement.
    cases=[(8,8,'forget','fp32',False),(8,8,'retain','fp32',False),(8,8,'combined','fp32',False),(8,2,'combined','fp32',False),(8,4,'combined','fp32',False),(4,4,'combined','fp32',False),(4,1,'combined','fp32',False),(8,8,'combined','bf16',False),(8,2,'combined','bf16',False),(8,8,'retain','fp32',True)]
    for gas,window,term,dtype,unequal in cases:
        label=f'old_g{gas}_w{window}_{term}_{dtype}'+('_variable_tokens' if unequal else '')
        cmd=[str(old),str(script),'--repo',str(ROOT/'repo'),'--output',str(TASK/(label+'.json')),'--gas',str(gas),'--window',str(window),'--term',term,'--dtype',dtype]
        if unequal:cmd+=['--unequal-tokens']
        run(label,cmd)
    upstream=TASK/'upstream'
    run('clone_upstream',['git','clone','--no-checkout','https://github.com/locuslab/open-unlearning.git',str(upstream)])
    commit='17cbbc87192e6934deb92875c359c91bbd837fb4'
    run('pin_upstream',['git','checkout','--detach',commit],cwd=upstream)
    envdir=TASK/'venv'
    run('create_venv',[sys.executable,'-m','venv','--system-site-packages',str(envdir)])
    py=envdir/'bin/python'
    import torch
    (TASK/'constraints.txt').write_text('torch=='+torch.__version__+'\n')
    env=dict(os.environ,DS_BUILD_OPS='0',PIP_CONFIG_FILE='/dev/null',PIP_INDEX_URL='https://pypi.org/simple',PIP_EXTRA_INDEX_URL='',HF_HOME=str(ROOT/'hf'),TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    run('install_new',[str(py),'-m','pip','install','-c',str(TASK/'constraints.txt'),'transformers==5.5.4','accelerate==1.13.0','deepspeed==0.15.4','bitsandbytes==0.50.2','peft>=0.18','datasets==3.0.1','hydra-core==1.3.2','hydra-colorlog==1.2.0','rouge-score==0.1.2','scipy==1.14.1','tensorboard==2.18.0','scikit-learn==1.5.2','wandb==0.21.4','sentencepiece','lm-eval==0.4.11'],env=env)
    run('freeze_new',[str(py),'-m','pip','freeze'],env=env)
    run('imports_new',[str(py),'-c','import torch,transformers,accelerate,deepspeed,peft,flash_attn,bitsandbytes; print(torch.__version__,transformers.__version__,accelerate.__version__,deepspeed.__version__,peft.__version__); assert transformers.__version__=="5.5.4" and accelerate.__version__=="1.13.0"'],env=env)
    for gas,window,term,dtype,unequal in cases:
        label=f'new_g{gas}_w{window}_{term}_{dtype}'+('_variable_tokens' if unequal else '')
        cmd=[str(py),str(script),'--repo',str(upstream),'--output',str(TASK/(label+'.json')),'--gas',str(gas),'--window',str(window),'--term',term,'--dtype',dtype]
        if unequal:cmd+=['--unequal-tokens']
        run(label,cmd,env=env)
    # Real-model gradient check uses exactly the cached 1B full checkpoint.
    assets=json.loads((ROOT/'assets_Llama-3.2-1B-Instruct_forget01.json').read_text())
    modelpath=ROOT/'hf/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots'/assets['model_revision']
    run('freeze_real_tofu',[str(old),str(ROOT/'repo/scripts/reproduction/npo/gb200/capture_tofu_gradient_batch.py')],env=env)
    for kind,python,repo in [('old',old,ROOT/'repo'),('new',py,upstream)]:
        for window in [8,2]:
            label=f'{kind}_real1b_w{window}_combined_bf16'
            run(label,[str(python),str(script),'--repo',str(repo),'--output',str(TASK/(label+'.json')),'--gas','8','--window',str(window),'--term','combined','--dtype','bf16','--model-path',str(modelpath)],env=env)
        for window,offset in [(8,0),(2,32)]:
            for term in ['forget','retain','combined']:
                label=f'{kind}_real1b_TOFU_w{window}_{term}_bf16'
                run(label,[str(python),str(script),'--repo',str(repo),'--output',str(TASK/(label+'.json')),'--gas','8','--window',str(window),'--term',term,'--dtype','bf16','--model-path',str(modelpath),'--batch-file',str(TASK/'tofu-frozen-40.pt'),'--offset',str(offset)],env=env)
    # Only after gradient checks: native new-stack full runs, no legacy patch.
    helpers=upstream/'scripts/reproduction/npo/gb200';helpers.mkdir(parents=True,exist_ok=True)
    for name in ['run_experiment.py','run_probe.py']:
        shutil.copy2(ROOT/'repo/scripts/reproduction/npo/gb200'/name,helpers/name)
    probe=(helpers/'run_probe.py').read_text()
    probe=probe.replace('        final.update(engine_global_steps=trainer.model_wrapped.global_steps,','        final.update(engine_global_steps=trainer.model_wrapped.global_steps,')
    # New wrapper microstep counters differ; our per-training_step counter is authoritative.
    probe += '\nif observed_trainer is not None:\n    expected={40:20,200:70,400:130}[len(observed_trainer.train_dataset)]\n    assert final["trainer_global_step"] == final["engine_global_steps"] == expected, final\n    assert final["epoch"] == 10, final\n    assert final["microsteps"] == len(observed_trainer.train_dataset)//4*10, final\n'
    (helpers/'run_probe.py').write_text(probe)
    finalize=upstream/'scripts/reproduction/npo/as-run';finalize.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'repo/scripts/reproduction/npo/as-run/finalize_h100_reproduction.py',finalize)
    newroot=TASK/'experiments';newroot.mkdir(exist_ok=True)
    (newroot/'repo').symlink_to(upstream,target_is_directory=True)
    (newroot/'hf').symlink_to(ROOT/'hf',target_is_directory=True)
    shutil.copy2(ROOT/'assets.json',newroot/'assets.json')
    for source in ROOT.glob('assets_*.json'):shutil.copy2(source,newroot/source.name)
    (newroot/'runs').mkdir()
    newenv=dict(env,NPO_ROOT=str(newroot))
    for split in ['forget01','forget05','forget10']:
        run('native_1b_'+split,[str(py),str(helpers/'run_experiment.py'),'--world-size','1','--stages','3','--attention','flash_attention_2','--seed','0','--model','Llama-3.2-1B-Instruct','--forget',split,'--tag','native_554_1130_20261006'],env=newenv)
    state.update(phase='DONE',finished=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception as e:
    state.update(phase='FAILED',error=repr(e));save();raise
