"""Finite H100 legacy validation; refuses busy devices and duplicate queue."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
root=Path('/home/ai/alpaca/results/reproduction/npo/scaling-validation-20261006')
root.mkdir(exist_ok=True)
lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'status.json').exists():raise RuntimeError('Refuse duplicate launch; preserve prior evidence')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise RuntimeError('GPUs busy; do not stop other tasks')
state={'phase':'RUNNING','pid':os.getpid(),'cells':[]}
def save():
    temp=root/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(root/'status.json')
save()
try:
    cases=[(4,'forget','fp32',None),(4,'retain','fp32',None),(4,'combined','fp32',None),(1,'combined','fp32',None),(4,'combined','bf16',None),(1,'combined','bf16',None)]
    model='/home/ai/.cache/huggingface/hub/models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/snapshots/cc5b31c69127d5da881608e640f2b453c446435b'
    if Path(model).exists():cases += [(4,'combined','bf16',model),(1,'combined','bf16',model)]
    for index,(window,term,dtype,modelpath) in enumerate(cases):
        label=f'{index}_w{window}_{term}_{dtype}'+('_real7b' if modelpath else '')
        state['current']=label;save()
        cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29671',str(root/'gradient_audit.py'),'--repo','/home/ai/alpaca','--output',str(root/(label+'.json')),'--gas','4','--window',str(window),'--term',term,'--dtype',dtype]
        if modelpath:cmd+=['--model-path',modelpath]
        (root/(label+'.command.json')).write_text(json.dumps(cmd))
        with (root/(label+'.log')).open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
        state['cells'].append({'label':label,'returncode':code});save()
        if code:raise RuntimeError(label+' failed')
    state['phase']='DONE';save()
except Exception as e:
    state.update(phase='FAILED',error=repr(e));save();raise
