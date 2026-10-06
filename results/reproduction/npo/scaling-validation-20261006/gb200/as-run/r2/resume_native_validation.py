"""Recover evaluation of the existing native f01 checkpoint, then two missing splits."""
import datetime,fcntl,hashlib,json,math,os,struct,subprocess,sys
from pathlib import Path

ROOT=Path('/data/npo-gb200-20261004')
BASE=ROOT/'validation-20261006'
PRIOR=ROOT/'validation-20261006-r1'
TASK=ROOT/'validation-20261006-r2'
TASK.mkdir(exist_ok=True)
lock=(TASK/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (TASK/'status.json').exists(),'Refuse duplicate queue'
assert json.loads((PRIOR/'status.json').read_text())['phase']=='FAILED'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
repo=BASE/'upstream';py=BASE/'venv/bin/python';experiments=PRIOR/'experiments'
env=dict(os.environ,NPO_ROOT=str(experiments),HF_HOME=str(ROOT/'hf'),TOKENIZERS_PARALLELISM='false',HYDRA_FULL_ERROR='1',WANDB_MODE='disabled')
state={'phase':'STARTING','pid':os.getpid(),'stages':[],'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
    tmp=TASK/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(TASK/'status.json')
def run(label,command):
    state.update(phase='RUNNING',current=label);save()
    (TASK/(label+'.command.json')).write_text(json.dumps(command,indent=2))
    with (TASK/(label+'.log')).open('w') as log:rc=subprocess.call(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['stages'].append({'label':label,'returncode':rc});save()
    if rc:raise RuntimeError(label+' failed; preserve evidence')
save()
try:
    cell=experiments/'runs/gb200_1gpu_zero3_flash_attention_2_s0_native_554_1130_20261006_r1'
    checkpoint=cell/'checkpoint'
    final=json.loads((checkpoint/'runtime_final_rank0.json').read_text())
    trainer=json.loads((checkpoint/'trainer_state.json').read_text())
    assert final['trainer_global_step']==final['engine_global_steps']==trainer['global_step']==20
    assert final['microsteps']==100 and final['epoch']==trainer['epoch']==10
    weights=checkpoint/'model.safetensors'
    with weights.open('rb') as stream:
        header_length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(header_length))
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    assert max(v['data_offsets'][1] for v in tensors.values())+8+header_length==weights.stat().st_size
    assert 'model.embed_tokens.weight' in tensors and 'model.norm.weight' in tensors
    config=json.loads((checkpoint/'config.json').read_text())
    for layer in range(config['num_hidden_layers']):
        for suffix in ['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','input_layernorm.weight','post_attention_layernorm.weight']:
            assert f'model.layers.{layer}.{suffix}' in tensors
    digest=hashlib.sha256()
    with weights.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
    proof={'checkpoint':str(checkpoint),'weights_bytes':weights.stat().st_size,'tensor_count':len(tensors),'weights_sha256':digest.hexdigest(),'counts':final,'header_and_required_layers_complete':True}
    (TASK/'checkpoint-forget01-verification.json').write_text(json.dumps(proof,indent=2))
    previous=json.loads((cell/'status.json').read_text())
    if not (cell/'status.failed-evaluating-r1.json').exists():
        (cell/'status.failed-evaluating-r1.json').write_text(json.dumps(previous,indent=2))
    command=json.loads((cell/'evaluating_command.json').read_text())
    command=[f'paths.output_dir={cell/"eval-recovered-r2"}' if x.startswith('paths.output_dir=') else x for x in command]
    run('recover_native_1b_forget01_evaluation',command)
    summary=cell/'eval-recovered-r2/TOFU_SUMMARY.json';raw=cell/'eval-recovered-r2/TOFU_EVAL.json'
    retain_arg=next(x for x in command if x.startswith('retain_logs_path='));retain=retain_arg.split('=',1)[1]
    run('audit_native_1b_forget01',[str(py),'scripts/reproduction/npo/as-run/finalize_h100_reproduction.py',str(summary),str(raw),retain,str(checkpoint/'trainer_state.json')])
    previous.update(phase='DONE',summary=json.loads(summary.read_text()),eval_dir=str(summary.parent),evaluation_recovery='bf16 NumPy serialization compatibility only; original checkpoint',updated=__import__('time').time())
    previous.pop('returncode',None);(cell/'status.json').write_text(json.dumps(previous,indent=2))
    for split in ['forget05','forget10']:
        run('native_1b_'+split,[str(py),'scripts/reproduction/npo/gb200/run_experiment.py','--world-size','1','--stages','3','--attention','flash_attention_2','--seed','0','--model','Llama-3.2-1B-Instruct','--forget',split,'--tag','native_554_1130_20261006_r2_1b_'+split])
    state.update(phase='DONE',finished=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception as error:
    state.update(phase='FAILED',error=repr(error));save();raise
