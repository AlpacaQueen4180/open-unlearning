"""Exclusive new early-checkpoint evaluations and paired prefix diagnostics.

No old run is retried. All paid Judge work stays outside this GPU queue.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/spf-beta05-full-20261010-r2')
ORIGINAL = Path('/data/spf-development-20261007-r1')
CONSTRUCTION = Path('/data/spf-npo-20261006-r4')
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
FREEZE = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
FULL = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
ADAPTER = '71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def bound(path,digest):
    raw=Path(path).read_bytes()
    if sha(raw)!=digest: raise ValueError('Bound file changed: '+Path(path).name)
    return json.loads(raw)
def write(path,value):
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def validate_plan(plan):
    expected=dict(parameter_delta={'adam_beta1':{'baseline':.9,'candidate':.5}},
        new_paid_judge_calls=0,retry_count=0,existing_training_repeated=False,formal_hexphi_selection=False,
        updates=625,epochs=5,examples=20000,unique_examples=4000,seed=0,
        learning_rate=1e-5,beta2=.999,weight_decay=.01,microbatch=4,accumulation=8,global_batch=32,warmup_steps=125)
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Single beta1 change to full SPF only')

def validate_failed_candidate(branch):
    status=branch.get('status') or {}
    if (status.get('pid')!=8483 or status.get('status')!='FAILED_NO_RETRY' or status.get('completed')!=[]
        or branch.get('optimizer_binding') is not None or branch.get('candidate_audit') is not None):
        raise ValueError('Bound original pre-step failure only')
    rows=status.get('stages',[])
    if (len(rows)!=1 or rows[0].get('name')!='train' or rows[0].get('process_pid')!=8486 or rows[0].get('returncode')!=1
        or rows[0].get('raw_log_sha256')!='9a37e105ef111e565fea1bd13ba800c76bcfe93d58fadb9ae9015d2404e967f8'):
        raise ValueError('Bound original command/raw failure required')
    expected={'run_spf_beta05_full.py':'0abffe63440a2fef5e3cbb05386431e1ab6cd54684f96bab6a3b3b423984a08e',
              'run_spf_beta05_full_queue.py':'176087bc5248c04f04acf17792348a7f12c26a02bdd59fe9796af97b992a9aaa',
              'plan.private.json':'d495b5b1b6ed1126fdf7938ccc79f612760bc7dffc15685e4f4f4e338709762e'}
    if any((branch.get('source_files',{}).get(name) or {}).get('sha256')!=digest for name,digest in expected.items()):
        raise ValueError('Original failed source/plan required')

def validate_failed_runtime():
    old=Path('/data/spf-beta05-full-20261009-r1')
    status=bound(old/'status.json','34da4de42f22765b36d46f873726d3f1f3453216407d1aa8adb8437a4282e0da')
    if (status['completed']!=[] or status['status']!='FAILED_NO_RETRY' or (old/'spf-beta05-full-s0').exists()
        or (old/'optimizer-binding.private.json').exists() or (old/'spf-beta05-full-s0-audit.json').exists()):
        raise ValueError('Do not rerun completed training')
    if sha((old/'train.raw.log').read_bytes())!='9a37e105ef111e565fea1bd13ba800c76bcfe93d58fadb9ae9015d2404e967f8':
        raise ValueError('Preserved failed raw required')
    for pid in (8483,8486,8554):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':
            raise ValueError('All original failure processes must exit')

def metadata_check(case):
    target=Path(case['checkpoint'])
    if target!=ROOT/'spf-beta05-full-s0' or case['step']!=625:raise ValueError('New completed candidate only')
    audit=bound(ROOT/'spf-beta05-full-s0-audit.json',case['candidate_audit_sha256'])
    if audit['status']!='pass' or audit['actual_updates']!=625 or audit['actual_examples']!=20000 or audit['fresh_reload']['status']!='pass':raise ValueError('Complete new actual audit/reload required')
    if case['weights_sha256']!=audit['final_checkpoint']['files'][0]['sha256']:raise ValueError('New complete audit weights required')
    if sorted(f.name for f in target.glob('*.safetensors'))!=['model.safetensors'] or (target/'model.safetensors').stat().st_size!=16060556616:raise ValueError('Exact new shard metadata')
    for name,digest in case['metadata_sha256'].items():bound(target/name,digest)
    if json.loads((target/'trainer_state.json').read_bytes())['global_step']!=625:raise ValueError('Full candidate counter')
    identity={k:v for k,v in case.items() if k!='identity_sha256'}
    if sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())!=case['identity_sha256']:raise ValueError('New candidate identity required')

def prepare_case(case,executor_plan):
    metadata_check(case)
    case_dir=ROOT/'candidate-development'
    case_dir.mkdir(exist_ok=False)
    for name,target in [('repo',ORIGINAL/'repo'),('bundle',ORIGINAL/'bundle')]:
        (case_dir/name).symlink_to(target,target_is_directory=True)
    old=ORIGINAL/'adapter-generation-api-r2'
    old_audit=bound(old/'adapter-audit.json',ADAPTER)
    audit=json.loads(json.dumps(old_audit))
    code=case_dir/'adapter-generation-api-r2/code';code.mkdir(parents=True)
    old_target=old_audit['checkpoint'];old_identity=old_audit['checkpoint_identity_sha256']
    for name,spec in old_audit['sources'].items():
        raw=(old/'code'/name).read_bytes()
        if sha(raw)!=spec['adapter_sha256']: raise ValueError('Original completed adapter drift')
        text=raw.decode('utf8')
        if name!='score_development_ifbench.py':
            if text.count(old_target)!=2 or text.count(old_identity)!=1:
                raise ValueError('Exact two loader/metadata paths and one identity edit required')
            text=text.replace(old_target,case['checkpoint']).replace(old_identity,case['identity_sha256'])
            restored=text.replace(case['checkpoint'],old_target).replace(case['identity_sha256'],old_identity)
            if restored.encode('utf8')!=raw: raise ValueError('Nonidentity adapter change')
        new=text.encode('utf8');compile(new,name,'exec');(code/name).write_bytes(new)
        audit['sources'][name].update(previous_adapter_sha256=sha(raw),adapter_sha256=sha(new),
                                    retarget_only_inverse_bytes_verified=True)
    helper=(old/'code/spf_development_generation.py').read_bytes()
    if sha(helper)!=audit['generation_helper_sha256']: raise ValueError('Generation helper changed')
    (code/'spf_development_generation.py').write_bytes(helper)
    audit.update(checkpoint=case['checkpoint'],checkpoint_identity_sha256=case['identity_sha256'],
                 checkpoint_files={**case['metadata_sha256'],'model.safetensors':case['weights_sha256']},
                 trajectory_step=case['step'],original_full_audit_sha256=FULL,
                 weight_byte_hash_verification='reuse newly completed beta05 full byte audit; no repeated weight hashing',
                 candidate_audit_sha256=case['candidate_audit_sha256'],parameter_delta={'adam_beta1':{'baseline':.9,'candidate':.5}},
                 intermediate_fresh_reload_numerical_probe_available=False)
    audit_path=code.parent/'adapter-audit.json'
    audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    os.environ.update(SPF_DIAG_CASE_TASK=str(case_dir),SPF_DIAG_IDENTITY=case['identity_sha256'],
                      SPF_DIAG_ADAPTER_SHA=sha(audit_path.read_bytes()))
    m=module(ROOT/'validation-code/spf_beta05_serialization_r2_development_executor.py','checkpoint_executor_'+str(case['step']))
    m.validate_dependencies(case_dir/'repo',bound(CONSTRUCTION/'frozen.json',FREEZE))
    for name,digest,count in m.DATA.values():
        if len(bound(case_dir/'bundle/assets'/name,digest))!=count: raise ValueError('Fixed development input mismatch')
    return m,case_dir,audit

def stage(state,name,command,env,guard,cwd=CONSTRUCTION/'repo'):
    guard.gpu_idle(guard.command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
    row=dict(name=name,command=command,cwd=str(cwd),environment_overrides=env,
             log=str(ROOT/(name+'.raw.log')),started_at=time.time())
    state['stage']=name;state['stages'].append(row);write(ROOT/'status.json',state)
    with Path(row['log']).open('xb') as f:
        proc=subprocess.Popen(command,cwd=cwd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
        row['process_pid']=proc.pid;write(ROOT/'status.json',state);row['returncode']=proc.wait()
    row.update(finished_at=time.time(),raw_log_sha256=sha(Path(row['log']).read_bytes()))
    write(ROOT/'status.json',state)
    if row['returncode']:raise RuntimeError('New stage failed; no retry: '+name)
    state['completed'].append(name);write(ROOT/'status.json',state)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','snapshot','snapshot-sha256','source-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if sys.executable!=PYTHON or os.name!='posix' or sha(Path(__file__).read_bytes())!=a.source_sha256:raise ValueError('Original runtime and new controller')
    plan=bound(a.plan,a.plan_sha256);validate_plan(plan)
    for path,digest in plan['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('Bound source drift: '+Path(path).name)
    guard=module(ORIGINAL/'validation-code/run_spf_development_recovery_guards.py','original_completion_guard')
    snapshot=bound(a.snapshot,a.snapshot_sha256);guard.validate_snapshot(snapshot,datetime.now(timezone.utc))
    validate_failed_candidate(snapshot['beta05_full_candidate']);validate_failed_runtime()
    bound(CONSTRUCTION/'spf-full-s0-audit.json',FULL);bound(CONSTRUCTION/'frozen.json',FREEZE)
    import fcntl
    with (ORIGINAL/'development-cuda.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'status.json').exists() or (ROOT/'spf-beta05-full-s0').exists():raise FileExistsError('Exclusive new candidate')
        state=dict(status='RUNNING',pid=os.getpid(),started_at=time.time(),completed=[],stages=[],plan_sha256=a.plan_sha256,
            source_sha256=a.source_sha256,parameter_delta=plan['parameter_delta'],new_paid_judge_calls=0,target_gate='insufficient_evidence')
        write(ROOT/'status.json',state)
        env=dict(HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache/hub',HF_HUB_CACHE='/data/smart-mfg/jimmy-lin/hf_cache/hub',
            HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PYTHONPATH=str(CONSTRUCTION/'repo/src'),
            TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
        target=ROOT/'spf-beta05-full-s0'
        try:
            train=ROOT/'validation-code/run_spf_beta05_full_serialization_r2.py'
            stage(state,'train',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',str(train),
                '--source-sha256',plan['source_sha256'][str(train)],'--plan-sha256',a.plan_sha256],env,guard)
            binding=json.loads((ROOT/'optimizer-binding.private.json').read_bytes())
            if binding['status']!='ACTUAL_BETA05_FULL_OPTIMIZER_BOUND' or binding['beta1']!=.5 or binding['beta2']!=.999 or not binding['actual_nonbeta_training_args_match']:raise ValueError('Actual optimizer binding missing')
            stage(state,'fresh-reload',[PYTHON,'-m','construction.cli','verify','--checkpoint',str(target)],env,guard)
            stage(state,'full-byte-audit',[PYTHON,str(ROOT/'validation-code/audit_spf_target.py'),'--target',str(target),'--output',str(ROOT/'spf-beta05-full-s0-audit.json')],{**env,'CUDA_VISIBLE_DEVICES':''},guard)
            audit_raw=(ROOT/'spf-beta05-full-s0-audit.json').read_bytes();audit=json.loads(audit_raw)
            resolved=json.loads((target/'resolved_training.json').read_bytes())
            if resolved['args']['adam_beta1']!=.5 or resolved['args']['adam_beta2']!=.999 or resolved['numerical_precision']!={'cuda_matmul_allow_tf32':False,'float32_matmul_precision':'highest'}:raise ValueError('Actual candidate beta/precision')
            case=dict(step=625,checkpoint=str(target),weights_sha256=audit['final_checkpoint']['files'][0]['sha256'],
                metadata_sha256={name:sha((target/name).read_bytes()) for name in ('config.json','generation_config.json','tokenizer_config.json','trainer_state.json')},
                candidate_audit_sha256=sha(audit_raw),baseline_full_audit_sha256=FULL,parameter_delta=plan['parameter_delta'])
            case['identity_sha256']=sha(json.dumps(case,sort_keys=True,separators=(',',':')).encode())
            write(ROOT/'candidate-identity.private.json',case)
            m,case_dir,adapter=prepare_case(case,plan)
            run=case_dir/'evaluation';run.mkdir(exist_ok=False)
            child_state=dict(status='RUNNING',pid=os.getpid(),commands=m.plan(run),completed=[],stages=[],controller_sha256=sha(Path(m.__file__).read_bytes()),checkpoint_step=625)
            m.write(run/'status.json',child_state)
            for name in m.STAGES:
                state['stage']='candidate-development/'+name;write(ROOT/'status.json',state)
                metadata_check(case);m.execute(child_state,run,name,guard)
            child_state.update(status='BETA05_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE',target_gate='insufficient_evidence');m.write(run/'status.json',child_state)
            state['completed'].append('candidate-development')
            state.update(status='BETA05_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE',finished_at=time.time())
        except BaseException as error:
            state.update(status='FAILED_NO_RETRY',error_type=type(error).__name__,finished_at=time.time());write(ROOT/'status.json',state);raise
        write(ROOT/'status.json',state)

if __name__=='__main__':main()
