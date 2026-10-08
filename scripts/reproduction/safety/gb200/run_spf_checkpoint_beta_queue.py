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

ROOT = Path('/data/spf-checkpoint-beta-20261008-r1')
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
    if (plan.get('checkpoint_steps') != [157,313,469] or plan.get('beta1_values') != [0.9,0.5]
            or plan.get('diagnostic_updates_per_beta') != 32 or plan.get('paid_judge_calls_in_gpu_queue') != 0
            or plan.get('proposed_new_judge_request_cap') != 2376 or plan.get('retry_count') != 0
            or plan.get('existing_final_checkpoint_repeated') is not False
            or plan.get('formal_hexphi_selection') is not False):
        raise ValueError('Fixed user-authorized early checkpoints and paired diagnostics required')

def metadata_check(case):
    target=Path(case['checkpoint'])
    if (case['step'] not in (157,313,469) or target != CONSTRUCTION/'spf-full-s0'/('checkpoint-'+str(case['step']))
            or set(case['metadata_sha256']) != {'config.json','generation_config.json','tokenizer_config.json','trainer_state.json'}):
        raise ValueError('Original early trajectory path and complete small metadata required')
    if sorted(f.name for f in target.glob('*.safetensors')) != ['model.safetensors']:
        raise ValueError('Original audited shard set required')
    if (target/'model.safetensors').stat().st_size != 16060556616:
        raise ValueError('Original audited shard size required')
    for name,digest in case['metadata_sha256'].items(): bound(target/name,digest)
    if json.loads((target/'trainer_state.json').read_bytes())['global_step'] != case['step']:
        raise ValueError('Early checkpoint counter mismatch')
    identity=dict(checkpoint=case['checkpoint'],step=case['step'],weights_sha256=case['weights_sha256'],
                  metadata_sha256=case['metadata_sha256'],original_full_audit_sha256=FULL)
    if sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()) != case['identity_sha256']:
        raise ValueError('New early checkpoint identity binding mismatch')

def prepare_case(case,executor_plan):
    metadata_check(case)
    case_dir=ROOT/('checkpoint-'+str(case['step']))
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
                 weight_byte_hash_verification='reuse original complete trajectory audit; no repeated weight hashing',
                 intermediate_fresh_reload_numerical_probe_available=False)
    audit_path=code.parent/'adapter-audit.json'
    audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    os.environ.update(SPF_DIAG_CASE_TASK=str(case_dir),SPF_DIAG_IDENTITY=case['identity_sha256'],
                      SPF_DIAG_ADAPTER_SHA=sha(audit_path.read_bytes()))
    m=module(ROOT/'validation-code/spf_checkpoint_executor.py','checkpoint_executor_'+str(case['step']))
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
    if sys.executable!=PYTHON or os.name!='posix' or sha(Path(__file__).read_bytes())!=a.source_sha256:
        raise ValueError('Exact runtime and new source binding required')
    plan=bound(a.plan,a.plan_sha256);validate_plan(plan)
    for path,digest in plan['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('New/existing source changed: '+Path(path).name)
    guard=module(ORIGINAL/'validation-code/run_spf_development_recovery_guards.py','original_completion_guard')
    snapshot=bound(a.snapshot,a.snapshot_sha256)
    guard.validate_snapshot(snapshot,datetime.now(timezone.utc))
    if snapshot['checkpoint_beta_diagnostic']['new_root_exists']:
        raise ValueError('Original snapshot must establish absent new run')
    original=bound(CONSTRUCTION/'spf-full-s0-audit.json',FULL);bound(CONSTRUCTION/'frozen.json',FREEZE)
    if [case['step'] for case in plan['cases']] != [157,313,469]:
        raise ValueError('Exactly the three new early checkpoints required')
    for case in plan['cases']:
        item=next(row for row in original['trajectory_checkpoints'] if row['step']==case['step'])
        if item['files'][0]['sha256'] != case['weights_sha256'] or len(item['files']) != 1:
            raise ValueError('Reuse original audited early weight hash required')
        metadata_check(case)
    import fcntl
    with (ORIGINAL/'development-cuda.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'status.json').exists():raise FileExistsError('Preserve existing new run')
        state=dict(status='RUNNING',pid=os.getpid(),started_at=time.time(),completed=[],stages=[],
                   plan_sha256=a.plan_sha256,source_sha256=a.source_sha256,new_paid_judge_calls=0,
                   new_scope='early157/313/469 plus 2x32 optimizer geometry; no final rerun')
        write(ROOT/'status.json',state)
        env=dict(HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache/hub',HF_HUB_CACHE='/data/smart-mfg/jimmy-lin/hf_cache/hub',
                 HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                 PYTHONPATH=str(CONSTRUCTION/'repo/src'),TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
        diagnostic=ROOT/'validation-code/spf_beta_geometry_diagnostic.py'
        try:
            stage(state,'observer-cpu-selftest',[PYTHON,str(diagnostic),'--self-test-cpu'],
                  {**env,'CUDA_VISIBLE_DEVICES':''},guard)
            for beta in plan['beta1_values']:
                name='beta-'+str(beta).replace('.','')
                command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',
                         str(diagnostic),'--beta1',str(beta),'--output',str(ROOT/name)]
                stage(state,name,command,env,guard)
                summary=json.loads((ROOT/name/'geometry-summary.private.json').read_bytes())
                if summary['actual_updates']!=32 or summary['actual_examples']!=1024 or summary['beta1']!=beta:
                    raise ValueError('New actual paired diagnostic incomplete')
            for case in plan['cases']:
                m,case_dir,audit=prepare_case(case,plan)
                run=case_dir/'evaluation';run.mkdir(exist_ok=False)
                state_case=dict(status='RUNNING',pid=os.getpid(),commands=m.plan(run),completed=[],stages=[],
                                controller_sha256=sha(Path(m.__file__).read_bytes()),checkpoint_step=case['step'])
                m.write(run/'status.json',state_case)
                for name in m.STAGES:
                    state['stage']='checkpoint-'+str(case['step'])+'/'+name;write(ROOT/'status.json',state)
                    metadata_check(case)
                    m.execute(state_case,run,name,guard)
                state_case.update(status='CHECKPOINT_EVALUATION_COMPLETE_PENDING_NEW_JUDGE',target_gate='insufficient_evidence')
                m.write(run/'status.json',state_case)
                state['completed'].append('checkpoint-'+str(case['step']));write(ROOT/'status.json',state)
            state.update(status='EARLY_CHECKPOINTS_AND_PREFIX_GEOMETRY_COMPLETE_PENDING_JUDGE',finished_at=time.time())
        except BaseException as error:
            state.update(status='FAILED_NO_RETRY',error_type=type(error).__name__,finished_at=time.time())
            write(ROOT/'status.json',state);raise
        write(ROOT/'status.json',state)

if __name__=='__main__':main()
