"""Finite smoke queue, after audited retain95 and separate CUDA executor acceptance.

This source is prepared locally. It neither deploys its package nor supplies the
still missing development CUDA acceptance. Initial training is never retried;
an explicit new reload run may reuse only a completed training export.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

CONSTRUCTION = Path('/data/spf-npo-20261006-r4')
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
IDENTITY = 'c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8'
FULL_AUDIT_SHA = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
FREEZE_SHA = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
ADAPTER_SHA = '71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de'
EXECUTORS = ('evaluate_construction_baseline.py', 'generate_development.py',
             'generate_development_conversation.py', 'evaluate_development_knowledge.py')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_json(path, expected):
    raw = Path(path).read_bytes()
    if not re.fullmatch('[0-9a-f]{64}', expected) or digest(raw) != expected:
        raise ValueError('Bound JSON SHA mismatch: '+str(path))
    return json.loads(raw)


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    tmp.replace(path)


def completed_reference(status, audit):
    names = ['spf-retain95-s0-'+suffix for suffix in ('train','reload','audit')]
    if (status.get('status') != 'CONSTRUCTION_DONE_PENDING_GATE_NPO'
            or status.get('pid') != 45219
            or any(name not in status.get('completed', []) for name in names)):
        raise ValueError('Original r4 reference train/reload/audit must be complete')
    for name in names:
        rows = [row for row in status['stages'] if row['name'] == name]
        if len(rows) != 1 or rows[0].get('returncode') != 0 or 'finished_at' not in rows[0]:
            raise ValueError('Completed reference command evidence missing: '+name)
    expected = dict(status='pass', method='spf', split='retain95', construction_seed=0,
                    actual_epochs=5.0, actual_updates=595, actual_examples=19000,
                    unique_examples=3800, manifest_sha256=FREEZE_SHA)
    if any(audit.get(k) != v for k,v in expected.items()):
        raise ValueError('Actual matched retain95 audit required')
    final = audit['final_checkpoint']
    if (audit['fresh_reload']['status'] != 'pass' or final['parameter_tensors'] != 291
            or final['layers'] != 32 or final['weights_bytes'] <= 16_000_000_000
            or not final['files']
            or sum(row['size'] for row in final['files']) != final['weights_bytes']
            or sum(row['tensors'] for row in final['files']) != 291
            or [row['step'] for row in audit['trajectory_checkpoints']] != [149,298,447,595]):
        raise ValueError('Complete reference export, trajectory and fresh reload required')


def process_exited(result):
    if result['returncode'] not in (0,1) or result['stderr'].strip():
        raise ValueError('Unable to establish original queue process state')
    rows = result['stdout'].strip().splitlines()
    if not rows or rows[0].split()[:2] != ['PID','STAT']:
        raise ValueError('Original queue ps header required')
    if result['returncode'] == 1 and len(rows) == 1:
        return
    if (result['returncode'] == 0 and len(rows) == 2 and rows[1].split()[0] == '45219'
            and rows[1].split()[1].startswith('Z')):
        return
    raise ValueError('Original queue still alive or process evidence ambiguous')


def gpu_idle(result):
    if result['returncode'] != 0 or result['stderr'].strip() or result['stdout'].strip():
        raise ValueError('All original GPU compute processes must have exited')


def validate_snapshot(snapshot, now):
    captured = datetime.fromisoformat(snapshot['snapshot_utc'])
    if captured.tzinfo is None or not 0 <= (now-captured).total_seconds() <= 600:
        raise ValueError('New complete snapshot within ten minutes required')
    if snapshot['launch']['pid'] != 45219 or snapshot['reference_audit'] is None:
        raise ValueError('Original r4 launch and completed reference audit required')
    completed_reference(snapshot['status'], snapshot['reference_audit'])
    process_exited(snapshot['process'])
    gpu_idle(snapshot['compute_processes'])
    if any(snapshot[k]['returncode'] != 0 for k in ('gpu','filesystem')):
        raise ValueError('Complete original GPU/filesystem snapshot required')


def development_accepted(acceptance, adapter):
    if (acceptance.get('status') != 'DEVELOPMENT_CUDA_EXECUTORS_PASS'
            or acceptance.get('checkpoint_identity_sha256') != IDENTITY
            or acceptance.get('checkpoint_audit_sha256') != FULL_AUDIT_SHA
            or acceptance.get('adapter_audit_sha256') != ADAPTER_SHA
            or acceptance.get('world_size') != 1
            or acceptance.get('generation_caps') != adapter['generation_caps']
            or acceptance.get('base_tokenizer_revision') != adapter['base_tokenizer_revision']
            or acceptance.get('external_judge_executed') is not False
            or set(acceptance.get('executors', {})) != set(EXECUTORS)):
        raise ValueError('Separate actual development CUDA acceptance still required')
    for name, row in acceptance['executors'].items():
        if (row.get('returncode') != 0 or row.get('gpu_runtime_validated') is not True
                or row.get('process_pid', 0) <= 0 or row.get('examples', 0) <= 0
                or row.get('source_sha256') != adapter['sources'][name]['adapter_sha256']
                or not re.fullmatch('[0-9a-f]{64}', row.get('raw_log_sha256',''))):
            raise ValueError('Actual executor command/log/CUDA evidence missing: '+name)


def command_result(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def exact_commands(c, contract_sha):
    root = c['task_root']
    return dict(cwd=c['repository'], environment=dict(
        SPF_NPO_SMOKE_CONTRACT=root+'/contract.private.json',
        SPF_NPO_SMOKE_CONTRACT_SHA256=contract_sha, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache', TOKENIZERS_PARALLELISM='false',
        WANDB_MODE='disabled', HYDRA_FULL_ERROR='1'),
        train=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
            root+'/code/spf_npo_native_probe.py','--config-path',c['config_dir'],
            '--config-name','spf_npo_smoke'],
        fresh_reload=[PYTHON,root+'/code/verify_spf_npo_smoke_reload.py',
            '--contract',root+'/contract.private.json','--contract-sha256',contract_sha])


def hydra_command(c, output):
    # Compose through actual Hydra; no native entrypoint or model imports.
    script = '''import json,sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
config_dir, output = sys.argv[1:]
raw = json.loads((Path(config_dir)/"spf_npo_smoke.yaml").read_bytes())
model = OmegaConf.load(Path(config_dir)/"model/Llama-3.1-8B-Instruct.yaml")
base = {k:v for k,v in raw.items() if k not in ("defaults","hydra")}
expected = OmegaConf.to_container(OmegaConf.merge({"model":model},base),resolve=True)
with initialize_config_dir(version_base=None,config_dir=config_dir):
    actual = OmegaConf.to_container(compose(config_name="spf_npo_smoke"),resolve=True)
if actual != expected: raise ValueError("Hydra/default composition differs from immutable smoke config")
Path(output).write_text(json.dumps(dict(status="HYDRA_COMPOSITION_PASS",config=actual),indent=2)+"\\n",encoding="utf8")
print(json.dumps(dict(status="HYDRA_COMPOSITION_PASS",model=actual["model"]["model_args"]["pretrained_model_name_or_path"])))
'''
    return [PYTHON,'-c',script,c['config_dir'],str(output)]


def stage(state, path, name, command, cwd, env, require_idle=False):
    row = dict(name=name, command=command, cwd=str(cwd),
               environment_overrides=env['overrides'], log=str(path.parent/(name+'.raw.log')),
               started_at=time.time())
    state.update(stage=name); state['stages'].append(row); atomic(path,state)
    if require_idle:
        idle = command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'])
        row['gpu_before_stage'] = idle; atomic(path,state); gpu_idle(idle)
    with Path(row['log']).open('xb') as stream:
        p = subprocess.Popen(command,cwd=cwd,env=env['values'],stdout=stream,stderr=subprocess.STDOUT)
        row['process_pid'] = p.pid; atomic(path,state)
        returncode = p.wait()
    row.update(returncode=returncode,finished_at=time.time(),
               raw_log_sha256=digest(Path(row['log']).read_bytes()))
    atomic(path,state)
    if returncode:
        raise RuntimeError(name+' failed; preserve evidence and do not retry training')
    state['completed'].append(name); atomic(path,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','package-sha256','controller-sha256','snapshot','snapshot-sha256',
                 'development-acceptance','development-acceptance-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('initial','reload-only'),default='initial')
    p.add_argument('--run-id',required=True)
    a=p.parse_args(); task=Path(a.task).resolve()
    if (task.parent != Path('/data') or not task.name.startswith('spf-npo-smoke-')
            or task.is_symlink() or str(task) != a.task or sys.executable != PYTHON
            or digest(Path(__file__).read_bytes()) != a.controller_sha256
            or not re.fullmatch('initial|reload-[a-z0-9-]+',a.run_id)
            or (a.mode=='initial') != (a.run_id=='initial')):
        raise ValueError('Exact isolated Linux task/interpreter/controller/run identity required')
    package=bound_json(task/'package.private.json',a.package_sha256)
    for name,spec in package['files'].items():
        file=(task/name).resolve()
        if not file.is_relative_to(task): raise ValueError('Package path escaped task')
        raw=file.read_bytes()
        if len(raw)!=spec['size'] or digest(raw)!=spec['sha256']:
            raise ValueError('Immutable package changed: '+name)
    c=bound_json(task/'contract.private.json',package['contract_sha256'])
    if c['task_root']!=str(task): raise ValueError('Package task identity changed')
    sys.path.insert(0,str(task/'code'))
    from prepare_spf_npo_smoke import validate_contract_config
    from spf_npo_runtime import audit_trace
    validate_contract_config(c,json.loads((task/'configs/spf_npo_smoke.yaml').read_bytes()),
                             json.loads((task/'deepspeed.json').read_bytes()))
    commands=json.loads((task/'commands.private.json').read_bytes())
    expected=exact_commands(c,package['contract_sha256'])
    if any(commands.get(k)!=v for k,v in expected.items()):
        raise ValueError('Exact sealed command/environment required')
    snapshot=bound_json(a.snapshot,a.snapshot_sha256)
    validate_snapshot(snapshot,datetime.now(timezone.utc))
    adapter=bound_json('/data/spf-development-20261007-r1/adapter/adapter-audit.json',ADAPTER_SHA)
    development=bound_json(a.development_acceptance,a.development_acceptance_sha256)
    development_accepted(development,adapter)
    import fcntl
    with (task/'smoke-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (task/'runs'/a.run_id).exists(): raise FileExistsError('Run exists; preserve original outcome')
        live_status=json.loads((CONSTRUCTION/'status.json').read_bytes())
        refraw=(CONSTRUCTION/'spf-retain95-s0-audit.json').read_bytes(); ref=json.loads(refraw)
        completed_reference(live_status,ref)
        if ref!=snapshot['reference_audit']: raise ValueError('Snapshot reference audit changed')
        process_exited(command_result(['ps','-p','45219','-o','pid,stat,args']))
        gpu_idle(command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
        gpu=command_result(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
        if gpu['returncode']!=0 or gpu['stderr'].strip() or len(gpu['stdout'].strip().splitlines())!=1:
            raise ValueError('Single original GPU required')
        if shutil.disk_usage(task).free < 80*1024**3:
            raise ValueError('80GiB free export workspace required; this is not a PVC quota assertion')
        if digest((CONSTRUCTION/'spf-full-s0-audit.json').read_bytes())!=FULL_AUDIT_SHA or digest((CONSTRUCTION/'frozen.json').read_bytes())!=FREEZE_SHA:
            raise ValueError('Original audited full target/freeze changed')
        for name,sha in c['native_source_sha256'].items():
            if digest((Path(c['repository'])/name).read_bytes())!=sha:
                raise ValueError('Native executor source changed: '+name)
        output=Path(c['output_dir'])
        if a.mode=='initial':
            if output.exists() or (task/'runs/initial').exists():
                raise FileExistsError('Training cannot replace or restart a checkpoint or initial run')
        else:
            previous=json.loads((task/'runs/initial/status.json').read_bytes())
            if ('train' not in previous['completed'] or previous.get('stage')!='fresh-reload'
                    or 'fresh-reload' in previous['completed'] or previous['pid']==os.getpid()):
                raise ValueError('Only a completed initial training export may be reused')
            old=Path('/proc')/str(previous['pid'])/'stat'
            if old.exists() and old.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':
                raise ValueError('Previous smoke queue still alive')
            runtime=json.loads((output/'spf-npo-runtime.private.json').read_bytes()); audit_trace(runtime)
            trainer_state=json.loads((output/'trainer_state.json').read_bytes())
            if trainer_state.get('global_step')!=2 or trainer_state.get('epoch')!=1.0 or not (output/'spf-npo-reload-capture.json').is_file():
                raise ValueError('Actual complete training and numerical capture required')
            existing_reload=output/'spf-npo-fresh-reload-audit.json'
            if existing_reload.exists() and json.loads(existing_reload.read_bytes()).get('fresh_process_reload_verified') is True:
                raise ValueError('Successful fresh reload is complete and must not be repeated')
        run=task/'runs'/a.run_id;run.mkdir(parents=True,exist_ok=False);path=run/'status.json'
        state=dict(status='RUNNING',pid=os.getpid(),mode=a.mode,completed=[],stages=[],
                   started_at=time.time(),snapshot_sha256=a.snapshot_sha256,
                   reference_audit_sha256=digest(refraw),development_acceptance_sha256=a.development_acceptance_sha256,
                   controller_sha256=a.controller_sha256,package_sha256=a.package_sha256,
                   contract_sha256=package['contract_sha256'],pilot_candidates_frozen=False,
                   training_resume_supported=False,target_gate_evaluated=False)
        atomic(path,state)
        values={**os.environ,**expected['environment'],'PYTHONPATH':c['repository']+'/src'}
        env=dict(values=values,overrides={**expected['environment'],'PYTHONPATH':c['repository']+'/src'})
        try:
            if a.mode=='initial':
                cpu_env=dict(values={**values,'CUDA_VISIBLE_DEVICES':''},
                             overrides={**env['overrides'],'CUDA_VISIBLE_DEVICES':''})
                stage(state,path,'hydra-compose',hydra_command(c,run/'hydra-compose.private.json'),c['repository'],cpu_env)
                stage(state,path,'train',expected['train'],c['repository'],env,require_idle=True)
            else:
                state['reused_training_export']=dict(initial_run='initial',runtime_sha256=digest((output/'spf-npo-runtime.private.json').read_bytes()))
                atomic(path,state)
            stage(state,path,'fresh-reload',expected['fresh_reload'],c['repository'],env,require_idle=True)
            reload=json.loads((output/'spf-npo-fresh-reload-audit.json').read_bytes())
            if reload.get('status')!='SMOKE_TRAINING_EXPORT_FRESH_RELOAD_PASS' or reload.get('fresh_process_reload_verified') is not True or reload.get('contract_sha256')!=package['contract_sha256']:
                raise ValueError('Actual fresh-process reload audit required')
            state.update(status='SMOKE_COMPLETE_PENDING_PILOT_FREEZE',finished_at=time.time())
        except BaseException as error:
            state.update(status='FAILED',error=repr(error),finished_at=time.time(),
                         recovery='Preserve raw evidence; completed training may only use explicit reload-only recovery')
            atomic(path,state); raise
        atomic(path,state)


if __name__=='__main__':
    main()
