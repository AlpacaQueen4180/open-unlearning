from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-beta05-lr5e6-full-20261010-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'beta05-lr5e6-full-launch-20261010.json').exists() and not (V/'beta05-lr5e6-full-launch-20261010.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-beta05-lr5e6-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-10T11:06:10.757057+00:00')).total_seconds() <= 600
previous=Path('/data/spf-beta05-full-20261010-r2')
assert sha((previous/'spf-beta05-full-s0-audit.json').read_bytes())=='63e72cb101b8056c938021e9518361280ed0df9f9238df04034342bf71d4110b'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os
expected={'run_spf_beta05_lr5e6_full_queue.py': {'size': 15630, 'sha256': 'eb6ca91386852591299d95e3f54e6309d1c5ba6b78064a8a8d726de3fc28cf08'}, 'run_spf_beta05_lr5e6_full.py': {'size': 6873, 'sha256': '8a362f7cbdf4a82b38ef4fc010833a6e1bda2c352a81d51e0a048d5019ec3f75'}, 'spf_beta05_lr5e6_development_executor.py': {'size': 16335, 'sha256': '535f7fe622ae6da568ce8a4f090b84410dacafa5dd6e8335782832b26cb9da27'}, 'plan.private.json': {'size': 3391, 'sha256': '8631a99a9f419037a04677c29384862506dfabbddaf9138d0741840a00ca0d6b'}, 'bound-snapshot.private.json': {'size': 14944, 'sha256': 'af300bfac39afe7b3c175db7f1daa685eda5d3303433a06014bcab015d710704'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_lower_lr_queue',V/'run_spf_beta05_lr5e6_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_beta05_lr5e6_full_queue.py'),
    '--source-sha256','eb6ca91386852591299d95e3f54e6309d1c5ba6b78064a8a8d726de3fc28cf08','--plan',str(V/'plan.private.json'),'--plan-sha256','8631a99a9f419037a04677c29384862506dfabbddaf9138d0741840a00ca0d6b',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','af300bfac39afe7b3c175db7f1daa685eda5d3303433a06014bcab015d710704']
receipt=V/'beta05-lr5e6-full-launch-20261010.json';log=V/'beta05-lr5e6-full-launch-20261010.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='af300bfac39afe7b3c175db7f1daa685eda5d3303433a06014bcab015d710704',plan_sha256='8631a99a9f419037a04677c29384862506dfabbddaf9138d0741840a00ca0d6b',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
