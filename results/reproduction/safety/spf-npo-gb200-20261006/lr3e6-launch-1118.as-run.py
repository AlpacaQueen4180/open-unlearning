from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-beta05-lr3e6-full-20261011-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'beta05-lr3e6-full-launch-20261011.json').exists() and not (V/'beta05-lr3e6-full-launch-20261011.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-beta05-lr3e6-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-11T03:24:03.961377+00:00')).total_seconds() <= 600
previous=Path('/data/spf-beta05-lr5e6-full-20261010-r1')
assert sha((previous/'spf-beta05-lr5e6-full-s0-audit.json').read_bytes())=='b42e934ac66d38a2918bd45302b362176653f49b1003e3f44d310f6f822118f5'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140,10758,10761,10829,11297,11577):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os
expected={'run_spf_beta05_lr3e6_full_queue.py': {'size': 15658, 'sha256': 'caba122d48d89e3fc40d9d7327300910fb614883f28a60ef50ea64d2d8e31a14'}, 'run_spf_beta05_lr3e6_full.py': {'size': 6873, 'sha256': '9e3fd1c8c14bb3771c33d6679af36639f0f78007a1a29308479ae433f5f720c2'}, 'spf_beta05_lr3e6_development_executor.py': {'size': 16335, 'sha256': '74fb508d71d667ef0de08fee3e471b85ba377bc00298a026bf1106a6dbecb643'}, 'plan.private.json': {'size': 3391, 'sha256': '0ef32eae9962ffa418e4d6d9f672dc2e67aeb51fa3b43cdc4905cd4086fec3c8'}, 'bound-snapshot.private.json': {'size': 15016, 'sha256': 'b482ab8261a6e793722bc22f4af32d6ba0ad62a6b31e1f69aabde725ffc0671b'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_lower_lr_queue',V/'run_spf_beta05_lr3e6_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_beta05_lr3e6_full_queue.py'),
    '--source-sha256','caba122d48d89e3fc40d9d7327300910fb614883f28a60ef50ea64d2d8e31a14','--plan',str(V/'plan.private.json'),'--plan-sha256','0ef32eae9962ffa418e4d6d9f672dc2e67aeb51fa3b43cdc4905cd4086fec3c8',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','b482ab8261a6e793722bc22f4af32d6ba0ad62a6b31e1f69aabde725ffc0671b']
receipt=V/'beta05-lr3e6-full-launch-20261011.json';log=V/'beta05-lr3e6-full-launch-20261011.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='b482ab8261a6e793722bc22f4af32d6ba0ad62a6b31e1f69aabde725ffc0671b',plan_sha256='0ef32eae9962ffa418e4d6d9f672dc2e67aeb51fa3b43cdc4905cd4086fec3c8',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
