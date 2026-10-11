from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'hybrid03-lr5e6-full-launch-20261012.json').exists() and not (V/'hybrid03-lr5e6-full-launch-20261012.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-hybrid03-lr5e6-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-11T19:34:57.174566+00:00')).total_seconds() <= 600
previous=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')
assert sha((previous/'spf-hybrid01-lr5e6-full-s0-audit.json').read_bytes())=='6ce8057e3d60ea705794c6c308b6a0563555caad05362fe7864a7db1d75c0931'
assert sha((previous/'hybrid-full-acceptance.private.json').read_bytes())=='5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140,10758,10761,10829,11297,11577,12367,12370,12438,14033,15066,15134,15635,15915,15918):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
import os
vfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os
expected={'run_spf_hybrid03_lr5e6_full_queue.py': {'size': 18854, 'sha256': '65305a2ba389818797b708a48db75d2dd1fa7a56d64b87de8e1d4d06069bdb47'}, 'run_spf_hybrid03_lr5e6_full.py': {'size': 8753, 'sha256': 'e30e9e059d96cc841030df47b9f16b94450388068857a2084819f4593ad4aa37'}, 'spf_hybrid03_lr5e6_development_executor.py': {'size': 16337, 'sha256': '2bd673f0bca4d5e0ca85f4bb8e08a7003e4350d8fd8cef95c8a652e51119ce62'}, 'spf_anchor_hybrid.py': {'size': 4864, 'sha256': 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc'}, 'verify_spf_anchor_hybrid03.py': {'size': 8546, 'sha256': '44da15a0287817ab4f20759890eee0f07fc5100d5258e2f301726c766bf2edd4'}, 'plan.private.json': {'size': 4243, 'sha256': '53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997'}, 'bound-snapshot.private.json': {'size': 20321, 'sha256': 'c0cf4fc7c1b74e56d2b472adff54ef1f75b666f4009092f9ffd94447aaac5e1e'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_lower_lr_queue',V/'run_spf_hybrid03_lr5e6_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_hybrid03_lr5e6_full_queue.py'),
    '--source-sha256','65305a2ba389818797b708a48db75d2dd1fa7a56d64b87de8e1d4d06069bdb47','--plan',str(V/'plan.private.json'),'--plan-sha256','53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','c0cf4fc7c1b74e56d2b472adff54ef1f75b666f4009092f9ffd94447aaac5e1e']
receipt=V/'hybrid03-lr5e6-full-launch-20261012.json';log=V/'hybrid03-lr5e6-full-launch-20261012.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='c0cf4fc7c1b74e56d2b472adff54ef1f75b666f4009092f9ffd94447aaac5e1e',plan_sha256='53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
