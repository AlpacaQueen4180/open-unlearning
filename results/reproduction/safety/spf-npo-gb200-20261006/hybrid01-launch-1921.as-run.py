from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'hybrid01-lr5e6-full-launch-20261011.json').exists() and not (V/'hybrid01-lr5e6-full-launch-20261011.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-hybrid01-lr5e6-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-11T11:35:21.715202+00:00')).total_seconds() <= 600
previous=Path('/data/spf-beta05-lr3e6-full-20261011-r1')
assert sha((previous/'spf-beta05-lr3e6-full-s0-audit.json').read_bytes())=='839156fd5c6d0e28847991188a95061d513727cf012ec912465697ca9450c181'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140,10758,10761,10829,11297,11577,12367,12370,12438):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
import os
vfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os
expected={'run_spf_hybrid01_lr5e6_full_queue.py': {'size': 18327, 'sha256': '7b5f82e1f285233288845c7dcdaf15a6612a53633764ec8f8360dbde28315fc9'}, 'run_spf_hybrid01_lr5e6_full.py': {'size': 8713, 'sha256': 'a3402bfc2bc0d31173e71e3f9240802f6e5cbdc62528e62644507e7676f4a2c8'}, 'spf_hybrid01_lr5e6_development_executor.py': {'size': 16337, 'sha256': '27ed16eda408ede0dd5401a29fbf7bc861fc2d4e57c3fa3fc4b7e0010c290da3'}, 'spf_anchor_hybrid.py': {'size': 4864, 'sha256': 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc'}, 'verify_spf_anchor_hybrid.py': {'size': 8518, 'sha256': 'd57d8ff1309c15a77beeb72ecbe0fe33a215c9b13fc8fb6f86c57ce1fffb8a73'}, 'plan.private.json': {'size': 4241, 'sha256': '4f60bd23f61c70dfcf06e909b347c07a27420ec60cecc8e9c71dc3b35a5580b5'}, 'bound-snapshot.private.json': {'size': 15191, 'sha256': '752384e32ec448c38c4dc87e7e65242bc979976ff383d75553467c17759275b4'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_lower_lr_queue',V/'run_spf_hybrid01_lr5e6_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_hybrid01_lr5e6_full_queue.py'),
    '--source-sha256','7b5f82e1f285233288845c7dcdaf15a6612a53633764ec8f8360dbde28315fc9','--plan',str(V/'plan.private.json'),'--plan-sha256','4f60bd23f61c70dfcf06e909b347c07a27420ec60cecc8e9c71dc3b35a5580b5',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','752384e32ec448c38c4dc87e7e65242bc979976ff383d75553467c17759275b4']
receipt=V/'hybrid01-lr5e6-full-launch-20261011.json';log=V/'hybrid01-lr5e6-full-launch-20261011.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='752384e32ec448c38c4dc87e7e65242bc979976ff383d75553467c17759275b4',plan_sha256='4f60bd23f61c70dfcf06e909b347c07a27420ec60cecc8e9c71dc3b35a5580b5',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
