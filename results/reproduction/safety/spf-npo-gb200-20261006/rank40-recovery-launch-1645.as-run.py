from pathlib import Path
import datetime,hashlib,json,os,subprocess
T=Path('/data/spf-hybrid01-rank40-full-20261012-r1');V=T/'validation-code';J=T/'runs/cpu-fixture-recovery-1539';sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-12T08:46:26.826505+00:00')).total_seconds()<=600
assert sha((T/'status.json').read_bytes())=='2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25'
assert not J.exists() and not (V/'hybrid01-rank40-cpu-recovery-launch-1539.json').exists() and not (V/'hybrid01-rank40-cpu-recovery-launch-1539.raw.log').exists()
assert all(not (T/name).exists() for name in ('engineering-spf-hybrid01-rank40-s0','spf-hybrid01-rank40-full-s0','candidate-development','optimizer-binding.private.json','hybrid-engineering-proof.private.json','hybrid-full-acceptance.private.json'))
for pid in (19460,19463,16733,16736,16947,17483,17763,17766,17834,18362,18642,18645,18802,18939,19076,19213,19348):
    path=Path('/proc')/str(pid)/'stat'
    assert not path.exists() or path.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert r.returncode==0 and not r.stdout.strip() and not r.stderr.strip()
vfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3

import importlib.util
expected={'run_spf_hybrid01_rank40_cpu_recovery_queue.py': {'size': 25049, 'sha256': 'dfb0f633424201e63abf98ed107f7f1c59ab41d8528e42ff34f725882b1f9b9a'}, 'run_spf_hybrid01_rank40_cpu_recovery.py': {'size': 10303, 'sha256': '4cc6b969be7a3fb52714a5e20c4c54c53b2f4b6f843d8549738d06685c05ca01'}, 'verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py': {'size': 8760, 'sha256': 'a998a42a44109da78c1789478881f1da8fc5753b53788844141b4bc5e65b447b'}, 'rank40-cpu-oracle-proof-1539.private.json': {'size': 1390, 'sha256': '5397aabbfcaa4d70b1b3f51004721d6d6f389212301ced815472c890bbfb6c4d'}, 'plan-cpu-fixture-recovery-1539.private.json': {'size': 5708, 'sha256': '49ce276792f785c05e102f0c82355ea20e4f7806d86459448ff3e3c9359cee6b'}, 'bound-snapshot-cpu-recovery-1539.private.json': {'size': 24767, 'sha256': '49a1538c4a6e0b058e116650796e57a83c7137b91e2f29dfbef6585cfe338c20'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan-cpu-fixture-recovery-1539.private.json').read_bytes())
for path,digest in plan['source_sha256'].items():assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_recovery_queue',V/'run_spf_hybrid01_rank40_cpu_recovery_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime();m.validate_failed_runtime(plan)
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot-cpu-recovery-1539.private.json').read_bytes());guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=[m.PYTHON,str(V/'run_spf_hybrid01_rank40_cpu_recovery_queue.py'),'--source-sha256','dfb0f633424201e63abf98ed107f7f1c59ab41d8528e42ff34f725882b1f9b9a','--plan',str(V/'plan-cpu-fixture-recovery-1539.private.json'),'--plan-sha256','49ce276792f785c05e102f0c82355ea20e4f7806d86459448ff3e3c9359cee6b','--snapshot',str(V/'bound-snapshot-cpu-recovery-1539.private.json'),'--snapshot-sha256','49a1538c4a6e0b058e116650796e57a83c7137b91e2f29dfbef6585cfe338c20']
receipt=V/'hybrid01-rank40-cpu-recovery-launch-1539.json';log=V/'hybrid01-rank40-cpu-recovery-launch-1539.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='49a1538c4a6e0b058e116650796e57a83c7137b91e2f29dfbef6585cfe338c20',plan_sha256='49ce276792f785c05e102f0c82355ea20e4f7806d86459448ff3e3c9359cee6b',original_failed_queue=19460,new_paid_api_calls=0,completed_cpu_fixture_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
