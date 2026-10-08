from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-checkpoint-beta-20261008-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'diagnostic-launch-20261008.json').exists() and not (V/'diagnostic-launch-20261008.raw.log').exists()
assert all(not (T/name).exists() for name in ('beta-09','beta-05','checkpoint-157','checkpoint-313','checkpoint-469'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-08T12:42:40.352013+00:00')).total_seconds()<=600
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652):
 p=Path('/proc')/str(pid)/'stat'
 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os,time
expected={'run_spf_checkpoint_beta_queue.py': {'size': 11936, 'sha256': 'ff9644e56ae589b234d18fa221ffe50f8ce41dc843c643753e99cfe9ccba43da'}, 'spf_checkpoint_executor.py': {'size': 16366, 'sha256': '95a28d74b00ad51423da541656007ff2e9d9d85f5d31039a9f5286325ae92609'}, 'spf_beta_geometry_diagnostic.py': {'size': 14724, 'sha256': 'c2b8576f6b4a558a767a95f2a2c8652783761f6e2afef68352d9f59b8150cf0d'}, 'plan.private.json': {'size': 4190, 'sha256': 'd047168aefa7e8ceb1b98de8299fbc2e79b14da1a79812be9f6218d24cca26b6'}, 'bound-snapshot.private.json': {'size': 66460, 'sha256': '8566cd24b26f815ea1d7d5cf08f93e133830b728f91c652ac18e41fd765a73da'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items():assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_queue',V/'run_spf_checkpoint_beta_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan)
for case in plan['cases']:m.metadata_check(case)
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'previous_guard')
guard.validate_snapshot(json.loads((V/'bound-snapshot.private.json').read_bytes()),datetime.datetime.now(datetime.timezone.utc))
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_checkpoint_beta_queue.py'),'--source-sha256','ff9644e56ae589b234d18fa221ffe50f8ce41dc843c643753e99cfe9ccba43da','--plan',str(V/'plan.private.json'),'--plan-sha256','d047168aefa7e8ceb1b98de8299fbc2e79b14da1a79812be9f6218d24cca26b6','--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','8566cd24b26f815ea1d7d5cf08f93e133830b728f91c652ac18e41fd765a73da']
receipt=V/'diagnostic-launch-20261008.json';log=V/'diagnostic-launch-20261008.raw.log'
with receipt.open('x',encoding='utf8') as handle:
 with log.open('xb') as stream:
  proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='8566cd24b26f815ea1d7d5cf08f93e133830b728f91c652ac18e41fd765a73da',plan_sha256='d047168aefa7e8ceb1b98de8299fbc2e79b14da1a79812be9f6218d24cca26b6',new_paid_api_calls=0,existing_final_training_repeated=False,existing_evaluations_repeated=False)
  handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
