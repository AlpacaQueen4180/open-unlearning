from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-beta05-full-20261009-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'beta05-full-launch-20261009.json').exists() and not (V/'beta05-full-launch-20261009.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-beta05-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-09T11:43:33.248433+00:00')).total_seconds()<=600
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110):
 p=Path('/proc')/str(pid)/'stat'
 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os,time
expected={'run_spf_beta05_full_queue.py': {'size': 12335, 'sha256': '176087bc5248c04f04acf17792348a7f12c26a02bdd59fe9796af97b992a9aaa'}, 'run_spf_beta05_full.py': {'size': 6317, 'sha256': '0abffe63440a2fef5e3cbb05386431e1ab6cd54684f96bab6a3b3b423984a08e'}, 'spf_beta05_development_executor.py': {'size': 16329, 'sha256': '301acf5c3b84ff0b2756e4c1481e89cb7b5bafc63eba58aa7fc704c88db68e94'}, 'audit_spf_target.py': {'size': 5809, 'sha256': '6fc5ba538b5b6ee1a8ffd003de62abb6f65a6ddf9b1df590afe20fec7769c283'}, 'plan.private.json': {'size': 2425, 'sha256': 'd495b5b1b6ed1126fdf7938ccc79f612760bc7dffc15685e4f4f4e338709762e'}, 'bound-snapshot.private.json': {'size': 66058, 'sha256': '92174ab04e2f4935012da2418665ca1349f4ebd5320d55c1f74e1e4aae786863'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items():assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_queue',V/'run_spf_beta05_full_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan)
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'previous_guard')
guard.validate_snapshot(json.loads((V/'bound-snapshot.private.json').read_bytes()),datetime.datetime.now(datetime.timezone.utc))
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_beta05_full_queue.py'),'--source-sha256','176087bc5248c04f04acf17792348a7f12c26a02bdd59fe9796af97b992a9aaa','--plan',str(V/'plan.private.json'),'--plan-sha256','d495b5b1b6ed1126fdf7938ccc79f612760bc7dffc15685e4f4f4e338709762e','--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','92174ab04e2f4935012da2418665ca1349f4ebd5320d55c1f74e1e4aae786863']
receipt=V/'beta05-full-launch-20261009.json';log=V/'beta05-full-launch-20261009.raw.log'
with receipt.open('x',encoding='utf8') as handle:
 with log.open('xb') as stream:
  proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='92174ab04e2f4935012da2418665ca1349f4ebd5320d55c1f74e1e4aae786863',plan_sha256='d495b5b1b6ed1126fdf7938ccc79f612760bc7dffc15685e4f4f4e338709762e',new_paid_api_calls=0,existing_final_training_repeated=False,existing_evaluations_repeated=False)
  handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
