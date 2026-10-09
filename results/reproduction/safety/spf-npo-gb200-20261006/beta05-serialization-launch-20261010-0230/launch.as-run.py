from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-beta05-full-20261010-r2');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'beta05-full-launch-20261009.json').exists() and not (V/'beta05-full-launch-20261009.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-beta05-full-s0','candidate-development','optimizer-binding.private.json'))
old=Path('/data/spf-beta05-full-20261009-r1')
assert sha((old/'status.json').read_bytes())=='34da4de42f22765b36d46f873726d3f1f3453216407d1aa8adb8437a4282e0da'
assert sha((old/'train.raw.log').read_bytes())=='9a37e105ef111e565fea1bd13ba800c76bcfe93d58fadb9ae9015d2404e967f8'
assert all(not (old/name).exists() for name in ('spf-beta05-full-s0','optimizer-binding.private.json','spf-beta05-full-s0-audit.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-09T18:27:13.801444+00:00')).total_seconds()<=600
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554):
 p=Path('/proc')/str(pid)/'stat'
 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os,time
expected={'run_spf_beta05_full_serialization_r2_queue.py': {'size': 14400, 'sha256': '9d734587cb49590f275b906405117640e6e1a472f43ac742a9a196dbe787c937'}, 'run_spf_beta05_full_serialization_r2.py': {'size': 6456, 'sha256': '33127eee907d343e18daf29fd064f8551048544ef03f7943dadb8339fd219e32'}, 'spf_beta05_serialization_r2_development_executor.py': {'size': 16329, 'sha256': '76c5fd9c0a6228643d3c9a11621cfa90db52fed65efde1f403d86a7f23c14191'}, 'audit_spf_target.py': {'size': 5809, 'sha256': '6fc5ba538b5b6ee1a8ffd003de62abb6f65a6ddf9b1df590afe20fec7769c283'}, 'plan.private.json': {'size': 2504, 'sha256': 'e5782f588434ba8b1ecb799882a1f6cef3a1c8347b48b47497b2c8304ebc7ecb'}, 'bound-snapshot.private.json': {'size': 69300, 'sha256': 'f75d42fc08f49d35a5ff4035358366369c0c74b66039663ee121784fc5c20cd7'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
assert sha((V/'bound-snapshot-20261010-0226.private.json').read_bytes())=='a49f89401a7bd89ddd772d5a135f06d18f7f3089784b9f0cfad14f2cd1aca512'
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items():assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_queue',V/'run_spf_beta05_full_serialization_r2_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan)
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'previous_guard')
guard.validate_snapshot(json.loads((V/'bound-snapshot-20261010-0226.private.json').read_bytes()),datetime.datetime.now(datetime.timezone.utc))
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_beta05_full_serialization_r2_queue.py'),'--source-sha256','9d734587cb49590f275b906405117640e6e1a472f43ac742a9a196dbe787c937','--plan',str(V/'plan.private.json'),'--plan-sha256','e5782f588434ba8b1ecb799882a1f6cef3a1c8347b48b47497b2c8304ebc7ecb','--snapshot',str(V/'bound-snapshot-20261010-0226.private.json'),'--snapshot-sha256','a49f89401a7bd89ddd772d5a135f06d18f7f3089784b9f0cfad14f2cd1aca512']
receipt=V/'beta05-full-launch-20261009.json';log=V/'beta05-full-launch-20261009.raw.log'
with receipt.open('x',encoding='utf8') as handle:
 with log.open('xb') as stream:
  proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='a49f89401a7bd89ddd772d5a135f06d18f7f3089784b9f0cfad14f2cd1aca512',plan_sha256='e5782f588434ba8b1ecb799882a1f6cef3a1c8347b48b47497b2c8304ebc7ecb',new_paid_api_calls=0,existing_final_training_repeated=False,existing_evaluations_repeated=False)
  handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
