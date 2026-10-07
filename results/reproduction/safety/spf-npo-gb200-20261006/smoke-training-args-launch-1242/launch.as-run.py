from pathlib import Path

import json,hashlib,subprocess,datetime

T=Path('/data/spf-npo-smoke-20261007-r1');V=T/'validation-code';R=T/'runs/training-args-recovery-1225'

sha=lambda b:hashlib.sha256(b).hexdigest()

assert not R.exists() and not (T/'checkpoint').exists() and not (V/'smoke-training-args-recovery-launch-1225.json').exists()

assert sha((T/'runs/cache-recovery-1155/status.json').read_bytes())=='f214eb901e4fbfdc7b638ca586463ded21680e41e3e8493d865dae3e07f5ba72'

assert sha((T/'package.private.json').read_bytes())=='cbad72d7386ca83904b5a003e3bcf5b378f44d1920b452dd0bc29369bff5fc8a'

assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-07T04:43:07.386981+00:00')).total_seconds()<=600

for pid in (193,202,270,824,830,961,1029,45219,47443):

 p=Path('/proc')/str(pid)/'stat'

 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'

p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True);assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()



import importlib.util,os,time

source=V/'run_spf_npo_smoke_training_args_recovery_queue.py';assert sha(source.read_bytes())=='341f2906434f9a7f4203cc99a672482e84d060487c9b1866ad78e1f062a84f97'

assert sha((V/'verify_spf_npo_training_args_api.py').read_bytes())=='c840357a27581a11aab3d43af2f00086c213bb460d53dbfc07edee34f250c841'
assert sha((V/'spf_npo_training_args_api.py').read_bytes())=='430feb8b97a3fb0dea5a9f293b50394300e0a947d5a0e2c0875f5c716e2fae52'

old=V/'run_spf_npo_smoke_api_r2_queue.py';assert sha(old.read_bytes())=='eedc4dfd16269db9fa84ef681ae289f86392c85c45ff7d0cedfda24bd6d17735'

spec=importlib.util.spec_from_file_location('old_guards',old);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

snapshot=V/'r4-snapshot-training-args-recovery-1242.private.json';s=g.bound_json(snapshot,'b991824b79e824d3dd78a74ab445fdd11d7521bf7e80bb7876bc52f726e2ead4');g.validate_snapshot(s,datetime.datetime.now(datetime.timezone.utc))

assert s['npo_smoke']['cache_recovery']['status']==json.loads((T/'runs/cache-recovery-1155/status.json').read_bytes())

dev=Path('/data/spf-development-20261007-r1');a=g.bound_json(dev/'cuda-development-r2/cuda-acceptance.private.json',g.DEVELOPMENT_ACCEPTANCE_SHA);adapter=g.bound_json(dev/'adapter-generation-api-r2/adapter-audit.json',g.ADAPTER_SHA);g.development_accepted(a,adapter)

command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(source),'--controller-sha256','341f2906434f9a7f4203cc99a672482e84d060487c9b1866ad78e1f062a84f97','--snapshot',str(snapshot),'--snapshot-sha256','b991824b79e824d3dd78a74ab445fdd11d7521bf7e80bb7876bc52f726e2ead4','--run-id','training-args-recovery-1225']

receipt=V/'smoke-training-args-recovery-launch-1225.json';log=V/'smoke-training-args-recovery-launch-1225.raw.log'

with receipt.open('x',encoding='utf8') as handle:

 with log.open('xb') as stream:

  p=subprocess.Popen(command,cwd=dev/'repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)

  result=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=p.pid,command=command,cwd=str(dev/'repo'),snapshot_sha256='b991824b79e824d3dd78a74ab445fdd11d7521bf7e80bb7876bc52f726e2ead4',started_at=time.time(),log=str(log),original_package_changed=False,hydra_composition_repeated=False,pilot_frozen=False,cache_diagnostic_repeated=False,actual_training_args_diagnostic_result_checked_by_launcher=False)

  handle.write(json.dumps(result,indent=2)+chr(10));handle.flush()

print(json.dumps(result))

