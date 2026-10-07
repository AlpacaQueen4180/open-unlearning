from pathlib import Path

import json,hashlib,subprocess,datetime

T=Path('/data/spf-npo-smoke-20261007-r1');V=T/'validation-code';R=T/'runs/cache-recovery-1155'

sha=lambda b:hashlib.sha256(b).hexdigest()

assert not R.exists() and not (T/'checkpoint').exists() and not (V/'smoke-cache-recovery-launch-1155.json').exists()

assert sha((T/'runs/initial/status.json').read_bytes())=='e473acc9a792eaaab8655093a0d774d001b131a2be68b6d7a69aced736835349'

assert sha((T/'package.private.json').read_bytes())=='cbad72d7386ca83904b5a003e3bcf5b378f44d1920b452dd0bc29369bff5fc8a'

assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-07T04:11:09.926620+00:00')).total_seconds()<=600

for pid in (193,202,270,45219,47443):

 p=Path('/proc')/str(pid)/'stat'

 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'

p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True);assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()



import importlib.util,os,time

source=V/'run_spf_npo_smoke_cache_recovery_queue.py';assert sha(source.read_bytes())=='8e9ee2c3aebf346664684f8ced0597619969e2f91970ce06ce8434da87bbb75a'

assert sha((V/'verify_spf_npo_tokenizer_cache.py').read_bytes())=='3b412c8480280a64a42f889eaf081ae4692038e3914cb003b1b0eb3ee383e1a2'

old=V/'run_spf_npo_smoke_api_r2_queue.py';assert sha(old.read_bytes())=='eedc4dfd16269db9fa84ef681ae289f86392c85c45ff7d0cedfda24bd6d17735'

spec=importlib.util.spec_from_file_location('old_guards',old);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

snapshot=V/'r4-snapshot-cache-recovery-1210.private.json';s=g.bound_json(snapshot,'a01217963b58d2bd4a112bac38130b0f47d3d26127ab08c3d4cde12824898635');g.validate_snapshot(s,datetime.datetime.now(datetime.timezone.utc))

assert s['npo_smoke']['status']==json.loads((T/'runs/initial/status.json').read_bytes())

dev=Path('/data/spf-development-20261007-r1');a=g.bound_json(dev/'cuda-development-r2/cuda-acceptance.private.json',g.DEVELOPMENT_ACCEPTANCE_SHA);adapter=g.bound_json(dev/'adapter-generation-api-r2/adapter-audit.json',g.ADAPTER_SHA);g.development_accepted(a,adapter)

command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(source),'--controller-sha256','8e9ee2c3aebf346664684f8ced0597619969e2f91970ce06ce8434da87bbb75a','--snapshot',str(snapshot),'--snapshot-sha256','a01217963b58d2bd4a112bac38130b0f47d3d26127ab08c3d4cde12824898635','--run-id','cache-recovery-1155']

receipt=V/'smoke-cache-recovery-launch-1155.json';log=V/'smoke-cache-recovery-launch-1155.raw.log'

with receipt.open('x',encoding='utf8') as handle:

 with log.open('xb') as stream:

  p=subprocess.Popen(command,cwd=dev/'repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)

  result=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=p.pid,command=command,cwd=str(dev/'repo'),snapshot_sha256='a01217963b58d2bd4a112bac38130b0f47d3d26127ab08c3d4cde12824898635',started_at=time.time(),log=str(log),original_package_changed=False,hydra_composition_repeated=False,pilot_frozen=False,actual_cache_diagnostic_result_checked_by_launcher=False)

  handle.write(json.dumps(result,indent=2)+chr(10));handle.flush()

print(json.dumps(result))

