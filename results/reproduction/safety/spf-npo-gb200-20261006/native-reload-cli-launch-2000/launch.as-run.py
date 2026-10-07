from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-npo-smoke-20261007-r1');V=T/'validation-code';R=T/'runs/reload-native-cli-1936'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not R.exists() and (T/'checkpoint').is_dir() and not (T/'checkpoint').is_symlink()
assert not (V/'smoke-native-reload-cli-launch-1936.json').exists() and not (V/'smoke-native-reload-cli-launch-1936.raw.log').exists()
assert sha((T/'runs/import-recovery-1257/status.json').read_bytes())=='0c361bd91dc9819f80be3013bf832262bb3bd93149e2c9d7dccaa12a603c9565'
assert sha((T/'runs/reload-native-1312/status.json').read_bytes())=='2f247e8e538c6fd4a7b412317fb599112c76ff126b974070f960f7ec4fbac828'
assert not (T/'runs/reload-native-1312/reload-audit.private.json').exists() and not (T/'runs/reload-native-1312/reload-comparison.private.pt').exists()
assert sha((T/'package.private.json').read_bytes())=='cbad72d7386ca83904b5a003e3bcf5b378f44d1920b452dd0bc29369bff5fc8a'
assert sha((T/'checkpoint/spf-npo-runtime.private.json').read_bytes())=='bf2ea667edf6772c86e1033aaedff3c71f7d8ee22a9279747c8fc053f787f88b'
assert sha((T/'checkpoint/spf-npo-reload-capture.json').read_bytes())=='a7eea385d0f9fd455ccbd911952f430ad2aff8529054a08d3782e06bd94fd045'
assert sha((T/'checkpoint/spf-npo-fresh-reload-audit.json').read_bytes())=='6e23113e71715a1e9dc75245ebf7a9d62af1067e8a21ec02454456e778ed1a02'
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-07T12:01:31.591656+00:00')).total_seconds()<=600
for pid in (193,202,270,824,830,961,1029,1977,1983,2252,2320,2953,2961,3029,3501,45219,47443,3889,3896):
 p=Path('/proc')/str(pid)/'stat'
 assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()

import importlib.util,os,time
source=V/'run_spf_npo_smoke_reload_cli_r2_queue.py';assert sha(source.read_bytes())=='2fb5f9c4287710589620d7e7fe0268c739c08b2036add9d39251be71706df0da'
assert sha((V/'verify_spf_npo_smoke_native_reload_cli_r2.py').read_bytes())=='289c718a2f9064b1a7d1542d0ea261c2befa765b5b85a241ec044ed4e1d4fe27'
old=V/'run_spf_npo_smoke_api_r2_queue.py';assert sha(old.read_bytes())=='eedc4dfd16269db9fa84ef681ae289f86392c85c45ff7d0cedfda24bd6d17735'
spec=importlib.util.spec_from_file_location('old_guards',old);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
snapshot=V/'r4-snapshot-native-reload-cli-2000.private.json';s=g.bound_json(snapshot,'0004b320493557cdf23bbc6c031b091cff6038de30e860b6d2601481900301e9');g.validate_snapshot(s,datetime.datetime.now(datetime.timezone.utc))
assert s['npo_smoke']['import_recovery']['status']==json.loads((T/'runs/import-recovery-1257/status.json').read_bytes())
dev=Path('/data/spf-development-20261007-r1');a=g.bound_json(dev/'cuda-development-r2/cuda-acceptance.private.json',g.DEVELOPMENT_ACCEPTANCE_SHA);adapter=g.bound_json(dev/'adapter-generation-api-r2/adapter-audit.json',g.ADAPTER_SHA);g.development_accepted(a,adapter)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(source),'--controller-sha256','2fb5f9c4287710589620d7e7fe0268c739c08b2036add9d39251be71706df0da','--snapshot',str(snapshot),'--snapshot-sha256','0004b320493557cdf23bbc6c031b091cff6038de30e860b6d2601481900301e9','--run-id','reload-native-cli-1936']
receipt=V/'smoke-native-reload-cli-launch-1936.json';log=V/'smoke-native-reload-cli-launch-1936.raw.log'
with receipt.open('x',encoding='utf8') as handle:
 with log.open('xb') as stream:
  p=subprocess.Popen(command,cwd=dev/'repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=p.pid,command=command,cwd=str(dev/'repo'),snapshot_sha256='0004b320493557cdf23bbc6c031b091cff6038de30e860b6d2601481900301e9',started_at=time.time(),log=str(log),original_checkpoint_modified=False,training_repeated=False,completed_diagnostics_repeated=False,pilot_frozen=False)
  handle.write(json.dumps(result,indent=2)+chr(10));handle.flush()
print(json.dumps(result))
