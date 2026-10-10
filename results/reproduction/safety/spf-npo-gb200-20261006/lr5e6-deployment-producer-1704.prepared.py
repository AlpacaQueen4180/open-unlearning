"""Prepare one finite new deployment after one successful hourly idle snapshot.

This producer does not execute any remote command, launch Judge, or read credentials.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import re
import tarfile
import urllib.parse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot',required=True)
parser.add_argument('--tag',required=True)
args=parser.parse_args()
assert re.fullmatch(r'[0-9]{8}-[0-9]{4}',args.tag)
root=Path('.').resolve()
work=root/'work/spf-npo-gb200-20261006'
campaign=work/'private/spf-safety-search-20261010'
preparation=campaign/'candidate-02-beta05-lr5e6-preparation-20261010-1704'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
snapshot_raw=Path(args.snapshot).read_bytes()
snapshot=json.loads(snapshot_raw)
folder=root/'scripts/reproduction/safety/gb200'
names=['run_spf_beta05_lr5e6_full_queue.py','run_spf_beta05_lr5e6_full.py','spf_beta05_lr5e6_development_executor.py']
evidence=json.loads((preparation/'evidence.private.json').read_bytes())
records_files={name:(folder/name).read_bytes() for name in names}
assert all(sha(raw)==evidence['sources'][name]['sha256'] for name,raw in records_files.items())
spec=importlib.util.spec_from_file_location('new_lower_lr_guards',folder/names[0])
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
guard_path=work/'private/development-completed-20261007-1140/validation-code/run_spf_development_recovery_guards.py'
assert sha(guard_path.read_bytes())=='64316b56536b77ee1aa11994c5c3aaf42aa01c61133178c4fcc35e5efe06a41f'
spec=importlib.util.spec_from_file_location('previous_original_idle_guard',guard_path)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
guard.validate_snapshot(snapshot,datetime.now(timezone.utc))
queue.validate_completed_predecessor(snapshot)
plan=json.loads((preparation/'candidate-plan-core.private.json').read_bytes())
queue.validate_plan(plan);queue.validate_search_decision(plan)
remote='/data/spf-beta05-lr5e6-full-20261010-r1'
previous='/data/spf-beta05-full-20261010-r2'
original='/data/spf-development-20261007-r1'
plan.update(source_sha256={remote+'/validation-code/'+name:sha(raw) for name,raw in records_files.items()},
    original_full_audit_sha256='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9',
    original_freeze_sha256='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e',
    baseline_resolved_sha256='d3aeba5342e5102124d7a12096cb4190c56c9617dabb75ed1df5a3b155bc354d',
    scope='New independent full SPF candidate from original Meta M0; beta1=.5 and lower learning rate5e-6; train then same six development stages.',
    new_paid_judge_budget_authorized=True,new_campaign_model_limit=50,
    this_new_candidate_model_judge_not_reserved_or_launched=True,
    gate='insufficient_evidence',
    limits=['Single seed; descriptive search only; preserve unverified human adjudication and paired clusters.',
        'All prior complete training/reloads/evaluations/paid successes remain unchanged.',
        'Original large-autograd/cross-platform/NPO limitations remain.',
        'No benchmark prompts or FormalHExPHI used as training data; original training-only anchor.'])
plan['source_sha256'].update({
    previous+'/validation-code/audit_spf_target.py':'6fc5ba538b5b6ee1a8ffd003de62abb6f65a6ddf9b1df590afe20fec7769c283',
    original+'/validation-code/run_spf_development_recovery_guards.py':'64316b56536b77ee1aa11994c5c3aaf42aa01c61133178c4fcc35e5efe06a41f',
    original+'/adapter-generation-api-r2/adapter-audit.json':'71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de',
    original+'/adapter-generation-api-r2/code/spf_development_generation.py':'7e729af218a2b5ffcc4584acde22fa881f8499e2415a8bdbb33dfd3cfa1b5c77'})
queue.validate_plan(plan);queue.validate_search_decision(plan)
records_files['plan.private.json']=dump(plan)
records_files['bound-snapshot.private.json']=snapshot_raw
out=campaign/('beta05-lr5e6-deployment-'+args.tag)
out.mkdir()
(out/'plan.private.json').write_bytes(records_files['plan.private.json'])
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for name,raw in records_files.items():
        item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
bundle=stream.getvalue();bundle_sha=sha(bundle)
(out/'transfer.private.tar.gz').write_bytes(bundle)
records={name:dict(size=len(raw),sha256=sha(raw)) for name,raw in records_files.items()}
commands=[]
def add(name,code):
    compile(code,name,'exec')
    (out/(name+'.as-run.py')).write_bytes(code.encode())
    payload=base64.b64encode(gzip.compress(code.encode())).decode()
    program="import base64,gzip;print('DEV_DEPLOY_BEGIN',flush=True);exec(compile(gzip.decompress(base64.b64decode('"+payload+"')),'<new_lower_lr_deploy>','exec'));print('DEV_DEPLOY_END',flush=True)"
    assert len(urllib.parse.quote(program,safe=''))<6000
    commands.append(dict(name=name,code=program))

common='''from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('REMOTE');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'beta05-lr5e6-full-launch-20261010.json').exists() and not (V/'beta05-lr5e6-full-launch-20261010.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-beta05-lr5e6-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('CAPTURED')).total_seconds() <= 600
previous=Path('/data/spf-beta05-full-20261010-r2')
assert sha((previous/'spf-beta05-full-s0-audit.json').read_bytes())=='63e72cb101b8056c938021e9518361280ed0df9f9238df04034342bf71d4110b'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
'''.replace('REMOTE',remote).replace('CAPTURED',snapshot['snapshot_utc'])
add('preflight',common+"assert not T.exists()\nprint(json.dumps(dict(status='NEW_LOWER_LR_PREFLIGHT_PASS',new_gpu_work_submitted=False)))")
transfer=remote+'/validation-code/transfer-'+args.tag+'/'+bundle_sha
encoded=base64.b64encode(bundle).decode()
pieces=[encoded[i:i+3500] for i in range(0,len(encoded),3500)]
for index,piece in enumerate(pieces):
    add('chunk-'+str(index),"from pathlib import Path\nimport json\np=Path('"+transfer+"');p.mkdir(parents=True,exist_ok=True);(p/'"+str(index)+"').open('x').write('"+piece+"');print(json.dumps(dict(chunk="+str(index)+")))")
seal=common+'''
import base64,tarfile,io
expected=RECORDS;d=Path('TRANSFER')
raw=base64.b64decode(''.join((d/str(i)).read_text() for i in range(CHUNKS)))
assert sha(raw)=='BUNDLE'
members={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    for item in archive.getmembers():
        assert item.isfile() and item.name in expected and item.name not in members and '/' not in item.name and chr(92) not in item.name
        data=archive.extractfile(item).read()
        assert len(data)==expected[item.name]['size'] and sha(data)==expected[item.name]['sha256']
        members[item.name]=data
assert set(members)==set(expected) and all(not (V/name).exists() for name in members)
for name,data in members.items(): (V/name).open('xb').write(data)
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
print(json.dumps(dict(status='NEW_LOWER_LR_SOURCES_PLAN_SNAPSHOT_SEALED',files=expected)))
'''.replace('RECORDS',repr(records)).replace('TRANSFER',transfer).replace('CHUNKS',str(len(pieces))).replace('BUNDLE',bundle_sha)
add('seal',seal)
launch=common+'''
import importlib.util,os
expected=RECORDS
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_lower_lr_queue',V/'run_spf_beta05_lr5e6_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_beta05_lr5e6_full_queue.py'),
    '--source-sha256','CONTROLLER','--plan',str(V/'plan.private.json'),'--plan-sha256','PLAN',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','SNAP']
receipt=V/'beta05-lr5e6-full-launch-20261010.json';log=V/'beta05-lr5e6-full-launch-20261010.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='SNAP',plan_sha256='PLAN',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
'''.replace('RECORDS',repr(records)).replace('CONTROLLER',records[names[0]]['sha256']).replace('PLAN',sha(records_files['plan.private.json'])).replace('SNAP',sha(snapshot_raw))
add('launch-once',launch)
(out/'commands.private.json').write_bytes(dump(commands))
(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
deployment=(work/'deploy_beta05_serialization_20261010_0117.ps1').read_text(encoding='utf8')
deployment=deployment.replace('private/beta05-serialization-deployment-20261010-0117',
    'private/spf-safety-search-20261010/beta05-lr5e6-deployment-'+args.tag)
deployment_path=work/('deploy_beta05_lr5e6_'+args.tag+'.ps1')
assert not deployment_path.exists()
deployment_path.write_bytes(deployment.encode())
(out/'deployment.as-run.ps1').write_bytes(deployment.encode())
(out/'bundle-metadata.private.json').write_bytes(dump(dict(bundle_sha256=bundle_sha,
    records=records,compressed_chunks=len(pieces),commands=len(commands),snapshot_sha256=sha(snapshot_raw),
    plan_sha256=sha(records_files['plan.private.json']),snapshot_utc=snapshot['snapshot_utc'],
    old_auditor_assets_or_models_retransferred=False,remote_commands_executed=0)))
print(json.dumps(dict(status='FINITE_NEW_LOWER_LR_DEPLOYMENT_PREPARED_NOT_EXECUTED',
    commands=len(commands),chunks=len(pieces),plan_sha256=sha(records_files['plan.private.json']),
    deployment_source=str(deployment_path))))
