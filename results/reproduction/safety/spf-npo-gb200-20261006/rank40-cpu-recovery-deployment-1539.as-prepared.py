"""Prepare a finite, exclusive recovery deployment only after a fresh eligible hourly snapshot.

No remote command is executed by this producer. Completed CPU proof and existing
executor/helper/auditor/assets are reused. No private Judge key is read.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,gzip,hashlib,importlib.util,io,json,re,sys,tarfile,urllib.parse
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
assert re.fullmatch(r'[0-9]{8}-[0-9]{4}',a.tag)
root=Path('.').resolve();work=root/'work/spf-npo-gb200-20261006';campaign=work/'private/spf-safety-search-20261010'
prepared=campaign/'rank40-cpu-recovery-preparation-20261012-1539';folder=root/'scripts/reproduction/safety/gb200'
sha=lambda raw:hashlib.sha256(raw).hexdigest();dump=lambda v:(json.dumps(v,indent=2)+'\n').encode()
snapshot_raw=Path(a.snapshot).read_bytes();snapshot=json.loads(snapshot_raw)
assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(snapshot['snapshot_utc'])).total_seconds()<=600
evidence=json.loads((prepared/'evidence.private.json').read_bytes())
names=['run_spf_hybrid01_rank40_cpu_recovery_queue.py','run_spf_hybrid01_rank40_cpu_recovery.py','verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py']
files={name:(folder/name).read_bytes() for name in names}
assert all(sha(raw)==evidence['sources'][name]['sha256'] for name,raw in files.items())
spec=importlib.util.spec_from_file_location('recovery_guard',folder/names[0]);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
guardpath=work/'private/development-completed-20261007-1140/validation-code/run_spf_development_recovery_guards.py'
assert sha(guardpath.read_bytes())=='64316b56536b77ee1aa11994c5c3aaf42aa01c61133178c4fcc35e5efe06a41f'
spec=importlib.util.spec_from_file_location('original_guard',guardpath);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
guard.validate_snapshot(snapshot,datetime.now(timezone.utc));q.validate_completed_predecessor(snapshot)
plan=json.loads((prepared/'plan-cpu-fixture-recovery-1539.private.json').read_bytes());q.validate_plan(plan);q.validate_search_decision(plan)
cpu=prepared/'rank40-cpu-oracle-proof-1539.private.json';assert sha(cpu.read_bytes())==plan['recovery_contract']['reused_cpu_proof_sha256'];q.validate_cpu_proof(json.loads(cpu.read_bytes()))
files.update({'rank40-cpu-oracle-proof-1539.private.json':cpu.read_bytes(),'plan-cpu-fixture-recovery-1539.private.json':dump(plan),'bound-snapshot-cpu-recovery-1539.private.json':snapshot_raw})
out=campaign/('rank40-cpu-recovery-deployment-'+a.tag);out.mkdir()
for name,raw in files.items():(out/name).write_bytes(raw)
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for name,raw in files.items():
        member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
bundle=stream.getvalue();bundle_sha=sha(bundle);(out/'transfer.private.tar.gz').write_bytes(bundle)
records={name:dict(size=len(raw),sha256=sha(raw)) for name,raw in files.items()};commands=[]
def add(name,code):
    compile(code,name,'exec');(out/(name+'.as-run.py')).write_bytes(code.encode())
    payload=base64.b64encode(gzip.compress(code.encode())).decode()
    program="import base64,gzip;print('DEV_DEPLOY_BEGIN',flush=True);exec(compile(gzip.decompress(base64.b64decode("+repr(payload)+")),'<rank40_cpu_recovery_deploy>','exec'));print('DEV_DEPLOY_END',flush=True)"
    assert len(urllib.parse.quote(program,safe=''))<6000
    commands.append(dict(name=name,code=program))
common='''from pathlib import Path
import datetime,hashlib,json,os,subprocess
T=Path('/data/spf-hybrid01-rank40-full-20261012-r1');V=T/'validation-code';J=T/'runs/cpu-fixture-recovery-1539';sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(CAPTURED)).total_seconds()<=600
assert sha((T/'status.json').read_bytes())=='2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25'
assert not J.exists() and not (V/'hybrid01-rank40-cpu-recovery-launch-1539.json').exists() and not (V/'hybrid01-rank40-cpu-recovery-launch-1539.raw.log').exists()
assert all(not (T/name).exists() for name in ('engineering-spf-hybrid01-rank40-s0','spf-hybrid01-rank40-full-s0','candidate-development','optimizer-binding.private.json','hybrid-engineering-proof.private.json','hybrid-full-acceptance.private.json'))
for pid in (19460,19463,16733,16736,16947,17483,17763,17766,17834,18362,18642,18645,18802,18939,19076,19213,19348):
    path=Path('/proc')/str(pid)/'stat'
    assert not path.exists() or path.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert r.returncode==0 and not r.stdout.strip() and not r.stderr.strip()
vfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3
'''.replace('CAPTURED',repr(snapshot['snapshot_utc']))
add('preflight',common+"assert all(not (V/name).exists() for name in "+repr(list(files))+ ")\nprint(json.dumps(dict(status='RECOVERY_PREFLIGHT_PASS',new_gpu_work_submitted=False)))")
transfer=(q.ROOT/'validation-code'/('cpu-recovery-transfer-'+a.tag)/bundle_sha).as_posix()
encoded=base64.b64encode(bundle).decode();pieces=[encoded[i:i+3500] for i in range(0,len(encoded),3500)]
for i,piece in enumerate(pieces):
    add('chunk-'+str(i),"from pathlib import Path\nimport json\np=Path("+repr(transfer)+");p.mkdir(parents=True,exist_ok=True);(p/"+repr(str(i))+").open('x').write("+repr(piece)+");print(json.dumps(dict(chunk="+str(i)+")))")
seal=common+'''
import base64,io,tarfile
expected=RECORDS;d=Path(TRANSFER);raw=base64.b64decode(''.join((d/str(i)).read_text() for i in range(CHUNKS)))
assert sha(raw)==BUNDLE
members={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
    for item in archive.getmembers():
        assert item.isfile() and item.name in expected and item.name not in members and '/' not in item.name and chr(92) not in item.name
        data=archive.extractfile(item).read();assert len(data)==expected[item.name]['size'] and sha(data)==expected[item.name]['sha256'];members[item.name]=data
assert set(members)==set(expected) and all(not (V/name).exists() for name in members)
for name,data in members.items():(V/name).open('xb').write(data)
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
print(json.dumps(dict(status='RECOVERY_SOURCES_PLAN_PROOF_SNAPSHOT_SEALED',files=expected)))
'''.replace('RECORDS',repr(records)).replace('TRANSFER',repr(transfer)).replace('CHUNKS',str(len(pieces))).replace('BUNDLE',repr(bundle_sha))
add('seal',seal)
launch=common+'''
import importlib.util
expected=RECORDS
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan-cpu-fixture-recovery-1539.private.json').read_bytes())
for path,digest in plan['source_sha256'].items():assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_recovery_queue',V/'run_spf_hybrid01_rank40_cpu_recovery_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime();m.validate_failed_runtime(plan)
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot-cpu-recovery-1539.private.json').read_bytes());guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=[m.PYTHON,str(V/'run_spf_hybrid01_rank40_cpu_recovery_queue.py'),'--source-sha256',CONTROLLER,'--plan',str(V/'plan-cpu-fixture-recovery-1539.private.json'),'--plan-sha256',PLAN,'--snapshot',str(V/'bound-snapshot-cpu-recovery-1539.private.json'),'--snapshot-sha256',SNAP]
receipt=V/'hybrid01-rank40-cpu-recovery-launch-1539.json';log=V/'hybrid01-rank40-cpu-recovery-launch-1539.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256=SNAP,plan_sha256=PLAN,original_failed_queue=19460,new_paid_api_calls=0,completed_cpu_fixture_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
'''.replace('RECORDS',repr(records)).replace('CONTROLLER',repr(sha(files[names[0]]))).replace('PLAN',repr(sha(files['plan-cpu-fixture-recovery-1539.private.json']))).replace('SNAP',repr(sha(snapshot_raw)))
add('launch-once',launch)
(out/'commands.private.json').write_bytes(dump(commands));(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
deployment=(work/'deploy_beta05_serialization_20261010_0117.ps1').read_text(encoding='utf8').replace('private/beta05-serialization-deployment-20261010-0117','private/spf-safety-search-20261010/rank40-cpu-recovery-deployment-'+a.tag)
deployment_path=work/('deploy_rank40_cpu_recovery_'+a.tag+'.ps1')
with deployment_path.open('xb') as f:f.write(deployment.encode())
(out/'deployment.as-run.ps1').write_bytes(deployment.encode())
(out/'bundle-metadata.private.json').write_bytes(dump(dict(bundle_sha256=bundle_sha,records=records,compressed_chunks=len(pieces),commands=len(commands),snapshot_utc=snapshot['snapshot_utc'],snapshot_sha256=sha(snapshot_raw),plan_sha256=sha(files['plan-cpu-fixture-recovery-1539.private.json']),old_sources_assets_models_or_auditor_retransferred=False,completed_cpu_fixture_repeated=False,remote_commands_executed=0)))
print(json.dumps(dict(status='FINITE_CPU_RECOVERY_DEPLOYMENT_PREPARED_NOT_EXECUTED',commands=len(commands),chunks=len(pieces),files=len(files),deployment_source=str(deployment_path))))
