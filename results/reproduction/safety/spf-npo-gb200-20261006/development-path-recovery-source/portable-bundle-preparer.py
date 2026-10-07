import pathlib,json,hashlib,base64,gzip,tarfile,io,urllib.parse,datetime
root=pathlib.Path(__file__).resolve().parents[2]
private=root/'work/spf-npo-gb200-20261006/private'
package=private/'development-api-recovery-20261007-0809-r3'
prepared=json.loads((package/'prepared.private.json').read_bytes())
snapshot=(private/'heartbeat-20261007-0809.json').read_bytes()
snap=json.loads(snapshot);snapsha=hashlib.sha256(snapshot).hexdigest()
assert snapsha=='50057139817d72bb239d9fd744972d7b198f3c6d069cfabe281493de9024b3c0'
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(snap['snapshot_utc'])).total_seconds() <= 600
assert snap['development_deployment']['status']['status']=='FAILED' and snap['development_deployment']['status']['completed']==[]
out=private/'development-recovery-deployment-20261007-0809';out.mkdir(exist_ok=False)
files={p.relative_to(package).as_posix():p.read_bytes() for p in package.rglob('*') if p.is_file() and p.name!='prepared.private.json'}
for name in ['run_spf_development_recovery_queue.py','run_spf_development_recovery_guards.py']:
    files['validation-code/'+name]=files.pop(name)
files['validation-code/r4-snapshot-0809.private.json']=snapshot
manifest={name:{'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in files.items()}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
    for name,raw in sorted(files.items()):
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
raw=buf.getvalue();bundle_sha=hashlib.sha256(raw).hexdigest();(out/'recovery-bundle.tar.gz').write_bytes(raw)
(out/'manifest.private.json').write_text(json.dumps(manifest,indent=2)+'\n')
commands=[]
def add(name,source):
    compile(source,name,'exec');payload=base64.b64encode(gzip.compress(source.encode())).decode()
    program="import base64,gzip;print('DEV_DEPLOY_BEGIN',flush=True);exec(compile(gzip.decompress(base64.b64decode('"+payload+"')),'<explicit_development_recovery>','exec'));print('DEV_DEPLOY_END',flush=True)"
    assert len(urllib.parse.quote(program,safe=''))<6000
    (out/(name+'.as-run.py')).write_text(source)
    commands.append({'name':name,'code':program})
remote='/data/spf-development-20261007-r1'
preflight="""import pathlib,hashlib,json,datetime,subprocess
t=pathlib.Path('/data/spf-development-20261007-r1')
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('CAPTURED')).total_seconds() <= 600
assert hashlib.sha256((t/'cuda-development-r1/status.json').read_bytes()).hexdigest()=='FAILED_SHA'
assert not (t/'cuda-development-r2').exists() and not (t/'adapter-generation-api-r2').exists()
assert not (t/'validation-code/development-recovery-launch-0809.json').exists()
g=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert g.returncode==0 and not g.stdout.strip() and not g.stderr.strip()
print(json.dumps({'status':'RECOVERY_DEPLOYMENT_PRECONDITIONS_PASS'}))
""".replace('CAPTURED',snap['snapshot_utc']).replace('FAILED_SHA',prepared['failed_run_status_sha256'])
add('preflight',preflight)
encoded=base64.b64encode(raw).decode();pieces=[encoded[i:i+3500] for i in range(0,len(encoded),3500)]
for i,piece in enumerate(pieces):
    source="import pathlib,json;d=pathlib.Path('"+remote+"/validation-code/transfer-0809-recovery/"+bundle_sha+"');d.mkdir(parents=True,exist_ok=True);(d/'"+str(i)+"').open('x').write('"+piece+"');print(json.dumps({'chunk':"+str(i)+"}))"
    add('bundle-chunk-'+str(i),source)
seal="""import pathlib,json,base64,hashlib,io,tarfile
t=pathlib.Path('/data/spf-development-20261007-r1');d=t/'validation-code/transfer-0809-recovery/BUNDLE_SHA'
b=base64.b64decode(''.join((d/str(i)).read_text() for i in range(CHUNKS)))
assert hashlib.sha256(b).hexdigest()=='BUNDLE_SHA'
expected=MANIFEST
assert all(not (t/name).exists() for name in expected)
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as archive:
 assert sorted(archive.getnames())==sorted(expected)
 for name,row in expected.items():
  raw=archive.extractfile(name).read();assert len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  p=(t/name).resolve();assert p.is_relative_to(t.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(raw)
for name,row in expected.items():assert hashlib.sha256((t/name).read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'status':'EXCLUSIVE_RECOVERY_FILES_SEALED','files':expected}))
""".replace('BUNDLE_SHA',bundle_sha).replace('CHUNKS',str(len(pieces))).replace('MANIFEST',repr(manifest))
add('bundle-seal',seal)
launcher="""import pathlib,json,hashlib,subprocess,os,time,importlib.util,datetime
t=pathlib.Path('/data/spf-development-20261007-r1');code=t/'validation-code';receipt=code/'development-recovery-launch-0809.json'
assert not receipt.exists() and not (t/'cuda-development-r2').exists()
source=code/'run_spf_development_recovery_queue.py';guard_source=code/'run_spf_development_recovery_guards.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='CONTROLLER_SHA'
assert hashlib.sha256(guard_source.read_bytes()).hexdigest()=='GUARD_SHA'
snapshot=code/'r4-snapshot-0809.private.json';assert hashlib.sha256(snapshot.read_bytes()).hexdigest()=='SNAPSHOT_SHA'
spec=importlib.util.spec_from_file_location('recovery_guards',guard_source);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
guard.validate_snapshot(json.loads(snapshot.read_bytes()),datetime.datetime.now(datetime.timezone.utc))
assert hashlib.sha256((t/'cuda-development-r1/status.json').read_bytes()).hexdigest()=='FAILED_SHA'
guard.process_exited(guard.command_result(['ps','-p','45219','-o','pid,stat,args']))
guard.gpu_idle(guard.command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(source),'--controller-sha256','CONTROLLER_SHA','--snapshot',str(snapshot),'--snapshot-sha256','SNAPSHOT_SHA']
env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WANDB_MODE':'disabled','PYTHONPATH':str(t/'repo/src'),'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}
log=code/'development-recovery-launch-0809.raw.log'
with receipt.open('x',encoding='utf8') as handle:
 with log.open('xb') as stream:
  process=subprocess.Popen(command,cwd=t/'repo',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=process.pid,command=command,cwd=str(t/'repo'),started_at=time.time(),log=str(log),parent_failed_pid=46851,completed_stages_repeated=[],training_executed_by_launcher=False,external_judge_executed=False)
  handle.write(json.dumps(result,indent=2)+'\\n');handle.flush()
print(json.dumps(result))
""".replace('CONTROLLER_SHA',prepared['controller_sha256']).replace('GUARD_SHA',prepared['guard_sha256']).replace('SNAPSHOT_SHA',snapsha).replace('FAILED_SHA',prepared['failed_run_status_sha256'])
add('launch-once',launcher)
(out/'commands.private.json').write_text(json.dumps(commands))
(out/'preparation-as-run.py').write_bytes(pathlib.Path(__file__).read_bytes())
ps=(root/'work/spf-npo-gb200-20261006/deploy_0754_r2.ps1').read_text()
ps=ps.replace('development-deployment-20261007-0754-r2','development-recovery-deployment-20261007-0809')
(root/'work/spf-npo-gb200-20261006/deploy_development_recovery_0809.ps1').write_text(ps,encoding='utf8')
(out/'deployment-as-run.ps1').write_text(ps,encoding='utf8')
print(json.dumps({'commands':len(commands),'files':len(files),'bundle_sha256':bundle_sha,'snapshot_sha256':snapsha}))
