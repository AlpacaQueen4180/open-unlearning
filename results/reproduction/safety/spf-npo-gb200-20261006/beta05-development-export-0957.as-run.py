"""Use only after the hourly snapshot confirms all six new development stages completed."""
from pathlib import Path
import os,sys,hashlib,json,importlib.util,io,tarfile,base64
T=Path('/data/spf-beta05-full-20261010-r2');V=T/'validation-code';C=T/'candidate-development';R=C/'evaluation'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
case=json.loads((T/'candidate-identity.private.json').read_bytes())
assert case['identity_sha256']=='9f8b68d4f657d2021c63b2aebe8ae76020b1c2d18e779c35451bd4b6d01c79c9'
adapterraw=(C/'adapter-generation-api-r2/adapter-audit.json').read_bytes()
adapter=json.loads(adapterraw);assert adapter['checkpoint_identity_sha256']==case['identity_sha256']
assert adapter['candidate_audit_sha256']=='63e72cb101b8056c938021e9518361280ed0df9f9238df04034342bf71d4110b'
os.environ.update(SPF_DIAG_CASE_TASK=str(C),SPF_DIAG_IDENTITY=case['identity_sha256'],SPF_DIAG_ADAPTER_SHA=sha(adapterraw))
source=V/'spf_beta05_serialization_r2_development_executor.py'
assert sha(source.read_bytes())=='76c5fd9c0a6228643d3c9a11621cfa90db52fed65efde1f403d86a7f23c14191'
spec=importlib.util.spec_from_file_location('completed_beta05_executor',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
state_raw=(R/'status.json').read_bytes();state=json.loads(state_raw)
assert state['status']=='BETA05_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE' and state['completed']==list(m.STAGES)
assert state['commands']==m.plan(R) and state['controller_sha256']==sha(source.read_bytes())
files={R/'status.json',source}
files.update(p for p in (C/'adapter-generation-api-r2').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for row in state['stages']:
 assert row['returncode']==0 and row['executor_command']==state['commands'][row['name']]
 assert m.actual_result(R,row['name'],row)==row['outputs']
 log=Path(row['log']);assert sha(log.read_bytes())==row['raw_log_sha256'];files.add(log)
 files.update(R/name for name in row['outputs'])
assert (R/'status.json').read_bytes()==state_raw
records=[];stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
 for path in sorted(files):
  assert path.resolve().is_relative_to(T) and not path.is_symlink()
  raw=path.read_bytes();assert len(raw)<15000000 and path.suffix not in ('.safetensors','.pt','.bin')
  name=path.relative_to(T).as_posix();records.append({'name':name,'remote_path':str(path),'size':len(raw),'sha256':sha(raw)})
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
raw=stream.getvalue()
print(json.dumps({'status':'IMMUTABLE_BETA05_COMPLETED_DEVELOPMENT_EXPORT','files':records,
 'archive_sha256':sha(raw),'archive_base64':base64.b64encode(raw).decode(),
 'model_identity_sha256':case['identity_sha256'],'completed_stages':state['completed'],
 'actual_outputs_commands_caps_cuda_sources_verified':True,'candidate_train_reload_audit_verified':True,
 'weights_read':False,'evaluations_repeated':0,'progress_snapshot':False,'paid_api_calls':0}))
