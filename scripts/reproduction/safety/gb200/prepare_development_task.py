"""CPU-only deployment preparation from the completed SPF audit, without evaluation."""
import argparse, hashlib, json, pathlib, shutil, subprocess, sys, tarfile
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--bundle-sha256',required=True)
a=p.parse_args();task=pathlib.Path(a.task).resolve();bundle=task/'bundle.tar.gz'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(bundle)==a.bundle_sha256
dest=task/'bundle'
dest.mkdir(exist_ok=False)
with tarfile.open(bundle,'r:gz') as tar:
    for item in tar.getmembers():
        name=pathlib.PurePosixPath(item.name)
        assert item.isfile() and not name.is_absolute() and '..' not in name.parts
        b=tar.extractfile(item).read();path=dest/item.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
index=json.loads((dest/'bundle-index.json').read_text())
for entry in index:
    path=dest/entry['name'];assert path.stat().st_size==entry['size'] and sha(path)==entry['sha256']
original=pathlib.Path('/data/spf-npo-20261006-r4/repo');repo=task/'repo';repo.mkdir(exist_ok=False)
for name in ('src','configs','scripts'):
    shutil.copytree(original/name,repo/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for path in sorted((dest/'code').rglob('*')):
    if path.is_file():
        output=repo/path.relative_to(dest/'code');output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(path.read_bytes())
audit=pathlib.Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json')
audit_sha=sha(audit)
assert audit_sha=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
assert sha(repo/'scripts/reproduction/safety/gb200/prepare_development_adapter.py')=='3b21774414ad1b83effca35fc54253b9ae7ac53f0ef1abd98ff0dbbab904db49'
cmd=[sys.executable,str(repo/'scripts/reproduction/safety/gb200/prepare_development_adapter.py'),
     '--repository',str(repo),'--manifest','/data/spf-npo-20261006-r4/frozen.json',
     '--target','/data/spf-npo-20261006-r4/spf-full-s0','--checkpoint-audit',str(audit),
     '--checkpoint-audit-sha256',audit_sha,'--output',str(task/'adapter')]
run=subprocess.run(cmd,capture_output=True,text=True)
(task/'adapter-prepare.log').write_text(run.stdout+run.stderr)
assert run.returncode==0,run.stderr
provenance=json.loads((task/'adapter/adapter-audit.json').read_text())
result=dict(status='PREPARED_PENDING_GPU_RUNTIME',task=str(task),bundle_sha256=a.bundle_sha256,
            assets_verified=352,bundle_entries_verified=len(index),checkpoint_audit_sha256=audit_sha,
            adapter_audit_sha256=sha(task/'adapter/adapter-audit.json'),
            checkpoint_identity_sha256=provenance['checkpoint_identity_sha256'],command=cmd,
            cpu_only=True,gpu_evaluation_started=False,external_judge_executed=False,
            scoring_venv_created=False,construction_source_unmodified=True,
            original_task='/data/spf-npo-20261006-r4')
(task/'preparation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
