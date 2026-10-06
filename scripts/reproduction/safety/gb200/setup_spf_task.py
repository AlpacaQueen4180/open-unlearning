"""Verify private source lineage and compose an isolated fixed-upstream SPF checkout."""
import argparse, hashlib, importlib, json, pathlib, shutil, subprocess, sys, tarfile

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);args=parser.parse_args()
    task=pathlib.Path(args.task).resolve()
    assert task.parent==pathlib.Path('/data') and task.name.startswith('spf-npo-')
    source=task/'source.tar.gz'
    assert sha(source)=='e82e22ef75d9774ac64d8bbe88a68f0bfc3b472ac880edce2dc2ed204b4351b0'
    with tarfile.open(source) as t:t.extractall(task/'source',filter='data')
    index=json.loads((task/'source/private/export-index.json').read_text())
    for e in index['entries']:assert sha(task/'source'/e['dest'])==e['sha256'],e['dest']
    base=pathlib.Path('/data/npo-gb200-20261004/validation-20261006/upstream')
    assert subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()=='17cbbc87192e6934deb92875c359c91bbd837fb4'
    assert sha(base/'src/evals/metrics/utils.py')=='acd98144cada6f02dea329bb78c77e317bd651d16fabbb576796253ae6c4ec7b'
    repo=task/'repo'
    assert not repo.exists(),'Use a new setup task after failed partial setup'
    shutil.copytree(base,repo,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    overlay=task/'overlay'
    for p in overlay.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json'):
            dest=repo/p.relative_to(overlay);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    # Add construction registrations to the NEW stack without replacing NPO code.
    registry=repo/'src/trainer/__init__.py';s=registry.read_text()
    s=s.replace('from trainer.base import FinetuneTrainer','from trainer.base import FinetuneTrainer\nfrom construction.runner import ConstructionTrainer, SafetyMixingTrainer, SPFTrainer')
    s=s.replace('_register_trainer(FinetuneTrainer)','_register_trainer(FinetuneTrainer)\n_register_trainer(ConstructionTrainer)\n_register_trainer(SafetyMixingTrainer)\n_register_trainer(SPFTrainer)')
    registry.write_text(s)
    parent=task/'source/private/frozen-parent.json';m=json.loads(parent.read_text())
    m['parent_manifest_sha256']=sha(parent)
    m['repository_commit']='17cbbc87192e6934deb92875c359c91bbd837fb4'
    m['anchor_path']=str(task/'source/private/prepared/anchor.jsonl')
    for v in m['datasets'].values():v['path']=str(task/'source/private/prepared'/pathlib.Path(v['path']).name)
    for i,v in enumerate(m['source_specs']):v['path']=str(task/'source/private/overlap'/f"{i}-{pathlib.Path(v['path']).name}")
    # Parent readiness is evidence for two H100 ranks only.
    m['parent_readiness']=m.pop('readiness');m['readiness']={'status':'pending','world_size':1,'global_batch':32}
    m['profile']={'hardware':'GB200','world_size':1,'global_batch':32,'initial_microbatch':4,'initial_accumulation':8}
    m['construction_code']={str(p.relative_to(repo)):sha(p) for p in repo.rglob('*') if p.is_file() and (p.suffix in ('.py','.json','.yaml') and '__pycache__' not in p.parts)}
    m['checkpoint_policy']='HF weights at 25/50/75/100 percent plus final export; no runtime; no resume'
    sys.path.insert(0,str(repo/'src'))
    # PYTHONPATH named this directory before it existed at process startup.
    importlib.invalidate_caches()
    from construction.protocol import audit_sources,write_json
    assert audit_sources(m['anchor_path'],m['anchor_sha256'],m['source_specs'])==m['audit']
    write_json(task/'prepared.json',m)
    env=json.loads(subprocess.check_output([sys.executable,'-c',"import json,torch,transformers,accelerate,deepspeed,bitsandbytes,flash_attn; print(json.dumps({m.__name__:m.__version__ for m in [torch,transformers,accelerate,deepspeed,bitsandbytes,flash_attn]}))"],text=True).splitlines()[-1])
    assert env['transformers']=='5.5.4' and env['accelerate']=='1.13.0' and env['deepspeed']=='0.15.4',env
    (task/'environment-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write_json(task/'setup-evidence.json',{'status':'pass','source_files':len(index['entries']),'source_archive_sha256':sha(source),'parent_sha256':sha(parent),'prepared_sha256':sha(task/'prepared.json'),'environment':env,'environment_freeze_sha256':sha(task/'environment-freeze.txt'),'construction_code':m['construction_code'],'storage_available_bytes':shutil.disk_usage(task).free,'planned_peak_disk_budget_bytes':600*1024**3,'pvc_exclusive_quota':'not advertised by RunAI workspace describe; shared filesystem free space is not a quota guarantee'})
    assert shutil.disk_usage(task).free>600*1024**3
    print(json.dumps({'status':'pass','prepared_sha256':sha(task/'prepared.json'),'environment':env}))

if __name__=='__main__':main()
