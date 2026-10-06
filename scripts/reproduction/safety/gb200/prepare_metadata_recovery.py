"""Reuse completed setup/assets/CPU/reference evidence after a metadata-only fix."""
import argparse,hashlib,json,pathlib,shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--task',required=True);a=p.parse_args()
    old=pathlib.Path(a.previous);new=pathlib.Path(a.task)
    assert new.parent==old.parent==pathlib.Path('/data') and new!=old
    s=json.loads((old/'status.json').read_text())
    required=['setup','verify_m0_assets','cpu_tests','tiny_reference']
    assert s['status']=='FAILED' and s['completed']==required and s['stage']=='tiny_zero3_world1'
    assert json.loads((old/'m0-assets.json').read_text())['status']=='pass'
    assert json.loads((old/'tiny-reference/acceptance.json').read_text())['reference_comparison']=='reference_created'
    assert not (new/'repo').exists() and not (new/'status.json').exists()
    shutil.copytree(old/'repo',new/'repo')
    prior=(old/'repo/src/construction/runner.py').read_text()
    updated=(new/'updates/src/construction/runner.py').read_text()
    added='        # Capture metadata before ZeRO replaces the optimizer with a wrapper.\n        self.optimizer_metadata = {"class": type(optimizer).__name__, "defaults": dict(optimizer.defaults)}\n'
    normalized=updated.replace(added,'').replace('"optimizer_defaults": self.optimizer_metadata["defaults"],','"optimizer_defaults": dict(self.optimizer.defaults),').replace('                        "base_optimizer": self.optimizer_metadata["class"],\n','')
    assert normalized==prior,'Reject non-metadata edits when reusing numerical reference'
    for file in (new/'updates').rglob('*'):
        if file.is_file():
            dest=new/'repo'/file.relative_to(new/'updates');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    shutil.copytree(old/'tiny-reference',new/'tiny-reference')
    m=json.loads((old/'prepared.json').read_text());m['recovery_parent_prepared_sha256']=sha(old/'prepared.json')
    m['construction_code']={str(f.relative_to(new/'repo')):sha(f) for f in (new/'repo').rglob('*') if f.is_file() and f.suffix in ('.py','.json','.yaml') and '__pycache__' not in f.parts}
    (new/'prepared.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    proofs={}
    for name in ['setup-evidence.json','environment-freeze.txt','m0-assets.json','cpu_tests.log','tiny-reference/acceptance.json','tiny-reference/gradients.pt','tiny-reference/parameters.pt']:
        proofs[name]={'source_task':old.name,'sha256':sha(old/name)}
    (new/'recovery-evidence.json').write_text(json.dumps({'status':'pass','fix_scope':'optimizer metadata only','previous_task':str(old),
        'reused_stages':required,'reused_artifact_hashes':proofs,'previous_runner_sha256':sha(old/'repo/src/construction/runner.py'),
        'updated_runner_sha256':sha(new/'repo/src/construction/runner.py'),'prepared_sha256':sha(new/'prepared.json')},indent=2))
    shutil.copy2(old/'environment-freeze.txt',new/'environment-freeze.txt')
    print(json.dumps({'status':'pass','task':str(new),'reused_stages':required,'prepared_sha256':sha(new/'prepared.json')}))

if __name__=='__main__':main()
