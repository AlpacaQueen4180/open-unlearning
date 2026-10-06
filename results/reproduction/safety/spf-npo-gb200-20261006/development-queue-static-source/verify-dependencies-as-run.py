import ast,hashlib,json,pathlib,sys,tempfile,subprocess,shutil
repo=pathlib.Path.cwd();code=repo/'scripts/reproduction/safety/gb200'
private=repo/'work/spf-npo-gb200-20261006/private'
previous=private/'development-queue-preparation-20261007-r1'
output=repo/'results/reproduction/safety/spf-npo-gb200-20261006/development-queue-dependencies-static-20261007.json'
root=private/'development-queue-preparation-20261007-r2';root.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(code/'run_spf_development_queue.py',root/'run_spf_development_queue.py')
sys.path.insert(0,str(code));import run_spf_development_queue as queue
old_ast=ast.parse((previous/'run_spf_development_queue.py').read_bytes())
new_ast=ast.parse((code/'run_spf_development_queue.py').read_bytes())
functions=('plan','environment','validate_cuda','validate_metadata','validate_rows','acceptance')
extract=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions}
assert extract(old_ast)==extract(new_ast)
rejected=[]
with tempfile.TemporaryDirectory(dir=root) as td:
    tmp=pathlib.Path(td);names=['src/construction/'+name+'.py' for name in ('__init__','baseline','data','learning_metrics','protocol','knowledge')]+['configs/model/Llama-3.1-8B-Instruct.yaml']
    for name in names:
        path=tmp/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic frozen bytes\n')
    manifest=dict(construction_code={name:sha(tmp/name) for name in names})
    queue.validate_dependencies(tmp,manifest)
    for name in ('src/construction/baseline.py','src/construction/data.py','src/construction/learning_metrics.py','src/construction/protocol.py','src/construction/knowledge.py','configs/model/Llama-3.1-8B-Instruct.yaml'):
        path=tmp/name;raw=path.read_bytes();path.write_bytes(raw+b'changed')
        try:queue.validate_dependencies(tmp,manifest)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
        path.write_bytes(raw)
    missing=dict(construction_code={k:v for k,v in manifest['construction_code'].items() if k!='src/construction/data.py'})
    try:queue.validate_dependencies(tmp,missing)
    except ValueError:rejected.append('missing_frozen_dependency_identity')
    else:raise AssertionError('missing dependency accepted')
result=dict(status='PASS_SYNTHETIC_FROZEN_DEPENDENCY_BYTES_ONLY',rejected_cases=rejected,rejected_count=len(rejected),
    source_sha256=sha(code/'run_spf_development_queue.py'),local_verifier_sha256=sha(pathlib.Path(__file__)),
    prior_result_sha256=sha(repo/'results/reproduction/safety/spf-npo-gb200-20261006/development-queue-static-20261007.json'),
    prior_tested_source_sha256=sha(previous/'run_spf_development_queue.py'),prior_tested_functions_ast_unchanged=list(functions),
    previous_28_cases_reexecuted=False,subprocesses_launched=0,torch_imported=False,gpu_runtime_validated=False,
    target_evaluation_executed=False,limitations=['All dependency bytes are synthetic; not an actual remote frozen-source or CUDA runtime check'])
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
(root/'result.json').write_bytes(output.read_bytes())
print(json.dumps(dict(status=result['status'],rejected_count=len(rejected),source_sha256=result['source_sha256'])))
