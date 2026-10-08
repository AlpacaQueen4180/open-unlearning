"""Prepare a new executor and scope evidence; no GPU, API or remote operation."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, sys

w=Path('work/spf-npo-gb200-20261006')
out=w/'private/checkpoint-beta-source-preparation-20261008'
out.mkdir(exist_ok=False)
old=w/'private/development-completed-20261007-1140/validation-code/run_spf_development_recovery_queue.py'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=old.read_bytes()
assert sha(raw)=='46252ad177ae1ef8d77f328700e919094a9df09538037a9e288739e83c0349ad'
text=raw.decode('utf8').replace('\r\n','\n')
text=text.replace("TASK = Path('/data/spf-development-20261007-r1')", "TASK = Path(os.environ['SPF_DIAG_CASE_TASK'])")
text=text.replace("CPU_PYTHON = str(TASK/'ifbench-cpu-venv/bin/python')", "CPU_PYTHON = '/data/spf-development-20261007-r1/ifbench-cpu-venv/bin/python'")
text=text.replace("ADAPTER_SHA = '71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de'", "ADAPTER_SHA = os.environ['SPF_DIAG_ADAPTER_SHA']")
text=text.replace("IDENTITY = 'c45e87fc69825b75f5d47345fb3c8502a4c1b4524f9d621411068bf3bc64f3a8'", "IDENTITY = os.environ['SPF_DIAG_IDENTITY']")
text=text.replace("path = Path(__file__).with_name('run_spf_development_recovery_guards.py')", "path = Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py')")
start=text.index('def acceptance(')
# The old acceptance/controller identity belongs to the completed final target.
text=text[:start]+'''def main():
    p = argparse.ArgumentParser(description='New early-checkpoint child; original actual-result validators')
    p.add_argument('--child-stage', choices=STAGES[:-1], required=True)
    p.add_argument('--child-run', required=True)
    a = p.parse_args()
    root = Path('/data/spf-checkpoint-beta-20261008-r1')
    if (sys.executable != PYTHON or os.name != 'posix'
            or TASK.parent != root or TASK.name not in ('checkpoint-157','checkpoint-313','checkpoint-469')
            or a.child_run != str(TASK/'evaluation') or len(IDENTITY) != 64 or len(ADAPTER_SHA) != 64):
        raise ValueError('Exact new checkpoint child identity and runtime required')
    child(a.child_stage, TASK/'evaluation')

if __name__ == '__main__': main()
'''
new=Path('scripts/reproduction/safety/gb200/spf_checkpoint_executor.py')
if new.exists():raise FileExistsError('Preserve previous new source')
new.write_bytes(text.encode('utf8'))
old_ast=ast.parse(raw);new_ast=ast.parse(text)
functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=functions(old_ast),functions(new_ast)
same=sorted(k for k in b if k not in ('guards','main'))
assert all(a[k]==b[k] for k in same)
audit_path=w/'private/development-completed-20261007-1140/adapter-generation-api-r2/adapter-audit.json'
audit=json.loads(audit_path.read_bytes())
assert sha(audit_path.read_bytes())=='71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de'
adapters=[]
for name,entry in audit['sources'].items():
    original=(audit_path.parent/'code'/name).read_bytes()
    assert sha(original)==entry['adapter_sha256']
    value=original.decode('utf8')
    if name!='score_development_ifbench.py':
        assert value.count(audit['checkpoint'])==2,(name,'checkpoint occurrences')
        assert value.count(audit['checkpoint_identity_sha256'])==1,(name,'identity occurrences')
        changed=value.replace(audit['checkpoint'],'/data/spf-npo-20261006-r4/spf-full-s0/checkpoint-157').replace(audit['checkpoint_identity_sha256'],'0'*64)
        restored=changed.replace('/data/spf-npo-20261006-r4/spf-full-s0/checkpoint-157',audit['checkpoint']).replace('0'*64,audit['checkpoint_identity_sha256'])
        assert restored.encode('utf8')==original
        compile(changed,name,'exec')
    adapters.append(dict(name=name,original_sha256=sha(original),inverse_bytes_equal=True))
queue=Path('scripts/reproduction/safety/gb200/run_spf_checkpoint_beta_queue.py')
spec=importlib.util.spec_from_file_location('new_scope_queue',queue);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=dict(checkpoint_steps=[157,313,469],beta1_values=[0.9,0.5],diagnostic_updates_per_beta=32,
          paid_judge_calls_in_gpu_queue=0,proposed_new_judge_request_cap=2376,retry_count=0,
          existing_final_checkpoint_repeated=False,formal_hexphi_selection=False)
m.validate_plan(plan)
negative=[]
for field,value in [('checkpoint_steps',[157,313,625]),('beta1_values',[0.9,0.7]),
                    ('diagnostic_updates_per_beta',125),('paid_judge_calls_in_gpu_queue',1),
                    ('proposed_new_judge_request_cap',2377),('retry_count',1),
                    ('existing_final_checkpoint_repeated',True),('formal_hexphi_selection',True)]:
    invalid=copy.deepcopy(plan);invalid[field]=value
    try:m.validate_plan(invalid)
    except ValueError:negative.append(field)
    else:raise AssertionError('New scope negative accepted: '+field)
sources={p.name:sha(p.read_bytes()) for p in (new,queue,Path('scripts/reproduction/safety/gb200/spf_beta_geometry_diagnostic.py'))}
for p in (new,queue,Path('scripts/reproduction/safety/gb200/spf_beta_geometry_diagnostic.py')):compile(p.read_bytes(),str(p),'exec')
result=dict(status='NEW_SCOPE_AND_SOURCE_AST_CHECKS_ONLY',unchanged_executor_function_ast=same,
            adapter_identity_inverse_bytes=adapters,new_scope_negative_rejections=negative,
            sources_sha256=sources,gpu_operations=0,model_weight_reads=0,new_api_calls=0,
            cpu_adamw_selftest_not_yet_run=True,paid_judge_scope_requires_new_cap=True,
            profile=plan,prefix_limit='All 32 updates are within original 125-update warmup; no late harmfulness inference')
(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
(out/'scope-source-checks.private.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
