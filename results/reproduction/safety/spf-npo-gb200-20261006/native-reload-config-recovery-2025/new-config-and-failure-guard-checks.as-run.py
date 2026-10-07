from pathlib import Path
import ast, copy, hashlib, json

w=Path('work/spf-npo-gb200-20261006'); p=w/'private'; src=Path('scripts/reproduction/safety/gb200')
first=p/'native-config-recovery-preparation-20261007-2025'
out=p/'native-config-recovery-preparation-20261007-2025-r2'; out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
(first/'preparation-check-failure.metadata.json').write_text(json.dumps(dict(
    error_type='AssertionError',operation='literal command spacing check',
    source_sha256=sha((w/'prepare_native_config_recovery_2025.py').read_bytes()),
    original_stream_bytes_captured=False,remote_commands=0,production_sources_compiled_and_written_before_check=True),indent=2)+'\n',encoding='utf8')
v=(first/'prior-verifier.py').read_bytes(); q=(first/'prior-controller.py').read_bytes()
vf=src/'verify_spf_npo_smoke_native_reload_config_r3.py'; qf=src/'run_spf_npo_smoke_reload_config_r3_queue.py'
vt=vf.read_bytes(); qt=qf.read_bytes()
def funcs(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
uv=[n for n in funcs(v) if n not in ['resolved_reference_config','main']]; uq=[n for n in funcs(q) if n!='main']
assert all(funcs(v)[n]==funcs(vt)[n] for n in uv)
assert all(funcs(q)[n]==funcs(qt)[n] for n in uq)
commands=[n.value for n in ast.walk(ast.parse(qt)) if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='command' for a in n.targets) and isinstance(n.value,ast.List)]
assert any(len(n.elts)>6 and isinstance(n.elts[5],ast.Constant) and n.elts[5].value=='--' and isinstance(n.elts[6],ast.Call) and ast.unparse(n.elts[6])=='str(verifier)' for n in commands)
ns=dict(json=json);fn=next(n for n in ast.parse(vt).body if isinstance(n,ast.FunctionDef) and n.name=='resolved_reference_config')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'isolated-new-config','exec'),ns)
cfg=dict(bf16=dict(enabled=True),train_batch_size=32,train_micro_batch_size_per_gpu=4,gradient_accumulation_steps=8,
         zero_optimization=dict(stage=3,reduce_bucket_size='auto',stage3_prefetch_bucket_size='auto',stage3_param_persistence_threshold='auto'))
before=copy.deepcopy(cfg); resolved=ns['resolved_reference_config'](cfg,4096)
assert cfg==before and type(resolved['zero_optimization']['stage3_prefetch_bucket_size']) is int
assert resolved['zero_optimization']==dict(stage=3,reduce_bucket_size=16777216,stage3_prefetch_bucket_size=15099494,stage3_param_persistence_threshold=40960)
state=json.loads((p/'native-reload-config-failure-20261007-2025/07-status.json').read_bytes())
guard=next(n for n in ast.parse(qt).body if isinstance(n,ast.FunctionDef) and n.name=='failed_native_config')
gn=dict(PYTHON='/data/npo-gb200-20261004/validation-20261006/venv/bin/python',TASK=Path('/data/spf-npo-smoke-20261007-r1'))
exec(compile(ast.Module(body=[guard],type_ignores=[]),'isolated-new-failure-guard','exec'),gn)
gn['failed_native_config'](state)
negative=[]
for key,value in [('pid',4075),('status','RUNNING'),('completed',['fresh-reload']),('training_repeated',True)]:
    bad=copy.deepcopy(state);bad[key]=value
    try:gn['failed_native_config'](bad)
    except ValueError:negative.append(key)
    else:raise AssertionError(key)
for key,value in [('returncode',0),('process_pid',4082),('raw_log_sha256','0'*64),('command',[])]:
    bad=copy.deepcopy(state);bad['stages'][0][key]=value
    try:gn['failed_native_config'](bad)
    except ValueError:negative.append(key)
    else:raise AssertionError(key)
e=dict(status='SOURCE_READY_CPU_AST_CONFIG_AND_NEW_FAILURE_GUARDS_ONLY',verifier_sha256=sha(vt),controller_sha256=sha(qt),
       unchanged_verifier_functions=uv,unchanged_controller_functions=uq,new_failure_negatives=negative,
       separator_ast_verified=True,original_configuration_not_mutated=True,corrected_prefetch_integer=15099494,
       hf_integer_rule_source_sha256='194c012cde03afdf8acacf0b94acc3e4e475f08cdaf9fa1f5be0ef6ac5a16bc3',
       original_reference_log_defaults=dict(reduce_bucket_size=500000000,prefetch_bucket_size=50000000,param_persistence_threshold=100000),
       target_bucket_resolution_inferred_from_installed_training_source=True,original_native_prepare_unchanged=True,
       subprocesses_launched=0,torch_imports=0,cuda_initialized=False,weights_read=False,model_reload_executed=False,
       completed_cases_repeated=False,threshold=dict(atol=.05,rtol=.01),pilot_frozen=False,deployed=False)
(out/'evidence.private.json').write_text(json.dumps(e,indent=2)+'\n',encoding='utf8')
print(json.dumps(e))
