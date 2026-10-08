from pathlib import Path
import ast,copy,hashlib,importlib.util,json,shutil
root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private'
run=p/'spf-checkpoint-judge-20261008/checkpoint-157-saved-key-r1'
source=run/'run_spf_checkpoint_development_judge_saved_key.py'
spec=importlib.util.spec_from_file_location('saved_key_judge_guard_test',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=json.loads((run/'plan.private.json').read_bytes());m.validate_profile(plan)
rejected=[]
for field,bad in [('credential_policy','new_key_memory_only'),('total_paid_api_request_limit',2377),('retries',1)]:
 value=copy.deepcopy(plan);value[field]=bad
 try:m.validate_profile(value)
 except ValueError:rejected.append(field)
 else:raise AssertionError(field)
out=p/'saved-key-judge-guards-20261008';out.mkdir(exist_ok=False)
shutil.copyfile(run.parent/'authorization.private.json',out/'authorization.private.json')
target=out/'checkpoint-157-saved-key-r1';target.mkdir()
newplan=copy.deepcopy(plan);newplan['budget_root']=str(out)
m.budget_check(target,newplan)
(out/'allocations/157.json').write_text(json.dumps(dict(checkpoint_step=157,reserved_request_limit=792)))
try:m.budget_check(target,newplan)
except FileExistsError:rejected.append('allocated_checkpoint')
else:raise AssertionError('existing allocation accepted')
original=(root/'scripts/reproduction/safety/run_spf_checkpoint_development_judge.py').read_bytes()
functions=lambda raw:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=functions(original),functions(source.read_bytes())
same=['digest','bound','atomic','safe_error','event','module','judge_rows']
assert all(a[name]==b[name] for name in same)
e=dict(status='SAVED_KEY_NEW_PROFILE_AND_EXISTING_BUDGET_GUARDS_PASS',rejected=rejected,unchanged_function_ast=same,
 key_file_content_read=False,credential_reads=0,api_calls=0,original_completed792_loop_repeated=False,
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(out/'checks.private.json').write_text(json.dumps(e,indent=2)+'\n',encoding='utf8')
(out/'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(e))
