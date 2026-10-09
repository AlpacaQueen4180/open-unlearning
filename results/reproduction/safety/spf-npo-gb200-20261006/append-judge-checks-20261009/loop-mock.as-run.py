from pathlib import Path
from types import SimpleNamespace
import asyncio,ast,copy,ctypes,hashlib,importlib.util,json,os
import httpx,openai
root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private'
run=p/'spf-checkpoint-judge-20261008/checkpoint-313-saved-key-r2-append'
source=run/'run_spf_checkpoint_judge_append_state.py'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
spec=importlib.util.spec_from_file_location('append_state_judge_test',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
os.environ['CUDA_VISIBLE_DEVICES']=''
args=SimpleNamespace(run=str(run),source_sha256=sha(source.read_bytes()),plan_sha256=sha((run/'plan.private.json').read_bytes()),preflight=True)
_,plan,packets,specs,hashes=m.preflight(args)
out=p/'append-state-judge-checks-20261009';out.mkdir(exist_ok=False)
old_source=(root/'scripts/reproduction/safety/run_spf_checkpoint_development_judge_saved_key.py').read_bytes()
functions=lambda raw:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old_functions=functions(old_source);new_functions=functions(source.read_bytes())
unchanged=['digest','bound','safe_error','event','module','judge_rows'];assert all(old_functions[name]==new_functions[name] for name in unchanged)
rejected=[]
for field,value in [('selected_packet_indices',list(range(218,792))),('new_calls',574),('total_paid_api_request_limit',2377),('retries',1),('checkpoint_step',157)]:
 bad=copy.deepcopy(plan);bad[field]=value
 try:m.validate_profile(bad)
 except ValueError:rejected.append(field)
 else:raise AssertionError(field)

# Real Windows sharing-mode failure in a synthetic file, then fresh-state writes.
locked=out/'locked-job-primitive';locked.mkdir();job=locked/'job.json';job.write_text('{}')
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_void_p]
kernel.CreateFileW.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.CreateFileW(str(job),0x80000000,1,None,3,0x80,None)
assert handle not in (None,ctypes.c_void_p(-1).value)
try:
 temporary=locked/'original-replace.tmp';temporary.write_text('{"new":1}')
 try:temporary.replace(job)
 except PermissionError:pass
 else:raise AssertionError('Synthetic deny-delete handle must refuse replacement')
 m.atomic(job,dict(test=1));m.atomic(job,dict(test=2))
 assert json.loads(job.read_bytes())=={} and len(list((locked/'state-snapshots').glob('*.json')))==2
finally:kernel.CloseHandle(handle)

def value_for(prop):
 if 'anyOf' in prop:return value_for(next(row for row in prop['anyOf'] if row.get('type')!='null'))
 if 'enum' in prop:return prop['enum'][0]
 if prop.get('type')=='string':return 'synthetic rationale'
 if prop.get('type')=='boolean':return False
 if prop.get('type')=='integer':return max(1,prop.get('minimum',1))
 if prop.get('type')=='number':return max(.5,prop.get('minimum',0))
 raise AssertionError('Synthetic schema fixture')
counts={'complete573':0,'postresponse_failure':0};mode={'name':'complete573'}
def mock(request):
 name=mode['name'];relative=counts[name];counts[name]+=1;index=plan['selected_packet_indices'][relative];row=packets[index]
 payload=json.loads(request.content);expected=json.dumps(row['messages'],ensure_ascii=False)
 if row['kind']=='harmful':expected='HARMFUL USER INSTRUCTION:\n'+row['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+row['messages'][1]['content']
 assert payload['input']==expected and payload['instructions']==specs[row['kind']][0]
 assert payload['model']==plan['model'] and payload['reasoning']=={'effort':'medium'} and payload['max_output_tokens']==4096
 schema=specs[row['kind']][1].model_json_schema();judgment={key:value_for(schema['properties'][key]) for key in schema['required']}
 return httpx.Response(200,json=dict(id=f'resp_mock_{index}',created_at=1,model=plan['model'],status='completed',
  usage=dict(input_tokens=1,output_tokens=1,total_tokens=2,input_tokens_details=dict(cached_tokens=0),output_tokens_details=dict(reasoning_tokens=0)),
  output=[dict(id=f'msg_mock_{index}',type='message',status='completed',role='assistant',content=[dict(type='output_text',text=json.dumps(judgment),annotations=[])])]))
original_atomic=m.atomic
async def test(name):
 target=out/name;target.mkdir();(target/'raw-responses').mkdir()
 state=dict(status='RUNNING',attempted=0,completed=0);mode['name']=name;fired=False
 def atomic(path,data):
  nonlocal fired
  if name=='postresponse_failure' and not fired and data.get('last_operation')=='persist_completed_state':
   fired=True;raise PermissionError(13,'synthetic persistence failure')
  original_atomic(path,data)
 m.atomic=atomic
 client=openai.AsyncOpenAI(api_key='synthetic-not-a-credential',base_url=m.ENDPOINT,max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock)))
 try:
  try:await m.judge_rows(client,target,plan,packets,specs,hashes,state)
  except PermissionError as error:
   assert name=='postresponse_failure';state.update(status='STOPPED_ON_EXECUTION_ERROR_NO_RETRY',failure=m.safe_error(error,state['last_operation']))
  original_atomic(target/'job.json',state);m.seal_final_job(target/'job.json',state)
 finally:m.atomic=original_atomic;await client.close()
 expected=573 if name=='complete573' else 1
 assert counts[name]==state['attempted']==state['completed']==expected
 assert len(list((target/'raw-responses').glob('*.json')))==expected
 events=[json.loads(line) for line in (target/'execution-events.private.jsonl').read_bytes().splitlines()]
 assert len(events)==expected*3
 snapshots=sorted((target/'state-snapshots').glob('*.json'))
 assert [json.loads(path.read_bytes())['sequence'] for path in snapshots]==list(range(1,len(snapshots)+1))
 assert json.loads(snapshots[-1].read_bytes())['job']==json.loads((target/'job.json').read_bytes())==state
 if name=='complete573':assert state['status']=='CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'
 else:assert state['failure']['operation']=='persist_completed_state'
asyncio.run(test('complete573'));asyncio.run(test('postresponse_failure'))
record=dict(status='NEW573_LOOP_IMMUTABLE_STATE_AND_FAILSTOP_MOCK_PASS',source_sha256=sha(source.read_bytes()),unchanged_function_ast=unchanged,
 mocked_http=counts,actual_windows_synthetic_deny_delete_replacement_rejected=True,immutable_state_updates_with_held_reader_pass=True,
 original_failure_cause_proven_to_be_reader_lock=False,scope_negatives_rejected=rejected,real_api_calls=0,credential_reads=0,
 original792_loop_repeated=False,prior219_or157_audit_repeated=False)
(out/'checks.private.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8');(out/'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record))
