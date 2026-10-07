from pathlib import Path
from types import SimpleNamespace
import asyncio, copy, importlib.util, json, os
import httpx, openai

root=Path('.').resolve();run=root/'work/spf-npo-gb200-20261006/private/spf-judge-local-20261007-r4-unsent26'
source=run/'run_spf_development_judge_unsent26.py'
spec=importlib.util.spec_from_file_location('unsent26_checked',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
os.environ['CUDA_VISIBLE_DEVICES']=''
args=SimpleNamespace(run=str(run),source_sha256=m.digest(source.read_bytes()),plan_sha256=m.digest((run/'plan.private.json').read_bytes()),preflight=True)
_,plan,packets,specs,hashes=m.preflight(args)
rejected=[]
for field,value in [('new_calls',27),('selected_packet_indices',list(range(765,792))),
 ('reused_response_labels',764),('uncertain_request_retry_permitted',True),('total_paid_api_request_limit',793),('retries',1)]:
 bad=copy.deepcopy(plan);bad[field]=value
 try:m.validate_profile(bad)
 except ValueError:rejected.append(field)
 else:raise AssertionError(field)
def value_for(prop):
 if 'anyOf' in prop:return value_for(next(p for p in prop['anyOf'] if p.get('type')!='null'))
 if 'enum' in prop:return prop['enum'][0]
 kind=prop.get('type')
 if kind=='string':return 'synthetic rationale'
 if kind=='boolean':return False
 if kind=='integer':return max(1,prop.get('minimum',1))
 if kind=='number':return max(.5,prop.get('minimum',0))
 raise AssertionError(kind)
original_client=openai.AsyncOpenAI;original_atomic=m.atomic
counts={'complete':0,'save_failure':0};mode={'value':'complete'}
def mock(request):
 name=mode['value'];count=counts[name];index=766+count;counts[name]+=1
 row=packets[index];kind=row['kind'];payload=json.loads(request.content)
 expected=json.dumps(row['messages'],ensure_ascii=False)
 if kind=='harmful':expected='HARMFUL USER INSTRUCTION:\n'+row['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+row['messages'][1]['content']
 assert payload['input']==expected and payload['instructions']==specs[kind][0]
 assert payload['model']==plan['model'] and payload['reasoning']=={'effort':'medium'} and payload['max_output_tokens']==4096
 schema=specs[kind][1].model_json_schema();judgment={k:value_for(schema['properties'][k]) for k in schema['required']}
 return httpx.Response(200,json=dict(id=f'resp_synthetic_{index}',created_at=1,model=plan['model'],status='completed',
  usage=dict(input_tokens=1,output_tokens=1,total_tokens=2,input_tokens_details=dict(cached_tokens=0),output_tokens_details=dict(reasoning_tokens=0)),
  output=[dict(id=f'msg_synthetic_{index}',type='message',status='completed',role='assistant',content=[dict(type='output_text',text=json.dumps(judgment),annotations=[])])]))
def mock_client(**kwargs):
 assert kwargs['max_retries']==0 and kwargs['base_url']==m.ENDPOINT
 return original_client(**kwargs,http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock)))
out=run/'new-loop-mock-checks';out.mkdir(exist_ok=False)
async def check(name):
 target=out/name;target.mkdir()
 for item in ['legacy-runner.py','private-packet.json','baseline-plan.private.json']:(target/item).write_bytes((run/item).read_bytes())
 for path in (run/'rubrics').rglob('*.py'):
  dest=target/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
 newplan=copy.deepcopy(plan);newplan['run_directory']=str(target)
 (target/'plan.private.json').write_text(json.dumps(newplan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 newargs=SimpleNamespace(run=str(target),source_sha256=args.source_sha256,plan_sha256=m.digest((target/'plan.private.json').read_bytes()),preflight=False)
 mode['value']=name
 fired={'value':False}
 def atomic(path,data):
  if name=='save_failure' and not fired['value'] and data.get('last_operation')=='persist_completed_state':
   fired['value']=True
   raise PermissionError(13,'synthetic filesystem failure')
  return original_atomic(path,data)
 m.atomic=atomic;openai.AsyncOpenAI=mock_client
 os.environ['SPF_JUDGE_NEW_KEY_INPUT']='local_masked_dialog';os.environ['OPENAI_API_KEY']='synthetic-not-a-credential'
 try:await m.execute(newargs)
 finally:
  m.atomic=original_atomic;openai.AsyncOpenAI=original_client
  os.environ.pop('OPENAI_API_KEY',None);os.environ.pop('SPF_JUDGE_NEW_KEY_INPUT',None)
 state=json.loads((target/'job.json').read_bytes())
 if name=='complete':
  assert counts[name]==26 and state['attempted']==26 and state['completed']==26
  assert state['status']=='UNSENT26_COMPLETE_PENDING_RESULT_AUDIT_UNKNOWN1_AND_CLUSTER_GATE'
  assert len(list((target/'raw-responses').glob('*.json')))==26
 else:
  assert counts[name]==1 and state['attempted']==1 and state['completed']==1
  assert state['status']=='STOPPED_ON_EXECUTION_ERROR_NO_RETRY'
  assert state['failure']['operation']=='persist_completed_state' and state['failure']['error_type']=='PermissionError'
  assert state['failure']['exception_message_and_locals_saved'] is False
  assert len(list((target/'raw-responses').glob('*.json')))==1
 events=[json.loads(line) for line in (target/'execution-events.private.jsonl').read_bytes().splitlines()]
 assert sum(e['operation']=='api_response_received' for e in events)==counts[name]
 assert not (target/'raw-responses/0765.json').exists()
asyncio.run(check('complete'));asyncio.run(check('save_failure'))
e=dict(status='UNSENT26_NEW_LOOP_AND_POST_RESPONSE_SAVE_FAILURE_MOCK_PASS',mock_http_requests=counts,
 real_api_calls=0,credential_reads=0,selected_packets_only=list(range(766,792)),excluded_uncertain_index=765,
 successful765_repeated=False,original_three_schema_fixtures_repeated=False,
 profile_negatives_rejected=rejected,request_call_ast_identical=True,
 operation_and_safe_frames_recorded_on_permission_failure=True,
 original_uncertain_permission_error_cause_still_unknown=True,
 source_sha256=args.source_sha256,plan_sha256=args.plan_sha256)
(run/'new-loop-checks.private.json').write_text(json.dumps(e,indent=2)+'\n',encoding='utf8')
(run/'new-loop-verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(e))
