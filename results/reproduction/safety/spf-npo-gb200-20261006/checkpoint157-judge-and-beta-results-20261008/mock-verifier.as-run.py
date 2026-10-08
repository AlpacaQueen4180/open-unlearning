from pathlib import Path
from types import SimpleNamespace
import asyncio,ast,copy,importlib.util,json,os,shutil
import httpx,openai

root=Path('.').resolve();private=root/'work/spf-npo-gb200-20261006/private'
run=private/'spf-checkpoint-judge-20261008/checkpoint-157-r1'
source=run/'run_spf_checkpoint_development_judge.py'
spec=importlib.util.spec_from_file_location('new_checkpoint_judge',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
os.environ['CUDA_VISIBLE_DEVICES']=''
args=SimpleNamespace(run=str(run),source_sha256=m.digest(source.read_bytes()),plan_sha256=m.digest((run/'plan.private.json').read_bytes()),preflight=True)
_,plan,packets,specs,hashes=m.preflight(args)
out=private/'checkpoint-judge-mock-20261008';out.mkdir(exist_ok=False)
rejected=[]
for field,value in [('total_paid_api_request_limit',2377),('new_calls',793),('retries',1),('checkpoint_step',625),('reused_response_labels',1),('previous_uncertain_request_retried',True)]:
 bad=copy.deepcopy(plan);bad[field]=value
 try:m.validate_profile(bad)
 except ValueError:rejected.append(field)
 else:raise AssertionError(field)
old=ast.parse((root/'scripts/reproduction/safety/run_spf_development_judge_unsent26.py').read_bytes())
new=ast.parse(source.read_bytes())
call=lambda tree:next(ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Await) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='parse')
assert call(old)==call(new)
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
 name=mode['value'];index=counts[name];counts[name]+=1
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
async def check(name):
 budget=out/name;budget.mkdir();target=budget/'checkpoint-157-r1';target.mkdir()
 shutil.copyfile(run.parent/'authorization.private.json',budget/'authorization.private.json')
 for item in ['private-packet.json','baseline-plan.private.json']:
  shutil.copyfile(run/item,target/item)
 shutil.copytree(run/'rubrics',target/'rubrics')
 newplan=copy.deepcopy(plan);newplan.update(run_directory=str(target),budget_root=str(budget))
 (target/'plan.private.json').write_text(json.dumps(newplan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 newargs=SimpleNamespace(run=str(target),source_sha256=args.source_sha256,plan_sha256=m.digest((target/'plan.private.json').read_bytes()),preflight=False)
 mode['value']=name;fired={'value':False}
 def atomic(path,data):
  if name=='save_failure' and not fired['value'] and data.get('last_operation')=='persist_completed_state':
   fired['value']=True;raise PermissionError(13,'synthetic filesystem failure')
  return original_atomic(path,data)
 m.atomic=atomic;openai.AsyncOpenAI=mock_client
 os.environ['SPF_JUDGE_NEW_KEY_INPUT']='local_masked_dialog';os.environ['OPENAI_API_KEY']='synthetic-not-a-credential'
 try:await m.execute(newargs)
 finally:
  m.atomic=original_atomic;openai.AsyncOpenAI=original_client
  os.environ.pop('OPENAI_API_KEY',None);os.environ.pop('SPF_JUDGE_NEW_KEY_INPUT',None)
 state=json.loads((target/'job.json').read_bytes());expected=792 if name=='complete' else 1
 assert counts[name]==expected and state['attempted']==expected and state['completed']==expected
 assert len(list((target/'raw-responses').glob('*.json')))==expected
 if name=='complete':assert state['status']=='CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'
 else:
  assert state['status']=='STOPPED_ON_EXECUTION_ERROR_NO_RETRY'
  assert state['failure']['operation']=='persist_completed_state' and state['failure']['error_type']=='PermissionError'
  assert state['failure']['exception_message_and_locals_saved'] is False
 events=[json.loads(line) for line in (target/'execution-events.private.jsonl').read_bytes().splitlines()]
 assert sum(e['operation']=='api_response_received' for e in events)==expected
 for index in range(expected):
  order=[e['operation'] for e in events if e.get('packet_index')==index]
  assert order[:3]==['intent_durable_before_api','api_request_started','api_response_received']
 allocation=json.loads((budget/'allocations/157.json').read_bytes());assert allocation['reserved_request_limit']==792
 try:m.budget_check(target,newplan)
 except FileExistsError:rejected.append(name+'_duplicate_allocation')
 else:raise AssertionError('duplicate allocation')
 for step in [313,469]:(budget/'allocations'/f'{step}.json').write_text(json.dumps(dict(checkpoint_step=step,reserved_request_limit=792)))
 try:m.budget_check(target,newplan)
 except FileExistsError:rejected.append(name+'_full_budget')
 else:raise AssertionError('cap')
asyncio.run(check('complete'));asyncio.run(check('save_failure'))
e=dict(status='NEW_CHECKPOINT_792_LOOP_AND_POST_RESPONSE_FILESYSTEM_FAILSTOP_MOCK_PASS',source_sha256=args.source_sha256,
 mock_http_requests=counts,real_api_calls=0,credential_reads=0,profile_and_allocation_negatives_rejected=rejected,
 original_api_call_ast_identical=True,durable_event_order_verified=True,original_completed_cases_repeated=False,
 checkpoint_labels_generated=False)
(out/'checks.private.json').write_text(json.dumps(e,indent=2)+'\n',encoding='utf8')
(out/'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(e))
