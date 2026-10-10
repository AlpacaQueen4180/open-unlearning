from pathlib import Path
import asyncio,hashlib,importlib.util,json,tempfile,sys,types
root=Path('.').resolve();s=root/'scripts/reproduction/safety'
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
runner=module(s/'run_spf_safety_search_judge.py','new_campaign_runner')
budget=module(s/'spf_safety_search_budget.py','new_campaign_budget')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert sha((s/'run_spf_checkpoint_judge_append_state.py').read_bytes())==runner.CORE_SHA
assert sha((s/'spf_safety_search_budget.py').read_bytes())==runner.BUDGET_SHA
class FakeResponse:
 id='synthetic-response-0';model=budget.PROFILE['model'];usage=None
 output_parsed=types.SimpleNamespace(model_dump=lambda:{'synthetic_fixture_only':True})
 def model_dump_json(self,indent=2):return json.dumps({'id':self.id,'model':self.model,'synthetic':True},indent=indent)
class FakeClient:
 def __init__(self):self.calls=0;self.responses=self
 async def parse(self,**kwargs):
  self.calls+=1
  assert kwargs['model']==budget.PROFILE['model'] and kwargs['reasoning']=={'effort':'medium'} and kwargs['max_output_tokens']==4096
  if self.calls==2:raise RuntimeError('synthetic API failure after one successful response')
  return FakeResponse()
checks=[]
with tempfile.TemporaryDirectory(prefix='spf-judge-hook-cpu-') as temp:
 base=Path(temp)
 for failure in ('api_after_one','state_after_receipt'):
  core=runner.module(s/'run_spf_checkpoint_judge_append_state.py',runner.CORE_SHA,'original_'+failure)
  campaign=budget.Campaign(base/(failure+'-campaign'))
  campaign.init({'max_model_judges':50,'requests_per_model':792,'mixing_harmful':13,'harmful_denominator':300,'judge_profile':budget.PROFILE,'development_counts':budget.KINDS},0)
  run=base/failure;run.mkdir();(run/'raw-responses').mkdir()
  packets=[{'id':str(i),'blind_id':'S'+str(i),'kind':kind,'messages':[{'content':'synthetic prompt'},{'content':'synthetic response'}]} for i,kind in enumerate(['harmful']*300+['benign']*350+['conversation']*142)]
  packet_path=run/'packet.json';packet_path.write_bytes(json.dumps(packets).encode())
  plan={'judge_profile':budget.PROFILE,'model_identity_sha256':sha(failure.encode()),'packet_sha256':sha(packet_path.read_bytes()),'new_campaign_model_limit':50,'development_outputs_and_candidate_audit_verified':True}
  plan_path=run/'plan.json';plan_path.write_bytes(json.dumps(plan).encode())
  slot=campaign.reserve(plan['model_identity_sha256'],packet_path,plan_path,run)
  if failure=='state_after_receipt':
   original_atomic=core.atomic
   def fail_state(path,value):
    if path.name=='job.json' and value.get('completed')==1:raise PermissionError('synthetic state failure')
    original_atomic(path,value)
   core.atomic=fail_state
  runner.attach_accounting(core,campaign,slot,run)
  client=FakeClient();state={'attempted':0,'completed':0}
  spec={kind:('synthetic original-call fixture',object,'fixture') for kind in budget.KINDS}
  loop_plan={'selected_packet_indices':[0,1],'model':budget.PROFILE['model']}
  try:asyncio.run(core.judge_rows(client,run,loop_plan,packets,spec,{kind:'fixture-sha' for kind in budget.KINDS},state))
  except (RuntimeError,PermissionError):pass
  else:raise AssertionError('Expected fail-stop integration')
  actual=campaign.status()
  assert actual['durable_receipts']==1
  assert (actual['request_intents'],client.calls)==((2,2) if failure=='api_after_one' else (1,1))
  assert (run/'raw-responses/0000.json').is_file()
  checks.append({'case':failure,'synthetic_calls':client.calls,'durable_intents':actual['request_intents'],'durable_receipts':1,'automatic_retries':0})
out=root/'work/spf-npo-gb200-20261006/private/spf-safety-search-20261010/new-judge-hooks-mock.private.json'
result={'status':'PASS_NEW_LEDGER_HOOKS_ON_UNCHANGED_ORIGINAL_PARSE_LOOP_ONLY','checks':checks,'real_api_calls':0,'credential_reads':0,'gpu_submissions':0,'original_loop_source_sha256':runner.CORE_SHA,'budget_source_sha256':runner.BUDGET_SHA,'runner_source_sha256':sha((s/'run_spf_safety_search_judge.py').read_bytes()),'actual_full_model_judge_complete':False,'original_schema_fixtures_or_completed_result_audits_repeated':False}
with out.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'status':result['status'],'new_hook_mock_cases':len(checks),'real_api_calls':0,'credential_reads':0}))
