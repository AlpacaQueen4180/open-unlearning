from pathlib import Path
from types import SimpleNamespace
import asyncio,hashlib,importlib.util,json,sys
import httpx
from openai import BadRequestError
root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private/spf-safety-search-20261010'
out=p/'model06-unsent788-new-integration-20261013-0051';out.mkdir(exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);loaded=importlib.util.module_from_spec(spec);sys.modules[name]=loaded;spec.loader.exec_module(loaded);return loaded
runner=module(root/'scripts/reproduction/safety/run_spf_safety_search_judge_unsent.py','rank40_new_unsent_test_runner')
corepath=root/'scripts/reproduction/safety/run_spf_checkpoint_judge_append_state.py'
assert sha(corepath.read_bytes())==runner.CORE_SHA
class Budget:
    def __init__(self):self.indices=[];self.receipts=[]
    def intent(self,slot,index,payload):
        assert slot==6 and index==4+len(self.indices) and 4<=index<792;self.indices.append(index)
    def receipt(self,slot,index,raw,response_id,model):
        assert slot==6 and index==4+len(self.receipts) and model=='gpt-5.6-terra';self.receipts.append(index)
class Parsed:
    def model_dump(self):return {'mock_label':'fixture'}
class Usage:
    input_tokens=1;output_tokens=1
    def model_dump(self):return dict(input_tokens=1,output_tokens=1,total_tokens=2)
class Responses:
    def __init__(self,error=False):self.calls=0;self.error=error
    async def parse(self,**kwargs):
        assert kwargs['model']=='gpt-5.6-terra' and kwargs['reasoning']=={'effort':'medium'} and kwargs['max_output_tokens']==4096
        index=4+self.calls;self.calls+=1
        if self.error:
            response=httpx.Response(400,request=httpx.Request('POST','https://example.invalid/mock'))
            raise BadRequestError('synthetic message excluded from saved metadata',response=response,
                body={'error':{'code':'mock_invalid_request','type':'mock_request_error','message':'do not save this body'}})
        result=SimpleNamespace(id='mock_response_'+str(index),model='gpt-5.6-terra',_request_id='mock_request',output_parsed=Parsed(),usage=Usage())
        result.model_dump_json=lambda indent=2:json.dumps(dict(id=result.id,model=result.model,mock_only=True),indent=indent)
        return result
async def case(name,error):
    run=out/name;run.mkdir();(run/'raw-responses').mkdir()
    core=module(corepath,'rank40_mock_core_'+name);budget=Budget();runner.attach_accounting(core,budget,6,run)
    client=SimpleNamespace(responses=Responses(error));state=dict(status='RUNNING',attempted=0,completed=0)
    packets=[dict(id=str(i),blind_id=str(i),kind='benign',messages=[dict(role='user',content='synthetic'),dict(role='assistant',content='synthetic')]) for i in range(792)]
    try:
        await core.judge_rows(client,run,dict(selected_packet_indices=list(range(4,792)),model='gpt-5.6-terra'),packets,
            dict(benign=('synthetic rubric',Parsed,'mock')),dict(benign='mock'),state)
    except BadRequestError as error_object:
        assert error
        safe=runner.safe_api_error(core,error_object,state['last_operation'])
        assert safe['http_status_code']==400 and safe['api_error_code']=='mock_invalid_request'
        assert safe['api_error_type']=='mock_request_error' and not safe['headers_response_body_message_and_key_saved']
        assert 'synthetic message' not in json.dumps(safe) and 'do not save this body' not in json.dumps(safe)
        assert client.responses.calls==1 and state['attempted']==1 and state['completed']==0
        assert budget.indices==[4] and budget.receipts==[] and not list((run/'raw-responses').iterdir())
        (out/'mock400-safe-metadata.json').write_text(json.dumps(safe,indent=2),encoding='utf8')
        return 1
    assert not error
    assert state['attempted']==state['completed']==client.responses.calls==788
    assert budget.indices==budget.receipts==list(range(4,792))
    assert 3 not in budget.indices and 0 not in budget.indices
    return client.responses.calls
positive=asyncio.run(case('new788_boundary_positive',False));failure=asyncio.run(case('new_first400_failure_stop',True))
plan=dict(campaign_slot=6,new_calls=788,verified_prior_successes=3,reserved_uncertain_indices=[3],
    previous_uncertain_request_retried=False,final_spf_candidate=True,spf_projection_rank=40)
negatives=0
for key,value in [('campaign_slot',7),('new_calls',789),('verified_prior_successes',4),('reserved_uncertain_indices',[]),
    ('previous_uncertain_request_retried',True),('final_spf_candidate',False),('spf_projection_rank',20)]:
    broken=dict(plan);broken[key]=value
    try:runner.validate_recovery(broken,None,None)
    except ValueError:negatives+=1
    else:raise AssertionError('Negative guard admitted unsafe recovery')
evidence=dict(status='NEW_UNSENT788_BOUNDARY_ACCOUNTING_FAILSTOP_AND_SAFE400_METADATA_MOCK_PASS',
    runner_sha256=sha((root/'scripts/reproduction/safety/run_spf_safety_search_judge_unsent.py').read_bytes()),
    original_core_sha256=runner.CORE_SHA,new_success_mock_parse_calls=positive,new_error_mock_parse_calls=failure,
    new_guard_negatives=negatives,selected_original_indices=list(range(4,792)),prior_or_uncertain_indices_retried=False,
    original_schema_fixtures_not_rerun=True,scope='new subset/accounting/failstop integration only; mock parsed objects, no actual SDK response equivalence or paid result',
    real_api_calls=0,credential_reads=0,actual_budget_mutations=0,new_gpu_submissions=0)
(out/'evidence.private.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
(out/'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=evidence['status'],mock_parse_calls=positive+failure,guard_negatives=negatives,real_api_calls=0,credential_reads=0)))
