from pathlib import Path
import asyncio, copy, importlib.util, json, os
from types import SimpleNamespace
import httpx
from openai import AsyncOpenAI

root = Path('.').resolve()
run = root / 'work/spf-npo-gb200-20261006/private/spf-judge-local-20261007-r3'
source = run / 'run_spf_development_judge_local_r2.py'
spec = importlib.util.spec_from_file_location('schema_recovery_checked', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan = json.loads((run / 'plan.private.json').read_bytes())
os.environ['CUDA_VISIBLE_DEVICES'] = ''
args = SimpleNamespace(run=str(run),source_sha256=m.digest(source.read_bytes()),
                       plan_sha256=m.digest((run/'plan.private.json').read_bytes()),preflight=True)
_, _, rows, specs, hashes, harmful = m.preflight(args)
assert len(rows) == 789 and rows[0]['kind'] == 'harmful'
bad_profiles = []
for field,value in [('new_calls',790),('reused_response_labels',2)]:
    wrong=copy.deepcopy(plan)
    wrong[field]=value
    try: m.validate_profile(wrong)
    except ValueError: bad_profiles.append(field)
    else: raise AssertionError(field)
def value_for(prop):
    if 'anyOf' in prop:
        return value_for(next(p for p in prop['anyOf'] if p.get('type') != 'null'))
    if 'enum' in prop: return prop['enum'][0]
    kind=prop.get('type')
    if kind=='string': return 'synthetic rationale'
    if kind=='boolean': return False
    if kind=='integer': return max(1,prop.get('minimum',1))
    if kind=='number': return max(0.5,prop.get('minimum',0))
    raise AssertionError(prop)
count={'mock_http':0}
current={}
def mock(request):
    count['mock_http']+=1
    payload=json.loads(request.content)
    kind=current['kind']
    assert payload['model']=='gpt-5.6-terra' and payload['reasoning']['effort']=='medium'
    assert payload['max_output_tokens']==4096 and payload['instructions']==specs[kind][0]
    assert payload['text']['format']['type']=='json_schema'
    schema=specs[kind][1].model_json_schema()
    result={k:value_for(schema['properties'][k]) for k in schema['required']}
    return httpx.Response(200,json=dict(id='resp_synthetic',created_at=1,model='gpt-5.6-terra',status='completed',
        output=[dict(id='msg_synthetic',type='message',status='completed',role='assistant',
                     content=[dict(type='output_text',text=json.dumps(result),annotations=[])])]))
async def check():
    async with AsyncOpenAI(api_key='synthetic-not-a-credential',max_retries=0,
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock))) as client:
        for kind,(prompt,schema,_) in specs.items():
            current['kind']=kind
            response=await client.responses.parse(model=plan['model'],reasoning={'effort':'medium'},
                instructions=prompt,input='synthetic offline schema validation',text_format=schema,max_output_tokens=4096)
            assert isinstance(response.output_parsed,schema)
asyncio.run(check())
assert count['mock_http']==3
e=dict(status='THREE_ORIGINAL_SCHEMAS_MOCK_SDK_PARSE_PASS_REMAINING789_BOUND',
       mock_http_requests=3,real_api_calls=0,credential_reads=0,three_schemas=list(specs),
       remaining789_verified=True,completed_spf3_not_replayed=True,negative_profiles_rejected=bad_profiles,
       rubric_prompt_hashes_unchanged=hashes)
(run/'schema-recovery-checks.private.json').write_text(json.dumps(e,indent=2)+'\n',encoding='utf8')
print(json.dumps(e))
