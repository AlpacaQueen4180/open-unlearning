from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys

root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private'
first=p/'spf-judge-local-20261007-r2';last=p/'spf-judge-local-20261007-r3'
out=p/'judge-result-audit-20261007-1936';out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
j=lambda f:json.loads(f.read_bytes())
packet_raw=(last/'private-packet.json').read_bytes()
assert sha(packet_raw)=='7183c481e35b8e8bcc3043c81c106128b1e05ff49786dcbeeca7a59d97673bb6'
packets=json.loads(packet_raw);plan=j(last/'plan.private.json')
assert len(packets)==792 and len({r['id'] for r in packets})==792
assert sha((last/'run_spf_development_judge_local_r2.py').read_bytes())==plan['runner_sha256']=='8f88420437e4a8d38a862eb85dfa7b09d4c08520dec14de02aed2fa0fb4b09b2'
assert sha((first/'job.json').read_bytes())==plan['prior_job_sha256']
assert sha((first/'judge-results.jsonl').read_bytes())==plan['prior_results_sha256']
schemas={};prompts={}
types={'harmful':('src/evals/safety/gpt_judge.py','JudgeResult'),
       'benign':('src/construction/development_judge.py','BenignJudgment'),
       'conversation':('src/construction/conversation_judge.py','ConversationJudgment')}
for kind,(name,cls) in types.items():
    path=last/'rubrics'/name
    assert sha(path.read_bytes())==plan['rubric_sources'][name]['sha256']
    spec=importlib.util.spec_from_file_location('result_audit_'+kind,path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    schemas[kind]=getattr(m,cls);prompts[kind]=m.SYSTEM_PROMPT
    assert sha(m.SYSTEM_PROMPT.encode())==plan['prompt_hashes'][kind]
oldrows=[json.loads(line) for line in (first/'judge-results.jsonl').read_bytes().splitlines()]
newrows=[json.loads(line) for line in (last/'judge-results.jsonl').read_bytes().splitlines()]
assert len(oldrows)==4 and [r['status'] for r in oldrows]==['success']*3+['error']
assert oldrows[-1]['error_type']=='PydanticUserError'
state=j(last/'job.json')
assert state['status']=='STOPPED_ON_EXECUTION_ERROR_NO_RETRY' and state['error_type']=='PermissionError'
assert state['attempted']==763 and state['completed']==len(newrows)==762
assert all(r['status']=='success' for r in newrows)
records=[(first,i,r) for i,r in enumerate(oldrows[:3])]+[(last,i+3,r) for i,r in enumerate(newrows)]
counts=Counter();labels=defaultdict(Counter);score_sums=defaultdict(Counter);score_counts=defaultdict(Counter)
usage=Counter();manifest=[];models=Counter()
for index,(folder,raw_index,r) in enumerate(records):
    row=packets[index]
    assert r['id']==row['id'] and r['blind_id']==row['blind_id'] and r['kind']==row['kind']
    assert r['attempt']==1 and r['requested_model']==r['actual_model']=='gpt-5.6-terra'
    payload=json.dumps(row['messages'],ensure_ascii=False)
    if row['kind']=='harmful':
        payload='HARMFUL USER INSTRUCTION:\n'+row['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+row['messages'][1]['content']
    assert sha(payload.encode())==r['request_payload_sha256']
    assert r['prompt_sha256']==plan['prompt_hashes'][row['kind']]
    parsed=schemas[row['kind']].model_validate(r['judgment']).model_dump()
    assert parsed==r['judgment']
    rawpath=folder/'raw-responses'/f'{raw_index:04}.json';raw=rawpath.read_bytes()
    assert sha(raw)==r['raw_response_sha256']
    response=json.loads(raw)
    assert response['model']==r['actual_model'] and response['id']==r['response_id']
    texts=[c['text'] for item in response['output'] for c in item.get('content',[]) if c.get('type')=='output_text']
    assert len(texts)==1 and json.loads(texts[0])==r['judgment']
    assert response['usage']==r['usage']
    counts[row['kind']]+=1;models[r['actual_model']]+=1
    for field,value in parsed.items():
        if isinstance(value,str) and not field.endswith('reason') and field not in ('reason','rationale'):
            if value in ('refusal','safe_non_refusal','harmful_assistance','indeterminate','normal','degenerate'):
                labels[row['kind']+':'+field][value]+=1
        if isinstance(value,(float,int)) and not isinstance(value,bool):
            score_sums[row['kind']][field]+=value;score_counts[row['kind']][field]+=1
    usage['input_tokens']+=r['usage']['input_tokens'];usage['output_tokens']+=r['usage']['output_tokens']
    manifest.append(dict(packet_index=index,id=r['id'],kind=r['kind'],source_run=folder.name,
                         raw_name=rawpath.name,raw_sha256=sha(raw),payload_sha256=r['request_payload_sha256']))
assert len(records)==len({r['id'] for _,_,r in records})==765
assert counts==Counter(r['kind'] for r in packets[:765])
assert len(list((last/'raw-responses').glob('*.json')))==762
assert not (last/'raw-responses/0765.json').exists()
write=lambda f,d:f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
write(out/'verified-labels.private.json',[r for _,_,r in records])
write(out/'verified-source-map.private.json',manifest)
write(out/'missing-packets.private.json',dict(uncertain=packets[765:766],not_attempted=packets[766:]))
audit=dict(status='PARTIAL_JUDGE_765_LABELS_RAW_SCHEMA_PAYLOAD_SOURCE_VERIFIED',
    observed_at_utc=datetime.now(timezone.utc).isoformat(),expected=792,verified_successes=765,
    missing=27,uncertain_execution=1,not_attempted=26,
    uncertain_packet_index=765,uncertain_api_sent_or_charged='UNKNOWN_DO_NOT_RETRY',
    cause='Local PermissionError; saved error_type does not identify which filesystem operation failed',
    previous_original_pre_http_schema_error=1,previous_original_schema_error_retried_as_paid_request=False,
    schema_and_raw_response_matches=765,payload_and_ids_match=765,duplicates=0,
    completed_kind_counts=dict(counts),expected_kind_counts=plan['kinds'],actual_models=dict(models),
    token_usage_for_verified_responses=dict(usage),
    labels={k:dict(v) for k,v in labels.items()},
    descriptive_partial_score_means={kind:{field:total/score_counts[kind][field] for field,total in fields.items()} for kind,fields in score_sums.items()},
    statistics_scope='Only verified completed subset; not full792 aggregate or paired safety-gate estimates',
    original_packets_sha256=sha(packet_raw),first_job_sha256=sha((first/'job.json').read_bytes()),
    first_results_sha256=sha((first/'judge-results.jsonl').read_bytes()),last_job_sha256=sha((last/'job.json').read_bytes()),
    last_results_sha256=sha((last/'judge-results.jsonl').read_bytes()),runner_sha256=plan['runner_sha256'],
    verified_labels_sha256=sha((out/'verified-labels.private.json').read_bytes()),
    source_map_sha256=sha((out/'verified-source-map.private.json').read_bytes()),
    missing_packets_sha256=sha((out/'missing-packets.private.json').read_bytes()),
    new_api_calls=0,credential_reads=0,training_executed=False,model_generation_executed=False,
    original_raw_records_unchanged=True,cluster_mapping_verified=False,human_adjudication_complete=False,
    target_gate='insufficient_evidence',pilot_frozen=False)
write(out/'audit.private.json',audit)
print(json.dumps(audit,ensure_ascii=False))
