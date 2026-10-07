"""Audit only new26 raw receipts; reuse the SHA-bound verified765 labels."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.metadata,importlib.util,json,os,sys
root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private'
run=p/'spf-judge-local-20261007-r6-unsent26';prior=p/'judge-result-audit-20261007-1936'
out=p/'judge-unsent26-result-audit-20261008-0340'
sha=lambda b:hashlib.sha256(b).hexdigest()
def bound(path,expected):
    raw=path.read_bytes();assert sha(raw)==expected,path.name
    return json.loads(raw)
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
versions={k:importlib.metadata.version(k) for k in ['openai','pydantic','python-dotenv']}
assert versions=={'openai':'2.54.0','pydantic':'2.13.4','python-dotenv':'1.2.3'}
job=json.loads((run/'job.json').read_bytes())
assert job['status']=='UNSENT26_COMPLETE_PENDING_RESULT_AUDIT_UNKNOWN1_AND_CLUSTER_GATE'
assert job['attempted']==job['completed']==26 and job['retries']==0 and job['statuses']=={'success':26}
assert job['actual_models']=={'gpt-5.6-terra':26} and job['uncertain_packet_indices']==[765]
plan=bound(run/'plan.private.json','5fa6dd0c1e3a919d88d161d9c93b5dbb018b2b1cb4566f8aeab01d34c2a5726b')
assert job['plan_sha256']==sha((run/'plan.private.json').read_bytes())
assert sha((run/'run_spf_development_judge_unsent26.py').read_bytes())==job['source_sha256']==plan['runner_sha256']=='128a7362a30d7e4b794fd1a707966f2bf3133cebf8233b1d99c4a12f241bea9e'
assert sha((run/'legacy-runner.py').read_bytes())=='8f88420437e4a8d38a862eb85dfa7b09d4c08520dec14de02aed2fa0fb4b09b2'
assert plan['selected_packet_indices']==list(range(766,792)) and not plan['uncertain_request_retry_permitted']
assert plan['total_paid_api_request_limit']==792 and plan['new_calls']==26 and plan['reused_response_labels']==765
assert plan['reasoning']=='medium' and plan['max_output_tokens']==4096 and plan['retries']==0
assert plan['api_base_url']=='https://api.openai.com/v1' and plan['credential_policy']=='new_key_memory_only'
packets=bound(run/'private-packet.json','7183c481e35b8e8bcc3043c81c106128b1e05ff49786dcbeeca7a59d97673bb6')
assert len(packets)==792 and len({r['id'] for r in packets})==792
old_audit=bound(prior/'audit.private.json','67cc4cd2d0eadd900f4abf8c6bd665adeffb7737ea28e5ee659dda2e383faea8')
old_labels=bound(prior/'verified-labels.private.json','9d8abdaa71055cc2871807e84783da250289943319723d50d08c745224e2a75f')
old_map=bound(prior/'verified-source-map.private.json','09cc8efed244fbad500384aa8eda1135f54cd7c3b44e0c53bacbbf527720ffcf')
bound(prior/'missing-packets.private.json','726a3fb9664ef6cbd511aa759484ea4cd2dec7577a59c6d4be6954374a81eecb')
assert old_audit['verified_successes']==len(old_labels)==len(old_map)==765
assert old_audit['uncertain_packet_index']==765 and old_audit['uncertain_execution']==1
for i,r in enumerate(old_labels):
    assert (r['id'],r['kind'])==(packets[i]['id'],packets[i]['kind'])
schemas={};prompt_versions={'harmful':'hexphi-two-dimensional-v1','benign':'benign-development-v1','conversation':'wildchat-development-quality-v1'}
types={'harmful':('src/evals/safety/gpt_judge.py','JudgeResult'),'benign':('src/construction/development_judge.py','BenignJudgment'),'conversation':('src/construction/conversation_judge.py','ConversationJudgment')}
for kind,(name,cls) in types.items():
    path=run/'rubrics'/name;assert sha(path.read_bytes())==plan['rubric_sources'][name]['sha256']
    spec=importlib.util.spec_from_file_location('new26_audit_'+kind,path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    schemas[kind]=getattr(m,cls);assert sha(m.SYSTEM_PROMPT.encode())==plan['prompt_hashes'][kind]
rows=[json.loads(x) for x in (run/'judge-results.jsonl').read_bytes().splitlines()]
events=[json.loads(x) for x in (run/'execution-events.private.jsonl').read_bytes().splitlines()]
assert len(rows)==26 and len(events)==78
assert sorted(f.name for f in (run/'raw-responses').glob('*.json'))==[f'{i:04}.json' for i in range(766,792)]
newmap=[];newusage=Counter()
for position,(index,r) in enumerate(zip(range(766,792),rows)):
    packet=packets[index];kind=packet['kind']
    assert r['packet_index']==index and r['id']==packet['id'] and r['blind_id']==packet['blind_id'] and r['kind']==kind
    assert r['status']=='success' and r['attempt']==1 and r['requested_model']==r['actual_model']=='gpt-5.6-terra'
    assert r['prompt_sha256']==plan['prompt_hashes'][kind] and r['prompt_version']==prompt_versions[kind]
    payload=json.dumps(packet['messages'],ensure_ascii=False)
    if kind=='harmful':payload='HARMFUL USER INSTRUCTION:\n'+packet['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+packet['messages'][1]['content']
    assert sha(payload.encode())==r['request_payload_sha256']
    assert schemas[kind].model_validate(r['judgment']).model_dump()==r['judgment']
    path=run/'raw-responses'/f'{index:04}.json';raw=path.read_bytes();assert sha(raw)==r['raw_response_sha256']
    response=json.loads(raw);assert response['id']==r['response_id'] and response['model']==r['actual_model'] and response['status']=='completed'
    texts=[c['text'] for item in response['output'] for c in item.get('content',[]) if c.get('type')=='output_text']
    assert len(texts)==1 and json.loads(texts[0])==r['judgment'] and response['usage']==r['usage']
    ev=events[position*3:position*3+3]
    assert [e['operation'] for e in ev]==['intent_durable_before_api','api_request_started','api_response_received']
    assert all(e['packet_index']==index for e in ev) and ev[0]['request_payload_sha256']==r['request_payload_sha256']
    assert ev[2]['response_id']==r['response_id'] and ev[2]['actual_model']==r['actual_model'] and ev[2]['request_id']==r['request_id']
    assert job['started_at']<=ev[0]['at']<=ev[1]['at']<=ev[2]['at']<=job['finished_at']
    assert r['request_started_at']<=ev[0]['at'] and ev[2]['at']<=r['request_finished_at']
    newusage.update(input_tokens=r['usage']['input_tokens'],output_tokens=r['usage']['output_tokens'])
    newmap.append(dict(packet_index=index,id=r['id'],kind=kind,source_run=run.name,raw_name=path.name,raw_sha256=sha(raw),payload_sha256=r['request_payload_sha256']))
assert dict(newusage)==job['usage']
combined=old_labels+rows;mapping=old_map+newmap
assert len(combined)==len({r['id'] for r in combined})==len({r['response_id'] for r in combined})==791
assert [r['packet_index'] for r in mapping]==list(range(765))+list(range(766,792))
counts=Counter();labels=defaultdict(Counter);sums=defaultdict(Counter);denoms=defaultdict(Counter);usage=Counter()
for r in combined:
    kind=r['kind'];counts[kind]+=1
    for field,value in r['judgment'].items():
        if field in ('safety_label','generation_quality','response_behavior','quality'):labels[kind+':'+field][value]+=1
        if isinstance(value,(int,float)) and not isinstance(value,bool):sums[kind][field]+=value;denoms[kind][field]+=1
    usage.update(input_tokens=r['usage']['input_tokens'],output_tokens=r['usage']['output_tokens'])
assert counts==Counter(r['kind'] for i,r in enumerate(packets) if i!=765)
assert dict(usage)==dict(Counter(old_audit['token_usage_for_verified_responses'])+newusage)
out.mkdir(exist_ok=False)
write(out/'new26-verified-labels.private.json',rows);write(out/'combined791-verified-labels.private.json',combined)
write(out/'combined791-source-map.private.json',mapping);write(out/'remaining-uncertain.private.json',dict(packet_index=765,packet=packets[765],state='UNKNOWN_DO_NOT_RETRY'))
manifest={}
paths=[run/'job.json',run/'plan.private.json',run/'judge-results.jsonl',run/'execution-events.private.jsonl',run/'run_spf_development_judge_unsent26.py']+sorted((run/'raw-responses').glob('*.json'))
for path in paths:manifest[path.relative_to(run).as_posix()]=dict(size=path.stat().st_size,sha256=sha(path.read_bytes()))
write(out/'new26-source-manifest.private.json',manifest)
audit=dict(status='NEW26_RAW_SCHEMA_PAYLOAD_EVENTS_VERIFIED_COMBINED791_WITH_UNKNOWN1',observed_at_utc=datetime.now(timezone.utc).isoformat(),new_verified_successes=26,prior765_reused_by_sha256=True,prior765_raw_reaudited=False,verified_successes=791,expected=792,uncertain_execution=1,uncertain_packet_index=765,uncertain_api_sent_or_charged='UNKNOWN_DO_NOT_RETRY',not_attempted=0,full792_complete=False,new_raw_schema_payload_order_matches=26,new_durable_intent_response_event_chains_verified=26,duplicates=0,actual_models={'gpt-5.6-terra':791},completed_kind_counts=dict(counts),new_token_usage=dict(newusage),combined_verified_token_usage=dict(usage),labels={k:dict(v) for k,v in labels.items()},descriptive_score_sums={k:dict(v) for k,v in sums.items()},descriptive_score_nonnull_denominators={k:dict(v) for k,v in denoms.items()},descriptive_score_means={k:{f:v/denoms[k][f] for f,v in c.items()} for k,c in sums.items()},runtime_versions=versions,runner_sha256=job['source_sha256'],plan_sha256=job['plan_sha256'],packet_sha256=plan['packet_sha256'],job_sha256=sha((run/'job.json').read_bytes()),results_sha256=sha((run/'judge-results.jsonl').read_bytes()),events_sha256=sha((run/'execution-events.private.jsonl').read_bytes()),prior_verified_labels_sha256=sha((prior/'verified-labels.private.json').read_bytes()),combined_labels_sha256=sha((out/'combined791-verified-labels.private.json').read_bytes()),combined_source_map_sha256=sha((out/'combined791-source-map.private.json').read_bytes()),new_source_manifest_sha256=sha((out/'new26-source-manifest.private.json').read_bytes()),paid_cap_accounting='791 confirmed +1 reserved uncertain = maximum original792',new_api_calls_by_auditor=0,credential_reads_by_auditor=0,uncertain_request_retried=False,m0_response_labels_transferred=False,cluster_mapping_verified=False,human_adjudication_complete=False,target_gate='insufficient_evidence',pilot_frozen=False,statistics_scope='SPF M_pre descriptive791 subset only; not full792, NPO final, FormalHExPHI or paired gate')
write(out/'audit.private.json',audit);(out/'auditor.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=audit['status'],verified_successes=791,uncertain=1,kinds=dict(counts),new_usage=dict(newusage),combined_usage=dict(usage),harmful_assistance=labels['harmful:safety_label']['harmful_assistance'],harmful_denominator=counts['harmful'],combined_labels_sha256=audit['combined_labels_sha256'],auditor_api_calls=0),ensure_ascii=False))
