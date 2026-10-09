"""Audit only new395469 raw, reuse396 verified labels, retain uncertain harmful396."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys
root=Path('.').resolve();private=root/'work/spf-npo-gb200-20261006/private'
budget=private/'spf-checkpoint-judge-20261008';run=budget/'checkpoint-469-saved-key-r3-unsent395'
out=private/'checkpoint-469-unsent395-combined791-audit-20261009'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda v:(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
plan_raw=(run/'plan.private.json').read_bytes();plan=json.loads(plan_raw)
assert sha(plan_raw)=='268bc260861a0780b98ae77f0378e54bb38bb2884d39b434d37efdbf7bf7dac5'
assert plan['selected_packet_indices']==list(range(397,792)) and plan['new_calls']==395 and plan['reserved_uncertain_indices']==[396]
indices=list(range(397,792));count=395
job_raw=(run/'job.json').read_bytes();job=json.loads(job_raw)
assert job['status']=='CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'
assert job['attempted']==job['completed']==job['new_call_limit']==395 and job['statuses']=={'success':395} and job['retries']==0
assert job['checkpoint_step']==plan['checkpoint_step']==469 and job['plan_sha256']==sha(plan_raw)
assert job['source_sha256']==plan['runner_sha256']=='0096ebf620510074f4a38f398137a785521364b83adfcb74ea53b83b3f52e13e'
assert sha((run/'run_spf_checkpoint469_unsent395.py').read_bytes())==plan['runner_sha256']
assert plan['model']=='gpt-5.6-terra' and plan['reasoning']=='medium' and plan['max_output_tokens']==4096
assert plan['api_base_url']==job['api_base_url']=='https://api.openai.com/v1'
proof_raw=Path(plan['prior469_audit']).read_bytes();proof=json.loads(proof_raw)
assert sha(proof_raw)==plan['prior469_audit_sha256']=='56fc11c3ffd3d987db8a999db5acb2d3c3804083e965d909ab827dc823c2d08b'
assert proof['verified_successes']==396 and proof['uncertain_packet_indices']==[396] and proof['verified_unsent_indices']==indices
prior=Path(plan['prior469_audit']).parent;prior_raw=(prior/'verified-labels.private.json').read_bytes()
assert sha(prior_raw)==proof['verified_labels_sha256'];prior_labels=json.loads(prior_raw)
source_raw=(prior/'verified-source-map.private.json').read_bytes();assert sha(source_raw)==proof['source_map_sha256']
prior_source_map=json.loads(source_raw)
old=budget/'checkpoint-469-saved-key-r2-append'
for name,key in [('job.json','job_sha256'),('judge-results.jsonl','results_sha256'),('execution-events.private.jsonl','events_sha256')]:assert sha((old/name).read_bytes())==proof[key]
allocation=json.loads((budget/'recoveries/469-r3-unsent395.json').read_bytes())
assert allocation['plan_sha256']==sha(plan_raw) and allocation['remaining_request_limit']==395 and allocation['reserved_uncertain']==1 and allocation['reuses_original_reservation']==792
packet_raw=(run/'private-packet.json').read_bytes();packets=json.loads(packet_raw)
assert sha(packet_raw)==proof['packet_sha256']==plan['packet_sha256']==job['packet_sha256']
snapshots=sorted((run/'state-snapshots').glob('*.json'));state_hashes=[]
assert [f.name for f in snapshots]==[f'{i:06}.json' for i in range(1,793)]
for sequence,path in enumerate(snapshots,1):
 raw=path.read_bytes();v=json.loads(raw);state=v['job']
 assert v['sequence']==sequence and state['plan_sha256']==sha(plan_raw) and state['pid']==job['pid']
 if sequence==1:assert state['attempted']==state['completed']==0
 elif sequence==792:assert state==job
 else:
  offset=sequence-2;relative=offset//2
  assert state['current_packet_index']==indices[relative] and state['attempted']==relative+1 and state['completed']==relative+(offset%2)
 state_hashes.append(dict(sequence=sequence,sha256=sha(raw)))
types = {
    'harmful': ('src/evals/safety/gpt_judge.py', 'JudgeResult', 'ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a', 'hexphi-two-dimensional-v1'),
    'benign': ('src/construction/development_judge.py', 'BenignJudgment', 'beb220e096e8aff1e60c15036fbf3d243b80fab32b28e35c43cea912f55c7d47', 'benign-development-v1'),
    'conversation': ('src/construction/conversation_judge.py', 'ConversationJudgment', 'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c', 'wildchat-development-quality-v1'),
}
schemas = {}
for kind, (name, cls, digest, _) in types.items():
    path = run / 'rubrics' / name
    assert sha(path.read_bytes()) == digest
    spec = importlib.util.spec_from_file_location('audit_checkpoint_' + kind, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    schemas[kind] = getattr(m, cls)
    assert sha(m.SYSTEM_PROMPT.encode()) == plan['prompt_hashes'][kind]
result_raw = (run / 'judge-results.jsonl').read_bytes()
rows = [json.loads(line) for line in result_raw.splitlines()]
event_raw = (run / 'execution-events.private.jsonl').read_bytes()
events = [json.loads(line) for line in event_raw.splitlines()]
assert len(rows) == count and len(events) == count*3
assert {path.name for path in (run / 'raw-responses').glob('*.json')} == {f'{i:04}.json' for i in indices}
counts, usage, models = Counter(), Counter(), Counter()
labels, score_sums, score_counts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
response_ids, source_map = set(), []
for relative, (index, row) in enumerate(zip(indices, rows)):
    packet=packets[index]
    kind = packet['kind']
    assert row['packet_index'] == index and row['status'] == 'success' and row['attempt'] == 1
    assert all(row[field] == packet[field] for field in ('id', 'blind_id', 'kind'))
    assert row['requested_model'] == row['actual_model'] == plan['model']
    assert row['prompt_sha256'] == plan['prompt_hashes'][kind] and row['prompt_version'] == types[kind][3]
    payload = json.dumps(packet['messages'], ensure_ascii=False)
    if kind == 'harmful':
        payload = 'HARMFUL USER INSTRUCTION:\n' + packet['messages'][0]['content'] + '\n\nMODEL RESPONSE:\n' + packet['messages'][1]['content']
    assert sha(payload.encode()) == row['request_payload_sha256']
    parsed = schemas[kind].model_validate(row['judgment']).model_dump()
    assert parsed == row['judgment']
    raw = (run / 'raw-responses' / f'{index:04}.json').read_bytes()
    assert sha(raw) == row['raw_response_sha256']
    response = json.loads(raw)
    assert response['model'] == row['actual_model'] and response['id'] == row['response_id']
    assert response['status'] == 'completed' and response['usage'] == row['usage']
    assert response['max_output_tokens'] == 4096 and response['reasoning']['effort'] == 'medium'
    texts = [c['text'] for item in response['output'] for c in item.get('content', []) if c.get('type') == 'output_text']
    assert len(texts) == 1 and json.loads(texts[0]) == parsed
    assert row['response_id'] not in response_ids
    response_ids.add(row['response_id'])
    triplet = events[relative * 3:relative * 3 + 3]
    assert [e['operation'] for e in triplet] == ['intent_durable_before_api', 'api_request_started', 'api_response_received']
    assert all(e['packet_index'] == index for e in triplet)
    assert triplet[0]['request_payload_sha256'] == row['request_payload_sha256']
    assert triplet[2]['response_id'] == row['response_id'] and triplet[2]['actual_model'] == plan['model']
    assert triplet[0]['at'] <= triplet[1]['at'] <= triplet[2]['at']
    counts[kind] += 1
    models[row['actual_model']] += 1
    usage.update(input_tokens=row['usage']['input_tokens'], output_tokens=row['usage']['output_tokens'])
    for field, value in parsed.items():
        if isinstance(value, str) and not field.endswith('reason') and field not in ('reason', 'rationale'):
            labels[kind + ':' + field][value] += 1
        if kind == 'conversation' and isinstance(value, int) and not isinstance(value, bool):
            score_sums[kind][field] += value
            score_counts[kind][field] += 1
    source_map.append(dict(packet_index=index, id=row['id'], kind=kind, raw_sha256=sha(raw), payload_sha256=row['request_payload_sha256']))
assert dict(usage)==job['usage'] and dict(models)==job['actual_models']
assert (run/'job.json').read_bytes()==job_raw and (run/'judge-results.jsonl').read_bytes()==result_raw
assert (run/'execution-events.private.jsonl').read_bytes()==event_raw
new_usage=dict(usage);rows=prior_labels+rows;source_map=prior_source_map+source_map
expected=list(range(396))+list(range(397,792))
assert [r['packet_index'] for r in rows]==expected and len(rows)==len(source_map)==791 and len({r['response_id'] for r in rows})==791
counts,usage,models=Counter(),Counter(),Counter();labels=defaultdict(Counter)
for index,row in zip(expected,rows):
 packet=packets[index];assert row['status']=='success' and all(row[k]==packet[k] for k in ('id','blind_id','kind'))
 kind=packet['kind'];counts[kind]+=1;models[row['actual_model']]+=1
 usage.update(input_tokens=row['usage']['input_tokens'],output_tokens=row['usage']['output_tokens'])
 for field,value in row['judgment'].items():
  if isinstance(value,str) and not field.endswith('reason') and field not in ('reason','rationale'):labels[kind+':'+field][value]+=1
assert dict(counts)=={'benign':350,'conversation':142,'harmful':299} and dict(models)=={'gpt-5.6-terra':791}
out.mkdir(exist_ok=False)
(out/'verified-labels.private.json').write_bytes(dump(rows));(out/'verified-source-map.private.json').write_bytes(dump(source_map))
(out/'state-snapshot-hashes.private.json').write_bytes(dump(state_hashes))
h=labels['harmful:safety_label']['harmful_assistance']
audit=dict(status='CHECKPOINT469791_CONFIRMED_UNCERTAIN_HARMFUL1',checkpoint_step=469,observed_at_utc=datetime.now(timezone.utc).isoformat(),
 verified_successes=791,new_raw_responses_verified=395,prior_verified_labels_reused=396,prior_raw_reaudited=False,
 missing=1,uncertain_packet_indices=[396],uncertain_kind='harmful',full792_complete=False,
 kind_counts=dict(counts),actual_models=dict(models),token_usage=dict(usage),new_token_usage=new_usage,
 labels={k:dict(v) for k,v in labels.items()},harmfulness=dict(confirmed_harmful_assistance=h,verified_denominator=299,descriptive_rate=h/299,
 full300_rate_bounds=[h/300,(h+1)/300],missing_label_imputed=False),
 plan_sha256=sha(plan_raw),packet_sha256=sha(packet_raw),runner_sha256=plan['runner_sha256'],prior396_audit_sha256=sha(proof_raw),
 job_sha256=sha(job_raw),results_sha256=sha(result_raw),events_sha256=sha(event_raw),
 verified_labels_sha256=sha((out/'verified-labels.private.json').read_bytes()),source_map_sha256=sha((out/'verified-source-map.private.json').read_bytes()),
 auditor_sha256=sha(Path(__file__).read_bytes()),state_snapshot_count=len(snapshots),retries=0,paid_cap=2376,
 new_api_calls=0,credential_reads=0,target_gate='insufficient_evidence',human_adjudication_complete=False,cluster_mapping_verified=False)
(out/'audit.private.json').write_bytes(dump(audit));(out/'audit.as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(audit))
