from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys
root=Path('.').resolve();private=root/'work/spf-npo-gb200-20261006/private'
run=private/'spf-checkpoint-judge-20261008/checkpoint-469-saved-key-r2-append'
out=private/'checkpoint-469-partial396-audit-20261009'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda v:(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
plan_raw=(run/'plan.private.json').read_bytes();plan=json.loads(plan_raw)
assert sha(plan_raw)=='85a3e6dda08cf985d22ce39236647e35231ef9e2e396debdd1da7ea771f52be8'
assert plan['selected_packet_indices']==list(range(792)) and plan['new_calls']==792
job_raw=(run/'job.json').read_bytes();job=json.loads(job_raw)
assert job['status']=='STOPPED_ON_EXECUTION_ERROR_NO_RETRY' and job['attempted']==397 and job['completed']==396
assert job['statuses']=={'success':396} and job['retries']==0 and job['current_packet_index']==396
assert job['failure']['error_type']=='InternalServerError' and job['failure']['operation']=='api_request_started'
assert job['checkpoint_step']==plan['checkpoint_step']==469 and job['plan_sha256']==sha(plan_raw)
assert job['source_sha256']==plan['runner_sha256']=='d2f09e9709578a9dfdd51d1de3efa1be0d092f6c47d2a9ed1baca6b6d7775570'
assert sha((run/'run_spf_checkpoint_judge_append_state.py').read_bytes())==plan['runner_sha256']
assert plan['model']=='gpt-5.6-terra' and plan['reasoning']=='medium' and plan['max_output_tokens']==4096 and plan['retries']==0
assert plan['api_base_url']==job['api_base_url']=='https://api.openai.com/v1'
packet_raw=(run/'private-packet.json').read_bytes();packets=json.loads(packet_raw)
assert sha(packet_raw)==plan['packet_sha256']==job['packet_sha256']=='619cd38da0d44315b1c3173bc723833fc20659e49195b93fe861e54a040068a9'
assert len(packets)==792 and len({row['id'] for row in packets})==792
indices=list(range(396));count=396
snapshots=sorted((run/'state-snapshots').glob('*.json'));state_hashes=[]
assert [f.name for f in snapshots]==[f'{i:06}.json' for i in range(1,796)]
for sequence,path in enumerate(snapshots,1):
 raw=path.read_bytes();v=json.loads(raw);state=v['job']
 assert v['sequence']==sequence and state['plan_sha256']==sha(plan_raw) and state['pid']==job['pid']
 if sequence==1:assert state['attempted']==state['completed']==0
 elif sequence==795:assert state==job
 elif sequence==794:assert state['attempted']==397 and state['completed']==396 and state['current_packet_index']==396
 else:
  offset=sequence-2;relative=offset//2
  assert state['current_packet_index']==relative and state['attempted']==relative+1 and state['completed']==relative+(offset%2)
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
assert len(rows) == count and len(events) == count*3+3
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
tail=events[count*3:]
assert [x['operation'] for x in tail]==['intent_durable_before_api','api_request_started','execution_stopped']
assert tail[0]['packet_index']==tail[1]['packet_index']==396 and tail[2]['attempted']==397 and tail[2]['completed']==396
assert tail[2]['failure']==job['failure']
assert tail[0]['at']<=tail[1]['at']<=tail[2]['at']
assert not (run/'raw-responses/0396.json').exists()
assert not any(x.get('packet_index',-1)>396 for x in events)
out.mkdir(exist_ok=False)
(out/'verified-labels.private.json').write_bytes(dump(rows))
(out/'verified-source-map.private.json').write_bytes(dump(source_map))
(out/'state-snapshot-hashes.private.json').write_bytes(dump(state_hashes))
audit=dict(status='STOPPED469_NEW396_RAW_VERIFIED_UNCERTAIN1_UNSENT395',checkpoint_step=469,
 observed_at_utc=datetime.now(timezone.utc).isoformat(),verified_successes=396,uncertain_requests=1,
 uncertain_packet_indices=[396],uncertain_kind=packets[396]['kind'],verified_unsent_indices=list(range(397,792)),
 source_and_events_prove_sequential_unsent_boundary=True,failed_sent_request_retried=False,
 plan_sha256=sha(plan_raw),packet_sha256=sha(packet_raw),runner_sha256=plan['runner_sha256'],
 checkpoint_identity_sha256=plan['checkpoint_identity_sha256'],ordered_ids_sha256=plan['ordered_ids_sha256'],
 job_sha256=sha(job_raw),results_sha256=sha(result_raw),events_sha256=sha(event_raw),
 verified_labels_sha256=sha((out/'verified-labels.private.json').read_bytes()),
 source_map_sha256=sha((out/'verified-source-map.private.json').read_bytes()),
 state_snapshot_count=len(snapshots),state_snapshot_hashes_sha256=sha((out/'state-snapshot-hashes.private.json').read_bytes()),
 kind_counts=dict(counts),actual_models=dict(models),token_usage=dict(usage),labels={k:dict(v) for k,v in labels.items()},
 failure=job['failure'],retries=0,real_api_calls=0,credential_reads=0,target_gate='insufficient_evidence',
 human_adjudication_complete=False,cluster_mapping_verified=False)
(out/'audit.private.json').write_bytes(dump(audit));(out/'audit.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(audit))
