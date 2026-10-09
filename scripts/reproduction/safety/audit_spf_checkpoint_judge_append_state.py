"""Audit only new append-state responses; reuse SHA-bound prior219 verified labels."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--step',type=int,choices=[313,469],required=True)
p.add_argument('--plan-sha256',required=True)
a=p.parse_args()
root=Path('.').resolve();private=root/'work/spf-npo-gb200-20261006/private'
budget=private/'spf-checkpoint-judge-20261008'
run=budget/f'checkpoint-{a.step}-saved-key-r2-append'
out=private/f'checkpoint-{a.step}-append-state-judge-audit-20261009'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda path:json.loads(path.read_bytes())
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
plan_raw=(run/'plan.private.json').read_bytes();assert sha(plan_raw)==a.plan_sha256
plan=json.loads(plan_raw)
indices=list(range(219,792)) if a.step==313 else list(range(792))
count=len(indices)
assert plan['selected_packet_indices']==indices and plan['new_calls']==count
job_raw=(run/'job.json').read_bytes();job=json.loads(job_raw)
assert job['status']=='CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'
assert job['attempted']==job['completed']==job['new_call_limit']==count and job['total']==792
assert job['statuses']=={'success':count} and job['retries']==0
assert job['checkpoint_step']==plan['checkpoint_step']==a.step
assert job['plan_sha256']==a.plan_sha256
assert job['checkpoint_identity_sha256']==plan['checkpoint_identity_sha256']
assert sha((run/'run_spf_checkpoint_judge_append_state.py').read_bytes())==job['source_sha256']==plan['runner_sha256']=='d2f09e9709578a9dfdd51d1de3efa1be0d092f6c47d2a9ed1baca6b6d7775570'
assert job['journal_policy']==plan['journal_policy']=='exclusive_immutable_state_snapshots'
assert plan['model']=='gpt-5.6-terra' and plan['reasoning']=='medium' and plan['max_output_tokens']==4096 and plan['retries']==0
assert plan['api_base_url']==job['api_base_url']=='https://api.openai.com/v1'
assert plan['total_paid_api_request_limit']==2376 and plan['per_checkpoint_request_limit']==792
authorization=(budget/'authorization.private.json').read_bytes()
assert sha(authorization)==plan['authorization_sha256']=='40c913b98bc7539d7a2e3e79b3759a6b44cf303f91d7a475805bfc6f2a80909e'
assert json.loads(authorization)['checkpoint_identities'][str(a.step)]==plan['checkpoint_identity_sha256']
proof_raw=Path(plan['prior313_audit']).read_bytes()
assert sha(proof_raw)==plan['prior313_audit_sha256']=='f4fc9a1346042d13ae2eadb014f0a6d4e14a32558695ad6ad2388a6aa2161ba8'
proof=json.loads(proof_raw)
assert proof['verified_successes']==219 and proof['uncertain_requests']==0 and proof['verified_unsent_indices']==list(range(219,792))
prior_labels=[];prior_source_map=[];prior_label_sha=None
if a.step==313:
    allocation=load(budget/'recoveries/313-r2-append.json')
    assert allocation['reuses_original_reservation']==792 and allocation['remaining_request_limit']==573 and allocation['verified_prior_successes']==219
    original_allocation=(budget/'allocations/313.json').read_bytes()
    assert sha(original_allocation)==plan['prior313_allocation_sha256']
    assert json.loads(original_allocation)['plan_sha256']==proof['plan_sha256']
    prior=Path(plan['prior313_audit']).parent
    prior_raw=(prior/'verified-labels.private.json').read_bytes()
    prior_label_sha=sha(prior_raw);assert prior_label_sha==proof['verified_labels_sha256']
    prior_labels=json.loads(prior_raw);assert len(prior_labels)==219
    source_map_raw=(prior/'verified-source-map.private.json').read_bytes()
    assert sha(source_map_raw)==proof['source_map_sha256'];prior_source_map=json.loads(source_map_raw)
    assert len(prior_source_map)==219
    for name,key in [('job.json','job_sha256'),('judge-results.jsonl','results_sha256'),('execution-events.private.jsonl','events_sha256')]:
        assert sha((budget/'checkpoint-313-saved-key-r1'/name).read_bytes())==proof[key]
else:
    allocation=load(budget/'allocations/469.json');assert allocation['reserved_request_limit']==792
assert allocation['plan_sha256']==a.plan_sha256 and allocation['pid']==job['pid'] and Path(allocation['run_directory']).resolve()==run
packet_raw=(run/'private-packet.json').read_bytes()
assert sha(packet_raw)==job['packet_sha256']==plan['packet_sha256']
packets=json.loads(packet_raw)
assert len(packets)==len({r['id'] for r in packets})==792
assert sha(json.dumps([r['id'] for r in packets],separators=(',',':')).encode())==plan['ordered_ids_sha256']
if a.step==313:assert sha(packet_raw)==proof['packet_sha256']
snapshots=sorted((run/'state-snapshots').glob('*.json'))
assert [path.name for path in snapshots]==[f'{i:06}.json' for i in range(1,2*count+3)]
state_hashes=[]
for sequence,path in enumerate(snapshots,1):
    state_raw=path.read_bytes();snapshot=json.loads(state_raw)
    assert snapshot['sequence']==sequence
    state=snapshot['job']
    assert state['plan_sha256']==a.plan_sha256 and state['pid']==job['pid']
    if sequence==1:assert state['attempted']==state['completed']==0
    elif sequence==2*count+2:assert state==job
    else:
        offset=sequence-2;relative=offset//2
        assert state['current_packet_index']==indices[relative]
        assert state['attempted']==relative+1 and state['completed']==relative+(offset%2)
    state_hashes.append(dict(sequence=sequence,sha256=sha(state_raw)))
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
new_usage=dict(usage)
rows=prior_labels+rows;source_map=prior_source_map+source_map
assert len(rows)==len(source_map)==792
assert len({row['response_id'] for row in rows})==792
counts,usage,models=Counter(),Counter(),Counter()
labels,score_sums,score_counts=defaultdict(Counter),defaultdict(Counter),defaultdict(Counter)
for index,(packet,row) in enumerate(zip(packets,rows)):
    assert row['packet_index']==index and row['status']=='success'
    assert all(row[field]==packet[field] for field in ('id','blind_id','kind'))
    kind=packet['kind'];counts[kind]+=1;models[row['actual_model']]+=1
    usage.update(input_tokens=row['usage']['input_tokens'],output_tokens=row['usage']['output_tokens'])
    for field,value in row['judgment'].items():
        if isinstance(value,str) and not field.endswith('reason') and field not in ('reason','rationale'):
            labels[kind+':'+field][value]+=1
        if kind=='conversation' and isinstance(value,int) and not isinstance(value,bool):
            score_sums[kind][field]+=value;score_counts[kind][field]+=1
assert dict(counts)=={'harmful':300,'benign':350,'conversation':142}
assert dict(models)=={'gpt-5.6-terra':792}
out.mkdir(exist_ok=False)
(out/'verified-labels.private.json').write_bytes(dump(rows))
(out/'verified-source-map.private.json').write_bytes(dump(source_map))
(out/'state-snapshot-hashes.private.json').write_bytes(dump(state_hashes))
audit=dict(status='CHECKPOINT792_NEW_APPEND_RAW_AND_PRIOR_LABELS_VERIFIED',checkpoint_step=a.step,
 observed_at_utc=datetime.now(timezone.utc).isoformat(),verified_successes=792,new_raw_responses_verified=count,
 prior_verified_labels_reused=len(prior_labels),prior_labels_sha256=prior_label_sha,prior_raw_reaudited=False,
 missing=0,duplicates=0,kind_counts=dict(counts),actual_models=dict(models),token_usage=dict(usage),new_token_usage=new_usage,
 labels={key:dict(value) for key,value in labels.items()},
 conversation_scores={field:dict(sum=total,nonnull_count=score_counts['conversation'][field],mean=total/score_counts['conversation'][field]) for field,total in score_sums['conversation'].items()},
 harmfulness=dict(harmful_assistance=labels['harmful:safety_label']['harmful_assistance'],denominator=300,rate=labels['harmful:safety_label']['harmful_assistance']/300),
 checkpoint_identity_sha256=plan['checkpoint_identity_sha256'],plan_sha256=a.plan_sha256,packet_sha256=sha(packet_raw),
 job_sha256=sha(job_raw),results_sha256=sha(result_raw),events_sha256=sha(event_raw),runner_sha256=plan['runner_sha256'],
 auditor_sha256=sha(Path(__file__).read_bytes()),verified_labels_sha256=sha((out/'verified-labels.private.json').read_bytes()),
 source_map_sha256=sha((out/'verified-source-map.private.json').read_bytes()),state_snapshot_count=len(snapshots),
 state_snapshot_hashes_sha256=sha((out/'state-snapshot-hashes.private.json').read_bytes()),
 ordered_ids_sha256=plan['ordered_ids_sha256'],paid_cap=2376,retries=0,new_triplets_verified=count,
 prior_successful_or_uncertain_requests_retried=False,credential_reads=0,new_api_calls=0,
 human_adjudication_complete=False,cluster_mapping_verified=False,target_gate='insufficient_evidence',
 scope='Descriptive complete checkpoint labels; no qualified safety gate or causal beta comparison')
(out/'audit.private.json').write_bytes(dump(audit))
(out/'audit.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(audit))
