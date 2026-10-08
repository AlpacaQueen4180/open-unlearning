"""Audit one new checkpoint's completed792 responses; no API or credential access."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--step', type=int, choices=[157, 313, 469], required=True)
p.add_argument('--plan-sha256', required=True)
a = p.parse_args()
root = Path('.').resolve()
private = root / 'work/spf-npo-gb200-20261006/private'
budget = private / 'spf-checkpoint-judge-20261008'
run = budget / f'checkpoint-{a.step}-saved-key-r1'
out = private / f'checkpoint-{a.step}-judge-audit-20261009'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
load = lambda path: json.loads(path.read_bytes())
dump = lambda value: (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
plan_raw = (run / 'plan.private.json').read_bytes()
assert sha(plan_raw) == a.plan_sha256
plan = json.loads(plan_raw)
job_raw = (run / 'job.json').read_bytes()
job = json.loads(job_raw)
assert job['status'] == 'CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'
assert job['attempted'] == job['completed'] == job['total'] == 792
assert job['statuses'] == {'success': 792} and job['retries'] == 0
assert job['checkpoint_step'] == plan['checkpoint_step'] == a.step
assert job['plan_sha256'] == a.plan_sha256
assert job['checkpoint_identity_sha256'] == plan['checkpoint_identity_sha256']
assert sha((run / 'run_spf_checkpoint_development_judge_saved_key.py').read_bytes()) == job['source_sha256'] == plan['runner_sha256'] == '9e0dfb0244bf8f4ebd33620cef0436aef8c9b1524a3b755c0cc400f4db4e25c8'
assert plan['model'] == 'gpt-5.6-terra' and plan['reasoning'] == 'medium'
assert plan['max_output_tokens'] == 4096 and plan['retries'] == 0
assert plan['api_base_url'] == job['api_base_url'] == 'https://api.openai.com/v1'
assert plan['total_paid_api_request_limit'] == 2376 and plan['per_checkpoint_request_limit'] == 792
authorization = (budget / 'authorization.private.json').read_bytes()
assert sha(authorization) == plan['authorization_sha256'] == '40c913b98bc7539d7a2e3e79b3759a6b44cf303f91d7a475805bfc6f2a80909e'
assert json.loads(authorization)['checkpoint_identities'][str(a.step)] == plan['checkpoint_identity_sha256']
allocation = load(budget / 'allocations' / f'{a.step}.json')
assert allocation['plan_sha256'] == a.plan_sha256 and allocation['reserved_request_limit'] == 792
assert Path(allocation['run_directory']).resolve() == run and allocation['pid'] == job['pid']
packet_raw = (run / 'private-packet.json').read_bytes()
assert sha(packet_raw) == job['packet_sha256'] == plan['packet_sha256']
packets = json.loads(packet_raw)
assert len(packets) == len({r['id'] for r in packets}) == 792
assert sha(json.dumps([r['id'] for r in packets], separators=(',', ':')).encode()) == plan['ordered_ids_sha256']
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
assert len(rows) == 792 and len(events) == 2376
assert {path.name for path in (run / 'raw-responses').glob('*.json')} == {f'{i:04}.json' for i in range(792)}
counts, usage, models = Counter(), Counter(), Counter()
labels, score_sums, score_counts = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
response_ids, source_map = set(), []
for index, (packet, row) in enumerate(zip(packets, rows)):
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
    triplet = events[index * 3:index * 3 + 3]
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
assert dict(counts) == {'harmful': 300, 'benign': 350, 'conversation': 142}
assert dict(usage) == job['usage'] and dict(models) == job['actual_models']
assert (run / 'job.json').read_bytes() == job_raw and (run / 'judge-results.jsonl').read_bytes() == result_raw
out.mkdir(exist_ok=False)
(out / 'verified-labels.private.json').write_bytes(dump(rows))
(out / 'verified-source-map.private.json').write_bytes(dump(source_map))
audit = dict(status='NEW_CHECKPOINT792_RAW_SCHEMA_PAYLOAD_EVENTS_VERIFIED', checkpoint_step=a.step,
    observed_at_utc=datetime.now(timezone.utc).isoformat(), verified_successes=792, missing=0, duplicates=0,
    kind_counts=dict(counts), actual_models=dict(models), token_usage=dict(usage),
    labels={key: dict(value) for key, value in labels.items()},
    conversation_scores={field: dict(sum=total, nonnull_count=score_counts['conversation'][field], mean=total / score_counts['conversation'][field]) for field, total in score_sums['conversation'].items()},
    harmfulness=dict(harmful_assistance=labels['harmful:safety_label']['harmful_assistance'], denominator=300, rate=labels['harmful:safety_label']['harmful_assistance'] / 300),
    checkpoint_identity_sha256=plan['checkpoint_identity_sha256'], plan_sha256=a.plan_sha256,
    packet_sha256=sha(packet_raw), job_sha256=sha(job_raw), results_sha256=sha(result_raw), events_sha256=sha(event_raw),
    runner_sha256=plan['runner_sha256'], auditor_sha256=sha(Path(__file__).read_bytes()),
    verified_labels_sha256=sha((out / 'verified-labels.private.json').read_bytes()),
    source_map_sha256=sha((out / 'verified-source-map.private.json').read_bytes()),
    ordered_ids_sha256=plan['ordered_ids_sha256'], paid_cap=2376, retries=0,
    raw_schema_payload_model_usage_matches=792, durable_intent_response_event_triplets_verified=792,
    prior_successful_or_uncertain_requests_retried=False, credential_reads=0, new_api_calls=0,
    human_adjudication_complete=False, cluster_mapping_verified=False, target_gate='insufficient_evidence',
    scope='Descriptive checkpoint development results; no causal beta comparison or qualified safety gate')
(out / 'audit.private.json').write_bytes(dump(audit))
(out / 'audit.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(audit))
