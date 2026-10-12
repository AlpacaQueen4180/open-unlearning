from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,importlib.util,io,json,sqlite3,sys,tarfile
root=Path('.').resolve();w=root/'work/spf-npo-gb200-20261006';p=w/'private/spf-safety-search-20261010';run=p/'model-06-hybrid01-rank40-full-r1'
sha=lambda raw:hashlib.sha256(raw).hexdigest();dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
def module(path,digest,name):
    assert sha(path.read_bytes())==digest
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result
audit_source=root/'scripts/reproduction/safety/audit_spf_safety_search_judge.py'
auditor=module(audit_source,'5913c335f9503c093d916bd033d4f9737b071c456ff6e6dbabfc58d7ea7cacb0','rank40_partial_audit_helpers')
exitraw=(p/'model06-old-process-exit-20261013-0051.private.json').read_bytes();exits=json.loads(exitraw)
assert exits['own_processes']==[{'pid':65544,'exited':True},{'pid':59696,'exited':True}]
planraw=(run/'plan.private.json').read_bytes();plan=json.loads(planraw)
assert sha(planraw)=='a6b33c4b0c0c58ccc298902b798eb30a33ac19dc7cb2ea895cd73efb55fab66c'
assert plan['judge_profile']==auditor.PROFILE and plan['model_identity_sha256']=='64d3fd783130bc4936a97cbc31da044dfd5e20487b35f2e71a21f6e20ad30c78'
packetraw=(run/'private-packet.json').read_bytes();packets=json.loads(packetraw);assert sha(packetraw)==plan['packet_sha256']
jobraw=(run/'job.json').read_bytes();job=json.loads(jobraw)
assert job['status']=='STOPPED_ON_EXECUTION_ERROR_NO_RETRY' and job['attempted']==4 and job['completed']==3 and job['current_packet_index']==3
assert job['failure']['error_type']=='BadRequestError' and job['failure']['operation']=='api_request_started'
assert job['retries']==0 and job['statuses']=={'success':3} and job['actual_models']=={'gpt-5.6-terra':3}
result_raw=(run/'judge-results.jsonl').read_bytes();rows=[json.loads(line) for line in result_raw.splitlines()]
event_raw=(run/'execution-events.private.jsonl').read_bytes();events=[json.loads(line) for line in event_raw.splitlines()]
assert len(rows)==3 and len(events)==12
assert [e['operation'] for e in events[-3:]]==['intent_durable_before_api','api_request_started','execution_stopped']
assert [events[-3]['packet_index'],events[-2]['packet_index']]==[3,3]
assert {f.name for f in (run/'raw-responses').glob('*.json')}=={'0000.json','0001.json','0002.json'}
with sqlite3.connect((p/'budget.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    allocation=dict(db.execute('SELECT * FROM slots WHERE slot=6').fetchone())
    intents=[dict(r) for r in db.execute('SELECT * FROM intents WHERE slot=6 ORDER BY idx')]
    receipts=[dict(r) for r in db.execute('SELECT * FROM receipts WHERE slot=6 ORDER BY idx')]
    saved_packets=[dict(r) for r in db.execute('SELECT * FROM packets WHERE slot=6 ORDER BY idx')]
assert allocation['status']=='RUNNING' and allocation['plan_sha']==sha(planraw) and allocation['packet_sha']==sha(packetraw)
assert len(intents)==4 and len(receipts)==3 and len(saved_packets)==792
assert [r['idx'] for r in intents]==list(range(4)) and [r['idx'] for r in receipts]==list(range(3))
schemas={}
for kind,(name,cls,digest,_) in auditor.TYPES.items():
    loaded=module(run/'rubrics'/name,digest,'rank40_first3_'+kind);schemas[kind]=getattr(loaded,cls)
    assert sha(loaded.SYSTEM_PROMPT.encode())==plan['prompt_hashes'][kind]
counts,usage=Counter(),Counter();raw_hashes={};seen=set()
for index,row in enumerate(rows):
    packet=packets[index];kind=packet['kind'];intent=intents[index];receipt=receipts[index];stored=saved_packets[index]
    assert row['packet_index']==intent['idx']==receipt['idx']==stored['idx']==index
    assert row['status']=='success' and row['attempt']==1
    assert all(row[k]==packet[k] for k in ('id','blind_id','kind'))
    assert row['requested_model']==row['actual_model']==receipt['actual_model']==auditor.PROFILE['model']
    assert row['prompt_sha256']==plan['prompt_hashes'][kind] and row['prompt_version']==auditor.TYPES[kind][3]
    payload=json.dumps(packet['messages'],ensure_ascii=False)
    if kind=='harmful':payload='HARMFUL USER INSTRUCTION:\n'+packet['messages'][0]['content']+'\n\nMODEL RESPONSE:\n'+packet['messages'][1]['content']
    assert sha(payload.encode())==row['request_payload_sha256']==intent['payload_sha']==stored['payload_sha']
    parsed=schemas[kind].model_validate(row['judgment']).model_dump();assert parsed==row['judgment']
    raw=(run/'raw-responses'/f'{index:04}.json').read_bytes();response=json.loads(raw)
    assert sha(raw)==row['raw_response_sha256']==receipt['raw_sha']
    assert response['model']==row['actual_model'] and response['id']==row['response_id'] and response['usage']==row['usage'] and response['status']=='completed'
    assert response['max_output_tokens']==4096 and response['reasoning']['effort']=='medium'
    texts=[c['text'] for item in response['output'] for c in item.get('content',[]) if c.get('type')=='output_text']
    assert len(texts)==1 and json.loads(texts[0])==parsed
    assert row['response_id'] not in seen and sha(row['response_id'].encode())==receipt['response_id_sha'];seen.add(row['response_id'])
    triplet=events[index*3:index*3+3];assert [e['operation'] for e in triplet]==['intent_durable_before_api','api_request_started','api_response_received']
    assert all(e['packet_index']==index for e in triplet)
    counts[kind]+=1;usage.update(input_tokens=row['usage']['input_tokens'],output_tokens=row['usage']['output_tokens']);raw_hashes[str(index)]=sha(raw)
assert dict(usage)==job['usage']
out=p/'model06-initial-failure-audit-20261013-0051';out.mkdir(exist_ok=False)
selected=[path for path in run.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
records=[];stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for path in sorted(selected):
        assert path.resolve().is_relative_to(run) and not path.is_symlink() and path.suffix not in ('.dpapi','.safetensors')
        raw=path.read_bytes();assert len(raw)<2000000
        name=path.relative_to(run).as_posix();records.append(dict(name=name,size=len(raw),sha256=sha(raw)))
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
archiveraw=stream.getvalue()
(out/'original-failure.private.tar.gz').write_bytes(archiveraw)
(out/'manifest.private.json').write_bytes(dump(dict(archive_sha256=sha(archiveraw),files=records,all_original_bytes_preserved=True,
    original_stdout_stderr_stream_files_preserved=True,credential_value_files_in_archive=False)))
(out/'verified-labels.private.json').write_bytes(dump(rows))
evidence=dict(status='MODEL06_FIRST3_RAW_SCHEMA_PAYLOAD_EVENTS_LEDGER_VERIFIED_WITH_ONE_API_FAILURE',recorded_utc=datetime.now(timezone.utc).isoformat(),
    campaign_slot=6,model_identity_sha256=plan['model_identity_sha256'],packet_sha256=sha(packetraw),plan_sha256=sha(planraw),
    job_sha256=sha(jobraw),results_sha256=sha(result_raw),events_sha256=sha(event_raw),
    verified_successes=3,verified_counts=dict(counts),token_usage=dict(usage),verified_raw_sha256_by_index=raw_hashes,
    verified_labels_sha256=sha((out/'verified-labels.private.json').read_bytes()),
    uncertain_packet_indices=[3],uncertain_kind=packets[3]['kind'],api_attempted_or_billing_not_reclassified_as_zero=True,
    verified_unsent_indices=list(range(4,792)),source_and_events_prove_sequential_unsent_boundary=True,
    source=job['source_sha256'],failure=job['failure'],original_api_error_detail_not_recorded=True,
    model_and_profile_unchanged=True,no_successful_or_sent_uncertain_retries=True,own_old_processes_exited=True,
    exit_evidence_sha256=sha(exitraw),archive_sha256=sha(archiveraw),archive_files=len(records),
    original_per_model_cap=792,confirmed3_plus_uncertain1_plus_unsent788=792,
    new_api_calls=0,credential_reads=0,prior5_paid_model_records_reaudited=False)
(out/'audit.private.json').write_bytes(dump(evidence));(out/'audit.as-run.py').write_bytes(Path(__file__).read_bytes())
budget=module(run/'spf_safety_search_budget.py',auditor.BUDGET_SHA,'rank40_initial_failure_budget')
closed=budget.Campaign(p).close_incomplete(6,sha((out/'audit.private.json').read_bytes()))
(out/'ledger-close-incomplete.private.json').write_bytes(dump(closed))
print(json.dumps(dict(status=evidence['status'],audit_sha256=sha((out/'audit.private.json').read_bytes()),archive_sha256=sha(archiveraw),
    successful_new_results=3,uncertain_index=3,uncertain_kind=evidence['uncertain_kind'],verified_unsent=788,
    total_model_reservations=closed['reserved_models'],complete_models=closed['complete_models'],incomplete_models=closed['incomplete_models'],new_api_calls=0)))
