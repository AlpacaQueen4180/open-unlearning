from pathlib import Path
import copy,hashlib,json,subprocess
root=Path('.').resolve();p=root/'work/spf-npo-gb200-20261006/private'
out=p/'append-auditor-queue-checks-20261009-r2';out.mkdir(exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
workspace=Path(__import__('tempfile').gettempdir())/'spf-aa-20261009-r2';fp=workspace/'work/spf-npo-gb200-20261006/private';budget=fp/'spf-checkpoint-judge-20261008'
run=budget/'checkpoint-313-saved-key-r2-append';run.mkdir(parents=True)
actual= p/'spf-checkpoint-judge-20261008/checkpoint-313-saved-key-r2-append'
for name in ['private-packet.json','run_spf_checkpoint_judge_append_state.py']:
 (run/name).write_bytes((actual/name).read_bytes())
for source in (actual/'rubrics').rglob('*.py'):
 target=run/source.relative_to(actual);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
proof_dir=fp/'checkpoint-313-partial219-audit-20261009';proof_dir.mkdir()
for name in ['audit.private.json','verified-labels.private.json','verified-source-map.private.json']:
 (proof_dir/name).write_bytes((p/'checkpoint-313-partial219-audit-20261009'/name).read_bytes())
for name in ['authorization.private.json','allocations/313.json','checkpoint-313-saved-key-r1/job.json','checkpoint-313-saved-key-r1/judge-results.jsonl','checkpoint-313-saved-key-r1/execution-events.private.jsonl']:
 target=budget/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((p/'spf-checkpoint-judge-20261008'/name).read_bytes())
plan=json.loads((actual/'plan.private.json').read_bytes());plan['prior313_audit']=str(proof_dir/'audit.private.json')
plan_raw=dump(plan);plan_sha=sha(plan_raw);(run/'plan.private.json').write_bytes(plan_raw)
mock=p/'append-state-judge-checks-20261009/complete573'
(run/'raw-responses').mkdir();rows=[json.loads(line) for line in (mock/'judge-results.jsonl').read_bytes().splitlines()]
for row in rows:
 name=f"{row['packet_index']:04}.json";response=json.loads((mock/'raw-responses'/name).read_bytes())
 # Only synthetic fixtures supply fields absent from the mock HTTP envelope.
 response['max_output_tokens']=4096;response['reasoning']={'effort':'medium'}
 raw=dump(response);(run/'raw-responses'/name).write_bytes(raw);row['raw_response_sha256']=sha(raw)
(run/'judge-results.jsonl').write_bytes(b''.join((json.dumps(row,ensure_ascii=False)+'\n').encode() for row in rows))
(run/'execution-events.private.jsonl').write_bytes((mock/'execution-events.private.jsonl').read_bytes())
metadata=dict(total=792,new_call_limit=573,retries=0,pid=12345,checkpoint_step=313,checkpoint_identity_sha256=plan['checkpoint_identity_sha256'],plan_sha256=plan_sha,source_sha256=plan['runner_sha256'],packet_sha256=plan['packet_sha256'],journal_policy=plan['journal_policy'],api_base_url=plan['api_base_url'])
states=[dict(attempted=0,completed=0,**metadata)]
for path in sorted((mock/'state-snapshots').glob('*.json')):
 state=json.loads(path.read_bytes())['job'];state.update(metadata);states.append(state)
(run/'state-snapshots').mkdir()
for sequence,state in enumerate(states,1):(run/'state-snapshots'/f'{sequence:06}.json').write_bytes(dump(dict(sequence=sequence,job=state)))
(run/'job.json').write_bytes(dump(states[-1]))
(budget/'recoveries').mkdir();(budget/'recoveries/313-r2-append.json').write_bytes(dump(dict(reuses_original_reservation=792,remaining_request_limit=573,verified_prior_successes=219,plan_sha256=plan_sha,pid=12345,run_directory=str(run))))
source=root/'scripts/reproduction/safety/audit_spf_checkpoint_judge_append_state.py'
python=p/'local-judge-venv/Scripts/python.exe'
result=subprocess.run([str(python),'-X','utf8',str(source),'--step','313','--plan-sha256',plan_sha],cwd=workspace,capture_output=True)
(out/'synthetic-audit.stdout.private.txt').write_bytes(result.stdout);(out/'synthetic-audit.stderr.private.txt').write_bytes(result.stderr)
assert result.returncode==0,'Synthetic combined audit rejected; inspect preserved streams'
audit=json.loads((fp/'checkpoint-313-append-state-judge-audit-20261009/audit.private.json').read_bytes())
assert audit['new_raw_responses_verified']==573 and audit['prior_verified_labels_reused']==219 and not audit['prior_raw_reaudited'] and audit['verified_successes']==792
prefix=source.read_text().split('types = {')[0];old_args=__import__('sys').argv
import os,sys
old_cwd=Path.cwd();os.chdir(workspace)
try:
 sys.argv=[str(source),'--step','313','--plan-sha256',plan_sha]
 changed=run/'state-snapshots/000002.json';original=changed.read_bytes();bad=json.loads(original);bad['job']['completed']=1;changed.write_bytes(dump(bad))
 try:exec(compile(prefix,str(source),'exec'),{})
 except AssertionError:pass
 else:raise AssertionError('Invalid intent-state completion must reject')
 finally:changed.write_bytes(original)
finally:sys.argv=old_args;os.chdir(old_cwd)
queue=p/'checkpoint313469-append-judge-queue-20261009'
config=json.loads((queue/'config.private.json').read_bytes());qsource=queue/'run_spf_checkpoint_judge_append_queue.ps1'
negative=[]
for name in ['cap2377','remaining1366','wrong313count','retry1','reversed_steps']:
 fixture=out/name;fixture.mkdir();bad=copy.deepcopy(config)
 if name=='cap2377':bad['total_authorized_cap']=2377
 if name=='remaining1366':bad['remaining_new_request_cap']=1366
 if name=='wrong313count':bad['cases'][0]['new_calls']=574
 if name=='retry1':bad['retries']=1
 if name=='reversed_steps':bad['cases'].reverse()
 raw=dump(bad);(fixture/'config.private.json').write_bytes(raw)
 result=subprocess.run([config['pwsh_executable'],'-NoProfile','-File',str(qsource),'-QueueDirectory',str(fixture),'-SourceSha256',sha(qsource.read_bytes()),'-ConfigSha256',sha(raw),'-Preflight'],capture_output=True)
 (fixture/'stdout.private.txt').write_bytes(result.stdout);(fixture/'stderr.private.txt').write_bytes(result.stderr)
 assert result.returncode!=0 and b'Only verified573 unsent313' in result.stderr
 assert not (fixture/'state-snapshots').exists();negative.append(name)
parser_code="$taskErrors=@();foreach($taskFile in $args){$taskTokens=$null;$taskErrorsOne=$null;[Management.Automation.Language.Parser]::ParseFile($taskFile,[ref]$taskTokens,[ref]$taskErrorsOne)|Out-Null;$taskErrors+=@($taskErrorsOne)};if($taskErrors.Count -ne 0){exit 1};'{\"error_count\":0}'"
parsed=subprocess.run([config['pwsh_executable'],'-NoProfile','-Command',parser_code,str(qsource),str(root/'work/spf-npo-gb200-20261006/launch_append_judge_queue_20261009.ps1')],capture_output=True)
(out/'parser.stdout.private.txt').write_bytes(parsed.stdout);(out/'parser.stderr.private.txt').write_bytes(parsed.stderr)
assert parsed.returncode==0
checks=dict(status='NEW573_SYNTHETIC_COMBINED_AUDIT_AND_QUEUE_GUARDS_PASS',auditor_sha256=sha(source.read_bytes()),queue_sha256=sha(qsource.read_bytes()),new_synthetic_raw_verified=573,prior219_verified_labels_sha256=plan['prior313_audit_sha256'],prior_raw_reaudited=False,state_negative_rejected=True,queue_negatives_rejected=negative,powershell_parser_error_count=0,new_api_calls=0,credential_reads=0,completed573_mock_http_loop_repeated=False,actual_paid_completion=False)
(out/'checks.private.json').write_bytes(dump(checks));(out/'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(checks))
