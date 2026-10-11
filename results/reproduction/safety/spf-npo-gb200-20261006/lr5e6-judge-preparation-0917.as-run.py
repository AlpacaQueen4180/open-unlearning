"""Prepare one new campaign packet from the completed beta05 LR5e-6 outputs; no key or API access."""
from pathlib import Path
from collections import Counter
import ast, base64, csv, hashlib, io, json, os, subprocess, tarfile

root = Path('.').resolve()
w = root / 'work/spf-npo-gb200-20261006'
p = w / 'private'
campaign = p / 'spf-safety-search-20261010'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda path: json.loads(path.read_bytes())
dump = lambda value: (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
identity = '9b83962184922d53dd122aa65394aed92fd6c4a121eff293595229b239946ab7'
export_path = campaign / 'lr5e6-completed-export-20261011-0917.private.json'
export = read(export_path)
assert export['status'] == 'IMMUTABLE_LR5E6_COMPLETED_TRAIN_DEVELOPMENT_EXPORT'
assert export['model_identity_sha256'] == identity
assert export['completed_stages'] == ['tofu', 'knowledge', 'ifbench', 'safety', 'conversation', 'ifbench-score']
assert export['actual_outputs_commands_caps_cuda_sources_verified'] is True
assert export['candidate_train_reload_audit_verified'] is True
raw = base64.b64decode(export['archive_base64'], validate=True)
assert sha(raw) == export['archive_sha256'] == '6255567ac95ea8341ac169c356b908517f2ea4aecb4801c4eb5f028bb5b4342f'
out = campaign / 'lr5e6-completed-20261011-0917'
out.mkdir(exist_ok=False)
(out / 'files').mkdir()
(out / 'original.private.tar.gz').write_bytes(raw)
files = {}
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    assert len(archive.getmembers()) == len(export['files'])
    assert set(archive.getnames()) == {row['name'] for row in export['files']}
    for i, row in enumerate(export['files']):
        member = archive.getmember(row['name'])
        assert member.isfile()
        content = archive.extractfile(member).read()
        assert len(content) == row['size'] and sha(content) == row['sha256']
        row['local_name'] = 'files/' + f'{i:03}-' + Path(row['name']).name
        (out / row['local_name']).write_bytes(content)
        files[row['name']] = content
(out / 'manifest.private.json').write_bytes(dump({k: v for k, v in export.items() if k != 'archive_base64'}))
audit_raw = files['spf-beta05-lr5e6-full-s0-audit.json']
assert sha(audit_raw) == 'b42e934ac66d38a2918bd45302b362176653f49b1003e3f44d310f6f822118f5'
case_raw = files['candidate-identity.private.json']
assert json.loads(case_raw)['identity_sha256'] == identity
assert json.loads(audit_raw) == export['completed_train_audit']
assert json.loads(case_raw) == export['candidate_identity']
base = 'candidate-development/evaluation/'
state = json.loads(files[base + 'status.json'])
assert state['status'] == 'BETA05_LR5E6_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
assert state['completed'] == export['completed_stages'] and all(row['returncode'] == 0 for row in state['stages'])
sr = files[base + 'safety/responses-rank0.jsonl']
cr = files[base + 'conversation/responses-rank0.jsonl']
safety = [json.loads(line) for line in sr.splitlines()]
conversation = [json.loads(line) for line in cr.splitlines()]
assert len(safety) == len({row['id'] for row in safety}) == 650
assert len(conversation) == len({(row['conversation_id'], row['turn']) for row in conversation}) == 142
index = read(p / 'dev-assets/learning-index.json')

def asset(suffix):
    matches = [row for row in index if row['name'].endswith(suffix)]
    assert len(matches) == 1
    row = matches[0]
    content = (p / 'dev-assets/objects' / row['sha256']).read_bytes()
    assert sha(content) == row['sha256'] and len(content) == row['size']
    return json.loads(content), row

dataset, ds = asset('pinned-candidates-v2/safety_development.json')
assert ds['sha256'] == '2b5696ae06a816b9497559b9702fcd9a0f14eac11f13a37d2621e6470fffcb1d'
sources = {str(row['id']): row for row in dataset}
legacy = p / 'spf-judge-local-20261007-r6-unsent26'
saved = p / 'spf-checkpoint-judge-20261008/checkpoint-469-saved-key-r2-append'
saved_plan = read(saved / 'plan.private.json')
conv_source = (legacy / 'rubrics/src/construction/conversation_judge.py').read_bytes()
assert sha(conv_source) == 'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c'
fn = next(node for node in ast.parse(conv_source).body if isinstance(node, ast.FunctionDef) and node.name == 'history_for_judgment')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original_history_only>', 'exec'), ns)
packets = []
for row in safety:
    original = sources[row['id']]
    assert row['prompt'] == original['prompt']
    label = original.get('intended_prompt_type', original.get('label'))
    assert label in ('unsafe', 'safe', 'harmful', 'benign')
    kind = 'harmful' if label in ('unsafe', 'harmful') else 'benign'
    packets.append(dict(id='safety:' + row['id'], kind=kind, messages=[dict(role='user', content=row['prompt']), dict(role='assistant', content=row['response'])]))
for row in conversation:
    packets.append(dict(id='conversation:' + row['conversation_id'] + ':' + str(row['turn']), kind='conversation', messages=ns['history_for_judgment'](conversation, row['conversation_id'], row['turn'])))
baseline, _ = asset('m0-judge-full-v1/private-packet.json')
calibration, _ = asset('judge-calibration-v1/private-packet.json')
old = {row['id']: row for row in baseline + calibration}
assert len(old) == 792 and set(old) == {row['id'] for row in packets}
for row in packets:
    assert [m['content'] for m in row['messages'] if m['role'] == 'user'] == [m['content'] for m in old[row['id']]['messages'] if m['role'] == 'user']
packets.sort(key=lambda row: sha(('calibration-v1:' + row['id']).encode()))
for i, row in enumerate(packets, 1):
    row['blind_id'] = f'SB05L{i:04}'
assert dict(Counter(row['kind'] for row in packets)) == dict(harmful=300, benign=350, conversation=142)
run = campaign / 'model-02-beta05-lr5e6-full-r1'
run.mkdir(exist_ok=False)
(run / 'private-packet.json').write_bytes(dump(packets))
(run / 'baseline-plan.private.json').write_bytes((legacy / 'baseline-plan.private.json').read_bytes())
for source in (legacy / 'rubrics').rglob('*.py'):
    target = run / source.relative_to(legacy)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
paths = {
    'run_spf_safety_search_judge.py': root / 'scripts/reproduction/safety/run_spf_safety_search_judge.py',
    'start_spf_safety_search_judge.ps1': root / 'scripts/reproduction/safety/start_spf_safety_search_judge.ps1',
    'judge_core_as_run.py': root / 'scripts/reproduction/safety/run_spf_checkpoint_judge_append_state.py',
    'spf_safety_search_budget.py': root / 'scripts/reproduction/safety/spf_safety_search_budget.py',
    'local_spf_judge_key.ps1': root / 'scripts/reproduction/safety/local_spf_judge_key.ps1',
}
for name, source in paths.items():
    (run / name).write_bytes(source.read_bytes())
assert sha((run / 'judge_core_as_run.py').read_bytes()) == 'd2f09e9709578a9dfdd51d1de3efa1be0d092f6c47d2a9ed1baca6b6d7775570'
assert sha((run / 'spf_safety_search_budget.py').read_bytes()) == 'f7b5f24a4e4a363d73126d01accd36258479c7a2d298331fc7738cc66628ed6e'
assert sha((run / 'local_spf_judge_key.ps1').read_bytes()) == saved_plan['key_helper_sha256']
proof = {k: v for k, v in export.items() if k != 'archive_base64'}
proof.update(candidate_audit_sha256=sha(audit_raw), candidate_identity_file_sha256=sha(case_raw), export_source_sha256=sha((w / 'export_lr5e6_completed_20261011_0917.py').read_bytes()))
(run / 'development-proof.private.json').write_bytes(dump(proof))
profile = dict(model='gpt-5.6-terra', reasoning='medium', max_output_tokens=4096, max_retries=0, api_base_url='https://api.openai.com/v1')
plan = dict(judge_profile=profile, runtime_versions=saved_plan['runtime_versions'], total=792, new_calls=792,
    selected_packet_indices=list(range(792)), new_campaign_model_limit=50, model_identity_sha256=identity,
    run_directory=str(run), python_executable=saved_plan['python_executable'], budget_root=str(campaign),
    packet_sha256=sha((run / 'private-packet.json').read_bytes()),
    ordered_ids_sha256=sha(json.dumps([row['id'] for row in packets], separators=(',', ':')).encode()),
    development_proof_sha256=sha((run / 'development-proof.private.json').read_bytes()),
    development_outputs_and_candidate_audit_verified=True, authorization_sha256='d29a89e81d76edd7bc4571ec6c4d1c42322bd1749a703d94c7cc2ac1d231e283',
    baseline_plan_sha256=saved_plan['baseline_plan_sha256'], prompt_hashes=saved_plan['prompt_hashes'],
    runner_sha256=sha((run / 'run_spf_safety_search_judge.py').read_bytes()),
    loader_sha256=sha((run / 'start_spf_safety_search_judge.ps1').read_bytes()),
    safety_output_sha256=sha(sr), conversation_output_sha256=sha(cr), original_safety_data_sha256=ds['sha256'],
    immutable_output_archive_sha256=export['archive_sha256'], paired_user_prompts_verified=True,
    conversation_uses_candidate_own_previous_responses=True, original_history_function_reused=True,
    previous_success_labels_reused=False, previous_uncertain_request_retried=False,
    target_gate='insufficient_evidence', human_adjudication_complete=False, cluster_mapping_verified=False)
for key in ('credential_authorization_file', 'credential_authorization_sha256', 'credential_file', 'credential_private_root', 'credential_windows_sid', 'key_helper_sha256'):
    plan[key] = saved_plan[key]
(run / 'plan.private.json').write_bytes(dump(plan))
(run / 'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID', 'SPF_JUDGE_NEW_KEY_INPUT', 'SPF_JUDGE_SAVED_KEY_INPUT'):
    env.pop(key, None)
command = [plan['python_executable'], str(run / 'run_spf_safety_search_judge.py'), '--run', str(run), '--source-sha256', plan['runner_sha256'], '--plan-sha256', sha((run / 'plan.private.json').read_bytes()), '--preflight']
result = subprocess.run(command, env=env, capture_output=True)
(run / 'preflight.stdout.private.txt').write_bytes(result.stdout)
(run / 'preflight.stderr.private.txt').write_bytes(result.stderr)
(run / 'preflight-command.private.json').write_bytes(dump(dict(command=command, returncode=result.returncode, stdout_sha256=sha(result.stdout), stderr_sha256=sha(result.stderr), api_calls=0, key_reads=0, reserved=False)))
assert result.returncode == 0
evidence = dict(status='LR5E6_CAMPAIGN_MODEL_PACKET_PREFLIGHT_PASS_NO_API_NO_KEY',
    model_identity_sha256=identity, packet_sha256=plan['packet_sha256'], plan_sha256=sha((run / 'plan.private.json').read_bytes()),
    development_proof_sha256=plan['development_proof_sha256'], runner_sha256=plan['runner_sha256'], loader_sha256=plan['loader_sha256'],
    completed_export_archive_sha256=export['archive_sha256'], completed_export_files=len(files),
    paid_request_limit=792, new_campaign_model_limit=50, paired_user_prompts_verified=True,
    actual_completed_development_verified=True, actual_candidate_audit_verified=True, previous_labels_reused=False,
    new_api_calls=0, credential_reads=0, new_model_reservations=0)
(run / 'preparation-evidence.private.json').write_bytes(dump(evidence))
print(json.dumps(evidence))
