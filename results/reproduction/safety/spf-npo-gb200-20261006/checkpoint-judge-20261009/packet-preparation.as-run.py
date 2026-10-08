"""Verify new immutable313/469 outputs and prepare own792 packets; API0/keyreads0."""
from pathlib import Path
from collections import Counter
import ast, base64, copy, csv, hashlib, io, json, os, subprocess, tarfile

root = Path('.').resolve()
w = root / 'work/spf-npo-gb200-20261006'
p = w / 'private'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
j = lambda path: json.loads(path.read_bytes())
dump = lambda value: (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
export = j(p / 'checkpoint313469-completed-export-20261009.private.json')
raw = base64.b64decode(export['archive_base64'], validate=True)
assert sha(raw) == export['archive_sha256'] == 'd9847aebcb38263ba335e6ce69cb927512c2286a1bfffd570baef1d206795ef1'
assert export['actual_complete_outputs_commands_caps_cuda_sources_verified'] and not export['weights_read']
out = p / 'checkpoint313469-completed-20261009'
budget = p / 'spf-checkpoint-judge-20261008'
assert all(not (budget / f'checkpoint-{step}-saved-key-r1').exists() for step in (313, 469))
out.mkdir(exist_ok=False)
(out / 'files').mkdir()
(out / 'original.private.tar.gz').write_bytes(raw)
files = {}
with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
    assert set(archive.getnames()) == {row['name'] for row in export['files']}
    for index, row in enumerate(export['files']):
        member = archive.getmember(row['name'])
        assert member.isfile()
        value = archive.extractfile(member).read()
        assert len(value) == row['size'] and sha(value) == row['sha256']
        row['local_name'] = 'files/' + f'{index:03}-' + Path(row['name']).name
        (out / row['local_name']).write_bytes(value)
        files[row['name']] = value
(out / 'manifest.private.json').write_bytes(dump({key: value for key, value in export.items() if key != 'archive_base64'}))
prior = p / 'checkpoint157-beta-completed-20261008'
prior_manifest = j(prior / 'manifest.private.json')
entry = next(row for row in prior_manifest['files'] if row['name'] == 'validation-code/plan.private.json')
original_plan_raw = (prior / entry['local_name']).read_bytes()
assert sha(original_plan_raw) == 'd047168aefa7e8ceb1b98de8299fbc2e79b14da1a79812be9f6218d24cca26b6'
original_plan = json.loads(original_plan_raw)
snapshot = j(p / 'checkpoint-beta-snapshot-20261009-0712.private.json')
assert json.loads(files['status.json']) == snapshot['checkpoint_beta_diagnostic']['status']
template_run = budget / 'checkpoint-157-saved-key-r1'
template = j(template_run / 'plan.private.json')
assert sha((template_run / 'plan.private.json').read_bytes()) == 'da3d1665426d9301c32ca2f01abc7d77590e39080849225f525156617073950d'
assert sha((budget / 'authorization.private.json').read_bytes()) == template['authorization_sha256']
assert sha((budget / 'saved-key-authorization.private.json').read_bytes()) == template['credential_authorization_sha256']
index = j(p / 'dev-assets/learning-index.json')
def asset(suffix):
    matches = [row for row in index if row['name'].endswith(suffix)]
    assert len(matches) == 1
    row = matches[0]
    value = (p / 'dev-assets/objects' / row['sha256']).read_bytes()
    assert len(value) == row['size'] and sha(value) == row['sha256']
    return json.loads(value), row
dataset, dataset_record = asset('pinned-candidates-v2/safety_development.json')
sources = {str(row['id']): row for row in dataset}
baseline, _ = asset('m0-judge-full-v1/private-packet.json')
calibration, _ = asset('judge-calibration-v1/private-packet.json')
old = {row['id']: row for row in baseline + calibration}
assert len(old) == 792
conv_source = (template_run / 'rubrics/src/construction/conversation_judge.py').read_bytes()
assert sha(conv_source) == 'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c'
fn = next(node for node in ast.parse(conv_source).body if isinstance(node, ast.FunctionDef) and node.name == 'history_for_judgment')
namespace = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original_history_only>', 'exec'), namespace)
summaries = []
for step in (313, 469):
    case = next(row for row in original_plan['cases'] if row['step'] == step)
    state_raw = files[f'checkpoint-{step}/evaluation/status.json']
    assert sha(state_raw) == export['case_status_sha256'][str(step)]
    state = json.loads(state_raw)
    assert state == snapshot['checkpoint_beta_diagnostic']['case_statuses'][str(step)]
    assert state['status'] == 'CHECKPOINT_EVALUATION_COMPLETE_PENDING_NEW_JUDGE'
    assert state['completed'] == ['tofu', 'knowledge', 'ifbench', 'safety', 'conversation', 'ifbench-score']
    base = f'checkpoint-{step}/evaluation/'
    safety_raw = files[base + 'safety/responses-rank0.jsonl']
    conv_raw = files[base + 'conversation/responses-rank0.jsonl']
    safety = [json.loads(line) for line in safety_raw.splitlines()]
    conversation = [json.loads(line) for line in conv_raw.splitlines()]
    assert len(safety) == len({row['id'] for row in safety}) == 650 and len(conversation) == 142
    packets = []
    for row in safety:
        source = sources[row['id']]
        assert row['prompt'] == source['prompt']
        label = source.get('intended_prompt_type', source.get('label'))
        assert label in ('unsafe', 'safe', 'harmful', 'benign')
        kind = 'harmful' if label in ('unsafe', 'harmful') else 'benign'
        packets.append(dict(id='safety:' + row['id'], kind=kind, messages=[dict(role='user', content=row['prompt']), dict(role='assistant', content=row['response'])]))
    for row in conversation:
        packets.append(dict(id='conversation:' + row['conversation_id'] + ':' + str(row['turn']), kind='conversation', messages=namespace['history_for_judgment'](conversation, row['conversation_id'], row['turn'])))
    assert set(old) == {row['id'] for row in packets}
    for row in packets:
        assert [m['content'] for m in row['messages'] if m['role'] == 'user'] == [m['content'] for m in old[row['id']]['messages'] if m['role'] == 'user']
    packets.sort(key=lambda row: sha(('calibration-v1:' + row['id']).encode()))
    for index, row in enumerate(packets, 1):
        row['blind_id'] = f'C{step}P{index:04}'
    assert dict(Counter(row['kind'] for row in packets)) == dict(harmful=300, benign=350, conversation=142)
    assert sha(json.dumps([row['id'] for row in packets], separators=(',', ':')).encode()) == template['ordered_ids_sha256']
    run = budget / f'checkpoint-{step}-saved-key-r1'
    run.mkdir(exist_ok=False)
    packet_raw = dump(packets)
    (run / 'private-packet.json').write_bytes(packet_raw)
    for name in ('baseline-plan.private.json', 'local_spf_judge_key.ps1', 'start_spf_checkpoint_judge_saved_key.ps1', 'run_spf_checkpoint_development_judge_saved_key.py'):
        (run / name).write_bytes((template_run / name).read_bytes())
    assert sha((run / 'local_spf_judge_key.ps1').read_bytes()) == template['key_helper_sha256']
    assert sha((run / 'start_spf_checkpoint_judge_saved_key.ps1').read_bytes()) == template['credential_dialog_sha256']
    assert sha((run / 'run_spf_checkpoint_development_judge_saved_key.py').read_bytes()) == template['runner_sha256']
    for source in (template_run / 'rubrics').rglob('*.py'):
        dest = run / source.relative_to(template_run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    plan = copy.deepcopy(template)
    for field in ('previous_waiting_run', 'previous_waiting_run_api0_required'):
        plan.pop(field, None)
    plan.update(checkpoint_step=step, checkpoint_identity_sha256=case['identity_sha256'], run_directory=str(run),
        packet_sha256=sha(packet_raw), safety_output_sha256=sha(safety_raw), conversation_output_sha256=sha(conv_raw),
        immutable_output_archive_sha256=export['archive_sha256'], original_safety_data_sha256=dataset_record['sha256'],
        preparation_source_sha256=sha(Path(__file__).read_bytes()), own_checkpoint_responses=True)
    (run / 'plan.private.json').write_bytes(dump(plan))
    selected = []
    for label in ('unsafe', 'safe', 'harmful', 'benign'):
        selected.extend([row for row in packets if row['id'].startswith('safety:') and sources[row['id'][7:]].get('intended_prompt_type', sources[row['id'][7:]].get('label')) == label][:10])
    cids = sorted({row['conversation_id'] for row in conversation}, key=lambda value: sha(('calibration-v1:' + value).encode()))[:5]
    selected.extend(row for row in packets if row['id'].startswith('conversation:') and row['id'].rsplit(':', 1)[0][13:] in cids)
    assert len(selected) == 50
    with (run / 'human-blind-review.private.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['blind_id', 'kind', 'conversation', 'human_label_or_overall', 'human_quality', 'human_reason'])
        writer.writeheader()
        for row in selected:
            writer.writerow(dict(blind_id=row['blind_id'], kind=row['kind'], conversation=json.dumps(row['messages'], ensure_ascii=False)))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID', 'OPENAI_LOG', 'SPF_JUDGE_NEW_KEY_INPUT', 'SPF_JUDGE_SAVED_KEY_INPUT'):
        env.pop(name, None)
    command = [plan['python_executable'], str(run / 'run_spf_checkpoint_development_judge_saved_key.py'), '--run', str(run), '--source-sha256', plan['runner_sha256'], '--plan-sha256', sha((run / 'plan.private.json').read_bytes()), '--preflight']
    result = subprocess.run(command, env=env, capture_output=True)
    (run / 'preflight.stdout.txt').write_bytes(result.stdout)
    (run / 'preflight.stderr.txt').write_bytes(result.stderr)
    assert result.returncode == 0
    summary = dict(checkpoint_step=step, status='OWN_CHECKPOINT792_PACKET_PREPARED_PREFLIGHT_PASS_API0',
        checkpoint_identity_sha256=case['identity_sha256'], plan_sha256=sha((run / 'plan.private.json').read_bytes()),
        packet_sha256=sha(packet_raw), safety_output_sha256=sha(safety_raw), conversation_output_sha256=sha(conv_raw),
        runner_sha256=plan['runner_sha256'], dialog_sha256=plan['credential_dialog_sha256'], key_helper_sha256=plan['key_helper_sha256'],
        ordered_ids_sha256=plan['ordered_ids_sha256'], per_checkpoint_cap=792, total_new_cap=2376,
        own_conversation_history=True, original_paired_user_prompts=True, label_transfer=False, new_api_calls=0, credential_reads=0)
    (run / 'preparation-evidence.private.json').write_bytes(dump(summary))
    (run / 'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
    summaries.append(summary)
record = dict(status='CHECKPOINT313469_NEW_PACKETS_READY', archive_sha256=export['archive_sha256'], export_files=len(files),
    cases=summaries, new_api_calls=0, credential_reads=0, generation_repeated=False, original_key_file_untouched=True)
(p / 'checkpoint313469-judge-prepared-20261009.private.json').write_bytes(dump(record))
print(json.dumps(record))
