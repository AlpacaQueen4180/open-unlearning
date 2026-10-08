from pathlib import Path
import copy, hashlib, json, subprocess
root = Path('.').resolve()
p = root / 'work/spf-npo-gb200-20261006/private'
original = p / 'checkpoint313469-judge-queue-20261009'
source = original / 'run_spf_checkpoint_judge_remaining_queue.ps1'
config = json.loads((original / 'config.private.json').read_bytes())
out = p / 'remaining-judge-queue-new-guards-20261009'
out.mkdir(exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
rejected = []
for name in ('cap2377', 'retry1', 'reversed_steps'):
    fixture = out / name
    fixture.mkdir()
    modified = copy.deepcopy(config)
    if name == 'cap2377': modified['total_authorized_cap'] = 2377
    if name == 'retry1': modified['retries'] = 1
    if name == 'reversed_steps': modified['cases'].reverse()
    raw = (json.dumps(modified, indent=2) + '\n').encode()
    (fixture / 'config.private.json').write_bytes(raw)
    result = subprocess.run([config['pwsh_executable'], '-NoProfile', '-File', str(source), '-QueueDirectory', str(fixture), '-SourceSha256', sha(source.read_bytes()), '-ConfigSha256', sha(raw), '-Preflight'], capture_output=True)
    (fixture / 'stdout.private.txt').write_bytes(result.stdout)
    (fixture / 'stderr.private.txt').write_bytes(result.stderr)
    assert result.returncode != 0 and b'Only remaining313/469 under original budget allowed' in result.stderr
    assert not (fixture / 'status.private.json').exists()
    rejected.append(name)
record = dict(status='NEW_REMAINING_QUEUE_SCOPE_NEGATIVES_REJECTED', rejected=rejected,
    source_sha256=sha(source.read_bytes()), real_api_calls=0, key_contents_read=False,
    original792_mock_loop_repeated=False, completed157_audit_repeated=False)
(out / 'checks.private.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
(out / 'verifier.as-run.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record))
