from pathlib import Path
import base64, hashlib, io, json, tarfile

t = Path('/data/spf-npo-smoke-20261007-r1')
r = t / 'runs/reload-native-config-2025'
v = t / 'validation-code'
sha = lambda b: hashlib.sha256(b).hexdigest()
s = json.loads((r / 'status.json').read_bytes())
assert s['status'] == 'FAILED' and s['pid'] == 4577 and s['completed'] == []
assert len(s['stages']) == 1
assert s['stages'][0]['returncode'] == 1 and s['stages'][0]['process_pid'] == 4584
assert sha((r / 'fresh-reload.raw.log').read_bytes()) == '9b06a9088e03084f0fc53263bfed3c8c4bda526e32c9cea8cf1b3c1cfe746f71'
audit_raw = (r / 'reload-audit.private.json').read_bytes()
assert sha(audit_raw) == 'f78d8208591969b3ee53735de7725e8702aa9417369bbef7242994b4e2e7b40d'
a = json.loads(audit_raw)
assert a['status'] == 'FAILED_NATIVE_CONTEXT_RELOAD' and a['pid'] == 4652
assert a['optimizer_created'] is True and a['training_steps_executed'] == 0 and a['backward_calls'] == 0
assert a['error'] == "ValueError('Reload must not construct an optimizer or perform a step')"
assert not (r / 'reload-comparison.private.pt').exists()
files = {
    r / 'status.json', r / 'fresh-reload.raw.log', r / 'reload-audit.private.json',
    v / 'smoke-native-reload-config-launch-2025.json',
    v / 'smoke-native-reload-config-launch-2025.raw.log',
    v / 'run_spf_npo_smoke_reload_config_r3_queue.py',
    v / 'verify_spf_npo_smoke_native_reload_config_r3.py',
    v / 'r4-snapshot-native-reload-config-0340.private.json',
    Path('/data/spf-development-20261007-r1/repo/src/trainer/unlearn/base.py'),
    Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/deepspeed/runtime/engine.py'),
}
records = []
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
    for f in sorted(files):
        raw = f.read_bytes()
        assert len(raw) < 500000
        name = f.as_posix().lstrip('/')
        records.append(dict(name=name, remote_path=f.as_posix(), size=len(raw), sha256=sha(raw)))
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
raw = buffer.getvalue()
print(json.dumps(dict(
    status='IMMUTABLE_NATIVE_RELOAD_OPTIMIZER_FAILURE_EXPORT', files=records,
    archive_sha256=sha(raw), archive_base64=base64.b64encode(raw).decode(),
    weights_read=False, model_evaluation_executed=False, training_executed=False,
    progress_snapshot=False, original_failed_verifier_loaded_291_weights=True,
    original_failed_verifier_logits_comparison_executed=False,
)))
