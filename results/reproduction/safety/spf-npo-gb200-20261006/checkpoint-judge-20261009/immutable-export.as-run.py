"""Immutable export of newly completed313/469 only; no progress query or weights."""
from pathlib import Path
import os, sys, hashlib, json, importlib.util, io, tarfile, base64
T = Path('/data/spf-checkpoint-beta-20261008-r1')
V = T / 'validation-code'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
planraw = (V / 'plan.private.json').read_bytes()
assert sha(planraw) == 'd047168aefa7e8ceb1b98de8299fbc2e79b14da1a79812be9f6218d24cca26b6'
plan = json.loads(planraw)
for path, digest in plan['source_sha256'].items():
    assert sha(Path(path).read_bytes()) == digest
parent_raw = (T / 'status.json').read_bytes()
parent = json.loads(parent_raw)
assert parent['status'] == 'EARLY_CHECKPOINTS_AND_PREFIX_GEOMETRY_COMPLETE_PENDING_JUDGE'
assert parent['completed'] == ['observer-cpu-selftest', 'beta-09', 'beta-05', 'checkpoint-157', 'checkpoint-313', 'checkpoint-469']
files = {T / 'status.json'}
states = {}
for step in (313, 469):
    C = T / f'checkpoint-{step}'
    R = C / 'evaluation'
    case = next(row for row in plan['cases'] if row['step'] == step)
    adapter_raw = (C / 'adapter-generation-api-r2/adapter-audit.json').read_bytes()
    adapter = json.loads(adapter_raw)
    assert adapter['checkpoint_identity_sha256'] == case['identity_sha256'] and adapter['trajectory_step'] == step
    os.environ.update(SPF_DIAG_CASE_TASK=str(C), SPF_DIAG_IDENTITY=case['identity_sha256'], SPF_DIAG_ADAPTER_SHA=sha(adapter_raw))
    spec = importlib.util.spec_from_file_location(f'completed_executor_{step}', V / 'spf_checkpoint_executor.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    state_raw = (R / 'status.json').read_bytes()
    state = json.loads(state_raw)
    assert state['status'] == 'CHECKPOINT_EVALUATION_COMPLETE_PENDING_NEW_JUDGE' and state['completed'] == list(m.STAGES)
    assert state['commands'] == m.plan(R) and state['controller_sha256'] == sha(Path(m.__file__).read_bytes())
    files.add(R / 'status.json')
    files.update(p for p in (C / 'adapter-generation-api-r2').rglob('*') if p.is_file())
    for row in state['stages']:
        assert row['returncode'] == 0 and row['executor_command'] == state['commands'][row['name']]
        assert m.actual_result(R, row['name'], row) == row['outputs']
        log = Path(row['log'])
        assert sha(log.read_bytes()) == row['raw_log_sha256']
        files.add(log)
        files.update(R / name for name in row['outputs'])
    assert (R / 'status.json').read_bytes() == state_raw
    states[str(step)] = sha(state_raw)
assert (T / 'status.json').read_bytes() == parent_raw
records = []
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as archive:
    for path in sorted(files):
        assert path.resolve().is_relative_to(T) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) < 15000000 and path.suffix not in ('.safetensors', '.pt', '.bin')
        name = path.relative_to(T).as_posix()
        records.append(dict(name=name, remote_path=str(path), size=len(raw), sha256=sha(raw)))
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
raw = stream.getvalue()
print(json.dumps(dict(status='IMMUTABLE_COMPLETED313469_EXPORT', files=records,
    archive_sha256=sha(raw), archive_base64=base64.b64encode(raw).decode(), case_status_sha256=states,
    actual_complete_outputs_commands_caps_cuda_sources_verified=True, weights_read=False,
    evaluations_repeated=0, progress_snapshot=False, paid_api_calls=0,
    completed157_or_beta_records_exported=False)))
