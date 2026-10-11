"""First immutable export of the completed new candidate; no model or progress work."""
from pathlib import Path
import os, sys, hashlib, json, importlib.util, io, tarfile, base64

T = Path('/data/spf-beta05-lr5e6-full-20261010-r1')
V = T / 'validation-code'
C = T / 'spf-beta05-lr5e6-full-s0'
D = T / 'candidate-development'
R = D / 'evaluation'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
planraw = (V / 'plan.private.json').read_bytes()
assert sha(planraw) == '8631a99a9f419037a04677c29384862506dfabbddaf9138d0741840a00ca0d6b'
plan = json.loads(planraw)
for path, expected in plan['source_sha256'].items():
    assert sha(Path(path).read_bytes()) == expected
auditraw = (T / 'spf-beta05-lr5e6-full-s0-audit.json').read_bytes()
assert sha(auditraw) == 'b42e934ac66d38a2918bd45302b362176653f49b1003e3f44d310f6f822118f5'
audit = json.loads(auditraw)
assert audit['status'] == 'pass' and audit['actual_updates'] == 625 and audit['actual_epochs'] == 5
assert audit['actual_examples'] == 20000 and audit['unique_examples'] == 4000
assert audit['fresh_reload']['status'] == 'pass' and audit['fresh_reload']['max_logit_error'] == 0
assert audit['final_checkpoint']['parameter_tensors'] == 291 and audit['final_checkpoint']['weights_bytes'] == 16060556616
trainer = json.loads((C / 'trainer_state.json').read_bytes())
assert trainer['global_step'] == 625 and trainer['epoch'] == 5
resolved = json.loads((C / 'resolved_training.json').read_bytes())
assert resolved['args']['adam_beta1'] == .5 and resolved['args']['adam_beta2'] == .999
assert resolved['args']['learning_rate'] == 5e-6 and resolved['args']['weight_decay'] == .01
binding = json.loads((T / 'optimizer-binding.private.json').read_bytes())
assert binding['status'] == 'ACTUAL_BETA05_LR5E6_FULL_OPTIMIZER_BOUND'
assert binding['actual_nonprofile_training_args_match'] and binding['original_training_loop_delegated']
assert binding['beta1'] == .5 and binding['beta2'] == .999 and binding['learning_rate'] == 5e-6
case = json.loads((T / 'candidate-identity.private.json').read_bytes())
assert case['candidate_audit_sha256'] == sha(auditraw) and case['step'] == 625
assert case['checkpoint'] == str(C) and case['weights_sha256'] == audit['final_checkpoint']['files'][0]['sha256']
identity = {k: v for k, v in case.items() if k != 'identity_sha256'}
assert sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()) == case['identity_sha256']
parent_raw = (T / 'status.json').read_bytes()
parent = json.loads(parent_raw)
assert parent['pid'] == 10758
assert parent['status'] == 'BETA05_LR5E6_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
assert parent['completed'] == ['train', 'fresh-reload', 'full-byte-audit', 'candidate-development']
assert parent['plan_sha256'] == sha(planraw)
assert parent['source_sha256'] == 'eb6ca91386852591299d95e3f54e6309d1c5ba6b78064a8a8d726de3fc28cf08'
adapterraw = (D / 'adapter-generation-api-r2/adapter-audit.json').read_bytes()
adapter = json.loads(adapterraw)
assert adapter['checkpoint_identity_sha256'] == case['identity_sha256']
assert adapter['candidate_audit_sha256'] == sha(auditraw)
os.environ.update(SPF_DIAG_CASE_TASK=str(D), SPF_DIAG_IDENTITY=case['identity_sha256'], SPF_DIAG_ADAPTER_SHA=sha(adapterraw))
source = V / 'spf_beta05_lr5e6_development_executor.py'
assert sha(source.read_bytes()) == '535f7fe622ae6da568ce8a4f090b84410dacafa5dd6e8335782832b26cb9da27'
spec = importlib.util.spec_from_file_location('completed_lr5e6_executor', source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
state_raw = (R / 'status.json').read_bytes()
state = json.loads(state_raw)
assert state['status'] == 'BETA05_LR5E6_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
assert state['completed'] == list(m.STAGES)
assert state['commands'] == m.plan(R) and state['controller_sha256'] == sha(source.read_bytes())
files = {T / 'status.json', T / 'spf-beta05-lr5e6-full-s0-audit.json',
         T / 'optimizer-binding.private.json', T / 'candidate-identity.private.json', R / 'status.json', source}
for name in ('target.json', 'resolved_training.json', 'trainer_state.json', 'trajectory.jsonl',
             'config.json', 'generation_config.json', 'tokenizer_config.json'):
    files.add(C / name)
for step in (157, 313, 469, 625):
    files.add(C / f'checkpoint-{step}' / 'trainer_state.json')
files.update(Path(path) for path in plan['source_sha256'] if Path(path).is_relative_to(T))
files.update(V / name for name in ('plan.private.json', 'bound-snapshot.private.json',
                                  'beta05-lr5e6-full-launch-20261010.json', 'beta05-lr5e6-full-launch-20261010.raw.log'))
expected_logs = {'train': '64e90f05d84b83a318e4c4fbbb00e0dcf5d6b1fd6f9924bedd47a473d808b829',
                 'fresh-reload': 'fa61f5d0fe19b5de2b8597f23da122793e56a016502b78441b79bad554ce42a2',
                 'full-byte-audit': '83bad1a86908a7850d98f9ec32b0811325f6fcb9c25ed868f446244a4155be11'}
assert len(parent['stages']) == 3
for row in parent['stages']:
    assert row['returncode'] == 0 and row['raw_log_sha256'] == expected_logs[row['name']]
    log = Path(row['log'])
    assert sha(log.read_bytes()) == expected_logs[row['name']]
    files.add(log)
files.update(p for p in (D / 'adapter-generation-api-r2').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for row in state['stages']:
    assert row['returncode'] == 0 and row['executor_command'] == state['commands'][row['name']]
    assert m.actual_result(R, row['name'], row) == row['outputs']
    log = Path(row['log'])
    assert sha(log.read_bytes()) == row['raw_log_sha256']
    files.add(log)
    files.update(R / name for name in row['outputs'])
assert (T / 'status.json').read_bytes() == parent_raw and (R / 'status.json').read_bytes() == state_raw
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
print(json.dumps(dict(status='IMMUTABLE_LR5E6_COMPLETED_TRAIN_DEVELOPMENT_EXPORT', files=records,
    archive_sha256=sha(raw), archive_base64=base64.b64encode(raw).decode(),
    model_identity_sha256=case['identity_sha256'], candidate_identity=case,
    completed_train_audit=audit, optimizer_binding=binding, completed_stages=state['completed'],
    actual_outputs_commands_caps_cuda_sources_verified=True, candidate_train_reload_audit_verified=True,
    weights_read=False, evaluations_repeated=0, progress_snapshot=False, paid_api_calls=0, credential_reads=0)))
