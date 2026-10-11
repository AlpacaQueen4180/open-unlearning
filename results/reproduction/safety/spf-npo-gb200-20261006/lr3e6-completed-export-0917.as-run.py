"""First immutable export of the completed new candidate; no model or progress work."""
from pathlib import Path
import os, sys, hashlib, json, importlib.util, io, tarfile, base64

T = Path('/data/spf-beta05-lr3e6-full-20261011-r1')
V = T / 'validation-code'
C = T / 'spf-beta05-lr3e6-full-s0'
D = T / 'candidate-development'
R = D / 'evaluation'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
planraw = (V / 'plan.private.json').read_bytes()
assert sha(planraw) == '0ef32eae9962ffa418e4d6d9f672dc2e67aeb51fa3b43cdc4905cd4086fec3c8'
plan = json.loads(planraw)
for path, expected in plan['source_sha256'].items():
    assert sha(Path(path).read_bytes()) == expected
auditraw = (T / 'spf-beta05-lr3e6-full-s0-audit.json').read_bytes()
assert sha(auditraw) == '839156fd5c6d0e28847991188a95061d513727cf012ec912465697ca9450c181'
audit = json.loads(auditraw)
assert audit['status'] == 'pass' and audit['actual_updates'] == 625 and audit['actual_epochs'] == 5
assert audit['actual_examples'] == 20000 and audit['unique_examples'] == 4000
assert audit['fresh_reload']['status'] == 'pass' and audit['fresh_reload']['max_logit_error'] == 0
assert audit['final_checkpoint']['parameter_tensors'] == 291 and audit['final_checkpoint']['weights_bytes'] == 16060556616
trainer = json.loads((C / 'trainer_state.json').read_bytes())
assert trainer['global_step'] == 625 and trainer['epoch'] == 5
resolved = json.loads((C / 'resolved_training.json').read_bytes())
assert resolved['args']['adam_beta1'] == .5 and resolved['args']['adam_beta2'] == .999
assert resolved['args']['learning_rate'] == 3e-6 and resolved['args']['weight_decay'] == .01
binding = json.loads((T / 'optimizer-binding.private.json').read_bytes())
assert binding['status'] == 'ACTUAL_BETA05_LR3E6_FULL_OPTIMIZER_BOUND'
assert binding['actual_nonprofile_training_args_match'] and binding['original_training_loop_delegated']
assert binding['beta1'] == .5 and binding['beta2'] == .999 and binding['learning_rate'] == 3e-6
case = json.loads((T / 'candidate-identity.private.json').read_bytes())
assert case['candidate_audit_sha256'] == sha(auditraw) and case['step'] == 625
assert case['checkpoint'] == str(C) and case['weights_sha256'] == audit['final_checkpoint']['files'][0]['sha256']
identity = {k: v for k, v in case.items() if k != 'identity_sha256'}
assert sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()) == case['identity_sha256']
parent_raw = (T / 'status.json').read_bytes()
parent = json.loads(parent_raw)
assert parent['pid'] == 12367
assert parent['status'] == 'BETA05_LR3E6_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
assert parent['completed'] == ['train', 'fresh-reload', 'full-byte-audit', 'candidate-development']
assert parent['plan_sha256'] == sha(planraw)
assert parent['source_sha256'] == 'caba122d48d89e3fc40d9d7327300910fb614883f28a60ef50ea64d2d8e31a14'
adapterraw = (D / 'adapter-generation-api-r2/adapter-audit.json').read_bytes()
adapter = json.loads(adapterraw)
assert adapter['checkpoint_identity_sha256'] == case['identity_sha256']
assert adapter['candidate_audit_sha256'] == sha(auditraw)
os.environ.update(SPF_DIAG_CASE_TASK=str(D), SPF_DIAG_IDENTITY=case['identity_sha256'], SPF_DIAG_ADAPTER_SHA=sha(adapterraw))
source = V / 'spf_beta05_lr3e6_development_executor.py'
assert sha(source.read_bytes()) == '74fb508d71d667ef0de08fee3e471b85ba377bc00298a026bf1106a6dbecb643'
spec = importlib.util.spec_from_file_location('completed_lr3e6_executor', source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
state_raw = (R / 'status.json').read_bytes()
state = json.loads(state_raw)
assert state['status'] == 'BETA05_LR3E6_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
assert state['completed'] == list(m.STAGES)
assert state['commands'] == m.plan(R) and state['controller_sha256'] == sha(source.read_bytes())
files = {T / 'status.json', R / 'status.json', source}
expected_logs = {'train': 'c05340da332adfb0ae7191c88a0170358666a6629696af1f1eeef470e3e9f003',
                 'fresh-reload': 'd6350057ce9fb833e9f82ab5c2d2bf94ed209f50f29f06dd833289046a914a91',
                 'full-byte-audit': '978be14dbe027ac3ad4b076e0ffdc3d6d9b679a5a47616595ad1160319e8bea2'}
assert len(parent['stages']) == 3
for row in parent['stages']:
    assert row['returncode'] == 0 and row['raw_log_sha256'] == expected_logs[row['name']]
    # Completed train raw logs are bound to the first immutable archive, never fetched again.
files.update(p for p in (D / 'adapter-generation-api-r2').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for row in state['stages']:
    assert row['returncode'] == 0 and row['executor_command'] == state['commands'][row['name']]
    assert m.actual_result(R, row['name'], row) == row['outputs']
    log = Path(row['log'])
    assert sha(log.read_bytes()) == row['raw_log_sha256']
    files.add(log)
    files.update(R / name for name in row['outputs'])
assert (T / 'status.json').read_bytes() == parent_raw and (R / 'status.json').read_bytes() == state_raw
# Only new development artifacts and final parent state enter this next archive.
retained_train_names = {'spf-beta05-lr3e6-full-s0/generation_config.json', 'spf-beta05-lr3e6-full-s0/tokenizer_config.json', 'spf-beta05-lr3e6-full-s0/trainer_state.json', 'full-byte-audit.raw.log', 'spf-beta05-lr3e6-full-s0/target.json', 'spf-beta05-lr3e6-full-s0/checkpoint-313/trainer_state.json', 'spf-beta05-lr3e6-full-s0/trajectory.jsonl', 'spf-beta05-lr3e6-full-s0/resolved_training.json', 'validation-code/beta05-lr3e6-full-launch-20261011.json', 'spf-beta05-lr3e6-full-s0/config.json', 'validation-code/plan.private.json', 'spf-beta05-lr3e6-full-s0/checkpoint-625/trainer_state.json', 'fresh-reload.raw.log', 'validation-code/run_spf_beta05_lr3e6_full.py', 'spf-beta05-lr3e6-full-s0/checkpoint-469/trainer_state.json', 'validation-code/bound-snapshot.private.json', 'validation-code/run_spf_beta05_lr3e6_full_queue.py', 'spf-beta05-lr3e6-full-s0-audit.json', 'optimizer-binding.private.json', 'validation-code/spf_beta05_lr3e6_development_executor.py', 'candidate-identity.private.json', 'spf-beta05-lr3e6-full-s0/checkpoint-157/trainer_state.json', 'train.raw.log'}
files = {path for path in files if path.relative_to(T).as_posix() not in retained_train_names}
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
print(json.dumps(dict(status='IMMUTABLE_LR3E6_COMPLETED_TRAIN_DEVELOPMENT_EXPORT', files=records,
    archive_sha256=sha(raw), archive_base64=base64.b64encode(raw).decode(),
    model_identity_sha256=case['identity_sha256'], candidate_identity=case,
    completed_train_audit=audit, optimizer_binding=binding, completed_stages=state['completed'],
    actual_outputs_commands_caps_cuda_sources_verified=True, candidate_train_reload_audit_verified=True,
    previous_train_export_archive_sha256='1310647eca2d4e5ad4064d8efac839809261056ad09f6ed80efb65b857baa3f2',
    retained_training_files_not_exported_again=True, weights_read=False, evaluations_repeated=0, progress_snapshot=False, paid_api_calls=0, credential_reads=0)))
