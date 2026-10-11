"""Save only newly completed train/reload/audit artifacts; no progress query or weights read."""
from pathlib import Path
import base64, hashlib, io, json, tarfile

T = Path('/data/spf-beta05-lr3e6-full-20261011-r1')
V = T / 'validation-code'
C = T / 'spf-beta05-lr3e6-full-s0'
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
assert audit['final_checkpoint']['parameter_tensors'] == 291
assert audit['final_checkpoint']['weights_bytes'] == 16060556616
state = json.loads((C / 'trainer_state.json').read_bytes())
assert state['global_step'] == 625 and state['epoch'] == 5
resolved = json.loads((C / 'resolved_training.json').read_bytes())
assert resolved['args']['adam_beta1'] == .5 and resolved['args']['adam_beta2'] == .999
assert resolved['args']['learning_rate'] == 3e-6 and resolved['args']['weight_decay'] == .01
binding = json.loads((T / 'optimizer-binding.private.json').read_bytes())
assert binding['status'] == 'ACTUAL_BETA05_LR3E6_FULL_OPTIMIZER_BOUND'
assert binding['actual_nonprofile_training_args_match'] and binding['original_training_loop_delegated']
assert binding['beta1'] == .5 and binding['beta2'] == .999 and binding['learning_rate'] == 3e-6
case = json.loads((T / 'candidate-identity.private.json').read_bytes())
assert case['candidate_audit_sha256'] == sha(auditraw) and case['step'] == 625
assert case['checkpoint'] == str(C)
assert case['weights_sha256'] == audit['final_checkpoint']['files'][0]['sha256']
identity = {k: v for k, v in case.items() if k != 'identity_sha256'}
assert sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()) == case['identity_sha256']

# All paths here belong to the three completed stages. Development/status is deliberately excluded.
files = {T / 'spf-beta05-lr3e6-full-s0-audit.json', T / 'optimizer-binding.private.json',
         T / 'candidate-identity.private.json'}
for name in ('target.json', 'resolved_training.json', 'trainer_state.json', 'trajectory.jsonl',
             'config.json', 'generation_config.json', 'tokenizer_config.json'):
    files.add(C / name)
for step in (157, 313, 469, 625):
    files.add(C / f'checkpoint-{step}' / 'trainer_state.json')
files.update(Path(path) for path in plan['source_sha256'] if Path(path).is_relative_to(T))
files.update(V / name for name in ('plan.private.json', 'bound-snapshot.private.json',
                                  'beta05-lr3e6-full-launch-20261011.json'))
logs = {'train.raw.log': 'c05340da332adfb0ae7191c88a0170358666a6629696af1f1eeef470e3e9f003',
        'fresh-reload.raw.log': 'd6350057ce9fb833e9f82ab5c2d2bf94ed209f50f29f06dd833289046a914a91',
        'full-byte-audit.raw.log': '978be14dbe027ac3ad4b076e0ffdc3d6d9b679a5a47616595ad1160319e8bea2'}
for name, expected in logs.items():
    assert sha((T / name).read_bytes()) == expected
    files.add(T / name)
records = []
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as archive:
    for path in sorted(files):
        assert path.resolve().is_relative_to(T) and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) < 15000000 and path.suffix not in ('.safetensors', '.bin', '.pt')
        name = path.relative_to(T).as_posix()
        records.append(dict(name=name, remote_path=str(path), size=len(raw), sha256=sha(raw)))
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
raw = stream.getvalue()
print(json.dumps(dict(status='IMMUTABLE_LR3E6_COMPLETED_TRAIN_RELOAD_AUDIT_EXPORT',
    files=records, archive_sha256=sha(raw), archive_base64=base64.b64encode(raw).decode(),
    completed_train_audit=audit, candidate_identity=case, optimizer_binding=binding,
    weights_read=False, actual_model_eval_repeated=0, paid_api_calls=0, credential_reads=0,
    evaluation_progress_queried=False, progress_snapshot=False)))
