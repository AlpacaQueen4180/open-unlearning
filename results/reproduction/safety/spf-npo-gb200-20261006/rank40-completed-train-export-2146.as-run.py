"""Export only newly completed hybrid full train/reload/audit; engineering and live development excluded."""
from pathlib import Path
import base64,hashlib,io,json,tarfile
T=Path('/data/spf-hybrid01-rank40-full-20261012-r1');C=T/'spf-hybrid01-rank40-full-s0'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
STAGES=[{'name': 'train', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1', '--', '/data/spf-hybrid01-rank40-full-20261012-r1/validation-code/run_spf_hybrid01_rank40_cpu_recovery.py', '--source-sha256', '4cc6b969be7a3fb52714a5e20c4c54c53b2f4b6f843d8549738d06685c05ca01', '--plan-sha256', '49ce276792f785c05e102f0c82355ea20e4f7806d86459448ff3e3c9359cee6b', '--plan-path', '/data/spf-hybrid01-rank40-full-20261012-r1/validation-code/plan-cpu-fixture-recovery-1539.private.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid01-rank40-full-20261012-r1/runs/cpu-fixture-recovery-1539/train.raw.log', 'started_at': 1791795254.3812895, 'process_pid': 21257, 'returncode': 0, 'finished_at': 1791811232.3371787, 'raw_log_sha256': '5dd16014009233dc53226426c80879dc136462c4862dbce0d7d4d7e5005dbe04'}, {'name': 'fresh-reload', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'construction.cli', 'verify', '--checkpoint', '/data/spf-hybrid01-rank40-full-20261012-r1/spf-hybrid01-rank40-full-s0'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid01-rank40-full-20261012-r1/runs/cpu-fixture-recovery-1539/fresh-reload.raw.log', 'started_at': 1791811232.3585029, 'process_pid': 21829, 'returncode': 0, 'finished_at': 1791811245.967812, 'raw_log_sha256': '45e9d40551409869899730dfbeb46c567acb90906d0f524afa19ea35eded98cf'}, {'name': 'full-byte-audit', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py', '--target', '/data/spf-hybrid01-rank40-full-20261012-r1/spf-hybrid01-rank40-full-s0', '--output', '/data/spf-hybrid01-rank40-full-20261012-r1/spf-hybrid01-rank40-full-s0-audit.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid01-rank40-full-20261012-r1/runs/cpu-fixture-recovery-1539/full-byte-audit.raw.log', 'started_at': 1791811245.9868686, 'process_pid': 22109, 'returncode': 0, 'finished_at': 1791811294.1042693, 'raw_log_sha256': '108232eb74048abd041cc7d7855b31be415e91d94e75354a9a664232da6f419b'}]
auditraw=(T/'spf-hybrid01-rank40-full-s0-audit.json').read_bytes()
assert sha(auditraw)=='071102b49e0112660df85907e892700cf5a7c479c36ce45cc6d2abd1627b839a'
audit=json.loads(auditraw)
assert audit['status']=='pass' and audit['actual_updates']==625 and audit['actual_epochs']==5
assert audit['actual_examples']==20000 and audit['unique_examples']==4000
assert audit['fresh_reload']['status']=='pass' and audit['fresh_reload']['max_logit_error']==0
assert audit['final_checkpoint']['parameter_tensors']==291 and audit['final_checkpoint']['weights_bytes']==16060556616
acceptanceraw=(T/'hybrid-full-acceptance.private.json').read_bytes()
assert sha(acceptanceraw)=='3050b474dee9f3e5ace7458ecd0590dd3ba38b392537acc6547b36dadf0dd19f'
acceptance=json.loads(acceptanceraw)
assert acceptance['status']=='HYBRID_ALL625_UPDATES_ANCHOR_ADDITION_RELOAD_AUDIT_PASS'
assert acceptance['spf_projection_rank']==40 and acceptance['all625_rank_boundaries_verified']
assert acceptance['updates']==625 and acceptance['method_family']=='spf_anchor_loss_hybrid' and acceptance['safety_loss_coefficient']==.1
assert acceptance['engineering_proof_sha256']=='a68b9782be482c1b6c160a1c88beafb747f8805492a6d6e70cf3b9d7de64844c'
assert acceptance['candidate_audit_sha256']==sha(auditraw) and acceptance['original_large_autograd_failures_resolved'] is False
state=json.loads((C/'trainer_state.json').read_bytes())
assert state['global_step']==625 and state['epoch']==5
resolved=json.loads((C/'resolved_training.json').read_bytes())
assert resolved['projection']['rank']==40
assert resolved['args']['adam_beta1']==.5 and resolved['args']['adam_beta2']==.999
assert resolved['args']['learning_rate']==5e-6 and resolved['args']['weight_decay']==.01
binding=json.loads((T/'optimizer-binding.private.json').read_bytes())
assert binding['status']=='ACTUAL_HYBRID01_RANK40_FULL_OPTIMIZER_BOUND'
assert binding['spf_projection_rank']==40 and binding['actual_nonprofile_method_args_match']
assert binding['engineering'] is False and binding['actual_nonprofile_training_args_match'] and binding['original_training_loop_delegated']
assert binding['beta1']==.5 and binding['beta2']==.999 and binding['learning_rate']==5e-6
assert binding['hybrid']['method_family']=='spf_anchor_loss_hybrid' and binding['hybrid']['safety_loss_coefficient']==.1
assert binding['wrapper_sha256']=='4cc6b969be7a3fb52714a5e20c4c54c53b2f4b6f843d8549738d06685c05ca01'
assert binding['plan_sha256']=='49ce276792f785c05e102f0c82355ea20e4f7806d86459448ff3e3c9359cee6b'
case=json.loads((T/'candidate-identity.private.json').read_bytes())
assert case['candidate_audit_sha256']==sha(auditraw) and case['step']==625 and case['checkpoint']==str(C)
assert case['weights_sha256']==audit['final_checkpoint']['files'][0]['sha256']
identity={k:v for k,v in case.items() if k!='identity_sha256'}
assert sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())==case['identity_sha256']
trajectory=[json.loads(line) for line in (C/'trajectory.jsonl').read_bytes().splitlines()][1:]
assert len(trajectory)==625
for step,row in enumerate(trajectory,1):
    assert row['spf_projection_rank']==40 and row['declared_projection_rank_verified_before_original_combine']
    assert row['step']==row['engine_global_steps']==row['scheduler_steps']==step
    assert row['method_family']=='spf_anchor_loss_hybrid' and row['safety_loss_coefficient']==.1
    assert row['anchor_gradient_added_before_original_clip_and_step'] and row['projection_before_anchor_addition']
    assert row['anchor_loss_reduction']=='assistant_token_mean' and row['backward_scale_wrt_gas'] is False
    assert row['extra_model_forwards']==row['extra_model_backwards']==0
files={T/name for name in ('spf-hybrid01-rank40-full-s0-audit.json','hybrid-full-acceptance.private.json','optimizer-binding.private.json','resolved-candidate-plan.private.json','candidate-identity.private.json')}
for name in ('target.json','resolved_training.json','trainer_state.json','trajectory.jsonl','config.json','generation_config.json','tokenizer_config.json'):
    files.add(C/name)
for step in (157,313,469,625):files.add(C/f'checkpoint-{step}'/'trainer_state.json')
for row in STAGES:
    path=Path(row['log']);assert path.parent==T/'runs/cpu-fixture-recovery-1539' and sha(path.read_bytes())==row['raw_log_sha256'];files.add(path)
records=[];stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for path in sorted(files):
        assert path.resolve().is_relative_to(T) and not path.is_symlink()
        raw=path.read_bytes();assert len(raw)<15000000 and path.suffix not in ('.safetensors','.bin','.pt')
        name=path.relative_to(T).as_posix();records.append(dict(name=name,remote_path=str(path),size=len(raw),sha256=sha(raw)))
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
raw=stream.getvalue()
print(json.dumps(dict(status='IMMUTABLE_HYBRID_COMPLETED_FULL_TRAIN_RELOAD_AUDIT_EXPORT',files=records,
    archive_sha256=sha(raw),archive_base64=base64.b64encode(raw).decode(),completed_train_audit=audit,
    candidate_identity=case,optimizer_binding=binding,full_hybrid_acceptance=acceptance,all625_hybrid_updates_verified=True,
    weights_read=False,actual_model_eval_repeated=0,paid_api_calls=0,credential_reads=0,
    engineering_raw_or_sources_reexported=False,evaluation_progress_queried=False,progress_snapshot=False)))
