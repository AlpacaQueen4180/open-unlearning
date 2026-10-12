"""Export only newly completed hybrid full train/reload/audit; engineering and live development excluded."""
from pathlib import Path
import base64,hashlib,io,json,tarfile
T=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1');C=T/'spf-hybrid03-lr5e6-full-s0'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
STAGES=[{'name': 'train', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1', '--', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/run_spf_hybrid03_lr5e6_full.py', '--source-sha256', 'e30e9e059d96cc841030df47b9f16b94450388068857a2084819f4593ad4aa37', '--plan-sha256', '53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/train.raw.log', 'started_at': 1791747456.1699207, 'process_pid': 17766, 'returncode': 0, 'finished_at': 1791763981.444035, 'raw_log_sha256': '09e62e3ee62d74056016dc413dcf3d3b75a121732faa078e0dad0369c7c7dee5'}, {'name': 'fresh-reload', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'construction.cli', 'verify', '--checkpoint', '/data/spf-hybrid03-lr5e6-full-20261012-r1/spf-hybrid03-lr5e6-full-s0'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/fresh-reload.raw.log', 'started_at': 1791763981.4620059, 'process_pid': 18362, 'returncode': 0, 'finished_at': 1791763995.0049007, 'raw_log_sha256': '93ba795ed040beaaad7fe1deaf09d9fedda8d6baa147c332c2c1e8fc0b0e15a5'}, {'name': 'full-byte-audit', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py', '--target', '/data/spf-hybrid03-lr5e6-full-20261012-r1/spf-hybrid03-lr5e6-full-s0', '--output', '/data/spf-hybrid03-lr5e6-full-20261012-r1/spf-hybrid03-lr5e6-full-s0-audit.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/full-byte-audit.raw.log', 'started_at': 1791763995.0230165, 'process_pid': 18642, 'returncode': 0, 'finished_at': 1791764042.399207, 'raw_log_sha256': '38163f52082cabbc39047e22c0cc28caf6d41e692b56097aa34dc88197d90fdb'}]
auditraw=(T/'spf-hybrid03-lr5e6-full-s0-audit.json').read_bytes()
assert sha(auditraw)=='d27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486'
audit=json.loads(auditraw)
assert audit['status']=='pass' and audit['actual_updates']==625 and audit['actual_epochs']==5
assert audit['actual_examples']==20000 and audit['unique_examples']==4000
assert audit['fresh_reload']['status']=='pass' and audit['fresh_reload']['max_logit_error']==0
assert audit['final_checkpoint']['parameter_tensors']==291 and audit['final_checkpoint']['weights_bytes']==16060556616
acceptanceraw=(T/'hybrid-full-acceptance.private.json').read_bytes()
assert sha(acceptanceraw)=='ea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53'
acceptance=json.loads(acceptanceraw)
assert acceptance['status']=='HYBRID_ALL625_UPDATES_ANCHOR_ADDITION_RELOAD_AUDIT_PASS'
assert acceptance['updates']==625 and acceptance['method_family']=='spf_anchor_loss_hybrid' and acceptance['safety_loss_coefficient']==.3
assert acceptance['engineering_proof_sha256']=='e2117ce5f8b045b5a21f45d6438a35745e88edbb14021f794542f6a912c11ace'
assert acceptance['candidate_audit_sha256']==sha(auditraw) and acceptance['original_large_autograd_failures_resolved'] is False
state=json.loads((C/'trainer_state.json').read_bytes())
assert state['global_step']==625 and state['epoch']==5
resolved=json.loads((C/'resolved_training.json').read_bytes())
assert resolved['args']['adam_beta1']==.5 and resolved['args']['adam_beta2']==.999
assert resolved['args']['learning_rate']==5e-6 and resolved['args']['weight_decay']==.01
binding=json.loads((T/'optimizer-binding.private.json').read_bytes())
assert binding['status']=='ACTUAL_HYBRID03_LR5E6_FULL_OPTIMIZER_BOUND'
assert binding['engineering'] is False and binding['actual_nonprofile_training_args_match'] and binding['original_training_loop_delegated']
assert binding['beta1']==.5 and binding['beta2']==.999 and binding['learning_rate']==5e-6
assert binding['hybrid']['method_family']=='spf_anchor_loss_hybrid' and binding['hybrid']['safety_loss_coefficient']==.3
assert binding['wrapper_sha256']=='e30e9e059d96cc841030df47b9f16b94450388068857a2084819f4593ad4aa37'
assert binding['plan_sha256']=='53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997'
case=json.loads((T/'candidate-identity.private.json').read_bytes())
assert case['candidate_audit_sha256']==sha(auditraw) and case['step']==625 and case['checkpoint']==str(C)
assert case['weights_sha256']==audit['final_checkpoint']['files'][0]['sha256']
identity={k:v for k,v in case.items() if k!='identity_sha256'}
assert sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())==case['identity_sha256']
trajectory=[json.loads(line) for line in (C/'trajectory.jsonl').read_bytes().splitlines()][1:]
assert len(trajectory)==625
for step,row in enumerate(trajectory,1):
    assert row['step']==row['engine_global_steps']==row['scheduler_steps']==step
    assert row['method_family']=='spf_anchor_loss_hybrid' and row['safety_loss_coefficient']==.3
    assert row['anchor_gradient_added_before_original_clip_and_step'] and row['projection_before_anchor_addition']
    assert row['anchor_loss_reduction']=='assistant_token_mean' and row['backward_scale_wrt_gas'] is False
    assert row['extra_model_forwards']==row['extra_model_backwards']==0
files={T/name for name in ('spf-hybrid03-lr5e6-full-s0-audit.json','hybrid-full-acceptance.private.json','optimizer-binding.private.json','resolved-candidate-plan.private.json','candidate-identity.private.json')}
for name in ('target.json','resolved_training.json','trainer_state.json','trajectory.jsonl','config.json','generation_config.json','tokenizer_config.json'):
    files.add(C/name)
for step in (157,313,469,625):files.add(C/f'checkpoint-{step}'/'trainer_state.json')
for row in STAGES:
    path=Path(row['log']);assert path.parent==T and sha(path.read_bytes())==row['raw_log_sha256'];files.add(path)
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
