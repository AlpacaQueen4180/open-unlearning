"""Immutable export of only four newly completed hybrid engineering stages; no live training query."""
from pathlib import Path
import base64,hashlib,io,json,tarfile
T=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')
V=T/'validation-code';E=T/'engineering-spf-hybrid01-s0'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
PINS={'run_spf_hybrid01_lr5e6_full.py': 'a3402bfc2bc0d31173e71e3f9240802f6e5cbdc62528e62644507e7676f4a2c8', 'run_spf_hybrid01_lr5e6_full_queue.py': '7b5f82e1f285233288845c7dcdaf15a6612a53633764ec8f8360dbde28315fc9', 'spf_hybrid01_lr5e6_development_executor.py': '27ed16eda408ede0dd5401a29fbf7bc861fc2d4e57c3fa3fc4b7e0010c290da3', 'spf_anchor_hybrid.py': 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', 'verify_spf_anchor_hybrid.py': 'd57d8ff1309c15a77beeb72ecbe0fe33a215c9b13fc8fb6f86c57ce1fffb8a73', 'plan.private.json': '4f60bd23f61c70dfcf06e909b347c07a27420ec60cecc8e9c71dc3b35a5580b5', 'bound-snapshot.private.json': '752384e32ec448c38c4dc87e7e65242bc979976ff383d75553467c17759275b4'}
PROOF_PINS={'hybrid-cpu-proof.private.json': '82038714b8837eb716bf17e80402b775007db532eb8acd61f7c020f255eda42a', 'hybrid-engineering-proof.private.json': 'a0a7bd95f69d6f14ce40115ce281c6c7effc9e0550195cafbea3a7c9228e8fae'}
STAGES=[{'name': 'hybrid-cpu-verification', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-hybrid01-lr5e6-full-20261011-r1/validation-code/verify_spf_anchor_hybrid.py', '--helper', '/data/spf-hybrid01-lr5e6-full-20261011-r1/validation-code/spf_anchor_hybrid.py', '--helper-sha256', 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', '--output', '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-cpu-proof.private.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-cpu-verification.raw.log', 'started_at': 1791718578.5771754, 'process_pid': 14036, 'returncode': 0, 'finished_at': 1791718582.4021082, 'raw_log_sha256': '5fb95acc7c9f8973e9d11725234a1dd3a7a30eeea726c0890f9d056284493aac'}, {'name': 'hybrid-engineering-train', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1', '--', '/data/spf-hybrid01-lr5e6-full-20261011-r1/validation-code/run_spf_hybrid01_lr5e6_full.py', '--source-sha256', 'a3402bfc2bc0d31173e71e3f9240802f6e5cbdc62528e62644507e7676f4a2c8', '--plan-sha256', '4f60bd23f61c70dfcf06e909b347c07a27420ec60cecc8e9c71dc3b35a5580b5', '--engineering'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-engineering-train.raw.log', 'started_at': 1791718582.4326553, 'process_pid': 14247, 'returncode': 0, 'finished_at': 1791718724.839153, 'raw_log_sha256': '433f6caed19667a5f39d95ee0af7023cafc464fccc568edcafb8da5b9914e16c'}, {'name': 'hybrid-engineering-reload', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'construction.cli', 'verify', '--checkpoint', '/data/spf-hybrid01-lr5e6-full-20261011-r1/engineering-spf-hybrid01-s0'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-engineering-reload.raw.log', 'started_at': 1791718724.8613167, 'process_pid': 14783, 'returncode': 0, 'finished_at': 1791718737.3840764, 'raw_log_sha256': '13c7fafc5974a4411f365fbf6dbc1548af3b83b3540575b04e468f75eff3747d'}, {'name': 'hybrid-engineering-verify', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-hybrid01-lr5e6-full-20261011-r1/validation-code/verify_spf_anchor_hybrid.py', '--helper', '/data/spf-hybrid01-lr5e6-full-20261011-r1/validation-code/spf_anchor_hybrid.py', '--helper-sha256', 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', '--engineering-target', '/data/spf-hybrid01-lr5e6-full-20261011-r1/engineering-spf-hybrid01-s0', '--output', '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-engineering-proof.private.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid01-lr5e6-full-20261011-r1/hybrid-engineering-verify.raw.log', 'started_at': 1791718737.4024653, 'process_pid': 15063, 'returncode': 0, 'finished_at': 1791718737.463727, 'raw_log_sha256': '3cf33fe5669158d0e739525c92605890e933a31894af74ed5fc4a145d6cfdaac'}]
files=set()
for name,digest in PINS.items():
    path=V/name
    assert sha(path.read_bytes())==digest
    files.add(path)
files.add(V/'hybrid01-lr5e6-full-launch-20261011.json')
for name,digest in PROOF_PINS.items():
    path=T/name
    assert sha(path.read_bytes())==digest
    files.add(path)
cpu=json.loads((T/'hybrid-cpu-proof.private.json').read_bytes())
engineering=json.loads((T/'hybrid-engineering-proof.private.json').read_bytes())
assert cpu['status']=='HYBRID_CPU_DIRECT_AUTOGRAD_TOKEN_MEAN_TWO_ADAM_UPDATES_RELOAD_PASS'
assert cpu['actual_cpu_updates']==2 and cpu['cuda_model_work']==0
assert engineering['status']=='HYBRID_REAL8B_ZERO3_TWO_UPDATES_TOKEN_MEAN_TAIL_RELOAD_PASS'
assert engineering['updates']==2 and engineering['window_examples']==[32,8] and engineering['microbatches']==[8,2]
assert engineering['fresh_reload']['status']=='pass' and engineering['old_large_autograd_failures_resolved'] is False
for row in STAGES:
    path=Path(row['log'])
    assert path.parent==T and sha(path.read_bytes())==row['raw_log_sha256']
    files.add(path)
for name in ('target.json','resolved_training.json','trainer_state.json','trajectory.jsonl','config.json','generation_config.json','tokenizer_config.json'):
    files.add(E/name)
for name in ('engineering-optimizer-binding.private.json','engineering-resolved-plan.private.json'):
    files.add(T/name)
assert sha((E/'trajectory.jsonl').read_bytes())==engineering['trajectory_sha256']
assert json.loads((E/'trainer_state.json').read_bytes())['global_step']==2
binding=json.loads((T/'engineering-optimizer-binding.private.json').read_bytes())
assert binding['engineering'] is True and binding['actual_nonprofile_training_args_match'] is True
assert binding['hybrid']['method_family']=='spf_anchor_loss_hybrid' and binding['hybrid']['safety_loss_coefficient']==.1
records=[];stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for path in sorted(files):
        assert path.resolve().is_relative_to(T) and not path.is_symlink()
        raw=path.read_bytes()
        assert len(raw)<15000000 and path.suffix not in ('.safetensors','.bin','.pt')
        name=path.relative_to(T).as_posix()
        records.append(dict(name=name,remote_path=str(path),size=len(raw),sha256=sha(raw)))
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
raw=stream.getvalue()
print(json.dumps(dict(status='IMMUTABLE_HYBRID_NEW_ENGINEERING_COMPLETION_EXPORT',files=records,
    archive_sha256=sha(raw),archive_base64=base64.b64encode(raw).decode(),cpu_proof=cpu,
    engineering_proof=engineering,engineering_binding=binding,completed_stage_records=STAGES,
    weights_read=False,progress_snapshot=False,training_progress_queried=False,
    engineering_repeated=0,new_api_calls=0,credential_reads=0)))
