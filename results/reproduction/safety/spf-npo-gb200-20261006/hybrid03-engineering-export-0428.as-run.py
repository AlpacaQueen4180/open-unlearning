"""Immutable export of only four newly completed hybrid engineering stages; no live training query."""
from pathlib import Path
import base64,hashlib,io,json,tarfile
T=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')
V=T/'validation-code';E=T/'engineering-spf-hybrid03-s0'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
PINS={'run_spf_hybrid03_lr5e6_full.py': 'e30e9e059d96cc841030df47b9f16b94450388068857a2084819f4593ad4aa37', 'run_spf_hybrid03_lr5e6_full_queue.py': '65305a2ba389818797b708a48db75d2dd1fa7a56d64b87de8e1d4d06069bdb47', 'spf_hybrid03_lr5e6_development_executor.py': '2bd673f0bca4d5e0ca85f4bb8e08a7003e4350d8fd8cef95c8a652e51119ce62', 'spf_anchor_hybrid.py': 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', 'verify_spf_anchor_hybrid03.py': '44da15a0287817ab4f20759890eee0f07fc5100d5258e2f301726c766bf2edd4', 'plan.private.json': '53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997', 'bound-snapshot.private.json': 'c0cf4fc7c1b74e56d2b472adff54ef1f75b666f4009092f9ffd94447aaac5e1e'}
PROOF_PINS={'hybrid-cpu-proof.private.json': 'a30986ee9fd7ccdecdbf85af09d10f2dfeba987b1b2366e3b2378ec3fb8be7e6', 'hybrid-engineering-proof.private.json': 'e2117ce5f8b045b5a21f45d6438a35745e88edbb14021f794542f6a912c11ace'}
STAGES=[{'name': 'hybrid-cpu-verification', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/verify_spf_anchor_hybrid03.py', '--helper', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/spf_anchor_hybrid.py', '--helper-sha256', 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', '--output', '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-cpu-proof.private.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-cpu-verification.raw.log', 'started_at': 1791747335.2086644, 'process_pid': 16736, 'returncode': 0, 'finished_at': 1791747338.5940971, 'raw_log_sha256': '68359edd380ef7c1e4eaa6710b104f17bf90b4b2306d9c3bf75c61f66b323240'}, {'name': 'hybrid-engineering-train', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1', '--', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/run_spf_hybrid03_lr5e6_full.py', '--source-sha256', 'e30e9e059d96cc841030df47b9f16b94450388068857a2084819f4593ad4aa37', '--plan-sha256', '53c070938dca650897b8e873d7933e0698ba208e11b1f7e9800fdb1c06397997', '--engineering'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-engineering-train.raw.log', 'started_at': 1791747338.6284702, 'process_pid': 16947, 'returncode': 0, 'finished_at': 1791747445.2735803, 'raw_log_sha256': '6bc8149561288e0b1b4af7d88d03a46dc63fbaa828c975ccd1335b08dcdf1b89'}, {'name': 'hybrid-engineering-reload', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '-m', 'construction.cli', 'verify', '--checkpoint', '/data/spf-hybrid03-lr5e6-full-20261012-r1/engineering-spf-hybrid03-s0'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-engineering-reload.raw.log', 'started_at': 1791747445.2916965, 'process_pid': 17483, 'returncode': 0, 'finished_at': 1791747456.0716448, 'raw_log_sha256': 'e0ed4586481714f66a479e2163cd57235b89871b0d1e529336a2c4425142c1d4'}, {'name': 'hybrid-engineering-verify', 'command': ['/data/npo-gb200-20261004/validation-20261006/venv/bin/python', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/verify_spf_anchor_hybrid03.py', '--helper', '/data/spf-hybrid03-lr5e6-full-20261012-r1/validation-code/spf_anchor_hybrid.py', '--helper-sha256', 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc', '--engineering-target', '/data/spf-hybrid03-lr5e6-full-20261012-r1/engineering-spf-hybrid03-s0', '--output', '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-engineering-proof.private.json'], 'cwd': '/data/spf-npo-20261006-r4/repo', 'environment_overrides': {'HF_HOME': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_CACHE': '/data/smart-mfg/jimmy-lin/hf_cache/hub', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'PYTHONPATH': '/data/spf-npo-20261006-r4/repo/src', 'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled', 'CUDA_VISIBLE_DEVICES': ''}, 'log': '/data/spf-hybrid03-lr5e6-full-20261012-r1/hybrid-engineering-verify.raw.log', 'started_at': 1791747456.0899127, 'process_pid': 17763, 'returncode': 0, 'finished_at': 1791747456.1528177, 'raw_log_sha256': 'df918604866087330c6b173da4cbf451769b29785bb7581c1fb2cc6f7e21346b'}]
files=set()
for name,digest in PINS.items():
    path=V/name
    assert sha(path.read_bytes())==digest
    files.add(path)
files.add(V/'hybrid03-lr5e6-full-launch-20261012.json')
for name,digest in PROOF_PINS.items():
    path=T/name
    assert sha(path.read_bytes())==digest
    files.add(path)
cpu=json.loads((T/'hybrid-cpu-proof.private.json').read_bytes())
engineering=json.loads((T/'hybrid-engineering-proof.private.json').read_bytes())
assert cpu['status']=='HYBRID_CPU_DIRECT_AUTOGRAD_TOKEN_MEAN_TWO_ADAM_UPDATES_RELOAD_PASS'
assert cpu['actual_cpu_updates']==2 and cpu['cuda_model_work']==0 and cpu['safety_loss_coefficient']==.3
assert engineering['status']=='HYBRID_REAL8B_ZERO3_TWO_UPDATES_TOKEN_MEAN_TAIL_RELOAD_PASS'
assert engineering['updates']==2 and engineering['window_examples']==[32,8] and engineering['microbatches']==[8,2] and engineering['safety_loss_coefficient']==.3
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
assert binding['hybrid']['method_family']=='spf_anchor_loss_hybrid' and binding['hybrid']['safety_loss_coefficient']==.3
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
