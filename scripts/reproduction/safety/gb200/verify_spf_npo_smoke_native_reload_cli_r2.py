"""Reload an already completed smoke in its original native DeepSpeed context.

The prior plain-HF failure remains immutable. This process performs no training,
optimizer construction, parameter update, manual rotary cast, or TF32 change.
DeepSpeed's original initialization is observable, including its module dtype
conversion; acceptance still requires the original captured logits and tolerance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

TASK = Path('/data/spf-npo-smoke-20261007-r1')
RUN_ID = 'reload-native-cli-1936'
CONTRACT_SHA = '958d8ae4a54b1c355443f67b731cdf04b65aec0f7128b723297746187c0353a7'
PROOFS = {
    'spf-npo-runtime.private.json': 'bf2ea667edf6772c86e1033aaedff3c71f7d8ee22a9279747c8fc053f787f88b',
    'trainer_state.json': 'd8c0278857ce75afae182083b0de84d2ada18b04faaf2a5252673a3d3ac42a0a',
    'spf-npo-reload-capture.json': 'a7eea385d0f9fd455ccbd911952f430ad2aff8529054a08d3782e06bd94fd045',
    'spf-npo-fresh-reload-audit.json': '6e23113e71715a1e9dc75245ebf7a9d62af1067e8a21ec02454456e778ed1a02',
    'reload-probe.private.pt': 'fca2cbb38de8f2481fa852199bdd4c01ba9529c8fef3dc5a715c10c5e6741193',
    'reload-comparison.private.pt': '01eaaeeb72bb53cbb12bb543efecf1474432d2043f86875e846030707389c242',
    'config.json': 'c9b2ececc4bfdad43b49faff7c48789b8afd5cbbc386b1ca0e2f43a3e292b475',
}
SITE = Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages')
SOURCES = {
    'deepspeed/runtime/engine.py': 'b4853189c5c5f5b3669a482778a2f44864cab7c0e863cf91e158ec4d9a955b9d',
    'deepspeed/runtime/zero/stage3.py': 'ca48f69f2698386d6f939c3e4d6fd5e6c70bbc33ed99baf572989f1407053485',
    'transformers/models/llama/modeling_llama.py': '46d313ec0f1116bcc1bb7f2379467ab249c140415c64fb758e2d5915cd8b89a5',
    'transformers/modeling_rope_utils.py': '55edda248757ae2ab38ee35bb79a4cb97660e31e40d95dc421e3863e43097930',
    'transformers/modeling_utils.py': 'b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_json(path, expected):
    if sha(path) != expected:
        raise ValueError('Preserved source/proof changed: '+str(path))
    return json.loads(Path(path).read_bytes())


def buffers(model):
    return {name: dict(shape=list(b.shape), dtype=str(b.dtype))
            for name, b in model.named_buffers()}


def resolved_reference_config(config, hidden_size):
    """Resolve the three original HF auto fields, then use native reference prep."""
    config = json.loads(json.dumps(config))
    z = config['zero_optimization']
    values = dict(reduce_bucket_size=hidden_size**2,
                  stage3_prefetch_bucket_size=0.9*hidden_size**2,
                  stage3_param_persistence_threshold=10*hidden_size)
    if (config['bf16'] != {'enabled': True} or z['stage'] != 3
            or config['train_batch_size'] != 32
            or config['train_micro_batch_size_per_gpu'] != 4
            or config['gradient_accumulation_steps'] != 8
            or 'optimizer' in config or 'scheduler' in config):
        raise ValueError('Original optimizer-free native reference context required')
    for name, value in values.items():
        if z[name] != 'auto':
            raise ValueError('Original auto field changed: '+name)
        z[name] = value
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    run = Path(a.run)
    if (run != TASK/'runs'/RUN_ID or run.is_symlink() or not run.is_dir()
            or sha(__file__) != a.source_sha256
            or sys.executable != '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'):
        raise ValueError('Bound new reload journal/source/interpreter required')
    for k, value in [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0'),
                     ('HF_HUB_OFFLINE', '1'), ('HF_DATASETS_OFFLINE', '1'),
                     ('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')]:
        if os.environ.get(k) != value:
            raise ValueError('Original world1/offline environment required: '+k)
    if (run/'reload-audit.private.json').exists():
        raise FileExistsError('Never repeat or replace a reload result')
    c = bound_json(TASK/'contract.private.json', CONTRACT_SHA)
    target = Path(c['output_dir'])
    if target != TASK/'checkpoint':
        raise ValueError('Completed checkpoint identity changed')
    for name, digest in PROOFS.items():
        if sha(target/name) != digest:
            raise ValueError('Completed train or failed comparison changed: '+name)
    for name, digest in c['code_sha256'].items():
        if sha(TASK/'code'/name) != digest:
            raise ValueError('Sealed observer/reload source changed')
    for name, digest in c['native_source_sha256'].items():
        if sha(Path(c['repository'])/name) != digest:
            raise ValueError('Native model/trainer source changed')
    for name, digest in SOURCES.items():
        if sha(SITE/name) != digest:
            raise ValueError('Installed native reload source changed: '+name)
    sys.path.insert(0, str(TASK/'code'))
    from spf_npo_runtime import audit_trace
    from prepare_spf_npo_smoke import validate_contract_config
    raw_config = json.loads((TASK/'configs/spf_npo_smoke.yaml').read_bytes())
    ds_config = json.loads((TASK/'deepspeed.json').read_bytes())
    validate_contract_config(c, raw_config, ds_config)
    runtime = json.loads((target/'spf-npo-runtime.private.json').read_bytes())
    audit_trace(runtime)
    capture = json.loads((target/'spf-npo-reload-capture.json').read_bytes())
    previous = json.loads((target/'spf-npo-fresh-reload-audit.json').read_bytes())
    if (previous['status'] != 'FAILED_RELOAD' or previous['fresh_process_reload_verified']
            or capture['capture_process_pid'] == os.getpid()
            or capture['comparison'] != dict(atol=0.05, rtol=0.01)):
        raise ValueError('Only the preserved failed independent reload may recover')
    # Reuse the full byte audit already produced by PID3501. No repeated 16GB audit.
    export = previous['final_checkpoint']
    if (export['parameter_tensors'] != 291 or export['layers'] != 32
            or export['weights_bytes'] != 16060556616 or len(export['files']) != 1
            or export['files'][0]['sha256'] != '84f3bbaa85abeaf0214d2a9cfe89f407c6a5954072e07b9de99475db47b32f69'):
        raise ValueError('Complete preserved HF export audit required')
    weight = target/export['files'][0]['name']
    before_stat = weight.stat()
    if before_stat.st_size != export['files'][0]['size']:
        raise ValueError('Audited checkpoint size changed')
    with weight.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_size < 10*1024**2:
            raise ValueError('Invalid exported tensor header')
        header = json.loads(stream.read(header_size))
    expected_names = set(header)-{'__metadata__'}
    if len(expected_names) != 291:
        raise ValueError('Complete 291 parameter header required')
    import torch
    import deepspeed
    import transformers
    if (transformers.__version__ != '5.5.4' or deepspeed.__version__ != '0.15.4'
            or torch.cuda.device_count() != 1):
        raise ValueError('Pinned single-GPU native reload required')
    def context():
        return dict(torch_version=torch.__version__, transformers_version=transformers.__version__,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    tf32_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
    initial_context = context()
    if initial_context != capture['numerical_context']:
        raise ValueError('Original numerical context required without changing settings')
    probe = torch.load(target/'reload-probe.private.pt', map_location='cpu', weights_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(target,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', local_files_only=True)
    if set(name for name, _ in model.named_parameters()) != expected_names:
        raise ValueError('Actual loaded 291 parameter names differ')
    loaded_buffers = buffers(model)
    # This is the unchanged native method used to prepare the original reference.
    # Its DeepSpeedEngine performs its own BF16 setup; no custom buffer mutation.
    sys.path.insert(0, c['repository']+'/src')
    from trainer.unlearn.base import UnlearnTrainer
    config = resolved_reference_config(ds_config, model.config.hidden_size)
    host = SimpleNamespace(accelerator=SimpleNamespace(state=SimpleNamespace(
        deepspeed_plugin=SimpleNamespace(deepspeed_config=config))))
    engine = UnlearnTrainer._prepare_deepspeed(host, model)
    prepared_buffers = buffers(engine.module)
    result = dict(status='FAILED_NATIVE_CONTEXT_RELOAD', pid=os.getpid(),
        capture_process_pid=capture['capture_process_pid'], source_sha256=a.source_sha256,
        contract_sha256=CONTRACT_SHA, comparison=capture['comparison'],
        original_plain_hf_reload_failure_preserved=True, export_audit_reused=export,
        checkpoint_byte_audit_repeated=False, loaded_buffers=loaded_buffers,
        native_prepared_buffers=prepared_buffers, numerical_context=context(),
        native_reference_prepare_source_sha256=c['native_source_sha256']['src/trainer/unlearn/base.py'],
        native_engine_source_sha256=SOURCES['deepspeed/runtime/engine.py'],
        manual_rotary_buffer_mutation=False, numerical_settings_changed=False,
        native_engine_bf16_setup=True, optimizer_created=engine.optimizer is not None,
        training_steps_executed=0, backward_calls=0, pilot_candidates_frozen=False,
        fresh_process_reload_verified=False)
    try:
        if engine.optimizer is not None or engine.global_steps != 0:
            raise ValueError('Reload must not construct an optimizer or perform a step')
        if context() != initial_context or prepared_buffers != capture['buffers']:
            raise ValueError('Native context and captured buffer metadata must match before forward')
        with torch.no_grad():
            logits = engine(**{k: v.cuda() for k, v in probe['inputs'].items()}).logits[0,-1].float().cpu()
        result['max_logit_error'] = float((logits-probe['logits']).abs().max())
        torch.save(dict(actual_logits=logits, expected_logits=probe['logits']),
                   run/'reload-comparison.private.pt')
        torch.testing.assert_close(logits, probe['logits'], atol=0.05, rtol=0.01)
        if (weight.stat().st_size, weight.stat().st_mtime_ns) != (before_stat.st_size, before_stat.st_mtime_ns):
            raise ValueError('Completed checkpoint metadata changed during read-only reload')
        if context() != initial_context or buffers(engine.module) != prepared_buffers:
            raise ValueError('Numerical settings or buffer metadata changed during forward')
        result.update(status='NATIVE_CONTEXT_SMOKE_FRESH_RELOAD_PASS',
                      fresh_process_reload_verified=True)
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        with (run/'reload-audit.private.json').open('x', encoding='utf8') as stream:
            stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status','max_logit_error','fresh_process_reload_verified')}))


if __name__ == '__main__':
    main()
