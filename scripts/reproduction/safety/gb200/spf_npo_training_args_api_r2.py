"""Source-bound Transformers 5.5.4 safetensors argument compatibility entrypoint.

The sealed config still requests save_safetensors=True. Only that removed
constructor keyword is omitted; this installed model save implementation always
uses safetensors. No Trainer/NPO arithmetic, weights or precision is patched.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

TASK = Path('/data/spf-npo-smoke-20261007-r1')
SITE = Path('/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers')
API_SHAS = {'training_args.py': '67e63d9b8a68a1547c0f3d17eac51034592da669d01cf03b4471b0b77f22a756',
            'trainer.py': 'e680536f179b7e2000188a671c0317c7fdce72cd4ceaf91dd72c2032f8c828d0',
            'modeling_utils.py': 'b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16'}
CONFIG_SHA = '2c753b5f656820ec7dffddfc57170845daf02a078462e18d98c793e3d9a1288b'
PROBE_SHA = '92edf63b14296d591e8e7cae4036e37e635280829e0a358ac873ac0a3e6ab5c1'
NATIVE_SHA = '714b61fea7afc2248739800baf9e2c76feee5111797ed83f49517a70187e1a5a'


IMPORT_TRACE_CAPTURE_SHA = '55e417fcb14e26f3093bfbdedcb5a2f048dadccf77458258191ff49eb050726a'
IMPORT_SOURCES = {'/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py': {'size': 114161, 'sha256': 'c05e6063736902b07d0b48e4a326ae7985b75430bacbee48014c954fd4e6cb05'}, '/usr/lib/python3.12/importlib/__init__.py': {'size': 4774, 'sha256': 'c9e1b3dbc619ac31e7017ac43668a20200872c1c0e79ae379c0dab6ed399b730'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers/training_args.py': {'size': 140158, 'sha256': '67e63d9b8a68a1547c0f3d17eac51034592da669d01cf03b4471b0b77f22a756'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers/trainer_utils.py': {'size': 49862, 'sha256': '951ebe7d0235be1de166ff17e0a5d45e6bcca4ea25be1ef93a4d777b680052d6'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/__init__.py': {'size': 8013, 'sha256': '0d75f5e7919bbc0906ba285d5b5309b0d49a16a9cd87524225eae1cd480f79af'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/auto.py': {'size': 9306, 'sha256': 'cb31d32a9bbb695b09e152bd94f5885428abba0876ce702a46908df5b07acbd6'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/peft_model.py': {'size': 173182, 'sha256': 'd22a1c50dea7a046ce28e71a79dfc20b1d96a2d238a9b457be6a88b6068e04d5'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/__init__.py': {'size': 5425, 'sha256': '74079426fd0adcc50e5d781f66d71118d2b40e98a712a0a448012040bd74a872'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/adalora/__init__.py': {'size': 1534, 'sha256': 'c493f7dbda463eaea6cd1abd093af6b19ee7e3e66bd0ac62995262e5c5e1fa82'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/adalora/config.py': {'size': 5896, 'sha256': 'f1b2eea2e36c7c69c5ab3a91cde4b851e6ddf40a4a806726c2ee89af23dcdc88'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/lora/__init__.py': {'size': 2358, 'sha256': '010dabf75b2d4c08e66f91c337be13a50d75b0aecf06ac41f36b308ffc5d37a2'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/lora/loraga.py': {'size': 7663, 'sha256': '9d48cdeb6e07711749aab39428b42b2c96b341c32aaa57ad62f8b08a46e4bb79'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/lora/model.py': {'size': 66161, 'sha256': '321964321e14ca0d9b4d81c4debcb486bae02193a576755b116ce35252e33555'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/tuners/lora/te.py': {'size': 4522, 'sha256': '339b2a5d979997d3fb2d18f79b1ff36ea3ded94359dc537fbdb0acee9f453686'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/import_utils.py': {'size': 7866, 'sha256': '4f27c2625f2c8d5cd4a32dc3efa7b747e317b9aaf2d405a8a77ae027d7b37fcd'}, '/usr/local/lib/python3.12/dist-packages/transformer_engine/__init__.py': {'size': 535, 'sha256': 'f18cc61b04d4958d5d01bd666ead3ff8e6fc5fa9da41e04798007d6c93f97369'}, '/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/__init__.py': {'size': 3770, 'sha256': 'dc437e3d9913e70ebaa4a9db11bf7dc4005d037579365e84fb73902574d28dfe'}, '/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py': {'size': 385722, 'sha256': 'ae489289d1962b997dc42b47e40f5d5accbf636c63454906c614667606946786'}, '/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/utils.py': {'size': 10842, 'sha256': 'fd7ec636e0c971e9bc7e9d9a5542c25d3551272432025f690cf1d1aea0ca3226'}, '/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py': {'size': 58288, 'sha256': '4091351b8a96fd8f015f9af160716a4102f4f6aec20d73b35746209912075034'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/peft/helpers.py': {'size': 22216, 'sha256': 'b8d66e60246cc547122dd798fdf7c814ba1c48c2837e68c751164a54d87ac9dc'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/bitsandbytes/__init__.py': {'size': 2222, 'sha256': 'a4e3e6bfc4cf9eff68b95eca823c4fa5e8402fe453e3be1eff4d48c4838cefa0'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/bitsandbytes/autograd/_functions.py': {'size': 17700, 'sha256': '77ca54c87b69ad470ac73e5d7dde94948f20de9de8b8764957728bc29831f37f'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/bitsandbytes/functional.py': {'size': 65287, 'sha256': '520a9e9d3afba11124734df60e48a27e38b6389305df888834405b435e285817'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/bitsandbytes/cextension.py': {'size': 17174, 'sha256': 'a7ce4d7bcb504bf7d268ce96658dbe832804b6f6ed999f61eaf2beeb9385c9bd'}, '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/bitsandbytes/cuda_specs.py': {'size': 3236, 'sha256': '6a4153476671439085a452de83940a15b54e678c73dd449ac78dd1c5accfdb69'}}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_config():
    raw = (TASK/'configs/spf_npo_smoke.yaml').read_bytes()
    if digest(raw) != CONFIG_SHA:
        raise ValueError('Original sealed smoke config required')
    return json.loads(raw)['trainer']['args']


def map_args(values, expected, arguments_class):
    if values != expected or values.get('save_safetensors') is not True:
        raise ValueError('Exactly the sealed arguments and safetensors=True required')
    signature = inspect.signature(arguments_class)
    if 'save_safetensors' in signature.parameters:
        raise ValueError('This compatibility entrypoint only supports the pinned removed API')
    mapped = dict(values)
    del mapped['save_safetensors']
    signature.bind(**mapped)
    return mapped


def audit_api(expected, *, bound_gpu_import=False):
    sources = {}
    for name, sha in API_SHAS.items():
        raw = (SITE/name).read_bytes()
        if digest(raw) != sha:
            raise ValueError('Installed serialization/TrainingArguments API source changed: '+name)
        sources[name] = raw.decode('utf8')
    import torch
    import transformers
    from transformers import TrainingArguments
    if transformers.__version__ != '5.5.4':
        raise ValueError('Pinned Transformers required')
    mapped = map_args(expected, expected, TrainingArguments)
    try:
        inspect.signature(TrainingArguments).bind(**expected)
    except TypeError:
        pass
    else:
        raise ValueError('Original removed keyword must fail signature binding')
    # These exact implementations were exported from the installed venv. The
    # source SHA binds the complete functions, including all control flow.
    if ('save_safetensors' in sources['trainer.py']
            or 'weights_name = SAFE_WEIGHTS_NAME' not in sources['modeling_utils.py']
            or 'safe_save_file(shard_state_dict, filename, metadata=metadata)' not in sources['modeling_utils.py']):
        raise ValueError('Installed mandatory safetensors serialization proof changed')
    if torch.cuda.is_initialized() and not bound_gpu_import:
        raise ValueError('Compatibility preflight must not initialize CUDA')
    return dict(status='TRAINING_ARGUMENTS_SIGNATURE_AND_SERIALIZATION_SOURCE_PASS',
                installed_api_sha256=API_SHAS, omitted_keywords=['save_safetensors'],
                omitted_value=True, remaining_arguments_exact=True,
                config_sha256=CONFIG_SHA, signature_bound=True,
                training_arguments_instantiated=False, cuda_initialized=torch.cuda.is_initialized(),
                weights_read=False, generation_calls=0, npo_training_executed=False)


def main():
    expected = bound_config()
    if sys.argv[1:] == ['--diagnose']:
        raise ValueError('Completed CPU diagnostic is reused; never rerun it through this entrypoint')
    original = TASK/'validation-code/spf_npo_training_args_api.py'
    if digest(original.read_bytes()) != '430feb8b97a3fb0dea5a9f293b50394300e0a947d5a0e2c0875f5c716e2fae52':
        raise ValueError('Preserved original API helper changed')
    for name, record in IMPORT_SOURCES.items():
        raw=Path(name).read_bytes()
        if len(raw)!=record['size'] or digest(raw)!=record['sha256']:
            raise ValueError('Actual traced import dependency changed: '+name)
    import torch
    if torch.cuda.is_initialized() or torch.cuda.device_count()!=1:
        raise ValueError('Fresh single-GPU process before bound import required')
    precision=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    proof = audit_api(expected,bound_gpu_import=True)
    if not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()!=0:
        raise ValueError('Exactly the observed capability-query context with zero allocated tensors required')
    if precision!=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32):
        raise ValueError('Import must preserve precision settings')
    import_proof=dict(status='ACTUAL_BOUND_GPU_IMPORT_CONTEXT_PASS',pid=os.getpid(),cuda_initialized_before=False,cuda_initialized_after=True,peak_allocated_bytes=0,precision_unchanged=True,weights_read=False,training_arguments_instantiated=False,npo_training_executed=False,independent_trace_capture_sha256=IMPORT_TRACE_CAPTURE_SHA)
    import_receipt=Path(os.environ['SPF_NPO_TRAINING_ARGS_API_RECEIPT']).with_name('training-args-import-preflight.private.json')
    if import_receipt.parent!=TASK/'runs/import-recovery-1257' or import_receipt.exists():
        raise ValueError('Explicit exclusive import preflight receipt required')
    import_receipt.write_text(json.dumps(import_proof)+chr(10),encoding='utf8')
    if sys.argv[1:] == ['--diagnose']:
        print(json.dumps(proof))
        return
    if sys.argv[1:] != ['--config-path',str(TASK/'configs'),'--config-name','spf_npo_smoke']:
        raise ValueError('Original native config arguments required')
    probe = TASK/'code/spf_npo_native_probe.py'
    if digest(probe.read_bytes()) != PROBE_SHA or (TASK/'checkpoint').exists():
        raise ValueError('Original observer and absent checkpoint required')
    repository = Path('/data/spf-development-20261007-r1/repo')
    if digest((repository/'src/trainer/__init__.py').read_bytes()) != NATIVE_SHA:
        raise ValueError('Original native trainer loader required')
    if digest(Path(__file__).read_bytes()) != os.environ['SPF_NPO_TRAINING_ARGS_API_SHA256']:
        raise ValueError('New compatibility entrypoint SHA required')
    receipt = Path(os.environ['SPF_NPO_TRAINING_ARGS_API_RECEIPT'])
    if receipt.parent != TASK/'runs/import-recovery-1257' or receipt.exists():
        raise ValueError('Explicit new exclusive API binding receipt required')
    sys.path.insert(0, str(repository/'src'))
    sys.path.insert(0, str(TASK/'code'))
    import trainer
    from transformers import TrainingArguments
    if trainer.TrainingArguments is not TrainingArguments:
        raise ValueError('Native TrainingArguments binding was already replaced')

    def compatible_arguments(**values):
        mapped = map_args(values, expected, TrainingArguments)
        result = TrainingArguments(**mapped)
        with receipt.open('x', encoding='utf8') as handle:
            handle.write(json.dumps(dict(status='ACTUAL_TRAINING_ARGUMENTS_API_BINDING_PASS',
                pid=os.getpid(), source_sha256=os.environ['SPF_NPO_TRAINING_ARGS_API_SHA256'],
                original_native_probe_sha256=PROBE_SHA, installed_api_sha256=API_SHAS,
                config_sha256=CONFIG_SHA, omitted_keywords=['save_safetensors'],
                remaining_arguments_exact=True, training_arguments_class_unchanged=True,
                npo_arithmetic_changed=False, precision_changed=False,import_preflight=import_proof))+chr(10))
        return result

    trainer.TrainingArguments = compatible_arguments
    # The original observer retains its complete guards and executes the same
    # native train.py. Only its loader's constructor binding above is adapted.
    runpy.run_path(str(probe), run_name='__main__')


if __name__ == '__main__':
    main()
