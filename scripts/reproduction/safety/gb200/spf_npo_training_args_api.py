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


def audit_api(expected):
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
    if torch.cuda.is_initialized():
        raise ValueError('Compatibility preflight must not initialize CUDA')
    return dict(status='TRAINING_ARGUMENTS_SIGNATURE_AND_SERIALIZATION_SOURCE_PASS',
                installed_api_sha256=API_SHAS, omitted_keywords=['save_safetensors'],
                omitted_value=True, remaining_arguments_exact=True,
                config_sha256=CONFIG_SHA, signature_bound=True,
                training_arguments_instantiated=False, cuda_initialized=False,
                weights_read=False, generation_calls=0, npo_training_executed=False)


def main():
    expected = bound_config()
    proof = audit_api(expected)
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
    if receipt.parent != TASK/'runs/training-args-recovery-1225' or receipt.exists():
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
                npo_arithmetic_changed=False, precision_changed=False))+chr(10))
        return result

    trainer.TrainingArguments = compatible_arguments
    # The original observer retains its complete guards and executes the same
    # native train.py. Only its loader's constructor binding above is adapted.
    runpy.run_path(str(probe), run_name='__main__')


if __name__ == '__main__':
    main()
