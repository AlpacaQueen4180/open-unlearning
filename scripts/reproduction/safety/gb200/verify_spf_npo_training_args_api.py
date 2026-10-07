"""CPU signature/serialization source diagnostic before an explicit new train run."""
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE_SHA = '430feb8b97a3fb0dea5a9f293b50394300e0a947d5a0e2c0875f5c716e2fae52'


def main():
    source = Path(__file__).with_name('spf_npo_training_args_api.py')
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Exact new compatibility source required')
    spec = importlib.util.spec_from_file_location('bound_training_arguments_api', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    expected = m.bound_config()
    proof = m.audit_api(expected)
    from transformers import TrainingArguments
    negatives = []
    for name, change in [('false-serialization', {'save_safetensors': False}),
                         ('unknown-keyword', {'unrecognized_argument': 1}),
                         ('changed-learning-rate', {'learning_rate': 3e-7})]:
        try:
            m.map_args({**expected, **change}, expected, TrainingArguments)
        except ValueError:
            negatives.append(name)
        else:
            raise ValueError('Unexpected compatibility acceptance: '+name)
    missing = dict(expected)
    del missing['save_safetensors']
    try:
        m.map_args(missing, expected, TrainingArguments)
    except ValueError:
        negatives.append('missing-required-serialization-intent')
    else:
        raise ValueError('Missing serialization intent accepted')
    proof.update(compatibility_source_sha256=SOURCE_SHA, negative_cases=negatives,
                 subprocesses_launched=0)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
