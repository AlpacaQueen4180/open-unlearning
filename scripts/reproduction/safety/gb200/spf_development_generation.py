"""Resolve the original greedy profile without inheriting checkpoint defaults.

Transformers 5.5.4 removed use_model_defaults. Resolve its documented global
defaults explicitly, then reject any remaining checkpoint-specific fallback.
This helper never changes the model's generation_config, buffers, or weights.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

IGNORED_METADATA = {'_commit_hash', '_from_model_config', 'transformers_version'}
INSTALLED_SOURCES = {
    'generation/utils.py': 'dde2df36821c0d724b5af47cb0ff71c3c4c1990c86d81b821911127ae4dc1254',
    'generation/configuration_utils.py': '0aaf06e21d844256eee23aac663672d0aa9d06ca1f7f59e9be655c761ccdb0a0',
}


def behavior(config):
    return {k: v for k, v in config.to_dict().items() if k not in IGNORED_METADATA}


def fixed_generation(model, supplied):
    import transformers
    from transformers.generation import configuration_utils, utils
    if transformers.__version__ != '5.5.4':
        raise ValueError('This recovery requires the observed Transformers 5.5.4 API')
    installed = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                 for name, module in [('generation/utils.py', utils),
                                      ('generation/configuration_utils.py', configuration_utils)]}
    if installed != INSTALLED_SOURCES:
        raise ValueError('Observed generation API source bytes changed')
    before = copy.deepcopy(model.generation_config.to_dict())
    explicit = copy.deepcopy(supplied)
    defaults = explicit._get_default_generation_params()
    explicit.update(**defaults, defaults_only=True)
    if explicit.do_sample is not False or explicit.num_beams != 1 or explicit.use_cache is not True:
        raise ValueError('Original greedy cached generation profile required')
    prepared, unused = model._prepare_generation_config(explicit)
    if unused or behavior(prepared) != behavior(explicit):
        raise ValueError('Checkpoint generation defaults would change the explicit profile')
    if model.generation_config.to_dict() != before:
        raise ValueError('Model generation configuration changed during resolution')
    record = dict(status='EXPLICIT_GENERATION_CONFIG_RESOLVED_WITHOUT_MODEL_FALLBACK',
                  transformers_version=transformers.__version__, supplied=supplied.to_dict(),
                  explicit=explicit.to_dict(), resolved=prepared.to_dict(),
                  model_generation_config_unchanged=True, unused_model_kwargs=unused,
                  installed_sources=installed)
    destination = os.environ.get('SPF_DEVELOPMENT_GENERATION_AUDIT')
    if not destination:
        raise ValueError('An exclusive private generation configuration receipt is required')
    with Path(destination).open('x', encoding='utf8') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    return prepared
