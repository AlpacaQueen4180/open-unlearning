"""New adapter checks only: source parity, world1 guards, and incomplete-audit refusal."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

repo = Path.cwd()
path = repo / 'scripts/reproduction/safety/gb200/prepare_development_adapter.py'
spec = importlib.util.spec_from_file_location('development_adapter', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source_checks = {}
for name, expected in module.SOURCE_SHA256.items():
    raw = (repo / 'scripts/reproduction/safety' / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    adapted, audit = module.adapt_source(name, raw, target="/data/SPF target's checkpoint",
                                       identity='a' * 64, repository='/data/fixed SPF repo')
    original_tree, adapted_tree = ast.parse(raw), ast.parse(adapted)
    def calls(tree, owner):
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == owner and node.func.attr == 'from_pretrained']
    for owner in ('AutoTokenizer', 'GenerationConfig'):
        assert [ast.dump(node) for node in calls(original_tree, owner)] == [ast.dump(node) for node in calls(adapted_tree, owner)]
    loads = calls(adapted_tree, 'AutoModelForCausalLM')
    if name != 'score_development_ifbench.py':
        assert len(loads) == 1 and isinstance(loads[0].args[0], ast.Constant)
        assert loads[0].args[0].value == "/data/SPF target's checkpoint"
        revision = next(item for item in loads[0].keywords if item.arg == 'revision')
        assert isinstance(revision.value, ast.Constant) and revision.value.value is None
        assert audit['exact_source_restored_after_inverse_edits'] is True
        main = next(node for node in adapted_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        assert isinstance(main.body[0], ast.If)
        guard = compile(ast.fix_missing_locations(ast.Module(body=[main.body[0]], type_ignores=[])), name, 'exec')
        environments = [({}, False), ({'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}, False),
                        ({'WORLD_SIZE': '2'}, True), ({'WORLD_SIZE': '1', 'RANK': '1'}, True),
                        ({'WORLD_SIZE': '1', 'LOCAL_RANK': '1'}, True)]
        for environment, should_fail in environments:
            failed = False
            try: exec(guard, {'os': SimpleNamespace(environ=environment)})
            except ValueError: failed = True
            assert failed is should_fail
    else:
        assert adapted == raw and not audit['edits'] and not loads
    try: module.adapt_source(name, raw + b'\n', target='/data/target', identity='a' * 64, repository='/data/repo')
    except ValueError: pass
    else: raise AssertionError('Source drift accepted')
    source_checks[name] = dict(source_sha256=expected, inverse_source_byte_parity=True,
                              tokenizer_generation_config_ast_unchanged=True,
                              local_checkpoint_load_ast_verified=name != 'score_development_ifbench.py',
                              world1_guards_verified=name != 'score_development_ifbench.py')

# Synthetic metadata exercises refusal. It is not a real checkpoint audit result.
fixture = dict(status='pass', method='spf', split='full', construction_seed=0, actual_epochs=5.0,
               actual_updates=625, actual_examples=20000, unique_examples=4000,
               fresh_reload=dict(status='pass'),
               final_checkpoint=dict(parameter_tensors=291, layers=32,
                                     files=[dict(name='model.safetensors', sha256='b' * 64, size=100)]))
module.validate_target_audit(fixture)
invalid = []
for field, value in dict(status='failed', method='standard', split='retain95', construction_seed=1,
                         actual_epochs=4.99, actual_updates=624, actual_examples=19999, unique_examples=3999).items():
    altered = copy.deepcopy(fixture); altered[field] = value; invalid.append((field, altered))
for field, value in dict(parameter_tensors=290, layers=31, files=[]).items():
    altered = copy.deepcopy(fixture); altered['final_checkpoint'][field] = value; invalid.append((field, altered))
altered = copy.deepcopy(fixture); altered['fresh_reload']['status'] = 'failed'; invalid.append(('reload', altered))
altered = copy.deepcopy(fixture); altered['final_checkpoint']['files'] *= 2; invalid.append(('duplicate_shard', altered))
for field, value in [('name', '../model.safetensors'), ('name', '..\\model.safetensors'), ('sha256', 'x' * 64)]:
    altered = copy.deepcopy(fixture); altered['final_checkpoint']['files'][0][field] = value; invalid.append((field, altered))
for label, altered in invalid:
    try: module.validate_target_audit(altered)
    except ValueError: pass
    else: raise AssertionError('Incomplete audit accepted: ' + label)
assert hashlib.sha256((repo / 'configs/model/Llama-3.1-8B-Instruct.yaml').read_bytes()).hexdigest() == module.TEMPLATE_SHA256
result = dict(status='pass', scope='new static development adapter verification only',
              adapter_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              execution_sources=source_checks, incomplete_audit_cases_rejected=len(invalid),
              checkpoint_fixture='synthetic metadata only; no checkpoint created or validated',
              template_sha256=module.TEMPLATE_SHA256, target_evaluation_executed=False,
              gpu_runtime_validation='not_run', spf_numerical_acceptance='not_tested_here',
              external_judge_executed=False, original_completed_tests_rerun=False,
              limitations=['Does not validate Transformers5.5.4 GPU generation or SPF target availability',
                           'Single GPU evaluation and post-training scores remain pending'])
out = repo / 'results/reproduction/safety/spf-npo-gb200-20261006/development-adapter-static-20261006.json'
out.write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps({key: result[key] for key in ('status', 'scope', 'adapter_source_sha256', 'incomplete_audit_cases_rejected', 'gpu_runtime_validation', 'target_evaluation_executed')}))
