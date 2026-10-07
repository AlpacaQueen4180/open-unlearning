"""Prepare an immutable API-only recovery of the failed development run.

No deployment, subprocess, imports of Torch, data reselection, or evaluation.
Only the observed first-stage failure with zero completed stages is supported.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ORIGINAL_CONTROLLER = '5346d5223b450496b704544b31416e7af9dfbe0853929c3390d387b969345478'
ORIGINAL_GUARD = 'd33d8b8d24f089112f74c4fe18adbb647c4e78b4e3bfecfd35bf3bce279d1ff6'
ORIGINAL_ADAPTER = 'ffb402da9ab4e390104f69bf69d060f169dc2cfb6a3978d1dcf0a3bb47a8013c'
FAILED_STATUS = '5f7c3d86378538d5888ddca2a4ac7dd15874e30b61ca9f394ad98f556560b1f3'
FAILED_ARCHIVE = 'bca2ce33d478839c7df27c3a24bf00d4ecb42bb0c5a7b71b578910cd60e5d094'
GENERATOR_NAMES = ('evaluate_construction_baseline.py', 'generate_development.py',
                   'generate_development_conversation.py')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('Observed evidence/source changed: '+str(path))
    return raw


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError('Expected exact recovery edit: '+old)
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    evidence, output = Path(a.evidence), Path(a.output)
    bound(evidence/'raw-evidence.tar.gz', FAILED_ARCHIVE)
    failed = json.loads(bound(evidence/'task/cuda-development-r1/status.json', FAILED_STATUS))
    if (failed['status'] != 'FAILED' or failed['completed'] != [] or failed['pid'] != 46851
            or len(failed['stages']) != 1 or failed['stages'][0]['returncode'] != 1
            or failed['stages'][0]['name'] != 'tofu'
            or (evidence/'task/cuda-development-r1/tofu/rows-rank0.jsonl').read_bytes()):
        raise ValueError('Only the observed failed first stage with no completed results is eligible')
    adapter = json.loads(bound(evidence/'task/adapter/adapter-audit.json', ORIGINAL_ADAPTER))
    helper = Path(__file__).with_name('spf_development_generation.py').read_bytes()
    code = {}
    updated = copy.deepcopy(adapter)
    for name, spec in adapter['sources'].items():
        raw = bound(evidence/'task/adapter/code'/name, spec['adapter_sha256'])
        text = raw.decode('utf8')
        edits = []
        if name in GENERATOR_NAMES:
            old = 'model.generate(use_model_defaults=False, do_sample=False, **'
            new = 'model.generate(**'
            text = replace(text, old, new); edits.append((old, new))
            old = 'def main():'
            new = 'from spf_development_generation import fixed_generation\n\ndef main():'
            text = replace(text, old, new); edits.append((old, new))
            old = '    eos = generation.eos_token_id' if name==GENERATOR_NAMES[0] else '        eos={config.eos_token_id}'
            new = ('    generation = fixed_generation(model, generation)\n'+old if name==GENERATOR_NAMES[0]
                   else '        config=fixed_generation(model,config)\n'+old)
            text = replace(text, old, new); edits.append((old, new))
            restored = text
            for old, new in reversed(edits):
                restored = replace(restored, new, old)
            if restored.encode() != raw:
                raise ValueError('Unexpected data/model/tokenizer/arithmetic edit')
        compile(text, name, 'exec')
        code[name] = text.encode()
        updated['sources'][name].update(previous_adapter_sha256=sha(raw), adapter_sha256=sha(code[name]),
                                       recovery_edits=[dict(before=o, after=n) for o,n in edits],
                                       previous_source_restored_after_inverse_edits=True)
    updated.update(parent_adapter_audit_sha256=ORIGINAL_ADAPTER,
                   recovery='explicit-generation-api-only-after-failed-empty-tofu-stage',
                   failed_run_status_sha256=FAILED_STATUS, failed_evidence_archive_sha256=FAILED_ARCHIVE,
                   generation_helper_sha256=sha(helper), runtime_validation='not_run',
                   model_generation_config_mutated=False)
    audit_raw = (json.dumps(updated, indent=2)+'\n').encode()
    audit_sha = sha(audit_raw)
    guard = bound(evidence/'task/validation-code/run_spf_npo_smoke_queue.py', ORIGINAL_GUARD).decode()
    guard = replace(guard, "ADAPTER_SHA = '"+ORIGINAL_ADAPTER+"'", "ADAPTER_SHA = '"+audit_sha+"'")
    guard_raw = guard.encode()
    controller = bound(evidence/'task/validation-code/run_spf_development_queue.py', ORIGINAL_CONTROLLER).decode()
    controller = replace(controller, "ADAPTER_SHA = '"+ORIGINAL_ADAPTER+"'", "ADAPTER_SHA = '"+audit_sha+"'")
    controller = replace(controller, "GUARD_SHA = '"+ORIGINAL_GUARD+"'", "GUARD_SHA = '"+sha(guard_raw)+"'")
    controller = replace(controller, "'run_spf_npo_smoke_queue.py'", "'run_spf_development_recovery_guards.py'")
    controller = replace(controller, "TASK/'adapter/code'", "TASK/'adapter-generation-api-r2/code'", 3)
    controller = replace(controller, "TASK/'adapter/adapter-audit.json'", "TASK/'adapter-generation-api-r2/adapter-audit.json'", 2)
    controller = replace(controller, "TASK/'cuda-development-r1'", "TASK/'cuda-development-r2'")
    controller = replace(controller, "    sys.argv = commands[stage][1:]", "    sys.path.insert(0, str(source.parent))\n    sys.argv = commands[stage][1:]")
    controller = replace(controller, "    env = environment(cpu=stage=='ifbench-score')", "    env = environment(cpu=stage=='ifbench-score')\n    if stage in ('tofu','ifbench','safety','conversation'):\n        env['SPF_DEVELOPMENT_GENERATION_AUDIT'] = str(run/(stage+'.generation.private.json'))")
    controller = replace(controller, "        files = sorted(p for p in directory.rglob('*') if p.is_file())+[run/(stage+'.cuda.private.json')]", """        files = sorted(p for p in directory.rglob('*') if p.is_file())+[run/(stage+'.cuda.private.json')]
        if stage in ('tofu','ifbench','safety','conversation'):
            generation_path = run/(stage+'.generation.private.json')
            generation = json.loads(generation_path.read_bytes())
            if (generation.get('status') != 'EXPLICIT_GENERATION_CONFIG_RESOLVED_WITHOUT_MODEL_FALLBACK'
                    or generation.get('model_generation_config_unchanged') is not True
                    or generation.get('unused_model_kwargs') != {}
                    or generation['resolved']['max_new_tokens'] != CAPS[stage]
                    or generation['resolved']['do_sample'] is not False
                    or generation['resolved']['num_beams'] != 1
                    or generation['resolved']['use_cache'] is not True):
                raise ValueError('Actual explicit generation configuration receipt required')
            files.append(generation_path)""")
    controller = replace(controller, "    guard = guards(); snapshot = read_bound(a.snapshot, a.snapshot_sha256)", """    guard = guards(); snapshot = read_bound(a.snapshot, a.snapshot_sha256)
    previous = read_bound(TASK/'cuda-development-r1/status.json', '"""+FAILED_STATUS+"""')
    if previous['status'] != 'FAILED' or previous['completed'] != [] or previous['pid'] != 46851:
        raise ValueError('Only unfinished stages from the preserved first-stage failure may recover')
    old_process = guard.command_result(['ps','-p','46851','-o','pid,stat,args'])
    old_rows = old_process['stdout'].strip().splitlines()
    if (old_process['stderr'].strip() or not old_rows or old_rows[0].split()[:2] != ['PID','STAT']
            or not ((old_process['returncode']==1 and len(old_rows)==1)
                    or (old_process['returncode']==0 and len(old_rows)==2
                        and old_rows[1].split()[0]=='46851' and old_rows[1].split()[1].startswith('Z')))):
        raise ValueError('Original failed development queue must have exited')""")
    controller = replace(controller, "    full = read_bound(CONSTRUCTION/'spf-full-s0-audit.json', FULL_SHA)", "    if digest((TASK/'adapter-generation-api-r2/code/spf_development_generation.py').read_bytes()) != adapter['generation_helper_sha256']:\n        raise ValueError('Bound generation API helper changed')\n    full = read_bound(CONSTRUCTION/'spf-full-s0-audit.json', FULL_SHA)")
    controller = replace(controller, "    validate_dependencies(TASK/'repo', read_bound(CONSTRUCTION/'frozen.json', FREEZE_SHA))", "    if digest((source.parent/'spf_development_generation.py').read_bytes()) != adapter['generation_helper_sha256']:\n        raise ValueError('Bound generation API helper changed in child')\n    validate_dependencies(TASK/'repo', read_bound(CONSTRUCTION/'frozen.json', FREEZE_SHA))")
    # The original completion checks, fixed-data plan, CUDA/row validators and all
    # finite stage ordering remain in place. New run/output paths preserve r1.
    for name, text in [('controller',controller),('guard',guard)]:
        compile(text, name, 'exec')
    original_tree = ast.parse(bound(evidence/'task/validation-code/run_spf_development_queue.py', ORIGINAL_CONTROLLER))
    updated_tree = ast.parse(controller)
    unchanged = ('validate_dependencies','validate_cuda','validate_metadata','validate_rows','plan','environment')
    for name in unchanged:
        old = next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new = next(n for n in updated_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        # plan's sole source directory replacement is separately constrained.
        if name=='plan':
            continue
        if ast.dump(old)!=ast.dump(new):
            raise ValueError('Unexpected change to completed static validation function: '+name)
    output.mkdir(parents=True,exist_ok=False)
    dest = output/'adapter-generation-api-r2/code';dest.mkdir(parents=True)
    for name,raw in code.items(): (dest/name).write_bytes(raw)
    (dest/'spf_development_generation.py').write_bytes(helper)
    (output/'adapter-generation-api-r2/adapter-audit.json').write_bytes(audit_raw)
    (output/'run_spf_development_recovery_queue.py').write_bytes(controller.encode())
    (output/'run_spf_development_recovery_guards.py').write_bytes(guard_raw)
    result = dict(status='PREPARED_API_RECOVERY_NOT_DEPLOYED_OR_LAUNCHED',
                  parent_controller_sha256=ORIGINAL_CONTROLLER, parent_guard_sha256=ORIGINAL_GUARD,
                  parent_adapter_audit_sha256=ORIGINAL_ADAPTER, adapter_audit_sha256=audit_sha,
                  controller_sha256=sha(controller.encode()), guard_sha256=sha(guard_raw),
                  helper_sha256=sha(helper), failed_evidence_archive_sha256=FAILED_ARCHIVE,
                  failed_run_status_sha256=FAILED_STATUS, completed_stages=[],
                  restored_adapter_sources_after_inverse_edits=True,
                  preserved_validator_functions=list(unchanged),
                  subprocesses_launched=0, gpu_evaluation_run=False, target_gate_evaluated=False)
    (output/'prepared.private.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
