"""Explicit unfinished-stage recovery of the preserved TrainingArguments API failure.

Reuse the successful Hydra composition and immutable twelve-file package. The
native observer, NPO arithmetic, config and reload thresholds remain bound to
their original bytes. This controller must run with a fresh complete snapshot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

TASK = Path('/data/spf-npo-smoke-20261007-r1')
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
ORIGINAL_CONTROLLER_SHA = 'eedc4dfd16269db9fa84ef681ae289f86392c85c45ff7d0cedfda24bd6d17735'
PACKAGE_SHA = 'cbad72d7386ca83904b5a003e3bcf5b378f44d1920b452dd0bc29369bff5fc8a'
FAILED_STATUS_SHA = '2b4f88241483b7e2c269052ba643d99ec0420bfb898c76f0fcaa27313a982d30'
FAILED_LOG_SHA = '526d37715c3048cfae36b629d162e95e8e0e7a25e0e94b465b5a30dffd07f43f'
HYDRA_SHA = '0327ee8e27ad04f1b589765b04c4cff3e6b93b706c9b01369144da8155b99036'
DIAGNOSTIC_SHA = 'c840357a27581a11aab3d43af2f00086c213bb460d53dbfc07edee34f250c841'
ORIGINAL_API_ENTRYPOINT_SHA = '430feb8b97a3fb0dea5a9f293b50394300e0a947d5a0e2c0875f5c716e2fae52'
API_ENTRYPOINT_SHA = 'bba745ba98304b4edb90037c1d24f776b277c00e42df5ac90e5f07ed3d7a90a7'
API_DIAGNOSTIC_RAW_SHA = '140365dfab20a5899e8b73ae223162a6c46a08e2b3bf2af91fd5dcb616ac91df'
CACHE_DIAGNOSTIC_RAW_SHA = '360de69cf844afa0807c1b74e904b6533890a0996c386dedf64549ec34097096'
CACHE = '/data/smart-mfg/jimmy-lin/hf_cache/hub'
CACHE_ENVIRONMENT = dict(HF_HOME=CACHE, HF_HUB_CACHE=CACHE,
                         HF_HUB_DISABLE_IMPLICIT_TOKEN='1')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exited(pid):
    stat = Path('/proc')/str(pid)/'stat'
    if stat.exists() and stat.read_text().rsplit(')', 1)[1].strip().split()[0] != 'Z':
        raise ValueError('Previous failed smoke process is still alive: '+str(pid))


def failed_before_training(state, checkpoint_exists, hydra):
    if (state.get('status') != 'FAILED' or state.get('pid') != 1977
            or state.get('mode') != 'unfinished-train-training-args-api-recovery'
            or state.get('completed') != ['training-args-diagnostic']
            or state.get('stage') != 'train' or checkpoint_exists
            or hydra.get('status') != 'HYDRA_COMPOSITION_PASS'):
        raise ValueError('Only the preserved pre-Trainer API failure without a checkpoint may recover')
    stages = state.get('stages', [])
    if (len(stages) != 2 or [r.get('name') for r in stages] != ['training-args-diagnostic', 'train']
            or [r.get('returncode') for r in stages] != [0, 1]
            or stages[1].get('process_pid') != 2252
            or stages[1].get('raw_log_sha256') != FAILED_LOG_SHA
            or stages[0].get('raw_log_sha256') != API_DIAGNOSTIC_RAW_SHA):
        raise ValueError('Exact completed cache diagnostic and failed native command required')
    verify_cache_diagnostic(state.get('cache_diagnostic', {}))
    verify_training_args_diagnostic(state.get('training_args_diagnostic', {}))


def recovered_environment(original):
    if original.get('HF_HOME') != '/data/smart-mfg/jimmy-lin/hf_cache':
        raise ValueError('Original sealed cache environment changed')
    if any(original.get(k) != '1' for k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE')):
        raise ValueError('Original offline flags must remain enabled')
    return {**original, **CACHE_ENVIRONMENT}


def verify_cache_diagnostic(receipt):
    expected = dict(status='NATIVE_OFFLINE_TOKENIZER_CACHE_MAPPING_PASS', cache_dir=CACHE,
                    hf_hub_cache=CACHE, model_id='meta-llama/Llama-3.1-8B-Instruct',
                    revision='0e9e39f249a16976918f6564b8830bc894c89659',
                    native_source_sha256='894e36a8471b3346ef672db896f51f40000270167ed70b955cec80351534a252',
                    old_cache_path_rejected=True, exact_vocab_and_template=True,
                    cuda_initialized=False, weights_read=False, generation_calls=0,
                    npo_training_executed=False)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual CPU native tokenizer diagnostic must pass before training')
    if receipt.get('vocab_size') != 128256 or receipt.get('eos_token_id') != 128009:
        raise ValueError('Fixed Meta tokenizer vocabulary/EOS required')
    if receipt.get('pad_token_id') != receipt['eos_token_id']:
        raise ValueError('Original native EOS-as-PAD behavior required')


def verify_training_args_diagnostic(receipt):
    expected = dict(status='TRAINING_ARGUMENTS_SIGNATURE_AND_SERIALIZATION_SOURCE_PASS',
        compatibility_source_sha256=ORIGINAL_API_ENTRYPOINT_SHA, omitted_keywords=['save_safetensors'],
        omitted_value=True, remaining_arguments_exact=True, signature_bound=True,
        training_arguments_instantiated=False, cuda_initialized=False, weights_read=False,
        generation_calls=0, npo_training_executed=False, subprocesses_launched=0,
        config_sha256='2c753b5f656820ec7dffddfc57170845daf02a078462e18d98c793e3d9a1288b',
        negative_cases=['false-serialization','unknown-keyword','changed-learning-rate',
                        'missing-required-serialization-intent'],
        installed_api_sha256={'training_args.py':'67e63d9b8a68a1547c0f3d17eac51034592da669d01cf03b4471b0b77f22a756',
            'trainer.py':'e680536f179b7e2000188a671c0317c7fdce72cd4ceaf91dd72c2032f8c828d0',
            'modeling_utils.py':'b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16'})
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError('Complete actual CPU signature/serialization proof required before training')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('controller-sha256', 'snapshot', 'snapshot-sha256', 'run-id'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    if (sys.executable != PYTHON or TASK.is_symlink()
            or a.run_id != 'import-recovery-1257'
            or digest(Path(__file__).read_bytes()) != a.controller_sha256):
        raise ValueError('Exact interpreter/source and new explicit recovery identity required')
    old_source = TASK/'validation-code/run_spf_npo_smoke_api_r2_queue.py'
    if digest(old_source.read_bytes()) != ORIGINAL_CONTROLLER_SHA:
        raise ValueError('Original smoke controller changed')
    spec = importlib.util.spec_from_file_location('original_smoke_guards', old_source)
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    package = g.bound_json(TASK/'package.private.json', PACKAGE_SHA)
    for name, record in package['files'].items():
        f = (TASK/name).resolve()
        if not f.is_relative_to(TASK):
            raise ValueError('Package path escaped original task')
        b = f.read_bytes()
        if len(b) != record['size'] or digest(b) != record['sha256']:
            raise ValueError('Original package member changed: '+name)
    c = g.bound_json(TASK/'contract.private.json', package['contract_sha256'])
    if c['task_root'] != str(TASK):
        raise ValueError('Original contract task identity changed')
    sys.path.insert(0, str(TASK/'code'))
    from prepare_spf_npo_smoke import validate_contract_config
    validate_contract_config(c, json.loads((TASK/'configs/spf_npo_smoke.yaml').read_bytes()),
                             json.loads((TASK/'deepspeed.json').read_bytes()))
    commands = json.loads((TASK/'commands.private.json').read_bytes())
    expected = g.exact_commands(c, package['contract_sha256'])
    if any(commands.get(k) != v for k, v in expected.items()):
        raise ValueError('Original sealed commands changed')
    previous = g.bound_json(TASK/'runs/training-args-recovery-1225/status.json', FAILED_STATUS_SHA)
    hydra = g.bound_json(TASK/'runs/initial/hydra-compose.private.json', HYDRA_SHA)
    output = Path(c['output_dir'])
    failed_before_training(previous, output.exists(), hydra)
    if digest((TASK/'runs/training-args-recovery-1225/train.raw.log').read_bytes()) != FAILED_LOG_SHA:
        raise ValueError('Preserved original loader failure changed')
    for pid in (193, 202, 270, 824, 830, 961, 1029, 1977, 1983, 2252, 2320):
        exited(pid)
    snapshot = g.bound_json(a.snapshot, a.snapshot_sha256)
    g.validate_snapshot(snapshot, datetime.now(timezone.utc))
    if snapshot['npo_smoke']['training_args_recovery']['status'] != previous:
        raise ValueError('Fresh complete snapshot must contain the exact failed original run')
    dev = Path('/data/spf-development-20261007-r1')
    acceptance = g.bound_json(dev/'cuda-development-r2/cuda-acceptance.private.json',
                              g.DEVELOPMENT_ACCEPTANCE_SHA)
    adapter = g.bound_json(dev/'adapter-generation-api-r2/adapter-audit.json', g.ADAPTER_SHA)
    g.development_accepted(acceptance, adapter)
    # Completed development outputs were already exported and validated. Bind
    # the immutable actual acceptance here without polling/rerunning those stages.
    diagnostic = TASK/'validation-code/verify_spf_npo_training_args_api.py'
    if digest(diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError('Bound CPU TrainingArguments diagnostic source changed')
    entrypoint = TASK/'validation-code/spf_npo_training_args_api_r2.py'
    if digest(entrypoint.read_bytes()) != API_ENTRYPOINT_SHA:
        raise ValueError('Explicit new TrainingArguments compatibility source changed')
    cache_raw = (TASK/'runs/cache-recovery-1155/cache-diagnostic.raw.log').read_bytes()
    if digest(cache_raw) != CACHE_DIAGNOSTIC_RAW_SHA:
        raise ValueError('Completed tokenizer cache proof changed')
    if json.loads(cache_raw.decode().splitlines()[-1]) != previous['cache_diagnostic']:
        raise ValueError('Actual preserved tokenizer receipt differs from completed stage')
    for name, sha in c['native_source_sha256'].items():
        if digest((Path(c['repository'])/name).read_bytes()) != sha:
            raise ValueError('Native source changed: '+name)
    import fcntl
    with (TASK/'smoke-queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = TASK/'runs'/a.run_id
        if run.exists() or output.exists():
            raise FileExistsError('Recovery run or checkpoint exists; never restart it')
        for pid in (193, 202, 270, 824, 830, 961, 1029, 1977, 1983, 2252, 2320):
            exited(pid)
        live = json.loads((g.CONSTRUCTION/'status.json').read_bytes())
        refraw = (g.CONSTRUCTION/'spf-retain95-s0-audit.json').read_bytes()
        ref = json.loads(refraw); g.completed_reference(live, ref)
        if ref != snapshot['reference_audit']:
            raise ValueError('Actual completed reference differs from bound snapshot')
        g.process_exited(g.command_result(['ps', '-p', '45219', '-o', 'pid,stat,args']))
        g.gpu_idle(g.command_result(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader']))
        gpu = g.command_result(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
        if gpu['returncode'] or gpu['stderr'].strip() or len(gpu['stdout'].strip().splitlines()) != 1:
            raise ValueError('Single original GPU required')
        if shutil.disk_usage(TASK).free < 80*1024**3:
            raise ValueError('Original free export workspace requirement must pass')
        for path, sha in ((g.CONSTRUCTION/'spf-full-s0-audit.json', g.FULL_AUDIT_SHA),
                          (g.CONSTRUCTION/'frozen.json', g.FREEZE_SHA)):
            if digest(path.read_bytes()) != sha:
                raise ValueError('Bound construction audit/freeze changed')
        run.mkdir(parents=True, exist_ok=False); path = run/'status.json'
        state = dict(status='RUNNING', pid=os.getpid(), mode='unfinished-train-bound-import-recovery',
                     completed=[], stages=[], started_at=time.time(),
                     controller_sha256=a.controller_sha256, package_sha256=PACKAGE_SHA,
                     contract_sha256=package['contract_sha256'], snapshot_sha256=a.snapshot_sha256,
                     reference_audit_sha256=digest(refraw),
                     development_acceptance_sha256=g.DEVELOPMENT_ACCEPTANCE_SHA,
                     reused_hydra_composition_sha256=HYDRA_SHA,
                     preserved_failed_status_sha256=FAILED_STATUS_SHA,
                     environment_changes=CACHE_ENVIRONMENT, original_run='training-args-recovery-1225',
                     reused_cache_diagnostic_raw_sha256=CACHE_DIAGNOSTIC_RAW_SHA,
                     cache_diagnostic=previous['cache_diagnostic'],training_args_diagnostic=previous['training_args_diagnostic'],reused_api_diagnostic_raw_sha256=API_DIAGNOSTIC_RAW_SHA,
                     compatibility_entrypoint_sha256=API_ENTRYPOINT_SHA,
                     training_resume_supported=False, pilot_candidates_frozen=False,
                     target_gate_evaluated=False)
        g.atomic(path, state)
        overrides = {**recovered_environment(expected['environment']),
                     'PYTHONPATH': c['repository']+'/src',
                     'SPF_NPO_TRAINING_ARGS_API_SHA256': API_ENTRYPOINT_SHA,
                     'SPF_NPO_TRAINING_ARGS_API_RECEIPT': str(run/'training-args-binding.private.json')}
        env = dict(values={**os.environ, **overrides}, overrides=overrides)
        cpu_env = dict(values={**env['values'], 'CUDA_VISIBLE_DEVICES': ''},
                       overrides={**overrides, 'CUDA_VISIBLE_DEVICES': ''})
        try:
            receipt=previous['training_args_diagnostic']
            api_raw=(TASK/'runs/training-args-recovery-1225/training-args-diagnostic.raw.log').read_bytes()
            if digest(api_raw)!=API_DIAGNOSTIC_RAW_SHA or json.loads(api_raw.decode().splitlines()[-1])!=receipt:
                raise ValueError('Completed actual API diagnostic changed')
            train = list(expected['train'])
            if train[5] != str(TASK/'code/spf_npo_native_probe.py'):
                raise ValueError('Original sealed distributed entrypoint changed')
            train[5] = str(entrypoint)
            g.stage(state, path, 'train', train, c['repository'], env, require_idle=True)
            binding = json.loads((run/'training-args-binding.private.json').read_bytes())
            if (binding.get('status') != 'ACTUAL_TRAINING_ARGUMENTS_API_BINDING_PASS'
                    or binding.get('source_sha256') != API_ENTRYPOINT_SHA
                    or binding.get('installed_api_sha256') != receipt['installed_api_sha256']
                    or binding.get('omitted_keywords') != ['save_safetensors']
                    or binding.get('remaining_arguments_exact') is not True
                    or binding.get('training_arguments_class_unchanged') is not True
                    or binding.get('npo_arithmetic_changed') is not False
                    or binding.get('precision_changed') is not False):
                raise ValueError('Actual API translation receipt required')
            state['training_args_binding'] = binding; g.atomic(path, state)
            g.stage(state, path, 'fresh-reload', expected['fresh_reload'],
                    c['repository'], env, require_idle=True)
            reload = json.loads((output/'spf-npo-fresh-reload-audit.json').read_bytes())
            if (reload.get('status') != 'SMOKE_TRAINING_EXPORT_FRESH_RELOAD_PASS'
                    or reload.get('fresh_process_reload_verified') is not True
                    or reload.get('contract_sha256') != package['contract_sha256']):
                raise ValueError('Actual complete smoke and fresh process reload required')
            state.update(status='SMOKE_COMPLETE_PENDING_PILOT_FREEZE', finished_at=time.time())
        except BaseException as error:
            state.update(status='FAILED', error=repr(error), finished_at=time.time(),
                         recovery='Preserve original failure and new journal; only unfinished stages may recover')
            g.atomic(path, state); raise
        g.atomic(path, state)


if __name__ == '__main__':
    main()
