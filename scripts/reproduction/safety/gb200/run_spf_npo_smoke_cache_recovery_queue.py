"""Explicit unfinished-stage recovery of the preserved tokenizer-cache failure.

Reuse the successful Hydra composition and immutable twelve-file package. The
native entrypoint, NPO arithmetic, config and reload thresholds remain bound to
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
FAILED_STATUS_SHA = 'e473acc9a792eaaab8655093a0d774d001b131a2be68b6d7a69aced736835349'
FAILED_LOG_SHA = 'fc26363a91a35358a432081d4cf6ec187a45bb140fa271d204a30ea03d0a4497'
HYDRA_SHA = '0327ee8e27ad04f1b589765b04c4cff3e6b93b706c9b01369144da8155b99036'
DIAGNOSTIC_SHA = '3b412c8480280a64a42f889eaf081ae4692038e3914cb003b1b0eb3ee383e1a2'
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
    if (state.get('status') != 'FAILED' or state.get('pid') != 193
            or state.get('mode') != 'initial' or state.get('completed') != ['hydra-compose']
            or state.get('stage') != 'train' or checkpoint_exists
            or hydra.get('status') != 'HYDRA_COMPOSITION_PASS'):
        raise ValueError('Only the preserved failed loader with no checkpoint may recover')
    stages = state.get('stages', [])
    if (len(stages) != 2 or [r.get('name') for r in stages] != ['hydra-compose', 'train']
            or [r.get('returncode') for r in stages] != [0, 1]
            or stages[1].get('process_pid') != 202
            or stages[1].get('raw_log_sha256') != FAILED_LOG_SHA):
        raise ValueError('Original successful composition and failed native command required')


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('controller-sha256', 'snapshot', 'snapshot-sha256', 'run-id'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    if (sys.executable != PYTHON or TASK.is_symlink()
            or not re.fullmatch('cache-recovery-[a-z0-9-]+', a.run_id)
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
    previous = g.bound_json(TASK/'runs/initial/status.json', FAILED_STATUS_SHA)
    hydra = g.bound_json(TASK/'runs/initial/hydra-compose.private.json', HYDRA_SHA)
    output = Path(c['output_dir'])
    failed_before_training(previous, output.exists(), hydra)
    if digest((TASK/'runs/initial/train.raw.log').read_bytes()) != FAILED_LOG_SHA:
        raise ValueError('Preserved original loader failure changed')
    for pid in (193, 202, 270):
        exited(pid)
    snapshot = g.bound_json(a.snapshot, a.snapshot_sha256)
    g.validate_snapshot(snapshot, datetime.now(timezone.utc))
    if snapshot['npo_smoke']['status'] != previous:
        raise ValueError('Fresh complete snapshot must contain the exact failed original run')
    dev = Path('/data/spf-development-20261007-r1')
    acceptance = g.bound_json(dev/'cuda-development-r2/cuda-acceptance.private.json',
                              g.DEVELOPMENT_ACCEPTANCE_SHA)
    adapter = g.bound_json(dev/'adapter-generation-api-r2/adapter-audit.json', g.ADAPTER_SHA)
    g.development_accepted(acceptance, adapter)
    # Completed development outputs were already exported and validated. Bind
    # the immutable actual acceptance here without polling/rerunning those stages.
    diagnostic = TASK/'validation-code/verify_spf_npo_tokenizer_cache.py'
    if digest(diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError('Bound CPU tokenizer diagnostic source changed')
    for name, sha in c['native_source_sha256'].items():
        if digest((Path(c['repository'])/name).read_bytes()) != sha:
            raise ValueError('Native source changed: '+name)
    import fcntl
    with (TASK/'smoke-queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = TASK/'runs'/a.run_id
        if run.exists() or output.exists():
            raise FileExistsError('Recovery run or checkpoint exists; never restart it')
        for pid in (193, 202, 270):
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
        state = dict(status='RUNNING', pid=os.getpid(), mode='unfinished-train-cache-recovery',
                     completed=[], stages=[], started_at=time.time(),
                     controller_sha256=a.controller_sha256, package_sha256=PACKAGE_SHA,
                     contract_sha256=package['contract_sha256'], snapshot_sha256=a.snapshot_sha256,
                     reference_audit_sha256=digest(refraw),
                     development_acceptance_sha256=g.DEVELOPMENT_ACCEPTANCE_SHA,
                     reused_hydra_composition_sha256=HYDRA_SHA,
                     preserved_failed_status_sha256=FAILED_STATUS_SHA,
                     environment_changes=CACHE_ENVIRONMENT, original_run='initial',
                     training_resume_supported=False, pilot_candidates_frozen=False,
                     target_gate_evaluated=False)
        g.atomic(path, state)
        overrides = {**recovered_environment(expected['environment']),
                     'PYTHONPATH': c['repository']+'/src'}
        env = dict(values={**os.environ, **overrides}, overrides=overrides)
        cpu_env = dict(values={**env['values'], 'CUDA_VISIBLE_DEVICES': ''},
                       overrides={**overrides, 'CUDA_VISIBLE_DEVICES': ''})
        try:
            g.stage(state, path, 'cache-diagnostic', [PYTHON, str(diagnostic)],
                    c['repository'], cpu_env)
            receipt = json.loads((run/'cache-diagnostic.raw.log').read_text().splitlines()[-1])
            verify_cache_diagnostic(receipt)
            state['cache_diagnostic'] = receipt; g.atomic(path, state)
            g.stage(state, path, 'train', expected['train'], c['repository'], env, require_idle=True)
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
