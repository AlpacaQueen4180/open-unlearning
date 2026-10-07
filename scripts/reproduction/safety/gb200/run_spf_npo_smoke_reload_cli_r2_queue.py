"""One new reload-only journal for the completed PID2953 smoke checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

TASK = Path('/data/spf-npo-smoke-20261007-r1')
RUN_ID = 'reload-native-cli-1936'
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
GUARDS_SHA = 'eedc4dfd16269db9fa84ef681ae289f86392c85c45ff7d0cedfda24bd6d17735'
PACKAGE_SHA = 'cbad72d7386ca83904b5a003e3bcf5b378f44d1920b452dd0bc29369bff5fc8a'
PREVIOUS_SHA = '0c361bd91dc9819f80be3013bf832262bb3bd93149e2c9d7dccaa12a603c9565'
VERIFIER_SHA = '289c718a2f9064b1a7d1542d0ea261c2befa765b5b85a241ec044ed4e1d4fe27'
LAUNCH_FAILURE_SHA = '2f247e8e538c6fd4a7b412317fb599112c76ff126b974070f960f7ec4fbac828'
ACCEPTANCE_SHA = '47fce3014faf1944d86d10f778bdbfd64458a84aa79975ae881a720e58a03af3'
API_SHA = 'bba745ba98304b4edb90037c1d24f776b277c00e42df5ac90e5f07ed3d7a90a7'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exited(pid):
    path = Path('/proc')/str(pid)/'stat'
    if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0] != 'Z':
        raise ValueError('Previous smoke process still alive: '+str(pid))


def completed_train_failed_reload(state, runtime, capture, failed):
    if (state.get('pid') != 2953 or state.get('status') != 'FAILED'
            or state.get('mode') != 'unfinished-train-bound-import-recovery'
            or state.get('completed') != ['train'] or state.get('stage') != 'fresh-reload'):
        raise ValueError('Only the exact completed-train/failed-reload run may recover')
    rows = state.get('stages', [])
    if (len(rows) != 2 or [r.get('name') for r in rows] != ['train','fresh-reload']
            or [r.get('returncode') for r in rows] != [0,1]
            or [r.get('process_pid') for r in rows] != [2961,3501]
            or rows[0].get('raw_log_sha256') != '3f6e7601d131e6abce6236080e07abdd8d72b8dc11afc4a7c2315a952b18141e'
            or rows[1].get('raw_log_sha256') != 'cddad0b89dfae148b4632130c7ac75c180134131182a731b6eee11be4ac9d8ae'):
        raise ValueError('Actual complete train and failed reload commands required')
    binding = state.get('training_args_binding', {})
    if (binding.get('status') != 'ACTUAL_TRAINING_ARGUMENTS_API_BINDING_PASS'
            or binding.get('pid') != 3029 or binding.get('source_sha256') != API_SHA
            or binding.get('remaining_arguments_exact') is not True
            or binding.get('training_arguments_class_unchanged') is not True
            or binding.get('npo_arithmetic_changed') is not False
            or binding.get('precision_changed') is not False):
        raise ValueError('Actual source/PID-bound original API binding required')
    if (runtime.get('trainer_global_step') != 2 or runtime.get('engine_global_steps') != 2
            or runtime.get('microsteps') != 10 or runtime.get('epoch') != 1.0
            or runtime.get('reference_fingerprint_unchanged') is not True
            or capture.get('capture_process_pid') != 3029
            or capture.get('comparison') != dict(atol=0.05,rtol=0.01)
            or failed.get('status') != 'FAILED_RELOAD'
            or failed.get('fresh_process_reload_verified') is not False):
        raise ValueError('Complete actual trace/capture and unsuccessful original reload required')



def failed_native_launcher(state):
    if (state.get('pid') != 3889 or state.get('status') != 'FAILED'
            or state.get('mode') != 'completed-train-native-context-reload-only'
            or state.get('completed') != [] or state.get('stage') != 'fresh-reload'
            or state.get('controller_sha256') != '9ee71c00d28fba2c998c68e94bf59584587be56a68da9abcb097c4ad20299ab3'
            or state.get('verifier_sha256') != '5364672a92d23fdc13b25809ae2364973ab8d81195729880988732b06ca2d2e0'
            or state.get('training_repeated') is not False):
        raise ValueError('Only the preserved failed torchrun launch may recover')
    rows = state.get('stages', [])
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
               (TASK/'validation-code/verify_spf_npo_smoke_native_reload.py').as_posix(),
               '--source-sha256', '5364672a92d23fdc13b25809ae2364973ab8d81195729880988732b06ca2d2e0',
               '--run', (TASK/'runs/reload-native-1312').as_posix()]
    if (len(rows) != 1 or rows[0].get('name') != 'fresh-reload'
            or rows[0].get('process_pid') != 3896 or rows[0].get('returncode') != 2
            or rows[0].get('command') != command
            or rows[0].get('raw_log_sha256') != '20b207faabc703d074421d0fee2891b8edd389406c9c96e187e44557920ff9c1'):
        raise ValueError('Exact failed parser command/source/PID/raw binding required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('controller-sha256','snapshot','snapshot-sha256','run-id'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    if (a.run_id != RUN_ID or sys.executable != PYTHON or TASK.is_symlink()
            or digest(Path(__file__).read_bytes()) != a.controller_sha256):
        raise ValueError('Exact new reload identity and interpreter/source required')
    source = TASK/'validation-code/run_spf_npo_smoke_api_r2_queue.py'
    if digest(source.read_bytes()) != GUARDS_SHA:
        raise ValueError('Original acceptance guards changed')
    spec = importlib.util.spec_from_file_location('smoke_reload_guards',source)
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    package = g.bound_json(TASK/'package.private.json',PACKAGE_SHA)
    for name, record in package['files'].items():
        path = (TASK/name).resolve()
        if not path.is_relative_to(TASK):
            raise ValueError('Package path escaped task')
        raw = path.read_bytes()
        if len(raw) != record['size'] or digest(raw) != record['sha256']:
            raise ValueError('Sealed package changed: '+name)
    c = g.bound_json(TASK/'contract.private.json',package['contract_sha256'])
    previous = g.bound_json(TASK/'runs/import-recovery-1257/status.json',PREVIOUS_SHA)
    failed_launcher = g.bound_json(TASK/'runs/reload-native-1312/status.json',LAUNCH_FAILURE_SHA)
    failed_native_launcher(failed_launcher)
    output = Path(c['output_dir'])
    if output != TASK/'checkpoint' or not output.is_dir() or output.is_symlink():
        raise ValueError('Already completed original checkpoint required')
    runtime = json.loads((output/'spf-npo-runtime.private.json').read_bytes())
    capture = json.loads((output/'spf-npo-reload-capture.json').read_bytes())
    failed = json.loads((output/'spf-npo-fresh-reload-audit.json').read_bytes())
    completed_train_failed_reload(previous,runtime,capture,failed)
    snapshot = g.bound_json(a.snapshot,a.snapshot_sha256)
    g.validate_snapshot(snapshot,datetime.now(timezone.utc))
    if (snapshot['npo_smoke']['import_recovery']['status'] != previous
            or snapshot['npo_smoke']['runtime'] != runtime
            or snapshot['npo_smoke']['fresh_reload_audit'] != failed):
        raise ValueError('New complete snapshot must bind completed checkpoint and failed reload')
    if snapshot['npo_smoke']['native_reload_only']['status'] != failed_launcher:
        raise ValueError('Fresh snapshot must bind the immutable parser failure')
    acceptance = g.bound_json(Path('/data/spf-development-20261007-r1/cuda-development-r2/cuda-acceptance.private.json'),ACCEPTANCE_SHA)
    adapter = g.bound_json(Path('/data/spf-development-20261007-r1/adapter-generation-api-r2/adapter-audit.json'),g.ADAPTER_SHA)
    g.development_accepted(acceptance,adapter)
    verifier = TASK/'validation-code/verify_spf_npo_smoke_native_reload_cli_r2.py'
    if digest(verifier.read_bytes()) != VERIFIER_SHA:
        raise ValueError('New native-context reload source changed')
    commands = json.loads((TASK/'commands.private.json').read_bytes())
    expected = g.exact_commands(c,package['contract_sha256'])
    if any(commands.get(k) != v for k,v in expected.items()):
        raise ValueError('Original sealed command contract changed')
    import fcntl
    with (TASK/'smoke-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run = TASK/'runs'/a.run_id
        if run.exists():
            raise FileExistsError('Never repeat or overwrite a reload journal')
        for pid in (193,202,270,824,830,961,1029,1977,1983,2252,2320,2953,2961,3029,3501,45219,47443,3889,3896):
            exited(pid)
        live = json.loads((g.CONSTRUCTION/'status.json').read_bytes())
        refraw = (g.CONSTRUCTION/'spf-retain95-s0-audit.json').read_bytes()
        g.completed_reference(live,json.loads(refraw))
        if json.loads(refraw) != snapshot['reference_audit']:
            raise ValueError('Completed reference changed')
        g.gpu_idle(g.command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
        gpu = g.command_result(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
        if gpu['returncode'] or gpu['stderr'].strip() or len(gpu['stdout'].strip().splitlines()) != 1:
            raise ValueError('Single original GPU required')
        if shutil.disk_usage(TASK).free < 80*1024**3:
            raise ValueError('Original free workspace guard must pass')
        for path, sha in ((g.CONSTRUCTION/'spf-full-s0-audit.json',g.FULL_AUDIT_SHA),
                          (g.CONSTRUCTION/'frozen.json',g.FREEZE_SHA)):
            if digest(path.read_bytes()) != sha:
                raise ValueError('Completed construction audit/freeze changed')
        run.mkdir(exist_ok=False); path = run/'status.json'
        state = dict(status='RUNNING',mode='completed-train-native-context-reload-only',pid=os.getpid(),
            completed=[],stages=[],started_at=time.time(),controller_sha256=a.controller_sha256,
            verifier_sha256=VERIFIER_SHA,snapshot_sha256=a.snapshot_sha256,
            preserved_original_status_sha256=PREVIOUS_SHA,original_run='import-recovery-1257',
            preserved_native_launcher_failure_sha256=LAUNCH_FAILURE_SHA,command_separator_recovery=True,
            training_repeated=False,completed_diagnostics_repeated=False,pilot_candidates_frozen=False,
            contract_sha256=package['contract_sha256'],target_gate_evaluated=False)
        g.atomic(path,state)
        overrides = {**expected['environment'], 'HF_HOME':'/data/smart-mfg/jimmy-lin/hf_cache/hub',
            'HF_HUB_CACHE':'/data/smart-mfg/jimmy-lin/hf_cache/hub','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1',
            'PYTHONPATH':c['repository']+'/src'}
        env = dict(values={**os.environ,**overrides},overrides=overrides)
        command = [PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',str(verifier),
                   '--source-sha256',VERIFIER_SHA,'--run',str(run)]
        try:
            g.stage(state,path,'fresh-reload',command,c['repository'],env,require_idle=True)
            result = json.loads((run/'reload-audit.private.json').read_bytes())
            if (result.get('status') != 'NATIVE_CONTEXT_SMOKE_FRESH_RELOAD_PASS'
                    or result.get('fresh_process_reload_verified') is not True
                    or result.get('training_steps_executed') != 0 or result.get('optimizer_created') is not False
                    or result.get('comparison') != dict(atol=0.05,rtol=0.01)
                    or result.get('manual_rotary_buffer_mutation') is not False
                    or result.get('numerical_settings_changed') is not False):
                raise ValueError('Actual unchanged-threshold independent native reload required')
            state.update(status='SMOKE_NATIVE_RELOAD_COMPLETE_PENDING_PILOT_REVIEW',finished_at=time.time(),reload=result)
            g.atomic(path,state)
        except Exception as error:
            state.update(status='FAILED',error=repr(error),finished_at=time.time(),
                recovery='Preserve new reload journal; never train this completed checkpoint again')
            g.atomic(path,state); raise
    print(json.dumps(dict(status=state['status'],training_repeated=False,pilot_frozen=False)))


if __name__ == '__main__':
    main()
