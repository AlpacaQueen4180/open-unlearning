"""Finite fixed-data SPF development queue; preparation is not CUDA acceptance.

Run only after the original retain95 audit and an idle-GPU snapshot. Execute the
unchanged, audited adapters in separate processes. No training or paid judges.
Private generations and row metrics remain in the isolated development task.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

TASK = Path(os.environ['SPF_DIAG_CASE_TASK'])
CONSTRUCTION = Path('/data/spf-npo-20261006-r4')
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
CPU_PYTHON = '/data/spf-development-20261007-r1/ifbench-cpu-venv/bin/python'
GUARD_SHA = '64316b56536b77ee1aa11994c5c3aaf42aa01c61133178c4fcc35e5efe06a41f'
ADAPTER_SHA = os.environ['SPF_DIAG_ADAPTER_SHA']
FULL_SHA = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
FREEZE_SHA = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
IDENTITY = os.environ['SPF_DIAG_IDENTITY']
META_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
ROOT = 'work/spf-p0p1-20260916/artifacts/'
DATA = {
    'knowledge': (ROOT+'development-candidates-v1/utility-development.json',
                  '8f6c4551cf8686515b80cb530c141928a0a39e89029179e722eff544cbbd4720', 1024),
    'ifbench': (ROOT+'development-extension-v1/pinned-candidates-v2/ifbench.json',
                '119782fe7a3d78f9276765b9c047dd722d2d61669c9d8076dbf5bf0f7dbd757c', 300),
    'safety': (ROOT+'development-extension-v1/pinned-candidates-v2/safety_development.json',
               '2b5696ae06a816b9497559b9702fcd9a0f14eac11f13a37d2621e6470fffcb1d', 650),
    'conversation': (ROOT+'development-extension-v1/wildchat-candidates-v1/context-eligible-v2.json',
                     'e392bbb41d5c98f9b25e204e379253a341fd1bd6813c1118da269b5edcdb621e', 71),
}
STAGES = ('tofu', 'knowledge', 'ifbench', 'safety', 'conversation', 'ifbench-score')
NAMES = {'tofu': 'evaluate_construction_baseline.py',
         'knowledge': 'evaluate_development_knowledge.py',
         'ifbench': 'generate_development.py', 'safety': 'generate_development.py',
         'conversation': 'generate_development_conversation.py',
         'ifbench-score': 'score_development_ifbench.py'}
CAPS = dict(tofu=200, safety=512, ifbench=2048, conversation=1024)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(path, sha):
    raw = Path(path).read_bytes()
    if digest(raw) != sha:
        raise ValueError('Bound source or metadata changed: '+str(path))
    return json.loads(raw)


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    tmp.replace(path)


def validate_dependencies(repository, manifest):
    template = 'configs/model/Llama-3.1-8B-Instruct.yaml'
    expected = {name: sha for name, sha in manifest['construction_code'].items()
                if name.startswith('src/construction/') or name == template}
    required = {'src/construction/'+name+'.py' for name in
                ('__init__','baseline','data','learning_metrics','protocol','knowledge')}
    if not required.issubset(expected) or template not in expected:
        raise ValueError('Frozen development dependency/template identities required')
    for name, sha in expected.items():
        path = (repository/name).resolve()
        if not path.is_relative_to(repository.resolve()) or digest(path.read_bytes()) != sha:
            raise ValueError('Frozen development dependency changed: '+name)


def guards():
    path = Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py')
    if digest(path.read_bytes()) != GUARD_SHA:
        raise ValueError('Original completion/snapshot guard source changed')
    spec = importlib.util.spec_from_file_location('spf_completion_guards', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(cpu=False):
    values = dict(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache', WANDB_MODE='disabled',
                  TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(TASK/'repo/src'),
                  WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
    if cpu:
        values['CUDA_VISIBLE_DEVICES'] = ''
    return values


def plan(run):
    """Exact commands use complete frozen data and the original generation caps."""
    manifest = str(CONSTRUCTION/'frozen.json')
    code = TASK/'adapter-generation-api-r2/code'
    commands = {'tofu': [PYTHON, str(code/NAMES['tofu']), '--manifest', manifest,
                          '--output', str(run/'tofu'), '--batch-size', '8',
                          '--max-new-tokens', '200']}
    for stage in DATA:
        name, sha, _ = DATA[stage]
        command = [PYTHON, str(code/NAMES[stage]), '--dataset', str(TASK/'bundle/assets'/name),
                   '--dataset-sha256', sha, '--manifest', manifest, '--output', str(run/stage)]
        if stage != 'knowledge':
            command += ['--max-new-tokens', str(CAPS[stage])]
        commands[stage] = command
    runtime = TASK/'bundle/assets'/ROOT/'development-extension-v1/ifbench-runtime'
    commands['ifbench-score'] = [CPU_PYTHON, str(code/NAMES['ifbench-score']),
        '--runtime', str(runtime), '--dataset', str(TASK/'bundle/assets'/DATA['ifbench'][0]),
        '--responses', str(run/'ifbench/responses-rank0.jsonl'), '--output', str(run/'ifbench-scores.private.json')]
    return {name: commands[name] for name in STAGES}


def validate_cuda(record, pid, source_sha):
    if (record.get('status') != 'CUDA_EXECUTOR_RETURNED'
            or record.get('pid') != pid or record.get('source_sha256') != source_sha
            or record.get('cuda_initialized') is not True or record.get('gpu_count') != 1
            or record.get('peak_allocated_bytes', 0) <= 0
            or record.get('checkpoint_identity_sha256') != IDENTITY):
        raise ValueError('Actual child CUDA/source/PID evidence required')


def validate_metadata(stage, metadata):
    if stage == 'ifbench-score':
        if metadata.get('status') != 'pass' or metadata.get('count') != 300 or metadata.get('dataset_sha256') != DATA['ifbench'][1]:
            raise ValueError('Complete new SPF IFBench scoring required')
        return
    if (metadata.get('status') != 'pass' or metadata.get('checkpoint_identity_sha256') != IDENTITY
            or metadata.get('tokenizer_revision') != META_REVISION or metadata.get('world_size') != 1):
        raise ValueError('Actual SPF development identity/world/tokenizer result required')
    if stage == 'tofu':
        if (metadata.get('teacher_forcing_max_length') != 512 or metadata.get('batch_size_per_rank') != 8
                or {s: v['count'] for s,v in metadata['aggregates'].items()} != dict(forget05=200, retain95=3800)
                or metadata.get('manifest_sha256') != FREEZE_SHA):
            raise ValueError('Complete 4000-row TOFU result with original teacher/batch required')
    else:
        if metadata.get('count') != DATA[stage][2] or metadata.get('dataset_sha256') != DATA[stage][1]:
            raise ValueError('Complete fixed development dataset result required')
        if stage == 'knowledge':
            if (metadata.get('protocol') != 'auxiliary-zero-shot-ABCD-summed-continuation-logprob-v1'
                    or metadata.get('num_fewshot') != 0 or metadata.get('chat_template') is not False):
                raise ValueError('Original auxiliary zero-shot knowledge protocol required')
            return
        if stage == 'conversation' and metadata.get('turns_per_conversation') != 2:
            raise ValueError('Both fixed conversation turns required')
    generation = metadata['generation']
    if (generation.get('max_new_tokens') != CAPS[stage] or generation.get('do_sample') is not False
            or generation.get('use_cache') is not True):
        raise ValueError('Original greedy generation caps required')


def validate_rows(stage, rows, source_rows=None):
    """Check row identities/counts without assigning safety labels or a gate."""
    if stage == 'tofu':
        if (len(rows) != 4000 or {r['id'] for r in rows} != {str(i) for i in range(4000)}
                or sum(r['split']=='forget05' for r in rows) != 200
                or sum(r['split']=='retain95' for r in rows) != 3800
                or any(not r.get('row_sha256') or any(not math.isfinite(v) for v in r['metrics'].values()) for r in rows)):
            raise ValueError('TOFU rows incomplete or nonfinite')
    elif stage == 'knowledge':
        if (len(rows) != 1024 or {r['id'] for r in rows} != {r['id'] for r in source_rows}
                or any(len(r['scores']) != 4 or not all(math.isfinite(v) for v in r['scores']) for r in rows)):
            raise ValueError('Knowledge rows incomplete or nonfinite')
    else:
        expected = {}
        for row in source_rows:
            if stage == 'conversation':
                for turn, prompt in enumerate(row['turns'], 1):
                    expected[str(row['id'])+':'+str(turn)] = prompt
            else:
                expected[str(row['id'])] = row['prompt']
        if (len(rows) != len(expected) or {r['id'] for r in rows} != set(expected)
                or any(r['prompt'] != expected[r['id']] or not isinstance(r['response'], str)
                       or not 0 <= r['generated_tokens'] <= CAPS[stage] for r in rows)):
            raise ValueError('Fixed prompt/ID/generation rows incomplete')


def actual_result(run, stage, row):
    code = TASK/'adapter-generation-api-r2/code'/NAMES[stage]
    source_sha = digest(code.read_bytes())
    if stage == 'ifbench-score':
        output = run/'ifbench-scores.private.json'
        validate_metadata(stage, json.loads(output.read_bytes()))
        files = [output]
    else:
        directory = run/stage
        metadata = directory/('public-summary.json' if stage in ('tofu','knowledge') else 'metadata-rank0.json')
        validate_metadata(stage, json.loads(metadata.read_bytes()))
        proof = read_bound(run/(stage+'.cuda.private.json'), row['cuda_receipt_sha256'])
        validate_cuda(proof, row['process_pid'], source_sha)
        rows_path = directory/('rows-rank0.jsonl' if stage=='tofu' else 'rows.jsonl' if stage=='knowledge' else 'responses-rank0.jsonl')
        rows = [json.loads(line) for line in rows_path.read_text(encoding='utf8').splitlines() if line.strip()]
        inputs = read_bound(TASK/'bundle/assets'/DATA[stage][0], DATA[stage][1]) if stage in DATA else None
        validate_rows(stage, rows, inputs)
        files = sorted(p for p in directory.rglob('*') if p.is_file())+[run/(stage+'.cuda.private.json')]
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
            files.append(generation_path)
    return {str(p.relative_to(run)): dict(size=p.stat().st_size, sha256=digest(p.read_bytes())) for p in files}


def child(stage, run):
    state = json.loads((run/'status.json').read_bytes())
    commands = plan(run)
    if state['commands'] != commands or state['controller_sha256'] != digest(Path(__file__).read_bytes()):
        raise ValueError('Exact child plan/controller required')
    adapter = read_bound(TASK/'adapter-generation-api-r2/adapter-audit.json', ADAPTER_SHA)
    source = Path(commands[stage][1]); sha = adapter['sources'][source.name]['adapter_sha256']
    if digest(source.read_bytes()) != sha:
        raise ValueError('Unchanged adapter source required')
    if digest((source.parent/'spf_development_generation.py').read_bytes()) != adapter['generation_helper_sha256']:
        raise ValueError('Bound generation API helper changed in child')
    validate_dependencies(TASK/'repo', read_bound(CONSTRUCTION/'frozen.json', FREEZE_SHA))
    # Import real Torch only in the separately launched GPU child. Observations do
    # not alter precision, model loading, inputs, generation, losses or buffers.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One actual CUDA device required')
    sys.path.insert(0, str(source.parent))
    sys.argv = commands[stage][1:]
    started = time.time()
    runpy.run_path(str(source), run_name='__main__')
    proof = dict(status='CUDA_EXECUTOR_RETURNED', pid=os.getpid(), source_sha256=sha,
                 checkpoint_identity_sha256=IDENTITY, started_at=started, finished_at=time.time(),
                 cuda_initialized=torch.cuda.is_initialized(), gpu_count=torch.cuda.device_count(),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    validate_cuda(proof, os.getpid(), sha)
    path = run/(stage+'.cuda.private.json')
    with path.open('x', encoding='utf8') as handle:
        handle.write(json.dumps(proof, indent=2)+'\n')


def execute(state, run, stage, guard):
    if stage != 'ifbench-score':
        guard.gpu_idle(guard.command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
    direct = state['commands'][stage]
    command = direct if stage=='ifbench-score' else [PYTHON, str(Path(__file__).resolve()), '--child-stage', stage, '--child-run', str(run)]
    env = environment(cpu=stage=='ifbench-score')
    if stage in ('tofu','ifbench','safety','conversation'):
        env['SPF_DEVELOPMENT_GENERATION_AUDIT'] = str(run/(stage+'.generation.private.json'))
    row = dict(name=stage, command=command, executor_command=direct, cwd=str(TASK/'repo'),
               environment_overrides=env, source_sha256=digest(Path(direct[1]).read_bytes()),
               log=str(run/(stage+'.raw.log')), started_at=time.time())
    state.update(stage=stage); state['stages'].append(row); write(run/'status.json', state)
    with Path(row['log']).open('xb') as stream:
        process = subprocess.Popen(command, cwd=row['cwd'], env={**os.environ, **env}, stdout=stream, stderr=subprocess.STDOUT)
        row['process_pid'] = process.pid; write(run/'status.json', state)
        row['returncode'] = process.wait()
    row.update(finished_at=time.time(), raw_log_sha256=digest(Path(row['log']).read_bytes()))
    write(run/'status.json', state)
    if row['returncode']:
        raise RuntimeError('Executor failed; preserve checkpoint and raw evidence: '+stage)
    if stage != 'ifbench-score':
        row['cuda_receipt_sha256'] = digest((run/(stage+'.cuda.private.json')).read_bytes())
    row['outputs'] = actual_result(run, stage, row)
    row['gpu_runtime_validated'] = stage != 'ifbench-score'
    row['examples'] = 4000 if stage=='tofu' else DATA['ifbench' if stage=='ifbench-score' else stage][2]
    state['completed'].append(stage); write(run/'status.json', state)


def main():
    p = argparse.ArgumentParser(description='New early-checkpoint child; original actual-result validators')
    p.add_argument('--child-stage', choices=STAGES[:-1], required=True)
    p.add_argument('--child-run', required=True)
    a = p.parse_args()
    root = Path('/data/spf-beta05-lr3e6-full-20261011-r1')
    if (sys.executable != PYTHON or os.name != 'posix'
            or TASK.parent != root or TASK.name != 'candidate-development'
            or a.child_run != str(TASK/'evaluation') or len(IDENTITY) != 64 or len(ADAPTER_SHA) != 64):
        raise ValueError('Exact new checkpoint child identity and runtime required')
    child(a.child_stage, TASK/'evaluation')

if __name__ == '__main__': main()
