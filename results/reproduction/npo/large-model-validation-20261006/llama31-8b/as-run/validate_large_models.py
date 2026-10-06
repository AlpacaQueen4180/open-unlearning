"""Finite 8B then 7B native validation; locks, no polling or prior-run mutation."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/npo-gb200-20261004')
BASE = ROOT / 'validation-20261006'
TASK = Path(os.environ.get('NPO_LARGE_VALIDATION_TASK', str(ROOT / 'model-validation-20261006')))
PREVIOUS = Path(os.environ['NPO_LARGE_VALIDATION_PREVIOUS']) if os.environ.get('NPO_LARGE_VALIDATION_PREVIOUS') else None
prior = json.loads((PREVIOUS / 'status.json').read_text()) if PREVIOUS else None
if prior:
    assert prior['phase'] == 'FAILED', 'Only resume an exited failed queue'
completed = {x['label'] for x in prior['stages'] if x['returncode'] == 0} if prior else set()
REPO = BASE / 'upstream'
SOURCE = ROOT / 'repo/scripts/reproduction/npo/gb200'
PY = BASE / 'venv/bin/python'
OLD_PY = ROOT / 'venv/bin/python'
TASK.mkdir(exist_ok=True)
lock = (TASK / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (TASK / 'status.json').exists(), 'Refuse duplicate validation'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy'
state = {'phase': 'STARTING', 'pid': os.getpid(), 'started': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'stages': [], 'models': {}}
env = dict(os.environ, HF_HOME=str(ROOT / 'hf'), TOKENIZERS_PARALLELISM='false', HYDRA_FULL_ERROR='1', WANDB_MODE='disabled')


def save():
    temporary = TASK / 'status.tmp'
    temporary.write_text(json.dumps(state, indent=2, default=str))
    temporary.replace(TASK / 'status.json')


def run(label, command):
    if label in completed:
        state['stages'].append({'label': label, 'returncode': 0, 'reused_from': str(PREVIOUS)})
        save()
        return
    state.update(phase='RUNNING', current=label)
    save()
    (TASK / (label + '.command.json')).write_text(json.dumps(command, indent=2))
    with (TASK / (label + '.log')).open('w') as log:
        rc = subprocess.call(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['stages'].append({'label': label, 'returncode': rc})
    save()
    if rc:
        raise RuntimeError(label + ' failed; preserve original evidence')


ASSET_CODE = '''
import pathlib,json,hashlib,sys,os
from huggingface_hub import HfApi,snapshot_download,hf_hub_download
root=pathlib.Path('/data/npo-gb200-20261004');task=pathlib.Path(os.environ.get('NPO_LARGE_VALIDATION_TASK',str(root/'model-validation-20261006')));model=sys.argv[1]
original=json.loads((root/'assets.json').read_text());repo_id='open-unlearning/tofu_'+model+'_full'
assets={'model_id':repo_id,'model_revision':original['model_revision'] if model=='Llama-2-7b-chat-hf' else HfApi().model_info(repo_id).sha,'dataset_revision':original['dataset_revision'],'eval_revision':original['eval_revision']}
dest=task/('assets_'+model+'.json');assert not dest.exists()
dest.write_text(json.dumps(assets,indent=2))
assets['model_path']=snapshot_download(repo_id,revision=assets['model_revision'],allow_patterns=['*.json','*.safetensors','tokenizer*','*.model'])
assets['retain_file']='tofu_'+model+'_retain99/TOFU_EVAL.json'
assets['retain_path']=hf_hub_download('open-unlearning/eval',assets['retain_file'],repo_type='dataset',revision=assets['eval_revision'])
assets['retain_sha256']=hashlib.sha256(pathlib.Path(assets['retain_path']).read_bytes()).hexdigest()
dest.write_text(json.dumps(assets,indent=2));print(json.dumps(assets))
'''

save()
try:
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == '17cbbc87192e6934deb92875c359c91bbd837fb4'
    assert hashlib.sha256((REPO / 'src/evals/metrics/utils.py').read_bytes()).hexdigest() == 'acd98144cada6f02dea329bb78c77e317bd651d16fabbb576796253ae6c4ec7b'
    template_cell = ROOT / 'validation-20261006-r1/experiments/runs/gb200_1gpu_zero3_flash_attention_2_s0_native_554_1130_20261006_r1'
    train_template = json.loads((template_cell / 'training_command.json').read_text())
    eval_template = json.loads((template_cell / 'evaluating_command.json').read_text())
    for model, short in [('Llama-3.1-8B-Instruct', 'llama31_8b'), ('Llama-2-7b-chat-hf', 'llama2_7b')]:
        if short + '_prepare_assets' in completed:
            original_assets = PREVIOUS / ('assets_' + model + '.json')
            (TASK / original_assets.name).write_bytes(original_assets.read_bytes())
        run(short + '_prepare_assets', [str(PY), '-c', ASSET_CODE, model])
        assets_file = TASK / ('assets_' + model + '.json')
        assets = json.loads(assets_file.read_text())
        batch = TASK / (short + '-tofu-frozen-40.pt')
        extra = []
        if PREVIOUS and (PREVIOUS / batch.name).exists():
            batch = PREVIOUS / batch.name
            extra = ['--resume-metadata', '--metadata-output', str(TASK / (short + '-tofu-frozen-40.json'))]
        run(short + '_freeze_tofu', [str(OLD_PY), str(SOURCE / 'capture_large_model_batch.py'), '--model', model, '--assets', str(assets_file), '--output', str(batch)] + extra)
        for window, offset in [(8, 0), (2, 32)]:
            label = short + '_TOFU_w' + str(window) + '_combined_bf16'
            run(label, [str(PY), '-m', 'torch.distributed.run', '--nproc_per_node=1', '--master_port=29681', str(SOURCE / 'gradient_audit.py'), '--repo', str(REPO), '--output', str(TASK / (label + '.json')), '--gas', '8', '--window', str(window), '--term', 'combined', '--dtype', 'bf16', '--model-path', assets['model_path'], '--batch-file', str(batch), '--offset', str(offset), '--ds-reference'])
        cell = TASK / 'runs' / (short + '_forget01_s0_native_smoke4')
        cell.mkdir(parents=True, exist_ok=False)
        checkpoint = cell / 'checkpoint'
        ds = json.loads((template_cell / 'deepspeed.json').read_text())
        (cell / 'deepspeed.json').write_text(json.dumps(ds, indent=2))
        import yaml
        ac = yaml.safe_load((template_cell / 'accelerate.yaml').read_text())
        ac['deepspeed_config']['deepspeed_config_file'] = str(cell / 'deepspeed.json')
        (cell / 'accelerate.yaml').write_text(yaml.safe_dump(ac))

        def transform(command, evaluation=False):
            converted = []
            for value in command:
                if value == str(template_cell / 'accelerate.yaml'):
                    value = str(cell / 'accelerate.yaml')
                elif value == 'scripts/reproduction/npo/gb200/run_probe.py':
                    value = 'scripts/reproduction/npo/gb200/short_native_probe.py'
                elif value.startswith('model='):
                    value = 'model=' + model
                elif value.startswith('task_name='):
                    value = 'task_name=' + cell.name + ('_eval' if evaluation else '')
                elif value.startswith('model.model_args.pretrained_model_name_or_path='):
                    value = 'model.model_args.pretrained_model_name_or_path=' + (str(checkpoint) if evaluation else assets['model_path'])
                elif value.startswith('model.tokenizer_args.pretrained_model_name_or_path='):
                    value = 'model.tokenizer_args.pretrained_model_name_or_path=' + assets['model_path']
                elif value.startswith('retain_logs_path='):
                    value = 'retain_logs_path=' + assets['retain_path']
                elif value.startswith('trainer.args.output_dir='):
                    value = 'trainer.args.output_dir=' + str(checkpoint)
                elif value.startswith('paths.output_dir='):
                    value = 'paths.output_dir=' + str(cell / 'eval')
                converted.append(value)
            return converted

        training = transform(train_template) + ['+trainer.args.max_steps=4']
        (cell / 'manifest.json').write_text(json.dumps({'model': model, 'assets': assets, 'seed': 0, 'split': 'forget01', 'smoke_only': True, 'alpha': 1, 'gamma': 1, 'beta': .1, 'planned_updates': 4, 'expected_epochs': 2, 'micro4_gas8_global32': True, 'native_no_legacy_patch': True, 'upstream': '17cbbc87192e6934deb92875c359c91bbd837fb4'}, indent=2))
        state['models'][model] = {'cell': str(cell), 'phase': 'TRAINING_SMOKE'}
        run(short + '_train_smoke4', training)
        state['models'][model]['phase'] = 'EVALUATING_SMOKE_CHECKPOINT'
        run(short + '_evaluate_smoke4', transform(eval_template, evaluation=True))
        run(short + '_audit_smoke_evaluation', [str(PY), 'scripts/reproduction/npo/as-run/finalize_h100_reproduction.py', str(cell / 'eval/TOFU_SUMMARY.json'), str(cell / 'eval/TOFU_EVAL.json'), assets['retain_path'], str(checkpoint / 'trainer_state.json')])
        state['models'][model].update(phase='VALIDATION_DONE', summary=json.loads((cell / 'eval/TOFU_SUMMARY.json').read_text()))
        save()
    state.update(phase='DONE', finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
except Exception as error:
    state.update(phase='FAILED', error=repr(error))
    save()
    raise
