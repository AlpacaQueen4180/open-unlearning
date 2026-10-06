"""Recover from TF32 failure without changing gates or rerunning valid CPU stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', required=True)
    p.add_argument('--task', required=True)
    a = p.parse_args()
    old, new = Path(a.previous), Path(a.task)
    assert old.parent == new.parent == Path('/data') and new != old
    status = json.loads((old / 'status.json').read_text())
    stages = ['setup', 'verify_m0_assets', 'cpu_tests', 'tiny_reference']
    assert status['status'] == 'FAILED' and status['stage'] == 'tiny_zero3_world1' and status['completed'] == stages
    assert not (new / 'repo').exists() and not (new / 'status.json').exists()
    diagnosis_path = new / 'projection-precision-diagnosis.json'
    diagnosis = json.loads(diagnosis_path.read_text())
    assert diagnosis['strict_all_pass'] and diagnosis['threshold_changed'] is False
    assert diagnosis['source_sha256'] == sha(old / 'repo/src/construction/projection.py')
    assert len(diagnosis['cases']) == 6 and all(r['original_atol'] == 2e-4 for r in diagnosis['cases'])
    assert all(r['ieee_check_max_abs'] > 2e-4 for r in diagnosis['cases'] if r['projection_mode'] == 'container_default')
    runner_path = Path('src/construction/runner.py')
    prior = (old / 'repo' / runner_path).read_text()
    updated = (new / 'updates' / runner_path).read_text()
    additions = [
        'from construction.numerics import configure_construction_cuda\n',
        "        # The GB200 container enables TF32; construction's FP32 gate requires IEEE matmul.\n        self.numerical_precision = configure_construction_cuda() if device.type == 'cuda' else {'device': 'cpu', 'unchanged': True}\n",
        '                        "numerical_precision": self.numerical_precision,\n',
    ]
    for addition in additions:
        assert updated.count(addition) == 1
        updated = updated.replace(addition, '')
    assert updated == prior, 'Reject unrelated trainer changes'
    projection_path = Path('src/construction/projection.py')
    prior_projection = (old / 'repo' / projection_path).read_text()
    updated_projection = (new / 'updates' / projection_path).read_text()
    wrapper = 'def project_block(task, safety, config=ProjectionConfig(), *, step=0, name=""):\n    with strict_projection_matmul(task.device):\n        return _project_block_fp32(task, safety, config, step=step, name=name)\n\n\n'
    normalized = updated_projection.replace('from construction.numerics import strict_projection_matmul\n', '').replace(wrapper, '').replace('def _project_block_fp32(', 'def project_block(')
    assert normalized == prior_projection, 'Original FP32 projection arithmetic must remain identical'
    check_path = Path('scripts/reproduction/safety/check_construction_zero3.py')
    prior_check = (old / 'repo' / check_path).read_text()
    updated_check = (new / 'updates' / check_path).read_text()
    parameter_block = 'if args.backend == "zero3":\n    from deepspeed.utils import safe_get_full_fp32_param\n    parameters = {n: safe_get_full_fp32_param(p).cpu() for n, p in trainer.model.named_parameters()}\nelse:\n    parameters = {n: p.detach().cpu() for n, p in trainer.model.named_parameters()}\n'
    raw_save = 'if trainer.backend.rank == 0:\n    torch.save(parameters, Path(args.output) / "raw-parameters.pt")\n    torch.save(gradient_trace, Path(args.output) / "raw-gradients.pt")\n'
    normalized_check = updated_check.replace(parameter_block + raw_save, '').replace('trainer.save_model()\n', parameter_block + 'trainer.save_model()\n').replace('        "numerical_precision": trainer.numerical_precision,\n', '')
    assert normalized_check == prior_check, 'Keep all original acceptance thresholds and comparisons'
    allowed = {runner_path, projection_path, check_path, Path('src/construction/numerics.py')}
    helper_root = Path('scripts/reproduction/safety/gb200')
    for file in (new / 'updates').rglob('*'):
        if file.is_file():
            relative = file.relative_to(new / 'updates')
            assert relative in allowed or relative.parent == helper_root
    shutil.copytree(old / 'repo', new / 'repo')
    for file in (new / 'updates').rglob('*'):
        if file.is_file():
            destination = new / 'repo' / file.relative_to(new / 'updates')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    shutil.copytree(old / 'tiny-reference', new / 'tiny-reference')
    verification = new / 'precision-verification.json'
    env = {**os.environ, 'PYTHONPATH': str(new / 'repo/src')}
    command = [sys.executable, str(new / 'repo' / helper_root / 'verify_precision_recovery.py'),
               '--previous-source', str(old / 'repo' / projection_path), '--output', str(verification)]
    with (new / 'precision-verification.log').open('w') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    assert json.loads(verification.read_text())['status'] == 'pass'
    manifest = json.loads((old / 'prepared.json').read_text())
    manifest['recovery_parent_prepared_sha256'] = sha(old / 'prepared.json')
    manifest['construction_numerical_policy'] = {'cuda_float32_matmul': 'highest', 'allow_tf32': False,
        'cpu_reference_unchanged': True, 'diagnosis_sha256': sha(diagnosis_path), 'verification_sha256': sha(verification)}
    manifest['construction_code'] = {str(f.relative_to(new / 'repo')): sha(f) for f in (new / 'repo').rglob('*')
        if f.is_file() and f.suffix in ('.py', '.json', '.yaml') and '__pycache__' not in f.parts}
    (new / 'prepared.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    proof = {'status': 'pass', 'fix_scope': 'strict CUDA FP32 matmul', 'previous_task': str(old),
        'reused_stages': stages, 'previous_recovery_sha256': sha(old / 'recovery-evidence.json'),
        'previous_status_sha256': sha(old / 'status.json'), 'previous_prepared_sha256': sha(old / 'prepared.json'),
        'previous_runner_sha256': sha(old / 'repo' / runner_path), 'updated_runner_sha256': sha(new / 'repo' / runner_path),
        'diagnosis_sha256': sha(diagnosis_path), 'verification_sha256': sha(verification),
        'original_projection_math_preserved': True, 'original_acceptance_thresholds_preserved': True,
        'cpu_reference_hashes': {str(f.relative_to(new / 'tiny-reference')): sha(f) for f in (new / 'tiny-reference').rglob('*') if f.is_file()},
        'prepared_sha256': sha(new / 'prepared.json'), 'verification_command': command}
    (new / 'recovery-evidence.json').write_text(json.dumps(proof, indent=2) + '\n')
    shutil.copy2(old / 'environment-freeze.txt', new / 'environment-freeze.txt')
    print(json.dumps({'status': 'pass', 'task': str(new), 'reused_stages': stages,
                      'prepared_sha256': proof['prepared_sha256'], 'recovery_sha256': sha(new / 'recovery-evidence.json')}))


if __name__ == '__main__':
    main()
