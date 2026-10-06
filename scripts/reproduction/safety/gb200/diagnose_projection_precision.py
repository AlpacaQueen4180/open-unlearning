"""Finite CUDA diagnosis; preserve the acceptance threshold and original source."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.projection_source)
    spec = importlib.util.spec_from_file_location('original_projection', source)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    initial = {'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
               'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
               'float32_matmul_precision': torch.get_float32_matmul_precision()}
    rows = []
    for seed in (12, 42, 0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        task = torch.randn(32, 24, device='cuda')
        safety = torch.randn_like(task)
        inputs = hashlib.sha256(task.cpu().numpy().tobytes() + safety.cpu().numpy().tobytes()).hexdigest()
        for strict in (False, True):
            torch.set_float32_matmul_precision('highest' if strict else initial['float32_matmul_precision'])
            torch.backends.cuda.matmul.allow_tf32 = False if strict else initial['allow_tf32']
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            projected = module.project_block(task, safety, module.ProjectionConfig(rank=3), step=7, name='acceptance')
            rng_preserved = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            removed = task - projected
            native_residual = (removed.T @ projected).abs().max().item()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            ieee_residual = (removed.T @ projected).abs().max().item()
            rows.append({'seed': seed, 'input_sha256': inputs, 'projection_mode': 'strict' if strict else 'container_default',
                         'native_check_max_abs': native_residual, 'ieee_check_max_abs': ieee_residual,
                         'original_atol': 2e-4, 'rng_preserved': rng_preserved,
                         'norm_nonincreasing': projected.norm().item() <= task.norm().item() + 1e-5,
                         'projected_sha256': hashlib.sha256(projected.cpu().numpy().tobytes()).hexdigest()})
    torch.set_float32_matmul_precision(initial['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = initial['allow_tf32']
    result = {'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'initial_precision': initial,
              'environment_overrides': {key: os.environ.get(key) for key in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')},
              'cases': rows, 'threshold_changed': False,
              'strict_all_pass': all(r['ieee_check_max_abs'] <= 2e-4 and r['rng_preserved'] and r['norm_nonincreasing'] for r in rows if r['projection_mode'] == 'strict')}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'Do not overwrite diagnostic evidence'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
