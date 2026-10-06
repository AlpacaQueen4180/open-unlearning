"""Verify the new CUDA policy, restoring state and preserving CPU reference math."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from construction.numerics import cuda_matmul_settings, strict_projection_matmul
from construction.projection import ProjectionConfig, project_block


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous-source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source = Path(a.previous_source)
    spec = importlib.util.spec_from_file_location('previous_projection', source)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    records = []
    for device in ('cpu', 'cuda'):
        for method in ('exact', 'lowrank'):
            torch.manual_seed(42)
            if device == 'cuda':
                torch.cuda.manual_seed(42)
            task = torch.randn(32, 24, device=device)
            safety = torch.randn_like(task)
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
            settings = cuda_matmul_settings()
            actual = project_block(task, safety, ProjectionConfig(rank=3, method=method), step=7, name='acceptance')
            assert cuda_matmul_settings() == settings
            assert torch.equal(cpu_rng, torch.random.get_rng_state())
            assert cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state())
            with strict_projection_matmul(task.device):
                expected = previous.project_block(task, safety, previous.ProjectionConfig(rank=3, method=method), step=7, name='acceptance')
                residual = ((task - actual).T @ actual).abs().max().item()
            assert torch.equal(actual, expected), 'CPU math or strict CUDA math changed'
            assert residual <= 2e-4
            records.append({'device': device, 'method': method, 'original_math_bitwise_equal': True,
                            'max_abs_orthogonality': residual, 'original_atol': 2e-4,
                            'rng_and_precision_restored': True,
                            'output_sha256': hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()})
    result = {'status': 'pass', 'cases': records, 'completed_training_repeated': False,
              'previous_projection_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    path = Path(a.output)
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
