"""Explicit FP32 CUDA matmul policy for construction, isolated from NPO."""
from contextlib import contextmanager

import torch


def cuda_matmul_settings():
    return {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def configure_construction_cuda():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    actual = cuda_matmul_settings()
    if actual['cuda_matmul_allow_tf32'] or actual['float32_matmul_precision'] != 'highest':
        raise RuntimeError('Construction requires strict FP32 CUDA matmul')
    return actual


@contextmanager
def strict_projection_matmul(device):
    if device.type != 'cuda':
        yield
        return
    previous = cuda_matmul_settings()
    try:
        configure_construction_cuda()
        yield
    finally:
        torch.set_float32_matmul_precision(previous['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = previous['cuda_matmul_allow_tf32']
