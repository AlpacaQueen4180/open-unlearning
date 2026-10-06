"""Blockwise SPF (Zhang et al., Algorithm 1), with isolated randomized SVD RNG.

Adapted from Zenodo 21289041 v1 (CC-BY-4.0). No safety examples are embedded.
Conflict detection is GLOBAL; this function projects one block only AFTER that test.
"""
import hashlib
import math
from dataclasses import dataclass

import torch
from construction.numerics import strict_projection_matmul


@dataclass(frozen=True)
class ProjectionConfig:
    rank: int = 20
    method: str = "lowrank"
    oversample: int = 5
    niter: int = 2
    seed: int = 0

    def __post_init__(self):
        if self.rank < 1 or self.oversample < 0 or self.niter < 0:
            raise ValueError("Invalid SVD rank/oversampling/iterations")
        if self.method not in ("exact", "lowrank"):
            raise ValueError("SVD method must be exact or lowrank")


def as_matrix(gradient):
    if gradient.ndim == 0:
        return gradient.reshape(1, 1)
    return gradient.reshape(gradient.shape[0], -1)


def project_block(task, safety, config=ProjectionConfig(), *, step=0, name=""):
    with strict_projection_matmul(task.device):
        return _project_block_fp32(task, safety, config, step=step, name=name)


def _project_block_fp32(task, safety, config=ProjectionConfig(), *, step=0, name=""):
    """Return an FP32 projection without modifying inputs or caller RNG states."""
    task = task.detach().float()
    if safety is None:
        return task.clone()
    if task.shape != safety.shape:
        raise ValueError("Task and safety block shapes differ")
    safety = safety.detach().to(device=task.device, dtype=torch.float32)
    if not torch.isfinite(task).all() or not torch.isfinite(safety).all():
        raise ValueError("Non-finite gradient")
    if not torch.count_nonzero(safety):
        return task.clone()
    gs, gt = as_matrix(safety), as_matrix(task)
    rank = min(config.rank, *gs.shape)
    if config.method == "exact":
        basis = torch.linalg.svd(gs, full_matrices=False).U[:, :rank]
    else:
        digest = hashlib.sha256(f"{config.seed}:{step}:{name}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        devices = [task.device.index] if task.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if task.is_cuda:
                with torch.cuda.device(task.device):
                    torch.cuda.manual_seed(seed)
            basis = torch.svd_lowrank(
                gs, q=min(min(gs.shape), rank + config.oversample), niter=config.niter
            )[0][:, :rank]
    return (gt - basis @ (basis.T @ gt)).reshape_as(task)


def project_gradients(task, safety, config=ProjectionConfig(), *, step=0):
    """Small-model reference: absent task gradients never become safety updates."""
    dot = sum(
        float((g.float() * safety[n].float()).sum())
        for n, g in task.items() if n in safety and safety[n] is not None
    )
    if not math.isfinite(dot) or any(not torch.isfinite(g).all() for g in task.values()):
        raise ValueError("Non-finite gradient")
    return {
        n: project_block(g, safety.get(n), config, step=step, name=n)
        if dot < 0 else g.detach().float().clone()
        for n, g in task.items()
    }, dot
