"""Explicit backward/step boundary; never use Accelerator.backward for SPF.

DeepSpeed 0.15.4 gradients are inspected after backward and before engine.step.
Only local shards are retained on CPU. Full matrices exist for ONE block at a time.
"""
import math

import torch
import torch.distributed as dist

from construction.projection import project_block


class GradientBackend:
    def __init__(self, model, optimizer, scheduler, *, engine=None, clip=1.0):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.engine, self.clip = engine, clip
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.device = next(model.parameters()).device
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.have_gradients = False
        if self.world > 1 and engine is None:
            raise ValueError("Distributed construction requires ZeRO-3")
        if engine is not None:
            if engine.zero_optimization_stage() != 3:
                raise ValueError("Only ZeRO-3 is supported")
            from deepspeed.utils import (
                safe_get_full_grad, safe_get_local_grad, safe_set_full_grad,
                safe_set_local_grad, safe_get_local_fp32_param,
            )
            self.get_full = safe_get_full_grad
            self.get_local = safe_get_local_grad
            self.set_full = safe_set_full_grad
            self.set_local = safe_set_local_grad
            self.get_weight = safe_get_local_fp32_param

    def reduce_sum(self, value):
        value = torch.as_tensor(value, dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(value)
        return float(value)

    def clear(self):
        if self.engine is not None:
            if self.have_gradients:
                for p in self.params.values():
                    g = self.get_local(p)
                    if g is not None:
                        self.set_local(p, torch.zeros_like(g))
            # engine.zero_grad only clears param.grad, NOT ZeRO's accumulation state.
            self.engine.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        if self.engine is not None:
            # bitsandbytes' paged states live outside PyTorch's allocator. Idle
            # reserved blocks otherwise evict managed optimizer pages on 80GB cards.
            torch.cuda.empty_cache()

    def backward(self, loss, *, boundary):
        if self.engine is not None:
            self.engine.set_gradient_accumulation_boundary(boundary)
            # Loss was normalized over the actual global effective batch already.
            self.engine.backward(loss, scale_wrt_gas=False)
        else:
            loss.backward()

    def snapshot(self):
        self.have_gradients = True
        result = {}
        for n, p in self.params.items():
            g = self.get_local(p) if self.engine is not None else p.grad
            if g is not None:
                result[n] = g.detach().float().cpu().clone()
        return result

    def weights(self):
        return {
            n: (self.get_weight(p) if self.engine is not None else p).detach().float().cpu().clone()
            for n, p in self.params.items()
        }

    def combine(self, task, safety, *, method, coefficient, projection, step):
        energy = self.reduce_sum(sum(float(g.square().sum()) for g in task.values()) +
                                 sum(float(g.square().sum()) for g in safety.values()))
        if not math.isfinite(energy):
            raise ValueError("Non-finite task/safety gradients")
        dot = self.reduce_sum(sum(
            float((g * safety[n]).sum()) for n, g in task.items() if n in safety
        ))
        if not math.isfinite(dot):
            raise ValueError("Non-finite global gradient inner product")
        before, after = 0.0, 0.0
        for name, p in self.params.items():
            gt, gs = task.get(name), safety.get(name)
            if gt is None and method == "mixing" and gs is not None:
                gt = torch.zeros_like(gs)
            if gt is None:
                if self.engine is None:
                    p.grad = None
                continue
            before += float(gt.square().sum())
            if self.engine is None:
                projected = gt.to(p.device)
                if method == "mixing" and gs is not None:
                    projected = projected + coefficient * gs.to(p.device)
                elif method == "spf" and dot < 0:
                    projected = project_block(projected, gs, projection, step=step, name=name)
                p.grad = projected.to(p.dtype)
                after += float(projected.square().sum())
                continue
            # All ranks participate in the SAME parameter order. Never SVD a shard.
            if method == "spf" and dot < 0 and gs is not None:
                torch.cuda.empty_cache()
                self.set_local(p, gt.to(p.device))
                full_task = self.get_full(p)
                self.set_local(p, gs.to(p.device))
                full_safety = self.get_full(p)
                if self.rank == 0:
                    projected = project_block(full_task, full_safety, projection, step=step, name=name)
                else:
                    projected = torch.empty_like(full_task, dtype=torch.float32)
                if self.world > 1:
                    dist.broadcast(projected, src=0)
                self.set_full(p, projected)
                del full_task, full_safety, projected
            else:
                local = gt + coefficient * gs if method == "mixing" and gs is not None else gt
                self.set_local(p, local.to(p.device))
            after += float(self.get_local(p).float().square().sum())
        return {"global_dot": dot, "conflict": dot < 0,
                "gradient_norm_before": math.sqrt(self.reduce_sum(before)),
                "gradient_norm_after": math.sqrt(self.reduce_sum(after))}

    def step(self):
        if self.engine is not None:
            before = self.engine.global_steps
            torch.cuda.empty_cache()
            self.engine.set_gradient_accumulation_boundary(True)
            self.engine.step()
            if not self.engine._step_applied:
                raise RuntimeError("Optimizer skipped an update; construction failed")
            if self.engine.global_steps != before + 1:
                raise RuntimeError('Construction must apply exactly one engine update per task window')
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()

    def update_norm(self, previous):
        squared = sum(float((w - previous[n]).square().sum()) for n, w in self.weights().items())
        return math.sqrt(self.reduce_sum(squared))
