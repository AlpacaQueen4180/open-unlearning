"""Add training-anchor CE gradients after the original SPF projection, before clipping.

The frozen runner computes both assistant-token means and their independent
gradients. This adapter consumes those gradients without another model forward
or backward. The family is explicitly SPF plus anchor loss, not pure SPF.
"""
import math


def add_anchor_gradient(backend, task, safety, stats, coefficient):
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Positive finite training-anchor coefficient required')
    if backend.world != 1 or not safety or set(task) != set(safety):
        raise ValueError('World1 and identical nonempty task/anchor gradient sets required')
    import torch
    squared = anchor_squared = after_dot = 0.0
    for name, parameter in backend.params.items():
        if name not in task:
            continue
        current = backend.get_local(parameter) if backend.engine is not None else parameter.grad
        anchor = safety[name].to(device=parameter.device, dtype=torch.float32)
        if current is None or current.shape != anchor.shape or not torch.isfinite(anchor).all():
            raise ValueError('Missing or invalid local anchor gradient')
        combined = current.detach().float() + coefficient * anchor
        if not torch.isfinite(combined).all():
            raise ValueError('Nonfinite hybrid gradient')
        if backend.engine is not None:
            backend.set_local(parameter, combined)
        else:
            parameter.grad = combined.to(parameter.dtype)
        squared += float(combined.square().sum())
        anchor_squared += float(anchor.square().sum())
        after_dot += float((combined * anchor).sum())
    result = dict(stats)
    result.update(method_family='spf_anchor_loss_hybrid', safety_loss_coefficient=coefficient,
        projection_before_anchor_addition=True, gradient_norm_after_spf=stats['gradient_norm_after'],
        gradient_norm_after=math.sqrt(backend.reduce_sum(squared)),
        anchor_gradient_norm=math.sqrt(backend.reduce_sum(anchor_squared)),
        anchor_addition_gradient_norm=coefficient * math.sqrt(backend.reduce_sum(anchor_squared)),
        hybrid_anchor_gradient_dot=backend.reduce_sum(after_dot),
        anchor_gradient_added_before_original_clip_and_step=True)
    return result


def install(trainer, coefficient):
    backend = trainer.backend
    if trainer.method != 'spf' or backend.world != 1 or backend.engine is None:
        raise ValueError('Original world1 ZeRO3 SPF construction required')
    original_snapshot, original_loss, original_combine = backend.snapshot, trainer._loss, backend.combine
    phase = {'name': 'task', 'anchor_loss': None}

    def loss(batch):
        value = original_loss(batch)
        if phase['name'] == 'anchor':
            if phase['anchor_loss'] is not None:
                raise ValueError('Exactly one original anchor forward per task window required')
            phase['anchor_loss'] = float(value.detach())
        return value

    def snapshot():
        result = original_snapshot()
        if phase['name'] == 'task':
            phase['name'] = 'anchor'
        elif phase['name'] == 'anchor' and phase['anchor_loss'] is not None:
            phase['name'] = 'combine'
        else:
            raise ValueError('Independent task then anchor gradients required')
        return result

    def combine(task, safety, *, method, coefficient: float, projection, step):
        if phase['name'] != 'combine' or method != 'spf':
            raise ValueError('Original SPF projection boundary required')
        stats = original_combine(task, safety, method=method, coefficient=coefficient,
                                 projection=projection, step=step)
        stats = add_anchor_gradient(backend, task, safety, stats, anchor_coefficient)
        stats.update(anchor_loss=phase['anchor_loss'],
                     weighted_anchor_loss=anchor_coefficient * phase['anchor_loss'],
                     anchor_loss_reduction='assistant_token_mean', anchor_normalized_weight_sum=1.0,
                     extra_model_forwards=0, extra_model_backwards=0)
        phase.update(name='task', anchor_loss=None)
        return stats

    anchor_coefficient = coefficient
    trainer._loss, backend.snapshot, backend.combine = loss, snapshot, combine
    return dict(method_family='spf_anchor_loss_hybrid', safety_loss_coefficient=coefficient,
                gradient_formula='original_SPF(task_gradient, anchor_gradient) + lambda * anchor_gradient',
                original_training_loop_and_loss_delegated=True, original_projection_delegated=True,
                original_backward_scaling_and_clip_step_delegated=True,
                anchor_source='frozen_training_only_anchor', development_used_as_training=False,
                additional_forward_backward_calls=0)
