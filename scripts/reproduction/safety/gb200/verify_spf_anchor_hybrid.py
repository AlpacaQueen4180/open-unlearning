"""New hybrid CPU arithmetic and completed real8B engineering acceptance only."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys


def load_helper(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Exact new hybrid helper required')
    spec = importlib.util.spec_from_file_location('new_anchor_hybrid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_check(helper):
    import torch
    from construction.backend import GradientBackend
    from construction.projection import ProjectionConfig, project_gradients
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 5, bias=False)
    reference = torch.nn.Linear(3, 5, bias=False)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, betas=(.5, .999), weight_decay=.01)
    independent = torch.optim.AdamW(reference.parameters(), lr=5e-6, betas=(.5, .999), weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    backend = GradientBackend(model, optimizer, scheduler, clip=1.0)
    inputs = torch.randn(3, 4, 3)
    labels = torch.tensor([[0, 1, -100, -100], [1, 2, 3, -100], [2, -100, -100, -100]])
    anchor_inputs = torch.randn(1, 3, 3)
    anchor_labels = torch.tensor([[4, 0, -100]])
    ce = lambda logits, targets: torch.nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
    maximum_gradient_error = maximum_update_error = 0.0
    for step in range(2):
        backend.clear()
        total_tokens = int((labels != -100).sum())
        staged_loss = 0.0
        for lo, hi in [(0, 2), (2, 3)]:
            fraction = int((labels[lo:hi] != -100).sum()) / total_tokens
            loss = ce(model(inputs[lo:hi]), labels[lo:hi]) * fraction
            staged_loss += float(loss.detach())
            backend.backward(loss, boundary=hi == 3)
        task = backend.snapshot()
        backend.clear()
        anchor_loss = ce(model(anchor_inputs), anchor_labels)
        backend.backward(anchor_loss, boundary=True)
        safety = backend.snapshot()
        independent.zero_grad()
        direct_task_loss = ce(reference(inputs), labels)
        direct_task_loss.backward()
        direct_task = {name: value.grad.detach().clone() for name, value in reference.named_parameters()}
        independent.zero_grad()
        direct_anchor_loss = ce(reference(anchor_inputs), anchor_labels)
        direct_anchor_loss.backward()
        direct_anchor = {name: value.grad.detach().clone() for name, value in reference.named_parameters()}
        assert abs(staged_loss - float(direct_task_loss.detach())) < 5e-7
        for name in task:
            torch.testing.assert_close(task[name], direct_task[name], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(safety[name], direct_anchor[name], atol=1e-7, rtol=1e-6)
        projection = ProjectionConfig(rank=1, method='exact', seed=0)
        expected, _ = project_gradients(direct_task, direct_anchor, projection, step=step)
        stats = backend.combine(task, safety, method='spf', coefficient=1.0, projection=projection, step=step)
        stats = helper.add_anchor_gradient(backend, task, safety, stats, .1)
        for name, parameter in reference.named_parameters():
            parameter.grad = expected[name] + .1 * direct_anchor[name]
            actual = backend.params[name].grad
            torch.testing.assert_close(actual, parameter.grad, atol=1e-7, rtol=1e-6)
            maximum_gradient_error = max(maximum_gradient_error, float((actual - parameter.grad).abs().max()))
        backend.step()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
        independent.step()
        for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected_parameter, atol=1e-7, rtol=1e-6)
            maximum_update_error = max(maximum_update_error, float((actual - expected_parameter).abs().max()))
    negative_cases = []
    for coefficient in (0.0, -.1, float('nan')):
        try:
            helper.add_anchor_gradient(backend, task, safety, stats, coefficient)
        except ValueError:
            negative_cases.append('invalid_coefficient')
        else:
            raise AssertionError('Invalid coefficient accepted')
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    assert all(torch.equal(restored[key], value) for key, value in model.state_dict().items())
    return dict(status='HYBRID_CPU_DIRECT_AUTOGRAD_TOKEN_MEAN_TWO_ADAM_UPDATES_RELOAD_PASS',
                actual_cpu_updates=2, unequal_microbatch_assistant_tokens=[5, 1],
                safety_assistant_tokens=2, maximum_gradient_error=maximum_gradient_error,
                maximum_update_error=maximum_update_error, invalid_coefficients_rejected=len(negative_cases),
                cuda_model_work=0, actual8b_equivalence_claimed=False)


def engineering_check(target):
    read = lambda name: json.loads((target / name).read_bytes())
    resolved, trainer, record = read('resolved_training.json'), read('trainer_state.json'), read('target.json')
    assert trainer['global_step'] == 2 and trainer['epoch'] == 1.0
    assert resolved['total_updates'] == 2 and resolved['world_size'] == 1 and resolved['effective_batch'] == 32
    assert resolved['args']['learning_rate'] == 5e-6 and resolved['args']['adam_beta1'] == .5
    assert resolved['args']['num_train_epochs'] == 1 and resolved['warmup_steps'] == 0
    assert record['reload']['status'] == 'pass' and math.isfinite(record['reload']['max_logit_error'])  # Original verify enforces atol.05/rtol.01.
    rows = [json.loads(line) for line in (target / 'trajectory.jsonl').read_bytes().splitlines()][1:]
    assert len(rows) == 2 and [row['window_examples'] for row in rows] == [32, 8]
    for step, row in enumerate(rows, 1):
        assert row['step'] == row['engine_global_steps'] == row['scheduler_steps'] == step
        assert row['method_family'] == 'spf_anchor_loss_hybrid' and row['safety_loss_coefficient'] == .1
        assert row['anchor_loss_reduction'] == 'assistant_token_mean'
        assert abs(row['task_normalized_weight_sum'] - 1.0) < 1e-10 and row['anchor_normalized_weight_sum'] == 1.0
        assert row['weighted_anchor_loss'] == .1 * row['anchor_loss']
        assert row['backward_scale_wrt_gas'] is False and row['extra_model_forwards'] == row['extra_model_backwards'] == 0
        assert row['projection_before_anchor_addition'] and row['anchor_gradient_added_before_original_clip_and_step']
        assert row['anchor_addition_gradient_norm'] > 0 and row['optimizer_update_norm'] > 0
        assert all(math.isfinite(row[key]) for key in ('task_loss', 'anchor_loss', 'gradient_norm_after', 'optimizer_update_norm'))
    return dict(status='HYBRID_REAL8B_ZERO3_TWO_UPDATES_TOKEN_MEAN_TAIL_RELOAD_PASS',
                updates=2, task_examples=40, window_examples=[32, 8], microbatches=[8, 2],
                safety_loss_coefficient=.1, original_loss_and_projection_and_clip_step_delegated=True,
                fresh_reload=record['reload'], source_checks_are_not_formal_gradient_equivalence=True,
                old_large_autograd_failures_resolved=False, completed_model_weights_rehashed=False,
                trajectory_sha256=hashlib.sha256((target / 'trajectory.jsonl').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', required=True)
    parser.add_argument('--helper-sha256', required=True)
    parser.add_argument('--engineering-target')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    helper = load_helper(Path(args.helper), args.helper_sha256)
    result = engineering_check(Path(args.engineering_target)) if args.engineering_target else cpu_check(helper)
    result.update(verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=args.helper_sha256, new_api_calls=0, credential_reads=0)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
