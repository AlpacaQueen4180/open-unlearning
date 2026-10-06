"""Fresh-process two-rank acceptance test, comparing against single-process results.

PYTHONPATH=src torchrun --nproc_per_node=2 scripts/reproduction/safety/check_construction_zero3.py --output /private/acceptance
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--method", choices=["standard", "mixing", "spf"], default="spf")
parser.add_argument("--backend", choices=["single", "zero3"], default="zero3")
parser.add_argument("--reference")
parser.add_argument('--global-batch', type=int, default=32)
parser.add_argument('--tail-examples', type=int, default=24)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fixtures", root / "tests/construction/test_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# This synthetic fixture is small enough to gather every gradient for numerical
# acceptance. Production construction never keeps full-model gradient copies.
from construction.backend import GradientBackend
gradient_trace = []
original_combine = GradientBackend.combine


def capture_combined_gradients(self, *positional, **keywords):
    result = original_combine(self, *positional, **keywords)
    gradients = {}
    for name, param in self.params.items():
        gradient = self.get_full(param) if self.engine is not None else param.grad
        gradients[name] = None if gradient is None else gradient.detach().float().cpu().clone()
    gradient_trace.append(gradients)
    return result


GradientBackend.combine = capture_combined_gradients
trainer = module.train_case(Path(args.output), args.method,
                            micro=args.global_batch if args.backend == "single" else 4, backend=args.backend,
                            global_batch=args.global_batch, tail_examples=args.tail_examples)
if dist.is_initialized():
    dist.barrier()
trajectory = [json.loads(line) for line in
              (Path(trainer.args.output_dir) / "trajectory.jsonl").read_text().splitlines()]
conflicting_updates = sum(row.get("conflict", False) for row in trajectory)
if args.method == "spf":
    assert conflicting_updates > 0, "SPF fixture must exercise the projection branch"
from construction.projection import ProjectionConfig, project_block
if args.backend == "zero3":
    from deepspeed.utils import safe_get_full_fp32_param
    parameters = {n: safe_get_full_fp32_param(p).cpu() for n, p in trainer.model.named_parameters()}
else:
    parameters = {n: p.detach().cpu() for n, p in trainer.model.named_parameters()}
if trainer.backend.rank == 0:
    torch.save(parameters, Path(args.output) / "raw-parameters.pt")
    torch.save(gradient_trace, Path(args.output) / "raw-gradients.pt")
task = torch.randn(32, 24, device=trainer.backend.device)
safety = torch.randn_like(task)
cpu_rng = torch.random.get_rng_state().clone()
cuda_rng = torch.cuda.get_rng_state().clone() if task.is_cuda else None
projected = project_block(task, safety, ProjectionConfig(rank=3), step=7, name="acceptance")
assert torch.equal(cpu_rng, torch.random.get_rng_state())
if cuda_rng is not None:
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
assert torch.equal(projected, project_block(task, safety, ProjectionConfig(rank=3), step=7, name="acceptance"))
removed = task - projected
torch.testing.assert_close(removed.T @ projected, torch.zeros(24, 24, device=task.device), atol=2e-4, rtol=0)
assert projected.norm() <= task.norm() + 1e-5
trainer.save_model()
if trainer.backend.rank == 0:
    if args.reference:
        expected = torch.load(args.reference, weights_only=True)
        for name, tensor in parameters.items():
            torch.testing.assert_close(tensor, expected[name], atol=5e-6, rtol=5e-5)
        expected_gradients = torch.load(Path(args.reference).with_name("gradients.pt"), weights_only=True)
        assert len(gradient_trace) == len(expected_gradients) == trainer.state.global_step
        for actual_step, expected_step in zip(gradient_trace, expected_gradients):
            assert actual_step.keys() == expected_step.keys()
            for name, actual in actual_step.items():
                torch.testing.assert_close(actual, expected_step[name], atol=5e-6, rtol=5e-5)
    destination = Path(args.output)
    from transformers import GPT2LMHeadModel
    loaded = GPT2LMHeadModel.from_pretrained(trainer.args.output_dir)
    for name, param in loaded.named_parameters():
        torch.testing.assert_close(param.detach().cpu(), parameters[name], atol=5e-6, rtol=5e-5)
    torch.save(parameters, destination / "parameters.pt")
    torch.save(gradient_trace, destination / "gradients.pt")
    (destination / "acceptance.json").write_text(json.dumps({
        "method": args.method, "backend": args.backend, "steps": trainer.state.global_step,
        "world_size": trainer.backend.world,
        "numerical_precision": trainer.numerical_precision,
        "global_batch": args.global_batch,
        "actual_windows": [r['window_examples'] for r in trajectory if r['step'] > 0],
        "token_denominators": [r['window_assistant_tokens'] for r in trajectory if r['step'] > 0],
        "scheduler_steps": trainer.lr_scheduler.last_epoch,
        "reference_comparison": "pass" if args.reference else "reference_created",
        "gradient_comparison": "pass" if args.reference else "reference_created",
        "serialized_reload": "pass",
        "randomized_svd_rng_and_orthogonality": "pass",
        "conflicting_updates": conflicting_updates,
    }, indent=2))
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
