"""Read-only instrumentation for native large-model NPO smoke training."""
import json
import math
import os
from pathlib import Path
import runpy
import sys

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / 'src'))
from trainer.unlearn.npo import NPO

original_loss = NPO.compute_loss
original_step = NPO.training_step
observed = None


def compute_loss(self, *args, **kwargs):
    result = original_loss(self, *args, **kwargs)
    loss = result[0] if isinstance(result, tuple) else result
    self._validation_raw_loss = float(loss.detach())
    return result


def training_step(self, model, inputs, *args, **kwargs):
    global observed
    observed = self
    self._validation_microsteps = getattr(self, '_validation_microsteps', 0) + 1
    if not hasattr(self, '_validation_backward_trace'):
        self._validation_backward_trace = []
        original_backward = model.backward

        def backward(loss, *backward_args, **backward_kwargs):
            raw = self._validation_raw_loss
            actual = float(loss.detach())
            window = self.current_gradient_accumulation_steps
            self._validation_backward_trace.append({
                'microstep': self._validation_microsteps,
                'raw_npo_loss': raw,
                'loss_entering_ds_backward': actual,
                'actual_window_microbatches': window,
                'ratio': actual / raw if raw else None,
                'expected_ratio': 1 / window,
                'boundary': model.is_gradient_accumulation_boundary(),
                'kwargs': backward_kwargs,
            })
            return original_backward(loss, *backward_args, **backward_kwargs)

        model.backward = backward
    return original_step(self, model, inputs, *args, **kwargs)


NPO.compute_loss = compute_loss
NPO.training_step = training_step
runpy.run_path(str(repo / 'src/train.py'), run_name='__main__')
assert observed is not None
trainer = observed
engine = trainer.model_wrapped
trace = trainer._validation_backward_trace
out = Path(trainer.args.output_dir)
runtime = {
    'trainer_global_step': trainer.state.global_step,
    'engine_global_steps': engine.global_steps,
    'engine_micro_steps': engine.micro_steps,
    'microsteps': trainer._validation_microsteps,
    'epoch': trainer.state.epoch,
    'attention': engine.module.config._attn_implementation,
    'world_size': trainer.accelerator.num_processes,
    'micro_batch': trainer.args.per_device_train_batch_size,
    'configured_gas': trainer.args.gradient_accumulation_steps,
    'optimizer': trainer.args.optim.value,
    'resolved_deepspeed': trainer.accelerator.state.deepspeed_plugin.deepspeed_config,
    'loss_scaling_trace': trace,
    'probe_policy': 'records compute_loss and engine.backward; no boundary/loss mutation',
}
(out / 'runtime_final_rank0.json').write_text(json.dumps(runtime, indent=2, default=str))
assert runtime['trainer_global_step'] == runtime['engine_global_steps'] == 4, runtime
assert runtime['microsteps'] == len(trace) == 20 and runtime['epoch'] == 2, runtime
assert runtime['attention'] == 'flash_attention_2'
assert runtime['world_size'] == 1 and runtime['micro_batch'] == 4 and runtime['configured_gas'] == 8
assert [x['actual_window_microbatches'] for x in trace] == ([8] * 8 + [2] * 2) * 2
assert [i + 1 for i, x in enumerate(trace) if x['boundary']] == [8, 10, 18, 20]
assert all(x['kwargs'].get('scale_wrt_gas') is False for x in trace)
assert all(math.isclose(x['ratio'], x['expected_ratio'], rel_tol=2e-6, abs_tol=1e-8) for x in trace)
print('NATIVE_SMOKE_AUDIT ' + json.dumps(runtime, default=str), flush=True)
