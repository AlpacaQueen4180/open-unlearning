"""Opt-in correction of epoch planning and DeepSpeed update boundaries only.

Keep the pinned stack and NPO loss scaling unchanged to isolate this intervention.
"""
import math
from pathlib import Path
import runpy
import sys

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / 'src'))
from trainer.unlearn.npo import NPO

original_values = NPO.set_initial_training_values
original_step = NPO.training_step

def corrected_values(self, args, dataloader, total_train_batch_size):
    values = list(original_values(self, args, dataloader, total_train_batch_size))
    if values[5] is None:
        raise ValueError('Corrected experiment requires a finite dataloader')
    updates = max(math.ceil(values[5] / args.gradient_accumulation_steps), 1)
    values[1] = updates
    if values[4]:
        values[6] = math.ceil(args.num_train_epochs * updates)
    else:
        values[0] = math.ceil(args.max_steps / updates)
    self._npo_corrected_boundaries = 0
    self._npo_corrected_plan = {'policy':'ceil_epoch_and_sync_ds_boundary_v1',
                               'dataloader_batches':values[5], 'updates_per_epoch':updates,
                               'planned_updates':values[6], 'configured_epochs':args.num_train_epochs,
                               'explicit_max_steps':args.max_steps,
                               'loss_scaling':'unchanged pinned NPO/Trainer behavior'}
    return tuple(values)

def corrected_step(self, model, inputs, *args, **kwargs):
    if self.is_deepspeed_enabled:
        # Trainer sets sync_gradients before entering training_step, including
        # the incomplete final group. Set before forward/backward for ZeRO3.
        boundary = bool(self.accelerator.sync_gradients)
        model.set_gradient_accumulation_boundary(boundary)
        self._npo_corrected_boundaries += int(boundary)
    return original_step(self, model, inputs, *args, **kwargs)

NPO.set_initial_training_values = corrected_values
NPO.training_step = corrected_step
runpy.run_path(str(Path(__file__).with_name('run_probe.py')), run_name='__main__')
