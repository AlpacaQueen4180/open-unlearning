"""Run the unchanged NPO entry point with auditable runtime metadata.

The hook only records runtime values; it does not change loss or gradients.
"""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo / "src"))
from trainer.unlearn.npo import NPO

original = NPO.training_step
observed_trainer = None


def training_step(self, model, inputs, *args, **kwargs):
    global observed_trainer
    observed_trainer = self
    self._npo_microsteps = getattr(self, "_npo_microsteps", 0) + 1
    if not getattr(self, "_npo_probe_recorded", False):
        self._npo_probe_recorded = True
        data = {
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": self.accelerator.num_processes,
            "deepspeed_enabled": self.is_deepspeed_enabled,
            "micro_batch": self.args.per_device_train_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "accelerator_gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
            "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
            "warmup_steps": self.args.warmup_steps,
            "max_steps": self.state.max_steps,
            "optimizer": self.args.optim.value,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "attention": self.model.config._attn_implementation,
        }
        if self.is_deepspeed_enabled:
            data["resolved_deepspeed"] = self.accelerator.state.deepspeed_plugin.deepspeed_config
            data["engine_gradient_accumulation_steps"] = model.gradient_accumulation_steps()
            data["engine_train_batch_size"] = model.train_batch_size()
            data["engine_zero_stage"] = model.zero_optimization_stage()
        if hasattr(self, '_npo_corrected_plan'):
            data['correction'] = self._npo_corrected_plan
        out = Path(self.args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"runtime_rank{data['rank']}.json").write_text(json.dumps(data, indent=2, default=str))
        print("NPO_RUNTIME " + json.dumps(data, default=str), flush=True)
    return original(self, model, inputs, *args, **kwargs)


NPO.training_step = training_step
runpy.run_path(str(repo / "src/train.py"), run_name="__main__")
if observed_trainer is not None:
    trainer = observed_trainer
    final = {"trainer_global_step": trainer.state.global_step,
             "microsteps": trainer._npo_microsteps, "epoch": trainer.state.epoch}
    if trainer.is_deepspeed_enabled:
        final.update(engine_global_steps=trainer.model_wrapped.global_steps,
                     engine_micro_steps=trainer.model_wrapped.micro_steps)
    if hasattr(trainer, '_npo_corrected_plan'):
        final['correction'] = trainer._npo_corrected_plan
        final['forced_update_boundaries'] = trainer._npo_corrected_boundaries
        assert final['trainer_global_step'] == final['engine_global_steps'] == trainer._npo_corrected_boundaries == trainer._npo_corrected_plan['planned_updates'], final
        if trainer._npo_corrected_plan['explicit_max_steps'] < 0:
            assert abs(final['epoch'] - trainer.args.num_train_epochs) < 1e-9, final
    rank = os.environ.get("RANK", "0")
    (Path(trainer.args.output_dir) / f"runtime_final_rank{rank}.json").write_text(json.dumps(final, indent=2))
