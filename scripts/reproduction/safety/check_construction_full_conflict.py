"""Two-update, no-checkpoint Llama stress test; synthetic gradients, NOT research data.

Forces safety=-task to exercise every nonzero block, including embedding/lm_head.
The second projection runs with the paged optimizer states already allocated.
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from construction.backend import GradientBackend
from construction.cli import load_recipe
from construction.data import ConstructionDataset, ConstructionCollator
from construction.protocol import read_rows, sha256, validate_manifest, write_json
from trainer import load_trainer
from construction.profile import accumulation, world_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4], default=4)
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    data = manifest['datasets']['full']
    if sha256(data['path']) != data['sha256']:
        raise ValueError('TOFU export checksum mismatch')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    recipe = load_recipe('spf')
    recipe['args'].update(output_dir=str(output), max_steps=2,
                          per_device_train_batch_size=args.microbatch,
                          gradient_accumulation_steps=accumulation(args.microbatch, world_size()))
    recipe['method_args'].update(anchor_path=manifest['anchor_path'],
                                 anchor_sha256=manifest['anchor_sha256'],
                                 save_intermediate=False, save_runtime=False)
    repo = Path(__file__).resolve().parents[3]
    template = OmegaConf.load(repo / 'configs/model/Llama-3.1-8B-Instruct.yaml').template_args
    torch.manual_seed(manifest['construction_seed'])
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model_id'], revision=manifest['model_revision'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_id'],
                                              revision=manifest['tokenizer_revision'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    trainer, _ = load_trainer(OmegaConf.create(recipe), model,
                              train_dataset=ConstructionDataset(read_rows(data['path']), tokenizer, template),
                              processing_class=tokenizer, data_collator=ConstructionCollator(tokenizer),
                              template_args=template)
    original = GradientBackend.combine
    events = []

    def force_conflict(backend, task, safety, **kwargs):
        started = time.perf_counter()
        populated = bool(backend.engine.optimizer.optimizer.state)
        synthetic = {name: -gradient for name, gradient in task.items()}
        result = original(backend, task, synthetic, **kwargs)
        if not result['conflict'] or result['gradient_norm_after'] > result['gradient_norm_before'] * 1.00001:
            raise AssertionError('Forced conflict projection failed')
        if kwargs['step'] == 1 and not populated:
            raise AssertionError('Second projection must include resident optimizer state')
        shapes = [(name, tuple(p.ds_shape)) for name, p in backend.params.items() if name in task]
        event = dict(result, step=kwargs['step'] + 1, optimizer_state_populated=populated,
                     blocks=len(shapes), largest_block=max(shapes, key=lambda item: __import__('math').prod(item[1])),
                     projection_seconds=time.perf_counter() - started)
        events.append(event)
        if backend.rank == 0:
            write_json(output / 'projection_progress.json', {'synthetic': True, 'events': events})
        return result

    GradientBackend.combine = force_conflict
    stop = threading.Event()
    peaks = [0, 0]
    samples = [0]

    def measure():
        while not stop.is_set():
            try:
                values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
                for i, value in enumerate(values.splitlines()[:2]):
                    peaks[i] = max(peaks[i], int(value) * 1024**2)
                samples[0] += 1
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(2)

    import os
    monitor = threading.Thread(target=measure, daemon=True) if int(os.environ.get('RANK', '0')) == 0 else None
    if monitor:
        monitor.start()
    try:
        result = trainer.train()
        if trainer.backend.rank == 0:
            assert result.global_step == 2 and len(events) == 2
            write_json(output / 'acceptance.json', {
                'status': 'pass', 'synthetic': True, 'research_evidence': False,
                'manifest_sha256': sha256(args.manifest), 'checkpoint_written': False,
                'microbatch': args.microbatch, 'accumulation': accumulation(args.microbatch, world_size()),
                'world_size': trainer.backend.world,
                'events': events, 'metrics': result.metrics,
                'nvml_sampled_peak_bytes': peaks, 'nvml_samples': samples[0],
                'nvml_interval_seconds': 2,
            })
    finally:
        GradientBackend.combine = original
        stop.set()
        if monitor:
            monitor.join(timeout=6)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
