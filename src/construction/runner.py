"""Construction-only trainers with one explicit optimizer update per task batch.

This loop intentionally does not call HF/Accelerate's DeepSpeed backward wrapper,
which can step the engine before the safety projection has been applied.
"""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import Trainer, get_scheduler
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput

from construction.backend import GradientBackend
from construction.projection import ProjectionConfig
from construction.profile import world_size
from construction.numerics import configure_construction_cuda


def task_windows(length, batch_size, seed, epoch):
    order = torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    for start in range(0, length, batch_size):
        yield order[start:start + batch_size]


def assistant_tokens(batch):
    return int((batch["labels"][..., 1:] != -100).sum())


class ConstructionTrainer:
    method = "standard"

    def __init__(self, model, args, train_dataset, data_collator, processing_class=None,
                 template_args=None, eval_dataset=None, evaluators=None, *,
                 anchor_path=None, anchor_sha256=None, backend="zero3",
                 zero3_config="configs/accelerate/zero_stage3_offload_config.json",
                 mixing_coefficient=1.0, rank=20, svd_method="lowrank",
                 oversample=5, niter=2, warmup_epochs=1.0,
                 save_intermediate=True, save_runtime=True, **kwargs):
        if kwargs:
            raise ValueError(f"Unknown construction options: {sorted(kwargs)}")
        if backend not in ("single", "zero3") or mixing_coefficient < 0:
            raise ValueError("Invalid backend or mixing coefficient")
        if args.fp16 or args.num_train_epochs != int(args.num_train_epochs):
            raise ValueError("Construction supports FP32/bf16 and integer epochs only")
        self.model, self.args, self.train_dataset = model, args, train_dataset
        self.data_collator, self.tokenizer = data_collator, processing_class
        self.template_args = template_args
        self.backend_name, self.zero3_config = backend, zero3_config
        self.coefficient, self.warmup_epochs = mixing_coefficient, warmup_epochs
        self.projection = ProjectionConfig(rank, svd_method, oversample, niter, args.seed)
        self.state = TrainerState()
        self.engine = None
        self.save_intermediate, self.save_runtime = save_intermediate, save_runtime
        self.anchor = None
        self.cpu_rss_peak = 0
        if self.method != "standard":
            from construction.protocol import read_anchor
            from data.utils import preprocess_chat_instance
            messages = read_anchor(anchor_path, anchor_sha256)["messages"]
            self.anchor = self.data_collator([preprocess_chat_instance(
                self.tokenizer, template_args, [messages[0]["content"]],
                [messages[1]["content"]], 512,
            )])
            if not assistant_tokens(self.anchor):
                raise ValueError("Anchor has no unmasked assistant tokens")

    def _setup(self, steps, warmup):
        torch.manual_seed(self.args.seed)
        if self.backend_name == "zero3":
            import deepspeed
            if deepspeed.__version__ != "0.15.4":
                raise RuntimeError("SPF backend validated against DeepSpeed 0.15.4 only")
            deepspeed.init_distributed()
            if dist.get_world_size() != world_size() or not torch.cuda.is_available():
                raise RuntimeError("Construction requires the declared one/two CUDA rank profile")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        else:
            if int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise RuntimeError("Single backend cannot run with multiple ranks")
            device = self.args.device
            self.model.to(device)
        # The GB200 container enables TF32; construction's FP32 gate requires IEEE matmul.
        self.numerical_precision = configure_construction_cuda() if device.type == 'cuda' else {'device': 'cpu', 'unchanged': True}
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        # Keep optimizer choice and parameter groups aligned with HF Trainer.
        names = Trainer.get_decay_parameter_names(self, self.model)
        groups = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and n in names],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and n not in names],
             "weight_decay": 0.0},
        ]
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        optimizer = cls(groups, **kwargs)
        # Capture metadata before ZeRO replaces the optimizer with a wrapper.
        self.optimizer_metadata = {"class": type(optimizer).__name__, "defaults": dict(optimizer.defaults)}
        scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer, warmup, steps)
        if self.backend_name == "zero3":
            config = json.loads(Path(self.zero3_config).read_text())
            config.update({
                "train_micro_batch_size_per_gpu": self.args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "train_batch_size": self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * dist.get_world_size(),
                "gradient_clipping": self.args.max_grad_norm,
                "zero_allow_untested_optimizer": True,
                "bf16": {"enabled": self.args.bf16},
                "data_types": {"grad_accum_dtype": "fp32"},
            })
            # Native DeepSpeed does not resolve HuggingFace's 'auto' bucket values.
            for key, value in {"reduce_bucket_size": 50_000_000,
                               "stage3_prefetch_bucket_size": 50_000_000,
                               "stage3_param_persistence_threshold": 100_000}.items():
                config["zero_optimization"][key] = value
            # ZeRO casts floating buffers to BF16, including nonpersistent rotary
            # frequencies that HF reconstructs in FP32 on checkpoint reload.
            # Preserve their original values, not a rounded BF16->FP32 conversion.
            rotary_buffers = [(module, module.inv_freq.detach().clone())
                              for module in self.model.modules()
                              if hasattr(module, "inv_freq") and
                              isinstance(module.inv_freq, torch.Tensor)]
            self.engine, optimizer, _, scheduler = deepspeed.initialize(
                model=self.model, optimizer=optimizer, lr_scheduler=scheduler, config=config,
            )
            for module, original in rotary_buffers:
                module.inv_freq = original.to(device=module.inv_freq.device)
                if hasattr(module, "original_inv_freq"):
                    module.original_inv_freq = module.inv_freq
            self.resolved_deepspeed_config = config
        self.backend = GradientBackend(self.model, optimizer, scheduler, engine=self.engine,
                                       clip=self.args.max_grad_norm)
        self.forward_model = self.engine if self.engine is not None else self.model
        self.optimizer, self.lr_scheduler = optimizer, scheduler

    def _move(self, batch):
        return {k: v.to(self.backend.device) for k, v in batch.items()
                if k in ("input_ids", "labels", "attention_mask")}

    def _loss(self, batch):
        # The existing model's assistant-token CE. Normalize microbatches below
        # to produce a token mean over the ACTUAL global effective batch.
        if not assistant_tokens(batch):
            raise ValueError("Example batch has no assistant labels")
        return self.forward_model(**batch).loss

    def train(self, resume_from_checkpoint=None, **kwargs):
        if resume_from_checkpoint or kwargs:
            raise ValueError("Construction resume is not supported; use a new run directory")
        world = world_size() if self.backend_name == "zero3" else 1
        effective = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * world
        per_epoch = math.ceil(len(self.train_dataset) / effective)
        if not per_epoch:
            raise ValueError("Empty task dataset")
        total = per_epoch * int(self.args.num_train_epochs)
        if self.args.max_steps > 0:
            total = min(total, self.args.max_steps)
        self._setup(total, int(self.warmup_epochs * per_epoch))
        self.model.train()
        out = Path(self.args.output_dir)
        if self.backend.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            if (out / "trajectory.jsonl").exists():
                raise FileExistsError("Refusing to overwrite an existing construction run")
            resolved = {"method": self.method, "args": self.args.to_dict(),
                        "optimizer": type(self.optimizer).__name__,
                        "scheduler": self.args.lr_scheduler_type.value,
                        "loss_reduction": "global_effective_batch_assistant_token_mean",
                        "warmup_steps": int(self.warmup_epochs * per_epoch),
                        "total_updates": total, "effective_batch": effective, "world_size": world,
                        "optimizer_defaults": self.optimizer_metadata["defaults"],
                        "base_optimizer": self.optimizer_metadata["class"],
                        "numerical_precision": self.numerical_precision,
                        "projection": vars(self.projection),
                        "sampler": "seed-plus-epoch-global-randperm-no-padding",
                        "cuda_empty_cache_at_boundaries": self.backend_name == "zero3",
                        "deepspeed": getattr(self, "resolved_deepspeed_config", None)}
            (out / "resolved_training.json").write_text(json.dumps(resolved, indent=2, default=str))
        start = time.perf_counter()
        examples = tokens = safety_tokens = input_tokens = 0
        unique = set()
        checkpoints = {0, *(math.ceil(total * f) for f in (.25, .5, .75, 1))}
        # Step 0 is represented by the immutable base revision; no duplicate 16GB save.
        self._record(out, {"step": 0, "examples": 0, "tokens": 0, "elapsed_seconds": 0})
        for epoch in range(int(self.args.num_train_epochs)):
            for ids in task_windows(len(self.train_dataset), effective, self.args.seed, epoch):
                self.backend.clear()
                local_ids = ids[self.backend.rank::self.backend.world]
                micro = self.args.per_device_train_batch_size
                batches = [self.data_collator([self.train_dataset[i] for i in local_ids[j:j+micro]])
                           for j in range(0, len(local_ids), micro)]
                global_tokens = self.backend.reduce_sum(sum(assistant_tokens(b) for b in batches))
                global_input_tokens = self.backend.reduce_sum(sum(int(b["attention_mask"].sum()) for b in batches))
                if global_tokens <= 0:
                    raise ValueError("Effective batch has no assistant tokens")
                # Empty final rank still executes matching forward/backward collectives.
                count = math.ceil(math.ceil(len(ids) / self.backend.world) / micro)
                task_loss = 0.0
                weight_sum = 0.0
                lr_before = self.optimizer.param_groups[0]['lr']
                for j in range(count):
                    present = j < len(batches)
                    batch = self._move(batches[j] if present else self.data_collator([self.train_dataset[ids[0]]]))
                    weight = assistant_tokens(batch) * self.backend.world / global_tokens if present else 0.0
                    weight_sum += weight / self.backend.world
                    loss = self._loss(batch) * weight
                    task_loss += float(loss.detach()) / self.backend.world
                    self.backend.backward(loss, boundary=j == count - 1)
                normalized_sum = self.backend.reduce_sum(weight_sum)
                if abs(normalized_sum - 1.0) > 1e-10:
                    raise AssertionError('Task assistant-token denominator is not normalized')
                task = self.backend.snapshot()
                safety = {}
                if self.method == "spf" or (self.method == "mixing" and self.coefficient != 0):
                    self.backend.clear()
                    anchor = self._move(self.anchor)
                    self.backend.backward(self._loss(anchor), boundary=True)
                    safety = self.backend.snapshot()
                    safety_tokens += assistant_tokens(anchor) * self.backend.world
                stats = self.backend.combine(task, safety, method=self.method,
                                             coefficient=self.coefficient, projection=self.projection,
                                             step=self.state.global_step)
                del task, safety
                previous = self.backend.weights()
                self.backend.step()
                stats["optimizer_update_norm"] = self.backend.update_norm(previous)
                del previous
                self.state.global_step += 1
                self.state.epoch = epoch + min(1.0, (examples % len(self.train_dataset) + len(ids)) / len(self.train_dataset))
                examples += len(ids)
                tokens += int(global_tokens)
                input_tokens += int(global_input_tokens)
                unique.update(ids)
                stats.update({"step": self.state.global_step, "examples": examples, "tokens": tokens,
                              "assistant_tokens": tokens, "input_tokens": input_tokens,
                              "unique_examples": len(unique), "safety_tokens_computed": safety_tokens,
                              "task_loss": self.backend.reduce_sum(task_loss),
                              "epoch": self.state.epoch, "window_examples": len(ids),
                              "window_assistant_tokens": int(global_tokens),
                              "task_microbatches_per_rank": count,
                              "task_normalized_weight_sum": normalized_sum,
                              "backward_scale_wrt_gas": False,
                              "lr_before": lr_before, "lr_after": self.optimizer.param_groups[0]['lr'],
                              "engine_global_steps": self.engine.global_steps if self.engine else self.state.global_step,
                              "scheduler_steps": self.lr_scheduler.last_epoch,
                              "elapsed_seconds": time.perf_counter() - start})
                self._record(out, stats)
                if self.save_intermediate and self.state.global_step in checkpoints:
                    self.save_model(str(out / f"checkpoint-{self.state.global_step}"))
                if self.state.global_step >= total:
                    break
            if self.state.global_step >= total:
                break
        self.save_state()
        return TrainOutput(self.state.global_step, stats["task_loss"], stats)

    def _record(self, out, row):
        import psutil
        row["cpu_rss_bytes"] = psutil.Process().memory_info().rss
        self.cpu_rss_peak = max(self.cpu_rss_peak, row["cpu_rss_bytes"])
        row["cpu_rss_peak_sampled_bytes"] = self.cpu_rss_peak
        if os.name == "posix":
            import resource
            row["cpu_rss_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        if self.backend.world > 1:
            ranks = [None] * self.backend.world
            dist.all_gather_object(ranks, {k: row.get(k) for k in ("cpu_rss_bytes", "cpu_rss_peak_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")})
            row["rank_resources"] = ranks
        if self.backend.rank == 0:
            with (out / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def save_model(self, output_dir=None):
        path = Path(output_dir or self.args.output_dir)
        if self.engine is not None:
            if self.save_runtime:
                self.engine.save_checkpoint(str(path / "runtime"), tag="state",
                                            client_state={"construction_step": self.state.global_step})
            self.engine.save_16bit_model(str(path), "pytorch_model.bin")
            if self.backend.rank == 0:
                self.model.config.save_pretrained(path)
                # Convert our own freshly written checkpoint to standard sharded
                # safetensors, avoiding torch.load restrictions in downstream HF.
                state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
                self.model.save_pretrained(path, state_dict=state, safe_serialization=True)
                del state
                (path / "pytorch_model.bin").unlink()
        elif self.backend.rank == 0:
            self.model.save_pretrained(path, safe_serialization=True)
        if self.backend.rank == 0:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
            self.state.save_to_json(str(path / "trainer_state.json"))
        if self.backend.world > 1:
            dist.barrier()

    def save_state(self):
        if self.backend.rank == 0:
            self.state.save_to_json(str(Path(self.args.output_dir) / "trainer_state.json"))

    def evaluate(self, **kwargs):
        raise RuntimeError("Construction evaluation is a separate frozen-checkpoint process")


class SafetyMixingTrainer(ConstructionTrainer):
    method = "mixing"


class SPFTrainer(ConstructionTrainer):
    method = "spf"
