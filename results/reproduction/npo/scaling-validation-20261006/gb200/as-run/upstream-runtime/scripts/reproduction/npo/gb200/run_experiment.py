"""Bounded GB200 NPO controls, with persistent commands, status, and metrics."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get("NPO_ROOT", "/data/npo-gb200-20261004"))
REPO = ROOT / "repo"
MODEL = "open-unlearning/tofu_Llama-2-7b-chat-hf_full"
RETAIN = "tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, choices=[1, 2], required=True)
    parser.add_argument("--stages", nargs="+", choices=["3", "2", "none"], default=["3", "none"])
    parser.add_argument("--attention", choices=["eager", "flash_attention_2"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--model", default="Llama-2-7b-chat-hf", choices=["Llama-2-7b-chat-hf", "Llama-3.2-1B-Instruct"])
    parser.add_argument("--forget", default="forget05", choices=["forget01", "forget05", "forget10"])
    parser.add_argument("--corrected", action="store_true")
    args = parser.parse_args()
    model_id = f"open-unlearning/tofu_{args.model}_full"
    retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.forget]
    retain_file = f"tofu_{args.model}_{retain_split}/TOFU_EVAL.json"
    import torch
    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError(f"Expected {args.world_size} GPUs, got {torch.cuda.device_count()}")
    if any("GB200" not in torch.cuda.get_device_name(i) for i in range(args.world_size)):
        raise RuntimeError("This experiment requires GB200 devices")
    os.environ.update(HF_HOME=str(ROOT / "hf"), TOKENIZERS_PARALLELISM="false", HYDRA_FULL_ERROR="1", WANDB_MODE="disabled")
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    api = HfApi()
    assets_file = ROOT / ("assets.json" if args.model == "Llama-2-7b-chat-hf" and args.forget == "forget05" else f"assets_{args.model}_{args.forget}.json")
    if assets_file.exists():
        assets = json.loads(assets_file.read_text())
    else:
        original = json.loads((ROOT / "assets.json").read_text())
        assets = {"model_revision": original['model_revision'] if args.model == "Llama-2-7b-chat-hf" else api.model_info(model_id).sha,
                  "eval_revision": api.dataset_info("open-unlearning/eval").sha,
                  "dataset_revision": original['dataset_revision']}
        assets['eval_revision'] = original['eval_revision']
        write_json(assets_file, assets)
    model_path = snapshot_download(model_id, revision=assets["model_revision"], allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"])
    retain_path = hf_hub_download("open-unlearning/eval", retain_file, repo_type="dataset", revision=assets["eval_revision"])
    assets.update(model=model_id, retain_file=retain_file)
    assets["retain_sha256"] = hashlib.sha256(Path(retain_path).read_bytes()).hexdigest()
    write_json(assets_file, assets)
    for stage in args.stages:
        name = f"gb200_{args.world_size}gpu_zero{stage}_{args.attention}_s{args.seed}_{args.tag}"
        if args.smoke_steps:
            name += f"_smoke{args.smoke_steps}"
        if args.corrected:
            name += '_corrected'
        run = ROOT / "runs" / name
        run.mkdir(parents=True, exist_ok=False)
        state = {"name": name, "phase": "PREPARING", "start": time.time(), "world_size": args.world_size,
                 "stage": stage, "seed":args.seed, "corrected":args.corrected, "model": args.model, "forget_split": args.forget, "attention": args.attention, "smoke_only": bool(args.smoke_steps), "assets": assets,
                 "packages": {p: metadata.version(p) for p in ("torch", "transformers", "accelerate", "deepspeed", "flash-attn", "bitsandbytes")}}
        write_json(run / "status.json", state)
        def execute(command, phase):
            state.update(phase=phase, updated=time.time())
            write_json(run / "status.json", state)
            write_json(run / f"{phase.lower()}_command.json", command)
            with (run / f"{phase.lower()}.log").open("w") as log:
                rc = subprocess.call(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if rc:
                state.update(phase=f"FAILED_{phase}", returncode=rc, updated=time.time())
                write_json(run / "status.json", state)
                raise RuntimeError(f"{name}: {phase} failed ({rc}); see {run}")
        checkpoint = run / "checkpoint"
        accum = 8 // args.world_size
        launch = [sys.executable]
        if stage != "none":
            ds = json.loads((REPO / "configs/accelerate/zero_stage3_offload_config.json").read_text())
            ds["zero_optimization"]["stage"] = int(stage)
            # Explicit integers make mismatched HF / DeepSpeed batch sizes fail loudly.
            ds.update(train_batch_size=32, train_micro_batch_size_per_gpu=4, gradient_accumulation_steps=accum)
            write_json(run / "deepspeed.json", ds)
            accelerate = {"compute_environment": "LOCAL_MACHINE", "distributed_type": "DEEPSPEED",
                          "num_machines": 1, "num_processes": args.world_size, "machine_rank": 0,
                          "use_cpu": False, "main_training_function": "main",
                          "deepspeed_config": {"deepspeed_config_file": str(run / "deepspeed.json"), "zero3_init_flag": stage == "3"}}
            import yaml
            (run / "accelerate.yaml").write_text(yaml.safe_dump(accelerate))
            launch += ["-m", "accelerate.commands.launch", "--config_file", str(run / "accelerate.yaml"), "--main_process_port", "29541"]
        elif args.world_size != 1:
            raise ValueError("The non-DeepSpeed control is defined for one GPU only")
        probe = 'corrected_probe.py' if args.corrected else 'run_probe.py'
        train = launch + [f"scripts/reproduction/npo/gb200/{probe}", "--config-name=unlearn.yaml",
                         "experiment=unlearn/tofu/default.yaml", "trainer=NPO", f"task_name={name}",
                         f"model={args.model}", f"forget_split={args.forget}", f"retain_split={retain_split}", f"holdout_split=holdout{args.forget[-2:]}",
                         f"model.model_args.pretrained_model_name_or_path={model_path}",
                         f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
                         f"model.model_args.attn_implementation={args.attention}",
                         f"retain_logs_path={retain_path}", f"trainer.args.output_dir={checkpoint}",
                         "trainer.args.per_device_train_batch_size=4", f"trainer.args.gradient_accumulation_steps={accum}",
                         "trainer.args.ddp_find_unused_parameters=true", "trainer.args.gradient_checkpointing=true",
                         f"trainer.args.seed={args.seed}", "trainer.args.eval_on_start=false", "trainer.args.eval_strategy=no",
                         "trainer.args.do_eval=false",
                         f"+data.forget.TOFU_QA_forget.args.hf_args.revision={assets['dataset_revision']}",
                         f"+data.retain.TOFU_QA_retain.args.hf_args.revision={assets['dataset_revision']}"]
        if args.smoke_steps:
            train += [f"+trainer.args.max_steps={args.smoke_steps}"]
        execute(train, "TRAINING")
        state["trainer_state"] = json.loads((checkpoint / "trainer_state.json").read_text())
        if not args.smoke_steps:
            evaluation = [sys.executable, "src/eval.py", "--config-name=eval.yaml", "experiment=eval/tofu/default.yaml",
                          f"model={args.model}", f"forget_split={args.forget}", f"holdout_split=holdout{args.forget[-2:]}", f"task_name={name}_eval",
                          f"model.model_args.pretrained_model_name_or_path={checkpoint}",
                          f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
                          f"model.model_args.attn_implementation={args.attention}",
                          f"retain_logs_path={retain_path}", f"paths.output_dir={run / 'eval'}", "eval.tofu.batch_size=32"]
            execute(evaluation, "EVALUATING")
            state["summary"] = json.loads((run / "eval/TOFU_SUMMARY.json").read_text())
            execute([sys.executable, "scripts/reproduction/npo/as-run/finalize_h100_reproduction.py",
                     str(run / "eval/TOFU_SUMMARY.json"), str(run / "eval/TOFU_EVAL.json"), retain_path,
                     str(checkpoint / "trainer_state.json")], "AUDITING")
        state.update(phase="SMOKE_PASSED" if args.smoke_steps else "DONE", updated=time.time())
        write_json(run / "status.json", state)
        print(json.dumps(state, default=str), flush=True)


if __name__ == "__main__":
    main()
