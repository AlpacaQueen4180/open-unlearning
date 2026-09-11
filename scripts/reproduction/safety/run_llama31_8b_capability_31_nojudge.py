#!/usr/bin/env python3
"""Run MMLU and generate unjudged MT-Bench answers for a fixed model registry."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


FASTCHAT_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"
MT_BENCH_SHA256 = "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7"
GB10 = "gb10@140.113.13.133"
GB10_PORT = "22026"
GB10_KEY = "/home/ai/.ssh/open_unlearning_to_gb10"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def validate_registry(registry: dict[str, Any]) -> list[dict[str, str]]:
    models = registry.get("models")
    if not isinstance(models, list) or len(models) != 31:
        raise ValueError(f"expected exactly 31 registry models, found {len(models or [])}")
    ids = [str(model.get("id")) for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("registry model IDs must be unique")
    for model in models:
        if model.get("source") not in {"hf", "h100", "gb10"} or not model.get("path"):
            raise ValueError(f"invalid registry entry: {model}")
    return models


def mmlu_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    subject_acc = {key: value for key, value in data.items() if key.startswith("mmlu_") and key.endswith("/acc")}
    if "mmlu/acc" not in data or len(subject_acc) != 57:
        raise ValueError(f"incomplete MMLU summary: group={data.get('mmlu/acc')}, subjects={len(subject_acc)}")
    return {"accuracy": data["mmlu/acc"], "accuracy_stderr": data.get("mmlu/acc_stderr"), "subjects": subject_acc}


def mt_bench_count(path: Path) -> int:
    if not path.is_file():
        return 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return len({(str(row.get("question_id")), row.get("turn_index")) for row in rows if row.get("status") == "success"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/ai/alpaca"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--queue-id", default="llama31_8b_capability_31_nojudge_20260911")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("ENABLE_API_JUDGE", "0") != "0":
        raise SystemExit("ENABLE_API_JUDGE must be 0")
    for key in list(os.environ):
        if key.startswith("OPENAI_") or key.startswith("AZURE_OPENAI_"):
            os.environ.pop(key, None)

    registry = json.loads(args.registry.read_text())
    models = validate_registry(registry)
    if args.dry_run:
        print(json.dumps({"api_judge_enabled": False, "count": len(models), "ids": [m["id"] for m in models]}, indent=2))
        return
    question_digest = __import__("hashlib").sha256(args.questions.read_bytes()).hexdigest()
    if question_digest != MT_BENCH_SHA256:
        raise SystemExit(f"MT-Bench checksum mismatch: {question_digest}")
    if args.workers != 2:
        raise SystemExit("this experiment is fixed to two one-GPU workers")
    if importlib.metadata.version("lm_eval") != "0.4.11":
        raise SystemExit(f"expected lm_eval 0.4.11, found {importlib.metadata.version('lm_eval')}")
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if len(gpu_query) != 2:
        raise SystemExit(f"expected exactly two visible H100 GPUs, found {len(gpu_query)}")
    for model in models:
        if model["source"] == "h100" and not Path(model["path"]).is_dir():
            raise SystemExit(f"missing H100 checkpoint: {model['path']}")
        if model["source"] == "gb10":
            check = subprocess.run(
                ["ssh", "-i", GB10_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 "-p", GB10_PORT, GB10, "test", "-d", model["path"]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if check.returncode:
                raise SystemExit(f"missing GB10 checkpoint: {model['path']}")

    root = args.root.resolve()
    env_dir = Path("/home/ai/miniforge3/envs/open-unlearning-repro")
    python = str(env_dir / "bin/python")
    artifacts = root / "safety_artifacts" / args.queue_id
    public = root / "results/reproduction/capability" / args.queue_id
    staging_root = root / "capability_staging" / args.queue_id
    status_path = artifacts / "queue.status.json"
    per_model = public / "models"
    for path in (artifacts / "logs", artifacts / "mt_bench_raw", public, per_model, staging_root):
        path.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for model in models:
        cached = per_model / f"{model['id']}.json"
        if cached.is_file():
            results[model["id"]] = json.loads(cached.read_text())

    def update_status(state: str, detail: str) -> None:
        with lock:
            atomic_json(status_path, {
                "status": state, "detail": detail, "completed": len(results),
                "failed": len(errors), "total": len(models), "api_judge_enabled": False,
                "updated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            })

    base_env = dict(os.environ)
    base_env.update({"PYTHONPATH": str(root / "src"), "TOKENIZERS_PARALLELISM": "false", "HYDRA_FULL_ERROR": "1"})

    def evaluate(model: dict[str, str], gpu: int) -> None:
        model_id, source, source_path = model["id"], model["source"], model["path"]
        stage = staging_root / f"gpu{gpu}" / model_id
        model_path = source_path
        update_status("RUNNING", f"{model_id}:resolve")
        try:
            if source == "h100":
                if not Path(source_path).is_dir():
                    raise FileNotFoundError(source_path)
            elif source == "gb10":
                if stage.exists():
                    shutil.rmtree(stage)
                stage.mkdir(parents=True)
                rsync_ssh = f"ssh -i {GB10_KEY} -o BatchMode=yes -o ConnectTimeout=15 -p {GB10_PORT}"
                command = ["rsync", "-a", "--partial", "-e", rsync_ssh, f"{GB10}:{source_path}/", f"{stage}/"]
                run(command, cwd=root, env=base_env, log=artifacts / "logs" / f"{model_id}.stage.log")
                model_path = str(stage)

            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            mmlu_out = root / "saves/eval" / args.queue_id / model_id
            update_status("RUNNING", f"{model_id}:mmlu")
            mmlu_path = mmlu_out / "LMEval_SUMMARY.json"
            if not mmlu_path.is_file():
                run([
                    python, "src/eval.py", "eval=lm_eval_mmlu_5shot", f"task_name={args.queue_id}_{model_id}",
                    "model=Llama-3.1-8B-Instruct", f"model.model_args.pretrained_model_name_or_path={model_path}",
                    f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
                    "model.model_args.attn_implementation=flash_attention_2", f"paths.output_dir={mmlu_out}",
                ], cwd=root, env=env, log=artifacts / "logs" / f"{model_id}.mmlu.log")
            mmlu = mmlu_summary(mmlu_path)

            mt_output = artifacts / "mt_bench_raw" / f"{model_id}.jsonl"
            update_status("RUNNING", f"{model_id}:mt_bench_generation")
            generated = mt_bench_count(mt_output)
            if generated != 160:
                run([
                    python, "-m", "evals.mt_bench_generate", "--model", model_path,
                    "--questions", str(args.questions), "--output", str(mt_output),
                    "--run-id", f"{args.queue_id}_{model_id}", "--checkpoint-id", model_id,
                    "--question-revision", FASTCHAT_COMMIT, "--seed", "0", "--max-new-tokens", "1024",
                ], cwd=root, env=env, log=artifacts / "logs" / f"{model_id}.mt_bench.log")
                generated = mt_bench_count(mt_output)
            if generated != 160:
                raise ValueError(f"expected 160 MT-Bench turns, found {generated}")
            model_result = {
                "source": source, "mmlu": mmlu,
                "mt_bench": {"successful_turns": generated, "judge_status": "pending_judge"},
            }
            atomic_json(per_model / f"{model_id}.json", model_result)
            with lock:
                results[model_id] = model_result
        except Exception as exc:
            with lock:
                errors[model_id] = str(exc)
            raise
        finally:
            if source == "gb10" and stage.exists():
                resolved_stage = stage.resolve()
                if staging_root.resolve() in resolved_stage.parents:
                    shutil.rmtree(resolved_stage)
            update_status("RUNNING", f"{model_id}:finished")

    update_status("RUNNING", "preflight_complete")
    worker_errors: list[Exception] = []
    remaining = [model for model in models if model["id"] not in results]

    def worker(gpu: int, assigned: list[dict[str, str]]) -> None:
        for model in assigned:
            try:
                evaluate(model, gpu)
            except Exception as exc:
                with lock:
                    worker_errors.append(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        partitions = [[model for index, model in enumerate(remaining) if index % args.workers == gpu] for gpu in range(args.workers)]
        futures = [executor.submit(worker, gpu, assigned) for gpu, assigned in enumerate(partitions)]
        for future in futures:
            future.result()

    payload = {
        "schema_version": "llama31-8b-capability-31-nojudge.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "api_judge_enabled": False,
        "protocol": {
            "mmlu": {"task": "mmlu", "subjects": 57, "num_fewshot": 5, "apply_chat_template": False, "seeds": 0},
            "mt_bench": {"fastchat_commit": FASTCHAT_COMMIT, "question_sha256": MT_BENCH_SHA256, "turns": 160, "judge_status": "pending_judge"},
        },
        "environment": {
            "lm_eval": importlib.metadata.version("lm_eval"),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
        },
        "models": dict(sorted(results.items())), "errors": dict(sorted(errors.items())),
    }
    atomic_json(public / "aggregate.json", payload)
    if worker_errors or len(results) != 31:
        update_status("FAILED", f"completed={len(results)} failed={len(errors)}")
        raise SystemExit(1)
    update_status("DONE", str(public / "aggregate.json"))


if __name__ == "__main__":
    main()
