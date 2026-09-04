"""Generate model responses for HEx-PHI with resumable JSONL output."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_id(prompt: str, category: Any = None) -> str:
    payload = json.dumps([category, prompt], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("HEx-PHI input must be a JSON array or JSONL records")
    prompts = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt", row.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"row {index} has no non-empty prompt/text string")
        category = row.get("category")
        prompts.append(
            {
                "prompt_id": str(row.get("id") or prompt_id(prompt, category)),
                "category": category,
                "prompt": prompt,
            }
        )
    ids = [row["prompt_id"] for row in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("HEx-PHI input contains duplicate prompt IDs")
    return prompts


def completed_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "success" and row.get("prompt_id"):
            completed.add(str(row["prompt_id"]))
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", choices=("legacy-300", "formal-330"), required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    dataset_sha256 = sha256_file(input_path)
    prompts = load_prompts(input_path)
    if args.limit:
        prompts = prompts[: args.limit]
    completed = completed_prompt_ids(output_path)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    do_sample = args.protocol == "legacy-300"
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.0,
    }
    if do_sample:
        generation_kwargs.update(temperature=1.0, top_p=1.0)

    model_revision = getattr(model.config, "_commit_hash", None)
    for item in prompts:
        if item["prompt_id"] in completed:
            continue
        messages = [{"role": "user", "content": item["prompt"]}]
        if tokenizer.chat_template is not None:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:
            inputs = tokenizer(item["prompt"], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        append_jsonl(
            output_path,
            {
                **item,
                "response": response,
                "status": "success",
                "run_id": args.run_id,
                "protocol": args.protocol,
                "dataset_sha256": dataset_sha256,
                "checkpoint_id": args.checkpoint_id,
                "checkpoint_sha256": args.checkpoint_sha256,
                "model_source": args.model,
                "model_revision": model_revision,
                "seed": args.seed,
                "generation": generation_kwargs,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    main()
