"""Generate resumable two-turn MT-Bench answers without calling a judge API."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORY_TEMPERATURES = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for index, row in enumerate(rows):
        question_id = str(row.get("question_id", ""))
        turns = row.get("turns")
        if not question_id or question_id in seen:
            raise ValueError(f"row {index} has a missing or duplicate question_id")
        if not isinstance(turns, list) or len(turns) != 2 or not all(
            isinstance(turn, str) and turn.strip() for turn in turns
        ):
            raise ValueError(f"question {question_id} must contain two non-empty turns")
        seen.add(question_id)
    return rows


def successful_turns(path: Path) -> dict[tuple[str, int], str]:
    completed: dict[tuple[str, int], str] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "success":
            continue
        key = (str(row.get("question_id")), int(row.get("turn_index", -1)))
        response = row.get("response")
        if key[0] and key[1] in (0, 1) and isinstance(response, str):
            completed[key] = response
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--question-revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    completed = successful_turns(args.output)
    dataset_sha256 = sha256_file(args.questions)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_revision = getattr(model.config, "_commit_hash", None)
    for question in questions:
        question_id = str(question["question_id"])
        category = str(question.get("category", "")).lower()
        temperature = CATEGORY_TEMPERATURES.get(category, 0.7)
        messages: list[dict[str, str]] = []
        for turn_index, prompt in enumerate(question["turns"]):
            messages.append({"role": "user", "content": prompt})
            existing = completed.get((question_id, turn_index))
            if existing is not None:
                messages.append({"role": "assistant", "content": existing})
                continue

            if tokenizer.chat_template is not None:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                transcript = "\n".join(
                    f"{message['role']}: {message['content']}" for message in messages
                ) + "\nassistant:"
                inputs = tokenizer(transcript, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            generation: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation.update(temperature=temperature, top_p=1.0)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation)
            response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            append_jsonl(
                args.output,
                {
                    "question_id": question_id,
                    "category": question.get("category"),
                    "turn_index": turn_index,
                    "prompt": prompt,
                    "response": response,
                    "status": "success",
                    "judge_status": "pending_judge",
                    "run_id": args.run_id,
                    "checkpoint_id": args.checkpoint_id,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "model_source": args.model,
                    "model_revision": model_revision,
                    "question_revision": args.question_revision,
                    "dataset_sha256": dataset_sha256,
                    "seed": args.seed,
                    "generation": generation,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            completed[(question_id, turn_index)] = response
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
