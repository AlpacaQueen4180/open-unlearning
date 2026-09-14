"""Resumable structured-output single-answer judge for pinned MT-Bench artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evals.safety.gpt_judge import is_policy_code, load_env_file


NEED_REF_CATEGORIES = {"math", "reasoning", "coding", "arena-hard-200"}
TERMINAL_STATUSES = {"success", "policy_blocked"}


class MTBenchJudgeResult(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = Field(min_length=1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_judge_prompts(path: Path) -> dict[str, dict[str, Any]]:
    prompts = {row["name"]: row for row in load_jsonl(path)}
    required = {"single-v1", "single-math-v1", "single-v1-multi-turn", "single-math-v1-multi-turn"}
    missing = sorted(required - prompts.keys())
    if missing:
        raise ValueError(f"missing judge prompts: {missing}")
    return prompts


def load_model_answers(path: Path) -> tuple[str, dict[str, list[str]]]:
    rows = load_jsonl(path)
    by_question: dict[str, dict[int, str]] = defaultdict(dict)
    checkpoint_ids = set()
    for row in rows:
        if row.get("status") != "success":
            continue
        question_id = str(row["question_id"])
        turn = int(row["turn_index"])
        by_question[question_id][turn] = row["response"]
        checkpoint_ids.add(str(row["checkpoint_id"]))
    if len(checkpoint_ids) != 1:
        raise ValueError(f"expected one checkpoint ID in {path}, found {checkpoint_ids}")
    answers = {}
    for question_id, turns in by_question.items():
        if set(turns) != {0, 1}:
            raise ValueError(f"question {question_id} is missing a turn in {path}")
        answers[question_id] = [turns[0], turns[1]]
    if len(answers) != 80:
        raise ValueError(f"expected 80 complete questions in {path}, found {len(answers)}")
    return checkpoint_ids.pop(), answers


def load_references(path: Path) -> dict[str, list[str]]:
    refs = {}
    for row in load_jsonl(path):
        turns = row["choices"][0]["turns"]
        refs[str(row["question_id"])] = turns
    return refs


def stable_id(model_id: str, question_id: str, turn: int) -> str:
    raw = json.dumps([model_id, question_id, turn], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def build_examples(
    answer_files: list[Path], questions_path: Path, refs_path: Path, prompts_path: Path
) -> list[dict[str, Any]]:
    questions = {str(row["question_id"]): row for row in load_jsonl(questions_path)}
    if len(questions) != 80:
        raise ValueError(f"expected 80 questions, found {len(questions)}")
    references = load_references(refs_path)
    prompts = load_judge_prompts(prompts_path)
    examples = []
    for answer_file in sorted(answer_files):
        model_id, answers = load_model_answers(answer_file)
        for question_id, question in questions.items():
            answer = answers[question_id]
            category = str(question["category"])
            needs_ref = category in NEED_REF_CATEGORIES
            refs = references.get(question_id)
            if needs_ref and (not refs or len(refs) != 2):
                raise ValueError(f"missing reference for {question_id} ({category})")

            prompt_name = "single-math-v1" if needs_ref else "single-v1"
            values = {"question": question["turns"][0], "answer": answer[0]}
            if needs_ref:
                values["ref_answer_1"] = refs[0]
            template = prompts[prompt_name]
            examples.append({
                "id": stable_id(model_id, question_id, 1), "model_id": model_id,
                "question_id": question_id, "turn": 1, "category": category,
                "prompt_name": prompt_name, "instructions": template["system_prompt"],
                "input": template["prompt_template"].format(**values),
            })

            prompt_name = "single-math-v1-multi-turn" if needs_ref else "single-v1-multi-turn"
            values = {
                "question_1": question["turns"][0], "question_2": question["turns"][1],
                "answer_1": answer[0], "answer_2": answer[1],
            }
            if needs_ref:
                values.update(ref_answer_1=refs[0], ref_answer_2=refs[1])
            template = prompts[prompt_name]
            examples.append({
                "id": stable_id(model_id, question_id, 2), "model_id": model_id,
                "question_id": question_id, "turn": 2, "category": category,
                "prompt_name": prompt_name, "instructions": template["system_prompt"],
                "input": template["prompt_template"].format(**values),
            })
    return examples


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest = {}
    for row in load_jsonl(path):
        latest[str(row["id"])] = row
    return latest


async def evaluate_one(example: dict[str, Any], client: Any, args: argparse.Namespace) -> dict[str, Any]:
    error = "unknown error"
    for attempt in range(args.retries + 1):
        try:
            response = await client.responses.parse(
                model=args.model,
                instructions=example["instructions"],
                input=example["input"],
                text_format=MTBenchJudgeResult,
                reasoning={"effort": args.reasoning_effort},
                timeout=args.timeout,
            )
            if response.output_parsed is None:
                raise ValueError(f"no parsed output (status={response.status})")
            return {
                **response.output_parsed.model_dump(), "status": "success",
                "attempts": attempt + 1, "judge_model_actual": str(response.model),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            body = getattr(exc, "body", None)
            code = getattr(exc, "code", None)
            if not code and isinstance(body, dict):
                code = body.get("code") or body.get("error", {}).get("code")
            if is_policy_code(code):
                return {"status": "policy_blocked", "policy_code": code, "error": error, "attempts": attempt + 1, "judge_model_actual": None}
            if attempt < args.retries:
                await asyncio.sleep(min(30.0, (2**attempt) + random.random()))
    return {"status": "error", "error": error, "attempts": args.retries + 1, "judge_model_actual": None}


async def run_judge(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    load_env_file(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    answer_files = sorted(args.answers_dir.glob("*.jsonl"))
    if len(answer_files) != args.expected_models:
        raise ValueError(f"expected {args.expected_models} answer files, found {len(answer_files)}")
    examples = build_examples(answer_files, args.questions, args.references, args.judge_prompts)
    expected = args.expected_models * 160
    if len(examples) != expected:
        raise ValueError(f"expected {expected} judgments, built {len(examples)}")
    latest = latest_rows(args.output)
    completed = {key for key, row in latest.items() if row.get("status") in TERMINAL_STATUSES}
    pending = [row for row in examples if row["id"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(max_retries=0)
    semaphore, write_lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def worker(example: dict[str, Any]) -> None:
        async with semaphore:
            judged = await evaluate_one(example, client, args)
        row = {
            **{key: value for key, value in example.items() if key not in {"instructions", "input"}},
            **judged, "judge_model_requested": args.model,
            "judge_reasoning_effort": args.reasoning_effort,
            "fastchat_commit": args.fastchat_commit,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        async with write_lock:
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()

    await asyncio.gather(*(worker(example) for example in pending))

    final = latest_rows(args.output)
    missing = {row["id"] for row in examples} - {key for key, row in final.items() if row.get("status") in TERMINAL_STATUSES}
    unresolved = [key for key, row in final.items() if row.get("status") not in TERMINAL_STATUSES]
    if missing or unresolved:
        raise RuntimeError(f"incomplete judge: missing={len(missing)} unresolved={len(unresolved)}")


def summarize(judged_path: Path, output: Path, expected_models: int) -> None:
    rows = list(latest_rows(judged_path).values())
    models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        models[row["model_id"]].append(row)
    if len(models) != expected_models:
        raise ValueError(f"expected {expected_models} models, found {len(models)}")

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [row for row in items if row.get("status") == "success"]
        blocked = [row for row in items if row.get("status") == "policy_blocked"]
        return {
            "expected": len(items), "successful": len(successful), "policy_blocked": len(blocked),
            "mean_score": sum(row["score"] for row in successful) / len(successful) if successful else None,
        }

    summaries = {}
    for model_id, items in sorted(models.items()):
        by_category = {category: stats([row for row in items if row["category"] == category]) for category in sorted({row["category"] for row in items})}
        summaries[model_id] = {
            "overall": stats(items),
            "turn_1": stats([row for row in items if row["turn"] == 1]),
            "turn_2": stats([row for row in items if row["turn"] == 2]),
            "by_category": by_category,
        }
    payload = {
        "schema_version": "llama31-8b-mtbench-terra.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge": {"model": "gpt-5.6-terra", "reasoning_effort": "medium", "mode": "single", "structured_output": True},
        "models": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--judge-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--expected-models", type=int, default=31)
    parser.add_argument("--fastchat-commit", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        asyncio.run(run_judge(args))
    summarize(args.output, args.summary, args.expected_models)


if __name__ == "__main__":
    main()
