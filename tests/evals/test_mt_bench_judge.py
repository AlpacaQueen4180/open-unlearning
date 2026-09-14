import json

from evals.mt_bench_judge import build_examples, stable_id


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_stable_id_separates_turns():
    assert stable_id("m", "101", 1) != stable_id("m", "101", 2)


def test_build_examples_routes_reference_categories(tmp_path):
    questions = tmp_path / "questions.jsonl"
    refs = tmp_path / "refs.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    answers = tmp_path / "answers.jsonl"
    write_jsonl(questions, [{"question_id": i, "category": "math" if i == 0 else "writing", "turns": ["q1", "q2"]} for i in range(80)])
    write_jsonl(refs, [{"question_id": i, "choices": [{"turns": ["r1", "r2"]}]} for i in range(80)])
    write_jsonl(prompts, [
        {"name": "single-v1", "system_prompt": "s", "prompt_template": "{question} {answer}"},
        {"name": "single-math-v1", "system_prompt": "s", "prompt_template": "{question} {answer} {ref_answer_1}"},
        {"name": "single-v1-multi-turn", "system_prompt": "s", "prompt_template": "{question_1} {answer_1} {question_2} {answer_2}"},
        {"name": "single-math-v1-multi-turn", "system_prompt": "s", "prompt_template": "{question_1} {answer_1} {question_2} {answer_2} {ref_answer_1} {ref_answer_2}"},
    ])
    answer_rows = []
    for i in range(80):
        for turn in (0, 1):
            answer_rows.append({"question_id": i, "turn_index": turn, "status": "success", "response": "a", "checkpoint_id": "model"})
    write_jsonl(answers, answer_rows)
    examples = build_examples([answers], questions, refs, prompts)
    assert len(examples) == 160
    assert examples[0]["prompt_name"] == "single-math-v1"
    assert examples[1]["prompt_name"] == "single-math-v1-multi-turn"
