import json
from pathlib import Path

import pytest

from evals.mt_bench_generate import load_questions, successful_turns


ROOT = Path(__file__).resolve().parents[2]


def test_capability_registry_is_fixed_at_31_models():
    path = ROOT / "configs/experiment/safety/llama31_8b_capability_31_nojudge.json"
    models = json.loads(path.read_text())["models"]
    assert len(models) == 31
    assert len({model["id"] for model in models}) == 31
    assert {model["source"] for model in models} == {"hf", "h100", "gb10"}


def test_nojudge_queues_have_no_judge_command():
    paths = [
        ROOT / "scripts/reproduction/safety/run_llama31_8b_public_f05_seeds1_4_nojudge_queue.sh",
        ROOT / "scripts/reproduction/safety/run_llama31_8b_nojudge_baselines.sh",
        ROOT / "scripts/reproduction/safety/run_llama31_8b_nojudge_completion_queue.sh",
    ]
    for path in paths:
        text = path.read_text()
        assert "evals.safety judge" not in text
        assert "ENABLE_API_JUDGE" in text


def test_mt_bench_questions_require_exactly_two_turns(tmp_path):
    valid = tmp_path / "valid.jsonl"
    valid.write_text(json.dumps({"question_id": 1, "category": "writing", "turns": ["a", "b"]}) + "\n")
    assert load_questions(valid)[0]["question_id"] == 1
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps({"question_id": 1, "turns": ["a"]}) + "\n")
    with pytest.raises(ValueError, match="two non-empty turns"):
        load_questions(invalid)


def test_mt_bench_resume_uses_only_success_rows(tmp_path):
    path = tmp_path / "answers.jsonl"
    rows = [
        {"question_id": 1, "turn_index": 0, "status": "success", "response": "first"},
        {"question_id": 1, "turn_index": 1, "status": "error", "response": "ignored"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert successful_turns(path) == {("1", 0): "first"}
