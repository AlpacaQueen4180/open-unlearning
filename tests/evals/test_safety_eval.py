import json

import pytest
from pydantic import ValidationError

from evals.safety.generate import completed_prompt_ids, load_prompts, prompt_id
from evals.safety.gpt_judge import JudgeResult, load_completed, stable_id, summarize_rows
from evals.safety.report import calibration, category_summaries


def test_judge_schema_and_confidence_bounds():
    valid = JudgeResult(safety_label="refusal", generation_quality="normal", safety_confidence=.9, generation_confidence=.8, safety_reason="refuses", generation_reason="coherent")
    assert valid.safety_label == "refusal"
    with pytest.raises(ValidationError):
        JudgeResult(safety_label="refusal", generation_quality="normal", safety_confidence=1.1, generation_confidence=.8, safety_reason="x", generation_reason="x")


def test_stable_ids_are_content_stable():
    assert stable_id("p", "r") == stable_id("p", "r")
    assert stable_id("p", "r") != stable_id("p", "different")
    assert prompt_id("p", 1) != prompt_id("p", 2)


def test_load_prompts_preserves_category(tmp_path):
    path = tmp_path / "hex.json"
    path.write_text(json.dumps([{"category": 7, "text": "prompt"}]), encoding="utf-8")
    rows = load_prompts(path)
    assert rows[0]["category"] == 7
    assert rows[0]["prompt"] == "prompt"


def test_resume_only_skips_success(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('\n'.join([json.dumps({"id": "ok", "prompt_id": "p1", "status": "success"}), json.dumps({"id": "retry", "prompt_id": "p2", "status": "error"}), "{broken"]) + "\n", encoding="utf-8")
    assert load_completed(path) == {"ok"}
    assert completed_prompt_ids(path) == {"p1"}


def test_summary_and_categories_are_two_dimensional():
    rows = [
        {"category": 1, "safety_label": "harmful_assistance", "generation_quality": "normal"},
        {"category": 1, "safety_label": "refusal", "generation_quality": "degenerate"},
    ]
    summary = summarize_rows(rows)
    assert summary["harmful_assistance_rate"] == .5
    assert summary["degeneration_rate"] == .5
    assert category_summaries(rows)["1"]["total_evaluated"] == 2


def test_calibration_reports_agreement_and_kappa():
    old = [{"id": "1", "safety_label": "refusal", "generation_quality": "normal"}, {"id": "2", "safety_label": "harmful_assistance", "generation_quality": "normal"}]
    new = [{"id": "1", "safety_label": "refusal", "generation_quality": "normal"}, {"id": "2", "safety_label": "safe_non_refusal", "generation_quality": "degenerate"}]
    report = calibration(old, new)
    assert report["safety"]["paired"] == 2
    assert report["safety"]["agreement_rate"] == .5
    assert report["generation_quality"]["agreement_rate"] == .5
