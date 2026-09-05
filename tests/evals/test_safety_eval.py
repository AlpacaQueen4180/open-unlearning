import json

import pytest
from pydantic import ValidationError

from evals.safety.generate import completed_prompt_ids, load_prompts, prompt_id
from evals.safety.gpt_judge import JudgeResult, load_completed, stable_id, summarize_rows, successful_rows, validate_judged
from evals.safety.report import calibration, category_summaries, summary_with_coverage


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


def test_policy_blocked_is_terminal_for_resume(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"id": "blocked", "status": "policy_blocked", "policy_code": "bio_policy"}) + "\n", encoding="utf-8")
    assert load_completed(path) == {"blocked"}


def test_validation_allows_error_history_after_successful_retry(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n", encoding="utf-8")
    row_id = stable_id("p", "r")
    judged = tmp_path / "judged.jsonl"
    success = {
        "id": row_id, "status": "success", "judge_model_actual": "gpt-5.6-terra",
        "safety_label": "refusal", "generation_quality": "normal",
        "safety_confidence": .9, "generation_confidence": .9,
        "safety_reason": "refused", "generation_reason": "coherent",
    }
    judged.write_text("\n".join((json.dumps({"id": row_id, "status": "error"}), json.dumps(success))) + "\n", encoding="utf-8")
    assert validate_judged(source, judged)["successful_unique"] == 1
    assert len(successful_rows(judged)) == 1


def test_summary_rejects_unresolved_errors(tmp_path):
    judged = tmp_path / "judged.jsonl"
    judged.write_text(json.dumps({"id": "x", "status": "error"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved"):
        successful_rows(judged)


def test_summary_reports_policy_coverage_and_harmfulness_bounds():
    rows = [
        {"id": "ok", "status": "success", "category": 1, "safety_label": "harmful_assistance", "generation_quality": "normal"},
        {"id": "blocked", "status": "policy_blocked", "category": 1},
    ]
    report = summary_with_coverage(rows)
    assert report["judge_coverage"] == {"total": 2, "successful": 1, "policy_blocked": 1, "successful_rate": .5}
    assert report["harmful_assistance_rate_bounds_all_inputs"] == {"lower": .5, "upper": 1.0}


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
