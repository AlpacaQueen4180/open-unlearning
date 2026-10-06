import copy
import json

import pytest

from construction.protocol import (ARCHIVE_SHA256, REQUIRED_ROLES, THRESHOLDS, audit_sources,
                                   read_anchor, sha256, target_gate, validate_manifest)


def anchor_file(tmp_path):
    path = tmp_path / "anchor.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "a unique anchor"},
                                            {"role": "assistant", "content": "A safe answer."}]}))
    return path


def test_checksum_mismatch(tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        read_anchor(anchor_file(tmp_path), "wrong")


def test_audit_missing_roles_and_normalized_collision(tmp_path):
    anchor = anchor_file(tmp_path)
    assert audit_sources(anchor, sha256(anchor), [])['status'] == 'insufficient_evidence'
    source = tmp_path / "test.json"
    source.write_text(json.dumps([{"prompt": " A UNIQUE   anchor "}]))
    audit = audit_sources(anchor, sha256(anchor), [{"role": "formal_hexphi", "path": str(source),
                         "sha256": sha256(source), "source": "fixture", "revision": "fixture-v1"}])
    assert audit["status"] == "fail"
    assert audit["collisions"][0]["count"] == 1


def test_complete_disjoint_audit(tmp_path):
    anchor = anchor_file(tmp_path)
    specs = []
    for role in REQUIRED_ROLES:
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps([{"question": f"unique {role}"}]))
        specs.append({"role": role, "source": "fixture", "revision": "v1", "path": str(path),
                      "sha256": sha256(path)})
    assert audit_sources(anchor, sha256(anchor), specs)["status"] == "pass"
    path = tmp_path / "safety_development.json"
    path.write_text((tmp_path / "formal_hexphi.json").read_text())
    for spec in specs:
        spec["sha256"] = sha256(spec["path"])
    assert audit_sources(anchor, sha256(anchor), specs)["status"] == "fail"


def evidence():
    return {"selection_source": "development", "target_id": "fixture", "overlap_audit_status": "pass",
            "baseline_audit": {"thresholds_frozen": True, "sha256": "a" * 64},
            "metrics": {k: 0.0 for k in THRESHOLDS}, "learning_comparable": True,
            "benign_overrefusal_acceptable": True,
            "learning": {s: {m: {"minimum": .1, "improvement_lower": .2}
                              for m in ("likelihood", "rouge", "extraction")}
                         for s in ("forget", "retain")}}


def test_gate_missing_not_equivalent_to_pass():
    item = evidence()
    assert target_gate(item)["status"] == "pass"
    del item["learning"]["forget"]["likelihood"]["minimum"]
    assert target_gate(item)["status"] == "insufficient_evidence"


def test_gate_requires_baseline_audit_even_with_positive_learning():
    item = evidence()
    del item["baseline_audit"]
    assert target_gate(item)["status"] == "insufficient_evidence"


def test_development_evidence_template_is_insufficient():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "configs/construction/development_evidence.example.json"
    assert target_gate(json.loads(path.read_text()))["status"] == "insufficient_evidence"


@pytest.mark.parametrize("metric", list(THRESHOLDS))
def test_threshold_exceedance_fails(metric):
    item = evidence()
    item["metrics"][metric] = THRESHOLDS[metric] + .01
    assert target_gate(item)["status"] == "fail"


def test_formal_selection_and_wrong_target_rejected():
    item = evidence()
    item["selection_source"] = "formal_hexphi"
    assert target_gate(item)["status"] == "fail"
    assert target_gate(evidence(), expected_target="other")["status"] == "fail"


def test_gate_distinguishes_failed_overlap_from_missing_audit():
    item = evidence()
    item["overlap_audit_status"] = "fail"
    assert target_gate(item)["status"] == "fail"
    item["overlap_audit_status"] = "insufficient_evidence"
    assert target_gate(item)["status"] == "insufficient_evidence"


def test_manifest_full_requires_audit_and_pinned_revisions():
    item = {"schema_version": 1, "archive_sha256": ARCHIVE_SHA256, "selection_source": "development",
            **{k: "a" * 40 for k in ("repository_commit", "model_revision", "tokenizer_revision", "tofu_revision")}}
    validate_manifest(item)
    with pytest.raises(ValueError):
        validate_manifest(item, full=True)
    item["model_revision"] = "main"
    with pytest.raises(ValueError):
        validate_manifest(item)


def test_freeze_requires_complete_same_manifest_same_profile_and_reload():
    from construction.protocol import freeze_manifest
    manifest = {"schema_version": 1, "archive_sha256": ARCHIVE_SHA256, "selection_source": "development",
                "audit": {"status": "pass"},
                **{k: "a" * 40 for k in ("repository_commit", "model_revision", "tokenizer_revision", "tofu_revision")}}
    smoke = {"status": "pass", "manifest_sha256": "prepared", "chosen_microbatch": 4,
             "chosen_accumulation": 4,
             "attempts": [{"method": m, "microbatch": 4, "status": "pass",
                           "target": {"metrics": {"step": 10}, "reload": {"status": "pass"}}}
                          for m in ("standard", "mixing", "spf")]}
    acceptance = {"backend": "zero3", "method": "spf", "reference_comparison": "pass",
                  "steps": 3, "scheduler_steps": 3, "serialized_reload": "pass", "gradient_comparison": "pass",
                  "conflicting_updates": 1}
    kwargs = dict(prepared_hash="prepared", smoke_hash="smoke", acceptance_hash="acceptance")
    result = freeze_manifest(manifest, smoke, acceptance, **kwargs)
    validate_manifest(result, full=True)
    smoke["attempts"][0]["microbatch"] = 2
    with pytest.raises(ValueError, match="All three"):
        freeze_manifest(manifest, smoke, acceptance, **kwargs)
    smoke["manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="exact prepared"):
        freeze_manifest(manifest, smoke, acceptance, **kwargs)
