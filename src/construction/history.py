"""Audit existing Experiment 2 aggregates without changing historical outputs."""
import json
from pathlib import Path
import subprocess
import sys

from construction.protocol import sha256, write_json


def audit_history(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rebuilt = output / "joint_analysis.json"
    subprocess.run([sys.executable, str(repo / "scripts/reproduction/safety/analyze_llama31_8b_joint_results.py"),
                    "--repo", str(repo), "--json-output", str(rebuilt),
                    "--report-output", str(output / "joint_report.md")], check=True)
    historical = repo / "results/reproduction/safety/llama31_8b_completion_20260915/joint_analysis.json"
    expected, actual = json.loads(historical.read_text()), json.loads(rebuilt.read_text())
    for value in (expected, actual):
        value.pop("created_at", None)
    if actual != expected:
        raise ValueError("Rebuilt joint aggregates differ from published historical aggregates")
    observations = actual["representative_seed_observations"]
    seen = set()
    for row in observations:
        key = row["condition"], row["seed"]
        if key in seen or row["seed"] not in (0, 2, 4):
            raise ValueError("Invalid or duplicated paired capability seed")
        seen.add(key)
    inputs = sorted((repo / "results/reproduction/safety/llama31_8b_completion_20260915").glob("*.json"))
    inputs += sorted((repo / "results/reproduction/capability/llama31_8b_completion_20260915").glob("*.json"))
    audit = {"status": "pass", "historical_aggregates_reproduced": True,
             "paired_seed_policy": [0, 2, 4], "paired_observations": len(observations),
             "protocol": actual["protocol"],
             "inputs": [{"path": p.relative_to(repo).as_posix(), "sha256": sha256(p)} for p in inputs],
             "judge_audit": {"level": "aggregate_provenance_only", "raw_row_rejudging": "not_performed",
                             "note": "Successful-row rates and unknown-label bounds retain existing definitions; no new judge calls."}}
    write_json(output / "audit.json", audit)
    return audit
