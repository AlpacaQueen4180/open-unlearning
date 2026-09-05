"""Aggregate GPT-judge results and compare old and new judges."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .gpt_judge import successful_rows, summarize_rows, validate_judged


def category_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("category", "unknown")), []).append(row)
    return {category: summarize_rows(group) for category, group in sorted(groups.items())}


def confusion(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    old_by_id, new_by_id = {row["id"]: row for row in old_rows}, {row["id"]: row for row in new_rows}
    common = sorted(old_by_id.keys() & new_by_id.keys())
    pairs = [(str(old_by_id[key][field]), str(new_by_id[key][field])) for key in common]
    labels = sorted({label for pair in pairs for label in pair})
    table = {old: {new: sum(pair == (old, new) for pair in pairs) for new in labels} for old in labels}
    agreement = sum(old == new for old, new in pairs) / len(pairs) if pairs else None
    if not pairs:
        kappa = None
    else:
        old_counts, new_counts = Counter(old for old, _ in pairs), Counter(new for _, new in pairs)
        expected = sum(old_counts[label] * new_counts[label] for label in labels) / (len(pairs) ** 2)
        kappa = (agreement - expected) / (1 - expected) if expected != 1 else 1.0
    return {"paired": len(pairs), "agreement_rate": agreement, "cohen_kappa": kappa, "confusion_old_to_new": table}


def calibration(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "old_summary": summarize_rows(old_rows),
        "new_summary": summarize_rows(new_rows),
        "harmfulness_rate_delta_new_minus_old": summarize_rows(new_rows)["harmful_assistance_rate"] - summarize_rows(old_rows)["harmful_assistance_rate"],
        "safety": confusion(old_rows, new_rows, "safety_label"),
        "generation_quality": confusion(old_rows, new_rows, "generation_quality"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", required=True)
    compare = sub.add_parser("calibrate")
    compare.add_argument("--old", required=True)
    compare.add_argument("--new", required=True)
    compare.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--input", required=True)
    verify.add_argument("--judged", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        rows = successful_rows(Path(args.input))
        report = {"overall": summarize_rows(rows), "by_category": category_summaries(rows)}
    elif args.command == "calibrate":
        report = calibration(successful_rows(Path(args.old)), successful_rows(Path(args.new)))
    else:
        print(json.dumps(validate_judged(Path(args.input), Path(args.judged)), indent=2))
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
