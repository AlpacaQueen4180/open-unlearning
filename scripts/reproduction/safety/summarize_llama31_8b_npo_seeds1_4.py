#!/usr/bin/env python3
"""Create a public-safe aggregate for the Llama-3.1-8B NPO seeds 1-4 queue."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import ks_2samp


TOFU_KEYS = (
    "forget_quality",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "model_utility",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def tofu_metrics(data: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = {key: data[key]["agg_value"] for key in TOFU_KEYS}
    pvalue = result["forget_quality"]
    result["forget_quality_log10"] = math.log10(pvalue) if pvalue > 0 else None
    left = [item["score"] for item in data["forget_truth_ratio"]["value_by_index"].values()]
    right = [item["score"] for item in reference["forget_truth_ratio"]["value_by_index"].values()]
    result["forget_quality_ks_statistic"] = float(ks_2samp(left, right).statistic)
    return result


def safety_metrics(data: dict[str, Any]) -> dict[str, Any]:
    overall = data["overall"]
    behavior = overall["safety_behavior"]
    quality = overall["generation_quality"]
    coverage = overall["judge_coverage"]
    blocked = coverage["policy_blocked"]
    errors = coverage["total"] - coverage["successful"] - blocked
    return {
        "judge_total": coverage["total"],
        "judge_successful": coverage["successful"],
        "judge_errors": errors,
        "policy_blocked": blocked,
        "judge_successful_rate": coverage["successful_rate"],
        "harmful_assistance_rate": behavior["harmful_assistance"]["rate"],
        "refusal_rate": behavior["refusal"]["rate"],
        "safe_non_refusal_rate": behavior["safe_non_refusal"]["rate"],
        "indeterminate_rate": behavior["indeterminate"]["rate"],
        "degeneration_rate": quality["degenerate"]["rate"],
        "harmful_among_normal_rate": overall["p_harmful_assistance_given_normal"],
        "harmful_lower_bound": overall["harmful_assistance_rate_bounds_all_inputs"]["lower"],
        "harmful_upper_bound": overall["harmful_assistance_rate_bounds_all_inputs"]["upper"],
    }


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 24:
        raise SystemExit(f"expected 24 completed cells, found {len(rows)}")

    cells: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tofu = load_json(row["tofu_eval"])
        reference = load_json(row["retain_reference"])
        missing = [key for key in TOFU_KEYS if key not in tofu]
        if missing:
            raise SystemExit(f"partial TOFU result for {row['name']}: {missing}")
        safety_data = load_json(row["safety_summary"])
        safety = safety_metrics(safety_data)
        condition = row["name"].rsplit("_seed", 1)[0]
        tofu_summary = tofu_metrics(tofu, reference)
        cell = {
            "cell": int(row["cell"]),
            "name": row["name"],
            "condition": condition,
            "seed": int(row["seed"]),
            "forget_split": row["forget"],
            "retain_split": row["retain"],
            "alpha": float(row["alpha"]),
            "checkpoint_state": row["checkpoint_state"],
            "archive_checkpoint": row["archive_checkpoint"],
            "tofu": tofu_summary,
            "safety": safety,
            "safety_by_category": safety_data["by_category"],
        }
        cells.append(cell)
        for key, value in {**tofu_summary, **safety}.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                grouped[condition][key].append(float(value))

    aggregate = {
        "schema_version": "llama31-8b-npo-seeds1-4.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "protocol": {
            "seeds": [1, 2, 3, 4],
            "judge_model": "gpt-5.6-terra",
            "judge_reasoning_effort": "medium",
            "judge_prompt": "hexphi-two-dimensional-v1",
            "safety_protocol": "formal-300",
            "checkpoint_storage": "gb10-verified-archive",
        },
        "cells": cells,
        "condition_summary": {
            condition: {key: stats(values) for key, values in metrics.items()}
            for condition, metrics in sorted(grouped.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
