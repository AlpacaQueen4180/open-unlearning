#!/usr/bin/env python3
"""Create a public-safe aggregate for no-Judge TOFU and generation queues."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
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


def metric_value(metric: Any) -> Any:
    return metric.get("agg_value") if isinstance(metric, dict) else metric


def tofu_metrics(data: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = {key: metric_value(data[key]) for key in TOFU_KEYS}
    pvalue = result["forget_quality"]
    result["forget_quality_log10"] = math.log10(pvalue) if pvalue and pvalue > 0 else None
    left = [item["score"] for item in data["forget_truth_ratio"]["value_by_index"].values()]
    right = [item["score"] for item in reference["forget_truth_ratio"]["value_by_index"].values()]
    result["forget_quality_ks_statistic"] = float(ks_2samp(left, right).statistic)
    return result


def successful_generation_count(path: str | Path) -> int:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return len({row.get("prompt_id") for row in rows if row.get("status") == "success"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    with args.ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} completed cells, found {len(rows)}")

    cells = []
    for row in rows:
        tofu = load_json(row["tofu_eval"])
        reference = load_json(row["retain_reference"])
        missing = sorted(set(TOFU_KEYS) - tofu.keys())
        if missing:
            raise SystemExit(f"partial TOFU result for {row['name']}: {missing}")
        generated = successful_generation_count(row["hexphi_generation"])
        if generated != 300:
            raise SystemExit(f"incomplete HEx-PHI generation for {row['name']}: {generated}")
        cells.append(
            {
                "cell": int(row["cell"]),
                "name": row["name"],
                "seed": int(row["seed"]),
                "forget_split": row["forget"],
                "retain_split": row["retain"],
                "alpha": float(row["alpha"]),
                "checkpoint_state": row["checkpoint_state"],
                "archive_checkpoint": row["archive_checkpoint"],
                "tofu": tofu_metrics(tofu, reference),
                "hexphi": {
                    "protocol": "formal-300",
                    "successful_generations": generated,
                    "judge_status": row["judge_status"],
                },
            }
        )

    payload = {
        "schema_version": "llama31-8b-nojudge-completion.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "api_judge_enabled": False,
        "judge_status": "pending_judge",
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
