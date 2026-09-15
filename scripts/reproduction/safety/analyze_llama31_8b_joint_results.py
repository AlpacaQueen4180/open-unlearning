#!/usr/bin/env python3
"""Build the public-safe joint TOFU, HEx-PHI, MMLU, and MT-Bench report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = {
    "local_full_forget05_npo_no_retain": ("local_f05_no", "自製 full, f05, no retain", "local_full"),
    "local_full_forget05_npo_retain95": ("local_f05_retain", "自製 full, f05, +retain95", "local_full"),
    "published_full_forget01_npo_no_retain": ("published_f01_no", "公開 full, f01, no retain", "published_full"),
    "published_full_forget01_npo_retain99": ("published_f01_retain", "公開 full, f01, +retain99", "published_full"),
    "published_full_forget05_npo_no_retain": ("published_f05_no", "公開 full, f05, no retain", "published_full"),
    "published_full_forget05_npo_retain95": ("published_f05_retain", "公開 full, f05, +retain95", "published_full"),
    "published_full_forget10_npo_no_retain": ("published_f10_no", "公開 full, f10, no retain", "published_full"),
    "published_full_forget10_npo_retain90": ("published_f10_retain", "公開 full, f10, +retain90", "published_full"),
}

CAPABILITY_PREFIX = {
    "local_f05_no": "local_f05_no_seed",
    "local_f05_retain": "local_f05_retain_seed",
    "published_f01_no": "published_f01_no_seed",
    "published_f01_retain": "published_f01_retain_seed",
    "published_f05_no": "published_f05_no_seed",
    "published_f05_retain": "published_f05_retain_seed",
    "published_f10_no": "published_f10_no_seed",
    "published_f10_retain": "published_f10_retain_seed",
}

BASELINE_LABELS = {
    "original_instruct": "Original instruct",
    "published_full": "公開 TOFU full",
    "local_full": "自製 TOFU full",
    "published_retain95": "公開 retain95 oracle",
    "local_retain95": "自製 retain95 oracle",
    "published_retain99": "公開 retain99 oracle",
    "published_retain90": "公開 retain90 oracle",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(metric: Any) -> Any:
    return metric.get("agg_value") if isinstance(metric, dict) else metric


def stat(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    lmean, rmean = statistics.mean(left), statistics.mean(right)
    numerator = sum((l - lmean) * (r - rmean) for l, r in zip(left, right))
    denominator = math.sqrt(sum((l - lmean) ** 2 for l in left) * sum((r - rmean) ** 2 for r in right))
    return numerator / denominator if denominator else None


def ks_statistic(left: list[float], right: list[float]) -> float:
    values = sorted(set(left + right))
    return max(abs(sum(x <= point for x in left) / len(left) - sum(x <= point for x in right) / len(right)) for point in values)


def tofu_from_eval(data: dict[str, Any], reference: dict[str, Any] | None = None) -> dict[str, Any]:
    keys = ("forget_quality", "forget_truth_ratio", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "retain_Truth_Ratio", "model_utility")
    result = {key: value(data.get(key)) for key in keys}
    if reference is not None:
        left = [item["score"] for item in data["forget_truth_ratio"]["value_by_index"].values()]
        right = [item["score"] for item in reference["forget_truth_ratio"]["value_by_index"].values()]
        result["forget_quality_ks_statistic"] = ks_statistic(left, right)
    return result


def safety_from_summary(data: dict[str, Any]) -> dict[str, Any]:
    overall = data["overall"]
    behavior, quality, coverage = overall["safety_behavior"], overall["generation_quality"], overall["judge_coverage"]
    return {
        "harmful_assistance_rate": behavior["harmful_assistance"]["rate"],
        "refusal_rate": behavior["refusal"]["rate"],
        "safe_non_refusal_rate": behavior["safe_non_refusal"]["rate"],
        "degeneration_rate": quality["degenerate"]["rate"],
        "policy_blocked": coverage["policy_blocked"],
        "judge_successful": coverage["successful"],
        "harmful_lower_bound": overall["harmful_assistance_rate_bounds_all_inputs"]["lower"],
        "harmful_upper_bound": overall["harmful_assistance_rate_bounds_all_inputs"]["upper"],
    }


def prior_safety(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "harmful_assistance_rate": data["harmful_assistance"]["rate"],
        "refusal_rate": data["refusal"]["rate"],
        "safe_non_refusal_rate": data["safe_non_refusal"]["rate"],
        "degeneration_rate": data["degenerate"]["rate"],
        "policy_blocked": data["policy_blocked"],
        "judge_successful": data["judge_successful"],
        "harmful_lower_bound": data["harmful_bounds_all_inputs"]["lower"],
        "harmful_upper_bound": data["harmful_bounds_all_inputs"]["upper"],
    }


def fmt(value_: float | None, digits: int = 4) -> str:
    return "—" if value_ is None else f"{value_:.{digits}f}"


def fmt_pct(value_: float | None) -> str:
    return "—" if value_ is None else f"{100 * value_:.2f}%"


def fmt_stat(item: dict[str, Any], digits: int = 4, percent: bool = False) -> str:
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    return f"{scale * item['mean']:.{digits}f}±{scale * item['sample_sd']:.{digits}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    safety_root = repo / "results/reproduction/safety/llama31_8b_completion_20260915"
    capability_root = repo / "results/reproduction/capability/llama31_8b_completion_20260915"

    prior0 = read_json(safety_root / "prior_seed0.json")
    prior14 = read_json(safety_root / "prior_seeds1_4.json")
    public_f05 = read_json(safety_root / "public_f05_seeds1_4_tofu.json")
    pending_safety = read_json(safety_root / "pending_hexphi_judged.json")["models"]
    supplemental_tofu = read_json(safety_root / "supplemental_tofu_metrics.json")
    mmlu = read_json(capability_root / "mmlu_and_mtbench_generation.json")["models"]
    mtbench = read_json(capability_root / "mtbench_terra.json")["models"]

    conditions: dict[str, list[dict[str, Any]]] = {canonical: [] for canonical, _, _ in CONDITIONS.values()}
    for source_name, cell in prior0["cells"].items():
        if source_name not in CONDITIONS:
            continue
        canonical, _, _ = CONDITIONS[source_name]
        conditions[canonical].append({"seed": 0, "tofu": cell["tofu"], "safety": prior_safety(cell["safety"])})
    for cell in prior14["cells"]:
        canonical, _, _ = CONDITIONS[cell["condition"]]
        conditions[canonical].append({"seed": int(cell["seed"]), "tofu": cell["tofu"], "safety": cell["safety"]})

    for variant in ("no", "retain"):
        canonical = f"published_f05_{variant}"
        stem = "published_full_forget05_npo_no_retain" if variant == "no" else "published_full_forget05_npo_retain95"
        seed0_safety = read_json(safety_root / f"public_f05_{variant}_seed0_safety.json")
        conditions[canonical].append({
            "seed": 0,
            "tofu": supplemental_tofu["public_f05_seed0"][variant],
            "safety": safety_from_summary(seed0_safety),
        })
        for cell in public_f05["cells"]:
            if cell["name"].startswith(stem + "_seed"):
                conditions[canonical].append({"seed": int(cell["seed"]), "tofu": cell["tofu"], "safety": safety_from_summary(pending_safety[cell["name"]])})

    capability_groups = {}
    for canonical, prefix in CAPABILITY_PREFIX.items():
        reps = []
        for seed in (0, 2, 4):
            model_id = prefix + str(seed)
            reps.append({
                "seed": seed,
                "model_id": model_id,
                "mmlu": mmlu[model_id]["mmlu"]["accuracy"],
                "mtbench": mtbench[model_id]["overall"]["mean_score"],
                "mtbench_turn1": mtbench[model_id]["turn_1"]["mean_score"],
                "mtbench_turn2": mtbench[model_id]["turn_2"]["mean_score"],
            })
        capability_groups[canonical] = reps

    condition_summary = {}
    observations = []
    for source_name, (canonical, label, baseline) in CONDITIONS.items():
        cells = sorted(conditions[canonical], key=lambda item: item["seed"])
        if [cell["seed"] for cell in cells] != [0, 1, 2, 3, 4]:
            raise ValueError(f"{canonical} does not have seeds 0-4")
        reps = capability_groups[canonical]
        forget_quality_values = [cell["tofu"]["forget_quality"] for cell in cells]
        summary = {
            "label": label,
            "baseline": baseline,
            "forget_quality_p_values": forget_quality_values,
            "forget_quality_log10": [math.log10(item) if item > 0 else None for item in forget_quality_values],
            "fq_pass_count": sum(item > 0.05 for item in forget_quality_values),
            "ks": stat([cell["tofu"]["forget_quality_ks_statistic"] for cell in cells]),
            "retain_truth_ratio": stat([cell["tofu"]["retain_Truth_Ratio"] for cell in cells]),
            "model_utility": stat([cell["tofu"]["model_utility"] for cell in cells]),
            "harmful": stat([cell["safety"]["harmful_assistance_rate"] for cell in cells]),
            "degeneration": stat([cell["safety"]["degeneration_rate"] for cell in cells]),
            "policy_blocked": stat([float(cell["safety"]["policy_blocked"]) for cell in cells]),
            "mmlu": stat([rep["mmlu"] for rep in reps]),
            "mtbench": stat([rep["mtbench"] for rep in reps]),
            "mtbench_turn1": stat([rep["mtbench_turn1"] for rep in reps]),
            "mtbench_turn2": stat([rep["mtbench_turn2"] for rep in reps]),
        }
        condition_summary[canonical] = summary
        by_seed = {cell["seed"]: cell for cell in cells}
        for rep in reps:
            cell = by_seed[rep["seed"]]
            observations.append({
                "condition": canonical, "seed": rep["seed"], **rep,
                "model_utility": cell["tofu"]["model_utility"],
                "harmful": cell["safety"]["harmful_assistance_rate"],
                "degeneration": cell["safety"]["degeneration_rate"],
            })

    baseline_tofu = {
        "original_instruct": None,
        "published_full": prior0["published_baselines"]["tofu_full"]["tofu"],
        "local_full": prior0["cells"]["local_full"]["tofu"],
        "published_retain95": prior0["published_baselines"]["retain95"]["tofu"],
        "local_retain95": prior0["cells"]["local_retain95"]["tofu"],
        "published_retain99": supplemental_tofu["oracle_tofu"]["retain99"],
        "published_retain90": supplemental_tofu["oracle_tofu"]["retain90"],
    }
    baseline_safety = {
        "original_instruct": safety_from_summary(read_json(safety_root / "original_instruct_safety.json")),
        "published_full": prior_safety(prior0["published_baselines"]["tofu_full"]["safety"]),
        "local_full": prior_safety(prior0["cells"]["local_full"]["safety"]),
        "published_retain95": prior_safety(prior0["published_baselines"]["retain95"]["safety"]),
        "local_retain95": prior_safety(prior0["cells"]["local_retain95"]["safety"]),
        "published_retain99": safety_from_summary(pending_safety["published_retain99_oracle"]),
        "published_retain90": safety_from_summary(pending_safety["published_retain90_oracle"]),
    }
    baselines = {}
    for model_id, label in BASELINE_LABELS.items():
        baselines[model_id] = {
            "label": label,
            "tofu": baseline_tofu[model_id],
            "safety": baseline_safety[model_id],
            "mmlu": mmlu[model_id]["mmlu"]["accuracy"],
            "mtbench": mtbench[model_id]["overall"]["mean_score"],
            "mtbench_turn1": mtbench[model_id]["turn_1"]["mean_score"],
            "mtbench_turn2": mtbench[model_id]["turn_2"]["mean_score"],
        }

    for canonical, summary in condition_summary.items():
        baseline = baselines[summary["baseline"]]
        summary["delta_vs_start"] = {
            "harmful_percentage_points": 100 * (summary["harmful"]["mean"] - baseline["safety"]["harmful_assistance_rate"]),
            "mmlu": summary["mmlu"]["mean"] - baseline["mmlu"],
            "mtbench": summary["mtbench"]["mean"] - baseline["mtbench"],
        }

    correlations = {}
    metric_pairs = (
        ("model_utility", "mmlu"), ("model_utility", "mtbench"),
        ("mmlu", "mtbench"), ("degeneration", "mtbench"), ("harmful", "mtbench"),
    )
    for left, right in metric_pairs:
        correlations[f"{left}_vs_{right}"] = {"n": len(observations), "pearson_r": pearson([row[left] for row in observations], [row[right] for row in observations])}
    normal = [row for row in observations if row["degeneration"] < 0.05]
    correlations["harmful_vs_mtbench_nondegenerate"] = {
        "n": len(normal), "pearson_r": pearson([row["harmful"] for row in normal], [row["mtbench"] for row in normal])
    }
    all_mmlu = [item["mmlu"]["accuracy"] for item in mmlu.values()]
    all_mt = [mtbench[key]["overall"]["mean_score"] for key in mmlu]
    correlations["mmlu_vs_mtbench_all31"] = {"n": 31, "pearson_r": pearson(all_mmlu, all_mt)}

    payload = {
        "schema_version": "llama31-8b-joint-analysis.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "protocol": {
            "tofu_and_hexphi_seeds": [0, 1, 2, 3, 4],
            "capability_representative_seeds": [0, 2, 4],
            "mmlu": "57 subjects, 5-shot, no chat template",
            "mtbench": "80 two-turn questions; gpt-5.6-terra medium single-answer judge",
            "hexphi": "formal-300; gpt-5.6-terra medium; successful-row point estimates",
        },
        "baselines": baselines,
        "conditions": condition_summary,
        "representative_seed_observations": observations,
        "correlations": correlations,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Llama-3.1-8B TOFU NPO：TOFU、Safety 與一般能力聯合分析",
        "",
        f"> 產生時間：{payload['created_at']}<br>",
        "> 分支：`repro/npo-h100-ada-5seed`<br>",
        "> 本報告只包含 aggregate statistics；不包含 HEx-PHI 原文、raw model responses、Judge rationale、API key 或 checkpoint 權重。",
        "",
        "## 一、結論摘要",
        "",
        "1. **Retain loss 最一致的作用是避免模型 collapse，但不保證恢復 safety。** 公開 full 的 f05／f10 no-retain 均在五個 seeds 全面 degeneration；加入 retain 後，TOFU utility、MMLU 與 MT-Bench 明顯恢復。",
        "2. **一般能力評估確認：低 harmfulness 若伴隨低 MT-Bench 或 degeneration，不能解讀為安全改善。** Collapse 模型的 HEx-PHI harmfulness 接近 0%，但開放式對話能力同時落至接近量表底部。",
        "3. **Safety outcome 具有強烈的 checkpoint、forget ratio 與 seed 依賴性。** 自製 f05 +retain95 的 harmfulness 高於其 full baseline；公開 f01 的 retain 效果接近零；公開 f05／f10 +retain 則平均低於公開 full，但跨 seed 波動明顯。",
        "4. **MMLU 與 MT-Bench 提供互補資訊。** MMLU 測知識／選擇題能力，MT-Bench 對 collapse、重複生成與多輪品質更敏感；因此 TOFU MU 正常並不足以單獨證明模型整體可用。",
        "5. **MT-Bench 分數是 Terra-judged internal comparison。** Judge 為 `gpt-5.6-terra` medium，不能直接當作官方 GPT-4 MT-Bench leaderboard 分數。",
        "",
        "## 二、Baseline 與 oracle",
        "",
        "| Checkpoint | TOFU MU | MMLU | MT-Bench | Turn 2 | Harmfulness | Degeneration | Blocked/300 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in BASELINE_LABELS:
        item = baselines[model_id]
        tofu_mu = item["tofu"]["model_utility"] if item["tofu"] else None
        lines.append(f"| {item['label']} | {fmt(tofu_mu)} | {fmt(item['mmlu'])} | {fmt(item['mtbench'], 2)} | {fmt(item['mtbench_turn2'], 2)} | {fmt_pct(item['safety']['harmful_assistance_rate'])} | {fmt_pct(item['safety']['degeneration_rate'])} | {item['safety']['policy_blocked']} |")

    lines += [
        "",
        "Original instruct 是 safety 參考點；TOFU full 與 retain oracles 已接觸 TOFU fine-tuning data，因此 harmfulness 明顯高於 original。Oracle 的角色是對應 forget split 的 retrain reference，不代表 safety oracle。",
        "",
        "## 三、八個 NPO 設定",
        "",
        "TOFU／HEx-PHI 使用五個 seeds；MMLU／MT-Bench 使用事前固定的 seeds 0、2、4。`ΔH` 是相對同一起始 full checkpoint 的 successful-row harmfulness 百分點差；`ΔMMLU`、`ΔMT` 同理。",
        "",
        "| 設定 | FQ>.05 | MU (5 seeds) | MMLU (3 seeds) | MT-Bench (3 seeds) | Harmfulness (5 seeds) | Degeneration | ΔH | ΔMMLU | ΔMT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source_name, (canonical, _, _) in CONDITIONS.items():
        item = condition_summary[canonical]; delta = item["delta_vs_start"]
        lines.append(
            f"| {item['label']} | {item['fq_pass_count']}/5 | {fmt_stat(item['model_utility'])} | {fmt_stat(item['mmlu'])} | {fmt_stat(item['mtbench'], 2)} | {fmt_stat(item['harmful'], 2, True)} | {fmt_stat(item['degeneration'], 2, True)} | {delta['harmful_percentage_points']:+.2f}pp | {delta['mmlu']:+.4f} | {delta['mtbench']:+.2f} |"
        )

    lines += [
        "",
        "### 3.1 TOFU forgetting diagnostics",
        "",
        "FQ p-value 與 `log10(p)` 依序列出 seeds 0–4；p-value 不做平均。KS 與 retain truth ratio 則報五個 seeds 的 mean±sample SD。",
        "",
        "| 設定 | FQ p-value (s0…s4) | log10(p) (s0…s4) | KS | Retain truth ratio |",
        "|---|---|---|---:|---:|",
    ]
    for canonical, _, _ in CONDITIONS.values():
        item = condition_summary[canonical]
        p_values = " / ".join(f"{number:.2e}" for number in item["forget_quality_p_values"])
        log_values = " / ".join("−∞" if number is None else f"{number:.2f}" for number in item["forget_quality_log10"])
        lines.append(
            f"| {item['label']} | {p_values} | {log_values} | {fmt_stat(item['ks'], 3)} | {fmt_stat(item['retain_truth_ratio'], 4)} |"
        )

    lines += [
        "",
        "### 3.2 公開 full：forget-ratio 趨勢",
        "",
        "- `forget01` with／without retain 都保有一般能力，兩者差異很小；retain99 並未帶來一致的額外 safety benefit。",
        "- `forget05` no-retain 在 5/5 seeds 全面 degeneration。retain95 能恢復 capability，但 safety 改善幅度隨 seed 變化。",
        "- `forget10` no-retain 同樣在 5/5 seeds collapse。retain90 避免全面崩潰，但既有結果顯示 FQ、harmfulness 與 capability 都有較大 seed variance。",
        "",
        "### 3.3 起始 checkpoint 敏感性",
        "",
        "自製 full 與公開 full 的 baseline TOFU metrics 接近，但 f05 NPO 動態不同：公開 no-retain 是一致全面 collapse，自製 no-retain 則是嚴重但程度不一的能力損失；with-retain 的 safety 方向也不同。這表示 baseline MU 接近不足以保證相同的 unlearning trajectory。",
        "",
        "## 四、跨指標關係",
        "",
        "以下 Pearson correlation 是探索性描述，不代表因果；NPO 部分使用固定 seeds 0、2、4。",
        "",
        "| 指標組合 | n | Pearson r | 解讀 |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "model_utility_vs_mmlu": "TOFU retain utility 與知識能力的一致程度",
        "model_utility_vs_mtbench": "TOFU utility 與開放式對話品質的一致程度",
        "mmlu_vs_mtbench": "兩種一般能力 benchmark 的一致程度",
        "degeneration_vs_mtbench": "生成崩潰是否對應低對話品質",
        "harmful_vs_mtbench": "全部 NPO reps 中 harmfulness 與能力的混合關係",
        "harmful_vs_mtbench_nondegenerate": "排除 degeneration≥5% 後的 safety／能力關係",
        "mmlu_vs_mtbench_all31": "全部 baseline、oracle 與 NPO checkpoint",
    }
    for key, item in correlations.items():
        lines.append(f"| `{key}` | {item['n']} | {fmt(item['pearson_r'], 3)} | {descriptions[key]} |")

    lines += [
        "",
        "能力與 harmfulness 的正相關可能主要反映 collapse：失去正常回答能力的模型既無法完成 MT-Bench，也無法提供可判定的 harmful assistance。只有在 degeneration 很低且 capability 可接受的子集中，safety 差異才較適合解讀為 alignment 變化。",
        "",
        "## 五、完整性與限制",
        "",
        "- 八個 NPO conditions 的 TOFU 與 HEx-PHI 均有 seeds 0–4；一般能力使用固定 seeds 0、2、4，而不是事後挑選結果。",
        "- Formal-300 為 deterministic generation；Judge 點估計只以成功判定列為分母，policy-blocked 必須搭配 coverage／bounds 解讀。",
        "- Baseline／oracle 多數只有單一 checkpoint；尚未量測多個 independently fine-tuned full checkpoints 的 starting-checkpoint variance。",
        "- MT-Bench 使用 Terra structured-output Judge，適合本研究內部比較，但沒有官方 GPT-4 Judge 的 leaderboard 可比性。",
        "- Correlation 樣本包含同一訓練設定的多個 seeds，不應視為彼此完全獨立的 24 個研究條件。",
        "",
        "## 六、建議下一步",
        "",
        "1. 對 non-degenerate checkpoints 做同 prompt 的 full→NPO label transition 與 paired bootstrap，主攻自製 f05 +retain95 和公開 f05／f10 +retain。",
        "2. 以人工分層抽查校準 Terra HEx-PHI labels：全收 policy-blocked，另抽 harmful、safe、degenerate 與 safe→harmful transitions。",
        "3. 若需要外部 MT-Bench 可比性，使用官方 GPT-4 Judge 對七個 baselines／oracles與固定代表 NPO checkpoints 做校準；不要直接把 Terra 分數對照 leaderboard。",
        "4. 若論文要主張 checkpoint-level robustness，至少再建立數個 independent full fine-tuning seeds，再各自執行固定的 f05 +retain／no-retain。",
        "",
        "## 七、Artifacts",
        "",
        "- Machine-readable joint analysis：`results/reproduction/safety/llama31_8b_completion_20260915/joint_analysis.json`",
        "- MMLU aggregate：`results/reproduction/capability/llama31_8b_completion_20260915/mmlu_and_mtbench_generation.json`",
        "- MT-Bench aggregate：`results/reproduction/capability/llama31_8b_completion_20260915/mtbench_terra.json`",
        "- 新增 HEx-PHI aggregate：`results/reproduction/safety/llama31_8b_completion_20260915/pending_hexphi_judged.json`",
    ]
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
