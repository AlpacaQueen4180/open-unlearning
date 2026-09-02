import json
import statistics
from pathlib import Path


root = Path("/home/ai/alpaca")
models = ["Llama-2-7b-chat-hf", "Llama-3.2-1B-Instruct"]
splits = ["forget01", "forget05", "forget10"]
metrics = ["forget_quality", "model_utility", "forget_truth_ratio"]
official = {
    ("Llama-2-7b-chat-hf", "forget01"): (0.40, 0.58, 0.65),
    ("Llama-2-7b-chat-hf", "forget05"): (0.09, 0.53, 0.71),
    ("Llama-2-7b-chat-hf", "forget10"): (0.42, 0.54, 0.73),
    ("Llama-3.2-1B-Instruct", "forget01"): (0.92, 0.56, 0.66),
    ("Llama-3.2-1B-Instruct", "forget05"): (0.14, 0.45, 0.70),
    ("Llama-3.2-1B-Instruct", "forget10"): (0.02, 0.46, 0.70),
}


def status_stem(model: str, split: str, seed: int) -> str:
    if seed == 0 and model == "Llama-2-7b-chat-hf" and split == "forget05":
        return (
            "tofu_Llama-2-7b-chat-hf_forget05_NPO_2xH100_zero3_flash2_"
            "micro4_acc4_e10_20260827_eval_retry2"
        )
    return (
        f"tofu_{model}_{split}_NPO_2xH100_zero3_flash2_"
        f"micro4_acc4_e10_seed{seed}_20260828"
    )


groups = []
for model in models:
    for split in splits:
        runs = []
        for seed in range(5):
            path = root / f"{status_stem(model, split, seed)}.status.json"
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "DONE":
                raise RuntimeError(f"incomplete run: {path}: {payload.get('status')}")
            runs.append(
                {
                    "seed": seed,
                    "status_path": str(path),
                    **{metric: payload[metric] for metric in metrics},
                }
            )

        summary = {}
        official_values = dict(zip(metrics, official[(model, split)]))
        for metric in metrics:
            values = [run[metric] for run in runs]
            mean = statistics.fmean(values)
            summary[metric] = {
                "mean": mean,
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
                "official": official_values[metric],
                "mean_minus_official": mean - official_values[metric],
            }
        groups.append(
            {
                "model": model,
                "forget_split": split,
                "seed_count": len(runs),
                "runs": runs,
                "summary": summary,
            }
        )

print(
    json.dumps(
        {
            "status": "DONE",
            "seed_count_per_cell": 5,
            "run_count": 30,
            "new_run_count": 24,
            "groups": groups,
        },
        indent=2,
    )
)
