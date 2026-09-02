import json
import statistics
from pathlib import Path


root = Path("/home/user/alpaca/open-unlearning")
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
h100_mean = {
    ("Llama-2-7b-chat-hf", "forget01"): (0.020022156940717246, 0.6205728683858436, 0.5514521107855948),
    ("Llama-2-7b-chat-hf", "forget05"): (0.5978063911389597, 0.4822205200709299, 0.7064399656100016),
    ("Llama-2-7b-chat-hf", "forget10"): (0.23797539465230785, 0.5208616852524031, 0.7204820860700715),
    ("Llama-3.2-1B-Instruct", "forget01"): (0.017756108346046486, 0.5896471856386354, 0.537798650807563),
    ("Llama-3.2-1B-Instruct", "forget05"): (0.28007264424627537, 0.43954721113160583, 0.6992570141404936),
    ("Llama-3.2-1B-Instruct", "forget10"): (0.11714945467176337, 0.48099719807898716, 0.6666502827325377),
}


def status_path(model: str, split: str, seed: int) -> Path:
    task = (
        f"tofu_{model}_{split}_NPO_1xAda6000_flash2_"
        f"micro8_acc4_e10_seed{seed}_20260829"
    )
    return root / f"{task}.status.json"


groups = []
for model in models:
    for split in splits:
        runs = []
        for seed in range(5):
            path = status_path(model, split, seed)
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
        h100_values = dict(zip(metrics, h100_mean[(model, split)]))
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
                "h100_five_seed_mean": h100_values[metric],
                "mean_minus_h100": mean - h100_values[metric],
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
            "hardware": "NVIDIA RTX 6000 Ada Generation",
            "topology": "1 GPU, microbatch 8, gradient accumulation 4, global batch 32",
            "seed_count_per_cell": 5,
            "run_count": 30,
            "groups": groups,
        },
        indent=2,
    )
)
