import glob
import json
import os

import numpy as np
from scipy.stats import ks_2samp


ROOT = "/home/gb10/open-unlearning"
retain_path = os.path.join(
    ROOT, "saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"
)


def values(path):
    data = json.load(open(path))
    by_index = data["forget_truth_ratio"]["value_by_index"]
    return np.asarray([by_index[str(i)]["score"] for i in range(len(by_index))], dtype=float)


def describe(x):
    symmetric = np.minimum(x, 1.0 / np.maximum(x, np.finfo(float).tiny))
    q = np.quantile(x, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    return {
        "n": int(x.size),
        "raw_mean": float(x.mean()),
        "raw_std": float(x.std()),
        "raw_quantiles_0_10_25_50_75_90_100": [float(v) for v in q],
        "fraction_above_1": float((x > 1).mean()),
        "symmetric_mean": float(symmetric.mean()),
    }


retain = values(retain_path)
result = {"retain95_reference": describe(retain), "runs": []}
pattern = os.path.join(
    ROOT, "saves/eval/tofu_Llama-2-7b-chat-hf*forget05*NPO*/TOFU_SUMMARY.json"
)
for summary_path in sorted(glob.glob(pattern)):
    eval_path = os.path.join(os.path.dirname(summary_path), "TOFU_EVAL.json")
    if not os.path.isfile(eval_path):
        continue
    run = values(eval_path)
    summary = json.load(open(summary_path))
    ks = ks_2samp(run, retain)
    result["runs"].append(
        {
            "run": os.path.basename(os.path.dirname(summary_path)),
            "summary": {
                key: summary.get(key)
                for key in [
                    "Forget Quality",
                    "Model Utility",
                    "Forget Truth Ratio",
                    "Forget QA Probability",
                    "Forget QA ROUGE",
                    "Extraction Strength",
                    "Privleak",
                ]
                if key in summary
            },
            "truth_ratio_scores": describe(run),
            "ks_vs_retain95": {"D": float(ks.statistic), "p": float(ks.pvalue)},
        }
    )

print(json.dumps(result, indent=2))
