import json
from pathlib import Path

import numpy as np
from scipy.stats import hmean, ks_2samp


ROOT = Path("/tmp/oldtofu")


def load(name):
    return json.load(open(ROOT / name))


def ratios(result):
    forget = result["eval_log_forget.json"]
    para = np.asarray(list(forget["avg_paraphrased_loss"].values()))
    pert = np.asarray(list(forget["average_perturb_loss"].values())).mean(axis=-1)
    return np.exp(pert - para)


def metrics(result, retain):
    names = {
        "eval_real_author_wo_options.json": "Real Authors",
        "eval_real_world_wo_options.json": "Real World",
        "eval_log.json": "Retain",
        "eval_log_forget.json": "Forget",
    }
    out = {}
    for key, label in names.items():
        item = result[key]
        if "eval_log" in key:
            prob = np.exp(-np.asarray(list(item["avg_gt_loss"].values()))).mean()
        else:
            true = np.exp(-np.asarray(list(item["avg_gt_loss"].values())))
            false = np.exp(-np.asarray(list(item["average_perturb_loss"].values())))
            prob = (true / np.concatenate([true[:, None], false], axis=1).sum(-1)).mean()
        rouge = np.asarray(list(item["rougeL_recall"].values())).mean()
        para = np.asarray(list(item["avg_paraphrased_loss"].values()))
        pert = np.asarray(list(item["average_perturb_loss"].values())).mean(axis=-1)
        raw = np.exp(pert - para)
        tr = np.minimum(raw, 1 / raw).mean() if key.endswith("forget.json") else np.maximum(0, 1 - 1 / raw).mean()
        out[f"Prob. {label}"] = float(prob)
        out[f"ROUGE {label}"] = float(rouge)
        out[f"Truth Ratio {label}"] = float(tr)
    utility = hmean([value for key, value in out.items() if "Forget" not in key])
    ks = ks_2samp(ratios(result), ratios(retain))
    return {
        "forget_quality": float(ks.pvalue),
        "ks_D": float(ks.statistic),
        "model_utility": float(utility),
        "forget_truth_ratio": out["Truth Ratio Forget"],
        "forget_Q_A_Prob": out["Prob. Forget"],
        "forget_Q_A_ROUGE": out["ROUGE Forget"],
    }


for full in ["full_wd0.json", "full_wd001.json"]:
    for retain in ["retain95_wd0.json", "retain95_wd001.json"]:
        print(json.dumps({"full": full, "retain": retain, **metrics(load(full), load(retain))}))
