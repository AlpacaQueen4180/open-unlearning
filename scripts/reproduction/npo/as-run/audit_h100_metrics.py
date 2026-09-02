import json
import math
import sys

import numpy as np
from scipy.stats import ks_2samp


eval_path, retain_path, summary_path, trainer_state_path = sys.argv[1:]
with open(eval_path, encoding="utf-8") as handle:
    evaluation = json.load(handle)
with open(retain_path, encoding="utf-8") as handle:
    retain = json.load(handle)
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
with open(trainer_state_path, encoding="utf-8") as handle:
    trainer_state = json.load(handle)


def truth_ratio_scores(payload):
    by_index = payload["forget_truth_ratio"]["value_by_index"]
    return np.asarray(
        [by_index[str(index)]["score"] for index in range(len(by_index))],
        dtype=np.float64,
    )


utility_keys = [
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "ra_Q_A_Prob_normalised",
    "ra_Q_A_ROUGE",
    "ra_Truth_Ratio",
    "wf_Q_A_Prob_normalised",
    "wf_Q_A_ROUGE",
    "wf_Truth_Ratio",
]
utility_values = [float(evaluation[key]["agg_value"]) for key in utility_keys]
utility = len(utility_values) / sum(1.0 / value for value in utility_values)

model_scores = truth_ratio_scores(evaluation)
retain_scores = truth_ratio_scores(retain)
truth_ratio = float(np.mean(np.minimum(model_scores, 1.0 / (model_scores + 1e-10))))
ks = ks_2samp(
    1.0 / (model_scores + 1e-10),
    1.0 / (retain_scores + 1e-10),
)

recomputed = {
    "forget_quality": float(ks.pvalue),
    "model_utility": float(utility),
    "forget_truth_ratio": truth_ratio,
}
reported = {key: float(summary[key]) for key in recomputed}
deltas = {key: recomputed[key] - reported[key] for key in recomputed}

for key, delta in deltas.items():
    if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{key} mismatch: delta={delta}")

result = {
    "status": "VERIFIED",
    "reported": reported,
    "recomputed": recomputed,
    "recompute_minus_reported": deltas,
    "forget_examples": int(model_scores.size),
    "retain_reference_examples": int(retain_scores.size),
    "ks_statistic": float(ks.statistic),
    "raw_truth_ratio_median": float(np.median(model_scores)),
    "raw_truth_ratio_mean": float(np.mean(model_scores)),
    "raw_truth_ratio_fraction_above_one": float(np.mean(model_scores > 1.0)),
    "utility_components": dict(zip(utility_keys, utility_values)),
    "trainer_state": {
        "global_step": trainer_state.get("global_step"),
        "epoch": trainer_state.get("epoch"),
        "max_steps": trainer_state.get("max_steps"),
    },
    "published_rounded": {
        "forget_quality": 0.09,
        "model_utility": 0.53,
        "forget_truth_ratio": 0.71,
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
