import json
import sys

from scipy.stats import ks_2samp


summary_path, eval_path, retain_path, trainer_state_path = sys.argv[1:]
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
with open(eval_path, encoding="utf-8") as handle:
    evaluation = json.load(handle)
with open(retain_path, encoding="utf-8") as handle:
    retain = json.load(handle)
with open(trainer_state_path, encoding="utf-8") as handle:
    trainer_state = json.load(handle)


def scores(data):
    by_index = data["forget_truth_ratio"]["value_by_index"]
    return [by_index[str(index)]["score"] for index in range(len(by_index))]


ks = ks_2samp(scores(evaluation), scores(retain))
result = {
    "status": "DONE",
    "global_step": trainer_state.get("global_step"),
    "epoch": trainer_state.get("epoch"),
    "forget_quality": summary["forget_quality"],
    "model_utility": summary["model_utility"],
    "forget_truth_ratio": summary["forget_truth_ratio"],
    "forget_Q_A_Prob": summary["forget_Q_A_Prob"],
    "forget_Q_A_ROUGE": summary["forget_Q_A_ROUGE"],
    "extraction_strength": summary["extraction_strength"],
    "privleak": summary["privleak"],
    "ks_D_200v200": float(ks.statistic),
    "ks_p_200v200": float(ks.pvalue),
    "summary_path": summary_path,
}
print(json.dumps(result, indent=2, sort_keys=True))
