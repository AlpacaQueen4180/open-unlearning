from scripts.reproduction.safety.summarize_llama31_8b_tofu_rebuild_npo_seed0 import (
    bounds_overlap,
    compare_metrics,
)


def test_bounds_overlap_inclusive():
    assert bounds_overlap({"lower": 0.60, "upper": 0.70}, {"lower": 0.70, "upper": 0.80})
    assert not bounds_overlap({"lower": 0.60, "upper": 0.69}, {"lower": 0.70, "upper": 0.80})


def test_core_metric_reproduction_threshold():
    baseline = {
        "model_utility": 0.60,
        "retain_Truth_Ratio": 0.50,
        "forget_Q_A_Prob": 0.80,
        "forget_Q_A_ROUGE": 0.70,
    }
    close = {key: value + 0.03 for key, value in baseline.items()}
    far = dict(close, model_utility=0.64)
    assert compare_metrics(close, baseline)["passes_0_03"]
    assert not compare_metrics(far, baseline)["passes_0_03"]
    incomplete = dict(close)
    incomplete["model_utility"] = None
    result = compare_metrics(incomplete, baseline)
    assert not result["complete"]
    assert not result["passes_0_03"]
