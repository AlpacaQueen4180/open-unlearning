from types import SimpleNamespace

import torch

from evals.metrics.utils import evaluate_probability


class _BFloat16Model:
    device = torch.device("cpu")

    def __call__(self, **_batch):
        logits = torch.zeros((1, 3, 4), dtype=torch.bfloat16)
        return SimpleNamespace(logits=logits)


def test_evaluate_probability_serializes_bfloat16_metrics():
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[-100, 1, 2]]),
    }

    result = evaluate_probability(_BFloat16Model(), batch)

    assert len(result) == 1
    assert isinstance(result[0]["prob"], float)
    assert isinstance(result[0]["avg_loss"], float)


if __name__ == "__main__":
    test_evaluate_probability_serializes_bfloat16_metrics()
