import math
import pytest
import torch
from construction.learning_metrics import suffix_fraction, teacher_forced_metrics


@pytest.mark.parametrize('prediction,expected', [([0, 0, 0], 0.), ([1, 2, 0], 0.),
                                                ([0, 2, 3], 2/3), ([1, 2, 3], 1.)])
def test_suffix_including_no_matching_suffix(prediction, expected):
    assert suffix_fraction(torch.tensor(prediction), torch.tensor([1, 2, 3])) == expected


def test_probability_scale_masks_and_causal_shift():
    result = teacher_forced_metrics(torch.zeros(1, 4, 4), torch.tensor([[-100, -100, 1, 2]]))[0]
    assert result['assistant_tokens'] == 2
    assert result['mean_log_probability'] == pytest.approx(-math.log(4))
    assert result['geometric_mean_token_probability'] == pytest.approx(.25)
    assert result['teacher_forced_suffix_fraction'] == 0


def test_empty_mask_is_not_a_successful_zero_score():
    with pytest.raises(ValueError, match='No unmasked'):
        teacher_forced_metrics(torch.zeros(1, 3, 4), torch.full((1, 3), -100))
