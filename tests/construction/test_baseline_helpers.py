import pytest
import torch
from construction.baseline import generation_batch, normalize_answer, split_membership


def test_membership_uses_complete_rows_and_rejects_ambiguity():
    full = [{'q':'a'}, {'q':'b'}, {'q':'c'}]
    assert split_membership(full, [full[1]]) == ['forget05','retain95','forget05']
    with pytest.raises(ValueError):
        split_membership(full+[full[0]], [full[1]])
    with pytest.raises(ValueError):
        split_membership(full, [{'q':'missing'}])


def test_generation_left_padding_excludes_answer_and_preserves_eos_prompt():
    batch = generation_batch([
        {'input_ids':torch.tensor([2,0,3,4]), 'labels':torch.tensor([-100,-100,3,4])},
        {'input_ids':torch.tensor([5,6]), 'labels':torch.tensor([-100,6])},
    ], 0)
    assert batch['input_ids'].tolist()==[[2,0],[0,5]]
    assert batch['attention_mask'].tolist()==[[1,1],[0,1]]


def test_exact_match_has_explicit_normalization():
    assert normalize_answer(' Ａ  b\n') == 'a b'
    assert normalize_answer('answer.') != normalize_answer('answer')
