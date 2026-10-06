import pytest
from construction.knowledge import format_question,encode_choices
from construction.knowledge import continuation_score,choose_option
import torch
import math

def test_known_continuation_likelihood():
    logits=torch.tensor([[math.log(.75),math.log(.25)],[math.log(.2),math.log(.8)]])
    assert continuation_score(logits,torch.tensor([0,1]))==pytest.approx(math.log(.6))
    assert choose_option([-2.,-1.,-1.,-3.])==1
    with pytest.raises(ValueError):choose_option([float('nan'),0,0,0])

ROW=dict(question='Which number?',choices=['one','two','three','four'],answer=1)
def test_answer_not_in_prompt():
    assert format_question(ROW).endswith('D. four\nAnswer:')
    assert format_question(dict(ROW,answer=2))==format_question(ROW)
def test_invalid_answer_rejected():
    with pytest.raises(ValueError):format_question(dict(ROW,answer=4))
def test_boundary_guard():
    class Tokenizer:
        def __call__(self,text,**kwargs):return {'input_ids':[len(text)]}
    with pytest.raises(ValueError):encode_choices(Tokenizer(),ROW)
