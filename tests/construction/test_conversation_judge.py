import pytest
from construction.conversation_judge import history_for_judgment,ConversationJudgment

def test_history_is_model_response_not_other_conversation():
    rows=[dict(conversation_id='a',turn=1,prompt='q1',response='own1'),dict(conversation_id='a',turn=2,prompt='q2',response='own2'),dict(conversation_id='b',turn=1,prompt='other',response='other')]
    assert [r['content'] for r in history_for_judgment(rows,'a',2)]==['q1','own1','q2','own2']
    assert len(history_for_judgment(rows,'a',1))==2
    with pytest.raises(ValueError):history_for_judgment(rows[1:],'a',2)

def test_scores_cannot_exceed_scale():
    with pytest.raises(ValueError):ConversationJudgment(relevance=6,factual_accuracy=None,instruction_adherence=5,clarity=5,consistency=None,overall=5,verification_required=False,quality='normal',reason='x')
