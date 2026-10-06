"""Subjectless auxiliary knowledge diagnostic; distinct from formal MMLU5shot."""
import math

def continuation_score(logits, targets):
    import torch
    if logits.ndim!=2 or targets.ndim!=1 or len(logits)!=len(targets) or not len(targets):
        raise ValueError('Expected aligned nonempty continuation')
    value=logits.float().log_softmax(-1).gather(1,targets[:,None]).sum().item()
    if not math.isfinite(value):raise ValueError('Nonfinite likelihood')
    return value

def choose_option(scores):
    if len(scores)!=4 or not all(math.isfinite(s) for s in scores):raise ValueError('Invalid scores')
    # Stable first-option tie break, reported separately by the evaluator.
    return max(range(4),key=lambda i:scores[i])

def format_question(row):
    choices=row['choices']
    if len(choices)!=4 or not all(isinstance(c,str) for c in choices):
        raise ValueError('Expected four text choices')
    if type(row['answer']) is not int or row['answer'] not in range(4):
        raise ValueError('Invalid answer index')
    return row['question']+'\n'+''.join(f'{letter}. {text}\n' for letter,text in zip('ABCD',choices))+'Answer:'

def encode_choices(tokenizer,row):
    prompt=format_question(row)
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    if not prefix:raise ValueError('Empty prompt')
    encoded=[]
    for letter in 'ABCD':
        ids=tokenizer(prompt+' '+letter,add_special_tokens=False)['input_ids']
        if ids[:len(prefix)]!=prefix or len(ids)==len(prefix):
            raise ValueError('Tokenizer boundary changed at answer continuation')
        encoded.append((ids,len(prefix)))
    return encoded
