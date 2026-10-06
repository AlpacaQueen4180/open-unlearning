"""Explicit construction-only scales; legacy evaluation remains unchanged.

Suffix accuracy is teacher-forced, not free-running answer extraction. Always
record its definition separately from generation exact match and ROUGE.
"""
import torch


def suffix_fraction(predictions, labels):
    if predictions.shape != labels.shape or labels.ndim != 1 or labels.numel() == 0:
        raise ValueError('Expected equally sized nonempty token vectors')
    correct = predictions.eq(labels).flip(0).long().cumprod(0)
    return float(correct.sum()) / labels.numel()


def teacher_forced_metrics(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError('Expected [batch, sequence, vocab] logits and matching labels')
    result = []
    for row, target in zip(logits[:, :-1], labels[:, 1:]):
        valid = target != -100
        target = target[valid]
        if not target.numel():
            raise ValueError('No unmasked causal target tokens')
        selected = row[valid].float()
        mean_logprob = selected.log_softmax(-1).gather(1, target[:, None]).mean()
        result.append({
            'assistant_tokens': target.numel(),
            'mean_log_probability': float(mean_logprob),
            'geometric_mean_token_probability': float(mean_logprob.exp()),
            'teacher_forced_suffix_fraction': suffix_fraction(selected.argmax(-1), target),
        })
    return result
