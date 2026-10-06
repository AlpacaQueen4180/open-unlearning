"""Matched, offline construction baseline evaluation helpers."""
import unicodedata
import torch
from construction.protocol import digest_json


def split_membership(full, retain):
    hashes = [digest_json(row) for row in full]
    retained = {digest_json(row) for row in retain}
    if len(set(hashes)) != len(hashes) or len(retained) != len(retain):
        raise ValueError('Ambiguous duplicate TOFU records')
    if not retained < set(hashes):
        raise ValueError('retain95 must be a strict subset of full')
    return [('retain95' if digest in retained else 'forget05') for digest in hashes]


def generation_batch(items, pad_id):
    prefixes = []
    for item in items:
        positions = torch.nonzero(item['labels'] != -100).flatten()
        if not len(positions) or int(positions[0]) == 0:
            raise ValueError('Need nonempty prompt before assistant tokens')
        prefixes.append(item['input_ids'][:int(positions[0])])
    width = max(len(ids) for ids in prefixes)
    ids = torch.full((len(prefixes), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, prefix in enumerate(prefixes):
        ids[i, -len(prefix):] = prefix
        attention[i, -len(prefix):] = 1
    return {'input_ids': ids, 'attention_mask': attention}


def normalize_answer(text):
    # Deliberately preserve punctuation: this is not token-F1 or fuzzy matching.
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
