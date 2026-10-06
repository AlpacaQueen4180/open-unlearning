import importlib.util
from pathlib import Path

from construction.protocol import prompt_hash

spec = importlib.util.spec_from_file_location('candidate_builder', Path(__file__).resolve().parents[2] /
                                            'scripts/reproduction/safety/prepare_development_candidates.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_selection_excludes_normalized_overlap_and_internal_duplicates():
    rows = [{'id': str(i), 'prompt': text} for i, text in enumerate([' blocked ', 'A', ' a ', 'B'])]
    kept, counts = builder.select(rows, {prompt_hash('BLOCKED')}, 10, 'fixture:')
    assert {prompt_hash(row['prompt']) for row in kept} == {prompt_hash('A'), prompt_hash('B')}
    assert counts['blocked_overlap'] == counts['internal_duplicate'] == 1


def test_unique_row_selection_is_order_independent_and_bounded():
    rows = [{'id': str(i), 'prompt': str(i)} for i in range(30)]
    forward, _ = builder.select(rows, set(), 5, 'fixture:')
    backward, _ = builder.select(reversed(rows), set(), 5, 'fixture:')
    assert forward == backward
    assert len(forward) == 5
