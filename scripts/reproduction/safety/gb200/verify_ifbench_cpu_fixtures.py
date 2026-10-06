"""Compare a bounded synthetic CPU fixture vector across the two frozen scorers.

This is a compatibility check, not target evaluation or full IFBench equivalence.
No historical benchmark prompts/responses are used and no files are written.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

REVISION = '1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'
AUDIT_SHA256 = '0dafaa7e87cb2d2cfc62f02a15716c52da66656d446b88137574c83aca0fe86f'
VERSIONS = {'absl-py': '2.5.0', 'click': '8.5.0', 'defusedxml': '0.7.1',
            'emoji': '2.14.1', 'immutabledict': '4.2.1', 'joblib': '1.5.3',
            'langdetect': '1.0.9', 'nltk': '3.10.3', 'regex': '2026.7.19',
            'six': '1.17.0', 'syllapy': '0.8.0', 'tqdm': '4.70.0'}


def fixtures():
    # All text below is newly authored synthetic input, not benchmark data.
    groups = [
        ('range', ['count:word_count_range'], [{'min_words': 3, 'max_words': 3}],
         ['amber copper silver', 'amber copper', 'header\namber copper silver\nfooter', '']),
        ('words', ['length_constraints:number_words'], [{'num_words': 4, 'relation': 'at least'}],
         ['One bright bird flew.', 'One bird.', '**One bright bird flew.**', '']),
        ('sentences', ['length_constraints:number_sentences'], [{'num_sentences': 2, 'relation': 'at least'}],
         ['The river is calm. The sky is clear.', 'The river is calm.',
          'Dr. Green arrived. She brought tea.', '']),
        ('stopwords', ['ratio:stop_words'], [{'percentage': 10}],
         ['amber copper silver', 'the and or', 'the amber and copper', '']),
        ('verb', ['words:start_verb'], [{}],
         ['Walk along the river.', 'The river is calm.', 'Write a clear sentence.', '']),
        ('english', ['language:response_language'], [{'language': 'en'}],
         ['The quiet library has many books about science and history.',
          'Bonjour, cette bibliotheque contient de nombreux livres sur la science et la nature.',
          'This is a complete English sentence about a beautiful garden.', '']),
        ('emoji', ['format:emoji'], [{}],
         ['The garden is bright 😀.', 'The garden is bright.',
          'The sky is blue 😀. The grass is green 🌱.', '']),
        ('syllables', ['words:odd_even_syllables'], [{}],
         ['cat apple bird', 'cat bird', 'apple cat apple', '']),
        ('names', ['count:person_names'], [{'N': 2}],
         ['Emma and Liam visited the garden.', 'Emma visited the garden.',
          'Sophia met Olivia.', '']),
        ('json', ['detectable_format:json_format'], [{}],
         ['{"color":"blue","count":2}', '{color:blue}',
          'Here is the result:\n{"color":"blue"}\nEnd of result.', '']),
        ('keywords', ['keywords:existence'], [{'keywords': ['amber', 'copper']}],
         ['amber copper silver', 'amber silver', '**amber** and **copper**', '']),
        ('multi', ['keywords:existence', 'punctuation:no_comma'],
         [{'keywords': ['amber']}, {}],
         ['amber copper', 'amber, copper', 'copper silver', '']),
    ]
    result = []
    for label, ids, kwargs, responses in groups:
        for n, response in enumerate(responses):
            result.append(dict(id=f'{label}-{n}', key=len(result),
                               prompt=f'Synthetic CPU scorer compatibility fixture {label}-{n}.',
                               instruction_id_list=ids, kwargs=kwargs, response=response))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    runtime = Path(args.runtime).resolve()
    raw_audit = (runtime/'audit.json').read_bytes()
    assert hashlib.sha256(raw_audit).hexdigest() == AUDIT_SHA256
    audit = json.loads(raw_audit)
    assert audit['revision'] == REVISION and len(audit['files']) == 9
    for name, entry in audit['files'].items():
        path = (runtime/name).resolve()
        path.relative_to(runtime)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    assert versions == VERSIONS
    os.environ['NLTK_DATA'] = str(runtime/'nltk_data')
    import nltk
    nltk.data.path.insert(0, os.environ['NLTK_DATA'])

    def no_download(*unused, **ignored):
        raise RuntimeError('Synthetic fixture scoring is offline; no NLTK download')

    nltk.download = no_download
    sys.path.insert(0, str(runtime))
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    random.seed(0)
    import evaluation_lib as ev
    rows = fixtures()
    fixture_bytes = json.dumps(rows, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=False).encode('utf8')
    records = []
    for row in rows:
        inp = ev.InputExample(key=row['key'], instruction_id_list=row['instruction_id_list'],
                              prompt=row['prompt'], kwargs=copy.deepcopy(row['kwargs']))
        record = dict(id=row['id'], instruction_id_list=row['instruction_id_list'])
        for mode, fn in [('strict', ev.test_instruction_following_strict),
                         ('loose', ev.test_instruction_following_loose)]:
            score = fn(copy.deepcopy(inp), {row['prompt']: row['response']})
            record[mode] = dict(prompt_pass=bool(score.follow_all_instructions),
                                instructions=[bool(x) for x in score.follow_instruction_list])
        records.append(record)
    # Independent obvious outcomes ensure an all-true/all-false vector cannot pass.
    by_id = {row['id']: row for row in records}
    assert by_id['range-0']['strict']['prompt_pass']
    assert not by_id['range-1']['strict']['prompt_pass']
    assert by_id['json-0']['strict']['prompt_pass']
    assert not by_id['json-1']['strict']['prompt_pass']
    assert not by_id['json-2']['strict']['prompt_pass'] and by_id['json-2']['loose']['prompt_pass']
    assert by_id['multi-1']['strict']['instructions'] == [True, False]
    assert all(not r['strict']['prompt_pass'] and not r['loose']['prompt_pass']
               for r in records if r['id'].endswith('-3'))
    vector_bytes = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf8')
    print('IFBENCH_CPU_FIXTURES_BEGIN', flush=True)
    print(json.dumps(dict(status='SYNTHETIC_FIXTURES_EXECUTED', python=sys.version,
                          architecture=platform.machine(), installed_versions=versions,
                          source_revision=REVISION, source_audit_sha256=AUDIT_SHA256,
                          fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                          vector_sha256=hashlib.sha256(vector_bytes).hexdigest(),
                          fixture_count=len(rows), records=records,
                          instruction_types=sorted({v for r in rows for v in r['instruction_id_list']}),
                          historical_scoring_repeated=False, target_scoring_executed=False,
                          gpu_runtime_validated=False, external_judge_executed=False,
                          full_score_equivalence_verified=False)))
    print('IFBENCH_CPU_FIXTURES_END', flush=True)


if __name__ == '__main__':
    main()
