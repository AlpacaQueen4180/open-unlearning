"""Offline official IFBench scoring of complete, ID-matched private generations."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['runtime','dataset','responses','output']:
        p.add_argument('--'+arg,required=True)
    args=p.parse_args()
    runtime=Path(args.runtime)
    audit=json.loads((runtime/'audit.json').read_text())
    for name,entry in audit['files'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==entry['sha256'],name
    os.environ['NLTK_DATA']=str(runtime/'nltk_data')
    import nltk
    nltk.data.path.insert(0,os.environ['NLTK_DATA'])
    def no_download(*a,**k):
        raise RuntimeError('Scoring is offline: provision NLTK resources first')
    nltk.download=no_download
    sys.path.insert(0,str(runtime))
    from langdetect import DetectorFactory
    DetectorFactory.seed=0
    random.seed(0)
    import evaluation_lib as ev
    rows=json.loads(Path(args.dataset).read_text())
    responses=[json.loads(line) for line in Path(args.responses).read_text().splitlines() if line.strip()]
    by_id={r['id']:r for r in responses}
    assert len(by_id)==len(responses)==len(rows)
    assert set(by_id)=={r['id'] for r in rows}
    records=[]
    for row in rows:
        response=by_id[row['id']]
        assert response['prompt']==row['prompt']
        assert isinstance(response['response'],str)
        inp=ev.InputExample(key=row['key'],instruction_id_list=row['instruction_id_list'],prompt=row['prompt'],kwargs=[{k:v for k,v in args.items() if v is not None} for args in row['kwargs']])
        result={'id':row['id']}
        for mode,fn in [('strict',ev.test_instruction_following_strict),('loose',ev.test_instruction_following_loose)]:
            score=fn(copy.deepcopy(inp),{row['prompt']:response['response']})
            result[mode]=dict(prompt_pass=score.follow_all_instructions,instructions=score.follow_instruction_list)
        records.append(result)
    summary={mode:dict(prompt_accuracy=sum(r[mode]['prompt_pass'] for r in records)/len(records),
                       instruction_accuracy=sum(sum(r[mode]['instructions']) for r in records)/sum(len(r[mode]['instructions']) for r in records))
             for mode in ['strict','loose']}
    result=dict(status='pass',count=len(records),summary=summary,records=records,source_revision=audit['revision'],
                dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                responses_sha256=hashlib.sha256(Path(args.responses).read_bytes()).hexdigest(),research_gate='insufficient_evidence')
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
