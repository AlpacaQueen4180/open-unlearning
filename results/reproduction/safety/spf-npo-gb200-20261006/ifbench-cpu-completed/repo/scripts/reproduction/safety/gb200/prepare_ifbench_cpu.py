"""Finite isolated CPU dependency/import preparation; never score historical rows or use GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    a=p.parse_args();task=Path(a.task).resolve()
    assert (task/'preparation.json').is_file()
    state_path=task/'ifbench-cpu-status.json'
    if state_path.exists():raise RuntimeError('CPU stage exists; inspect its evidence before recovery')
    state=dict(status='RUNNING',pid=os.getpid(),started_at=time.time(),commands=[],
               gpu_evaluation_started=False,historical_scoring_repeated=False,external_judge_executed=False)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def run(stage,cmd):
        state['stage']=stage;row=dict(stage=stage,command=cmd,started_at=time.time())
        state['commands'].append(row);save()
        with (task/f'ifbench-cpu-{stage}.log').open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600,
                                  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PIP_DISABLE_PIP_VERSION_CHECK':'1'})
        row.update(returncode=result.returncode,finished_at=time.time());save()
        if result.returncode:raise RuntimeError(stage+' failed; preserve installation/import evidence')
    save()
    try:
        env=task/'ifbench-cpu-venv'
        if env.exists():raise RuntimeError('Isolated scorer venv already exists')
        run('create',[sys.executable,'-m','venv',str(env)])
        py=str(env/'bin/python')
        pins=task/'repo/scripts/reproduction/safety/gb200/ifbench-cpu-h100-20261006.txt'
        assert hashlib.sha256(pins.read_bytes()).hexdigest()=='4dc9bcac4eb9f169c9f4ad1fe1e4f436dbd32de41e8d4c6d10fa2fadc2eea9ec'
        run('install',[py,'-m','pip','install','--no-input','--requirement',str(pins),
                       '--report',str(task/'ifbench-cpu-pip-report.json')])
        run('check',[py,'-m','pip','check'])
        assets=task/'bundle/assets'
        index=json.loads((task/'bundle/learning-index.json').read_text())
        candidates=[e for e in index if e['name'].endswith('/ifbench-runtime/audit.json')]
        assert len(candidates)==1
        runtime=(assets/candidates[0]['name']).parent
        code='''import hashlib,importlib,importlib.metadata as md,json,os,pathlib,platform,sys
runtime=pathlib.Path(sys.argv[1]);audit=json.loads((runtime/'audit.json').read_text())
assert audit['revision']=='1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'
for name,entry in audit['files'].items():
    assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==entry['sha256'],name
os.environ['NLTK_DATA']=str(runtime/'nltk_data')
import nltk
nltk.data.path.insert(0,os.environ['NLTK_DATA'])
def no_download(*a,**k):raise RuntimeError('Offline scorer forbids NLTK download')
nltk.download=no_download
sys.path.insert(0,str(runtime))
roots=['absl','langdetect','nltk','immutabledict','emoji','syllapy','evaluation_lib']
for name in roots:importlib.import_module(name)
pins=pathlib.Path(sys.argv[2]).read_text().splitlines();versions={}
for line in pins:
    if '==' in line and not line.startswith('#'):
        name,expected=line.split('==');actual=md.version(name);assert actual==expected,(name,actual,expected);versions[name]=actual
record=dict(status='IMPORT_PASS_PENDING_SCORE_EQUIVALENCE',python=sys.version,platform=platform.machine(),
            installed_versions=versions,import_roots=roots,source_revision=audit['revision'],
            historical_scoring_repeated=False,target_scoring_executed=False,score_equivalence_verified=False,
            gpu_runtime_validated=False,external_judge_executed=False)
pathlib.Path(sys.argv[3]).write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
        query=task/'ifbench-cpu-import-query.py';query.write_text(code)
        run('imports',[py,str(query),str(runtime),str(pins),str(task/'ifbench-cpu-imports.json')])
        state.update(status='IMPORT_PASS_PENDING_SCORE_EQUIVALENCE',finished_at=time.time(),
                     scoring_venv=str(env),score_equivalence_verified=False)
    except BaseException as error:
        state.update(status='FAILED',error=str(error),finished_at=time.time());save();raise
    save();print(json.dumps({k:state[k] for k in ('status','pid','stage')}))


if __name__=='__main__':main()
