from pathlib import Path
import json,hashlib,subprocess,datetime
T=Path('/data/spf-hybrid01-rank40-full-20261012-r1');V=T/'validation-code'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert not (T/'status.json').exists() and not (V/'hybrid01-rank40-full-launch-20261012.json').exists() and not (V/'hybrid01-rank40-full-launch-20261012.raw.log').exists()
assert all(not (T/name).exists() for name in ('spf-hybrid01-rank40-full-s0','candidate-development','optimizer-binding.private.json'))
assert 0 <= (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat('2026-10-12T05:52:12.051399+00:00')).total_seconds() <= 600
previous=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')
assert sha((previous/'spf-hybrid03-lr5e6-full-s0-audit.json').read_bytes())=='d27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486'
assert sha((previous/'hybrid-full-acceptance.private.json').read_bytes())=='ea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53'
for pid in (45219,47443,2953,2961,3029,3501,3889,3896,4074,4081,4149,4577,4584,4652,5110,8483,8486,8554,9069,9072,9140,10758,10761,10829,11297,11577,12367,12370,12438,14033,15066,15134,15635,15915,15918,16733,16736,16947,17483,17763,17766,17834,18362,18642,18645,18802,18939,19076,19213,19348):
    p=Path('/proc')/str(pid)/'stat'
    assert not p.exists() or p.read_text().rsplit(')',1)[1].strip().split()[0]=='Z'
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0 and not p.stdout.strip() and not p.stderr.strip()
assert sha(Path('/data/spf-npo-20261006-r4/spf-full-s0-audit.json').read_bytes())=='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
import os
vfs=os.statvfs('/data');assert vfs.f_bavail*vfs.f_frsize>=160*1024**3
assert sha(Path('/data/spf-npo-20261006-r4/frozen.json').read_bytes())=='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'

import importlib.util,os
expected={'run_spf_hybrid01_rank40_full_queue.py': {'size': 19358, 'sha256': '36f0c3d74b7e5f9bb4bd9a03fcfc0e5d991f26b903998742aa492602d53f488e'}, 'run_spf_hybrid01_rank40_full.py': {'size': 10172, 'sha256': '755345d730e4f2d3cb2aecb225c38b545bc230be342a6132483801dc103b8727'}, 'spf_hybrid01_rank40_development_executor.py': {'size': 16338, 'sha256': '851c57fd60fd36280dc84fcdbdb9631df1124d73326999b906afe8348b346c40'}, 'spf_anchor_hybrid.py': {'size': 4864, 'sha256': 'fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc'}, 'verify_spf_anchor_hybrid_rank40.py': {'size': 8630, 'sha256': '548814b7d849fc9e58f40441f176ad463f1aa71854c235fed1daf998d5884be1'}, 'plan.private.json': {'size': 4510, 'sha256': 'ef346727c0948dcdb9ce92832b33e455683422bdee2612b5f106b1a5bfbd3829'}, 'bound-snapshot.private.json': {'size': 20336, 'sha256': '52f57a0bb9238ad9ca7ccedab117cdefeef77b28e3e445e095b426073730483e'}}
assert all(sha((V/name).read_bytes())==row['sha256'] for name,row in expected.items())
plan=json.loads((V/'plan.private.json').read_bytes())
for path,digest in plan['source_sha256'].items(): assert sha(Path(path).read_bytes())==digest
spec=importlib.util.spec_from_file_location('new_rank40_queue',V/'run_spf_hybrid01_rank40_full_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.validate_plan(plan);m.validate_search_decision(plan);m.validate_predecessor_runtime()
guard=m.module(Path('/data/spf-development-20261007-r1/validation-code/run_spf_development_recovery_guards.py'),'original_idle_guard')
snapshot=json.loads((V/'bound-snapshot.private.json').read_bytes())
guard.validate_snapshot(snapshot,datetime.datetime.now(datetime.timezone.utc));m.validate_completed_predecessor(snapshot)
command=['/data/npo-gb200-20261004/validation-20261006/venv/bin/python',str(V/'run_spf_hybrid01_rank40_full_queue.py'),
    '--source-sha256','36f0c3d74b7e5f9bb4bd9a03fcfc0e5d991f26b903998742aa492602d53f488e','--plan',str(V/'plan.private.json'),'--plan-sha256','ef346727c0948dcdb9ce92832b33e455683422bdee2612b5f106b1a5bfbd3829',
    '--snapshot',str(V/'bound-snapshot.private.json'),'--snapshot-sha256','52f57a0bb9238ad9ca7ccedab117cdefeef77b28e3e445e095b426073730483e']
receipt=V/'hybrid01-rank40-full-launch-20261012.json';log=V/'hybrid01-rank40-full-launch-20261012.raw.log'
with receipt.open('x',encoding='utf8') as handle:
    with log.open('xb') as stream:
        proc=subprocess.Popen(command,cwd='/data/spf-npo-20261006-r4/repo',env={**os.environ,
            'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled'},
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(status='SUBMITTED_RESULT_NOT_CHECKED',pid=proc.pid,command=command,log=str(log),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_sha256='52f57a0bb9238ad9ca7ccedab117cdefeef77b28e3e445e095b426073730483e',plan_sha256='ef346727c0948dcdb9ce92832b33e455683422bdee2612b5f106b1a5bfbd3829',
            new_paid_api_calls=0,previous_completed_training_or_evaluations_repeated=False)
        handle.write(json.dumps(value,indent=2)+chr(10));handle.flush();os.fsync(handle.fileno())
print(json.dumps(value))
