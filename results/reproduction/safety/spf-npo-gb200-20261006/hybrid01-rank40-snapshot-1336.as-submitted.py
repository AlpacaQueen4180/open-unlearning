from pathlib import Path
import datetime
import hashlib
import json
import subprocess

construction=Path('/data/spf-npo-20261006-r4')
previous=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')
candidate=Path('/data/spf-hybrid01-rank40-full-20261012-r1')

def read(path):
    return json.loads(path.read_bytes()) if path.is_file() else None

def record(path):
    return dict(size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.is_file() else None

def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=20)
    return dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)

def branch(path,target,launch_name,names):
    validation=path/'validation-code'
    launch=read(validation/launch_name)
    result=dict(new_root_exists=path.exists(),launch=launch,status=read(path/'status.json'),
        checkpoint_exists=(path/target).exists(),candidate_audit=record(path/(target+'-audit.json')),
        source_files={name:record(validation/name) for name in names},
        process=command(['ps','-p',str(launch['pid']),'-o','pid,stat,args']) if launch else None,
        completed_raw_outputs_or_weights_not_repolled=True)
    if result['status'] and result['status']['status'] in ('RUNNING','FAILED_NO_RETRY'):
        stage=result['status'].get('stage','')
        log=path/(stage+'.raw.log') if not stage.startswith('candidate-development/') else path/'candidate-development/evaluation'/(stage.split('/',1)[1]+'.raw.log')
        result['current_log']=log.read_text()[-6000:] if log.is_file() else None
    return result

launch=read(construction/'launch.json')
print(json.dumps(dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    launch=launch,status=read(construction/'status.json'),
    process=command(['ps','-p',str(launch['pid']),'-o','pid,stat,args']),
    gpu=command(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader']),
    compute_processes=command(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),
    filesystem=command(['df','-B1','/data']),reference_audit=read(construction/'spf-retain95-s0-audit.json'),
    original_full_audit=record(construction/'spf-full-s0-audit.json'),original_freeze=record(construction/'frozen.json'),
    hybrid03_lr5e6_candidate=branch(previous,'spf-hybrid03-lr5e6-full-s0','hybrid03-lr5e6-full-launch-20261012.json',
        ['run_spf_hybrid03_lr5e6_full.py','run_spf_hybrid03_lr5e6_full_queue.py','spf_hybrid03_lr5e6_development_executor.py',
         'spf_anchor_hybrid.py','verify_spf_anchor_hybrid03.py','plan.private.json','bound-snapshot.private.json']),
    previous_hybrid_full_acceptance=record(previous/'hybrid-full-acceptance.private.json'),
    hybrid01_rank40_candidate=branch(candidate,'spf-hybrid01-rank40-full-s0','hybrid01-rank40-full-launch-20261012.json',
        ['run_spf_hybrid01_rank40_full.py','run_spf_hybrid01_rank40_full_queue.py','spf_hybrid01_rank40_development_executor.py',
         'spf_anchor_hybrid.py','verify_spf_anchor_hybrid_rank40.py','plan.private.json','bound-snapshot.private.json']),
    hybrid_engineering_cpu_proof=record(candidate/'hybrid-cpu-proof.private.json'),
    hybrid_engineering_proof=record(candidate/'hybrid-engineering-proof.private.json'),
    hybrid_full_acceptance=record(candidate/'hybrid-full-acceptance.private.json'),
    completed_train_reload_development_raw_or_weight_bytes_not_repeated=True)))
