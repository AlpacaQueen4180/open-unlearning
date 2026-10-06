"""Finite, exclusive SPF acceptance/full/reference queue; stop on every failure."""
import argparse,fcntl,json,os,pathlib,subprocess,sys,time

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
    task=pathlib.Path(a.task).resolve();task.mkdir(exist_ok=True)
    lock=(task/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (task/'status.json').exists():raise RuntimeError('Task status exists; inspect and create explicit recovery tag')
    state={'status':'RUNNING','pid':os.getpid(),'started_at':time.time(),'completed':[],'stages':[]}
    atomic(task/'status.json',state)
    repo=task/'repo';py=sys.executable
    env={**os.environ,'PYTHONPATH':str(repo/'src'),'SPF_TASK':str(task),
         'HF_HUB_CACHE':'/data/smart-mfg/jimmy-lin/hf_cache/hub','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','WANDB_DISABLED':'true'}
    def stage(name,cmd,cwd=repo):
        state.update(stage=name,stage_started_at=time.time());atomic(task/'status.json',state)
        row={'name':name,'command':cmd,'cwd':str(cwd),'log':str(task/f'{name}.log'),'started_at':time.time()}
        state['stages'].append(row);atomic(task/'status.json',state)
        with open(row['log'],'w') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=r.returncode,finished_at=time.time());atomic(task/'status.json',state)
        if r.returncode:raise RuntimeError(f'{name} failed ({r.returncode}); preserve evidence')
        state['completed'].append(name);atomic(task/'status.json',state)
    launch=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=1']
    helper='scripts/reproduction/safety/gb200/'
    try:
        recovery=task/'recovery-evidence.json'
        if recovery.exists():
            reused=json.loads(recovery.read_text())
            assert reused['status']=='pass' and reused['fix_scope']=='optimizer metadata only'
            state['reused_stages']=reused['reused_stages'];state['completed'].extend(reused['reused_stages'])
            atomic(task/'status.json',state)
        else:
            stage('setup',[py,str(task/'overlay'/helper/'setup_spf_task.py'),'--task',str(task)],task)
            stage('verify_m0_assets',[py,helper+'verify_m0_assets.py'])
            stage('cpu_tests',[py,'-m','pytest','-q','tests/construction/test_projection.py','tests/construction/test_training.py','tests/construction/test_protocol.py','tests/construction/test_profile.py','tests/construction/test_npo_regression.py'])
        check='scripts/reproduction/safety/check_construction_zero3.py'
        if not recovery.exists():
            stage('tiny_reference',[py,check,'--backend','single','--output',str(task/'tiny-reference')])
        stage('tiny_zero3_world1',launch+[check,'--output',str(task/'tiny-zero3'),'--reference',str(task/'tiny-reference/parameters.pt')])
        stage('production_conflict',launch+['scripts/reproduction/safety/check_construction_full_conflict.py','--manifest',str(task/'prepared.json'),'--output',str(task/'production-conflict')])
        stage('production_smokes',[py,'scripts/reproduction/safety/run_construction_smoke.py','--manifest',str(task/'prepared.json'),'--output',str(task/'smoke'),'--world-size','1'])
        stage('freeze',[py,'-m','construction.cli','freeze','--manifest',str(task/'prepared.json'),'--smoke-summary',str(task/'smoke/summary.json'),'--backend-acceptance',str(task/'tiny-zero3/acceptance.json'),'--output',str(task/'frozen.json')])
        micro=str(json.loads((task/'frozen.json').read_text())['readiness']['microbatch'])
        for split,name in [('full','spf-full-s0'),('retain95','spf-retain95-s0')]:
            target=task/name
            stage(name+'-train',launch+['-m','construction.cli','run','--manifest',str(task/'frozen.json'),'--method','spf','--mode','full','--split',split,'--microbatch',micro,'--weights-only','--output',str(target)])
            stage(name+'-reload',[py,'-m','construction.cli','verify','--checkpoint',str(target)])
            stage(name+'-audit',[py,helper+'audit_spf_target.py','--target',str(target),'--output',str(task/(name+'-audit.json'))])
        state.update(status='CONSTRUCTION_DONE_PENDING_GATE_NPO',finished_at=time.time())
    except BaseException as e:
        state.update(status='FAILED',error=str(e),finished_at=time.time());atomic(task/'status.json',state);raise
    atomic(task/'status.json',state)

if __name__=='__main__':main()
