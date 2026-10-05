"""Ten finite serial runs, gated on an actual partial-group corrected smoke."""
import datetime
import json
import os
import subprocess
import sys
import traceback
from run_experiment import ROOT, REPO, write_json
from run_matrix import MODELS, TARGETS
from audit_cell import audit_cell

PREFIX='gb200_1gpu_zero3_flash_attention_2'

def command(model,split,seed,tag,corrected,smoke=0):
    cmd=[sys.executable,str(REPO/'scripts/reproduction/npo/gb200/run_experiment.py'),'--world-size','1','--stages','3','--attention','flash_attention_2','--model',model,'--forget',split,'--seed',str(seed),'--tag',tag]
    if corrected:cmd+=['--corrected']
    if smoke:cmd+=['--smoke-steps',str(smoke)]
    return cmd

def main():
    lock=ROOT/'followup.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    assert not (ROOT/'matrix.lock').exists(), 'original queue still running'
    status=ROOT/'followup-status.json'
    old=json.loads(status.read_text()) if status.exists() else {}
    state={'phase':'PREFLIGHT','pid':os.getpid(),'cells':old.get('cells',[]),'baseline_matrix':str(ROOT/'matrix-status.json'),'policy':'ceil_epoch_and_sync_ds_boundary_v1','loss_scaling':'unchanged pinned NPO/Trainer behavior'}
    write_json(status,state)
    try:
        smoke_name=f'{PREFIX}_s0_boundarycheck_Llama-3.2-1B-Instruct_forget01_smoke4_corrected'
        smoke=ROOT/'runs'/smoke_name
        state.update(current=smoke_name);write_json(status,state)
        if not smoke.exists():subprocess.run(command(MODELS[1],'forget01',0,'boundarycheck_Llama-3.2-1B-Instruct_forget01',True,4),check=True)
        smoke_status=json.loads((smoke/'status.json').read_text())
        assert smoke_status['phase']=='SMOKE_PASSED'
        proof=json.loads((smoke/'checkpoint/runtime_final_rank0.json').read_text())
        assert proof['trainer_global_step']==proof['engine_global_steps']==proof['forced_update_boundaries']==4 and proof['microsteps']==20 and proof['epoch']==2,proof
        state['smoke_proof']=proof
        jobs=[(mi,si,0,True) for mi in range(2) for si in range(3)] + [(mi,0,seed,False) for mi in range(2) for seed in [1,2]]
        for mi,si,seed,corrected in jobs:
            model=MODELS[mi];split=['forget01','forget05','forget10'][si]
            tag=f'followup_{model}_{split}'
            name=f'{PREFIX}_s{seed}_{tag}'+('_corrected' if corrected else '')
            cell=next((c for c in state['cells'] if c['run']==name),None)
            if cell is None:
                cell={'model':model,'split':split,'seed':seed,'corrected':corrected,'run':name,'docs_repro':dict(zip(['forget_quality','model_utility','forget_truth_ratio'],TARGETS[mi][si]))};state['cells'].append(cell)
            state.update(phase='RUNNING',current=name);write_json(status,state)
            if not (ROOT/'runs'/name).exists():subprocess.run(command(model,split,seed,tag,corrected),check=True)
            audit_cell(cell)
            baseline=next(c for c in json.loads((ROOT/'matrix-status.json').read_text())['cells'] if c['model']==model and c['split']==split)
            cell['baseline_seed0']=baseline['summary']
            cell['delta_original_seed0']={k:v-baseline['summary'][k] for k,v in cell['summary'].items()}
            write_json(status,state)
        state.update(phase='DONE',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception:
        state.update(phase='FAILED',error=traceback.format_exc());raise
    finally:
        write_json(status,state);lock.unlink()

if __name__=='__main__':main()
