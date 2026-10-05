"""Finite serial seed-zero matrix; persistent state, no training polling."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_experiment import ROOT, REPO, write_json

MODELS = ['Llama-2-7b-chat-hf', 'Llama-3.2-1B-Instruct']
TARGETS = [[(.4,.58,.65),(.09,.53,.71),(.42,.54,.73)],[(.92,.56,.66),(.14,.45,.70),(.02,.46,.70)]]

def main():
    lock = ROOT / 'matrix.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    state = {'phase':'PREPARING','pid':os.getpid(),'cells':[], 'source_sha256':hashlib.sha256((REPO/'docs/repro.md').read_bytes()).hexdigest(), 'seed':0,'global_batch':32}
    status = ROOT/'matrix-status.json'
    write_json(status,state)
    try:
        subprocess.run([sys.executable,'-m','pip','install','--ignore-installed','--no-deps','bitsandbytes==0.50.2'],check=True,env={**os.environ,'PIP_CONFIG_FILE':'/dev/null','PIP_INDEX_URL':'https://pypi.org/simple','PIP_EXTRA_INDEX_URL':''})
        subprocess.run([sys.executable,'-c','import torch, bitsandbytes, flash_attn, transformers, deepspeed; print(torch.__version__, bitsandbytes.__version__)'],check=True)
        for mi,model in enumerate(MODELS):
            for si,split in enumerate(['forget01','forget05','forget10']):
                tag = f'matrix_{model}_{split}'
                name = 'gb200_1gpu_zero3_flash_attention_2_s0_' + ('main' if mi == 0 and si == 1 else tag)
                run = ROOT/'runs'/name
                cell = {'model':model,'split':split,'run':name,'docs_repro':dict(zip(['forget_quality','model_utility','forget_truth_ratio'],TARGETS[mi][si]))}
                state['cells'].append(cell); state.update(phase='RUNNING',current=name)
                write_json(status,state)
                if not run.exists():
                    subprocess.run([sys.executable,str(REPO/'scripts/reproduction/npo/gb200/run_experiment.py'),'--world-size','1','--stages','3','--attention','flash_attention_2','--model',model,'--forget',split,'--tag',tag],check=True)
                run_state=json.loads((run/'status.json').read_text())
                if run_state['phase'] != 'DONE':
                    raise RuntimeError(f'Preserved incomplete run {name}: {run_state["phase"]}; repair before retry')
                import numpy as np
                from scipy.stats import ks_2samp,hmean
                raw=json.loads((run/'eval/TOFU_EVAL.json').read_text())
                summary=json.loads((run/'eval/TOFU_SUMMARY.json').read_text())
                assets=run_state['assets']
                reference_file=assets.get('retain_file','tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json')
                reference_path=ROOT/'hf/hub/datasets--open-unlearning--eval/snapshots'/assets['eval_revision']/reference_file
                assert hashlib.sha256(reference_path.read_bytes()).hexdigest()==assets['retain_sha256']
                reference=json.loads(reference_path.read_text())
                a=np.array([v['score'] for v in raw['forget_truth_ratio']['value_by_index'].values()])
                b=np.array([v['score'] for v in reference['forget_truth_ratio']['value_by_index'].values()])
                ks=ks_2samp(a,b)
                keys=['retain_Q_A_Prob','retain_Q_A_ROUGE','retain_Truth_Ratio','ra_Q_A_Prob_normalised','ra_Q_A_ROUGE','ra_Truth_Ratio','wf_Q_A_Prob_normalised','wf_Q_A_ROUGE','wf_Truth_Ratio']
                recomputed={'forget_quality':float(ks.pvalue),'model_utility':float(hmean([raw[k]['agg_value'] for k in keys])),'forget_truth_ratio':float(np.minimum(a,1/(a+1e-10)).mean())}
                assert max(abs(summary[k]-v) for k,v in recomputed.items())<1e-12
                checkpoint=run/'checkpoint'
                index=checkpoint/'model.safetensors.index.json'
                shards=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
                assert all((checkpoint/s).is_file() and (checkpoint/s).stat().st_size>0 for s in shards)
                cell.update(phase='DONE',summary=recomputed,ks_D=float(ks.statistic),samples=[len(a),len(b)],delta_docs={k:recomputed[k]-v for k,v in cell['docs_repro'].items()},runtime=json.loads((checkpoint/'runtime_final_rank0.json').read_text()),checkpoint_shards={s:(checkpoint/s).stat().st_size for s in shards})
                write_json(status,state)
        state.update(phase='DONE',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception:
        state.update(phase='FAILED',error=traceback.format_exc())
        raise
    finally:
        write_json(status,state)
        lock.unlink()

if __name__=='__main__': main()
