"""Independent score, checkpoint and update audit for all follow-up cells."""
import hashlib
import json
import numpy as np
from scipy.stats import ks_2samp, hmean
from run_experiment import ROOT

def audit_cell(cell):
    run = ROOT/'runs'/cell['run']
    state=json.loads((run/'status.json').read_text())
    assert state['phase']=='DONE' and not state['smoke_only']
    raw=json.loads((run/'eval/TOFU_EVAL.json').read_text())
    summary=json.loads((run/'eval/TOFU_SUMMARY.json').read_text())
    a=state['assets']
    ref=ROOT/'hf/hub/datasets--open-unlearning--eval/snapshots'/a['eval_revision']/a.get('retain_file','tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json')
    assert hashlib.sha256(ref.read_bytes()).hexdigest()==a['retain_sha256']
    reference=json.loads(ref.read_text())
    scores=lambda x: np.array([v['score'] for v in x['forget_truth_ratio']['value_by_index'].values()])
    observed,expected=scores(raw),scores(reference)
    assert len(observed)==len(expected)=={'forget01':40,'forget05':200,'forget10':400}[cell['split']]
    ks=ks_2samp(observed,expected)
    keys=['retain_Q_A_Prob','retain_Q_A_ROUGE','retain_Truth_Ratio','ra_Q_A_Prob_normalised','ra_Q_A_ROUGE','ra_Truth_Ratio','wf_Q_A_Prob_normalised','wf_Q_A_ROUGE','wf_Truth_Ratio']
    recomputed={'forget_quality':float(ks.pvalue),'model_utility':float(hmean([raw[k]['agg_value'] for k in keys])),'forget_truth_ratio':float(np.minimum(observed,1/(observed+1e-10)).mean())}
    deltas={k:abs(summary[k]-v) for k,v in recomputed.items()}
    assert max(deltas.values())<1e-12
    checkpoint=run/'checkpoint'; index=checkpoint/'model.safetensors.index.json'
    shards=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    assert all((checkpoint/s).is_file() and (checkpoint/s).stat().st_size>0 for s in shards)
    runtime=json.loads((checkpoint/'runtime_final_rank0.json').read_text())
    trainer=json.loads((checkpoint/'trainer_state.json').read_text())
    assert runtime['trainer_global_step']==trainer['global_step'] and runtime['epoch']==trainer['epoch']
    if cell['corrected']:
        updates={'forget01':20,'forget05':70,'forget10':130}[cell['split']]
        assert runtime['trainer_global_step']==runtime['engine_global_steps']==runtime['forced_update_boundaries']==updates
        assert runtime['epoch']==10 and runtime['microsteps']=={'forget01':100,'forget05':500,'forget10':1000}[cell['split']]
    cell.update(phase='DONE',summary=recomputed,independent_deltas=deltas,ks_D=float(ks.statistic),samples=[len(observed),len(expected)],runtime=runtime,delta_docs={k:recomputed[k]-v for k,v in cell['docs_repro'].items()},checkpoint_shards={s:(checkpoint/s).stat().st_size for s in shards})
    return cell
