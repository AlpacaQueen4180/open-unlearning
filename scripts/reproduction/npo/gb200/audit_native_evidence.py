"""Independent native counts/metrics audit using exact integer KS path counting."""
import argparse,bisect,hashlib,json,math
from pathlib import Path

def ks_exact_two_sided(left,right):
    """Count all n+m label paths; exclude paths touching the observed D band."""
    a,b=sorted(left),sorted(right);n,m=len(a),len(b)
    distance=max(abs(bisect.bisect_right(a,x)*m-bisect.bisect_right(b,x)*n) for x in set(a+b))
    if distance==0:return 0.0,1.0
    previous=[0]*(m+1)
    for i in range(n+1):
        row=[0]*(m+1)
        for j in range(m+1):
            if abs(i*m-j*n)>=distance:continue
            row[j]=1 if i==j==0 else (previous[j] if i else 0)+(row[j-1] if j else 0)
        previous=row
    total=math.comb(n+m,n)
    return distance/(n*m),(total-previous[m])/total

def audit(cell,reference,checkpoint_proof):
    state=json.loads((cell/'status.json').read_text());assert state['phase']=='DONE' and not state['smoke_only']
    split=state['forget_split'];size={'forget01':40,'forget05':200,'forget10':400}[split]
    updates={'forget01':20,'forget05':70,'forget10':130}[split]
    eval_dir=cell/Path(state.get('eval_dir','eval')).name
    raw=json.loads((eval_dir/'TOFU_EVAL.json').read_text());summary=json.loads((eval_dir/'TOFU_SUMMARY.json').read_text())
    reference_bytes=reference.read_bytes();assert hashlib.sha256(reference_bytes).hexdigest()==state['assets']['retain_sha256']
    ref=json.loads(reference_bytes)
    scores=lambda data:[v['score'] for v in data['forget_truth_ratio']['value_by_index'].values()]
    left,right=scores(raw),scores(ref);assert len(left)==len(right)==size
    for key in ['forget_Q_A_Prob','forget_Q_A_ROUGE']:
        assert len(raw[key]['value_by_index'])==size
    statistic,pvalue=ks_exact_two_sided(left,right)
    keys=['retain_Q_A_Prob','retain_Q_A_ROUGE','retain_Truth_Ratio','ra_Q_A_Prob_normalised','ra_Q_A_ROUGE','ra_Truth_Ratio','wf_Q_A_Prob_normalised','wf_Q_A_ROUGE','wf_Truth_Ratio']
    utility=[raw[k]['agg_value'] for k in keys];assert all(x>=0 for x in utility)
    computed={'forget_quality':pvalue,'model_utility':0.0 if 0 in utility else len(utility)/sum(1/x for x in utility),'forget_truth_ratio':sum(min(x,1/(x+1e-10)) for x in left)/len(left)}
    deltas={k:abs(summary[k]-v) for k,v in computed.items()};assert max(deltas.values())<1e-12,deltas
    runtime=json.loads((cell/'checkpoint/runtime_final_rank0.json').read_text());trainer=json.loads((cell/'checkpoint/trainer_state.json').read_text());proof=json.loads(checkpoint_proof.read_text())
    assert runtime['trainer_global_step']==runtime['engine_global_steps']==trainer['global_step']==updates
    assert runtime['microsteps']==size//4*10 and runtime['epoch']==trainer['epoch']==10
    assert proof['header_and_required_layers_complete'] and proof['counts']==runtime
    docs=dict(zip(computed,{'forget01':[.92,.56,.66],'forget05':[.14,.45,.70],'forget10':[.02,.46,.70]}[split]))
    result={'model':state['model'],'split':split,'seed':state['seed'],'phase':'DONE','stack':'native TF5.5.4/Accel1.13.0/DS0.15.4; evaluator serialization casts','summary':computed,'ks_D':statistic,'samples':[len(left),len(right)],'independent_deltas':deltas,'ks_method':'two-sided exact integer lattice-path probability','runtime':runtime,'checkpoint_proof':proof,'delta_docs':{k:computed[k]-v for k,v in docs.items()},'reference_sha256':state['assets']['retain_sha256']}
    (cell/'independent-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cell',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--checkpoint-proof',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(audit(args.cell,args.reference,args.checkpoint_proof),indent=2))
