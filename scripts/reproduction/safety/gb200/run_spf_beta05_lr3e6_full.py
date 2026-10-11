"""New full SPF candidate: beta1=.5 and learning_rate=3e-6; delegate the complete frozen construction loop."""
import argparse,copy,hashlib,io,json,os,sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/data/spf-beta05-lr3e6-full-20261011-r1')
BASE=Path('/data/spf-npo-20261006-r4')
PYTHON='/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
FREEZE='122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
FULL='7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def bound(path,digest):
    raw=Path(path).read_bytes()
    if sha(raw)!=digest:raise ValueError('Source-bound original changed: '+Path(path).name)
    return json.loads(raw)

def save(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())

def candidate_recipe(original):
    recipe=copy.deepcopy(original)
    recipe['args']['adam_beta1']=.5
    recipe['args']['adam_beta2']=.999
    recipe['args']['learning_rate']=3e-6
    if (original['args']['learning_rate']!=1e-5 or original['args']['weight_decay']!=.01
            or original['args']['num_train_epochs']!=5 or original['args'].get('adam_beta1',.9)!=.9
            or original['args'].get('adam_beta2',.999)!=.999):
        raise ValueError('Original full SPF recipe required')
    return recipe

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-sha256','plan-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if sys.executable!=PYTHON or os.name!='posix' or sha(Path(__file__).read_bytes())!=a.source_sha256:
        raise ValueError('Exact new source and original Linux runtime required')
    plan=bound(ROOT/'validation-code/plan.private.json',a.plan_sha256)
    manifest=bound(BASE/'frozen.json',FREEZE);bound(BASE/'spf-full-s0-audit.json',FULL)
    if plan['parameter_delta']!={'adam_beta1':{'baseline':.9,'candidate':.5},'learning_rate':{'baseline':1e-5,'candidate':3e-6}} or plan['new_paid_judge_calls']!=0:
        raise ValueError('Exact beta1 and lower learning rate relative to original SPF')
    for name,digest in manifest['construction_code'].items():
        if sha((BASE/'repo'/name).read_bytes())!=digest:raise ValueError('Frozen source changed: '+name)
    if (ROOT/'spf-beta05-lr3e6-full-s0').exists():raise FileExistsError('New candidate must be absent')
    baseline=bound(BASE/'spf-full-s0/resolved_training.json','d3aeba5342e5102124d7a12096cb4190c56c9617dabb75ed1df5a3b155bc354d')
    if baseline['args']['adam_beta1']!=.9 or baseline['args']['adam_beta2']!=.999:raise ValueError('Actual baseline betas')
    sys.path.insert(0,str(BASE/'repo/src'))
    import construction.cli as cli
    original_load=cli.load_recipe
    original=copy.deepcopy(original_load('spf'));candidate=candidate_recipe(original)
    def load_recipe(method):
        if method!='spf':raise ValueError('Only pure SPF')
        actual=original_load(method)
        if actual!=original:raise ValueError('Original recipe drift')
        return candidate_recipe(actual)
    cli.load_recipe=load_recipe
    args=SimpleNamespace(manifest=str(BASE/'frozen.json'),method='spf',mode='full',split='full',
       microbatch=4,weights_only=True,output=str(ROOT/'spf-beta05-lr3e6-full-s0'),dry_run=True)
    capture=io.StringIO()
    with redirect_stdout(capture):cli.run(args)
    dry=json.loads(capture.getvalue());actual=dry['recipe']
    expected=copy.deepcopy(original)
    expected['args'].update(actual['args'])
    # Check every original supplied field except the explicit beta1 and original
    # runtime output/batch/seed fields assigned by the unmodified CLI.
    runtime_fields={'output_dir','per_device_train_batch_size','gradient_accumulation_steps','seed','adam_beta1','adam_beta2','learning_rate'}
    for key,value in original['args'].items():
        if key not in runtime_fields and actual['args'][key]!=value:raise ValueError('Unrelated recipe change: '+key)
    if (actual['args']['adam_beta1']!=.5 or actual['args']['adam_beta2']!=.999 or actual['args']['learning_rate']!=3e-6
         or actual['args']['gradient_accumulation_steps']!=8 or actual['args']['per_device_train_batch_size']!=4
         or actual['args']['seed']!=0 or actual['method_args']['save_intermediate'] is not True
         or actual['method_args']['save_runtime'] is not False):raise ValueError('Full candidate counters/config')
    import trainer
    original_trainer=trainer.load_trainer
    def load_trainer(*args,**kwargs):
        result=original_trainer(*args,**kwargs);tr=result[0]
        original_setup=tr._setup
        def setup(*args,**kwargs):
            value=original_setup(*args,**kwargs)
            if tuple(tr.optimizer_metadata['defaults']['betas'])!=(.5,.999):raise ValueError('Actual optimizer beta mismatch')
            if tr.args.learning_rate!=3e-6 or tr.optimizer_metadata['defaults']['lr']!=3e-6:raise ValueError('Actual lower learning rate required')
            actual_args=json.loads(json.dumps(tr.args.to_dict(),default=str))
            # HF serializes hub_token with a redaction placeholder. Compare all
            # other actual TrainingArguments to the successful baseline.
            excluded={'output_dir','adam_beta1','learning_rate','hub_token'}
            diffs={k:(baseline['args'].get(k),v) for k,v in actual_args.items() if k not in excluded and baseline['args'].get(k)!=v}
            if diffs:raise ValueError('Unexpected actual TrainingArguments delta: '+','.join(sorted(diffs)))
            save(ROOT/'optimizer-binding.private.json',dict(status='ACTUAL_BETA05_LR3E6_FULL_OPTIMIZER_BOUND',
                beta1=.5,beta2=.999,learning_rate=3e-6,optimizer=tr.optimizer_metadata,baseline_resolved_sha256=plan['baseline_resolved_sha256'],
                plan_sha256=a.plan_sha256,wrapper_sha256=a.source_sha256,frozen_cli_sha256=manifest['construction_code']['src/construction/cli.py'],
                original_setup_delegated=True,original_training_loop_delegated=True,actual_nonprofile_training_args_match=True,
                comparison_serialization='same json.dumps(default=str) as original resolved_training',pid=os.getpid()))
            return value
        tr._setup=setup
        return result
    trainer.load_trainer=load_trainer
    save(ROOT/'resolved-candidate-plan.private.json',dict(original_recipe=original,candidate_recipe=candidate,
         original_cli_dry_run=dry,plan_sha256=a.plan_sha256,source_sha256=a.source_sha256,
         parent_frozen_manifest_sha256=FREEZE,parameter_delta=plan['parameter_delta'],resumes_old_weights=False))
    args.dry_run=False
    try:cli.run(args)
    finally:cli.load_recipe=original_load;trainer.load_trainer=original_trainer

if __name__=='__main__':main()
