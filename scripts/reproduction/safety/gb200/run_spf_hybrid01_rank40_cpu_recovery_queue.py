"""Exclusive lower-learning-rate SPF candidate after audited beta1=.5 Judge result.

No old run is retried. All paid Judge work stays outside this GPU queue.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/spf-hybrid01-rank40-full-20261012-r1')
JOURNAL = ROOT/'runs/cpu-fixture-recovery-1539'
ORIGINAL = Path('/data/spf-development-20261007-r1')
CONSTRUCTION = Path('/data/spf-npo-20261006-r4')
PYTHON = '/data/npo-gb200-20261004/validation-20261006/venv/bin/python'
FREEZE = '122edbe2c09fcd2903e5b8949657f3dcaf3bf5cbc527e05b4cce91a29fb04d9e'
FULL = '7a7036b903ced64e18307e90523cc7ada73699ddb3d79498bfd4063f86ed8cf9'
ADAPTER = '71c3c240ab6418a140fb34dca9fb2892ba61f079c9a7f46c0c3806c15dc963de'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def bound(path,digest):
    raw=Path(path).read_bytes()
    if sha(raw)!=digest: raise ValueError('Bound file changed: '+Path(path).name)
    return json.loads(raw)
def write(path,value):
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def validate_plan(plan):
    expected={'parameter_delta': {'adam_beta1': {'baseline': 0.9, 'candidate': 0.5}, 'learning_rate': {'baseline': 1e-05, 'candidate': 5e-06}, 'safety_loss_coefficient': {'baseline': 0.0, 'candidate': 0.1}, 'projection_rank': {'baseline': 20, 'candidate': 40}}, 'new_paid_judge_calls': 0, 'retry_count': 0, 'existing_training_repeated': False, 'formal_hexphi_selection': False, 'updates': 625, 'epochs': 5, 'examples': 20000, 'unique_examples': 4000, 'seed': 0, 'method_family': 'spf_anchor_loss_hybrid', 'safety_loss_coefficient': 0.1, 'projection_rank': 40, 'learning_rate': 5e-06, 'beta2': 0.999, 'weight_decay': 0.01, 'microbatch': 4, 'accumulation': 8, 'global_batch': 32, 'warmup_steps': 125}
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Exact lambda.1 rank40 candidate required')

def validate_completed_predecessor(snapshot):
    branch=snapshot.get('hybrid03_lr5e6_candidate') or {};status=branch.get('status') or {}
    if (status.get('pid')!=16733 or status.get('status')!='HYBRID03_LR5E6_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
        or status.get('completed')!=['hybrid-cpu-verification','hybrid-engineering-train','hybrid-engineering-reload','hybrid-engineering-verify','train','fresh-reload','full-byte-audit','candidate-development']
        or (branch.get('candidate_audit') or {}).get('sha256')!='d27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486'
        or (snapshot.get('previous_hybrid_full_acceptance') or {}).get('sha256')!='ea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53'
        or branch.get('checkpoint_exists') is not True):raise ValueError('Actual complete predecessor required')
    rows=status.get('stages',[])
    if len(rows)!=7 or any(row.get('returncode')!=0 for row in rows):raise ValueError('All predecessor engineering/train stages complete')
    process=branch.get('process') or {};lines=process.get('stdout','').strip().splitlines()
    if process.get('stderr','').strip() or process.get('returncode') not in (0,1) or not lines or lines[0].split()[:2]!=['PID','STAT']:raise ValueError('Unambiguous process state')
    if not ((process['returncode']==1 and len(lines)==1) or (process['returncode']==0 and len(lines)==2 and lines[1].split()[0]=='16733' and lines[1].split()[1].startswith('Z'))):raise ValueError('Predecessor must exit')
    branch=snapshot.get('hybrid01_rank40_candidate') or {}
    failed=branch.get('status') or {}
    if branch.get('new_root_exists') is not True or (branch.get('launch') or {}).get('pid')!=19460 or failed.get('pid')!=19460 or failed.get('status')!='FAILED_NO_RETRY' or failed.get('stage')!='hybrid-cpu-verification' or failed.get('completed')!=[] or branch.get('checkpoint_exists') is not False:raise ValueError('Exact original CPU-only failure required')
    if (snapshot.get('primary_failure_status_record') or {}).get('sha256')!='2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25':raise ValueError('Original immutable failed status')
    recovery=snapshot.get('rank40_cpu_fixture_recovery') or {}
    if recovery.get('journal_exists') is not False or recovery.get('launch') is not None or recovery.get('status') is not None:raise ValueError('New recovery must be absent')
    if any(snapshot.get(name) is not None for name in ('hybrid_engineering_proof','hybrid_full_acceptance')):raise ValueError('No completed engineering or full training may be repeated')

def validate_search_decision(plan):
    expected={'previous_model_identity_sha256': '4ab6af98cf33e20e50394530d2fe6de69aa9e985188cd23cbfefd9a4f002d33d', 'previous_judge_audit_sha256': '7c1a87cefbec6e1e65e30154c4a81d9642311a9182d311934fc02f30e9184d30', 'previous_harmful_assistance': 33, 'harmful_denominator': 300, 'mixing_harmful': 13, 'campaign_authorization_sha256': 'd29a89e81d76edd7bc4571ec6c4d1c42322bd1749a703d94c7cc2ac1d231e283', 'previous_complete_model_count': 5, 'model_limit': 50, 'development_used_as_training': False, 'selected_control_identity_sha256': '7c3636e230dc3c53ee824572488191ae103f5c5fa4f85af28eb7b79a4ef14962', 'selected_control_judge_audit_sha256': 'a6ca483d3e4dd34944d93be18460596c8c944c206a13244ade5ef52c6e38cf89', 'selected_control_learning_rate': 5e-06, 'controlled_delta_from_selected_control': {'projection_rank': {'baseline': 20, 'candidate': 40}}, 'selected_control_safety_loss_coefficient': 0.1, 'selected_control_projection_rank': 20, 'selected_control_harmful_assistance': 27}
    if plan.get('search_decision')!=expected:raise ValueError('Audited best lambda.1 control and rank20to40 decision required')

def validate_predecessor_runtime():
    previous=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')
    if sha((previous/'spf-hybrid03-lr5e6-full-s0-audit.json').read_bytes())!='d27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486':raise ValueError('Predecessor audit binding')
    if sha((previous/'hybrid-full-acceptance.private.json').read_bytes())!='ea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53':raise ValueError('Predecessor acceptance binding')
    for pid in (16733,16736,16947,17483,17763,17766,17834,18362,18642,18645,18802,18939,19076,19213,19348):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':raise ValueError('All predecessor processes must exit')


def validate_cpu_proof(proof):
    expected=dict(status='HYBRID_CPU_DIRECT_AUTOGRAD_TOKEN_MEAN_TWO_ADAM_UPDATES_RELOAD_PASS',actual_cpu_updates=2,
        spf_projection_rank=40,safety_loss_coefficient=.1,loss_oracle_dtype='torch.float64',parameter_gradient_projection_dtype='torch.float32',
        original_tolerances_unchanged=True,verifier_sha256='a998a42a44109da78c1789478881f1da8fc5753b53788844141b4bc5e65b447b',
        helper_sha256='fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc',execution_mode='in_memory_exact_source_bound_cpu_function_only',
        verifier_file_deployed=False,cuda_initialized=False,hf_weights_read=False,actual8b_training=0,api_calls=0,credential_reads=0,
        unequal_microbatch_assistant_tokens=[56,16],safety_assistant_tokens=56)
    if any(proof.get(k)!=v for k,v in expected.items()):raise ValueError('Exact successfully checked CPU source and scope required')
    if proof.get('maximum_gradient_error')!=1.4901161193847656e-8 or proof.get('maximum_update_error')!=7.450580596923828e-9:raise ValueError('Actual CPU errors required')
    expected_pins={'src/construction/backend.py':'c3d009890289f94f10a8d795d1ed035046e290cb10378f426a43d93738b59e74','src/construction/projection.py':'7776c68373ef7edc19bc2f9b7d4a6900f154c97be0c215517b71202c98ea643a','src/construction/numerics.py':'6fdbb14da03eddc0371203003cef24e2c2c29b873c27827f9677bf1dae9cd164'}
    if proof.get('frozen_dependency_sha256')!=expected_pins:raise ValueError('Original CPU dependencies required')

def validate_failed_runtime(plan):
    if plan.get('recovery_contract')!={'original_failed_status_sha256':'2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25','original_cpu_raw_sha256':'b833452612cefcf2fd97b1f6fec007c058db761e41eae63f5f61f7f08d5cb263','original_plan_sha256':'ef346727c0948dcdb9ce92832b33e455683422bdee2612b5f106b1a5bfbd3829','reused_cpu_proof_sha256':'5397aabbfcaa4d70b1b3f51004721d6d6f389212301ced815472c890bbfb6c4d','original_failure_archive_sha256':'3f5c58114b262673e8f54d93b14eec2adcff57d75186908eb0ef26fc9a0d346d','recovery_journal':'/data/spf-hybrid01-rank40-full-20261012-r1/runs/cpu-fixture-recovery-1539','completed_cpu_case_repeated':False}:raise ValueError('Exact unfinished-stage recovery contract')
    failed=bound(ROOT/'status.json',plan['recovery_contract']['original_failed_status_sha256'])
    original_plan=bound(ROOT/'validation-code/plan.private.json',plan['recovery_contract']['original_plan_sha256'])
    for path,digest in original_plan['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('Original failed sources preserved')
    command=[PYTHON,str(ROOT/'validation-code/verify_spf_anchor_hybrid_rank40.py'),'--helper',str(ROOT/'validation-code/spf_anchor_hybrid.py'),'--helper-sha256','fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc','--output',str(ROOT/'hybrid-cpu-proof.private.json')]
    rows=failed.get('stages',[])
    if failed.get('pid')!=19460 or failed.get('status')!='FAILED_NO_RETRY' or failed.get('completed')!=[] or len(rows)!=1 or rows[0].get('process_pid')!=19463 or rows[0].get('command')!=command or rows[0].get('returncode')!=1:raise ValueError('Original failed CPU exact command and PID')
    if sha((ROOT/'hybrid-cpu-verification.raw.log').read_bytes())!=plan['recovery_contract']['original_cpu_raw_sha256']:raise ValueError('Original failed raw preserved')
    launch=json.loads((ROOT/'validation-code/hybrid01-rank40-full-launch-20261012.json').read_bytes())
    if launch.get('pid')!=19460 or launch.get('plan_sha256')!=plan['recovery_contract']['original_plan_sha256']:raise ValueError('Original launch binding')
    for pid in (19460,19463):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':raise ValueError('Original failed processes must exit')
    forbidden=('engineering-spf-hybrid01-rank40-s0','spf-hybrid01-rank40-full-s0','candidate-development','engineering-optimizer-binding.private.json','optimizer-binding.private.json','engineering-resolved-plan.private.json','resolved-candidate-plan.private.json','hybrid-engineering-proof.private.json','hybrid-full-acceptance.private.json','spf-hybrid01-rank40-full-s0-audit.json','hybrid-cpu-proof.private.json')
    if JOURNAL.exists() or any((ROOT/name).exists() for name in forbidden):raise FileExistsError('Only proved-unfinished stages may run')
    proof=bound(ROOT/'validation-code/rank40-cpu-oracle-proof-1539.private.json',plan['recovery_contract']['reused_cpu_proof_sha256']);validate_cpu_proof(proof)

def metadata_check(case):
    target=Path(case['checkpoint'])
    if target!=ROOT/'spf-hybrid01-rank40-full-s0' or case['step']!=625:raise ValueError('New completed candidate only')
    audit=bound(ROOT/'spf-hybrid01-rank40-full-s0-audit.json',case['candidate_audit_sha256'])
    if audit['status']!='pass' or audit['actual_updates']!=625 or audit['actual_examples']!=20000 or audit['fresh_reload']['status']!='pass':raise ValueError('Complete new actual audit/reload required')
    if case['weights_sha256']!=audit['final_checkpoint']['files'][0]['sha256']:raise ValueError('New complete audit weights required')
    if sorted(f.name for f in target.glob('*.safetensors'))!=['model.safetensors'] or (target/'model.safetensors').stat().st_size!=16060556616:raise ValueError('Exact new shard metadata')
    for name,digest in case['metadata_sha256'].items():bound(target/name,digest)
    if json.loads((target/'trainer_state.json').read_bytes())['global_step']!=625:raise ValueError('Full candidate counter')
    identity={k:v for k,v in case.items() if k!='identity_sha256'}
    if sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())!=case['identity_sha256']:raise ValueError('New candidate identity required')

def prepare_case(case,executor_plan):
    metadata_check(case)
    case_dir=ROOT/'candidate-development'
    case_dir.mkdir(exist_ok=False)
    for name,target in [('repo',ORIGINAL/'repo'),('bundle',ORIGINAL/'bundle')]:
        (case_dir/name).symlink_to(target,target_is_directory=True)
    old=ORIGINAL/'adapter-generation-api-r2'
    old_audit=bound(old/'adapter-audit.json',ADAPTER)
    audit=json.loads(json.dumps(old_audit))
    code=case_dir/'adapter-generation-api-r2/code';code.mkdir(parents=True)
    old_target=old_audit['checkpoint'];old_identity=old_audit['checkpoint_identity_sha256']
    for name,spec in old_audit['sources'].items():
        raw=(old/'code'/name).read_bytes()
        if sha(raw)!=spec['adapter_sha256']: raise ValueError('Original completed adapter drift')
        text=raw.decode('utf8')
        if name!='score_development_ifbench.py':
            if text.count(old_target)!=2 or text.count(old_identity)!=1:
                raise ValueError('Exact two loader/metadata paths and one identity edit required')
            text=text.replace(old_target,case['checkpoint']).replace(old_identity,case['identity_sha256'])
            restored=text.replace(case['checkpoint'],old_target).replace(case['identity_sha256'],old_identity)
            if restored.encode('utf8')!=raw: raise ValueError('Nonidentity adapter change')
        new=text.encode('utf8');compile(new,name,'exec');(code/name).write_bytes(new)
        audit['sources'][name].update(previous_adapter_sha256=sha(raw),adapter_sha256=sha(new),
                                    retarget_only_inverse_bytes_verified=True)
    helper=(old/'code/spf_development_generation.py').read_bytes()
    if sha(helper)!=audit['generation_helper_sha256']: raise ValueError('Generation helper changed')
    (code/'spf_development_generation.py').write_bytes(helper)
    audit.update(checkpoint=case['checkpoint'],checkpoint_identity_sha256=case['identity_sha256'],
                 checkpoint_files={**case['metadata_sha256'],'model.safetensors':case['weights_sha256']},
                 trajectory_step=case['step'],original_full_audit_sha256=FULL,
                 weight_byte_hash_verification='reuse newly completed rank40 full byte audit; no repeated weight hashing',
                 candidate_audit_sha256=case['candidate_audit_sha256'],parameter_delta={'adam_beta1': {'baseline': 0.9, 'candidate': 0.5}, 'learning_rate': {'baseline': 1e-05, 'candidate': 5e-06}, 'safety_loss_coefficient': {'baseline': 0.0, 'candidate': 0.1}, 'projection_rank': {'baseline': 20, 'candidate': 40}},
                 intermediate_fresh_reload_numerical_probe_available=False)
    audit_path=code.parent/'adapter-audit.json'
    audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    os.environ.update(SPF_DIAG_CASE_TASK=str(case_dir),SPF_DIAG_IDENTITY=case['identity_sha256'],
                      SPF_DIAG_ADAPTER_SHA=sha(audit_path.read_bytes()))
    m=module(ROOT/'validation-code/spf_hybrid01_rank40_development_executor.py','checkpoint_executor_'+str(case['step']))
    m.validate_dependencies(case_dir/'repo',bound(CONSTRUCTION/'frozen.json',FREEZE))
    for name,digest,count in m.DATA.values():
        if len(bound(case_dir/'bundle/assets'/name,digest))!=count: raise ValueError('Fixed development input mismatch')
    return m,case_dir,audit

def stage(state,name,command,env,guard,cwd=CONSTRUCTION/'repo'):
    guard.gpu_idle(guard.command_result(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']))
    row=dict(name=name,command=command,cwd=str(cwd),environment_overrides=env,
             log=str(JOURNAL/(name+'.raw.log')),started_at=time.time())
    state['stage']=name;state['stages'].append(row);write(JOURNAL/'status.json',state)
    with Path(row['log']).open('xb') as f:
        proc=subprocess.Popen(command,cwd=cwd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
        row['process_pid']=proc.pid;write(JOURNAL/'status.json',state);row['returncode']=proc.wait()
    row.update(finished_at=time.time(),raw_log_sha256=sha(Path(row['log']).read_bytes()))
    write(JOURNAL/'status.json',state)
    if row['returncode']:raise RuntimeError('New stage failed; no retry: '+name)
    state['completed'].append(name);write(JOURNAL/'status.json',state)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','snapshot','snapshot-sha256','source-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if sys.executable!=PYTHON or os.name!='posix' or sha(Path(__file__).read_bytes())!=a.source_sha256:raise ValueError('Original runtime and new controller')
    plan=bound(a.plan,a.plan_sha256);validate_plan(plan)
    for path,digest in plan['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('Bound source drift: '+Path(path).name)
    guard=module(ORIGINAL/'validation-code/run_spf_development_recovery_guards.py','original_completion_guard')
    snapshot=bound(a.snapshot,a.snapshot_sha256);guard.validate_snapshot(snapshot,datetime.now(timezone.utc))
    validate_completed_predecessor(snapshot);validate_search_decision(plan);validate_predecessor_runtime();validate_failed_runtime(plan)
    bound(CONSTRUCTION/'spf-full-s0-audit.json',FULL);bound(CONSTRUCTION/'frozen.json',FREEZE)
    import fcntl
    with (ORIGINAL/'development-cuda.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_failed_runtime(plan)
        JOURNAL.mkdir(parents=True,exist_ok=False)
        state=dict(status='RUNNING',pid=os.getpid(),started_at=time.time(),completed=[],stages=[],plan_sha256=a.plan_sha256,
            source_sha256=a.source_sha256,parameter_delta=plan['parameter_delta'],new_paid_judge_calls=0,method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1,target_gate='insufficient_evidence')
        write(JOURNAL/'status.json',state)
        env=dict(HF_HOME='/data/smart-mfg/jimmy-lin/hf_cache/hub',HF_HUB_CACHE='/data/smart-mfg/jimmy-lin/hf_cache/hub',
            HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',PYTHONPATH=str(CONSTRUCTION/'repo/src'),
            TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
        target=ROOT/'spf-hybrid01-rank40-full-s0'
        try:
            train=ROOT/'validation-code/run_spf_hybrid01_rank40_cpu_recovery.py'
            helper=ROOT/'validation-code/spf_anchor_hybrid.py'
            verifier=ROOT/'validation-code/verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py'
            verify_command=[PYTHON,str(verifier),'--helper',str(helper),'--helper-sha256',plan['source_sha256'][str(helper)]]
            state['completed'].append('hybrid-cpu-verification-reused')
            state['stages'].append(dict(name='hybrid-cpu-verification-reused',returncode=0,reused=True,process_pid=None,proof_sha256=plan['recovery_contract']['reused_cpu_proof_sha256'],new_cpu_processes=0))
            write(JOURNAL/'status.json',state)
            stage(state,'hybrid-engineering-train',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',str(train),
                '--source-sha256',plan['source_sha256'][str(train)],'--plan-sha256',a.plan_sha256,'--plan-path',a.plan,'--engineering'],env,guard)
            stage(state,'hybrid-engineering-reload',[PYTHON,'-m','construction.cli','verify','--checkpoint',str(ROOT/'engineering-spf-hybrid01-rank40-s0')],env,guard)
            stage(state,'hybrid-engineering-verify',verify_command+['--engineering-target',str(ROOT/'engineering-spf-hybrid01-rank40-s0'),
                '--output',str(ROOT/'hybrid-engineering-proof.private.json')],{**env,'CUDA_VISIBLE_DEVICES':''},guard)
            stage(state,'train',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',str(train),
                '--source-sha256',plan['source_sha256'][str(train)],'--plan-sha256',a.plan_sha256,'--plan-path',a.plan],env,guard)
            binding=json.loads((ROOT/'optimizer-binding.private.json').read_bytes())
            if binding['hybrid']['method_family']!='spf_anchor_loss_hybrid' or binding['hybrid']['safety_loss_coefficient']!=.1 or binding['engineering'] or binding['spf_projection_rank']!=40 or not binding['actual_nonprofile_method_args_match']:raise ValueError('Actual hybrid binding required')
            if binding['status']!='ACTUAL_HYBRID01_RANK40_FULL_OPTIMIZER_BOUND' or binding['beta1']!=.5 or binding['beta2']!=.999 or binding['learning_rate']!=5e-6 or not binding['actual_nonprofile_training_args_match']:raise ValueError('Actual optimizer binding missing')
            stage(state,'fresh-reload',[PYTHON,'-m','construction.cli','verify','--checkpoint',str(target)],env,guard)
            stage(state,'full-byte-audit',[PYTHON,str(Path('/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py')),'--target',str(target),'--output',str(ROOT/'spf-hybrid01-rank40-full-s0-audit.json')],{**env,'CUDA_VISIBLE_DEVICES':''},guard)
            audit_raw=(ROOT/'spf-hybrid01-rank40-full-s0-audit.json').read_bytes();audit=json.loads(audit_raw)
            resolved=json.loads((target/'resolved_training.json').read_bytes())
            hybrid_rows=[json.loads(line) for line in (target/'trajectory.jsonl').read_bytes().splitlines()][1:]
            if len(hybrid_rows)!=625 or any(row.get('method_family')!='spf_anchor_loss_hybrid' or row.get('safety_loss_coefficient')!=.1 or not row.get('anchor_gradient_added_before_original_clip_and_step') or row.get('spf_projection_rank')!=40 or not row.get('declared_projection_rank_verified_before_original_combine') for row in hybrid_rows):raise ValueError('Every actual full update must include training anchor loss')
            write(ROOT/'hybrid-full-acceptance.private.json',dict(status='HYBRID_ALL625_UPDATES_ANCHOR_ADDITION_RELOAD_AUDIT_PASS',updates=625,spf_projection_rank=40,all625_rank_boundaries_verified=True,method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1,engineering_proof_sha256=sha((ROOT/'hybrid-engineering-proof.private.json').read_bytes()),candidate_audit_sha256=sha(audit_raw),original_large_autograd_failures_resolved=False))
            if resolved['projection']['rank']!=40:raise ValueError('Serialized rank40 required')
            if resolved['args']['adam_beta1']!=.5 or resolved['args']['adam_beta2']!=.999 or resolved['args']['learning_rate']!=5e-6 or resolved['numerical_precision']!={'cuda_matmul_allow_tf32':False,'float32_matmul_precision':'highest'}:raise ValueError('Actual candidate beta/precision')
            case=dict(step=625,checkpoint=str(target),weights_sha256=audit['final_checkpoint']['files'][0]['sha256'],
                metadata_sha256={name:sha((target/name).read_bytes()) for name in ('config.json','generation_config.json','tokenizer_config.json','trainer_state.json')},
                candidate_audit_sha256=sha(audit_raw),baseline_full_audit_sha256=FULL,parameter_delta=plan['parameter_delta'])
            case['identity_sha256']=sha(json.dumps(case,sort_keys=True,separators=(',',':')).encode())
            write(ROOT/'candidate-identity.private.json',case)
            m,case_dir,adapter=prepare_case(case,plan)
            run=case_dir/'evaluation';run.mkdir(exist_ok=False)
            child_state=dict(status='RUNNING',pid=os.getpid(),commands=m.plan(run),completed=[],stages=[],controller_sha256=sha(Path(m.__file__).read_bytes()),checkpoint_step=625)
            m.write(run/'status.json',child_state)
            for name in m.STAGES:
                state['stage']='candidate-development/'+name;write(JOURNAL/'status.json',state)
                metadata_check(case);m.execute(child_state,run,name,guard)
            child_state.update(status='HYBRID01_RANK40_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE',target_gate='insufficient_evidence');m.write(run/'status.json',child_state)
            state['completed'].append('candidate-development')
            state.update(status='HYBRID01_RANK40_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE',finished_at=time.time())
        except BaseException as error:
            state.update(status='FAILED_NO_RETRY',error_type=type(error).__name__,finished_at=time.time());write(JOURNAL/'status.json',state);raise
        write(JOURNAL/'status.json',state)

if __name__=='__main__':main()
