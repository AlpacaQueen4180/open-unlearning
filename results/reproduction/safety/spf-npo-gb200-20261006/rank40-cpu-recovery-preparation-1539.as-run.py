"""Prepare explicit unfinished-stage recovery, reusing the successful CPU check by exact bytes."""
from pathlib import Path
import ast,base64,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
work=Path('work/spf-npo-gb200-20261006');campaign=work/'private/spf-safety-search-20261010';folder=Path('scripts/reproduction/safety/gb200')
out=campaign/'rank40-cpu-recovery-preparation-20261012-1539';out.mkdir()
sha=lambda raw:hashlib.sha256(raw).hexdigest();dump=lambda v:(json.dumps(v,indent=2)+'\n').encode()
def exclusive(path,raw):
    with path.open('xb') as f:f.write(raw)
result=json.loads((campaign/'rank40-cpu-oracle-r3-20261012-1539.private.json').read_bytes())
assert result['returncode']==0 and result['child_source_sha256']=='23d61585809a2f4e02b5bebdeba0d57b950e7bf42175b26bb90429d8e0a963fc'
for key in ('stdout','stderr'):
    raw=base64.b64decode(result[key+'_base64']);assert sha(raw)==result[key+'_sha256']
    exclusive(out/('cpu-child.'+key+'.private.bin'),raw)
proof_raw=base64.b64decode(result['stdout_base64']);proof=json.loads(proof_raw)
assert proof==result['cpu_proof'] and sha(proof_raw)=='5397aabbfcaa4d70b1b3f51004721d6d6f389212301ced815472c890bbfb6c4d'
assert proof['actual_cpu_updates']==2 and proof['cuda_initialized'] is False and proof['verifier_sha256']=='a998a42a44109da78c1789478881f1da8fc5753b53788844141b4bc5e65b447b'
exclusive(out/'rank40-cpu-oracle-proof-1539.private.json',proof_raw)
failure=campaign/'rank40-cpu-failure-20261012-1539'
old_plan=json.loads((failure/'005-plan.private.json').read_bytes())
original_wrapper=(folder/'run_spf_hybrid01_rank40_full.py').read_bytes()
original_queue=(folder/'run_spf_hybrid01_rank40_full_queue.py').read_bytes()
assert sha(original_queue)=='36f0c3d74b7e5f9bb4bd9a03fcfc0e5d991f26b903998742aa492602d53f488e'
assert sha(original_wrapper)==old_plan['source_sha256']['/data/spf-hybrid01-rank40-full-20261012-r1/validation-code/run_spf_hybrid01_rank40_full.py']
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old));return text.replace(old,new)
wrapper=original_wrapper.decode().replace('\r\n','\n')
wrapper=replace(wrapper,"for name in ('source-sha256','plan-sha256'):","for name in ('source-sha256','plan-sha256','plan-path'):")
wrapper=replace(wrapper,"plan=bound(ROOT/'validation-code/plan.private.json',a.plan_sha256)","if Path(a.plan_path)!=ROOT/'validation-code/plan-cpu-fixture-recovery-1539.private.json':raise ValueError('Exact new recovery plan path')\n    plan=bound(Path(a.plan_path),a.plan_sha256)")
queue=original_queue.decode().replace('\r\n','\n')
queue=replace(queue,"ORIGINAL = Path", "JOURNAL = ROOT/'runs/cpu-fixture-recovery-1539'\nORIGINAL = Path")
old="if branch.get('new_root_exists') is not False or branch.get('launch') is not None or branch.get('status') is not None or branch.get('checkpoint_exists') is not False:raise ValueError('New candidate must be absent')"
new="""failed=branch.get('status') or {}
    if branch.get('new_root_exists') is not True or (branch.get('launch') or {}).get('pid')!=19460 or failed.get('pid')!=19460 or failed.get('status')!='FAILED_NO_RETRY' or failed.get('stage')!='hybrid-cpu-verification' or failed.get('completed')!=[] or branch.get('checkpoint_exists') is not False:raise ValueError('Exact original CPU-only failure required')
    if (snapshot.get('primary_failure_status_record') or {}).get('sha256')!='2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25':raise ValueError('Original immutable failed status')
    recovery=snapshot.get('rank40_cpu_fixture_recovery') or {}
    if recovery.get('journal_exists') is not False or recovery.get('launch') is not None or recovery.get('status') is not None:raise ValueError('New recovery must be absent')
    if any(snapshot.get(name) is not None for name in ('hybrid_engineering_proof','hybrid_full_acceptance')):raise ValueError('No completed engineering or full training may be repeated')"""
queue=replace(queue,old,new)
extra='''
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

'''
queue=replace(queue,'def metadata_check(case):',extra+'def metadata_check(case):')
queue=queue.replace("write(ROOT/'status.json',state)","write(JOURNAL/'status.json',state)")
queue=replace(queue,"log=str(ROOT/(name+'.raw.log'))","log=str(JOURNAL/(name+'.raw.log'))")
queue=replace(queue,"validate_completed_predecessor(snapshot);validate_search_decision(plan);validate_predecessor_runtime()","validate_completed_predecessor(snapshot);validate_search_decision(plan);validate_predecessor_runtime();validate_failed_runtime(plan)")
queue=replace(queue,"if (ROOT/'status.json').exists() or (ROOT/'spf-hybrid01-rank40-full-s0').exists():raise FileExistsError('Exclusive new candidate')","validate_failed_runtime(plan)\n        JOURNAL.mkdir(parents=True,exist_ok=False)")
queue=replace(queue,"train=ROOT/'validation-code/run_spf_hybrid01_rank40_full.py'","train=ROOT/'validation-code/run_spf_hybrid01_rank40_cpu_recovery.py'")
queue=replace(queue,"verifier=ROOT/'validation-code/verify_spf_anchor_hybrid_rank40.py'","verifier=ROOT/'validation-code/verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py'")
queue=replace(queue,"stage(state,'hybrid-cpu-verification',verify_command+['--output',str(ROOT/'hybrid-cpu-proof.private.json')],{**env,'CUDA_VISIBLE_DEVICES':''},guard)","state['completed'].append('hybrid-cpu-verification-reused')\n            state['stages'].append(dict(name='hybrid-cpu-verification-reused',returncode=0,reused=True,process_pid=None,proof_sha256=plan['recovery_contract']['reused_cpu_proof_sha256'],new_cpu_processes=0))\n            write(JOURNAL/'status.json',state)")
queue=replace(queue,"'--plan-sha256',a.plan_sha256","'--plan-sha256',a.plan_sha256,'--plan-path',a.plan",2)
queue=replace(queue,"'reuse newly completed hybrid03 full byte audit; no repeated weight hashing'","'reuse newly completed rank40 full byte audit; no repeated weight hashing'")
texts={'run_spf_hybrid01_rank40_cpu_recovery.py':wrapper,'run_spf_hybrid01_rank40_cpu_recovery_queue.py':queue}
for name,text in texts.items():compile(text,name,'exec')
def functions(text):return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(text)) if isinstance(node,ast.FunctionDef)}
oldf=functions(original_wrapper.decode());newf=functions(wrapper)
unchanged_wrapper=['candidate_recipe','load_recipe','load_trainer','setup','rank_bound_combine']
assert all(oldf[n]==newf[n] for n in unchanged_wrapper)
oldf=functions(original_queue.decode());newf=functions(queue)
unchanged_queue=['sha','bound','write','module','validate_plan','validate_search_decision','validate_predecessor_runtime','metadata_check']
assert all(oldf[n]==newf[n] for n in unchanged_queue)
for name,text in texts.items():exclusive(folder/name,text.encode());exclusive(out/name,text.encode())
verifier_name='verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py';verifier=(folder/verifier_name).read_bytes();assert sha(verifier)==proof['verifier_sha256'];exclusive(out/verifier_name,verifier)
plan=copy.deepcopy(old_plan)
remote='/data/spf-hybrid01-rank40-full-20261012-r1';validation=remote+'/validation-code/'
plan['source_sha256'].update({validation+name:sha(raw) for name,raw in {**{n:t.encode() for n,t in texts.items()},verifier_name:verifier}.items()})
plan['recovery_contract']=dict(original_failed_status_sha256='2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25',original_cpu_raw_sha256='b833452612cefcf2fd97b1f6fec007c058db761e41eae63f5f61f7f08d5cb263',original_plan_sha256='ef346727c0948dcdb9ce92832b33e455683422bdee2612b5f106b1a5bfbd3829',reused_cpu_proof_sha256=sha(proof_raw),original_failure_archive_sha256='3f5c58114b262673e8f54d93b14eec2adcff57d75186908eb0ef26fc9a0d346d',recovery_journal=remote+'/runs/cpu-fixture-recovery-1539',completed_cpu_case_repeated=False)
plan['scope']='CPU scalar reference precision corrected; exact actual successful CPU proof reused. Only unfinished original rank40 engineering/full/development stages in new journal; all training arithmetic and thresholds unchanged.'
exclusive(out/'plan-cpu-fixture-recovery-1539.private.json',dump(plan))
spec=importlib.util.spec_from_file_location('new_recovery',folder/'run_spf_hybrid01_rank40_cpu_recovery_queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.validate_plan(plan);q.validate_search_decision(plan);q.validate_cpu_proof(proof)
snapshot=json.loads((campaign/'heartbeat-20261012-1539-rank40.private.json').read_bytes())
snapshot.update(primary_failure_status_record={'sha256':'2e27ea9dcdfda80e5a14e8b85802a8b1bc3687569279707e0de50ea530620d25'},rank40_cpu_fixture_recovery={'journal_exists':False,'launch':None,'status':None})
q.validate_completed_predecessor(snapshot)
negative=[]
for key,value in [('loss_oracle_dtype','torch.float32'),('verifier_sha256','0'*64),('actual_cpu_updates',1),('spf_projection_rank',20),('cuda_initialized',True),('maximum_gradient_error',1.0)]:
    bad=copy.deepcopy(proof);bad[key]=value
    try:q.validate_cpu_proof(bad)
    except ValueError:negative.append('cpu_'+key)
    else:raise AssertionError(key)
for key,value in [('journal_exists',True),('launch',{'pid':19460}),('status',{'status':'RUNNING'})]:
    bad=copy.deepcopy(snapshot);bad['rank40_cpu_fixture_recovery'][key]=value
    try:q.validate_completed_predecessor(bad)
    except ValueError:negative.append('recovery_'+key)
    else:raise AssertionError(key)
bad=copy.deepcopy(snapshot);bad['hybrid01_rank40_candidate']['status']['completed']=['train']
try:q.validate_completed_predecessor(bad)
except ValueError:negative.append('completed_train')
else:raise AssertionError('completed train')
monitor=(work/'snapshot_spf_search_hybrid01_rank40_20261012.py').read_text(encoding='utf8').replace('\r\n','\n')
monitor=replace(monitor,"if result['status'] and result['status']['status'] in ('RUNNING','FAILED_NO_RETRY'):","if path!=candidate and result['status'] and result['status']['status'] in ('RUNNING','FAILED_NO_RETRY'):")
lo=monitor.index('launch=read(construction');monitor=monitor[:lo]+'''recovery_journal=candidate/'runs/cpu-fixture-recovery-1539'
recovery_launch=read(candidate/'validation-code/hybrid01-rank40-cpu-recovery-launch-1539.json')
recovery=dict(journal_exists=recovery_journal.exists(),launch=recovery_launch,status=read(recovery_journal/'status.json'),
    process=command(['ps','-p',str(recovery_launch['pid']),'-o','pid,stat,args']) if recovery_launch else None,
    source_files={name:record(candidate/'validation-code'/name) for name in ('run_spf_hybrid01_rank40_cpu_recovery.py','run_spf_hybrid01_rank40_cpu_recovery_queue.py','verify_spf_anchor_hybrid_rank40_cpu_oracle_r2.py','plan-cpu-fixture-recovery-1539.private.json','rank40-cpu-oracle-proof-1539.private.json','bound-snapshot-cpu-recovery-1539.private.json')})
if recovery['status'] and recovery['status']['status'] in ('RUNNING','FAILED_NO_RETRY'):
    stage=recovery['status'].get('stage','')
    log=recovery_journal/(stage+'.raw.log') if not stage.startswith('candidate-development/') else candidate/'candidate-development/evaluation'/(stage.split('/',1)[1]+'.raw.log')
    recovery['current_log']=log.read_text()[-6000:] if log.is_file() else None
if recovery_launch and (candidate/'validation-code/hybrid01-rank40-cpu-recovery-launch-1539.raw.log').is_file() and recovery['status'] is None:
    recovery['early_failure_log']=(candidate/'validation-code/hybrid01-rank40-cpu-recovery-launch-1539.raw.log').read_text()[-6000:]
'''+monitor[lo:]
monitor=replace(monitor,'hybrid_engineering_cpu_proof=record(',"primary_failure_status_record=record(candidate/'status.json'),\n    primary_failure_raw_archive_sha256='3f5c58114b262673e8f54d93b14eec2adcff57d75186908eb0ef26fc9a0d346d',\n    rank40_cpu_fixture_recovery=recovery,\n    hybrid_engineering_cpu_proof=record(")
compile(monitor,'new recovery monitor','exec')
exclusive(work/'snapshot_spf_rank40_cpu_recovery_20261012_1539.py',monitor.encode())
evidence=dict(status='CPU_RECOVERY_SOURCES_AND_SCHEMA_PREPARED_NOT_DEPLOYED',cpu_actual_proof=proof,cpu_actual_proof_sha256=sha(proof_raw),
 sources={name:dict(size=len(raw),sha256=sha(raw)) for name,raw in {**{n:t.encode() for n,t in texts.items()},verifier_name:verifier}.items()},
 original_failure_archive_sha256=plan['recovery_contract']['original_failure_archive_sha256'],recovery_contract=plan['recovery_contract'],
 unchanged_wrapper_functions_ast=unchanged_wrapper,unchanged_queue_functions_ast=unchanged_queue,
 new_schema_negatives_rejected=negative,local_schema_subprocesses=0,local_schema_cuda=0,completed_cpu_case_not_repeated=True,
 new_monitor_sha256=sha(monitor.encode()),original_status_and_sources_preserved=True,actual8b_training=0,new_gpu_submissions=0,api_calls=0,credential_reads=0)
exclusive(out/'evidence.private.json',dump(evidence));exclusive(out/'preparation.as-run.py',Path(__file__).read_bytes())
print(json.dumps(dict(status=evidence['status'],new_sources=3,negative_cases=len(negative),cpu_gradient_error=proof['maximum_gradient_error'],cpu_update_error=proof['maximum_update_error'],new_gpu_submissions=0)))
