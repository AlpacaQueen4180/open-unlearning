"""One declared projection-rank change from the best audited lambda.1 control."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json

root=Path('.').resolve();work=root/'work/spf-npo-gb200-20261006';campaign=work/'private/spf-safety-search-20261010';folder=root/'scripts/reproduction/safety/gb200'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
out=campaign/'candidate-06-hybrid01-rank40-preparation-20261012-1336';out.mkdir()
(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
auditraw=(campaign/'model-05-hybrid03-result-audit-20261012-1336/audit.private.json').read_bytes();audit=json.loads(auditraw)
assert sha(auditraw)=='7c1a87cefbec6e1e65e30154c4a81d9642311a9182d311934fc02f30e9184d30'
assert audit['harmful_assistance']==33 and audit['verified_counts']==dict(harmful=300,benign=350,conversation=142)
budget=json.loads((campaign/'model-05-hybrid03-result-audit-20261012-1336/ledger-finish-receipt.private.json').read_bytes())
assert budget['status']=='ACTIVE' and budget['reserved_models']==budget['complete_models']==5 and budget['running_models']==0
assert budget['request_intents']==budget['durable_receipts']==3960
control=json.loads((root/'results/reproduction/safety/spf-npo-gb200-20261006/spf-hybrid01-judge-complete-hybrid03-launch-20261012-0327.json').read_bytes())
assert control['audit_sha256']=='a6ca483d3e4dd34944d93be18460596c8c944c206a13244ade5ef52c6e38cf89' and control['harmful_assistance']==27
assert control['candidate']['safety_loss_coefficient']==.1 and control['candidate']['learning_rate']==5e-6
oldprep=campaign/'candidate-05-hybrid03-preparation-20261012-0327';old_evidence=json.loads((oldprep/'evidence.private.json').read_bytes())
oldnames=['run_spf_hybrid03_lr5e6_full.py','run_spf_hybrid03_lr5e6_full_queue.py','spf_hybrid03_lr5e6_development_executor.py']
raws={name:(folder/name).read_bytes() for name in oldnames}
assert all(sha(raw)==old_evidence['sources'][name]['sha256'] for name,raw in raws.items())
best_wrapper=(folder/'run_spf_hybrid01_lr5e6_full.py').read_bytes()
assert sha(best_wrapper)=='a3402bfc2bc0d31173e71e3f9240802f6e5cbdc62528e62644507e7676f4a2c8'
verifier_original=(folder/'verify_spf_anchor_hybrid.py').read_bytes()
assert sha(verifier_original)=='d57d8ff1309c15a77beeb72ecbe0fe33a215c9b13fc8fb6f86c57ce1fffb8a73'
helper=(folder/'spf_anchor_hybrid.py').read_bytes();assert sha(helper)=='fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc'
plan=json.loads((oldprep/'candidate-plan-core.private.json').read_bytes())
plan['safety_loss_coefficient']=.1;plan['projection_rank']=40
plan['parameter_delta']['safety_loss_coefficient']['candidate']=.1
plan['parameter_delta']['projection_rank']={'baseline':20,'candidate':40}
decision=plan['search_decision']
decision.update(previous_model_identity_sha256=audit['model_identity_sha256'],previous_judge_audit_sha256=sha(auditraw),previous_harmful_assistance=33,
 previous_complete_model_count=5,selected_control_identity_sha256=control['candidate']['model_identity_sha256'],selected_control_judge_audit_sha256=control['audit_sha256'],
 selected_control_safety_loss_coefficient=.1,selected_control_projection_rank=20,selected_control_harmful_assistance=27,
 controlled_delta_from_selected_control={'projection_rank':{'baseline':20,'candidate':40}})
delta=repr(plan['parameter_delta'])
def retarget(text):
    return text.replace('hybrid03_lr5e6','hybrid01_rank40').replace('hybrid03-lr5e6','hybrid01-rank40').replace('HYBRID03_LR5E6','HYBRID01_RANK40').replace('engineering-spf-hybrid03-s0','engineering-spf-hybrid01-rank40-s0')
wrapper=best_wrapper.decode().replace('\r\n','\n').replace('hybrid01_lr5e6','hybrid01_rank40').replace('hybrid01-lr5e6','hybrid01-rank40').replace('HYBRID01_LR5E6','HYBRID01_RANK40').replace('20261011-r1','20261012-r1').replace('engineering-spf-hybrid01-s0','engineering-spf-hybrid01-rank40-s0')
wrapper=wrapper.replace('lambda=.1; frozen','lambda=.1, projection rank40; frozen')
needle="recipe['args']['learning_rate']=5e-6";assert wrapper.count(needle)==1
wrapper=wrapper.replace(needle,needle+"\n    if original['method_args'].get('rank')!=20:raise ValueError('Original rank20 recipe required')\n    recipe['method_args']['rank']=40")
start=wrapper.index("if plan['parameter_delta']!=");end=wrapper.index(" or plan['new_paid_judge_calls']",start)
wrapper=wrapper[:start]+"if plan['parameter_delta']!="+delta+wrapper[end:]
anchor='    import trainer';assert wrapper.count(anchor)==1
wrapper=wrapper.replace(anchor,"""    if actual['method_args']['rank']!=40:raise ValueError('Declared rank40 required')
    for key,value in original['method_args'].items():
        if key!='rank' and actual['method_args'][key]!=value:raise ValueError('Unrelated method argument changed: '+key)
"""+anchor)
needle="if proof['status']!='HYBRID_REAL8B_ZERO3_TWO_UPDATES_TOKEN_MEAN_TAIL_RELOAD_PASS' or proof['helper_sha256']!=plan['source_sha256'][str(helper_path)]:"
assert wrapper.count(needle)==1
wrapper=wrapper.replace(needle,needle[:-1]+" or proof['safety_loss_coefficient']!=.1 or proof['spf_projection_rank']!=40:")
anchor='            hybrid_binding=hybrid.install(tr,.1)';assert wrapper.count(anchor)==1
wrapper=wrapper.replace(anchor,"""            expected_projection=copy.deepcopy(baseline['projection']);expected_projection['rank']=40
            if vars(tr.projection)!=expected_projection:raise ValueError('Only declared projection rank may change')
            original_rank_combine=tr.backend.combine
            def rank_bound_combine(task,safety,*,method,coefficient,projection,step):
                if method!='spf' or vars(projection)!=expected_projection:raise ValueError('Actual rank40 projection boundary required')
                stats=original_rank_combine(task,safety,method=method,coefficient=coefficient,projection=projection,step=step)
                stats.update(spf_projection_rank=40,declared_projection_rank_verified_before_original_combine=True)
                return stats
            tr.backend.combine=rank_bound_combine
"""+anchor)
wrapper=wrapper.replace('dict(hybrid=hybrid_binding,engineering=a.engineering','dict(spf_projection_rank=40,actual_nonprofile_method_args_match=True,hybrid=hybrid_binding,engineering=a.engineering')
wrapper=wrapper.replace("dict(method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1,engineering=a.engineering","dict(method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1,spf_projection_rank=40,engineering=a.engineering")

queue=retarget(raws[oldnames[1]].decode().replace('\r\n','\n'))
import io,tokenize
tokens=list(tokenize.generate_tokens(io.StringIO(queue).readline))
queue=tokenize.untokenize([t._replace(string='.1' if t.string=='.3' else '0.1') if t.type==tokenize.NUMBER and t.string in ('.3','0.3') else t for t in tokens])
queue=queue.replace('verify_spf_anchor_hybrid03.py','verify_spf_anchor_hybrid_rank40.py')
start=queue.index('def validate_plan(');end=queue.index('def validate_completed_predecessor(',start)
expected={k:plan[k] for k in ['parameter_delta','new_paid_judge_calls','retry_count','existing_training_repeated','formal_hexphi_selection','updates','epochs','examples','unique_examples','seed','method_family','safety_loss_coefficient','projection_rank','learning_rate','beta2','weight_decay','microbatch','accumulation','global_batch','warmup_steps']}
queue=queue[:start]+"def validate_plan(plan):\n    expected="+repr(expected)+"\n    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Exact lambda.1 rank40 candidate required')\n\n"+queue[end:]
start=queue.index('def validate_completed_predecessor(');end=queue.index('def validate_search_decision(',start)
previous_guard='''def validate_completed_predecessor(snapshot):
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
    if branch.get('new_root_exists') is not False or branch.get('launch') is not None or branch.get('status') is not None or branch.get('checkpoint_exists') is not False:raise ValueError('New candidate must be absent')

'''
queue=queue[:start]+previous_guard+queue[end:]
start=queue.index('def validate_search_decision(');end=queue.index('def validate_predecessor_runtime(',start)
queue=queue[:start]+"def validate_search_decision(plan):\n    expected="+repr(decision)+"\n    if plan.get('search_decision')!=expected:raise ValueError('Audited best lambda.1 control and rank20to40 decision required')\n\n"+queue[end:]
start=queue.index('def validate_predecessor_runtime(');end=queue.index('def metadata_check(',start)
queue=queue[:start]+'''def validate_predecessor_runtime():
    previous=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')
    if sha((previous/'spf-hybrid03-lr5e6-full-s0-audit.json').read_bytes())!='d27ab4ec05ddfedcc7b0a2d5b7260c32d5971918613d1326af26f825a6ddd486':raise ValueError('Predecessor audit binding')
    if sha((previous/'hybrid-full-acceptance.private.json').read_bytes())!='ea346a7696d1302acdcae591e99e65cdbbfb65635e594d3e9865d17afc18bb53':raise ValueError('Predecessor acceptance binding')
    for pid in (16733,16736,16947,17483,17763,17766,17834,18362,18642,18645,18802,18939,19076,19213,19348):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':raise ValueError('All predecessor processes must exit')

'''+queue[end:]
needle="if binding['hybrid']['method_family']!='spf_anchor_loss_hybrid' or binding['hybrid']['safety_loss_coefficient']!=.1 or binding['engineering']:"
assert queue.count(needle)==1;queue=queue.replace(needle,needle[:-1]+" or binding['spf_projection_rank']!=40 or not binding['actual_nonprofile_method_args_match']:")
needle="or not row.get('anchor_gradient_added_before_original_clip_and_step') for row in hybrid_rows)"
assert queue.count(needle)==1;queue=queue.replace(needle,"or not row.get('anchor_gradient_added_before_original_clip_and_step') or row.get('spf_projection_rank')!=40 or not row.get('declared_projection_rank_verified_before_original_combine') for row in hybrid_rows)")
queue=queue.replace("updates=625,method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1","updates=625,spf_projection_rank=40,all625_rank_boundaries_verified=True,method_family='spf_anchor_loss_hybrid',safety_loss_coefficient=.1")
needle="if resolved['args']['adam_beta1']!=.5";assert queue.count(needle)==1
queue=queue.replace(needle,"if resolved['projection']['rank']!=40:raise ValueError('Serialized rank40 required')\n            "+needle)
start=queue.index("parameter_delta={'adam_beta1':",queue.index('def prepare_case'));end=queue.index(',\n',start)
queue=queue[:start]+"parameter_delta="+delta+queue[end:]
executor=retarget(raws[oldnames[2]].decode().replace('\r\n','\n'))

verifier=verifier_original.decode().replace('\r\n','\n')
verifier=verifier.replace('torch.nn.Linear(3, 5, bias=False)','torch.nn.Linear(48, 64, bias=False)')
start=verifier.index('    inputs = torch.randn(');end=verifier.index('    maximum_gradient_error',start)
verifier=verifier[:start]+'''    inputs = torch.randn(3, 32, 48)
    labels = torch.arange(96).reshape(3, 32) % 32
    labels[0,30:]=-100;labels[1,26:]=-100;labels[2,16:]=-100
    anchor_inputs = inputs[:2].clone()
    anchor_labels = torch.arange(64).reshape(2,32) % 32 + 32
    anchor_labels[1,24:]=-100
    ce = lambda logits, targets: torch.nn.functional.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
'''+verifier[end:]
verifier=verifier.replace("ProjectionConfig(rank=1, method='exact', seed=0)","ProjectionConfig(rank=40, method='lowrank', oversample=5, niter=2, seed=0)")
verifier=verifier.replace('expected, _ = project_gradients','expected, direct_dot = project_gradients')
needle='        stats = backend.combine(task, safety,';assert verifier.count(needle)==1
verifier=verifier.replace(needle,"        assert direct_dot < 0, 'New rank40 fixture must exercise the original conflict projection'\n"+needle)
start=verifier.index('    negative_cases = []');end=verifier.index('    buffer = io.BytesIO()',start)
verifier=verifier[:start]+verifier[end:]
verifier=verifier.replace("actual_cpu_updates=2, unequal_microbatch_assistant_tokens=[5, 1]","spf_projection_rank=40, safety_loss_coefficient=.1, actual_cpu_updates=2, unequal_microbatch_assistant_tokens=[56, 16]")
verifier=verifier.replace('safety_assistant_tokens=2','safety_assistant_tokens=56').replace('invalid_coefficients_rejected=len(negative_cases)','old_coefficient_negative_cases_not_repeated=True')
needle="    assert resolved['total_updates'] == 2";assert verifier.count(needle)==1
verifier=verifier.replace(needle,"    assert resolved['projection']['rank']==40\n"+needle)
needle="        assert row['method_family'] == 'spf_anchor_loss_hybrid'";assert verifier.count(needle)==1
verifier=verifier.replace(needle,"        assert row['spf_projection_rank']==40 and row['declared_projection_rank_verified_before_original_combine']\n"+needle)
verifier=verifier.replace('updates=2, task_examples=40','spf_projection_rank=40, updates=2, task_examples=40')

texts={'run_spf_hybrid01_rank40_full.py':wrapper,'run_spf_hybrid01_rank40_full_queue.py':queue,
 'spf_hybrid01_rank40_development_executor.py':executor,'verify_spf_anchor_hybrid_rank40.py':verifier}
records={}
for name,text in texts.items():
    raw=text.encode();compile(raw,name,'exec');records[name]=dict(size=len(raw),sha256=sha(raw))
for name,text in texts.items():
    with (folder/name).open('xb') as stream:stream.write(text.encode())
    (out/name).write_bytes(text.encode())
(out/'spf_anchor_hybrid.py').write_bytes(helper)
def functions(text):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
unchanged=['sha','bound','write','module','stage'];old_functions=functions(raws[oldnames[1]].decode());new_functions=functions(queue)
assert all(old_functions[name]==new_functions[name] for name in unchanged)
assert functions(verifier_original.decode())['load_helper']==functions(verifier)['load_helper'] and functions(verifier_original.decode())['main']==functions(verifier)['main']
assert '/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py' in queue
spec=importlib.util.spec_from_file_location('rank40_queue',folder/'run_spf_hybrid01_rank40_full_queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.validate_plan(plan);q.validate_search_decision(plan)
old_snapshot=json.loads((campaign/'heartbeat-20261012-1142-hybrid03.private.json').read_bytes())
synthetic={'hybrid03_lr5e6_candidate':old_snapshot['hybrid03_lr5e6_candidate'],'previous_hybrid_full_acceptance':old_snapshot['hybrid_full_acceptance'],
 'hybrid01_rank40_candidate':dict(new_root_exists=False,launch=None,status=None,checkpoint_exists=False)}
q.validate_completed_predecessor(synthetic)
negatives=[]
for key,value in [('projection_rank',20),('safety_loss_coefficient',.3),('learning_rate',3e-6),('method_family','spf')]:
    bad=copy.deepcopy(plan);bad[key]=value
    try:q.validate_plan(bad)
    except ValueError:negatives.append(key)
    else:raise AssertionError(key)
for key,value in [('selected_control_projection_rank',40),('selected_control_safety_loss_coefficient',.3),('selected_control_judge_audit_sha256','0'*64),('previous_judge_audit_sha256','0'*64),('previous_harmful_assistance',27),('previous_complete_model_count',4),('development_used_as_training',True)]:
    bad=copy.deepcopy(plan);bad['search_decision'][key]=value
    try:q.validate_search_decision(bad)
    except ValueError:negatives.append(key)
    else:raise AssertionError(key)
for key,value in [('pid',14033),('completed',['train'])]:
    bad=copy.deepcopy(synthetic);bad['hybrid03_lr5e6_candidate']['status'][key]=value
    try:q.validate_completed_predecessor(bad)
    except ValueError:negatives.append('predecessor_'+key)
    else:raise AssertionError(key)
bad=copy.deepcopy(synthetic);bad['hybrid01_rank40_candidate']['new_root_exists']=True
try:q.validate_completed_predecessor(bad)
except ValueError:negatives.append('new_root_exists')
else:raise AssertionError('new root')
(out/'candidate-plan-core.private.json').write_bytes(dump(plan))
evidence=dict(status='RANK40_SOURCE_RECIPE_DECISION_PREDECESSOR_SCHEMA_ONLY',sources=records,reused_helper_sha256=sha(helper),
 previous_sources={name:sha(raw) for name,raw in raws.items()},best_control_wrapper_sha256=sha(best_wrapper),
 controlled_delta_from_best_control={'projection_rank':{'baseline':20,'candidate':40}},best_control_lambda=.1,best_control_harmful=27,
 best_control_judge_audit_sha256=control['audit_sha256'],immediate_previous_judge_audit_sha256=sha(auditraw),immediate_previous_harmful=33,
 unchanged_queue_functions_ast=unchanged,verifier_loader_and_main_ast_unchanged=True,executor_identity_path_only=True,
 original_generic_anchor_helper_unchanged=True,original_auditor_route_unchanged=True,original_thresholds_unchanged=True,
 new_schema_negatives_rejected=negatives,new_cpu_rank40_72task56anchor_token_fixture_not_yet_run=True,
 new_actual8b_rank40_engineering_not_yet_run=True,subprocesses=0,cuda=0,model_evaluations=0,api_calls=0,credential_reads=0,
 old_completed_cases_repeated=False,original_large_autograd_failures_resolved=False)
(out/'evidence.private.json').write_bytes(dump(evidence))
print(json.dumps(dict(status=evidence['status'],new_sources=4,new_negatives=len(negatives),parameter_delta=plan['parameter_delta'],remote_commands=0,cuda=0,api=0)))
