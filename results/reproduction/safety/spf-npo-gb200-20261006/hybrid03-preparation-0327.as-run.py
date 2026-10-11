"""Prepare the single lambda .1 -> .3 candidate from the newly audited result.

No subprocess, credential read, model evaluation or remote mutation.
"""
from pathlib import Path
import ast, copy, hashlib, importlib.util, io, json, tokenize

root=Path('.').resolve();work=root/'work/spf-npo-gb200-20261006'
campaign=work/'private/spf-safety-search-20261010';folder=root/'scripts/reproduction/safety/gb200'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda value:(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
out=campaign/'candidate-05-hybrid03-preparation-20261012-0327';out.mkdir()
(out/'preparation.as-run.py').write_bytes(Path(__file__).read_bytes())
audit_raw=(campaign/'model-04-hybrid01-result-audit-20261012-0327/audit.private.json').read_bytes()
audit=json.loads(audit_raw)
assert sha(audit_raw)=='a6ca483d3e4dd34944d93be18460596c8c944c206a13244ade5ef52c6e38cf89'
assert audit['harmful_assistance']==27 and audit['verified_counts']==dict(harmful=300,benign=350,conversation=142)
budget=json.loads((campaign/'model-04-hybrid01-result-audit-20261012-0327/ledger-finish-receipt.private.json').read_bytes())
assert budget['status']=='ACTIVE' and budget['reserved_models']==budget['complete_models']==4 and budget['running_models']==0
assert budget['request_intents']==budget['durable_receipts']==3168
oldprep=campaign/'candidate-04-hybrid01-preparation-20261011-1921-r3'
old_evidence=json.loads((oldprep/'evidence.private.json').read_bytes())
oldnames=['run_spf_hybrid01_lr5e6_full.py','run_spf_hybrid01_lr5e6_full_queue.py','spf_hybrid01_lr5e6_development_executor.py','verify_spf_anchor_hybrid.py']
originals={name:(folder/name).read_bytes() for name in oldnames}
assert all(sha(raw)==old_evidence['sources'][name]['sha256'] for name,raw in originals.items())
helper=(folder/'spf_anchor_hybrid.py').read_bytes()
assert sha(helper)=='fdb2e1254dcae25c33ee788123eb2a0ca29e2d23edae0db7971c72d8c15b45cc'

def retarget(text):
    return text.replace('hybrid01','hybrid03').replace('HYBRID01','HYBRID03').replace('20261011-r1','20261012-r1')

def coefficients(text):
    # Change only exact numeric coefficient tokens, preserve all precision/rate constants.
    tokens=list(tokenize.generate_tokens(io.StringIO(text).readline))
    return tokenize.untokenize([t._replace(string='.3' if t.string=='.1' else '0.3')
        if t.type==tokenize.NUMBER and t.string in ('.1','0.1') else t for t in tokens])

plan=json.loads((oldprep/'candidate-plan-core.private.json').read_bytes())
plan['parameter_delta']['safety_loss_coefficient']['candidate']=.3
plan['safety_loss_coefficient']=.3
decision=plan['search_decision']
decision.update(previous_model_identity_sha256=audit['model_identity_sha256'],previous_judge_audit_sha256=sha(audit_raw),
    previous_harmful_assistance=27,previous_complete_model_count=4,
    selected_control_identity_sha256=audit['model_identity_sha256'],selected_control_judge_audit_sha256=sha(audit_raw),
    controlled_delta_from_selected_control={'safety_loss_coefficient':{'baseline':.1,'candidate':.3}})

wrapper=coefficients(retarget(originals[oldnames[0]].decode().replace('\r\n','\n')))
wrapper=wrapper.replace('lambda=.1','lambda=.3')
needle="if proof['status']!='HYBRID_REAL8B_ZERO3_TWO_UPDATES_TOKEN_MEAN_TAIL_RELOAD_PASS' or proof['helper_sha256']!=plan['source_sha256'][str(helper_path)]:"
assert wrapper.count(needle)==1
wrapper=wrapper.replace(needle,needle[:-1]+" or proof['safety_loss_coefficient']!=.3:")

queue=coefficients(retarget(originals[oldnames[1]].decode().replace('\r\n','\n')))
start=queue.index('def validate_completed_predecessor(');end=queue.index('def validate_search_decision(',start)
predecessor='''def validate_completed_predecessor(snapshot):
    branch=snapshot.get('hybrid01_lr5e6_candidate') or {}
    status=branch.get('status') or {}
    if (status.get('pid')!=14033 or status.get('status')!='HYBRID01_LR5E6_FULL_TRAIN_RELOAD_AUDIT_DEVELOPMENT_COMPLETE_PENDING_NEW_JUDGE'
        or status.get('completed')!=['hybrid-cpu-verification','hybrid-engineering-train','hybrid-engineering-reload','hybrid-engineering-verify','train','fresh-reload','full-byte-audit','candidate-development']
        or (branch.get('candidate_audit') or {}).get('sha256')!='6ce8057e3d60ea705794c6c308b6a0563555caad05362fe7864a7db1d75c0931'
        or (snapshot.get('previous_hybrid_full_acceptance') or {}).get('sha256')!='5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c'
        or branch.get('checkpoint_exists') is not True):
        raise ValueError('Completed hybrid01 train/reload/audit/development and all625 acceptance required')
    rows=status.get('stages',[])
    if len(rows)!=7 or any(row.get('returncode')!=0 for row in rows):
        raise ValueError('All seven predecessor engineering/train/reload/audit stages complete')
    process=branch.get('process') or {}
    lines=process.get('stdout','').strip().splitlines()
    if process.get('stderr','').strip() or process.get('returncode') not in (0,1) or not lines or lines[0].split()[:2]!=['PID','STAT']:
        raise ValueError('Unambiguous predecessor process state required')
    if not ((process['returncode']==1 and len(lines)==1) or (process['returncode']==0 and len(lines)==2 and lines[1].split()[0]=='14033' and lines[1].split()[1].startswith('Z'))):
        raise ValueError('Predecessor queue must exit')
    branch=snapshot.get('hybrid03_lr5e6_candidate') or {}
    if branch.get('new_root_exists') is not False or branch.get('launch') is not None or branch.get('status') is not None or branch.get('checkpoint_exists') is not False:
        raise ValueError('Exclusive new candidate absent at snapshot required')

'''
queue=queue[:start]+predecessor+queue[end:]
start=queue.index('def validate_search_decision(');end=queue.index('def validate_predecessor_runtime(',start)
queue=queue[:start]+"def validate_search_decision(plan):\n    expected="+repr(decision)+"\n    if plan.get('search_decision')!=expected:raise ValueError('Exact audited hybrid01 control and lambda .3 decision required')\n\n"+queue[end:]
start=queue.index('def validate_predecessor_runtime(');end=queue.index('def metadata_check(',start)
queue=queue[:start]+'''def validate_predecessor_runtime():
    previous=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')
    if sha((previous/'spf-hybrid01-lr5e6-full-s0-audit.json').read_bytes())!='6ce8057e3d60ea705794c6c308b6a0563555caad05362fe7864a7db1d75c0931':
        raise ValueError('Preserved predecessor audit binding required')
    if sha((previous/'hybrid-full-acceptance.private.json').read_bytes())!='5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c':
        raise ValueError('Original all625 hybrid acceptance required')
    for pid in (14033,15066,15134,15635,15915,15918):
        path=Path('/proc')/str(pid)/'stat'
        if path.exists() and path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z':
            raise ValueError('All predecessor training processes must exit')

'''+queue[end:]
queue=queue.replace('verify_spf_anchor_hybrid.py','verify_spf_anchor_hybrid03.py')
# Do not retarget the unchanged installed auditor route.
assert '/data/spf-beta05-full-20261010-r2/validation-code/audit_spf_target.py' in queue
executor=retarget(originals[oldnames[2]].decode().replace('\r\n','\n'))
verifier=coefficients(originals[oldnames[3]].decode().replace('\r\n','\n'))
verifier=verifier.replace("(0.0, -.3, float('nan'))","(0.0, -.1, float('nan'))")
verifier=verifier.replace('actual_cpu_updates=2, unequal_microbatch', 'safety_loss_coefficient=.3, actual_cpu_updates=2, unequal_microbatch')
sources={'run_spf_hybrid03_lr5e6_full.py':wrapper,'run_spf_hybrid03_lr5e6_full_queue.py':queue,
    'spf_hybrid03_lr5e6_development_executor.py':executor,'verify_spf_anchor_hybrid03.py':verifier}
records={}
for name,text in sources.items():
    raw=text.encode();compile(raw,name,'exec');records[name]=dict(sha256=sha(raw),size=len(raw))
    with (folder/name).open('xb') as stream:stream.write(raw)
    (out/name).write_bytes(raw)
(out/'spf_anchor_hybrid.py').write_bytes(helper)

def functions(text):
    return {f.name:ast.dump(f,include_attributes=False) for f in ast.parse(text).body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
old_functions=functions(originals[oldnames[1]].decode())
new_functions=functions(queue)
unchanged=['sha','bound','write','module','stage']
assert all(old_functions[name]==new_functions[name] for name in unchanged)
assert functions(originals[oldnames[0]].decode())['candidate_recipe']==functions(wrapper)['candidate_recipe']
assert functions(originals[oldnames[3]].decode())['load_helper']==functions(verifier)['load_helper']
assert functions(originals[oldnames[3]].decode())['main']==functions(verifier)['main']
assert executor==retarget(originals[oldnames[2]].decode().replace('\r\n','\n'))
spec=importlib.util.spec_from_file_location('lambda03_queue',(folder/'run_spf_hybrid03_lr5e6_full_queue.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.validate_plan(plan);q.validate_search_decision(plan)
old_snapshot=json.loads((campaign/'heartbeat-20261012-0225-hybrid01.private.json').read_bytes())
synthetic={'hybrid01_lr5e6_candidate':old_snapshot['hybrid01_lr5e6_candidate'],
    'previous_hybrid_full_acceptance':old_snapshot['hybrid_full_acceptance'],
    'hybrid03_lr5e6_candidate':dict(new_root_exists=False,launch=None,status=None,checkpoint_exists=False)}
q.validate_completed_predecessor(synthetic)
negatives=[]
for key,value in [('safety_loss_coefficient',.1),('learning_rate',3e-6),('method_family','spf'),('epochs',1)]:
    bad=copy.deepcopy(plan);bad[key]=value
    try:q.validate_plan(bad)
    except ValueError:negatives.append(key)
    else:raise AssertionError(key)
for key,value in [('previous_judge_audit_sha256','0'*64),('previous_harmful_assistance',38),('development_used_as_training',True),('previous_complete_model_count',3)]:
    bad=copy.deepcopy(plan);bad['search_decision'][key]=value
    try:q.validate_search_decision(bad)
    except ValueError:negatives.append(key)
    else:raise AssertionError(key)
for key,value in [('pid',12367),('completed',['train'])]:
    bad=copy.deepcopy(synthetic);bad['hybrid01_lr5e6_candidate']['status'][key]=value
    try:q.validate_completed_predecessor(bad)
    except ValueError:negatives.append('predecessor_'+key)
    else:raise AssertionError(key)
bad=copy.deepcopy(synthetic);bad['hybrid03_lr5e6_candidate']['new_root_exists']=True
try:q.validate_completed_predecessor(bad)
except ValueError:negatives.append('new_root_exists')
else:raise AssertionError('new root')
(out/'candidate-plan-core.private.json').write_bytes(dump(plan))
evidence=dict(status='LAMBDA03_SOURCES_COMPILE_EXACT_RECIPE_DECISION_PREDECESSOR_SCHEMA_ONLY',sources=records,
    reused_helper_sha256=sha(helper),previous_sources={name:sha(raw) for name,raw in originals.items()},
    unchanged_queue_functions_ast=unchanged,candidate_recipe_ast_unchanged=True,executor_identity_path_only=True,
    verifier_loader_and_main_ast_unchanged=True,new_schema_negatives_rejected=negatives,
    controlled_delta={'safety_loss_coefficient':{'baseline':.1,'candidate':.3}},previous_judge_audit_sha256=sha(audit_raw),
    original_auditor_route_unchanged=True,original_thresholds_unchanged=True,subprocesses=0,cuda=0,model_evaluations=0,
    api_calls=0,credential_reads=0,old_completed_cases_repeated=False)
(out/'evidence.private.json').write_bytes(dump(evidence))

# Retarget the prepared producer. Keep original helper bytes, auditor route and finite protocol.
producer=(work/'prepare_spf_hybrid01_deployment_20261011.py').read_text(encoding='utf8')
producer=retarget(producer).replace('candidate-04-hybrid03-preparation-20261011-1921-r3','candidate-05-hybrid03-preparation-20261012-0327')
producer=producer.replace('verify_spf_anchor_hybrid.py','verify_spf_anchor_hybrid03.py').replace('lambda.1','lambda.3')
producer=producer.replace("assert all(sha(raw)==evidence['sources'][name]['sha256'] for name,raw in records_files.items())",
    "assert all(sha(raw)==(evidence['reused_helper_sha256'] if name=='spf_anchor_hybrid.py' else evidence['sources'][name]['sha256']) for name,raw in records_files.items())")
producer=producer.replace("previous='/data/spf-beta05-lr3e6-full-20261012-r1'","previous='/data/spf-hybrid01-lr5e6-full-20261011-r1'")
producer=producer.replace("previous=Path('/data/spf-beta05-lr3e6-full-20261012-r1')\nassert sha((previous/'spf-beta05-lr3e6-full-s0-audit.json').read_bytes())=='839156fd5c6d0e28847991188a95061d513727cf012ec912465697ca9450c181'",
    "previous=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')\nassert sha((previous/'spf-hybrid01-lr5e6-full-s0-audit.json').read_bytes())=='6ce8057e3d60ea705794c6c308b6a0563555caad05362fe7864a7db1d75c0931'\nassert sha((previous/'hybrid-full-acceptance.private.json').read_bytes())=='5457809bceb7539fb13618ee3e3b8c736a2224d02fa838a0fe59a90e68621c6c'")
producer=producer.replace('12367,12370,12438):','12367,12370,12438,14033,15066,15134,15635,15915,15918):')
producer=producer.replace('hybrid03-lr5e6-full-launch-20261011','hybrid03-lr5e6-full-launch-20261012')
compile(producer,'new_producer','exec')
with (work/'prepare_spf_hybrid03_deployment_20261012.py').open('xb') as stream:stream.write(producer.encode())

monitor=(work/'snapshot_spf_search_hybrid01_20261011.py').read_text(encoding='utf8')
monitor=monitor.replace("previous=Path('/data/spf-beta05-lr3e6-full-20261011-r1')","previous=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')")
monitor=monitor.replace("candidate=Path('/data/spf-hybrid01-lr5e6-full-20261011-r1')","candidate=Path('/data/spf-hybrid03-lr5e6-full-20261012-r1')")
start=monitor.index('    beta05_lr3e6_candidate=');end=monitor.index('    hybrid_engineering_cpu_proof=',start)
monitor=monitor[:start]+'''    hybrid01_lr5e6_candidate=branch(previous,'spf-hybrid01-lr5e6-full-s0','hybrid01-lr5e6-full-launch-20261011.json',
        ['run_spf_hybrid01_lr5e6_full.py','run_spf_hybrid01_lr5e6_full_queue.py','spf_hybrid01_lr5e6_development_executor.py',
         'spf_anchor_hybrid.py','verify_spf_anchor_hybrid.py','plan.private.json','bound-snapshot.private.json']),
    previous_hybrid_full_acceptance=record(previous/'hybrid-full-acceptance.private.json'),
    hybrid03_lr5e6_candidate=branch(candidate,'spf-hybrid03-lr5e6-full-s0','hybrid03-lr5e6-full-launch-20261012.json',
        ['run_spf_hybrid03_lr5e6_full.py','run_spf_hybrid03_lr5e6_full_queue.py','spf_hybrid03_lr5e6_development_executor.py',
         'spf_anchor_hybrid.py','verify_spf_anchor_hybrid03.py','plan.private.json','bound-snapshot.private.json']),
'''+monitor[end:]
compile(monitor,'new_monitor','exec')
with (work/'snapshot_spf_search_hybrid03_20261012.py').open('xb') as stream:stream.write(monitor.encode())
observe=(work/'observe_hybrid_predeployment_20261011_1921.ps1').read_text(encoding='utf8')
observe=observe.replace('20261011-1921','20261012-0327').replace('snapshot_spf_search_hybrid01_20261011.py','snapshot_spf_search_hybrid03_20261012.py')
observe=observe.replace('7eff5c2d674ab63f9e17412bb71c6bb990aafd217952a9f0d88364d41e4c057d',sha(monitor.encode()))
observe=observe.replace('heartbeat-20261012-0327-hybrid01','heartbeat-20261012-0327-hybrid03').replace('hybrid01-monitoring','hybrid03-monitoring')
observe=observe.replace('$taskSnapshot.beta05_lr3e6_candidate.status','$taskSnapshot.hybrid01_lr5e6_candidate.status')
observe=observe.replace('$taskSnapshot.hybrid01_lr5e6_candidate.new_root_exists','$taskSnapshot.hybrid03_lr5e6_candidate.new_root_exists')
with (work/'observe_hybrid03_predeployment_20261012_0327.ps1').open('xb') as stream:stream.write(observe.encode())
print(json.dumps(dict(status=evidence['status'],new_sources=4,reused_helper=True,negatives=len(negatives),
    snapshot_source_sha256=sha(monitor.encode()),remote_commands=0,cuda=0,api=0)))
