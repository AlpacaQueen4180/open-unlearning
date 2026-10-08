"""New early-checkpoint Judge; three exclusive792 allocations under authorized2376 cap."""
import argparse,asyncio,hashlib,importlib.metadata,importlib.util,json,os,sys,time,traceback,uuid
from pathlib import Path
from collections import Counter
ENDPOINT='https://api.openai.com/v1'
VERSIONS={'openai':'2.54.0','pydantic':'2.13.4','python-dotenv':'1.2.3'}
RUBRICS={
 'src/evals/safety/gpt_judge.py':'ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a',
 'src/construction/development_judge.py':'beb220e096e8aff1e60c15036fbf3d243b80fab32b28e35c43cea912f55c7d47',
 'src/construction/conversation_judge.py':'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def bound(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('Bound evidence changed: ' + path.name)
    return json.loads(raw)

def atomic(path, data):
    # Unique temporary identity avoids sharing a .tmp filename with a reader.
    # Filesystem failures stop execution; this is not a retry mechanism.
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def safe_error(error, operation):
    return dict(error_type=type(error).__name__, operation=operation,
                errno=getattr(error, 'errno', None), winerror=getattr(error, 'winerror', None),
                frames=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                        for f in traceback.extract_tb(error.__traceback__)],
                exception_message_and_locals_saved=False)

def event(run, state, operation, **details):
    state['last_operation'] = operation
    with (run / 'execution-events.private.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(at=time.time(), operation=operation, **details)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

def validate_profile(plan):
    expected=dict(total=792,new_calls=792,reused_response_labels=0,selected_packet_indices=list(range(792)),
        total_paid_api_request_limit=2376,per_checkpoint_request_limit=792,retries=0,
        model='gpt-5.6-terra',reasoning='medium',max_output_tokens=4096,api_base_url=ENDPOINT,
        credential_policy='local_windows_dpapi_current_user_file',execution_location='local_windows',
        runtime_versions=VERSIONS,human_adjudication_complete=False,cluster_mapping_verified=False,
        previous_success_labels_reused=False,previous_uncertain_request_retried=False)
    if any(plan.get(k)!=v for k,v in expected.items()) or plan.get('checkpoint_step') not in (157,313,469):
        raise ValueError('Only three new792 checkpoint allocations under authorized2376 cap allowed')

def budget_check(run,plan):
    root=Path(plan['budget_root']).resolve()
    if run.parent!=root or run.name!='checkpoint-'+str(plan['checkpoint_step'])+'-saved-key-r1':
        raise ValueError('Exclusive new checkpoint run identity required')
    authorization=bound(root/'authorization.private.json',plan['authorization_sha256'])
    if (authorization['authorized_cap']!=2376 or authorization['allowed_checkpoints']!=[157,313,469]
            or authorization['per_checkpoint_request_limit']!=792 or authorization['max_retries']!=0
            or authorization['user_authorization']!='可以進行Judge'
            or authorization['checkpoint_identities'][str(plan['checkpoint_step'])]!=plan['checkpoint_identity_sha256']):
        raise ValueError('Explicit human budget and checkpoint identity required')
    allocations=root/'allocations';allocations.mkdir(exist_ok=True)
    rows=[]
    for path in allocations.glob('*.json'):
        entry=json.loads(path.read_bytes())
        if (path.name!=str(entry['checkpoint_step'])+'.json' or entry['checkpoint_step'] not in (157,313,469)
                or entry['reserved_request_limit']!=792):raise ValueError('Unexpected budget allocation')
        rows.append(entry)
    if sum(r['reserved_request_limit'] for r in rows)+792>2376 or (allocations/(str(plan['checkpoint_step'])+'.json')).exists():
        raise FileExistsError('Never repeat an allocated checkpoint or exceed new budget')
    return root

def preflight(args):
    run=Path(args.run).resolve()
    if run.is_symlink() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or digest(Path(__file__).read_bytes())!=args.source_sha256:
        raise ValueError('Source and local CPU-only environment binding required')
    plan=bound(run/'plan.private.json',args.plan_sha256);validate_profile(plan)
    storage=bound(Path(plan['credential_authorization_file']),plan['credential_authorization_sha256'])
    if storage.get('human_request')!='把key存在檔案裡面，我不想要每次都要貼key' or storage.get('encryption')!='Windows DPAPI CurrentUser':
        raise ValueError('Explicit user-authorized encrypted local key storage required')
    expected_key=Path(plan['credential_private_root']).resolve()/'local-judge-credentials/openai-judge-key.dpapi'
    if Path(plan['credential_file']).resolve()!=expected_key:
        raise ValueError('Dedicated private saved key path required')
    if Path(plan['run_directory']).resolve()!=run or Path(plan['python_executable']).resolve()!=Path(sys.executable).resolve():
        raise ValueError('Exact new run and isolated interpreter required')
    if {k:importlib.metadata.version(k) for k in VERSIONS}!=VERSIONS:raise ValueError('Pinned SDK profile required')
    packets=bound(run/'private-packet.json',plan['packet_sha256'])
    if (len(packets)!=792 or len({r['id'] for r in packets})!=792
            or dict(Counter(r['kind'] for r in packets))!={'harmful':300,'benign':350,'conversation':142}
            or digest(json.dumps([r['id'] for r in packets],separators=(',',':')).encode())!=plan['ordered_ids_sha256']):
        raise ValueError('Complete new checkpoint packet and original ordered IDs required')
    for name,expected in RUBRICS.items():
        if digest((run/'rubrics'/name).read_bytes())!=expected:raise ValueError('Original rubric changed')
    harmful=module('checkpoint_harmful',run/'rubrics/src/evals/safety/gpt_judge.py')
    benign=module('checkpoint_benign',run/'rubrics/src/construction/development_judge.py')
    conversation=module('checkpoint_conversation',run/'rubrics/src/construction/conversation_judge.py')
    specs={'harmful':(harmful.SYSTEM_PROMPT,harmful.JudgeResult,'hexphi-two-dimensional-v1'),
        'benign':(benign.SYSTEM_PROMPT,benign.BenignJudgment,'benign-development-v1'),
        'conversation':(conversation.SYSTEM_PROMPT,conversation.ConversationJudgment,'wildchat-development-quality-v1')}
    hashes={kind:digest(value[0].encode()) for kind,value in specs.items()}
    baseline=bound(run/'baseline-plan.private.json',plan['baseline_plan_sha256'])
    if hashes!=baseline['prompt_hashes'] or hashes!=plan['prompt_hashes']:raise ValueError('Original prompts required')
    for _,schema,_ in specs.values():schema.model_json_schema()
    if any((run/name).exists() for name in ['job.json','execution.lock','judge-results.jsonl','raw-responses','execution-events.private.jsonl']):
        raise FileExistsError('Never repeat sent or uncertain requests')
    budget_check(run,plan)
    return run,plan,packets,specs,hashes


async def judge_rows(client, run, plan, packets, specs, hashes, state):
    statuses, usage, models = Counter(), Counter(), Counter()
    state['last_operation'] = 'open_results_exclusive'
    with (run / 'judge-results.jsonl').open('x', encoding='utf8') as stream:
        for index in plan['selected_packet_indices']:
            row = packets[index]
            prompt, schema, version = specs[row['kind']]
            payload = json.dumps(row['messages'], ensure_ascii=False)
            if row['kind'] == 'harmful':
                payload = ('HARMFUL USER INSTRUCTION:\n' + row['messages'][0]['content']
                           + '\n\nMODEL RESPONSE:\n' + row['messages'][1]['content'])
            result = dict(id=row['id'], blind_id=row['blind_id'], kind=row['kind'],
                          packet_index=index, requested_model=plan['model'], prompt_version=version,
                          prompt_sha256=hashes[row['kind']], request_payload_sha256=digest(payload.encode()),
                          attempt=1, request_started_at=time.time())
            state.update(attempted=state['attempted'] + 1, current_packet_index=index,
                         last_operation='persist_intent_before_request')
            atomic(run / 'job.json', state)
            event(run, state, 'intent_durable_before_api', packet_index=index,
                  request_payload_sha256=result['request_payload_sha256'])
            event(run, state, 'api_request_started', packet_index=index)
            response = await client.responses.parse(
                model=plan['model'], reasoning={'effort': 'medium'}, instructions=prompt,
                input=payload, text_format=schema, max_output_tokens=4096)
            event(run, state, 'api_response_received', packet_index=index,
                  response_id=response.id, actual_model=response.model,
                  request_id=getattr(response, '_request_id', None))
            state['last_operation'] = 'persist_raw_response_exclusive'
            raw = (response.model_dump_json(indent=2) + '\n').encode()
            with (run / 'raw-responses' / f'{index:04}.json').open('xb') as raw_stream:
                raw_stream.write(raw)
                raw_stream.flush()
                os.fsync(raw_stream.fileno())
            judgment = response.output_parsed
            result.update(actual_model=response.model, response_id=response.id,
                          request_id=getattr(response, '_request_id', None),
                          usage=response.usage.model_dump() if response.usage else None,
                          status='success' if judgment is not None else 'unparsed',
                          judgment=judgment.model_dump() if judgment is not None else None,
                          raw_response_sha256=digest(raw), request_finished_at=time.time())
            if response.model != plan['model']:
                result['status'] = 'unexpected_model'
            state['last_operation'] = 'persist_result_and_fsync'
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            statuses[result['status']] += 1
            models[response.model] += 1
            if response.usage:
                usage.update(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
            state.update(completed=state['completed'] + 1, statuses=dict(statuses),
                         usage=dict(usage), actual_models=dict(models), last_operation='persist_completed_state')
            atomic(run / 'job.json', state)
            if result['status'] != 'success':
                state['status'] = 'STOPPED_ON_RESULT_ERROR_NO_RETRY'
                return
        state['status'] = 'CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE'

async def execute(args):
    run,plan,packets,specs,hashes=preflight(args)
    if args.preflight:
        print(json.dumps(dict(status='NEW_CHECKPOINT_JUDGE_PREFLIGHT_PASS',checkpoint_step=plan['checkpoint_step'],new_call_limit=792,authorized_cap=2376,new_api_calls=0,credential_reads=0)))
        return
    if (os.environ.get('SPF_JUDGE_SAVED_KEY_INPUT')!='local_windows_dpapi_current_user_file' or not os.environ.get('OPENAI_API_KEY')
            or os.environ.get('OPENAI_ORG_ID') or os.environ.get('OPENAI_PROJECT_ID')):
        raise ValueError('Key must be loaded from user-authorized local DPAPI file')
    root=budget_check(run,plan)
    with (root/'allocations'/(str(plan['checkpoint_step'])+'.json')).open('x',encoding='utf8') as f:
        json.dump(dict(checkpoint_step=plan['checkpoint_step'],reserved_request_limit=792,run_directory=str(run),plan_sha256=args.plan_sha256,pid=os.getpid()),f)
        f.flush();os.fsync(f.fileno())
    with (run/'execution.lock').open('x') as f:f.write(str(os.getpid()))
    (run/'raw-responses').mkdir(exist_ok=False)
    state=dict(status='RUNNING',pid=os.getpid(),total=792,new_call_limit=792,authorized_new_total_cap=2376,
        checkpoint_step=plan['checkpoint_step'],checkpoint_identity_sha256=plan['checkpoint_identity_sha256'],
        reused_spf_successful_labels=0,attempted=0,completed=0,retries=0,
        source_sha256=args.source_sha256,plan_sha256=args.plan_sha256,packet_sha256=plan['packet_sha256'],
        started_at=time.time(),target_gate='insufficient_evidence',human_adjudication_complete=False,
        cluster_mapping_verified=False,credential_policy='local_windows_dpapi_current_user_file',api_base_url=ENDPOINT,
        authorization=plan['authorization'],last_operation='persist_initial_state')
    client=None
    try:
        atomic(run/'job.json',state)
        from openai import AsyncOpenAI
        key=os.environ.pop('OPENAI_API_KEY');state['last_operation']='construct_client'
        client=AsyncOpenAI(api_key=key,base_url=ENDPOINT,max_retries=0,timeout=90);del key
        await judge_rows(client,run,plan,packets,specs,hashes,state)
    except BaseException as error:
        failure=safe_error(error,state['last_operation']);state.update(status='STOPPED_ON_EXECUTION_ERROR_NO_RETRY',failure=failure)
        try:event(run,state,'execution_stopped',failure=failure,attempted=state['attempted'],completed=state['completed'])
        except Exception:pass
    finally:
        state['finished_at']=time.time()
        if client is not None:
            try:await client.close()
            except Exception as error:state['client_close_error']=safe_error(error,'close_client')
        atomic(run/'job.json',state);atomic(run/'public-summary.json',state)
    print(json.dumps(dict(status=state['status'],checkpoint_step=plan['checkpoint_step'],attempted=state['attempted'],completed=state['completed'])))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','source-sha256','plan-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--preflight',action='store_true')
    try:asyncio.run(execute(p.parse_args()))
    except Exception as error:
        print(json.dumps(dict(status='CHECKPOINT_JUDGE_PREFLIGHT_OR_EXECUTION_REJECTED',failure=safe_error(error,'main'))));sys.exit(1)

if __name__=='__main__':main()
