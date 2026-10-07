"""Judge only the 26 verified unsent SPF packets; preserve the uncertain request."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

LEGACY_SHA = '8f88420437e4a8d38a862eb85dfa7b09d4c08520dec14de02aed2fa0fb4b09b2'
PRIOR_JOB_SHA = '28dccd42f8b0550ba8d9b2cdef80e4c528f19c766241fb44f362b06a6e16cf79'
PRIOR_RESULTS_SHA = 'eac622c3a4b46d56ea85db873637d7aa873900f89934f9775cee4f04771876d9'
LABELS_SHA = '9d8abdaa71055cc2871807e84783da250289943319723d50d08c745224e2a75f'
MISSING_SHA = '726a3fb9664ef6cbd511aa759484ea4cd2dec7577a59c6d4be6954374a81eecb'
ENDPOINT = 'https://api.openai.com/v1'
PACKET_SHA = '7183c481e35b8e8bcc3043c81c106128b1e05ff49786dcbeeca7a59d97673bb6'


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


def validate_profile(plan):
    expected = dict(total=792, new_calls=26, reused_response_labels=765,
                    uncertain_packet_indices=[765], selected_packet_indices=list(range(766, 792)),
                    total_paid_api_request_limit=792, retries=0, model='gpt-5.6-terra',
                    reasoning='medium', max_output_tokens=4096, api_base_url=ENDPOINT,
                    packet_sha256=PACKET_SHA, credential_policy='new_key_memory_only',
                    execution_location='local_windows', human_adjudication_complete=False,
                    cluster_mapping_verified=False, uncertain_request_retry_permitted=False)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Only verified unsent26 under the original paid cap are allowed')


def preflight(args):
    run = Path(args.run).resolve()
    if (run.is_symlink() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or digest(Path(__file__).read_bytes()) != args.source_sha256):
        raise ValueError('Source and CPU-only environment binding required')
    plan = bound(run / 'plan.private.json', args.plan_sha256)
    validate_profile(plan)
    if (Path(plan['run_directory']).resolve() != run
            or Path(plan['python_executable']).resolve() != Path(sys.executable).resolve()):
        raise ValueError('Exact run and isolated interpreter required')
    if digest((run / 'legacy-runner.py').read_bytes()) != LEGACY_SHA:
        raise ValueError('Original request and rubric implementation changed')
    spec = importlib.util.spec_from_file_location('spf_unsent26_legacy', run / 'legacy-runner.py')
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    if {k: legacy.importlib.metadata.version(k) for k in legacy.VERSIONS} != legacy.VERSIONS:
        raise ValueError('Pinned SDK versions required')
    packets = bound(run / 'private-packet.json', PACKET_SHA)
    if len(packets) != 792 or len({r['id'] for r in packets}) != 792:
        raise ValueError('Unchanged complete SPF packet required')
    for name, expected in legacy.RUBRICS.items():
        if digest((run / 'rubrics' / name).read_bytes()) != expected:
            raise ValueError('Original rubric source changed')
    harmful = legacy.module('unsent26_harmful', run / 'rubrics/src/evals/safety/gpt_judge.py')
    benign = legacy.module('unsent26_benign', run / 'rubrics/src/construction/development_judge.py')
    conversation = legacy.module('unsent26_conversation', run / 'rubrics/src/construction/conversation_judge.py')
    specs = {'harmful': (harmful.SYSTEM_PROMPT, harmful.JudgeResult, 'hexphi-two-dimensional-v1'),
             'benign': (benign.SYSTEM_PROMPT, benign.BenignJudgment, 'benign-development-v1'),
             'conversation': (conversation.SYSTEM_PROMPT, conversation.ConversationJudgment,
                              'wildchat-development-quality-v1')}
    hashes = {k: digest(v[0].encode()) for k, v in specs.items()}
    baseline = bound(run / 'baseline-plan.private.json', plan['baseline_plan_sha256'])
    if hashes != plan['prompt_hashes'] or hashes != baseline['prompt_hashes']:
        raise ValueError('Original rubric prompt hashes required')
    for _, schema, _ in specs.values():
        schema.model_json_schema()
    prior = Path(plan['prior_run_directory']).resolve()
    old = bound(prior / 'job.json', PRIOR_JOB_SHA)
    if digest((prior / 'judge-results.jsonl').read_bytes()) != PRIOR_RESULTS_SHA:
        raise ValueError('Original durable results changed')
    audit_dir = Path(plan['verified_audit_directory']).resolve()
    audit = bound(audit_dir / 'audit.private.json', plan['verified_audit_sha256'])
    bound(audit_dir / 'verified-labels.private.json', LABELS_SHA)
    bound(audit_dir / 'missing-packets.private.json', MISSING_SHA)
    if (old['status'] != 'STOPPED_ON_EXECUTION_ERROR_NO_RETRY' or old['attempted'] != 763
            or old['completed'] != 762 or old.get('error_type') != 'PermissionError'
            or audit['verified_successes'] != 765 or audit['uncertain_packet_index'] != 765
            or audit['not_attempted'] != 26 or audit['uncertain_execution'] != 1
            or audit['last_job_sha256'] != PRIOR_JOB_SHA or audit['last_results_sha256'] != PRIOR_RESULTS_SHA):
        raise ValueError('Verified stopped job and unsent boundary required')
    if any((prior / 'raw-responses' / f'{index:04}.json').exists() for index in range(765, 792)):
        raise ValueError('Missing boundary changed; do not repeat any possible response')
    for name in ['job.json', 'execution.lock', 'judge-results.jsonl', 'raw-responses',
                 'execution-events.private.jsonl']:
        if (run / name).exists():
            raise FileExistsError('Never overwrite or repeat an existing execution')
    return run, plan, packets, specs, hashes


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
        state['status'] = 'UNSENT26_COMPLETE_PENDING_RESULT_AUDIT_UNKNOWN1_AND_CLUSTER_GATE'


async def execute(args):
    run, plan, packets, specs, hashes = preflight(args)
    if args.preflight:
        print(json.dumps(dict(status='UNSENT26_PREFLIGHT_PASS_NO_API_CALLS', new_call_limit=26,
                              reused_spf_successes=765, excluded_uncertain_packet_index=765,
                              schemas_verified=True, new_api_calls=0, credential_reads=0)))
        return
    if (os.environ.get('SPF_JUDGE_NEW_KEY_INPUT') != 'local_masked_dialog'
            or not os.environ.get('OPENAI_API_KEY')
            or os.environ.get('OPENAI_ORG_ID') or os.environ.get('OPENAI_PROJECT_ID')):
        raise ValueError('New key from the local masked input required')
    with (run / 'execution.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    (run / 'raw-responses').mkdir(exist_ok=False)
    state = dict(status='RUNNING', pid=os.getpid(), total=792, new_call_limit=26,
                 reused_spf_successful_labels=765, uncertain_packet_indices=[765],
                 attempted=0, completed=0, retries=0, source_sha256=args.source_sha256,
                 plan_sha256=args.plan_sha256, packet_sha256=PACKET_SHA, started_at=time.time(),
                 target_gate='insufficient_evidence', pilot_frozen=False,
                 human_adjudication_complete=False, cluster_mapping_verified=False,
                 credential_policy='new_key_memory_only', api_base_url=ENDPOINT,
                 authorization=plan['authorization'], last_operation='persist_initial_state')
    client = None
    try:
        atomic(run / 'job.json', state)
        from openai import AsyncOpenAI
        key = os.environ.pop('OPENAI_API_KEY')
        state['last_operation'] = 'construct_client'
        client = AsyncOpenAI(api_key=key, base_url=ENDPOINT, max_retries=0, timeout=90)
        del key
        await judge_rows(client, run, plan, packets, specs, hashes, state)
    except BaseException as error:
        failure = safe_error(error, state['last_operation'])
        state.update(status='STOPPED_ON_EXECUTION_ERROR_NO_RETRY', failure=failure)
        # Best effort alternate evidence only; never repeat a request or failed mutation.
        try:
            event(run, state, 'execution_stopped', failure=failure,
                  attempted=state['attempted'], completed=state['completed'])
        except Exception:
            pass
    finally:
        state['finished_at'] = time.time()
        if client is not None:
            try:
                await client.close()
            except Exception as error:
                state['client_close_error'] = safe_error(error, 'close_client')
        atomic(run / 'job.json', state)
        atomic(run / 'public-summary.json', state)
    print(json.dumps(dict(status=state['status'], attempted=state['attempted'], completed=state['completed'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'source-sha256', 'plan-sha256']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--preflight', action='store_true')
    try:
        asyncio.run(execute(parser.parse_args()))
    except Exception as error:
        print(json.dumps(dict(status='UNSENT26_PREFLIGHT_OR_EXECUTION_REJECTED',
                              failure=safe_error(error, 'main'))))
        sys.exit(1)


if __name__ == '__main__':
    main()
