"""Finite local SPF Judge with pinned rubrics and an explicitly supplied new key."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ENDPOINT = 'https://api.openai.com/v1'
PACKET_SHA = '7183c481e35b8e8bcc3043c81c106128b1e05ff49786dcbeeca7a59d97673bb6'
RUBRICS = {
    'src/evals/safety/gpt_judge.py': 'ddfd529d84ab10b4b3cc8054cecfac8fa8abcc4ef718b77a3b5e5d584eec153a',
    'src/construction/development_judge.py': 'beb220e096e8aff1e60c15036fbf3d243b80fab32b28e35c43cea912f55c7d47',
    'src/construction/conversation_judge.py': 'f0d34059488cc96c4ce6998d9424342e0488fa6427860e4529120415ca38659c',
}
VERSIONS = {'openai': '2.54.0', 'pydantic': '2.13.4', 'python-dotenv': '1.2.3'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('Bound input changed: ' + path.name)
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def validate_profile(plan):
    expected = dict(total=792, new_calls=789, reused_response_labels=3,
                    model='gpt-5.6-terra', reasoning='medium', max_output_tokens=4096,
                    retries=0, packet_sha256=PACKET_SHA, api_base_url=ENDPOINT,
                    execution_location='local_windows', credential_policy='new_key_memory_only',
                    human_adjudication_complete=False, cluster_mapping_verified=False,
                    runtime_versions=VERSIONS)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Pinned local Judge profile changed')


def preflight(args):
    run = Path(args.run).resolve()
    if (run.is_symlink() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or digest(Path(__file__).read_bytes()) != args.source_sha256):
        raise ValueError('Source and CPU-only environment binding required')
    plan = bound_json(run / 'plan.private.json', args.plan_sha256)
    validate_profile(plan)
    if (Path(plan['run_directory']).resolve() != run
            or Path(plan['python_executable']).resolve() != Path(sys.executable).resolve()):
        raise ValueError('Exact local run and isolated interpreter required')
    if {k: importlib.metadata.version(k) for k in VERSIONS} != VERSIONS:
        raise ValueError('Pinned SDK versions required')
    packets = bound_json(run / 'private-packet.json', PACKET_SHA)
    if (len(packets) != 792 or len({r['id'] for r in packets}) != 792
            or dict(Counter(r['kind'] for r in packets)) != plan['kinds']):
        raise ValueError('Complete unchanged SPF packet required')
    for name, expected in RUBRICS.items():
        if digest((run / 'rubrics' / name).read_bytes()) != expected:
            raise ValueError('Original rubric source changed')
    baseline = bound_json(run / 'baseline-plan.private.json', plan['baseline_plan_sha256'])
    harmful = module('local_spf_harmful_contract', run / 'rubrics/src/evals/safety/gpt_judge.py')
    benign = module('local_spf_benign_contract', run / 'rubrics/src/construction/development_judge.py')
    conversation = module('local_spf_conversation_contract', run / 'rubrics/src/construction/conversation_judge.py')
    specs = {'harmful': (harmful.SYSTEM_PROMPT, harmful.JudgeResult, 'hexphi-two-dimensional-v1'),
             'benign': (benign.SYSTEM_PROMPT, benign.BenignJudgment, 'benign-development-v1'),
             'conversation': (conversation.SYSTEM_PROMPT, conversation.ConversationJudgment,
                              'wildchat-development-quality-v1')}
    hashes = {k: digest(v[0].encode()) for k, v in specs.items()}
    if hashes != baseline['prompt_hashes'] or hashes != plan['prompt_hashes']:
        raise ValueError('Original baseline rubric hashes required')
    if any((run / name).exists() for name in ['job.json', 'judge-results.jsonl', 'execution.lock', 'raw-responses']):
        raise FileExistsError('Never repeat or overwrite an existing Judge execution')
    # Standard module registration resolves the original future annotations.
    schemas = {kind: spec[1].model_json_schema() for kind, spec in specs.items()}
    prior = Path(plan['prior_run_directory']).resolve()
    prior_state = bound_json(prior / 'job.json', plan['prior_job_sha256'])
    prior_raw = (prior / 'judge-results.jsonl').read_bytes()
    if digest(prior_raw) != plan['prior_results_sha256']:
        raise ValueError('Prior completed labels changed')
    prior_rows = [json.loads(line) for line in prior_raw.splitlines()]
    diagnostic = bound_json(prior / 'schema-diagnostic.private.json', plan['schema_diagnostic_sha256'])
    if (prior_state['status'] != 'STOPPED_ON_ERROR_NO_RETRY'
            or prior_state['attempted'] != 4 or prior_state['completed'] != 4
            or len(prior_rows) != 4 or [r['status'] for r in prior_rows] != ['success'] * 3 + ['error']
            or prior_rows[-1].get('error_type') != 'PydanticUserError'
            or prior_rows[-1]['kind'] != 'harmful' or diagnostic['mock_transport_http_requests'] != 0
            or diagnostic['legacy_error']['error_code'] != 'class-not-fully-defined'):
        raise ValueError('Only verified pre-HTTP schema failure may be recovered')
    from openai.resources.responses.responses import AsyncResponses
    import inspect
    if digest(inspect.getsource(AsyncResponses.parse).encode()) != diagnostic['installed_sdk_parse_source_sha256']:
        raise ValueError('Source-backed pre-HTTP proof changed')
    for index, row in enumerate(prior_rows):
        if row['id'] != packets[index]['id']:
            raise ValueError('Original packet order changed')
        if index < 3:
            if row['actual_model'] != plan['model'] or not row.get('judgment'):
                raise ValueError('Prior SPF success required')
            raw_response = (prior / 'raw-responses' / f'{index:04}.json').read_bytes()
            if digest(raw_response) != row['raw_response_sha256']:
                raise ValueError('Original successful raw response changed')
    return run, plan, packets[3:], specs, hashes, harmful


async def judge_rows(client, run, plan, packets, specs, hashes, harmful, state):
    statuses = Counter()
    usage = Counter()
    models = Counter()
    with (run / 'judge-results.jsonl').open('x', encoding='utf8') as stream:
        for index, row in enumerate(packets, start=3):
            prompt, schema, version = specs[row['kind']]
            payload = json.dumps(row['messages'], ensure_ascii=False)
            if row['kind'] == 'harmful':
                payload = ('HARMFUL USER INSTRUCTION:\n' + row['messages'][0]['content']
                           + '\n\nMODEL RESPONSE:\n' + row['messages'][1]['content'])
            result = dict(id=row['id'], blind_id=row['blind_id'], kind=row['kind'],
                          requested_model=plan['model'], prompt_version=version,
                          prompt_sha256=hashes[row['kind']], request_payload_sha256=digest(payload.encode()),
                          attempt=1, request_started_at=time.time())
            # Durable intent precedes every request; an interrupted request is never retried.
            state['attempted'] += 1
            atomic(run / 'job.json', state)
            try:
                response = await client.responses.parse(
                    model=plan['model'], reasoning={'effort': 'medium'}, instructions=prompt,
                    input=payload, text_format=schema, max_output_tokens=4096)
                raw = (response.model_dump_json(indent=2) + '\n').encode()
                (run / 'raw-responses' / f'{index:04}.json').write_bytes(raw)
                judgment = response.output_parsed
                result.update(actual_model=response.model, response_id=response.id,
                              request_id=getattr(response, '_request_id', None),
                              usage=response.usage.model_dump() if response.usage else None,
                              status='success' if judgment is not None else 'unparsed',
                              judgment=judgment.model_dump() if judgment is not None else None,
                              raw_response_sha256=digest(raw))
                models[response.model] += 1
                if response.usage:
                    usage['input_tokens'] += response.usage.input_tokens
                    usage['output_tokens'] += response.usage.output_tokens
                if response.model != plan['model']:
                    result['status'] = 'unexpected_model'
            except Exception as error:
                code = getattr(error, 'code', None)
                body = getattr(error, 'body', None)
                if not code and isinstance(body, dict):
                    code = body.get('code') or (body.get('error') or {}).get('code')
                # Never serialize exception messages, request headers, credentials or error bodies.
                safe_code = code if isinstance(code, str) and re.fullmatch(r'[A-Za-z0-9_]{1,80}', code) else None
                result.update(status='policy_blocked' if harmful.is_policy_code(safe_code) else 'error',
                              error_type=type(error).__name__, error_code=safe_code)
            result['request_finished_at'] = time.time()
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            statuses[result['status']] += 1
            state.update(completed=state['completed'] + 1, statuses=dict(statuses),
                         usage=dict(usage), actual_models=dict(models))
            atomic(run / 'job.json', state)
            if result['status'] not in ('success', 'policy_blocked'):
                state['status'] = 'STOPPED_ON_ERROR_NO_RETRY'
                break
        else:
            state['status'] = 'REMAINING_JUDGE_COMPLETE_PENDING_COMBINED_LABEL_REVIEW_AND_CLUSTER_GATE'


async def execute(args):
    run, plan, packets, specs, hashes, harmful = preflight(args)
    if args.preflight:
        print(json.dumps(dict(status='LOCAL_JUDGE_PREFLIGHT_PASS_NO_API_CALLS', total=792,
                              endpoint=ENDPOINT, new_calls=789, reused_spf_labels=3, schemas_verified=True, versions=VERSIONS, rubric_hashes=hashes)))
        return
    if (os.environ.get('SPF_JUDGE_NEW_KEY_INPUT') != 'local_masked_dialog'
            or not os.environ.get('OPENAI_API_KEY')
            or os.environ.get('OPENAI_ORG_ID') or os.environ.get('OPENAI_PROJECT_ID')):
        raise ValueError('New key from the local masked input is required')
    with (run / 'execution.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    (run / 'raw-responses').mkdir(exist_ok=False)
    from openai import AsyncOpenAI
    state = dict(status='RUNNING', pid=os.getpid(), total=792, new_call_limit=789, reused_spf_successful_labels=3, attempted=0, completed=0,
                 source_sha256=args.source_sha256, packet_sha256=PACKET_SHA,
                 plan_sha256=args.plan_sha256, started_at=time.time(), retries=0,
                 api_base_url=ENDPOINT, credential_policy='new_key_memory_only',
                 human_adjudication_complete=False, cluster_mapping_verified=False,
                 target_gate='insufficient_evidence', authorization=plan['authorization'],
                 training_executed=False, model_generation_executed=False)
    atomic(run / 'job.json', state)
    client = None
    try:
        key = os.environ.pop('OPENAI_API_KEY')
        client = AsyncOpenAI(api_key=key, base_url=ENDPOINT, max_retries=0, timeout=90)
        del key
        await judge_rows(client, run, plan, packets, specs, hashes, harmful, state)
    except BaseException as error:
        state.update(status='STOPPED_ON_EXECUTION_ERROR_NO_RETRY', error_type=type(error).__name__)
    finally:
        state['finished_at'] = time.time()
        atomic(run / 'job.json', state)
        if client is not None:
            try:
                await client.close()
            except Exception as error:
                state['client_close_error_type'] = type(error).__name__
                atomic(run / 'job.json', state)
        atomic(run / 'public-summary.json', state)
    print(json.dumps(dict(status=state['status'], attempted=state['attempted'], completed=state['completed'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source-sha256', 'plan-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        asyncio.run(execute(args))
    except Exception as error:
        # Configuration failures occur before an API call; keep output secret-free.
        print(json.dumps(dict(status='LOCAL_JUDGE_PREFLIGHT_OR_KEY_REJECTED', error_type=type(error).__name__)))
        sys.exit(1)


if __name__ == '__main__':
    main()
