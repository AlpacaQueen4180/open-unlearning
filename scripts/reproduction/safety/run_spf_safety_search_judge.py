"""New 50-model campaign Judge, using the original parsed-call loop and durable budget hooks."""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

CORE_SHA = 'd2f09e9709578a9dfdd51d1de3efa1be0d092f6c47d2a9ed1baca6b6d7775570'
BUDGET_SHA = 'f7b5f24a4e4a363d73126d01accd36258479c7a2d298331fc7738cc66628ed6e'
AUTH_SHA = 'd29a89e81d76edd7bc4571ec6c4d1c42322bd1749a703d94c7cc2ac1d231e283'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError('Source or evidence binding changed')
    return json.loads(raw)


def module(path, expected, name):
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError('Original core or campaign accounting source changed')
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def preflight(args):
    run = Path(args.run).resolve()
    if run.is_symlink() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Private local CPU-only run required')
    if digest(Path(__file__).read_bytes()) != args.source_sha256:
        raise ValueError('Exact new runner required')
    plan = bound(run / 'plan.private.json', args.plan_sha256)
    budget = module(run / 'spf_safety_search_budget.py', BUDGET_SHA, 'search_budget_bound')
    core = module(run / 'judge_core_as_run.py', CORE_SHA, 'original_search_judge_core')
    if plan['judge_profile'] != budget.PROFILE or plan['runtime_versions'] != core.VERSIONS:
        raise ValueError('Original endpoint, model, parse profile and pinned SDK required')
    if {k: importlib.metadata.version(k) for k in core.VERSIONS} != core.VERSIONS:
        raise ValueError('Actual pinned SDK versions required')
    if plan['selected_packet_indices'] != list(range(792)) or plan['new_campaign_model_limit'] != 50:
        raise ValueError('Exactly one full new792 model under new50 authorization')
    if (Path(plan['run_directory']).resolve() != run
            or Path(plan['python_executable']).resolve() != Path(sys.executable).resolve()):
        raise ValueError('Exact new run and isolated interpreter required')
    campaign = budget.Campaign(plan['budget_root'])
    bound(campaign.root / 'authorization.private.json', AUTH_SHA)
    if campaign.status()['status'] != 'ACTIVE':
        raise ValueError('Campaign stopped')
    proof = bound(run / 'development-proof.private.json', plan['development_proof_sha256'])
    if (proof['model_identity_sha256'] != plan['model_identity_sha256']
            or proof['completed_stages'] != ['tofu', 'knowledge', 'ifbench', 'safety', 'conversation', 'ifbench-score']
            or proof['actual_outputs_commands_caps_cuda_sources_verified'] is not True
            or proof['candidate_train_reload_audit_verified'] is not True):
        raise ValueError('Actual completed new model and development proof required')
    packets = bound(run / 'private-packet.json', plan['packet_sha256'])
    if (len(packets) != 792 or len({r['id'] for r in packets}) != 792
            or dict(Counter(r['kind'] for r in packets)) != budget.KINDS
            or digest(json.dumps([r['id'] for r in packets], separators=(',', ':')).encode()) != plan['ordered_ids_sha256']):
        raise ValueError('Original complete ordered development packets required')
    storage = bound(Path(plan['credential_authorization_file']), plan['credential_authorization_sha256'])
    if storage['human_request'] != '把key存在檔案裡面，我不想要每次都要貼key' or storage['encryption'] != 'Windows DPAPI CurrentUser':
        raise ValueError('Explicit saved-key authorization required')
    expected_key = Path(plan['credential_private_root']).resolve() / 'local-judge-credentials/openai-judge-key.dpapi'
    if Path(plan['credential_file']).resolve() != expected_key:
        raise ValueError('Dedicated DPAPI credential path required')
    for name, expected in core.RUBRICS.items():
        if digest((run / 'rubrics' / name).read_bytes()) != expected:
            raise ValueError('Original rubric changed')
    harmful = core.module('search_harmful', run / 'rubrics/src/evals/safety/gpt_judge.py')
    benign = core.module('search_benign', run / 'rubrics/src/construction/development_judge.py')
    conversation = core.module('search_conversation', run / 'rubrics/src/construction/conversation_judge.py')
    specs = {'harmful': (harmful.SYSTEM_PROMPT, harmful.JudgeResult, 'hexphi-two-dimensional-v1'),
             'benign': (benign.SYSTEM_PROMPT, benign.BenignJudgment, 'benign-development-v1'),
             'conversation': (conversation.SYSTEM_PROMPT, conversation.ConversationJudgment, 'wildchat-development-quality-v1')}
    hashes = {kind: digest(value[0].encode()) for kind, value in specs.items()}
    baseline = bound(run / 'baseline-plan.private.json', plan['baseline_plan_sha256'])
    if hashes != baseline['prompt_hashes'] or hashes != plan['prompt_hashes']:
        raise ValueError('Original three prompt hashes required')
    for _, schema, _ in specs.values():
        schema.model_json_schema()
    if any((run / name).exists() for name in ('job.json', 'execution.lock', 'judge-results.jsonl',
                                            'raw-responses', 'execution-events.private.jsonl', 'state-snapshots')):
        raise FileExistsError('Never repeat a launched or uncertain Judge')
    # Supply only the original scalar fields to the unchanged Judge loop.
    plan.update(model=budget.PROFILE['model'], total=792, new_calls=792)
    return run, plan, packets, specs, hashes, core, campaign


def attach_accounting(core, campaign, slot, run):
    original_event, original_atomic = core.event, core.atomic
    last_completed = [0]

    def event(run_path, state, operation, **details):
        if operation == 'intent_durable_before_api':
            campaign.intent(slot, details['packet_index'], details['request_payload_sha256'])
        original_event(run_path, state, operation, **details)

    def atomic(path, state):
        completed = state.get('completed', 0)
        if path.name == 'job.json' and completed > last_completed[0]:
            if completed != last_completed[0] + 1:
                raise ValueError('One durable new response per completion required')
            index = state['current_packet_index']
            row = json.loads((run / 'judge-results.jsonl').read_bytes().splitlines()[-1])
            raw = (run / 'raw-responses' / f'{index:04}.json').read_bytes()
            if row['packet_index'] != index or row['raw_response_sha256'] != digest(raw):
                raise ValueError('Original durable raw/result binding changed')
            if row['status'] == 'success':
                campaign.receipt(slot, index, digest(raw), row['response_id'], row['actual_model'])
            last_completed[0] = completed
        original_atomic(path, state)

    core.event, core.atomic = event, atomic


async def execute(args):
    run, plan, packets, specs, hashes, core, campaign = preflight(args)
    reservation_path = run / 'reservation.private.json'
    if args.preflight:
        if args.reserve:
            if reservation_path.exists():
                raise FileExistsError('Reservation already recorded; never relaunch')
            slot = campaign.reserve(plan['model_identity_sha256'], run / 'private-packet.json',
                                    run / 'plan.private.json', run)
            core.atomic(reservation_path, {'slot': slot, 'plan_sha256': args.plan_sha256,
                                         'model_identity_sha256': plan['model_identity_sha256']})
        print(json.dumps({'status': 'NEW_SEARCH_MODEL_JUDGE_PREFLIGHT_PASS', 'api_calls': 0, 'key_reads': 0,
                          'reserved_before_key_access': args.reserve}))
        return
    reservation = json.loads(reservation_path.read_bytes())
    slot = reservation['slot']
    allocation = next(r for r in campaign.status()['slots'] if r['slot'] == slot)
    if (reservation['plan_sha256'] != args.plan_sha256 or allocation['plan_sha'] != args.plan_sha256
            or allocation['status'] != 'RUNNING' or allocation['run_path'] != str(run)):
        raise ValueError('Original active reservation required')
    if (os.environ.get('SPF_JUDGE_SAVED_KEY_INPUT') != 'local_windows_dpapi_current_user_file'
            or not os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_ORG_ID') or os.environ.get('OPENAI_PROJECT_ID')):
        raise ValueError('Key must come from authorized local saved-file loader')
    with (run / 'execution.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    (run / 'raw-responses').mkdir(exist_ok=False)
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'total': 792, 'new_call_limit': 792,
             'model_identity_sha256': plan['model_identity_sha256'], 'campaign_slot': slot,
             'attempted': 0, 'completed': 0, 'retries': 0, 'source_sha256': args.source_sha256,
             'plan_sha256': args.plan_sha256, 'packet_sha256': plan['packet_sha256'],
             'started_at': time.time(), 'target_gate': 'insufficient_evidence',
             'human_adjudication_complete': False, 'cluster_mapping_verified': False,
             'last_operation': 'persist_initial_state'}
    attach_accounting(core, campaign, slot, run)
    client = None
    try:
        core.atomic(run / 'job.json', state)
        from openai import AsyncOpenAI
        key = os.environ.pop('OPENAI_API_KEY')
        client = AsyncOpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=90)
        del key
        await core.judge_rows(client, run, plan, packets, specs, hashes, state)
        if state['status'] == 'CHECKPOINT_JUDGE_COMPLETE_PENDING_RESULT_AUDIT_AND_HUMAN_CLUSTER_GATE':
            state['status'] = 'SEARCH_MODEL_JUDGE_COMPLETE_PENDING_ACTUAL_RESULT_AUDIT'
    except BaseException as error:
        state.update(status='STOPPED_ON_EXECUTION_ERROR_NO_RETRY', failure=core.safe_error(error, state['last_operation']))
        try:
            core.event(run, state, 'execution_stopped', failure=state['failure'])
        except Exception:
            pass
    finally:
        state['finished_at'] = time.time()
        if client is not None:
            try:
                await client.close()
            except Exception as error:
                state['client_close_error'] = core.safe_error(error, 'close_client')
        core.atomic(run / 'job.json', state)
        core.seal_final_job(run / 'job.json', state)
    print(json.dumps({'status': state['status'], 'slot': slot,
                      'attempted': state['attempted'], 'completed': state['completed']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source-sha256', 'plan-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--reserve', action='store_true')
    args = parser.parse_args()
    if args.reserve and not args.preflight:
        parser.error('Reserve only during preflight before saved-key access')
    asyncio.run(execute(args))


if __name__ == '__main__':
    main()
